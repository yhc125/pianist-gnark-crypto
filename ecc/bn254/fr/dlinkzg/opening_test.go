package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestMonomialRectangularCommitmentsAndSourceLink(t *testing.T) {
	tauY, tauZ := element(19), element(23)
	srs, err := NewMonomialSRS(8, 8, tauY, tauZ)
	if err != nil {
		t.Fatal(err)
	}
	rectangle := [][]fr.Element{
		elements(1, 2, 3, 4, 5),
		elements(6, 7, 8, 9, 10),
		elements(11, 12, 13, 14, 15),
		elements(16, 17, 18, 19, 20),
	}
	commitment, err := CommitRect(rectangle, srs)
	if err != nil {
		t.Fatal(err)
	}
	expectedCommitment := scaleG1(srs.G1Rect[0][0], EvalRect(rectangle, tauY, tauZ))
	if !commitment.Equal(&expectedCommitment) {
		t.Fatal("rectangular commitment does not evaluate at (tauY,tauZ)")
	}

	zPolynomial := elements(3, 1, 4, 1, 5)
	zCommitment, err := CommitZ(zPolynomial, srs)
	if err != nil {
		t.Fatal(err)
	}
	expectedZ := scaleG1(srs.G1Rect[0][0], Eval(zPolynomial, tauZ))
	if !zCommitment.Equal(&expectedZ) {
		t.Fatal("Z commitment uses the wrong SRS direction")
	}
	yPolynomial := elements(2, 7, 1, 8)
	yCommitment, err := CommitY(yPolynomial, srs)
	if err != nil {
		t.Fatal(err)
	}
	expectedY := scaleG1(srs.G1Rect[0][0], Eval(yPolynomial, tauY))
	if !yCommitment.Equal(&expectedY) {
		t.Fatal("Y commitment uses the wrong SRS direction")
	}

	beta, zChallenge := element(7), element(11)
	proof, err := OpenSourceLink(rectangle, beta, zChallenge, srs)
	if err != nil {
		t.Fatal(err)
	}
	wantValue := EvalRect(rectangle, beta, zChallenge)
	if !proof.ClaimedValue.Equal(&wantValue) {
		t.Fatal("source-link prover returned the wrong point value")
	}
	if err := VerifySourceLink(commitment, beta, zChallenge, proof, srs); err != nil {
		t.Fatal(err)
	}

	tamperedValue := proof
	tamperedValue.ClaimedValue.Add(&tamperedValue.ClaimedValue, onePtr())
	if err := VerifySourceLink(commitment, beta, zChallenge, tamperedValue, srs); !errors.Is(err, ErrVerifySourceLink) {
		t.Fatalf("tampered source value was not rejected: %v", err)
	}
	tamperedProof := proof
	tamperedProof.PiZ = addG1(tamperedProof.PiZ, srs.G1Rect[0][0])
	if err := VerifySourceLink(commitment, beta, zChallenge, tamperedProof, srs); !errors.Is(err, ErrVerifySourceLink) {
		t.Fatalf("tampered source quotient was not rejected: %v", err)
	}
}

func TestDeltaBatchedVerification(t *testing.T) {
	srs, err := NewMonomialSRS(8, 8, element(19), element(23))
	if err != nil {
		t.Fatal(err)
	}
	rectangle := [][]fr.Element{
		elements(1, 2, 3, 4, 5),
		elements(6, 7, 8, 9, 10),
		elements(11, 12, 13, 14, 15),
		elements(16, 17, 18, 19, 20),
	}
	beta, zChallenge := element(7), element(11)
	sourceCommitment, err := CommitRect(rectangle, srs)
	if err != nil {
		t.Fatal(err)
	}
	sourceProof, err := OpenSourceLink(rectangle, beta, zChallenge, srs)
	if err != nil {
		t.Fatal(err)
	}

	var betaInverse fr.Element
	betaInverse.Inverse(&beta)
	gPoints := []fr.Element{zChallenge, beta, betaInverse}
	lPoints := []fr.Element{beta, betaInverse}
	gInputs := sameSetInputs([][]fr.Element{
		elements(3, 1, 4, 1, 5, 9),
		elements(2, 6, 5, 3, 5, 8),
		elements(9, 7, 9, 3, 2, 3),
	}, gPoints)
	lInputs := sameSetInputs([][]fr.Element{
		elements(8, 4, 6, 2, 6),
		elements(4, 3, 3, 8, 3),
		elements(2, 7, 9, 5, 0),
		elements(2, 8, 8, 4, 1),
	}, lPoints)
	kappa := element(13)
	nestedResult, err := BuildNestedSetQuotient(gInputs, gPoints, lInputs, lPoints, kappa)
	if err != nil {
		t.Fatal(err)
	}

	gCommitments := commitSameSetPolynomials(t, gInputs, srs)
	lCommitments := commitSameSetPolynomials(t, lInputs, srs)
	numeratorG, err := FoldSameSetCommitments(gCommitments, nestedResult.Outer.Interpolants, kappa, srs)
	if err != nil {
		t.Fatal(err)
	}
	numeratorL, err := FoldSameSetCommitments(lCommitments, nestedResult.Inner.Interpolants, kappa, srs)
	if err != nil {
		t.Fatal(err)
	}
	directNumeratorG, err := CommitZ(nestedResult.Outer.Numerator, srs)
	if err != nil {
		t.Fatal(err)
	}
	if !numeratorG.Equal(&directNumeratorG) {
		t.Fatal("induced G numerator commitment differs from direct commitment")
	}
	directNumeratorL, err := CommitZ(nestedResult.Inner.Numerator, srs)
	if err != nil {
		t.Fatal(err)
	}
	if !numeratorL.Equal(&directNumeratorL) {
		t.Fatal("induced L numerator commitment differs from direct commitment")
	}
	wN, err := CommitZ(nestedResult.Quotient, srs)
	if err != nil {
		t.Fatal(err)
	}
	statement := DeltaBatchStatement{
		SourceCommitment: sourceCommitment,
		SourceValue:      sourceProof.ClaimedValue,
		Beta:             beta,
		ZChallenge:       zChallenge,
		OuterNumerator:   numeratorG,
		InnerNumerator:   numeratorL,
		OuterVanishing:   nestedResult.Outer.Vanishing,
		InnerVanishing:   nestedResult.Inner.Vanishing,
		InnerScale:       nestedResult.InnerScale,
	}
	proof := DeltaBatchProof{
		PiZ: sourceProof.PiZ,
		PiY: sourceProof.PiY,
		WN:  wN,
	}
	delta := element(17)
	if err := VerifyDeltaBatch(statement, proof, delta, srs); err != nil {
		t.Fatal(err)
	}

	tamperedProof := proof
	tamperedProof.WN = addG1(tamperedProof.WN, srs.G1Rect[0][0])
	if err := VerifyDeltaBatch(statement, tamperedProof, delta, srs); !errors.Is(err, ErrVerifyDeltaBatch) {
		t.Fatalf("tampered same-set quotient was not rejected: %v", err)
	}
	tamperedStatement := statement
	tamperedStatement.SourceValue.Add(&tamperedStatement.SourceValue, onePtr())
	if err := VerifyDeltaBatch(tamperedStatement, proof, delta, srs); !errors.Is(err, ErrVerifyDeltaBatch) {
		t.Fatalf("tampered source value was not rejected by delta batch: %v", err)
	}
	var zero fr.Element
	if err := VerifyDeltaBatch(statement, proof, zero, srs); !errors.Is(err, ErrInvalidChallenge) {
		t.Fatalf("zero delta should be rejected, got %v", err)
	}
}

func sameSetInputs(polynomials [][]fr.Element, points []fr.Element) []SameSetInput {
	result := make([]SameSetInput, len(polynomials))
	for i := range polynomials {
		result[i] = SameSetInput{
			Polynomial:    polynomials[i],
			ClaimedValues: evaluations(polynomials[i], points),
		}
	}
	return result
}

func commitSameSetPolynomials(t *testing.T, inputs []SameSetInput, srs *MonomialSRS) []bn254.G1Affine {
	t.Helper()
	result := make([]bn254.G1Affine, len(inputs))
	for i := range inputs {
		var err error
		result[i], err = CommitZ(inputs[i].Polynomial, srs)
		if err != nil {
			t.Fatal(err)
		}
	}
	return result
}
