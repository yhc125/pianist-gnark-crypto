package dlinkzg

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/stretchr/testify/require"
)

func TestSplitSRSMatchesMonolithicCommitments(t *testing.T) {
	const parties = 4
	const degreeBound = 8
	tauY := element(17)
	tauZ := element(19)
	monolithic, err := NewMonomialSRS(parties, degreeBound, tauY, tauZ)
	require.NoError(t, err)

	rectangle := make([][]fr.Element, parties)
	var aggregate bn254.G1Jac
	for rank := 0; rank < parties; rank++ {
		row, rowErr := NewDeterministicPartyRowSRS(
			parties,
			degreeBound,
			rank,
			tauY,
			tauZ,
		)
		require.NoError(t, rowErr)
		require.Equal(t, rank, row.Rank)
		require.Equal(t, degreeBound, len(row.G1SemanticRow))
		require.Equal(t, degreeBound, len(row.G1Row))
		require.Equal(t, degreeBound, len(row.G1ZShared))

		polynomial := make([]fr.Element, degreeBound-rank)
		for j := range polynomial {
			polynomial[j].SetUint64(uint64(1 + 13*rank + j))
		}
		rectangle[rank] = polynomial

		got, commitErr := row.CommitRow(polynomial)
		require.NoError(t, commitErr)
		expected, commitErr := CommitRect(singleRankRectangle(
			parties,
			rank,
			polynomial,
		), monolithic)
		require.NoError(t, commitErr)
		require.True(t, got.Equal(&expected), "rank %d commitment", rank)

		gotZ, commitErr := row.CommitZ(polynomial)
		require.NoError(t, commitErr)
		expectedZ, commitErr := CommitZ(polynomial, monolithic)
		require.NoError(t, commitErr)
		require.True(t, gotZ.Equal(&expectedZ), "rank %d shared-Z commitment", rank)
		if rank > 0 {
			require.False(t, got.Equal(&gotZ), "rank %d mixed and Y^0 rows were conflated", rank)
		}

		var gotJac bn254.G1Jac
		gotJac.FromAffine(&got)
		aggregate.AddAssign(&gotJac)
	}

	wantAggregate, err := CommitRect(rectangle, monolithic)
	require.NoError(t, err)
	var gotAggregate bn254.G1Affine
	gotAggregate.FromJacobian(&aggregate)
	require.True(t, gotAggregate.Equal(&wantAggregate))

	coordinator, err := NewDeterministicCoordinatorSRS(parties, tauY, tauZ)
	require.NoError(t, err)
	yPolynomial := []fr.Element{element(3), element(5), element(7), element(11)}
	gotY, err := coordinator.CommitY(yPolynomial)
	require.NoError(t, err)
	wantY, err := CommitY(yPolynomial, monolithic)
	require.NoError(t, err)
	require.True(t, gotY.Equal(&wantY))
	coordinatorZPolynomial := []fr.Element{element(13), element(17), element(19), element(23)}
	gotCoordinatorZ, err := coordinator.CommitZ(coordinatorZPolynomial)
	require.NoError(t, err)
	wantCoordinatorZ, err := CommitZ(coordinatorZPolynomial, monolithic)
	require.NoError(t, err)
	require.True(t, gotCoordinatorZ.Equal(&wantCoordinatorZ))

	verifier := NewDeterministicVerifierSRS(tauY, tauZ)
	require.NoError(t, verifier.Validate())
	for i := range verifier.G1ZVerifier {
		require.True(t, verifier.G1ZVerifier[i].Equal(&monolithic.G1Rect[0][i]))
	}
	for i := range verifier.G2Y {
		require.True(t, verifier.G2Y[i].Equal(&monolithic.G2Y[i]))
	}
	for i := range verifier.G2Z {
		require.True(t, verifier.G2Z[i].Equal(&monolithic.G2Z[i]))
	}

	beta := element(41)
	zChallenge := element(43)
	sourceCommitment, err := CommitRect(rectangle, monolithic)
	require.NoError(t, err)
	sourceProof, err := OpenSourceLink(rectangle, beta, zChallenge, monolithic)
	require.NoError(t, err)
	require.NoError(t, verifier.VerifySourceLink(
		sourceCommitment,
		beta,
		zChallenge,
		sourceProof,
	))
}

func TestSemanticRowMatchesNativeTaylorShift(t *testing.T) {
	const parties = 4
	const degreeBound = 16
	const rank = 2
	tauY := element(17)
	tauZ := element(19)
	sigma := element(23)
	row, err := NewDeterministicPartyRowSRSWithShift(
		parties,
		degreeBound,
		rank,
		tauY,
		tauZ,
		sigma,
	)
	require.NoError(t, err)

	polynomial := elements(2, 3, 5, 7, 11, 13, 17, 19)
	semantic, err := row.CommitSemantic(polynomial)
	require.NoError(t, err)
	shifted := FastTaylorShift(polynomial, sigma)
	nativeShifted, err := row.CommitRow(shifted)
	require.NoError(t, err)
	require.True(t, semantic.Equal(&nativeShifted))

	nativeUnshifted, err := row.CommitRow(polynomial)
	require.NoError(t, err)
	require.False(t, semantic.Equal(&nativeUnshifted))

	var tauX fr.Element
	tauX.Add(&tauZ, &sigma)
	semanticMonolithic, err := NewMonomialSRS(parties, degreeBound, tauY, tauX)
	require.NoError(t, err)
	want, err := CommitRect(
		singleRankRectangle(parties, rank, polynomial),
		semanticMonolithic,
	)
	require.NoError(t, err)
	require.True(t, semantic.Equal(&want))
}

func TestUnshiftedPartyConstructorCompatibility(t *testing.T) {
	row, err := NewDeterministicPartyRowSRS(4, 8, 1, element(29), element(31))
	require.NoError(t, err)
	polynomial := elements(1, 4, 9, 16)
	semantic, err := row.CommitSemantic(polynomial)
	require.NoError(t, err)
	native, err := row.CommitRow(polynomial)
	require.NoError(t, err)
	require.True(t, semantic.Equal(&native))
}

func TestSplitVerifierMatchesMonolithicDeltaBatch(t *testing.T) {
	tauY := element(47)
	tauZ := element(53)
	monolithic, err := NewMonomialSRS(4, 8, tauY, tauZ)
	require.NoError(t, err)
	verifier := NewDeterministicVerifierSRS(tauY, tauZ)

	rectangle := [][]fr.Element{
		elements(1, 2, 3, 4),
		elements(5, 6, 7, 8),
		elements(9, 10, 11, 12),
		elements(13, 14, 15, 16),
	}
	beta := element(59)
	zChallenge := element(61)
	sourceCommitment, err := CommitRect(rectangle, monolithic)
	require.NoError(t, err)
	sourceProof, err := OpenSourceLink(rectangle, beta, zChallenge, monolithic)
	require.NoError(t, err)

	var betaInverse fr.Element
	betaInverse.Inverse(&beta)
	gPoints := []fr.Element{zChallenge, beta, betaInverse}
	lPoints := []fr.Element{beta, betaInverse}
	gInputs := sameSetInputs([][]fr.Element{
		elements(2, 3, 5, 7, 11),
		elements(13, 17, 19, 23, 29),
		elements(31, 37, 41, 43, 47),
	}, gPoints)
	lInputs := sameSetInputs([][]fr.Element{
		elements(3, 1, 4, 1),
		elements(5, 9, 2, 6),
		elements(5, 3, 5, 8),
		elements(9, 7, 9, 3),
	}, lPoints)
	kappa := element(67)
	gResult, err := BuildSameSetQuotient(gInputs, gPoints, kappa)
	require.NoError(t, err)
	lResult, err := BuildSameSetQuotient(lInputs, lPoints, kappa)
	require.NoError(t, err)
	gCommitments := commitSameSetPolynomials(t, gInputs, monolithic)
	lCommitments := commitSameSetPolynomials(t, lInputs, monolithic)
	numeratorG, err := FoldSameSetCommitments(gCommitments, gResult.Interpolants, kappa, monolithic)
	require.NoError(t, err)
	numeratorL, err := FoldSameSetCommitments(lCommitments, lResult.Interpolants, kappa, monolithic)
	require.NoError(t, err)
	splitNumeratorG, err := verifier.FoldSameSetCommitments(gCommitments, gResult.Interpolants, kappa)
	require.NoError(t, err)
	require.True(t, splitNumeratorG.Equal(&numeratorG))
	splitNumeratorL, err := verifier.FoldSameSetCommitments(lCommitments, lResult.Interpolants, kappa)
	require.NoError(t, err)
	require.True(t, splitNumeratorL.Equal(&numeratorL))
	wG, err := CommitZ(gResult.Quotient, monolithic)
	require.NoError(t, err)
	wL, err := CommitZ(lResult.Quotient, monolithic)
	require.NoError(t, err)

	statement := DeltaBatchStatement{
		SourceCommitment: sourceCommitment,
		SourceValue:      sourceProof.ClaimedValue,
		Beta:             beta,
		ZChallenge:       zChallenge,
		NumeratorG:       numeratorG,
		NumeratorL:       numeratorL,
		VanishingG:       gResult.Vanishing,
		VanishingL:       lResult.Vanishing,
	}
	proof := DeltaBatchProof{
		PiZ: sourceProof.PiZ,
		PiY: sourceProof.PiY,
		WG:  wG,
		WL:  wL,
	}
	delta := element(71)
	require.NoError(t, VerifyDeltaBatch(statement, proof, delta, monolithic))
	require.NoError(t, verifier.VerifyDeltaBatch(statement, proof, delta))

	tampered := proof
	tampered.WL = addG1(tampered.WL, verifier.G1ZVerifier[0])
	require.ErrorIs(t, verifier.VerifyDeltaBatch(statement, tampered, delta), ErrVerifyDeltaBatch)
}

func TestSplitSRSMemoryShapes(t *testing.T) {
	const parties = 16
	const degreeBound = 64
	tauY := element(23)
	tauZ := element(29)

	row, err := NewDeterministicPartyRowSRSWithShift(parties, degreeBound, 7, tauY, tauZ, element(31))
	require.NoError(t, err)
	require.Len(t, row.G1SemanticRow, degreeBound)
	require.Len(t, row.G1Row, degreeBound)
	require.Len(t, row.G1ZShared, degreeBound)
	require.Equal(t, 7, row.Rank)
	require.Equal(t, parties, row.Parties)

	coordinator, err := NewDeterministicCoordinatorSRS(parties, tauY, tauZ)
	require.NoError(t, err)
	require.Len(t, coordinator.G1Y, parties)
	require.Len(t, coordinator.G1ZShared, parties)

	verifier := NewDeterministicVerifierSRS(tauY, tauZ)
	require.Len(t, verifier.G1ZVerifier, verifierG1ZPowers)
	require.Len(t, verifier.G2Y, verifierYPowers)
	require.Len(t, verifier.G2Z, verifierZPowers)
	require.NotEmpty(t, DeterministicSplitSRSNotice)
}

func TestSplitSRSRejectsMalformedRankAndDegree(t *testing.T) {
	tauY := element(31)
	tauZ := element(37)

	_, err := NewDeterministicPartyRowSRS(1, 8, 0, tauY, tauZ)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = NewDeterministicPartyRowSRS(4, 3, 0, tauY, tauZ)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = NewDeterministicPartyRowSRS(4, 8, -1, tauY, tauZ)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = NewDeterministicPartyRowSRS(4, 8, 4, tauY, tauZ)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = NewDeterministicPartyRowSRSWithShift(4, 8, 4, tauY, tauZ, element(41))
	require.ErrorIs(t, err, ErrInvalidSRS)

	row, err := NewDeterministicPartyRowSRS(4, 8, 2, tauY, tauZ)
	require.NoError(t, err)
	_, err = row.CommitSemantic(make([]fr.Element, 9))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	_, err = row.CommitRow(make([]fr.Element, 9))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	_, err = row.CommitZ(make([]fr.Element, 9))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	row.G1Row = row.G1Row[:7]
	_, err = row.CommitSemantic(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitRow(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)

	row, err = NewDeterministicPartyRowSRS(4, 8, 2, tauY, tauZ)
	require.NoError(t, err)
	row.G1SemanticRow = row.G1SemanticRow[:7]
	_, err = row.CommitSemantic(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitRow(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)

	row, err = NewDeterministicPartyRowSRS(4, 8, 2, tauY, tauZ)
	require.NoError(t, err)
	row.G1ZShared = row.G1ZShared[:7]
	_, err = row.CommitSemantic(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitRow(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = row.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	var nilRow *PartyRowSRS
	_, err = nilRow.CommitSemantic(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = nilRow.CommitRow(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = nilRow.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)

	_, err = NewDeterministicCoordinatorSRS(1, tauY, tauZ)
	require.ErrorIs(t, err, ErrInvalidSRS)
	coordinator, err := NewDeterministicCoordinatorSRS(4, tauY, tauZ)
	require.NoError(t, err)
	_, err = coordinator.CommitY(make([]fr.Element, 5))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	_, err = coordinator.CommitZ(make([]fr.Element, 5))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	coordinator.G1Y = coordinator.G1Y[:3]
	_, err = coordinator.CommitY(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = coordinator.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	coordinator, err = NewDeterministicCoordinatorSRS(4, tauY, tauZ)
	require.NoError(t, err)
	coordinator.G1ZShared = coordinator.G1ZShared[:3]
	_, err = coordinator.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	var nilCoordinator *CoordinatorSRS
	_, err = nilCoordinator.CommitY(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)
	_, err = nilCoordinator.CommitZ(nil)
	require.ErrorIs(t, err, ErrInvalidSRS)

	verifier := NewDeterministicVerifierSRS(tauY, tauZ)
	tooWideStatement := DeltaBatchStatement{
		VanishingG: make([]fr.Element, verifierZPowers+1),
		VanishingL: []fr.Element{fr.One()},
	}
	_, _, generator1, _ := bn254.Generators()
	tooWideProof := DeltaBatchProof{PiZ: generator1}
	tooWideDelta := fr.One()
	require.ErrorIs(
		t,
		verifier.VerifyDeltaBatch(tooWideStatement, tooWideProof, tooWideDelta),
		ErrPolynomialTooWide,
	)
	_, err = verifier.CommitZ(make([]fr.Element, verifierG1ZPowers+1))
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	_, err = verifier.FoldSameSetCommitments(
		[]bn254.G1Affine{verifier.G1ZVerifier[0]},
		nil,
		fr.One(),
	)
	require.ErrorIs(t, err, ErrMismatchedInput)
	_, err = verifier.FoldSameSetCommitments(
		[]bn254.G1Affine{verifier.G1ZVerifier[0]},
		[][]fr.Element{make([]fr.Element, verifierG1ZPowers+1)},
		fr.One(),
	)
	require.ErrorIs(t, err, ErrPolynomialTooWide)
	verifier.G1ZVerifier = verifier.G1ZVerifier[:2]
	require.ErrorIs(t, verifier.Validate(), ErrInvalidSRS)
	verifier = NewDeterministicVerifierSRS(tauY, tauZ)
	verifier.G2Z = verifier.G2Z[:3]
	require.ErrorIs(t, verifier.Validate(), ErrInvalidSRS)
	var nilVerifier *VerifierSRS
	require.ErrorIs(t, nilVerifier.Validate(), ErrInvalidSRS)
}

func singleRankRectangle(parties, rank int, polynomial []fr.Element) [][]fr.Element {
	result := make([][]fr.Element, parties)
	result[rank] = polynomial
	return result
}
