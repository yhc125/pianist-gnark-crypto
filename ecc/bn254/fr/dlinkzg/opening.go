package dlinkzg

import (
	"errors"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

var (
	ErrInvalidRectangle = errors.New("dlinkzg: invalid rectangular polynomial")
	ErrInvalidChallenge = errors.New("dlinkzg: challenge must be nonzero")
	ErrVerifySourceLink = errors.New("dlinkzg: source-link verification failed")
	ErrVerifyDeltaBatch = errors.New("dlinkzg: delta-batched verification failed")
)

// SourceLinkProof proves D(beta,zChallenge)=ClaimedValue using the two
// directional quotients from Equation (dlinkzg-source-link).
type SourceLinkProof struct {
	PiZ          bn254.G1Affine
	PiY          bn254.G1Affine
	ClaimedValue fr.Element
}

// DeltaBatchStatement contains A0's source statement and the two induced
// same-set numerator commitments A1 and A2 from Equation (final-pairing).
type DeltaBatchStatement struct {
	SourceCommitment bn254.G1Affine
	SourceValue      fr.Element
	Beta             fr.Element
	ZChallenge       fr.Element
	NumeratorG       bn254.G1Affine
	NumeratorL       bn254.G1Affine
	VanishingG       []fr.Element
	VanishingL       []fr.Element
}

// DeltaBatchProof contains the four quotient commitments on the right-hand
// side of Equation (final-pairing).
type DeltaBatchProof struct {
	PiZ bn254.G1Affine
	PiY bn254.G1Affine
	WG  bn254.G1Affine
	WL  bn254.G1Affine
}

// EvalRect evaluates sum_{i,j} coefficients[i][j]Y^iZ^j at (y,z).
func EvalRect(coefficients [][]fr.Element, y, z fr.Element) fr.Element {
	rows := make([]fr.Element, len(coefficients))
	for i := range coefficients {
		rows[i] = Eval(coefficients[i], z)
	}
	return Eval(rows, y)
}

// OpenSourceLink constructs the canonical Z-then-Y quotient decomposition.
// Soundness does not rely on this decomposition being unique.
func OpenSourceLink(coefficients [][]fr.Element, beta, zChallenge fr.Element, srs *MonomialSRS) (SourceLinkProof, error) {
	var proof SourceLinkProof
	if err := validateSRS(srs); err != nil {
		return proof, err
	}
	if len(coefficients) == 0 || len(coefficients) > len(srs.G1Rect) {
		return proof, ErrInvalidRectangle
	}

	qZ := make([][]fr.Element, len(coefficients))
	rowValues := make([]fr.Element, len(coefficients))
	for i := range coefficients {
		if len(coefficients[i]) == 0 || len(coefficients[i]) > len(srs.G1Rect[i]) {
			return SourceLinkProof{}, ErrInvalidRectangle
		}
		qZ[i], rowValues[i] = SyntheticDivision(coefficients[i], zChallenge)
	}
	qY, value := SyntheticDivision(rowValues, beta)
	var err error
	proof.PiZ, err = CommitRect(qZ, srs)
	if err != nil {
		return SourceLinkProof{}, err
	}
	proof.PiY, err = CommitY(qY, srs)
	if err != nil {
		return SourceLinkProof{}, err
	}
	proof.ClaimedValue = value
	return proof, nil
}

// VerifySourceLink checks the bivariate point-ideal opening equation.
func VerifySourceLink(commitment bn254.G1Affine, beta, zChallenge fr.Element, proof SourceLinkProof, srs *MonomialSRS) error {
	if err := validateSRS(srs); err != nil {
		return err
	}
	a0 := subtractG1Scalar(commitment, srs.G1Rect[0][0], proof.ClaimedValue)
	zDirection := subtractG2Scalar(srs.G2Z[1], srs.G2Z[0], zChallenge)
	yDirection := subtractG2Scalar(srs.G2Y[1], srs.G2Y[0], beta)
	negPiZ := negG1(proof.PiZ)
	negPiY := negG1(proof.PiY)
	ok, err := bn254.PairingCheck(
		[]bn254.G1Affine{a0, negPiZ, negPiY},
		[]bn254.G2Affine{srs.G2Z[0], zDirection, yDirection},
	)
	if err != nil {
		return err
	}
	if !ok {
		return ErrVerifySourceLink
	}
	return nil
}

// FoldSameSetCommitments derives C_F from commitments to p and the public
// interpolants l_p, without recommitting to the numerator polynomial.
func FoldSameSetCommitments(commitments []bn254.G1Affine, interpolants [][]fr.Element, kappa fr.Element, srs *MonomialSRS) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if len(commitments) != len(interpolants) {
		return result, ErrMismatchedInput
	}
	var resultJac bn254.G1Jac
	var power fr.Element
	power.SetOne()
	for i := range commitments {
		interpolantCommitment, err := CommitZ(interpolants[i], srs)
		if err != nil {
			return bn254.G1Affine{}, err
		}
		difference := subtractG1(commitments[i], interpolantCommitment)
		scaled := scaleG1(difference, power)
		var scaledJac bn254.G1Jac
		scaledJac.FromAffine(&scaled)
		resultJac.AddAssign(&scaledJac)
		power.Mul(&power, &kappa)
	}
	result.FromJacobian(&resultJac)
	return result, nil
}

// VerifyDeltaBatch checks the source link and both same-set quotients in one
// five-pairing product, as in Equation (final-pairing).
func VerifyDeltaBatch(statement DeltaBatchStatement, proof DeltaBatchProof, delta fr.Element, srs *MonomialSRS) error {
	if delta.IsZero() {
		return ErrInvalidChallenge
	}
	if err := validateSRS(srs); err != nil {
		return err
	}
	zG, err := commitZInG2(statement.VanishingG, srs)
	if err != nil {
		return err
	}
	zL, err := commitZInG2(statement.VanishingL, srs)
	if err != nil {
		return err
	}

	a0 := subtractG1Scalar(statement.SourceCommitment, srs.G1Rect[0][0], statement.SourceValue)
	var deltaSquared fr.Element
	deltaSquared.Square(&delta)
	left := addG1(a0, scaleG1(statement.NumeratorG, delta))
	left = addG1(left, scaleG1(statement.NumeratorL, deltaSquared))

	zDirection := subtractG2Scalar(srs.G2Z[1], srs.G2Z[0], statement.ZChallenge)
	yDirection := subtractG2Scalar(srs.G2Y[1], srs.G2Y[0], statement.Beta)
	negPiZ := negG1(proof.PiZ)
	negPiY := negG1(proof.PiY)
	negWG := negG1(scaleG1(proof.WG, delta))
	negWL := negG1(scaleG1(proof.WL, deltaSquared))
	ok, err := bn254.PairingCheck(
		[]bn254.G1Affine{left, negPiZ, negPiY, negWG, negWL},
		[]bn254.G2Affine{srs.G2Z[0], zDirection, yDirection, zG, zL},
	)
	if err != nil {
		return err
	}
	if !ok {
		return ErrVerifyDeltaBatch
	}
	return nil
}

func subtractG1Scalar(point, base bn254.G1Affine, scalar fr.Element) bn254.G1Affine {
	return subtractG1(point, scaleG1(base, scalar))
}

func subtractG2Scalar(point, base bn254.G2Affine, scalar fr.Element) bn254.G2Affine {
	scaled := scaleG2(base, scalar)
	var pointJac, scaledJac bn254.G2Jac
	pointJac.FromAffine(&point)
	scaledJac.FromAffine(&scaled)
	pointJac.SubAssign(&scaledJac)
	var result bn254.G2Affine
	result.FromJacobian(&pointJac)
	return result
}

func subtractG1(a, b bn254.G1Affine) bn254.G1Affine {
	var aJac, bJac bn254.G1Jac
	aJac.FromAffine(&a)
	bJac.FromAffine(&b)
	aJac.SubAssign(&bJac)
	var result bn254.G1Affine
	result.FromJacobian(&aJac)
	return result
}

func addG1(a, b bn254.G1Affine) bn254.G1Affine {
	var aJac, bJac bn254.G1Jac
	aJac.FromAffine(&a)
	bJac.FromAffine(&b)
	aJac.AddAssign(&bJac)
	var result bn254.G1Affine
	result.FromJacobian(&aJac)
	return result
}

func negG1(point bn254.G1Affine) bn254.G1Affine {
	var result bn254.G1Affine
	result.Neg(&point)
	return result
}
