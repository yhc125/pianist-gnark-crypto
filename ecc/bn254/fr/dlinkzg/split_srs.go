package dlinkzg

import (
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

const (
	// DeterministicSplitSRSNotice marks the constructors in this file as
	// benchmark/test helpers. Production deployments must derive the same role
	// views from an MPC-generated rectangular SRS and erase every trapdoor.
	DeterministicSplitSRSNotice = "deterministic split SRS: benchmark/test only; use an MPC-generated SRS in production"

	verifierYPowers = 2
	verifierZPowers = 4
)

// PartyRowSRS is the rank-local row of a rectangular monomial SRS. G1Row[j]
// equals [tauY^Rank tauZ^j]_1. It contains no other party row and no G2 power.
type PartyRowSRS struct {
	Rank        int
	Parties     int
	DegreeBound int
	G1Row       []bn254.G1Affine
}

// CoordinatorSRS is the root-only Y column of a rectangular monomial SRS.
// G1Y[i] equals [tauY^i]_1. It contains no Z row and no G2 power.
type CoordinatorSRS struct {
	Parties int
	G1Y     []bn254.G1Affine
}

// VerifierSRS is the constant-size verifier view used by the source link and
// the degree-three/degree-two same-set batches. G2Y contains powers 0 and 1;
// G2Z contains powers 0 through 3.
type VerifierSRS struct {
	G1Base bn254.G1Affine
	G2Y    []bn254.G2Affine
	G2Z    []bn254.G2Affine
}

// NewDeterministicPartyRowSRS constructs one party row directly, without
// materializing the M-by-T rectangle. It is benchmark/test-only; see
// DeterministicSplitSRSNotice. The trapdoors are used transiently and are not
// retained in PartyRowSRS.
func NewDeterministicPartyRowSRS(parties, degreeBound, rank int, tauY, tauZ fr.Element) (*PartyRowSRS, error) {
	if parties < 2 || degreeBound < verifierZPowers || rank < 0 || rank >= parties {
		return nil, ErrInvalidSRS
	}
	_, _, generator1, _ := bn254.Generators()
	zScalars := powers(tauZ, degreeBound)
	yRank := fieldExponent(tauY, rank)
	for j := range zScalars {
		zScalars[j].Mul(&zScalars[j], &yRank)
	}
	return &PartyRowSRS{
		Rank:        rank,
		Parties:     parties,
		DegreeBound: degreeBound,
		G1Row:       batchScalarMultiplicationG1(&generator1, zScalars),
	}, nil
}

// NewDeterministicCoordinatorSRS constructs the root-only Y column without
// materializing any party Z row. It is benchmark/test-only; see
// DeterministicSplitSRSNotice. The trapdoor is not retained.
func NewDeterministicCoordinatorSRS(parties int, tauY fr.Element) (*CoordinatorSRS, error) {
	if parties < 2 {
		return nil, ErrInvalidSRS
	}
	_, _, generator1, _ := bn254.Generators()
	return &CoordinatorSRS{
		Parties: parties,
		G1Y:     batchScalarMultiplicationG1(&generator1, powers(tauY, parties)),
	}, nil
}

// NewDeterministicVerifierSRS constructs the constant-size verifier view. It
// is benchmark/test-only; see DeterministicSplitSRSNotice. Neither trapdoor is
// retained.
func NewDeterministicVerifierSRS(tauY, tauZ fr.Element) *VerifierSRS {
	_, _, generator1, generator2 := bn254.Generators()
	return &VerifierSRS{
		G1Base: generator1,
		G2Y: batchScalarMultiplicationG2(
			&generator2,
			powers(tauY, verifierYPowers),
		),
		G2Z: batchScalarMultiplicationG2(
			&generator2,
			powers(tauZ, verifierZPowers),
		),
	}
}

// Validate checks that the party view has exactly one complete rank row.
func (srs *PartyRowSRS) Validate() error {
	if srs == nil || srs.Parties < 2 || srs.DegreeBound < verifierZPowers ||
		srs.Rank < 0 || srs.Rank >= srs.Parties || len(srs.G1Row) != srs.DegreeBound {
		return ErrInvalidSRS
	}
	return nil
}

// Validate checks that the coordinator view has exactly the declared Y
// column and no rectangular row material.
func (srs *CoordinatorSRS) Validate() error {
	if srs == nil || srs.Parties < 2 || len(srs.G1Y) != srs.Parties {
		return ErrInvalidSRS
	}
	return nil
}

// Validate checks the constant verifier-key shape required by the protocol.
func (srs *VerifierSRS) Validate() error {
	if srs == nil || len(srs.G2Y) != verifierYPowers || len(srs.G2Z) != verifierZPowers {
		return ErrInvalidSRS
	}
	return nil
}

// CommitRow commits p in this party's fixed rank:
// [sum_j p[j] tauY^Rank tauZ^j]_1.
func (srs *PartyRowSRS) CommitRow(p []fr.Element) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if err := srs.Validate(); err != nil {
		return result, err
	}
	if len(p) > srs.DegreeBound {
		return result, ErrPolynomialTooWide
	}
	if len(p) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(
		srs.G1Row[:len(p)],
		p,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

// CommitY commits a coefficient polynomial in the root-only Y column:
// [sum_i coefficients[i] tauY^i]_1.
func (srs *CoordinatorSRS) CommitY(coefficients []fr.Element) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if err := srs.Validate(); err != nil {
		return result, err
	}
	if len(coefficients) > srs.Parties {
		return result, ErrPolynomialTooWide
	}
	if len(coefficients) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(
		srs.G1Y[:len(coefficients)],
		coefficients,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

// VerifySourceLink checks a source-link proof using only the constant-size
// verifier view.
func (srs *VerifierSRS) VerifySourceLink(commitment bn254.G1Affine, beta, zChallenge fr.Element, proof SourceLinkProof) error {
	if err := srs.Validate(); err != nil {
		return err
	}
	a0 := subtractG1Scalar(commitment, srs.G1Base, proof.ClaimedValue)
	zDirection := subtractG2Scalar(srs.G2Z[1], srs.G2Z[0], zChallenge)
	yDirection := subtractG2Scalar(srs.G2Y[1], srs.G2Y[0], beta)
	ok, err := bn254.PairingCheck(
		[]bn254.G1Affine{a0, negG1(proof.PiZ), negG1(proof.PiY)},
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

// VerifyDeltaBatch checks the final five-pairing product using only the
// constant-size verifier view. Numerator commitments are part of statement;
// callers derive them from the public polynomial commitments and interpolants.
func (srs *VerifierSRS) VerifyDeltaBatch(statement DeltaBatchStatement, proof DeltaBatchProof, delta fr.Element) error {
	if delta.IsZero() {
		return ErrInvalidChallenge
	}
	if err := srs.Validate(); err != nil {
		return err
	}
	zG, err := srs.commitZInG2(statement.VanishingG)
	if err != nil {
		return err
	}
	zL, err := srs.commitZInG2(statement.VanishingL)
	if err != nil {
		return err
	}

	a0 := subtractG1Scalar(statement.SourceCommitment, srs.G1Base, statement.SourceValue)
	var deltaSquared fr.Element
	deltaSquared.Square(&delta)
	left := addG1(a0, scaleG1(statement.NumeratorG, delta))
	left = addG1(left, scaleG1(statement.NumeratorL, deltaSquared))
	zDirection := subtractG2Scalar(srs.G2Z[1], srs.G2Z[0], statement.ZChallenge)
	yDirection := subtractG2Scalar(srs.G2Y[1], srs.G2Y[0], statement.Beta)
	ok, err := bn254.PairingCheck(
		[]bn254.G1Affine{
			left,
			negG1(proof.PiZ),
			negG1(proof.PiY),
			negG1(scaleG1(proof.WG, delta)),
			negG1(scaleG1(proof.WL, deltaSquared)),
		},
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

func (srs *VerifierSRS) commitZInG2(coefficients []fr.Element) (bn254.G2Affine, error) {
	var result bn254.G2Affine
	if err := srs.Validate(); err != nil {
		return result, err
	}
	if len(coefficients) > len(srs.G2Z) {
		return result, ErrPolynomialTooWide
	}
	if len(coefficients) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(
		srs.G2Z[:len(coefficients)],
		coefficients,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

func fieldExponent(base fr.Element, exponent int) fr.Element {
	result := fr.One()
	for exponent > 0 {
		if exponent&1 == 1 {
			result.Mul(&result, &base)
		}
		base.Square(&base)
		exponent >>= 1
	}
	return result
}

// The generated batch-scalar-multiplication helpers consume canonical scalar
// limbs, whereas fr.Element arithmetic stores Montgomery-form limbs.
func batchScalarMultiplicationG1(base *bn254.G1Affine, scalars []fr.Element) []bn254.G1Affine {
	regular := append([]fr.Element(nil), scalars...)
	for i := range regular {
		regular[i].FromMont()
	}
	return bn254.BatchScalarMultiplicationG1(base, regular)
}

func batchScalarMultiplicationG2(base *bn254.G2Affine, scalars []fr.Element) []bn254.G2Affine {
	regular := append([]fr.Element(nil), scalars...)
	for i := range regular {
		regular[i].FromMont()
	}
	return bn254.BatchScalarMultiplicationG2(base, regular)
}
