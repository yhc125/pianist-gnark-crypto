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

	verifierYPowers   = 2
	verifierZPowers   = 4
	verifierG1ZPowers = 3
)

// PartyRowSRS contains the three O(T) G1 rows needed by one prover.
// G1SemanticRow[j] equals [tauY^Rank tauX^j]_1, where tauX=tauZ+sigma, for
// semantic X-coordinate commitments in W0--W2 and the fixed PIOP columns.
// G1Row[j] equals [tauY^Rank tauZ^j]_1 for translated source commitments and
// piZ, while G1ZShared[j] equals [tauZ^j]_1 for the logical Y^0 commitments in
// U0--U3. It contains no other party rank and no G2 power.
type PartyRowSRS struct {
	Rank          int
	Parties       int
	DegreeBound   int
	G1SemanticRow []bn254.G1Affine
	G1Row         []bn254.G1Affine
	G1ZShared     []bn254.G1Affine
}

// CoordinatorSRS contains the O(M) root-only Y column and the O(M) Z prefix
// needed to commit t_0, t_1, and h_xi, all of which have degree less than M.
// It contains no mixed party row and no G2 power.
type CoordinatorSRS struct {
	Parties   int
	G1Y       []bn254.G1Affine
	G1ZShared []bn254.G1Affine
}

// VerifierSRS is the constant-size verifier view used by the source link and
// the degree-three/degree-two same-set batches. G1ZVerifier contains powers 0
// through 2 for public interpolant commitments, G2Y contains powers 0 and 1,
// and G2Z contains powers 0 through 3.
type VerifierSRS struct {
	G1ZVerifier []bn254.G1Affine
	G2Y         []bn254.G2Affine
	G2Z         []bn254.G2Affine
}

// NewDeterministicPartyRowSRS preserves the original unshifted benchmark/test
// constructor. Its semantic and native mixed rows coincide because sigma=0.
// New code that uses the protocol's semantic X coordinate should call
// NewDeterministicPartyRowSRSWithShift.
func NewDeterministicPartyRowSRS(parties, degreeBound, rank int, tauY, tauZ fr.Element) (*PartyRowSRS, error) {
	return NewDeterministicPartyRowSRSWithShift(
		parties,
		degreeBound,
		rank,
		tauY,
		tauZ,
		fr.Element{},
	)
}

// NewDeterministicPartyRowSRSWithShift constructs the semantic mixed X row,
// native mixed Z row, and shared Y^0 Z row directly, without materializing an
// M-by-T rectangle. It is benchmark/test-only; see
// DeterministicSplitSRSNotice. The trapdoors and sigma are used transiently
// and are not retained in PartyRowSRS.
func NewDeterministicPartyRowSRSWithShift(parties, degreeBound, rank int, tauY, tauZ, sigma fr.Element) (*PartyRowSRS, error) {
	if parties < 2 || degreeBound < verifierZPowers || rank < 0 || rank >= parties {
		return nil, ErrInvalidSRS
	}
	_, _, generator1, _ := bn254.Generators()
	zScalars := powers(tauZ, degreeBound)
	g1ZShared := batchScalarMultiplicationG1(&generator1, zScalars)
	yRank := fieldExponent(tauY, rank)
	nativeMixedScalars := append([]fr.Element(nil), zScalars...)
	for j := range nativeMixedScalars {
		nativeMixedScalars[j].Mul(&nativeMixedScalars[j], &yRank)
	}
	var tauX fr.Element
	tauX.Add(&tauZ, &sigma)
	semanticMixedScalars := powers(tauX, degreeBound)
	for j := range semanticMixedScalars {
		semanticMixedScalars[j].Mul(&semanticMixedScalars[j], &yRank)
	}
	return &PartyRowSRS{
		Rank:          rank,
		Parties:       parties,
		DegreeBound:   degreeBound,
		G1SemanticRow: batchScalarMultiplicationG1(&generator1, semanticMixedScalars),
		G1Row:         batchScalarMultiplicationG1(&generator1, nativeMixedScalars),
		G1ZShared:     g1ZShared,
	}, nil
}

// NewDeterministicCoordinatorSRS constructs the root Y column and length-M Z prefix
// without materializing a mixed party rectangle. It is benchmark/test-only;
// see DeterministicSplitSRSNotice. Neither trapdoor is retained.
func NewDeterministicCoordinatorSRS(parties int, tauY, tauZ fr.Element) (*CoordinatorSRS, error) {
	if parties < 2 {
		return nil, ErrInvalidSRS
	}
	_, _, generator1, _ := bn254.Generators()
	return &CoordinatorSRS{
		Parties:   parties,
		G1Y:       batchScalarMultiplicationG1(&generator1, powers(tauY, parties)),
		G1ZShared: batchScalarMultiplicationG1(&generator1, powers(tauZ, parties)),
	}, nil
}

// NewDeterministicVerifierSRS constructs the constant-size verifier view. It
// is benchmark/test-only; see DeterministicSplitSRSNotice. Neither trapdoor is
// retained.
func NewDeterministicVerifierSRS(tauY, tauZ fr.Element) *VerifierSRS {
	_, _, generator1, generator2 := bn254.Generators()
	return &VerifierSRS{
		G1ZVerifier: batchScalarMultiplicationG1(
			&generator1,
			powers(tauZ, verifierG1ZPowers),
		),
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
		srs.Rank < 0 || srs.Rank >= srs.Parties ||
		len(srs.G1SemanticRow) != srs.DegreeBound ||
		len(srs.G1Row) != srs.DegreeBound || len(srs.G1ZShared) != srs.DegreeBound {
		return ErrInvalidSRS
	}
	return nil
}

// Validate checks that the coordinator view has exactly the declared Y
// column and no rectangular row material.
func (srs *CoordinatorSRS) Validate() error {
	if srs == nil || srs.Parties < 2 ||
		len(srs.G1Y) != srs.Parties || len(srs.G1ZShared) != srs.Parties {
		return ErrInvalidSRS
	}
	return nil
}

// Validate checks the constant verifier-key shape required by the protocol.
func (srs *VerifierSRS) Validate() error {
	if srs == nil || len(srs.G1ZVerifier) != verifierG1ZPowers ||
		len(srs.G2Y) != verifierYPowers || len(srs.G2Z) != verifierZPowers {
		return ErrInvalidSRS
	}
	return nil
}

// CommitSemantic commits p in this party's semantic X coordinate:
// [sum_j p[j] tauY^Rank (tauZ+sigma)^j]_1. It is used by W0--W2 and the fixed
// PIOP columns. Equivalently, it commits FastTaylorShift(p,sigma) through the
// native mixed row.
func (srs *PartyRowSRS) CommitSemantic(p []fr.Element) (bn254.G1Affine, error) {
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
		srs.G1SemanticRow[:len(p)],
		p,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

// CommitRow commits p in this party's native Z coordinate:
// [sum_j p[j] tauY^Rank tauZ^j]_1.
// It is retained separately from CommitSemantic for translated sources and
// piZ.
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

// CommitZ commits a logical univariate polynomial in the shared Y^0 row:
// [sum_j p[j] tauZ^j]_1. This is distinct from CommitRow when Rank is nonzero
// and is used for g_{j,i}, S_i^lin, W_{G,i}, and W_{L,i}.
func (srs *PartyRowSRS) CommitZ(p []fr.Element) (bn254.G1Affine, error) {
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
		srs.G1ZShared[:len(p)],
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

// CommitZ commits a coordinator polynomial in the shared Y^0 row.
func (srs *CoordinatorSRS) CommitZ(coefficients []fr.Element) (bn254.G1Affine, error) {
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
		srs.G1ZShared[:len(coefficients)],
		coefficients,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

// CommitZ commits a public degree-at-most-two interpolant using only the
// verifier view.
func (srs *VerifierSRS) CommitZ(coefficients []fr.Element) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if err := srs.Validate(); err != nil {
		return result, err
	}
	if len(coefficients) > len(srs.G1ZVerifier) {
		return result, ErrPolynomialTooWide
	}
	if len(coefficients) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(
		srs.G1ZVerifier[:len(coefficients)],
		coefficients,
		ecc.MultiExpConfig{ScalarsMont: true},
	)
	return result, err
}

// FoldSameSetCommitments derives a same-set numerator commitment using only
// the constant verifier G1 view. Consequently every public interpolant must
// have degree at most two.
func (srs *VerifierSRS) FoldSameSetCommitments(commitments []bn254.G1Affine, interpolants [][]fr.Element, kappa fr.Element) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if len(commitments) != len(interpolants) {
		return result, ErrMismatchedInput
	}
	if err := srs.Validate(); err != nil {
		return result, err
	}
	var resultJac bn254.G1Jac
	power := fr.One()
	for i := range commitments {
		interpolantCommitment, err := srs.CommitZ(interpolants[i])
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

// VerifySourceLink checks a source-link proof using only the constant-size
// verifier view.
func (srs *VerifierSRS) VerifySourceLink(commitment bn254.G1Affine, beta, zChallenge fr.Element, proof SourceLinkProof) error {
	if err := srs.Validate(); err != nil {
		return err
	}
	a0 := subtractG1Scalar(commitment, srs.G1ZVerifier[0], proof.ClaimedValue)
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

	a0 := subtractG1Scalar(statement.SourceCommitment, srs.G1ZVerifier[0], statement.SourceValue)
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
