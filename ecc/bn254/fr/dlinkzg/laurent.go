package dlinkzg

import (
	"errors"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

var ErrVerifyLaurentIdentity = errors.New("dlinkzg: Laurent functional identity failed")

// LocalLaurentInput contains the polynomials and public weights in one
// party's instance of Equation (local-dlinkzg-witness). Polynomials may have
// different slice lengths; missing high coefficients are interpreted as zero.
type LocalLaurentInput struct {
	G    [3][]fr.Element
	PsiQ [3][]fr.Element
	P    [3]fr.Element
	Xi   fr.Element
	HXi  []fr.Element
	PsiR []fr.Element
	Nu   fr.Element
	T0   []fr.Element
	T1   []fr.Element
	AXi  []fr.Element
	BXi  []fr.Element
}

// BuildLocalLaurent constructs S_i^lin from Equation
// (local-dlinkzg-witness). The Xi identifiers are zero based, so the three
// circuit scales are P_j, Xi*P_j, and Xi^2*P_j. This generic path accumulates
// correlations with FastOffDiagBatch in O(T log T). Protocol callers whose
// circuit queries come from Q_T use the translated-query entry point below.
// OffDiag is retained only as the simple reference used by tests.
func BuildLocalLaurent(input LocalLaurentInput) []fr.Element {
	return buildLocalLaurent(input, nil, nil, false)
}

// BuildLocalLaurentWithDomain is BuildLocalLaurent with a reusable minimal
// convolution domain supplied by preprocessing.
func BuildLocalLaurentWithDomain(input LocalLaurentInput, domain *fft.Domain) []fr.Element {
	return buildLocalLaurent(input, domain, nil, false)
}

// BuildLocalLaurentWithGeometricCircuitQueries is BuildLocalLaurentWithDomain
// specialized for translated coefficient-MLE circuit queries. ratios[j] is
// the public z_j for which PsiQ[j][k] = PsiQ[j][0] * z_j^k. The function
// checks that relation before using the linear-time geometric recurrence; an
// arbitrary non-geometric input retains the generic FFT-convolution path.
func BuildLocalLaurentWithGeometricCircuitQueries(
	input LocalLaurentInput,
	ratios [3]fr.Element,
	domain *fft.Domain,
) []fr.Element {
	return buildLocalLaurent(input, domain, &ratios, false)
}

// BuildLocalLaurentWithTranslatedCircuitQueries is the protocol-specialized
// circuit path. For the translated coefficient-MLE queries, the public
// identity
//
//	PsiQ[j][k] = ratios[j]^k / P[j]
//
// makes the P[j] factor in Equation (local-dlinkzg-witness) cancel. Protocol
// callers that derived ratios and P from Q_T may therefore omit the three
// length-T PsiQ coefficient vectors entirely. The generic checked entry point
// above remains available for arbitrary library inputs.
func BuildLocalLaurentWithTranslatedCircuitQueries(
	input LocalLaurentInput,
	ratios [3]fr.Element,
	domain *fft.Domain,
) []fr.Element {
	return buildLocalLaurent(input, domain, &ratios, true)
}

type localLaurentMonomialTerm struct {
	degree      int
	coefficient fr.Element
	right       []fr.Element
	scale       fr.Element
}

type localLaurentGeometricTerm struct {
	left  []fr.Element
	ratio fr.Element
	scale fr.Element
}

func buildLocalLaurent(
	input LocalLaurentInput,
	domain *fft.Domain,
	circuitRatios *[3]fr.Element,
	translatedQueries bool,
) []fr.Element {
	left := make([][]fr.Element, 0, 6)
	right := make([][]fr.Element, 0, 6)
	scales := make([]fr.Element, 0, 6)
	geometric := make([]localLaurentGeometricTerm, 0, len(input.G))
	maxLength := 0
	var xiPower fr.Element
	xiPower.SetOne()
	for j := range input.G {
		var scale fr.Element
		scale.Mul(&xiPower, &input.P[j])
		normalization, isGeometric := fr.Element{}, false
		if translatedQueries && circuitRatios != nil && len(input.G[j]) > 0 {
			// P[j] * (1/P[j]) cancels before constructing the witness.
			scale = xiPower
			normalization.SetOne()
			isGeometric = true
		} else if circuitRatios != nil && len(input.G[j]) == len(input.PsiQ[j]) {
			normalization, isGeometric = localLaurentGeometricNormalization(
				input.PsiQ[j], circuitRatios[j],
			)
		}
		if isGeometric {
			scale.Mul(&scale, &normalization)
			geometric = append(geometric, localLaurentGeometricTerm{
				left: input.G[j], ratio: circuitRatios[j], scale: scale,
			})
		} else {
			left = append(left, input.G[j])
			right = append(right, input.PsiQ[j])
			scales = append(scales, scale)
		}
		maxLength = maxLocalLaurentLength(maxLength, input.G[j], input.PsiQ[j])
		xiPower.Mul(&xiPower, &input.Xi)
	}

	monomials := make([]localLaurentMonomialTerm, 0, 3)
	specialLeft := [3][]fr.Element{input.HXi, input.T0, input.T1}
	specialRight := [3][]fr.Element{input.PsiR, input.AXi, input.BXi}
	specialScales := [3]fr.Element{input.Nu, fr.One(), fr.One()}
	for term := range specialLeft {
		maxLength = maxLocalLaurentLength(maxLength, specialLeft[term], specialRight[term])
		degree, coefficient, isMonomial := localLaurentMonomial(specialLeft[term])
		if isMonomial {
			monomials = append(monomials, localLaurentMonomialTerm{
				degree: degree, coefficient: coefficient,
				right: specialRight[term], scale: specialScales[term],
			})
			continue
		}
		left = append(left, specialLeft[term])
		right = append(right, specialRight[term])
		scales = append(scales, specialScales[term])
	}

	result := FastOffDiagBatchWithDomain(left, right, scales, domain)
	if maxLength >= 2 && len(result) < maxLength-1 {
		grown := make([]fr.Element, maxLength-1)
		copy(grown, result)
		result = grown
	}
	if len(geometric) > 0 {
		geometricPowers := make([]fr.Element, maxLength)
		geometricContribution := make([]fr.Element, maxLength-1)
		for term := range geometric {
			length := len(geometric[term].left) - 1
			fillFastOffDiagGeometric(
				geometricContribution[:length],
				geometric[term].left,
				geometric[term].ratio,
				geometricPowers,
			)
			addScaled(result, geometricContribution[:length], geometric[term].scale)
		}
	}
	for term := range monomials {
		addLocalLaurentMonomialOffDiag(result, monomials[term])
	}
	return normalizedCopy(result)
}

func localLaurentGeometricNormalization(polynomial []fr.Element, ratio fr.Element) (fr.Element, bool) {
	if len(polynomial) == 0 {
		return fr.Element{}, false
	}
	normalization := polynomial[0]
	expected := normalization
	for index := 1; index < len(polynomial); index++ {
		expected.Mul(&expected, &ratio)
		if !polynomial[index].Equal(&expected) {
			return fr.Element{}, false
		}
	}
	return normalization, true
}

func maxLocalLaurentLength(current int, polynomials ...[]fr.Element) int {
	for polynomial := range polynomials {
		if len(polynomials[polynomial]) > current {
			current = len(polynomials[polynomial])
		}
	}
	return current
}

func localLaurentMonomial(polynomial []fr.Element) (int, fr.Element, bool) {
	degree := -1
	var coefficient fr.Element
	for index := range polynomial {
		if polynomial[index].IsZero() {
			continue
		}
		if degree >= 0 {
			return 0, fr.Element{}, false
		}
		degree = index
		coefficient = polynomial[index]
	}
	return degree, coefficient, true
}

func addLocalLaurentMonomialOffDiag(result []fr.Element, term localLaurentMonomialTerm) {
	if term.degree < 0 || term.scale.IsZero() {
		return
	}
	// Only offsets that can reach a declared coefficient of term.right can be
	// nonzero. In the protocol these monomial terms and their public weight
	// vectors have length M even though the Laurent witness has length T.
	maxDelta := term.degree
	if rightDelta := len(term.right) - 1 - term.degree; rightDelta > maxDelta {
		maxDelta = rightDelta
	}
	if maxDelta > len(result) {
		maxDelta = len(result)
	}
	for delta := 1; delta <= maxDelta; delta++ {
		var value, contribution fr.Element
		if term.degree >= delta && term.degree-delta < len(term.right) {
			value.Set(&term.right[term.degree-delta])
		}
		if term.degree+delta < len(term.right) {
			value.Add(&value, &term.right[term.degree+delta])
		}
		contribution.Mul(&term.coefficient, &value).Mul(&contribution, &term.scale)
		result[delta-1].Add(&result[delta-1], &contribution)
	}
}

// LocalLaurentDiagonal returns a_lin,i for an honest local polynomial
// instance. It is the un-doubled constant coefficient of the symmetric
// Laurent expression. Protocol code should compare this value with the
// independently computed public target rather than use it to define that
// target.
func LocalLaurentDiagonal(input LocalLaurentInput) fr.Element {
	var diagonal fr.Element
	var xiPower fr.Element
	xiPower.SetOne()
	for j := range input.G {
		var scale fr.Element
		scale.Mul(&xiPower, &input.P[j])
		addScaledInnerProduct(&diagonal, input.G[j], input.PsiQ[j], scale)
		xiPower.Mul(&xiPower, &input.Xi)
	}
	addScaledInnerProduct(&diagonal, input.HXi, input.PsiR, input.Nu)
	addScaledInnerProduct(&diagonal, input.T0, input.AXi, fr.One())
	addScaledInnerProduct(&diagonal, input.T1, input.BXi, fr.One())
	return diagonal
}

// EvalLocalLaurentLeft evaluates the complete left side of Equation
// (dlinkzg-functional-identity) at beta. Beta must be nonzero because the
// expression also evaluates every polynomial at beta^{-1}.
func EvalLocalLaurentLeft(input LocalLaurentInput, beta fr.Element) (fr.Element, error) {
	var result fr.Element
	if beta.IsZero() {
		return result, ErrInvalidChallenge
	}
	var betaInverse fr.Element
	betaInverse.Inverse(&beta)

	var xiPower fr.Element
	xiPower.SetOne()
	for j := range input.G {
		var scale fr.Element
		scale.Mul(&xiPower, &input.P[j])
		addScaledSymmetricValue(&result, input.G[j], input.PsiQ[j], beta, betaInverse, scale)
		xiPower.Mul(&xiPower, &input.Xi)
	}
	addScaledSymmetricValue(&result, input.HXi, input.PsiR, beta, betaInverse, input.Nu)
	addScaledSymmetricValue(&result, input.T0, input.AXi, beta, betaInverse, fr.One())
	addScaledSymmetricValue(&result, input.T1, input.BXi, beta, betaInverse, fr.One())
	return result, nil
}

// DeriveLaurentInverseValue reconstructs S_i^lin(beta^{-1}) from the retained
// local left-hand value, target, and S_i^lin(beta), as in Equation
// (local-laurent-derived-value). This is the value omitted from U2.
func DeriveLaurentInverseValue(beta, left, target, witnessAtBeta fr.Element) (fr.Element, error) {
	var result fr.Element
	if beta.IsZero() {
		return result, ErrInvalidChallenge
	}
	var twiceTarget, positive fr.Element
	twiceTarget.Double(&target)
	positive.Mul(&beta, &witnessAtBeta)
	result.Sub(&left, &twiceTarget).Sub(&result, &positive).Mul(&result, &beta)
	return result, nil
}

// VerifyLocalLaurentIdentity checks the slot-local form of Equation
// (dlinkzg-functional-identity). target is supplied independently so that a
// false terminal claim cannot be hidden by recomputing the diagonal from the
// witness polynomials.
func VerifyLocalLaurentIdentity(input LocalLaurentInput, witness []fr.Element, target, beta fr.Element) error {
	left, err := EvalLocalLaurentLeft(input, beta)
	if err != nil {
		return err
	}
	var betaInverse fr.Element
	betaInverse.Inverse(&beta)

	var right, term fr.Element
	right.Double(&target)
	witnessAtBeta := Eval(witness, beta)
	term.Mul(&beta, &witnessAtBeta)
	right.Add(&right, &term)
	witnessAtBetaInverse := Eval(witness, betaInverse)
	term.Mul(&betaInverse, &witnessAtBetaInverse)
	right.Add(&right, &term)
	if !left.Equal(&right) {
		return ErrVerifyLaurentIdentity
	}
	return nil
}

func addScaledPolynomial(dst, src []fr.Element, scale fr.Element) []fr.Element {
	if len(dst) < len(src) {
		grown := make([]fr.Element, len(src))
		copy(grown, dst)
		dst = grown
	}
	addScaled(dst, src, scale)
	return dst
}

func addScaledInnerProduct(dst *fr.Element, a, b []fr.Element, scale fr.Element) {
	length := len(a)
	if len(b) < length {
		length = len(b)
	}
	for i := 0; i < length; i++ {
		var term fr.Element
		term.Mul(&a[i], &b[i]).Mul(&term, &scale)
		dst.Add(dst, &term)
	}
}

func addScaledSymmetricValue(dst *fr.Element, f, d []fr.Element, beta, betaInverse, scale fr.Element) {
	fBeta := Eval(f, beta)
	fBetaInverse := Eval(f, betaInverse)
	dBeta := Eval(d, beta)
	dBetaInverse := Eval(d, betaInverse)
	var value, term fr.Element
	value.Mul(&fBeta, &dBetaInverse)
	term.Mul(&fBetaInverse, &dBeta)
	value.Add(&value, &term).Mul(&value, &scale)
	dst.Add(dst, &value)
}
