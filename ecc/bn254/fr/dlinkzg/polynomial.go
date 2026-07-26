// Package dlinkzg contains the algebraic building blocks used by the
// distributed linear-functional compiler in the paper.  This first slice is
// deliberately small and deterministic: polynomials are represented in the
// monomial basis, with the coefficient of X^i stored at index i.
package dlinkzg

import (
	"errors"
	"fmt"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

var (
	ErrMismatchedInput = errors.New("dlinkzg: mismatched input lengths")
	ErrDuplicatePoint  = errors.New("dlinkzg: interpolation points are not distinct")
	ErrDivisionByZero  = errors.New("dlinkzg: polynomial division by zero")
	ErrInexactDivision = errors.New("dlinkzg: polynomial division has a non-zero remainder")
	ErrInvalidMLE      = errors.New("dlinkzg: coefficient vector does not fit the MLE point")
)

// SameSetInput is one polynomial and its claimed evaluations on a common
// ordered point set.
type SameSetInput struct {
	Polynomial    []fr.Element
	ClaimedValues []fr.Element
}

// SameSetResult records every polynomial produced by the exact same-set
// reduction from Equation (dlinkzg-multipoint-quotients) in the paper.
type SameSetResult struct {
	Interpolants [][]fr.Element
	Numerator    []fr.Element
	Vanishing    []fr.Element
	Quotient     []fr.Element
}

// Eval evaluates a coefficient-form polynomial at point using Horner's rule.
// The empty slice denotes the zero polynomial.
func Eval(p []fr.Element, point fr.Element) fr.Element {
	var result fr.Element
	for i := len(p) - 1; i >= 0; i-- {
		result.Mul(&result, &point).Add(&result, &p[i])
	}
	return result
}

// SyntheticDivision divides p by X-point and returns the quotient and
// remainder.  It does not mutate p.
func SyntheticDivision(p []fr.Element, point fr.Element) ([]fr.Element, fr.Element) {
	var remainder fr.Element
	if len(p) == 0 {
		return nil, remainder
	}
	if len(p) == 1 {
		return nil, p[0]
	}

	quotient := make([]fr.Element, len(p)-1)
	quotient[len(quotient)-1] = p[len(p)-1]
	for i := len(quotient) - 2; i >= 0; i-- {
		quotient[i].Mul(&quotient[i+1], &point).Add(&quotient[i], &p[i+1])
	}
	remainder.Mul(&quotient[0], &point).Add(&remainder, &p[0])
	return quotient, remainder
}

// TaylorShift returns the coefficients of p(X+shift).  The output has the
// same length as p, including any declared high zero coefficients.
func TaylorShift(p []fr.Element, shift fr.Element) []fr.Element {
	if len(p) == 0 {
		return nil
	}

	result := []fr.Element{p[len(p)-1]}
	for k := len(p) - 2; k >= 0; k-- {
		next := make([]fr.Element, len(result)+1)
		next[0].Mul(&result[0], &shift).Add(&next[0], &p[k])
		for i := 1; i < len(result); i++ {
			next[i].Mul(&result[i], &shift).Add(&next[i], &result[i-1])
		}
		next[len(result)] = result[len(result)-1]
		result = next
	}
	return result
}

// EqualityWeights returns (chi_i(point))_i in little-endian Boolean-index
// order: coordinate k controls bit k of i.
func EqualityWeights(point []fr.Element) []fr.Element {
	weights := make([]fr.Element, 1)
	weights[0].SetOne()
	one := fr.One()
	for k := range point {
		next := make([]fr.Element, 2*len(weights))
		var oneMinus fr.Element
		oneMinus.Sub(&one, &point[k])
		for i := range weights {
			next[i].Mul(&weights[i], &oneMinus)
			next[i+len(weights)].Mul(&weights[i], &point[k])
		}
		weights = next
	}
	return weights
}

// MLEEval evaluates a coefficient vector's multilinear extension.  Missing
// high coefficients are interpreted as zero.
func MLEEval(coefficients, point []fr.Element) (fr.Element, error) {
	var result fr.Element
	if len(point) >= 63 || uint64(len(coefficients)) > uint64(1)<<uint(len(point)) {
		return result, ErrInvalidMLE
	}
	weights := EqualityWeights(point)
	var term fr.Element
	for i := range coefficients {
		term.Mul(&coefficients[i], &weights[i])
		result.Add(&result, &term)
	}
	return result, nil
}

// OffDiag computes the positive off-diagonal witness from Equation
// (offdiag).  Shorter inputs are padded with zeros to T=max(len(f),len(d)).
func OffDiag(f, d []fr.Element) []fr.Element {
	t := len(f)
	if len(d) > t {
		t = len(d)
	}
	if t < 2 {
		return nil
	}

	result := make([]fr.Element, t-1)
	for delta := 1; delta < t; delta++ {
		for k := 0; k+delta < t; k++ {
			var term fr.Element
			if k+delta < len(f) && k < len(d) {
				term.Mul(&f[k+delta], &d[k])
				result[delta-1].Add(&result[delta-1], &term)
			}
			if k < len(f) && k+delta < len(d) {
				term.Mul(&f[k], &d[k+delta])
				result[delta-1].Add(&result[delta-1], &term)
			}
		}
	}
	return result
}

// Interpolate returns the unique degree-<len(points) polynomial taking the
// supplied values.  Points must be pairwise distinct.
func Interpolate(points, values []fr.Element) ([]fr.Element, error) {
	if len(points) != len(values) {
		return nil, ErrMismatchedInput
	}
	if len(points) == 0 {
		return zeroPolynomial(), nil
	}

	result := make([]fr.Element, len(points))
	for i := range points {
		basis := []fr.Element{fr.One()}
		var denominator fr.Element
		denominator.SetOne()
		for j := range points {
			if i == j {
				continue
			}
			var difference fr.Element
			difference.Sub(&points[i], &points[j])
			if difference.IsZero() {
				return nil, ErrDuplicatePoint
			}
			denominator.Mul(&denominator, &difference)
			var negPoint fr.Element
			negPoint.Neg(&points[j])
			basis = multiply(basis, []fr.Element{negPoint, fr.One()})
		}
		var scale fr.Element
		scale.Div(&values[i], &denominator)
		addScaled(result, basis, scale)
	}
	return result, nil
}

// VanishingPolynomial returns product_i (X-points[i]).
func VanishingPolynomial(points []fr.Element) []fr.Element {
	result := []fr.Element{fr.One()}
	for i := range points {
		var negPoint fr.Element
		negPoint.Neg(&points[i])
		result = multiply(result, []fr.Element{negPoint, fr.One()})
	}
	return result
}

// ExactQuotient divides numerator by divisor and fails if the remainder is
// nonzero.  The zero polynomial has the canonical representation [0].
func ExactQuotient(numerator, divisor []fr.Element) ([]fr.Element, error) {
	numerator = normalizedCopy(numerator)
	divisor = normalizedCopy(divisor)
	if isZeroPolynomial(divisor) {
		return nil, ErrDivisionByZero
	}
	if isZeroPolynomial(numerator) {
		return zeroPolynomial(), nil
	}
	if len(numerator) < len(divisor) {
		return nil, ErrInexactDivision
	}

	remainder := append([]fr.Element(nil), numerator...)
	quotient := make([]fr.Element, len(numerator)-len(divisor)+1)
	var inverseLeading fr.Element
	inverseLeading.Inverse(&divisor[len(divisor)-1])
	for k := len(remainder) - 1; k >= len(divisor)-1; k-- {
		qIndex := k - len(divisor) + 1
		quotient[qIndex].Mul(&remainder[k], &inverseLeading)
		for j := range divisor {
			var term fr.Element
			term.Mul(&quotient[qIndex], &divisor[j])
			remainder[qIndex+j].Sub(&remainder[qIndex+j], &term)
		}
	}
	for i := 0; i < len(divisor)-1 && i < len(remainder); i++ {
		if !remainder[i].IsZero() {
			return nil, ErrInexactDivision
		}
	}
	return normalizedCopy(quotient), nil
}

// BuildSameSetQuotient implements the ordered kappa batch in Equation
// (dlinkzg-multipoint-quotients).  Identifiers are the zero-based input
// positions.  Exact divisibility is checked, rather than assumed.
func BuildSameSetQuotient(inputs []SameSetInput, points []fr.Element, kappa fr.Element) (SameSetResult, error) {
	var result SameSetResult
	if len(inputs) == 0 || len(points) == 0 {
		return result, fmt.Errorf("%w: empty same-set batch", ErrMismatchedInput)
	}

	result.Interpolants = make([][]fr.Element, len(inputs))
	var power fr.Element
	power.SetOne()
	for i := range inputs {
		if len(inputs[i].ClaimedValues) != len(points) {
			return SameSetResult{}, ErrMismatchedInput
		}
		interpolant, err := Interpolate(points, inputs[i].ClaimedValues)
		if err != nil {
			return SameSetResult{}, err
		}
		result.Interpolants[i] = interpolant
		difference := subtract(inputs[i].Polynomial, interpolant)
		if len(result.Numerator) < len(difference) {
			grown := make([]fr.Element, len(difference))
			copy(grown, result.Numerator)
			result.Numerator = grown
		}
		addScaled(result.Numerator, difference, power)
		power.Mul(&power, &kappa)
	}
	result.Numerator = normalizedCopy(result.Numerator)
	result.Vanishing = VanishingPolynomial(points)
	quotient, err := ExactQuotient(result.Numerator, result.Vanishing)
	if err != nil {
		return SameSetResult{}, err
	}
	result.Quotient = quotient
	return result, nil
}

func zeroPolynomial() []fr.Element {
	return make([]fr.Element, 1)
}

func normalizedCopy(p []fr.Element) []fr.Element {
	last := len(p) - 1
	for last > 0 && p[last].IsZero() {
		last--
	}
	if last < 0 {
		return zeroPolynomial()
	}
	result := make([]fr.Element, last+1)
	copy(result, p[:last+1])
	return result
}

func isZeroPolynomial(p []fr.Element) bool {
	if len(p) == 0 {
		return true
	}
	for i := range p {
		if !p[i].IsZero() {
			return false
		}
	}
	return true
}

func addScaled(dst, src []fr.Element, scale fr.Element) {
	for i := range src {
		var term fr.Element
		term.Mul(&src[i], &scale)
		dst[i].Add(&dst[i], &term)
	}
}

func subtract(a, b []fr.Element) []fr.Element {
	length := len(a)
	if len(b) > length {
		length = len(b)
	}
	result := make([]fr.Element, length)
	copy(result, a)
	for i := range b {
		result[i].Sub(&result[i], &b[i])
	}
	return normalizedCopy(result)
}

func multiply(a, b []fr.Element) []fr.Element {
	if len(a) == 0 || len(b) == 0 || isZeroPolynomial(a) || isZeroPolynomial(b) {
		return zeroPolynomial()
	}
	result := make([]fr.Element, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			var term fr.Element
			term.Mul(&a[i], &b[j])
			result[i+j].Add(&result[i+j], &term)
		}
	}
	return normalizedCopy(result)
}
