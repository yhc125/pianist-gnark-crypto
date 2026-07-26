package dlinkzg

import (
	"errors"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
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
// circuit scales are P_j, Xi*P_j, and Xi^2*P_j.
func BuildLocalLaurent(input LocalLaurentInput) []fr.Element {
	var witness []fr.Element
	var xiPower fr.Element
	xiPower.SetOne()
	for j := range input.G {
		var scale fr.Element
		scale.Mul(&xiPower, &input.P[j])
		witness = addScaledPolynomial(witness, OffDiag(input.G[j], input.PsiQ[j]), scale)
		xiPower.Mul(&xiPower, &input.Xi)
	}
	witness = addScaledPolynomial(witness, OffDiag(input.HXi, input.PsiR), input.Nu)
	witness = addScaledPolynomial(witness, OffDiag(input.T0, input.AXi), fr.One())
	witness = addScaledPolynomial(witness, OffDiag(input.T1, input.BXi), fr.One())
	return normalizedCopy(witness)
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
