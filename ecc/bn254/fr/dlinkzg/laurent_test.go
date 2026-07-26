package dlinkzg

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestLocalLaurentFunctionalIdentityAndCanonicalInverse(t *testing.T) {
	input, target := fixedLocalLaurentInstance(t)
	witness := BuildLocalLaurent(input)
	beta := element(19)

	if err := VerifyLocalLaurentIdentity(input, witness, target, beta); err != nil {
		t.Fatalf("honest Laurent identity failed: %v", err)
	}
	left, err := EvalLocalLaurentLeft(input, beta)
	if err != nil {
		t.Fatal(err)
	}
	witnessAtBeta := Eval(witness, beta)
	derived, err := DeriveLaurentInverseValue(beta, left, target, witnessAtBeta)
	if err != nil {
		t.Fatal(err)
	}
	var betaInverse fr.Element
	betaInverse.Inverse(&beta)
	want := Eval(witness, betaInverse)
	if !derived.Equal(&want) {
		t.Fatal("canonical beta-inverse value differs from the committed witness evaluation")
	}

	diagonal := LocalLaurentDiagonal(input)
	if !diagonal.Equal(&target) {
		t.Fatal("independent paper target differs from the Laurent diagonal")
	}
}

func TestLocalLaurentIdentityRejectsTampering(t *testing.T) {
	input, target := fixedLocalLaurentInstance(t)
	witness := BuildLocalLaurent(input)
	beta := element(19)

	tamperedTarget := target
	tamperedTarget.Add(&tamperedTarget, onePtr())
	if err := VerifyLocalLaurentIdentity(input, witness, tamperedTarget, beta); !errors.Is(err, ErrVerifyLaurentIdentity) {
		t.Fatalf("tampered target was not rejected: %v", err)
	}

	tamperedWitness := append([]fr.Element(nil), witness...)
	tamperedWitness[0].Add(&tamperedWitness[0], onePtr())
	if err := VerifyLocalLaurentIdentity(input, tamperedWitness, target, beta); !errors.Is(err, ErrVerifyLaurentIdentity) {
		t.Fatalf("tampered witness was not rejected: %v", err)
	}

	tamperedInput := input
	tamperedInput.G[1] = append([]fr.Element(nil), input.G[1]...)
	tamperedInput.G[1][2].Add(&tamperedInput.G[1][2], onePtr())
	if err := VerifyLocalLaurentIdentity(tamperedInput, witness, target, beta); !errors.Is(err, ErrVerifyLaurentIdentity) {
		t.Fatalf("tampered source polynomial was not rejected: %v", err)
	}

	zero := element(0)
	if _, err := EvalLocalLaurentLeft(input, zero); !errors.Is(err, ErrInvalidChallenge) {
		t.Fatalf("zero beta was not rejected by the Laurent evaluator: %v", err)
	}
	if _, err := DeriveLaurentInverseValue(zero, fr.Element{}, target, fr.Element{}); !errors.Is(err, ErrInvalidChallenge) {
		t.Fatalf("zero beta was not rejected by the inverse derivation: %v", err)
	}
}

func TestBuildLocalLaurentIsAdditive(t *testing.T) {
	first, _ := fixedLocalLaurentInstance(t)
	second := first
	for j := range second.G {
		second.G[j] = scalePolynomial(first.G[j], element(uint64(j+2)))
	}
	second.HXi = scalePolynomial(first.HXi, element(7))
	second.T0 = scalePolynomial(first.T0, element(11))
	second.T1 = scalePolynomial(first.T1, element(13))

	combined := first
	for j := range combined.G {
		combined.G[j] = addPolynomials(first.G[j], second.G[j])
	}
	combined.HXi = addPolynomials(first.HXi, second.HXi)
	combined.T0 = addPolynomials(first.T0, second.T0)
	combined.T1 = addPolynomials(first.T1, second.T1)

	want := addPolynomials(BuildLocalLaurent(first), BuildLocalLaurent(second))
	got := BuildLocalLaurent(combined)
	requirePolynomialEqual(t, want, got)
}

func TestBuildLocalLaurentFastMatchesNaiveRandomized(t *testing.T) {
	rng := rand.New(rand.NewSource(0xD11C0DE))
	lengths := []int{0, 1, 2, 3, 5, 8, 13, 16, 17, 31}
	for trial := 0; trial < 80; trial++ {
		input := randomLocalLaurentInput(rng, lengths)
		want := buildLocalLaurentNaive(input)
		got := BuildLocalLaurent(input)
		requirePolynomialEqual(t, want, got)
	}
}

func TestBuildLocalLaurentFastRandomizedAdditivityAndIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(0xADD1717E))
	lengths := []int{0, 1, 2, 4, 7, 8, 11, 16, 23}
	for trial := 0; trial < 40; trial++ {
		first := randomLocalLaurentInput(rng, lengths)
		second := randomLocalLaurentInput(rng, lengths)
		copyLaurentWeights(&second, first)
		combined := addLocalLaurentSources(first, second)

		wantSum := addPolynomials(BuildLocalLaurent(first), BuildLocalLaurent(second))
		requirePolynomialEqual(t, wantSum, BuildLocalLaurent(combined))

		for _, input := range []LocalLaurentInput{first, second, combined} {
			witness := BuildLocalLaurent(input)
			target := LocalLaurentDiagonal(input)
			beta := randomNonzeroElement(rng)
			if err := VerifyLocalLaurentIdentity(input, witness, target, beta); err != nil {
				t.Fatalf("trial %d randomized Laurent identity: %v", trial, err)
			}
		}
	}
}

func buildLocalLaurentNaive(input LocalLaurentInput) []fr.Element {
	var witness []fr.Element
	xiPower := fr.One()
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

func randomLocalLaurentInput(rng *rand.Rand, lengths []int) LocalLaurentInput {
	var input LocalLaurentInput
	for j := range input.G {
		input.G[j] = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
		input.PsiQ[j] = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
		input.P[j] = randomElement(rng)
	}
	input.Xi = randomElement(rng)
	input.HXi = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	input.PsiR = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	input.Nu = randomElement(rng)
	input.T0 = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	input.T1 = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	input.AXi = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	input.BXi = randomLaurentPolynomial(rng, chooseLength(rng, lengths))
	return input
}

func randomLaurentPolynomial(rng *rand.Rand, length int) []fr.Element {
	result := make([]fr.Element, length)
	for i := range result {
		result[i] = randomElement(rng)
	}
	// Preserve declared high zero coefficients in many cases. FastOffDiag and
	// OffDiag must agree on the declared length, not only the normalized degree.
	if length > 0 {
		zeros := rng.Intn(length + 1)
		for i := length - zeros; i < length; i++ {
			result[i].SetZero()
		}
	}
	return result
}

func randomElement(rng *rand.Rand) fr.Element {
	var result fr.Element
	result.SetUint64(rng.Uint64())
	return result
}

func randomNonzeroElement(rng *rand.Rand) fr.Element {
	for {
		result := randomElement(rng)
		if !result.IsZero() {
			return result
		}
	}
}

func chooseLength(rng *rand.Rand, lengths []int) int {
	return lengths[rng.Intn(len(lengths))]
}

func copyLaurentWeights(destination *LocalLaurentInput, source LocalLaurentInput) {
	destination.PsiQ = source.PsiQ
	destination.P = source.P
	destination.Xi = source.Xi
	destination.PsiR = source.PsiR
	destination.Nu = source.Nu
	destination.AXi = source.AXi
	destination.BXi = source.BXi
}

func addLocalLaurentSources(first, second LocalLaurentInput) LocalLaurentInput {
	combined := first
	for j := range combined.G {
		combined.G[j] = addPolynomials(first.G[j], second.G[j])
	}
	combined.HXi = addPolynomials(first.HXi, second.HXi)
	combined.T0 = addPolynomials(first.T0, second.T0)
	combined.T1 = addPolynomials(first.T1, second.T1)
	return combined
}

func fixedLocalLaurentInstance(t *testing.T) (LocalLaurentInput, fr.Element) {
	t.Helper()
	const slot = 2
	r := elements(2, 3)
	chiR := EqualityWeights(r)[slot]
	rows := [3][]fr.Element{
		elements(1, 4, 2, 8, 5, 7, 3, 6),
		elements(9, 2, 6, 5, 3, 5, 8, 9),
		elements(7, 9, 3, 2, 3, 8, 4, 6),
	}
	qPoints := [3][]fr.Element{
		elements(2, 5, 7),
		elements(3, 4, 6),
		elements(5, 8, 9),
	}
	pFactors := [3]fr.Element{element(11), element(13), element(17)}
	xi := element(5)
	nu := element(7)
	zChallenge := element(23)

	var input LocalLaurentInput
	input.Xi = xi
	input.Nu = nu
	input.P = pFactors
	input.PsiR = EqualityWeights(r)
	var dLink fr.Element
	var xiPower fr.Element
	xiPower.SetOne()
	for j := range rows {
		input.G[j] = scalePolynomial(rows[j], chiR)
		input.PsiQ[j] = EqualityWeights(qPoints[j])
		d := Eval(rows[j], zChallenge)
		var term fr.Element
		term.Mul(&xiPower, &d)
		dLink.Add(&dLink, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	input.HXi = monomial(slot, dLink)

	t0Value := element(29)
	t1Value := element(31)
	input.T0 = monomial(slot, t0Value)
	input.T1 = monomial(slot, t1Value)
	input.AXi = make([]fr.Element, 4)
	input.BXi = make([]fr.Element, 4)
	treePoints := [5][]fr.Element{
		elements(2, 7),
		elements(3, 8),
		elements(4, 9),
		elements(5, 10),
		elements(6, 11),
	}
	qTree := [5]fr.Element{element(2), element(3), element(5), element(7), element(11)}
	xiPower = element(1)
	for exponent := 0; exponent < 3; exponent++ {
		xiPower.Mul(&xiPower, &xi)
	}
	for c := range treePoints {
		weights := EqualityWeights(treePoints[c])
		var oneMinusQ, aScale, bScale fr.Element
		oneMinusQ.Sub(onePtr(), &qTree[c])
		aScale.Mul(&xiPower, &oneMinusQ)
		bScale.Mul(&xiPower, &qTree[c])
		addScaled(input.AXi, weights, aScale)
		addScaled(input.BXi, weights, bScale)
		xiPower.Mul(&xiPower, &xi)
	}

	// Compute a_lin,i from the independent local targets in Appendix B rather
	// than by calling LocalLaurentDiagonal.
	var target fr.Element
	xiPower.SetOne()
	for j := range rows {
		rowMLE, err := MLEEval(rows[j], qPoints[j])
		if err != nil {
			t.Fatal(err)
		}
		var localCircuit, term fr.Element
		localCircuit.Mul(&chiR, &rowMLE).Mul(&localCircuit, &pFactors[j])
		term.Mul(&xiPower, &localCircuit)
		target.Add(&target, &term)
		xiPower.Mul(&xiPower, &xi)
	}
	var linkTarget fr.Element
	linkTarget.Mul(&chiR, &dLink).Mul(&linkTarget, &nu)
	target.Add(&target, &linkTarget)
	xiPower = element(1)
	for exponent := 0; exponent < 3; exponent++ {
		xiPower.Mul(&xiPower, &xi)
	}
	for c := range treePoints {
		chiTree := EqualityWeights(treePoints[c])[slot]
		var treeValue, term, oneMinusQ fr.Element
		oneMinusQ.Sub(onePtr(), &qTree[c])
		treeValue.Mul(&oneMinusQ, &t0Value)
		term.Mul(&qTree[c], &t1Value)
		treeValue.Add(&treeValue, &term).Mul(&treeValue, &chiTree).Mul(&treeValue, &xiPower)
		target.Add(&target, &treeValue)
		xiPower.Mul(&xiPower, &xi)
	}
	return input, target
}

func monomial(degree int, coefficient fr.Element) []fr.Element {
	result := make([]fr.Element, degree+1)
	result[degree] = coefficient
	return result
}

func scalePolynomial(polynomial []fr.Element, scale fr.Element) []fr.Element {
	result := make([]fr.Element, len(polynomial))
	for i := range polynomial {
		result[i].Mul(&polynomial[i], &scale)
	}
	return result
}

func addPolynomials(a, b []fr.Element) []fr.Element {
	length := len(a)
	if len(b) > length {
		length = len(b)
	}
	result := make([]fr.Element, length)
	copy(result, a)
	for i := range b {
		result[i].Add(&result[i], &b[i])
	}
	return normalizedCopy(result)
}
