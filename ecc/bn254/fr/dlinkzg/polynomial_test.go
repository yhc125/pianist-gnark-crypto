package dlinkzg

import (
	"errors"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestPolynomialPrimitives(t *testing.T) {
	p := elements(3, 2, 5, 7)
	point := element(9)

	quotient, remainder := SyntheticDivision(p, point)
	reconstructed := multiply(quotient, []fr.Element{neg(point), fr.One()})
	reconstructed[0].Add(&reconstructed[0], &remainder)
	requirePolynomialEqual(t, p, reconstructed)
	evaluation := Eval(p, point)
	if !remainder.Equal(&evaluation) {
		t.Fatal("synthetic division returned the wrong remainder")
	}

	shift := element(11)
	shifted := TaylorShift(p, shift)
	query := element(13)
	var translatedQuery fr.Element
	translatedQuery.Add(&query, &shift)
	got, want := Eval(shifted, query), Eval(p, translatedQuery)
	if !got.Equal(&want) {
		t.Fatal("Taylor shift does not preserve p(X+shift)")
	}

	weights := EqualityWeights(elements(2, 3))
	if len(weights) != 4 {
		t.Fatalf("unexpected equality-weight length: %d", len(weights))
	}
	var weightSum fr.Element
	for i := range weights {
		weightSum.Add(&weightSum, &weights[i])
	}
	if !weightSum.Equal(onePtr()) {
		t.Fatal("equality weights do not sum to one")
	}
	coefficients := elements(5, 7, 11, 13)
	mle, err := MLEEval(coefficients, elements(2, 3))
	if err != nil {
		t.Fatal(err)
	}
	var dot, term fr.Element
	for i := range coefficients {
		term.Mul(&coefficients[i], &weights[i])
		dot.Add(&dot, &term)
	}
	if !mle.Equal(&dot) {
		t.Fatal("MLE evaluation and equality-weight inner product disagree")
	}
	if _, err := MLEEval(elements(1, 2, 3), []fr.Element{element(4)}); !errors.Is(err, ErrInvalidMLE) {
		t.Fatalf("expected ErrInvalidMLE, got %v", err)
	}
}

func TestOffDiagLaurentIdentity(t *testing.T) {
	f := elements(2, 3, 5, 7, 11)
	d := elements(13, 17, 19, 23, 29)
	s := OffDiag(f, d)
	beta := element(31)
	var betaInverse fr.Element
	betaInverse.Inverse(&beta)

	var left, term fr.Element
	left.Mul(valuePtr(Eval(f, beta)), valuePtr(Eval(d, betaInverse)))
	term.Mul(valuePtr(Eval(f, betaInverse)), valuePtr(Eval(d, beta)))
	left.Add(&left, &term)

	var diagonal fr.Element
	for i := range f {
		term.Mul(&f[i], &d[i])
		diagonal.Add(&diagonal, &term)
	}
	diagonal.Double(&diagonal)
	var positive, negative fr.Element
	positive.Mul(&beta, valuePtr(Eval(s, beta)))
	negative.Mul(&betaInverse, valuePtr(Eval(s, betaInverse)))
	var right fr.Element
	right.Add(&diagonal, &positive).Add(&right, &negative)
	if !left.Equal(&right) {
		t.Fatal("OffDiag does not satisfy the symmetric Laurent identity")
	}
}

func TestInterpolationAndExactSameSetQuotient(t *testing.T) {
	points := elements(2, 5, 7)
	p0 := elements(3, 1, 4, 1, 5, 9)
	p1 := elements(2, 6, 5, 3, 5, 8)
	inputs := []SameSetInput{
		{Polynomial: p0, ClaimedValues: evaluations(p0, points)},
		{Polynomial: p1, ClaimedValues: evaluations(p1, points)},
	}
	kappa := element(11)
	result, err := BuildSameSetQuotient(inputs, points, kappa)
	if err != nil {
		t.Fatal(err)
	}
	for i := range result.Interpolants {
		for j := range points {
			got := Eval(result.Interpolants[i], points[j])
			if !got.Equal(&inputs[i].ClaimedValues[j]) {
				t.Fatalf("interpolant %d misses point %d", i, j)
			}
		}
	}
	for i := range points {
		value := Eval(result.Vanishing, points[i])
		if !value.IsZero() {
			t.Fatalf("vanishing polynomial misses point %d", i)
		}
	}
	reconstructed := multiply(result.Quotient, result.Vanishing)
	requirePolynomialEqual(t, result.Numerator, reconstructed)

	tampered := append([]fr.Element(nil), inputs[0].ClaimedValues...)
	tampered[1].Add(&tampered[1], onePtr())
	badInputs := append([]SameSetInput(nil), inputs...)
	badInputs[0].ClaimedValues = tampered
	if _, err := BuildSameSetQuotient(badInputs, points, kappa); !errors.Is(err, ErrInexactDivision) {
		t.Fatalf("tampered value should make the quotient inexact, got %v", err)
	}

	duplicatePoints := elements(2, 2)
	if _, err := Interpolate(duplicatePoints, elements(1, 1)); !errors.Is(err, ErrDuplicatePoint) {
		t.Fatalf("expected duplicate-point rejection, got %v", err)
	}
}

func element(value uint64) fr.Element {
	var result fr.Element
	result.SetUint64(value)
	return result
}

func elements(values ...uint64) []fr.Element {
	result := make([]fr.Element, len(values))
	for i := range values {
		result[i].SetUint64(values[i])
	}
	return result
}

func evaluations(p, points []fr.Element) []fr.Element {
	result := make([]fr.Element, len(points))
	for i := range points {
		result[i] = Eval(p, points[i])
	}
	return result
}

func neg(value fr.Element) fr.Element {
	var result fr.Element
	result.Neg(&value)
	return result
}

func onePtr() *fr.Element {
	one := fr.One()
	return &one
}

func valuePtr(value fr.Element) *fr.Element {
	return &value
}

func requirePolynomialEqual(t *testing.T, want, got []fr.Element) {
	t.Helper()
	want = normalizedCopy(want)
	got = normalizedCopy(got)
	if len(want) != len(got) {
		t.Fatalf("polynomial lengths differ: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		if !want[i].Equal(&got[i]) {
			t.Fatalf("coefficient %d differs", i)
		}
	}
}
