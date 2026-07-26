package dlinkzg

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

// maxFFTCardinality is the largest power-of-two FFT supported by the BN254
// scalar field in this version of gnark-crypto.
const maxFFTCardinality = 1 << 28

// FastTaylorShift returns the coefficients of p(X+shift) in O(n log n)
// field operations. The output follows TaylorShift's convention: it has the
// same length as p, including declared high zero coefficients.
//
// The reduction uses
//
//	[X^k]p(X+shift) = 1/k! sum_{j>=0} p[k+j](k+j)! shift^j/j!,
//
// so all coefficients are obtained from one convolution after reversing the
// factorial-scaled coefficients of p. It panics if the padded convolution
// exceeds the largest FFT supported by the BN254 scalar field.
func FastTaylorShift(p []fr.Element, shift fr.Element) []fr.Element {
	n := len(p)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return append([]fr.Element(nil), p...)
	}
	checkedConvolutionLength(n, n)

	reversedScaled := make([]fr.Element, n)
	shiftOverFactorial := make([]fr.Element, n)

	var factorial fr.Element
	factorial.SetOne()
	for i := 0; i < n; i++ {
		reversedScaled[n-1-i].Mul(&p[i], &factorial)
		if i+1 < n {
			var next fr.Element
			next.SetUint64(uint64(i + 1))
			factorial.Mul(&factorial, &next)
		}
	}

	shiftOverFactorial[0].SetOne()
	for i := 1; i < n; i++ {
		shiftOverFactorial[i].Mul(&shiftOverFactorial[i-1], &shift)
	}
	var inverseFactorial fr.Element
	inverseFactorial.Inverse(&factorial)
	for i := n - 1; i >= 0; i-- {
		shiftOverFactorial[i].Mul(&shiftOverFactorial[i], &inverseFactorial)
		if i > 0 {
			var current fr.Element
			current.SetUint64(uint64(i))
			inverseFactorial.Mul(&inverseFactorial, &current)
		}
	}

	convolution := fftConvolution(reversedScaled, shiftOverFactorial)
	result := make([]fr.Element, n)
	for k := 0; k < n; k++ {
		result[k] = convolution[n-1-k]
	}
	// Walk down from 1/(n-1)! to avoid one inversion per coefficient.
	inverseFactorial.Inverse(&factorial)
	for k := n - 1; k >= 0; k-- {
		result[k].Mul(&result[k], &inverseFactorial)
		if k > 0 {
			var current fr.Element
			current.SetUint64(uint64(k))
			inverseFactorial.Mul(&inverseFactorial, &current)
		}
	}
	return result
}

// FastOffDiag computes the same positive off-diagonal witness as OffDiag in
// O(T log T) field operations, where T=max(len(f),len(d)). Short inputs are
// zero-padded exactly as in OffDiag.
//
// If c=f*reverse(d), the two directional correlations for offset delta are
// c[T-1+delta] and c[T-1-delta]. Thus one convolution supplies every output.
// It panics if the padded convolution exceeds the field's FFT capacity.
func FastOffDiag(f, d []fr.Element) []fr.Element {
	t := len(f)
	if len(d) > t {
		t = len(d)
	}
	if t < 2 {
		return nil
	}
	checkedConvolutionLength(t, t)

	left := make([]fr.Element, t)
	rightReversed := make([]fr.Element, t)
	copy(left, f)
	for i := range d {
		rightReversed[t-1-i] = d[i]
	}

	convolution := fftConvolution(left, rightReversed)
	result := make([]fr.Element, t-1)
	for delta := 1; delta < t; delta++ {
		result[delta-1].Add(
			&convolution[t-1+delta],
			&convolution[t-1-delta],
		)
	}
	return result
}

// fftConvolution returns the full linear convolution of two non-empty
// coefficient vectors. A DIF forward transform and DIT inverse transform let
// the pointwise product remain in bit-reversed order, avoiding two explicit
// permutations.
func fftConvolution(a, b []fr.Element) []fr.Element {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	linearLength := checkedConvolutionLength(len(a), len(b))

	domain := fft.NewDomain(uint64(linearLength))
	left := make([]fr.Element, domain.Cardinality)
	right := make([]fr.Element, domain.Cardinality)
	copy(left, a)
	copy(right, b)

	domain.FFT(left, fft.DIF)
	domain.FFT(right, fft.DIF)
	for i := range left {
		left[i].Mul(&left[i], &right[i])
	}
	domain.FFTInverse(left, fft.DIT)
	return left[:linearLength]
}

func checkedConvolutionLength(leftLength, rightLength int) int {
	if leftLength <= 0 || rightLength <= 0 {
		panic("dlinkzg: convolution inputs must be non-empty")
	}
	if leftLength > int(^uint(0)>>1)-rightLength+1 {
		panic("dlinkzg: convolution length overflows int")
	}
	linearLength := leftLength + rightLength - 1
	if linearLength > maxFFTCardinality {
		panic(fmt.Sprintf("dlinkzg: convolution length %d exceeds BN254 FFT capacity %d", linearLength, maxFFTCardinality))
	}
	return linearLength
}
