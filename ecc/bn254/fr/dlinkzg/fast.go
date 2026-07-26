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
	return FastTaylorShiftBatch([][]fr.Element{p}, shift)[0]
}

// FastTaylorShiftBatch shifts equal-length polynomials with one shared FFT of
// the shift/factorial kernel. Unequal lengths retain FastTaylorShift semantics
// through the independent fallback path.
func FastTaylorShiftBatch(polynomials [][]fr.Element, shift fr.Element) [][]fr.Element {
	results := make([][]fr.Element, len(polynomials))
	if len(polynomials) == 0 {
		return results
	}
	n := len(polynomials[0])
	for i := range polynomials {
		if len(polynomials[i]) != n {
			for j := range polynomials {
				results[j] = FastTaylorShift(polynomials[j], shift)
			}
			return results
		}
	}
	if n == 0 {
		return results
	}
	if n == 1 {
		for i := range polynomials {
			results[i] = append([]fr.Element(nil), polynomials[i]...)
		}
		return results
	}

	linearLength := checkedConvolutionLength(n, n)
	domain := fft.NewDomain(uint64(linearLength))
	kernel := make([]fr.Element, domain.Cardinality)
	kernel[0].SetOne()
	for i := 1; i < n; i++ {
		kernel[i].Mul(&kernel[i-1], &shift)
	}
	var factorial fr.Element
	factorial.SetOne()
	for i := 1; i < n; i++ {
		var current fr.Element
		current.SetUint64(uint64(i))
		factorial.Mul(&factorial, &current)
	}
	var inverseFactorial fr.Element
	inverseFactorial.Inverse(&factorial)
	for i := n - 1; i >= 0; i-- {
		kernel[i].Mul(&kernel[i], &inverseFactorial)
		if i > 0 {
			var current fr.Element
			current.SetUint64(uint64(i))
			inverseFactorial.Mul(&inverseFactorial, &current)
		}
	}
	domain.FFT(kernel, fft.DIF)

	for polynomial := range polynomials {
		convolution := make([]fr.Element, domain.Cardinality)
		factorial.SetOne()
		for i := 0; i < n; i++ {
			convolution[n-1-i].Mul(&polynomials[polynomial][i], &factorial)
			if i+1 < n {
				var next fr.Element
				next.SetUint64(uint64(i + 1))
				factorial.Mul(&factorial, &next)
			}
		}
		domain.FFT(convolution, fft.DIF)
		for i := range convolution {
			convolution[i].Mul(&convolution[i], &kernel[i])
		}
		domain.FFTInverse(convolution, fft.DIT)

		result := make([]fr.Element, n)
		inverseFactorial.Inverse(&factorial)
		for k := n - 1; k >= 0; k-- {
			result[k].Mul(&convolution[n-1-k], &inverseFactorial)
			if k > 0 {
				var current fr.Element
				current.SetUint64(uint64(k))
				inverseFactorial.Mul(&inverseFactorial, &current)
			}
		}
		results[polynomial] = result
	}
	return results
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
