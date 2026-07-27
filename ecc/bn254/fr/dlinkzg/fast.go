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
	if len(polynomials) == 0 {
		return make([][]fr.Element, 0)
	}
	precomputation := NewFastTaylorShiftPrecomputation(len(polynomials[0]), shift, nil)
	return FastTaylorShiftBatchWithPrecomputation(polynomials, precomputation)
}

// FastTaylorShiftBatchWithDomain is FastTaylorShiftBatch with a reusable
// minimal convolution domain supplied by preprocessing.
func FastTaylorShiftBatchWithDomain(polynomials [][]fr.Element, shift fr.Element, domain *fft.Domain) [][]fr.Element {
	if len(polynomials) == 0 {
		return make([][]fr.Element, 0)
	}
	precomputation := NewFastTaylorShiftPrecomputation(len(polynomials[0]), shift, domain)
	return FastTaylorShiftBatchWithPrecomputation(polynomials, precomputation)
}

// FastTaylorShiftPrecomputation contains the witness-independent factorials
// and transformed shift kernel for one polynomial length and public shift.
type FastTaylorShiftPrecomputation struct {
	length            int
	shift             fr.Element
	domain            *fft.Domain
	factorials        []fr.Element
	inverseFactorials []fr.Element
	kernelSpectrum    []fr.Element
}

// NewFastTaylorShiftPrecomputation moves the shared Taylor kernel FFT and
// factorial tables out of the online proof.
func NewFastTaylorShiftPrecomputation(length int, shift fr.Element, reusableDomain *fft.Domain) *FastTaylorShiftPrecomputation {
	precomputation := &FastTaylorShiftPrecomputation{length: length, shift: shift}
	if length <= 1 {
		return precomputation
	}
	linearLength := checkedConvolutionLength(length, length)
	precomputation.domain = reusableFastConvolutionDomain(linearLength, reusableDomain)
	precomputation.factorials = make([]fr.Element, length)
	precomputation.inverseFactorials = make([]fr.Element, length)
	precomputation.factorials[0].SetOne()
	for index := 1; index < length; index++ {
		var current fr.Element
		current.SetUint64(uint64(index))
		precomputation.factorials[index].Mul(&precomputation.factorials[index-1], &current)
	}
	precomputation.inverseFactorials[length-1].Inverse(&precomputation.factorials[length-1])
	for index := length - 1; index > 0; index-- {
		var current fr.Element
		current.SetUint64(uint64(index))
		precomputation.inverseFactorials[index-1].Mul(&precomputation.inverseFactorials[index], &current)
	}

	precomputation.kernelSpectrum = make([]fr.Element, precomputation.domain.Cardinality)
	power := fr.One()
	for index := 0; index < length; index++ {
		precomputation.kernelSpectrum[index].Mul(&power, &precomputation.inverseFactorials[index])
		power.Mul(&power, &shift)
	}
	precomputation.domain.FFT(precomputation.kernelSpectrum, fft.DIF)
	return precomputation
}

// PolynomialLength reports the coefficient width fixed by preprocessing.
func (precomputation *FastTaylorShiftPrecomputation) PolynomialLength() int {
	if precomputation == nil {
		return 0
	}
	return precomputation.length
}

// FastTaylorShiftBatchWithPrecomputation shifts an equal-length batch without
// rebuilding the domain, factorial tables, or shift-kernel spectrum.
func FastTaylorShiftBatchWithPrecomputation(
	polynomials [][]fr.Element,
	precomputation *FastTaylorShiftPrecomputation,
) [][]fr.Element {
	results := make([][]fr.Element, len(polynomials))
	if len(polynomials) == 0 {
		return results
	}
	if precomputation == nil {
		panic("dlinkzg: nil Taylor-shift precomputation")
	}
	n := len(polynomials[0])
	for i := range polynomials {
		if len(polynomials[i]) != n || n != precomputation.length {
			for j := range polynomials {
				results[j] = FastTaylorShift(polynomials[j], precomputation.shift)
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

	convolution := make([]fr.Element, precomputation.domain.Cardinality)
	for polynomial := range polynomials {
		for index := range convolution {
			convolution[index].SetZero()
		}
		for i := 0; i < n; i++ {
			convolution[n-1-i].Mul(&polynomials[polynomial][i], &precomputation.factorials[i])
		}
		precomputation.domain.FFT(convolution, fft.DIF)
		for i := range convolution {
			convolution[i].Mul(&convolution[i], &precomputation.kernelSpectrum[i])
		}
		precomputation.domain.FFTInverse(convolution, fft.DIT)

		result := make([]fr.Element, n)
		for k := 0; k < n; k++ {
			result[k].Mul(&convolution[n-1-k], &precomputation.inverseFactorials[k])
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

// FastOffDiagGeometric computes FastOffDiag(f, d) in O(T) field operations
// for the structured vector d[k] = ratio^k with len(d) = len(f). The
// recurrence is valid for every ratio, including zero, and does not mutate f.
//
// For positive offset delta, write
//
//	c_delta = sum_k f[k+delta] ratio^k
//	        + ratio^delta sum_k f[k] ratio^k.
//
// All suffix values in the first sum and all bounded prefixes in the second
// sum are obtained by one linear pass each.
func FastOffDiagGeometric(f []fr.Element, ratio fr.Element) []fr.Element {
	t := len(f)
	if t < 2 {
		return nil
	}

	result := make([]fr.Element, t-1)
	powers := make([]fr.Element, t)
	fillFastOffDiagGeometric(result, f, ratio, powers)
	return result
}

// fillFastOffDiagGeometric overwrites destination with the structured result
// while reusing a caller-owned power table. It is the allocation-free inner
// loop used when several circuit terms share Laurent-witness workspaces.
func fillFastOffDiagGeometric(destination, f []fr.Element, ratio fr.Element, powers []fr.Element) {
	t := len(f)
	if t < 2 {
		return
	}
	if len(destination) < t-1 || len(powers) < t {
		panic("dlinkzg: geometric off-diagonal workspace is too short")
	}
	powers = powers[:t]
	powers[0].SetOne()
	for index := 1; index < t; index++ {
		powers[index].Mul(&powers[index-1], &ratio)
	}

	tail := f[t-1]
	for delta := t - 1; delta >= 1; delta-- {
		if delta < t-1 {
			tail.Mul(&tail, &ratio).Add(&tail, &f[delta])
		}
		destination[delta-1] = tail
	}

	var prefix fr.Element
	for index := 0; index < t-1; index++ {
		var term fr.Element
		term.Mul(&f[index], &powers[index])
		prefix.Add(&prefix, &term)

		delta := t - 1 - index
		term.Mul(&prefix, &powers[delta])
		destination[delta-1].Add(&destination[delta-1], &term)
	}
}

// FastOffDiagBatch returns sum_i scales[i] * FastOffDiag(left[i], right[i]).
// It uses linearity in the Fourier domain to share the inverse FFT across the
// complete batch. Inputs may have different declared lengths; every pair is
// zero-padded to the largest batch length, which preserves the coefficient for
// each positive offset.
func FastOffDiagBatch(left, right [][]fr.Element, scales []fr.Element) []fr.Element {
	return fastOffDiagBatch(left, right, scales, nil)
}

// FastOffDiagBatchWithDomain is FastOffDiagBatch with a reusable minimal
// convolution domain supplied by preprocessing.
func FastOffDiagBatchWithDomain(left, right [][]fr.Element, scales []fr.Element, domain *fft.Domain) []fr.Element {
	return fastOffDiagBatch(left, right, scales, domain)
}

func fastOffDiagBatch(left, right [][]fr.Element, scales []fr.Element, reusableDomain *fft.Domain) []fr.Element {
	if len(left) != len(right) || len(left) != len(scales) {
		panic("dlinkzg: mismatched off-diagonal batch")
	}
	t := 0
	for i := range left {
		if len(left[i]) > t {
			t = len(left[i])
		}
		if len(right[i]) > t {
			t = len(right[i])
		}
	}
	if t < 2 {
		return nil
	}

	linearLength := checkedConvolutionLength(t, t)
	domain := reusableFastConvolutionDomain(linearLength, reusableDomain)
	accumulator := make([]fr.Element, domain.Cardinality)
	leftSpectrum := make([]fr.Element, domain.Cardinality)
	rightSpectrum := make([]fr.Element, domain.Cardinality)
	for pair := range left {
		if scales[pair].IsZero() || len(left[pair]) == 0 || len(right[pair]) == 0 {
			continue
		}
		for index := range leftSpectrum {
			leftSpectrum[index].SetZero()
			rightSpectrum[index].SetZero()
		}
		copy(leftSpectrum, left[pair])
		for i := range right[pair] {
			rightSpectrum[t-1-i] = right[pair][i]
		}
		domain.FFT(leftSpectrum, fft.DIF)
		domain.FFT(rightSpectrum, fft.DIF)
		for i := range accumulator {
			var term fr.Element
			term.Mul(&leftSpectrum[i], &rightSpectrum[i]).Mul(&term, &scales[pair])
			accumulator[i].Add(&accumulator[i], &term)
		}
	}
	domain.FFTInverse(accumulator, fft.DIT)

	result := make([]fr.Element, t-1)
	for delta := 1; delta < t; delta++ {
		result[delta-1].Add(
			&accumulator[t-1+delta],
			&accumulator[t-1-delta],
		)
	}
	return result
}

func reusableFastConvolutionDomain(linearLength int, domain *fft.Domain) *fft.Domain {
	if domain == nil {
		return fft.NewDomain(uint64(linearLength))
	}
	cardinality := int(domain.Cardinality)
	if cardinality < linearLength {
		panic("dlinkzg: reusable FFT domain has the wrong cardinality")
	}
	return domain
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
