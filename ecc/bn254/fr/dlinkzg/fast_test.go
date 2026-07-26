package dlinkzg

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestFastTaylorShiftMatchesReference(t *testing.T) {
	random := rand.New(rand.NewSource(0x5441594c4f52))
	for size := 1; size <= 1024; size *= 2 {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			p := deterministicElements(random, size)
			before := append([]fr.Element(nil), p...)
			shift := deterministicElements(random, 1)[0]
			want := TaylorShift(p, shift)
			got := FastTaylorShift(p, shift)
			assertElementsEqual(t, got, want)
			assertElementsEqual(t, p, before)
		})
	}
}

func TestFastTaylorShiftEdgeCases(t *testing.T) {
	var zero fr.Element
	seven := fr.NewElement(7)

	tests := []struct {
		name  string
		p     []fr.Element
		shift fr.Element
	}{
		{name: "empty", p: nil, shift: seven},
		{name: "constant", p: []fr.Element{fr.NewElement(11)}, shift: seven},
		{name: "zero_shift", p: []fr.Element{fr.NewElement(3), fr.NewElement(4), fr.NewElement(5)}, shift: zero},
		{name: "declared_high_zeros", p: []fr.Element{fr.NewElement(3), fr.NewElement(4), zero, zero}, shift: seven},
		{name: "zero_polynomial", p: make([]fr.Element, 8), shift: seven},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := append([]fr.Element(nil), test.p...)
			got := FastTaylorShift(test.p, test.shift)
			want := TaylorShift(test.p, test.shift)
			assertElementsEqual(t, got, want)
			assertElementsEqual(t, test.p, before)
		})
	}
}

func TestFastOffDiagMatchesReference(t *testing.T) {
	random := rand.New(rand.NewSource(0x4f464644494147))
	for size := 1; size <= 2048; size *= 2 {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			f := deterministicElements(random, size)
			d := deterministicElements(random, size)
			want := OffDiag(f, d)
			got := FastOffDiag(f, d)
			assertElementsEqual(t, got, want)
		})
	}
}

func TestFastOffDiagEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		f    []fr.Element
		d    []fr.Element
	}{
		{name: "both_empty"},
		{name: "left_constant", f: []fr.Element{fr.NewElement(1)}},
		{name: "right_constant", d: []fr.Element{fr.NewElement(2)}},
		{name: "left_empty", d: []fr.Element{fr.NewElement(1), fr.NewElement(2), fr.NewElement(3)}},
		{name: "right_empty", f: []fr.Element{fr.NewElement(1), fr.NewElement(2), fr.NewElement(3)}},
		{name: "left_shorter", f: []fr.Element{fr.NewElement(1)}, d: []fr.Element{fr.NewElement(2), fr.NewElement(3), fr.NewElement(4), fr.NewElement(5)}},
		{name: "right_shorter", f: []fr.Element{fr.NewElement(1), fr.NewElement(2), fr.NewElement(3), fr.NewElement(4)}, d: []fr.Element{fr.NewElement(5)}},
		{name: "declared_zeros", f: []fr.Element{fr.NewElement(1), {}, {}, {}}, d: []fr.Element{{}, fr.NewElement(2)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			beforeF := append([]fr.Element(nil), test.f...)
			beforeD := append([]fr.Element(nil), test.d...)
			got := FastOffDiag(test.f, test.d)
			want := OffDiag(test.f, test.d)
			assertElementsEqual(t, got, want)
			assertElementsEqual(t, test.f, beforeF)
			assertElementsEqual(t, test.d, beforeD)
		})
	}
}

func TestCheckedConvolutionLengthBoundaries(t *testing.T) {
	if got := checkedConvolutionLength(maxFFTCardinality/2, maxFFTCardinality/2+1); got != maxFFTCardinality {
		t.Fatalf("exact capacity: got %d, want %d", got, maxFFTCardinality)
	}
	for _, test := range []struct {
		name  string
		left  int
		right int
	}{
		{name: "empty", left: 0, right: 1},
		{name: "over_capacity", left: maxFFTCardinality/2 + 1, right: maxFFTCardinality/2 + 1},
		{name: "int_overflow", left: int(^uint(0) >> 1), right: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			checkedConvolutionLength(test.left, test.right)
		})
	}
}

func BenchmarkTaylorShiftImplementations(b *testing.B) {
	for _, size := range []int{64, 256, 1024, 4096} {
		p := benchmarkElements(size)
		shift := fr.NewElement(23)
		b.Run(fmt.Sprintf("reference/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = TaylorShift(p, shift)
			}
		})
		b.Run(fmt.Sprintf("fast/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = FastTaylorShift(p, shift)
			}
		})
	}
}

func BenchmarkOffDiagImplementations(b *testing.B) {
	for _, size := range []int{64, 256, 1024, 4096} {
		f := benchmarkElements(size)
		d := benchmarkElements(size)
		b.Run(fmt.Sprintf("reference/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = OffDiag(f, d)
			}
		})
		b.Run(fmt.Sprintf("fast/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = FastOffDiag(f, d)
			}
		})
	}
}

func deterministicElements(random *rand.Rand, size int) []fr.Element {
	result := make([]fr.Element, size)
	for i := range result {
		result[i].SetUint64(random.Uint64())
	}
	return result
}

func benchmarkElements(size int) []fr.Element {
	result := make([]fr.Element, size)
	for i := range result {
		result[i].SetUint64(uint64(17*i + 3))
	}
	return result
}

func assertElementsEqual(t *testing.T, got, want []fr.Element) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !got[i].Equal(&want[i]) {
			t.Fatalf("coefficient %d mismatch: got %s, want %s", i, got[i].String(), want[i].String())
		}
	}
}
