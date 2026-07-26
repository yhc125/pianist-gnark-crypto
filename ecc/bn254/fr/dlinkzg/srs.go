package dlinkzg

import (
	"errors"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

var (
	ErrInvalidSRS        = errors.New("dlinkzg: invalid monomial rectangular SRS")
	ErrPolynomialTooWide = errors.New("dlinkzg: polynomial exceeds the SRS rectangle")
)

// MonomialSRS is a transparent test-only rectangular SRS.  G1Rect[i][j]
// equals [tauY^i tauZ^j]_1, while G2Y and G2Z contain the corresponding
// one-variable powers in G2.  Production code must replace NewMonomialSRS
// with an MPC-generated SRS and must not retain either trapdoor.
type MonomialSRS struct {
	G1Rect [][]bn254.G1Affine
	G2Y    []bn254.G2Affine
	G2Z    []bn254.G2Affine
}

// NewMonomialSRS constructs a deterministic rectangular test SRS with the
// requested number of Y and Z powers.
func NewMonomialSRS(yPowers, zPowers int, tauY, tauZ fr.Element) (*MonomialSRS, error) {
	if yPowers < 2 || zPowers < 2 {
		return nil, ErrInvalidSRS
	}
	_, _, generator1, generator2 := bn254.Generators()

	yScalars := powers(tauY, yPowers)
	zScalars := powers(tauZ, zPowers)
	srs := &MonomialSRS{
		G1Rect: make([][]bn254.G1Affine, yPowers),
		G2Y:    make([]bn254.G2Affine, yPowers),
		G2Z:    make([]bn254.G2Affine, zPowers),
	}
	for i := range srs.G1Rect {
		srs.G1Rect[i] = make([]bn254.G1Affine, zPowers)
		for j := range srs.G1Rect[i] {
			var scalar fr.Element
			scalar.Mul(&yScalars[i], &zScalars[j])
			srs.G1Rect[i][j] = scaleG1(generator1, scalar)
		}
		srs.G2Y[i] = scaleG2(generator2, yScalars[i])
	}
	for j := range srs.G2Z {
		srs.G2Z[j] = scaleG2(generator2, zScalars[j])
	}
	return srs, nil
}

// CommitRect commits to sum_{i,j} coefficients[i][j] Y^i Z^j.
func CommitRect(coefficients [][]fr.Element, srs *MonomialSRS) (bn254.G1Affine, error) {
	var result bn254.G1Affine
	if err := validateSRS(srs); err != nil {
		return result, err
	}
	if len(coefficients) > len(srs.G1Rect) {
		return result, ErrPolynomialTooWide
	}
	var bases []bn254.G1Affine
	var scalars []fr.Element
	for i := range coefficients {
		if len(coefficients[i]) > len(srs.G1Rect[i]) {
			return result, ErrPolynomialTooWide
		}
		bases = append(bases, srs.G1Rect[i][:len(coefficients[i])]...)
		scalars = append(scalars, coefficients[i]...)
	}
	if len(bases) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(bases, scalars, ecc.MultiExpConfig{ScalarsMont: true})
	return result, err
}

// CommitZ commits to a polynomial embedded in the Y^0 row.
func CommitZ(coefficients []fr.Element, srs *MonomialSRS) (bn254.G1Affine, error) {
	return CommitRect([][]fr.Element{coefficients}, srs)
}

// CommitY commits to a polynomial embedded in the Z^0 column.
func CommitY(coefficients []fr.Element, srs *MonomialSRS) (bn254.G1Affine, error) {
	rows := make([][]fr.Element, len(coefficients))
	for i := range coefficients {
		rows[i] = []fr.Element{coefficients[i]}
	}
	return CommitRect(rows, srs)
}

func commitZInG2(coefficients []fr.Element, srs *MonomialSRS) (bn254.G2Affine, error) {
	var result bn254.G2Affine
	if err := validateSRS(srs); err != nil {
		return result, err
	}
	if len(coefficients) > len(srs.G2Z) {
		return result, ErrPolynomialTooWide
	}
	if len(coefficients) == 0 {
		return result, nil
	}
	_, err := result.MultiExp(srs.G2Z[:len(coefficients)], coefficients, ecc.MultiExpConfig{ScalarsMont: true})
	return result, err
}

func validateSRS(srs *MonomialSRS) error {
	if srs == nil || len(srs.G1Rect) < 2 || len(srs.G1Rect[0]) < 2 || len(srs.G2Y) < 2 || len(srs.G2Z) < 2 {
		return ErrInvalidSRS
	}
	return nil
}

func powers(base fr.Element, count int) []fr.Element {
	result := make([]fr.Element, count)
	result[0].SetOne()
	for i := 1; i < count; i++ {
		result[i].Mul(&result[i-1], &base)
	}
	return result
}

func scaleG1(point bn254.G1Affine, scalar fr.Element) bn254.G1Affine {
	var scalarBig big.Int
	scalar.ToBigIntRegular(&scalarBig)
	var result bn254.G1Affine
	result.ScalarMultiplication(&point, &scalarBig)
	return result
}

func scaleG2(point bn254.G2Affine, scalar fr.Element) bn254.G2Affine {
	var scalarBig big.Int
	scalar.ToBigIntRegular(&scalarBig)
	var result bn254.G2Affine
	result.ScalarMultiplication(&point, &scalarBig)
	return result
}
