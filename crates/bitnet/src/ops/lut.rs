//! TL1 lookup-table kernel — zero arithmetic in the hot loop.
//!
//! Instead of unpacking weights and doing conditional add/sub, we precompute
//! every possible dot product for groups of 2 ternary weights with their
//! corresponding activations.
//!
//! ## How it works
//!
//! Two ternary weights w0, w1 ∈ {-1, 0, +1} produce 3² = 9 possible
//! combinations. For each pair of activations (a0, a1), we precompute
//! all 9 outcomes:
//!
//! ```text
//! index  w0  w1  result
//! 0      -1  -1  -a0 - a1
//! 1      -1   0  -a0
//! 2      -1  +1  -a0 + a1
//! 3       0  -1       - a1
//! 4       0   0         0
//! 5       0  +1       + a1
//! 6      +1  -1  +a0 - a1
//! 7      +1   0  +a0
//! 8      +1  +1  +a0 + a1
//! ```
//!
//! The packed weight pair (4 bits from I2S encoding) maps to one of these
//! 9 entries. The inner loop becomes: `acc += lut[index]`.
//!
//! ## Encoding
//!
//! Two 2-bit weights pack into 4 bits. Our encoding is:
//! - w0 in bits 0-1, w1 in bits 2-3.
//! - We map the 4-bit value to a LUT index via a 16-entry remap table
//!   (only 9 of 16 slots are valid; unused slots map to 0).

use crate::tensor::TernaryTensor;

/// Precomputed lookup table for one pair of activations.
///
/// Contains 9 entries indexed by the ternary weight-pair combination.
type Lut9 = [i32; 9];

/// Map a 4-bit packed weight pair to a LUT-9 index.
///
/// The 4-bit value encodes two 2-bit ternary weights: w0 in bits 0-1,
/// w1 in bits 2-3. Our ternary encoding: 0b00=-1, 0b01=0, 0b10=+1.
///
/// This table maps each 4-bit value (0..16) to a LUT-9 index (0..9),
/// or 4 (the zero entry) for invalid/unused bit patterns.
const PAIR_TO_LUT_INDEX: [usize; 16] = [
    // w1=00(-1): w0=00(-1), 01(0), 10(+1), 11(unused)
    0, 3, 6, 4,  // 0b00_00, 0b00_01, 0b00_10, 0b00_11
    // w1=01(0): w0=00(-1), 01(0), 10(+1), 11(unused)
    1, 4, 7, 4,  // 0b01_00, 0b01_01, 0b01_10, 0b01_11
    // w1=10(+1): w0=00(-1), 01(0), 10(+1), 11(unused)
    2, 5, 8, 4,  // 0b10_00, 0b10_01, 0b10_10, 0b10_11
    // w1=11(unused)
    4, 4, 4, 4,  // all unused → zero
];

/// Build the 9-entry LUT for a pair of activations.
#[inline]
fn build_lut9(a0: i32, a1: i32) -> Lut9 {
    [
        -a0 - a1, // 0: (-1, -1)
        -a0,      // 1: (-1,  0)
        -a0 + a1, // 2: (-1, +1)
             -a1, // 3: ( 0, -1)
              0,  // 4: ( 0,  0)
              a1, // 5: ( 0, +1)
         a0 - a1, // 6: (+1, -1)
         a0,      // 7: (+1,  0)
         a0 + a1, // 8: (+1, +1)
    ]
}

/// TL1 lookup-table matrix-vector product: y = W · x.
///
/// Processes weights in pairs, using precomputed 9-entry tables.
/// The inner loop is pure table lookup — no arithmetic on activations.
pub fn lut_matvec(weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
    assert_eq!(weights.cols(), input.len(), "dimension mismatch");

    let rows = weights.rows();
    let cols = weights.cols();
    let mut output = vec![0i32; rows];

    for (row, out) in output.iter_mut().enumerate() {
        *out = lut_dot(weights, row, input, cols);
    }

    output
}

/// Single row dot product using LUT-9 kernel.
///
/// Handles non-byte-aligned row starts by falling back to scalar
/// for partial slots at the beginning, then processing pairs.
#[inline]
fn lut_dot(weights: &TernaryTensor, row: usize, input: &[i8], cols: usize) -> i32 {
    let (row_bytes, start_offset, _) = weights.row_bytes(row);
    let mut acc: i32 = 0;
    let mut col = 0;

    for (byte_i, &byte) in row_bytes.iter().enumerate() {
        let slot_start = if byte_i == 0 { start_offset } else { 0 };

        // Extract the valid slots from this byte for our row.
        let mut slots = [0u8; 4];
        let mut n_slots = 0;
        for slot in slot_start..4 {
            if col + n_slots >= cols { break; }
            slots[n_slots] = (byte >> (slot * 2)) & 0b11;
            n_slots += 1;
        }

        // Process slots in pairs for LUT, scalar fallback for odd remainder.
        let mut s = 0;
        while s + 1 < n_slots {
            let w0_bits = slots[s];
            let w1_bits = slots[s + 1];
            let pair_bits = (w0_bits | (w1_bits << 2)) as usize;
            let a0 = input[col] as i32;
            let a1 = input[col + 1] as i32;
            let lut = build_lut9(a0, a1);
            acc += lut[PAIR_TO_LUT_INDEX[pair_bits]];
            col += 2;
            s += 2;
        }
        if s < n_slots {
            acc += scalar_ternary_mul(slots[s], input[col]);
            col += 1;
        }
    }

    acc
}

/// Scalar fallback for single leftover weights.
#[inline(always)]
fn scalar_ternary_mul(weight_bits: u8, activation: i8) -> i32 {
    match weight_bits {
        0b00 => -(activation as i32),
        0b10 => activation as i32,
        _    => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Ternary;
    use crate::ops::matmul::ternary_matvec;

    fn weights_from_i8(values: &[i8], rows: usize, cols: usize) -> TernaryTensor {
        let ternary: Vec<Ternary> = values.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        TernaryTensor::pack(&ternary, rows, cols)
    }

    #[test]
    fn lut9_values() {
        let lut = build_lut9(10, 20);
        assert_eq!(lut[0], -30);  // (-1,-1): -10-20
        assert_eq!(lut[1], -10);  // (-1, 0): -10
        assert_eq!(lut[2],  10);  // (-1,+1): -10+20
        assert_eq!(lut[3], -20);  // ( 0,-1):    -20
        assert_eq!(lut[4],   0);  // ( 0, 0):      0
        assert_eq!(lut[5],  20);  // ( 0,+1):    +20
        assert_eq!(lut[6], -10);  // (+1,-1): +10-20
        assert_eq!(lut[7],  10);  // (+1, 0): +10
        assert_eq!(lut[8],  30);  // (+1,+1): +10+20
    }

    #[test]
    fn pair_index_mapping() {
        // pair_bits = w0_bits | (w1_bits << 2)
        // LUT index = ternary_idx(w0)*3 + ternary_idx(w1)

        // w0=-1(0b00), w1=-1(0b00) → pair=0b0000=0, idx=0*3+0=0
        assert_eq!(PAIR_TO_LUT_INDEX[0b0000], 0);
        // w0=-1(0b00), w1=0(0b01) → pair=0b0100=4, idx=0*3+1=1
        assert_eq!(PAIR_TO_LUT_INDEX[0b0100], 1);
        // w0=+1(0b10), w1=+1(0b10) → pair=0b1010=10, idx=2*3+2=8
        assert_eq!(PAIR_TO_LUT_INDEX[0b1010], 8);
        // w0=0(0b01), w1=0(0b01) → pair=0b0101=5, idx=1*3+1=4
        assert_eq!(PAIR_TO_LUT_INDEX[0b0101], 4);
        // w0=+1(0b10), w1=-1(0b00) → pair=0b0010=2, idx=2*3+0=6
        assert_eq!(PAIR_TO_LUT_INDEX[0b0010], 6);
    }

    /// LUT kernel must produce identical results to I2S kernel.
    #[test]
    fn lut_matches_i2s_aligned() {
        let w = weights_from_i8(&[
             1, -1,  0,  1,
            -1,  0,  1, -1,
             0,  1, -1,  0,
        ], 3, 4);
        let x = vec![10i8, 20, 30, 40];

        let i2s_result = ternary_matvec(&w, &x);
        let lut_result = lut_matvec(&w, &x);
        assert_eq!(i2s_result, lut_result);
    }

    /// LUT kernel handles non-aligned column counts.
    #[test]
    fn lut_matches_i2s_unaligned() {
        let w = weights_from_i8(&[
            1, -1, 0, 1, -1,
        ], 1, 5);
        let x = vec![10i8, 20, 30, 40, 50];

        let i2s_result = ternary_matvec(&w, &x);
        let lut_result = lut_matvec(&w, &x);
        assert_eq!(i2s_result, lut_result);
    }

    #[test]
    fn lut_matches_i2s_odd_cols() {
        // 3 columns — tests single-weight fallback.
        let w = weights_from_i8(&[1, -1, 1], 1, 3);
        let x = vec![5i8, 10, 15];

        let i2s_result = ternary_matvec(&w, &x);
        let lut_result = lut_matvec(&w, &x);
        assert_eq!(i2s_result, lut_result);
    }

    /// Exhaustive correctness: test all 9 weight-pair combos.
    #[test]
    fn lut_all_nine_combos() {
        let weights_pairs: [(i8, i8); 9] = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ];

        for &(w0, w1) in &weights_pairs {
            let w = weights_from_i8(&[w0, w1], 1, 2);
            let x = vec![42i8, 17];

            let i2s = ternary_matvec(&w, &x);
            let lut = lut_matvec(&w, &x);
            assert_eq!(i2s, lut, "mismatch for weights ({w0}, {w1})");
        }
    }

    /// Multi-row LUT correctness.
    #[test]
    fn lut_multi_row() {
        let w = weights_from_i8(&[
             1,  0,  0,  0,
             0,  1,  0,  0,
            -1, -1, -1, -1,
        ], 3, 4);
        let x = vec![5i8, 10, 15, 20];

        let i2s_result = ternary_matvec(&w, &x);
        let lut_result = lut_matvec(&w, &x);
        assert_eq!(i2s_result, lut_result);
    }

    /// Large dimension — stress test.
    #[test]
    fn lut_large_dimension() {
        let cols = 512;
        let rows = 64;
        let w_vals: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 })
            .collect();
        let w = weights_from_i8(&w_vals, rows, cols);
        let x: Vec<i8> = (0..cols).map(|i| (i % 127) as i8).collect();

        let i2s_result = ternary_matvec(&w, &x);
        let lut_result = lut_matvec(&w, &x);
        assert_eq!(i2s_result, lut_result);
    }
}
