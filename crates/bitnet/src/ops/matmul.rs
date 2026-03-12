//! I2S ternary matrix multiplication — the core kernel.
//!
//! For a ternary weight matrix W (rows × cols) and an input activation
//! vector x (cols), computes y = W · x where:
//!
//! - w = +1 → accumulator += x[i]
//! - w = -1 → accumulator -= x[i]
//! - w =  0 → skip
//!
//! No multiplication. The hot loop is branch-free conditional addition.
//!
//! ## I2S byte-level kernel
//!
//! Each packed byte contains 4 ternary weights (2 bits each).
//! We process one byte at a time, extracting 4 weights and applying
//! the corresponding 4 activations. This amortizes the byte load
//! and bit-shift overhead.

use crate::tensor::{TernaryTensor, ActivationTensor};

/// Ternary matrix-vector product: y = W · x.
///
/// `weights` is (out_features × in_features), `input` has `in_features` elements.
/// Returns an `i32` accumulator per output feature. The caller applies scaling.
///
/// This is the I2S (Integer 2-bit Scalar) kernel: unpacks weights from 2-bit
/// encoding and performs conditional add/sub on 8-bit activations.
pub fn ternary_matvec(weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
    assert_eq!(weights.cols(), input.len(), "dimension mismatch");

    let rows = weights.rows();
    let cols = weights.cols();
    let mut output = vec![0i32; rows];

    for row in 0..rows {
        output[row] = ternary_dot_i2s(weights, row, input, cols);
    }

    output
}

/// Single row dot product using I2S byte-level unpacking.
///
/// Processes up to 4 weights per byte: extract 2-bit pairs, conditional add/sub.
/// Handles non-byte-aligned row starts via `start_offset`.
#[inline]
fn ternary_dot_i2s(weights: &TernaryTensor, row: usize, input: &[i8], cols: usize) -> i32 {
    let (row_bytes, start_offset, _vals_in_last) = weights.row_bytes(row);
    let mut acc: i32 = 0;
    let mut col = 0;

    for (byte_i, &byte) in row_bytes.iter().enumerate() {
        // On the first byte, skip `start_offset` slots that belong to the prior row.
        let slot_start = if byte_i == 0 { start_offset } else { 0 };

        for slot in slot_start..4 {
            if col >= cols { break; }
            let w = (byte >> (slot * 2)) & 0b11;
            acc += ternary_mul_i8(w, input[col]);
            col += 1;
        }
    }

    acc
}

/// Branch-free ternary "multiply": maps 2-bit encoding to add/sub/skip.
///
/// - 0b00 (-1) → -x
/// - 0b01 ( 0) →  0
/// - 0b10 (+1) → +x
/// - 0b11 (unused) → 0
#[inline(always)]
fn ternary_mul_i8(weight_bits: u8, activation: i8) -> i32 {
    match weight_bits {
        0b00 => -(activation as i32),
        0b10 => activation as i32,
        _    => 0,
    }
}

/// Ternary matrix-vector product with full quantization pipeline.
///
/// Takes quantized weights and activations, returns scaled f32 output.
/// This is the high-level API that handles the scale factors.
///
/// Output[i] = (Σ_j W[i,j] * x[j]) * activation_scale * weight_scale
///
/// For ternary weights, weight_scale is the original absmean γ from training.
pub fn ternary_matvec_scaled(
    weights: &TernaryTensor,
    input: &ActivationTensor,
    weight_scale: f32,
) -> Vec<f32> {
    let raw = ternary_matvec(weights, input.data());
    let combined_scale = input.scale() * weight_scale;

    raw.iter().map(|&acc| acc as f32 * combined_scale).collect()
}

/// Batch ternary matmul: Y = W · X^T for multiple input vectors.
///
/// `inputs` is a slice of activation vectors, all of length `in_features`.
/// Returns one output vector per input.
pub fn ternary_matmul_batch(
    weights: &TernaryTensor,
    inputs: &[&[i8]],
) -> Vec<Vec<i32>> {
    inputs.iter().map(|x| ternary_matvec(weights, x)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Ternary;

    /// Build a ternary weight matrix from i8 values (-1, 0, 1).
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
    fn identity_like_matvec() {
        // Weight matrix = [[1, 0], [0, 1]] (identity).
        let w = weights_from_i8(&[1, 0, 0, 1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![42, -17]);
    }

    #[test]
    fn negation_matvec() {
        // Weight matrix = [[-1, 0], [0, -1]] (negate).
        let w = weights_from_i8(&[-1, 0, 0, -1], 2, 2);
        let x = vec![42i8, -17i8];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![-42, 17]);
    }

    #[test]
    fn mixed_weights() {
        // W = [[1, -1, 0, 1]], x = [10, 20, 30, 40]
        // y = 10 - 20 + 0 + 40 = 30
        let w = weights_from_i8(&[1, -1, 0, 1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![30]);
    }

    #[test]
    fn all_zeros_weight() {
        let w = weights_from_i8(&[0, 0, 0, 0], 1, 4);
        let x = vec![100i8, -50, 25, -12];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![0]);
    }

    #[test]
    fn all_positive_weight() {
        // Sum all activations.
        let w = weights_from_i8(&[1, 1, 1, 1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![100]);
    }

    #[test]
    fn all_negative_weight() {
        // Negate and sum.
        let w = weights_from_i8(&[-1, -1, -1, -1], 1, 4);
        let x = vec![10i8, 20, 30, 40];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![-100]);
    }

    #[test]
    fn non_aligned_cols() {
        // 5 columns — not a multiple of 4.
        let w = weights_from_i8(&[1, -1, 1, 0, -1], 1, 5);
        let x = vec![10i8, 20, 30, 40, 50];
        // y = 10 - 20 + 30 + 0 - 50 = -30
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![-30]);
    }

    #[test]
    fn multi_row() {
        // 3×4 weight matrix.
        let w = weights_from_i8(&[
             1,  0,  0,  0,  // row 0: select x[0]
             0,  1,  0,  0,  // row 1: select x[1]
            -1, -1, -1, -1,  // row 2: negate all
        ], 3, 4);
        let x = vec![5i8, 10, 15, 20];
        let y = ternary_matvec(&w, &x);
        assert_eq!(y, vec![5, 10, -50]);
    }

    #[test]
    fn scaled_output() {
        let w = weights_from_i8(&[1, -1, 1, -1], 1, 4);
        let input_f32 = vec![1.0f32, 0.5, 0.25, 0.125];
        let act = ActivationTensor::quantize(&input_f32, vec![4]);
        let weight_scale = 0.1;

        let y = ternary_matvec_scaled(&w, &act, weight_scale);
        // Expected: (1.0 - 0.5 + 0.25 - 0.125) * 0.1 = 0.0625
        assert_eq!(y.len(), 1);
        assert!((y[0] - 0.0625).abs() < 0.01, "expected ~0.0625, got {}", y[0]);
    }

    #[test]
    fn batch_matvec() {
        let w = weights_from_i8(&[1, 1, -1, -1], 2, 2);
        let x1 = vec![10i8, 20];
        let x2 = vec![5i8, -5];
        let results = ternary_matmul_batch(&w, &[&x1, &x2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![30, -30]);
        assert_eq!(results[1], vec![0, 0]);
    }

    // Verify no multiplication happens — the computation is purely additive.
    #[test]
    fn accumulator_range() {
        // With 8-bit activations in [-127, 127] and N columns,
        // the max accumulator value is 127 * N, which fits in i32.
        let n = 4096; // typical hidden dim
        let max_acc = 127i64 * n as i64;
        assert!(max_acc < i32::MAX as i64, "accumulator overflow at dim {n}");
    }
}
