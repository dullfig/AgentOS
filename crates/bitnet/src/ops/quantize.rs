//! Activation quantization — absmax scaling to 8-bit.
//!
//! BitNet b1.58 quantizes activations per-token to 8-bit using absmax:
//!
//!   scale = max(|x|) / 127
//!   x_quant = round(clamp(x / scale, -127, 127))
//!
//! This module provides in-place and allocating variants, plus the
//! dequantization path for layer outputs.

/// Compute the absmax scale factor for a float slice.
///
/// Returns `max(|x|) / 127.0`. If all values are zero, returns 1.0.
#[inline]
pub fn absmax_scale(values: &[f32]) -> f32 {
    let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax < f32::EPSILON { 1.0 } else { absmax / 127.0 }
}

/// Quantize a float slice to i8 using the given scale.
///
/// `quantized[i] = round(clamp(values[i] / scale, -127, 127))`
pub fn quantize_with_scale(values: &[f32], scale: f32, out: &mut [i8]) {
    assert_eq!(values.len(), out.len());
    let inv_scale = 1.0 / scale;
    for (v, q) in values.iter().zip(out.iter_mut()) {
        *q = (*v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Quantize a float slice, computing scale automatically.
///
/// Returns `(quantized_data, scale)`.
pub fn quantize_absmax(values: &[f32]) -> (Vec<i8>, f32) {
    let scale = absmax_scale(values);
    let mut out = vec![0i8; values.len()];
    quantize_with_scale(values, scale, &mut out);
    (out, scale)
}

/// Dequantize i8 values back to f32 using the given scale.
///
/// `output[i] = quantized[i] * scale`
pub fn dequantize(quantized: &[i8], scale: f32, out: &mut [f32]) {
    assert_eq!(quantized.len(), out.len());
    for (q, v) in quantized.iter().zip(out.iter_mut()) {
        *v = *q as f32 * scale;
    }
}

/// Dequantize to a new Vec<f32>.
pub fn dequantize_alloc(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&q| q as f32 * scale).collect()
}

/// Per-token quantization for a batch of vectors.
///
/// Each row of `values` (of length `cols`) is quantized independently.
/// Returns `(quantized_data, scales)` where scales has one entry per row.
pub fn quantize_per_token(values: &[f32], cols: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(values.len() % cols, 0, "values length must be a multiple of cols");
    let rows = values.len() / cols;
    let mut quantized = vec![0i8; values.len()];
    let mut scales = vec![0.0f32; rows];

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_slice = &values[start..end];
        let scale = absmax_scale(row_slice);
        scales[row] = scale;
        quantize_with_scale(row_slice, scale, &mut quantized[start..end]);
    }

    (quantized, scales)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmax_basic() {
        let vals = vec![1.0f32, -3.0, 2.5, 0.0];
        let scale = absmax_scale(&vals);
        assert!((scale - 3.0 / 127.0).abs() < f32::EPSILON);
    }

    #[test]
    fn absmax_all_zero() {
        let vals = vec![0.0f32; 8];
        let scale = absmax_scale(&vals);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn quantize_roundtrip() {
        let vals = vec![1.0f32, -0.5, 0.0, 0.75, -1.0];
        let (quant, scale) = quantize_absmax(&vals);
        let restored = dequantize_alloc(&quant, scale);

        for (orig, rest) in vals.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.02, "expected ~{orig}, got {rest}");
        }
    }

    #[test]
    fn quantize_boundary_values() {
        // Exactly ±127 should be preserved.
        let vals = vec![127.0f32, -127.0];
        let (quant, scale) = quantize_absmax(&vals);
        assert!((scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(quant, vec![127i8, -127i8]);
    }

    #[test]
    fn quantize_clamping() {
        // With scale=1.0, values >127 should clamp.
        let vals = vec![200.0f32, -200.0];
        let (quant, _) = quantize_absmax(&vals);
        // absmax = 200, scale = 200/127 ≈ 1.575
        // 200/1.575 ≈ 127 → clamped to 127
        assert_eq!(quant[0], 127);
        assert_eq!(quant[1], -127);
    }

    #[test]
    fn per_token_independence() {
        // Two tokens with different scales.
        let vals = vec![
            1.0, 0.5,       // token 0: absmax=1.0
            100.0, -50.0,   // token 1: absmax=100.0
        ];
        let (quant, scales) = quantize_per_token(&vals, 2);
        assert_eq!(quant.len(), 4);
        assert_eq!(scales.len(), 2);

        // Token 0: scale = 1.0/127.0
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-6);
        // Token 1: scale = 100.0/127.0
        assert!((scales[1] - 100.0 / 127.0).abs() < 1e-4);

        // Verify max quantized values are ±127.
        assert_eq!(quant[0], 127);   // 1.0 → 127
        assert_eq!(quant[2], 127);   // 100.0 → 127
    }

    #[test]
    fn dequantize_in_place() {
        let quant = vec![127i8, -64, 0, 32];
        let scale = 0.01;
        let mut out = vec![0.0f32; 4];
        dequantize(&quant, scale, &mut out);
        assert!((out[0] - 1.27).abs() < 1e-5);
        assert!((out[1] - -0.64).abs() < 1e-5);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - 0.32).abs() < 1e-5);
    }
}
