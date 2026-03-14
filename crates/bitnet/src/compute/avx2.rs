//! AVX2 compute backend — 32-wide SIMD ternary matmul.
//!
//! Key insight: `vpabsb` + `vpsignb` collapse the ternary branch
//! (add/sub/skip) into a single SIMD instruction per 32 weights.
//!
//! ## Algorithm
//!
//! For each row of the weight matrix:
//! 1. Unpack 8 packed bytes → 32 ternary sign values (+1, 0, -1)
//! 2. Load 32 activation bytes
//! 3. `_mm256_sign_epi8(activations, signs)` → conditional negate/zero
//! 4. `_mm256_maddubs_epi16` + `_mm256_madd_epi16` → horizontal partial sums
//! 5. Final horizontal sum of i32 lanes
//!
//! Expected speedup: 8-16× over scalar for large matrices.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::tensor::TernaryTensor;
use super::ComputeBackend;

/// AVX2 SIMD backend.
#[derive(Debug, Clone, Copy)]
pub struct Avx2Backend;

impl ComputeBackend for Avx2Backend {
    fn name(&self) -> &str { "avx2" }

    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
        assert_eq!(weights.cols(), input.len(), "dimension mismatch");
        // SAFETY: We only construct Avx2Backend after checking is_x86_feature_detected!("avx2")
        unsafe { avx2_ternary_matvec(weights, input) }
    }
}

/// SIMD unpack: 8 packed bytes → 32 sign bytes via nibble lookup.
///
/// Each packed byte holds 4 ternary weights at 2 bits each:
///   bits [1:0] = w0, bits [3:2] = w1, bits [5:4] = w2, bits [7:6] = w3
///
/// Encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1, 0b11 = 0
///
/// Uses `pshufb` as a 4-bit → byte lookup table:
///   - Split each byte into low nibble (w0, w1) and high nibble (w2, w3)
///   - Two lookups per nibble extract the two weights
///   - Interleave to reconstruct the original order
///
/// # Safety
/// Requires SSSE3 (pshufb) — implied by AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_32_ternary_simd(packed_ptr: *const u8) -> __m256i {
    // Lookup tables (128-bit, used by pshufb which indexes by low 4 bits):
    //
    // lut_lo: indexed by nibble, returns sign for the LOW 2 bits of nibble
    //   nibble & 0x3 = {0→-1, 1→0, 2→+1, 3→0}, repeated across indices 0-15
    let lut_lo = _mm_setr_epi8(-1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0);

    // lut_hi: indexed by nibble, returns sign for the HIGH 2 bits of nibble
    //   (nibble >> 2) = {0→-1, 1→0, 2→+1, 3→0}
    //   indices 0-3 → -1, 4-7 → 0, 8-11 → +1, 12-15 → 0
    let lut_hi = _mm_setr_epi8(-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0);

    // Load 8 packed bytes (64 bits) into low half of __m128i
    let raw = _mm_loadl_epi64(packed_ptr as *const __m128i);

    // Split into low and high nibbles
    let mask_0f = _mm_set1_epi8(0x0F);
    let lo_nib = _mm_and_si128(raw, mask_0f);
    let hi_nib = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

    // 4 lookups: each nibble gives 2 weight signs
    let w0 = _mm_shuffle_epi8(lut_lo, lo_nib);  // bits [1:0] of each byte
    let w1 = _mm_shuffle_epi8(lut_hi, lo_nib);  // bits [3:2] of each byte
    let w2 = _mm_shuffle_epi8(lut_lo, hi_nib);  // bits [5:4] of each byte
    let w3 = _mm_shuffle_epi8(lut_hi, hi_nib);  // bits [7:6] of each byte

    // Interleave: [w0_0, w1_0, w2_0, w3_0, w0_1, w1_1, w2_1, w3_1, ...]
    let pair01 = _mm_unpacklo_epi8(w0, w1);     // [w0_0,w1_0, w0_1,w1_1, ..., w0_7,w1_7]
    let pair23 = _mm_unpacklo_epi8(w2, w3);     // [w2_0,w3_0, w2_1,w3_1, ..., w2_7,w3_7]
    let quad_lo = _mm_unpacklo_epi16(pair01, pair23); // bytes 0-3 → 16 signs
    let quad_hi = _mm_unpackhi_epi16(pair01, pair23); // bytes 4-7 → 16 signs

    // Combine into 256-bit result
    _mm256_set_m128i(quad_hi, quad_lo)
}

/// Scalar fallback unpack for testing/verification.
#[cfg(test)]
fn unpack_32_ternary_scalar(packed: &[u8]) -> [i8; 32] {
    let mut signs = [0i8; 32];
    for (byte_i, &byte) in packed.iter().enumerate().take(8) {
        for slot in 0..4 {
            let bits = (byte >> (slot * 2)) & 0b11;
            signs[byte_i * 4 + slot] = match bits {
                0b00 => -1,
                0b10 => 1,
                _ => 0,
            };
        }
    }
    signs
}

/// AVX2 ternary matrix-vector product.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
unsafe fn avx2_ternary_matvec(weights: &TernaryTensor, input: &[i8]) -> Vec<i32> {
    let rows = weights.rows();
    let cols = weights.cols();
    let mut output = vec![0i32; rows];

    let packed = weights.packed_data();
    let bytes_per_row = cols.div_ceil(4);

    // Precompute constants hoisted out of the loop
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);

    #[allow(clippy::needless_range_loop)]
    for row in 0..rows {
        let row_start = row * bytes_per_row;
        let row_data = &packed[row_start..row_start + bytes_per_row];

        let mut acc = _mm256_setzero_si256();
        let mut col = 0;

        // Main SIMD loop: 32 values (8 packed bytes) per iteration
        while col + 32 <= cols {
            let byte_offset = col / 4;

            // SIMD unpack: 8 packed bytes → 32 sign bytes via nibble lookup
            let signs_v = unpack_32_ternary_simd(row_data[byte_offset..].as_ptr());

            // Load 32 activation values
            let acts_v = _mm256_loadu_si256(input[col..].as_ptr() as *const __m256i);

            // sign_epi8: conditional negate/zero based on ternary weight
            let products = _mm256_sign_epi8(acts_v, signs_v);

            // Horizontal partial sum: 32×i8 → 8×i32
            let sum_pairs = _mm256_maddubs_epi16(ones_u8, products); // 32→16 i16
            let sum_quads = _mm256_madd_epi16(sum_pairs, ones_16);   // 16→8 i32

            acc = _mm256_add_epi32(acc, sum_quads);
            col += 32;
        }

        // Horizontal sum of 8 i32 lanes
        let mut row_acc = hsum_epi32(acc);

        // Scalar tail for remaining < 32 values
        while col < cols {
            let byte_offset = col / 4;
            let slot = col % 4;
            let bits = (row_data[byte_offset] >> (slot * 2)) & 0b11;
            let w: i32 = match bits {
                0b00 => -1,
                0b10 => 1,
                _ => 0,
            };
            row_acc += w * (input[col] as i32);
            col += 1;
        }

        output[row] = row_acc;
    }

    output
}

/// Horizontal sum of 8 packed i32 values in a __m256i.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    // Split into two 128-bit halves and add
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);

    // Horizontal add: [a,b,c,d] → [a+b,c+d,a+b,c+d]
    let shuf = _mm_shuffle_epi32(sum128, 0b_01_00_11_10); // swap pairs
    let sum64 = _mm_add_epi32(sum128, shuf);

    // Final pair
    let shuf2 = _mm_shuffle_epi32(sum64, 0b_00_01_00_01);
    let sum32 = _mm_add_epi32(sum64, shuf2);

    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

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
    fn unpack_scalar_roundtrip() {
        let mut packed = [0u8; 8];
        // Byte 0: [-1, 0, +1, 0] = [0b00, 0b01, 0b10, 0b01] = 0b01_10_01_00
        packed[0] = 0b01_10_01_00;
        let signs = unpack_32_ternary_scalar(&packed);
        assert_eq!(signs[0], -1);
        assert_eq!(signs[1], 0);
        assert_eq!(signs[2], 1);
        assert_eq!(signs[3], 0);
        for i in 4..32 {
            assert_eq!(signs[i], -1, "slot {i} should be -1 for 0b00 encoding");
        }
    }

    #[test]
    fn unpack_simd_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Test all 256 possible byte values in each position
        for test_byte in 0..=255u8 {
            let mut packed = [0u8; 8];
            packed[0] = test_byte;
            let scalar = unpack_32_ternary_scalar(&packed);
            let simd = unsafe {
                let v = unpack_32_ternary_simd(packed.as_ptr());
                let mut out = [0i8; 32];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
                out
            };
            assert_eq!(&scalar[..4], &simd[..4],
                "mismatch for byte {test_byte:#04x}: scalar={:?} simd={:?}",
                &scalar[..4], &simd[..4]);
        }
        // Test with all 8 positions populated
        for pattern in [0x00u8, 0x55, 0xAA, 0xFF, 0x24, 0x93, 0xDB, 0x6E] {
            let packed = [pattern; 8];
            let scalar = unpack_32_ternary_scalar(&packed);
            let simd = unsafe {
                let v = unpack_32_ternary_simd(packed.as_ptr());
                let mut out = [0i8; 32];
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
                out
            };
            assert_eq!(scalar, simd, "mismatch for pattern {pattern:#04x}");
        }
    }

    #[test]
    fn avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("skipping AVX2 test — not available");
            return;
        }

        let backend_avx2 = Avx2Backend;
        let backend_scalar = super::super::scalar::ScalarBackend;

        // Test with various sizes including non-32-aligned.
        // Inputs clamped to [-127, 127] — same as real quantized activations.
        // (_mm256_sign_epi8 can't negate -128; absmax quantization guarantees this.)
        for cols in [32, 64, 128, 100, 256, 2560] {
            let rows = 4;
            let n = rows * cols;
            let values: Vec<i8> = (0..n).map(|i| match i % 3 { 0 => -1, 1 => 0, _ => 1 }).collect();
            let w = weights_from_i8(&values, rows, cols);

            let input: Vec<i8> = (0..cols).map(|i| {
                let v = (i % 254) as i32 - 127; // range [-127, 126], avoids -128
                v as i8
            }).collect();

            let y_scalar = backend_scalar.ternary_matvec(&w, &input);
            let y_avx2 = backend_avx2.ternary_matvec(&w, &input);

            assert_eq!(y_scalar, y_avx2, "mismatch at cols={cols}");
        }
    }

    #[test]
    fn avx2_identity_2x2() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Pad to 32 columns (minimum SIMD width) — extra cols are zero-weight
        let mut values = vec![0i8; 2 * 32];
        values[0] = 1;  // (0,0) = +1
        values[33] = 1; // (1,1) = +1
        let w = weights_from_i8(&values, 2, 32);

        let mut input = vec![0i8; 32];
        input[0] = 42;
        input[1] = -17;

        let y = Avx2Backend.ternary_matvec(&w, &input);
        assert_eq!(y[0], 42);
        assert_eq!(y[1], -17);
    }

    #[test]
    fn avx2_all_negative_weights() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let cols = 64;
        let values = vec![-1i8; cols];
        let w = weights_from_i8(&values, 1, cols);
        let input: Vec<i8> = (0..cols).map(|i| (i + 1) as i8).collect();

        let y = Avx2Backend.ternary_matvec(&w, &input);
        let expected: i32 = -(1..=cols as i32).sum::<i32>();
        assert_eq!(y[0], expected);
    }

    #[test]
    fn avx2_real_model_dims() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // BitNet b1.58-2B: embed_dim=2560, intermediate=6912
        let cols = 2560;
        let rows = 8; // just a few rows to keep the test fast
        let n = rows * cols;
        let values: Vec<i8> = (0..n).map(|i| match i % 7 {
            0 | 1 | 2 => 1,
            3 | 4 => -1,
            _ => 0,
        }).collect();
        let w = weights_from_i8(&values, rows, cols);
        let input: Vec<i8> = (0..cols).map(|i| {
            let v = ((i * 7 + 3) % 254) as i32 - 127; // range [-127, 126]
            v.clamp(-127, 127) as i8
        }).collect();

        let y_scalar = super::super::scalar::ScalarBackend.ternary_matvec(&w, &input);
        let y_avx2 = Avx2Backend.ternary_matvec(&w, &input);
        assert_eq!(y_scalar, y_avx2);
    }
}
