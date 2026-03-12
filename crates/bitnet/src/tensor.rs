//! Tensor types for 1.58-bit inference.
//!
//! Ternary weights are packed 4 per byte (2 bits each).
//! Activations are quantized to 8-bit integers with an f32 scale factor.

use std::fmt;

// ---------------------------------------------------------------------------
// Ternary value
// ---------------------------------------------------------------------------

/// A single ternary weight: {-1, 0, +1}.
///
/// Encoded as 2 bits: 0b00 = -1, 0b01 = 0, 0b10 = +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Ternary {
    Neg  = 0b00,
    Zero = 0b01,
    Pos  = 0b10,
}

impl Ternary {
    /// Decode a 2-bit value from the encoding scheme.
    #[inline]
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Ternary::Neg,
            0b01 => Ternary::Zero,
            0b10 => Ternary::Pos,
            _    => Ternary::Zero, // 0b11 is unused, treat as zero
        }
    }

    /// The integer value: -1, 0, or +1.
    #[inline]
    pub fn value(self) -> i8 {
        match self {
            Ternary::Neg  => -1,
            Ternary::Zero =>  0,
            Ternary::Pos  =>  1,
        }
    }
}

// ---------------------------------------------------------------------------
// Ternary tensor — 2-bit packed weight storage
// ---------------------------------------------------------------------------

/// A dense tensor of ternary weights, packed 4 values per byte.
///
/// Memory layout: values are packed in little-endian bit order within each byte.
/// Byte layout: `[w0:2][w1:2][w2:2][w3:2]` where w0 occupies bits 0-1.
///
/// For a weight matrix of shape (rows, cols), the total storage is
/// `ceil(rows * cols / 4)` bytes — a 16× reduction vs f32.
#[derive(Clone)]
pub struct TernaryTensor {
    /// Packed 2-bit ternary values, 4 per byte.
    data: Vec<u8>,
    /// Number of rows (output features).
    rows: usize,
    /// Number of columns (input features).
    cols: usize,
}

impl TernaryTensor {
    /// Create a new ternary tensor from unpacked values.
    ///
    /// `values` must have exactly `rows * cols` elements.
    pub fn pack(values: &[Ternary], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols, "value count must equal rows × cols");

        let n = values.len();
        let packed_len = (n + 3) / 4;
        let mut data = vec![0u8; packed_len];

        for (i, &v) in values.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= (v as u8) << bit_offset;
        }

        Self { data, rows, cols }
    }

    /// Create a zero-initialized ternary tensor.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let packed_len = (n + 3) / 4;
        // 0b01 = Zero for each slot → 0b01_01_01_01 = 0x55
        let data = vec![0x55u8; packed_len];
        Self { data, rows, cols }
    }

    /// Number of rows (output dimension).
    #[inline]
    pub fn rows(&self) -> usize { self.rows }

    /// Number of columns (input dimension).
    #[inline]
    pub fn cols(&self) -> usize { self.cols }

    /// Read a single ternary weight at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Ternary {
        debug_assert!(row < self.rows && col < self.cols);
        let idx = row * self.cols + col;
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        Ternary::from_bits(self.data[byte_idx] >> bit_offset)
    }

    /// Set a single ternary weight at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Ternary) {
        debug_assert!(row < self.rows && col < self.cols);
        let idx = row * self.cols + col;
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let mask = !(0b11u8 << bit_offset);
        self.data[byte_idx] = (self.data[byte_idx] & mask) | ((value as u8) << bit_offset);
    }

    /// Raw packed data (for SIMD kernels).
    #[inline]
    pub fn packed_data(&self) -> &[u8] { &self.data }

    /// Number of packed bytes.
    #[inline]
    pub fn packed_len(&self) -> usize { self.data.len() }

    /// Unpack an entire row into a Vec of Ternary values.
    pub fn unpack_row(&self, row: usize) -> Vec<Ternary> {
        debug_assert!(row < self.rows);
        let start = row * self.cols;
        (0..self.cols)
            .map(|c| {
                let idx = start + c;
                let byte_idx = idx / 4;
                let bit_offset = (idx % 4) * 2;
                Ternary::from_bits(self.data[byte_idx] >> bit_offset)
            })
            .collect()
    }

    /// Iterate over the packed bytes for a single row.
    ///
    /// Returns `(byte_slice, start_offset, values_in_last_byte)`:
    /// - `start_offset`: number of 2-bit slots to skip in the first byte
    ///   (non-zero when `row * cols` is not a multiple of 4).
    /// - `values_in_last_byte`: valid value count in the last byte (1-4).
    pub fn row_bytes(&self, row: usize) -> (&[u8], usize, usize) {
        debug_assert!(row < self.rows);
        let start_val = row * self.cols;
        let end_val = start_val + self.cols;
        let start_byte = start_val / 4;
        let start_offset = start_val % 4;
        let end_byte = (end_val + 3) / 4;
        let end_remainder = end_val % 4;
        let vals_in_last = if end_remainder == 0 { 4 } else { end_remainder };
        (&self.data[start_byte..end_byte], start_offset, vals_in_last)
    }

    /// Create from raw packed bytes (for GGUF loading).
    ///
    /// `data` must have at least `ceil(rows * cols / 4)` bytes.
    pub fn from_packed(data: Vec<u8>, rows: usize, cols: usize) -> Self {
        let expected = (rows * cols + 3) / 4;
        assert!(data.len() >= expected, "packed data too short");
        Self { data, rows, cols }
    }
}

impl fmt::Debug for TernaryTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TernaryTensor({}×{}, {} bytes packed)", self.rows, self.cols, self.data.len())
    }
}

// ---------------------------------------------------------------------------
// Activation tensor — 8-bit quantized with f32 scale
// ---------------------------------------------------------------------------

/// A quantized activation vector or matrix.
///
/// Values are stored as `i8` with a per-tensor scale factor such that
/// `real_value ≈ quantized_value * scale`.
///
/// Quantization uses absmax: `scale = max(|x|) / 127.0`,
/// `quantized = round(x / scale)`.
#[derive(Clone)]
pub struct ActivationTensor {
    /// Quantized 8-bit values.
    data: Vec<i8>,
    /// Scale factor: `real = quantized * scale`.
    scale: f32,
    /// Shape dimensions.
    shape: Vec<usize>,
}

impl ActivationTensor {
    /// Quantize a float slice to 8-bit using absmax scaling.
    ///
    /// `shape` describes the tensor dimensions (e.g., `[batch, features]`).
    /// The total number of elements must equal the product of shape dimensions.
    pub fn quantize(values: &[f32], shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected, "value count must match shape");

        let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax < f32::EPSILON { 1.0 } else { absmax / 127.0 };
        let inv_scale = 1.0 / scale;

        let data: Vec<i8> = values
            .iter()
            .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        Self { data, scale, shape }
    }

    /// Create from pre-quantized data.
    pub fn from_raw(data: Vec<i8>, scale: f32, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "data length must match shape");
        Self { data, scale, shape }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&q| q as f32 * self.scale).collect()
    }

    /// Raw quantized data.
    #[inline]
    pub fn data(&self) -> &[i8] { &self.data }

    /// Scale factor.
    #[inline]
    pub fn scale(&self) -> f32 { self.scale }

    /// Shape dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] { &self.shape }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    /// Whether the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

impl fmt::Debug for ActivationTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActivationTensor({:?}, scale={:.6})", self.shape, self.scale)
    }
}

// ---------------------------------------------------------------------------
// Float tensor — for layers that stay in f32 (embeddings, norms)
// ---------------------------------------------------------------------------

/// A simple f32 tensor for non-quantized operations (embeddings, RMSNorm, logits).
#[derive(Clone)]
pub struct FloatTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl FloatTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "data length must match shape");
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    #[inline]
    pub fn data(&self) -> &[f32] { &self.data }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] { &mut self.data }

    #[inline]
    pub fn shape(&self) -> &[usize] { &self.shape }

    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Quantize this float tensor to an 8-bit activation tensor.
    pub fn to_activation(&self) -> ActivationTensor {
        ActivationTensor::quantize(&self.data, self.shape.clone())
    }
}

impl fmt::Debug for FloatTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FloatTensor({:?})", self.shape)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_value_roundtrip() {
        assert_eq!(Ternary::Neg.value(), -1);
        assert_eq!(Ternary::Zero.value(), 0);
        assert_eq!(Ternary::Pos.value(), 1);
    }

    #[test]
    fn ternary_bits_roundtrip() {
        for t in [Ternary::Neg, Ternary::Zero, Ternary::Pos] {
            assert_eq!(Ternary::from_bits(t as u8), t);
        }
    }

    #[test]
    fn pack_unpack_identity() {
        let values = vec![
            Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos,
            Ternary::Zero, Ternary::Neg, Ternary::Neg, Ternary::Pos,
        ];
        let tensor = TernaryTensor::pack(&values, 2, 4);

        // Verify dimensions.
        assert_eq!(tensor.rows(), 2);
        assert_eq!(tensor.cols(), 4);
        assert_eq!(tensor.packed_len(), 2); // 8 values / 4 per byte

        // Verify individual access.
        assert_eq!(tensor.get(0, 0), Ternary::Pos);
        assert_eq!(tensor.get(0, 1), Ternary::Neg);
        assert_eq!(tensor.get(0, 2), Ternary::Zero);
        assert_eq!(tensor.get(0, 3), Ternary::Pos);
        assert_eq!(tensor.get(1, 0), Ternary::Zero);
        assert_eq!(tensor.get(1, 1), Ternary::Neg);
        assert_eq!(tensor.get(1, 2), Ternary::Neg);
        assert_eq!(tensor.get(1, 3), Ternary::Pos);

        // Verify row unpack.
        assert_eq!(tensor.unpack_row(0), &values[0..4]);
        assert_eq!(tensor.unpack_row(1), &values[4..8]);
    }

    #[test]
    fn pack_non_aligned_cols() {
        // 3 cols — not a multiple of 4, exercises padding logic.
        let values = vec![
            Ternary::Pos, Ternary::Neg, Ternary::Zero,
            Ternary::Neg, Ternary::Pos, Ternary::Pos,
        ];
        let tensor = TernaryTensor::pack(&values, 2, 3);
        assert_eq!(tensor.rows(), 2);
        assert_eq!(tensor.cols(), 3);

        assert_eq!(tensor.get(0, 0), Ternary::Pos);
        assert_eq!(tensor.get(0, 2), Ternary::Zero);
        assert_eq!(tensor.get(1, 0), Ternary::Neg);
        assert_eq!(tensor.get(1, 2), Ternary::Pos);
    }

    #[test]
    fn zeros_are_zero() {
        let tensor = TernaryTensor::zeros(4, 8);
        for r in 0..4 {
            for c in 0..8 {
                assert_eq!(tensor.get(r, c), Ternary::Zero);
            }
        }
    }

    #[test]
    fn set_value() {
        let mut tensor = TernaryTensor::zeros(2, 4);
        tensor.set(0, 1, Ternary::Pos);
        tensor.set(1, 3, Ternary::Neg);

        assert_eq!(tensor.get(0, 0), Ternary::Zero);
        assert_eq!(tensor.get(0, 1), Ternary::Pos);
        assert_eq!(tensor.get(1, 3), Ternary::Neg);
    }

    #[test]
    fn row_bytes_aligned() {
        let values = vec![Ternary::Pos; 8];
        let tensor = TernaryTensor::pack(&values, 2, 4);
        let (bytes, start_offset, vals_in_last) = tensor.row_bytes(0);
        assert_eq!(bytes.len(), 1); // 4 values = 1 byte
        assert_eq!(start_offset, 0);
        assert_eq!(vals_in_last, 4);
    }

    #[test]
    fn row_bytes_unaligned() {
        let values = vec![Ternary::Pos; 10];
        let tensor = TernaryTensor::pack(&values, 2, 5);
        let (bytes, start_offset, vals_in_last) = tensor.row_bytes(0);
        assert_eq!(bytes.len(), 2); // ceil(5/4) = 2 bytes
        assert_eq!(start_offset, 0); // first row always starts at 0
        assert_eq!(vals_in_last, 1); // 5 % 4 = 1

        // Row 1 starts at value 5 → byte 1, offset 1.
        let (bytes1, start_offset1, vals_in_last1) = tensor.row_bytes(1);
        assert_eq!(start_offset1, 1); // 5 % 4 = 1
        assert!(bytes1.len() >= 2); // spans 2 bytes
        assert_eq!(vals_in_last1, 2); // (5+5)%4 = 2
    }

    #[test]
    fn from_packed_roundtrip() {
        let values = vec![Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos];
        let original = TernaryTensor::pack(&values, 1, 4);
        let restored = TernaryTensor::from_packed(original.packed_data().to_vec(), 1, 4);
        for c in 0..4 {
            assert_eq!(original.get(0, c), restored.get(0, c));
        }
    }

    #[test]
    fn activation_quantize_dequantize() {
        let values = vec![1.0, -0.5, 0.0, 0.75, -1.0];
        let quant = ActivationTensor::quantize(&values, vec![5]);

        assert_eq!(quant.len(), 5);
        assert_eq!(quant.shape(), &[5]);

        // Dequantize and check within tolerance.
        let restored = quant.dequantize();
        for (orig, rest) in values.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.02, "expected ~{orig}, got {rest}");
        }
    }

    #[test]
    fn activation_zero_input() {
        let values = vec![0.0; 4];
        let quant = ActivationTensor::quantize(&values, vec![4]);
        let restored = quant.dequantize();
        for v in &restored {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn activation_scale_correctness() {
        let values = vec![127.0, -127.0];
        let quant = ActivationTensor::quantize(&values, vec![2]);
        // absmax = 127.0, scale = 127.0/127.0 = 1.0
        assert!((quant.scale() - 1.0).abs() < f32::EPSILON);
        assert_eq!(quant.data(), &[127i8, -127i8]);
    }

    #[test]
    fn float_tensor_to_activation() {
        let ft = FloatTensor::new(vec![0.5, -0.3, 0.0, 1.0], vec![2, 2]);
        let act = ft.to_activation();
        assert_eq!(act.shape(), &[2, 2]);
        assert_eq!(act.len(), 4);
    }

    // 16× compression ratio test.
    #[test]
    fn compression_ratio() {
        let n = 1024;
        let f32_bytes = n * 4;
        let ternary_bytes = (n + 3) / 4;
        let ratio = f32_bytes as f64 / ternary_bytes as f64;
        assert!((ratio - 16.0).abs() < 0.1, "expected ~16× compression, got {ratio}×");
    }
}
