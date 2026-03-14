//! GGUF model file parser for loading ternary (1.58-bit) LLM weights.
//!
//! Implements the GGUF v3 format as used by llama.cpp and BitNet.cpp.
//! Supports ternary tensor types TQ1_0 and TQ2_0, plus F32/F16/BF16 for
//! non-quantized layers (embeddings, norms, output heads).
//!
//! No mmap, no unsafe, no external ML deps — just std + thiserror.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use thiserror::Error;
use tracing::{debug, info};

use crate::tensor::{FloatTensor, Ternary, TernaryTensor};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46554747; // 'G','G','U','F' as little-endian u32

/// Supported GGUF version.
const GGUF_VERSION: u32 = 3;

/// Default tensor data alignment in bytes.
const DEFAULT_ALIGNMENT: usize = 32;

/// TQ2_0 block size: 256 elements per block.
const TQ2_BLOCK_ELEMENTS: usize = 256;

/// TQ2_0 block byte size: 64 data bytes + 2 scale bytes.
const TQ2_BLOCK_BYTES: usize = 66;

/// TQ1_0 block size: 256 elements per block.
const TQ1_BLOCK_ELEMENTS: usize = 256;

/// TQ1_0 block byte size: 48 base-3 + 4 tail + 2 scale.
const TQ1_BLOCK_BYTES: usize = 54;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur while parsing a GGUF file.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("bad magic number: expected 0x46475547, got 0x{0:08X}")]
    BadMagic(u32),

    #[error("unsupported GGUF version: {0} (expected {GGUF_VERSION})")]
    UnsupportedVersion(u32),

    #[error("invalid metadata value type: {0}")]
    InvalidValueType(u32),

    #[error("invalid UTF-8 string in GGUF metadata")]
    InvalidString(#[from] std::string::FromUtf8Error),

    #[error("unsupported tensor type code: {0}")]
    UnsupportedTensorType(u32),

    #[error("tensor shape mismatch: expected {expected} elements, got {actual}")]
    TensorShapeMismatch { expected: u64, actual: u64 },

    #[error("missing metadata key: {0}")]
    MissingMetadata(String),

    #[error("metadata type mismatch for key '{key}': expected {expected}")]
    MetadataTypeMismatch {
        key: String,
        expected: &'static str,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, GgufError>;

// ---------------------------------------------------------------------------
// GGML tensor types
// ---------------------------------------------------------------------------

/// Tensor element types supported by GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    /// BitNet.cpp I2_S: 2-bit packed ternary + single per-tensor float scale.
    I2S = 36,
}

impl TryFrom<u32> for GgmlType {
    type Error = GgufError;

    fn try_from(v: u32) -> Result<Self> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            30 => Ok(GgmlType::BF16),
            34 => Ok(GgmlType::TQ1_0),
            35 => Ok(GgmlType::TQ2_0),
            36 => Ok(GgmlType::I2S),
            _ => Err(GgufError::UnsupportedTensorType(v)),
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

/// A typed metadata value from the GGUF header.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            MetadataValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor info
// ---------------------------------------------------------------------------

/// Metadata about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Shape dimensions, outermost first (reversed from GGUF storage order).
    pub shape: Vec<usize>,
    /// Element type.
    pub ggml_type: GgmlType,
    /// Byte offset from start of tensor data section.
    pub offset: u64,
    /// Total number of elements.
    pub n_elements: u64,
}

// ---------------------------------------------------------------------------
// Model config
// ---------------------------------------------------------------------------

/// LLaMA-family hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub embedding_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub context_length: u32,
    pub intermediate_size: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// RoPE type: 0 = interleaved (llama.cpp NORM), 2 = halved (NeoX/HF).
    /// -1 or absent defaults to interleaved.
    pub rope_type: i32,
    /// Hidden activation function name (e.g., "silu", "relu2", "gelu").
    /// Defaults to "silu" if not specified.
    pub hidden_act: String,
    /// Model name from `general.name` metadata (e.g., "Falcon3-7B-Instruct-1.58bit").
    pub model_name: Option<String>,
}

// ---------------------------------------------------------------------------
// GgufReader — generic little-endian binary reader
// ---------------------------------------------------------------------------

/// A little-endian binary reader over any `Read + Seek` source.
struct GgufReader<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> GgufReader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Read a GGUF string: u64 length prefix + UTF-8 bytes.
    fn read_gguf_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }

    /// Read a metadata value given its type code.
    fn read_metadata_value(&mut self, type_id: u32) -> Result<MetadataValue> {
        match type_id {
            0 => Ok(MetadataValue::U8(self.read_u8()?)),
            1 => Ok(MetadataValue::I8(self.read_i8()?)),
            2 => Ok(MetadataValue::U16(self.read_u16()?)),
            3 => Ok(MetadataValue::I16(self.read_i16()?)),
            4 => Ok(MetadataValue::U32(self.read_u32()?)),
            5 => Ok(MetadataValue::I32(self.read_i32()?)),
            6 => Ok(MetadataValue::F32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_bool()?)),
            8 => Ok(MetadataValue::String(self.read_gguf_string()?)),
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_metadata_value(elem_type)?);
                }
                Ok(MetadataValue::Array(items))
            }
            10 => Ok(MetadataValue::U64(self.read_u64()?)),
            11 => Ok(MetadataValue::I64(self.read_i64()?)),
            12 => Ok(MetadataValue::F64(self.read_f64()?)),
            _ => Err(GgufError::InvalidValueType(type_id)),
        }
    }

    fn stream_position(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }
}

// ---------------------------------------------------------------------------
// Float conversion helpers
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision (f16) value to f32.
///
/// Layout: 1 sign | 5 exponent | 10 mantissa
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: convert to normalized f32
            // Value = (-1)^sign * 2^(-14) * (mant / 1024)
            let val = (mant as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    } else {
        // Normalized: rebias exponent from f16 bias (15) to f32 bias (127)
        let f32_exp = exp + 127 - 15;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert a BF16 value to f32. BF16 is just the upper 16 bits of f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// TQ2_0 unpacking
// ---------------------------------------------------------------------------

/// Remap table for TQ2_0: GGUF encoding → our ternary encoding.
///
/// GGUF TQ2_0: 0 = -1, 1 = 0, 2 = +1
/// Our encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1
const TQ2_REMAP: [u8; 4] = [
    0b00, // GGUF 0 → Neg
    0b01, // GGUF 1 → Zero
    0b10, // GGUF 2 → Pos
    0b01, // GGUF 3 → Zero (unused, safe fallback)
];

/// Unpack a TQ2_0 block (256 elements, 66 bytes) into ternary values.
///
/// Returns the 256 ternary values and the per-block f16 scale.
fn unpack_tq2_0_block(block: &[u8]) -> ([Ternary; TQ2_BLOCK_ELEMENTS], f32) {
    debug_assert_eq!(block.len(), TQ2_BLOCK_BYTES);

    let mut values = [Ternary::Zero; TQ2_BLOCK_ELEMENTS];

    // First 64 bytes: 2 bits per value, 4 values per byte
    for (byte_idx, &b) in block[..64].iter().enumerate() {
        for slot in 0..4 {
            let gguf_code = (b >> (slot * 2)) & 0x03;
            let val_idx = byte_idx * 4 + slot;
            values[val_idx] = Ternary::from_bits(TQ2_REMAP[gguf_code as usize]);
        }
    }

    // Last 2 bytes: f16 scale
    let scale_bits = u16::from_le_bytes([block[64], block[65]]);
    let scale = f16_to_f32(scale_bits);

    (values, scale)
}

/// Unpack TQ2_0 tensor data into a `TernaryTensor` and averaged weight scale.
fn unpack_tq2_0(data: &[u8], n_elements: u64) -> (TernaryTensor, f32) {
    let n = n_elements as usize;
    let n_blocks = n.div_ceil(TQ2_BLOCK_ELEMENTS);

    let mut all_values = Vec::with_capacity(n);
    let mut scale_sum = 0.0f64;

    for block_idx in 0..n_blocks {
        let start = block_idx * TQ2_BLOCK_BYTES;
        let block = &data[start..start + TQ2_BLOCK_BYTES];
        let (values, scale) = unpack_tq2_0_block(block);

        let remaining = n - block_idx * TQ2_BLOCK_ELEMENTS;
        let count = remaining.min(TQ2_BLOCK_ELEMENTS);
        all_values.extend_from_slice(&values[..count]);
        scale_sum += scale.abs() as f64;
    }

    let avg_scale = (scale_sum / n_blocks as f64) as f32;

    // Pack into our 2-bit format
    let packed_len = n.div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    for (i, &v) in all_values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= (v as u8) << bit_offset;
    }

    // Shape is set by caller; use 1D here
    (TernaryTensor::from_packed(packed, 1, n), avg_scale)
}

// ---------------------------------------------------------------------------
// I2_S unpacking (BitNet.cpp native format)
// ---------------------------------------------------------------------------

/// Unpack I2_S data: 2-bit packed ternary with a single per-tensor float scale.
///
/// Layout: `[n_elements/4 bytes of packed data] [4 bytes f32 scale]`
/// Encoding: 0=-1, 1=0, 2=+1 (same as TQ2_0).
///
/// **Interleaved packing** (from BitNet.cpp `quantize_i2_s`):
/// Values are packed in blocks of `QK_I2_S = 128` elements into 32 bytes.
/// Within each 32-byte block, byte `p` (0..31) contains 4 values from
/// positions `p`, `p+32`, `p+64`, `p+96` within the 128-element block:
///   - bits 7:6 = value at offset `p + 0*32`
///   - bits 5:4 = value at offset `p + 1*32`
///   - bits 3:2 = value at offset `p + 2*32`
///   - bits 1:0 = value at offset `p + 3*32`
fn unpack_i2s(data: &[u8], n_elements: u64) -> (TernaryTensor, f32) {
    let n = n_elements as usize;
    let packed_bytes = n.div_ceil(4);

    // Scale is the last 4 bytes after the packed data
    let scale = if data.len() >= packed_bytes + 4 {
        let scale_bytes = &data[packed_bytes..packed_bytes + 4];
        f32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]])
    } else {
        1.0
    };

    // I2_S: 128-element blocks, 32 bytes per block.
    // Each byte holds 4 ternary values from 4 interleaved groups of 32:
    //   bits 7:6 → group 0 (offset p),    bits 5:4 → group 1 (offset p+32),
    //   bits 3:2 → group 2 (offset p+64), bits 1:0 → group 3 (offset p+96).
    // GGUF encoding: 0 = -1, 1 = 0, 2 = +1 (remapped via TQ2_REMAP).
    const QK: usize = 128;
    const BLOCK_BYTES: usize = QK / 4; // 32

    let mut all_values = vec![Ternary::Zero; n];

    let n_full_blocks = n / QK;
    let remainder = n % QK;

    for block_idx in 0..n_full_blocks {
        let byte_start = block_idx * BLOCK_BYTES;
        let val_start = block_idx * QK;

        for p in 0..BLOCK_BYTES {
            let b = data[byte_start + p];
            all_values[val_start + p]      = Ternary::from_bits(TQ2_REMAP[((b >> 6) & 0x03) as usize]);
            all_values[val_start + p + 32] = Ternary::from_bits(TQ2_REMAP[((b >> 4) & 0x03) as usize]);
            all_values[val_start + p + 64] = Ternary::from_bits(TQ2_REMAP[((b >> 2) & 0x03) as usize]);
            all_values[val_start + p + 96] = Ternary::from_bits(TQ2_REMAP[(b & 0x03) as usize]);
        }
    }

    // Handle remainder
    if remainder > 0 {
        let byte_start = n_full_blocks * BLOCK_BYTES;
        let val_start = n_full_blocks * QK;
        let mut idx = 0;
        let rem_bytes = remainder.div_ceil(4);
        for p in 0..rem_bytes {
            let b = data[byte_start + p];
            for shift in [6, 4, 2, 0] {
                if idx >= remainder { break; }
                let code = (b >> shift) & 0x03;
                all_values[val_start + idx] = Ternary::from_bits(TQ2_REMAP[code as usize]);
                idx += 1;
            }
        }
    }

    // Repack into our internal 2-bit format
    let mut packed = vec![0u8; packed_bytes];
    for (i, &v) in all_values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= (v as u8) << bit_offset;
    }

    (TernaryTensor::from_packed(packed, 1, n), scale)
}

// ---------------------------------------------------------------------------
// TQ1_0 unpacking
// ---------------------------------------------------------------------------

/// Decode a single byte of base-3 packed data into 5 ternary values.
///
/// Each byte encodes 5 values in base 3 (3^5 = 243 ≤ 255).
/// Values: 0 = -1, 1 = 0, 2 = +1.
fn decode_base3(byte: u8) -> [Ternary; 5] {
    let mut result = [Ternary::Zero; 5];
    let mut val = byte as u16; // u16 to avoid overflow during division

    for item in &mut result {
        let rem = (val % 3) as u8;
        *item = match rem {
            0 => Ternary::Neg,
            1 => Ternary::Zero,
            2 => Ternary::Pos,
            _ => unreachable!(),
        };
        val /= 3;
    }

    result
}

/// Unpack a TQ1_0 block (256 elements, 54 bytes) into ternary values.
///
/// Layout:
/// - Bytes 0..48: 48 base-3 bytes → 240 values (5 values per byte)
/// - Bytes 48..52: 4 bytes of 2-bit packed values → 16 values (4 per byte)
/// - Bytes 52..54: f16 scale
fn unpack_tq1_0_block(block: &[u8]) -> ([Ternary; TQ1_BLOCK_ELEMENTS], f32) {
    debug_assert_eq!(block.len(), TQ1_BLOCK_BYTES);

    let mut values = [Ternary::Zero; TQ1_BLOCK_ELEMENTS];

    // First 240 values: base-3 packed, 5 per byte
    for (byte_idx, &b) in block[..48].iter().enumerate() {
        let decoded = decode_base3(b);
        let base = byte_idx * 5;
        values[base..base + 5].copy_from_slice(&decoded);
    }

    // Last 16 values: 2-bit packed (same encoding as TQ2_0)
    for byte_idx in 0..4 {
        let b = block[48 + byte_idx];
        for slot in 0..4 {
            let gguf_code = (b >> (slot * 2)) & 0x03;
            let val_idx = 240 + byte_idx * 4 + slot;
            values[val_idx] = Ternary::from_bits(TQ2_REMAP[gguf_code as usize]);
        }
    }

    // Scale: f16 at bytes 52..54
    let scale_bits = u16::from_le_bytes([block[52], block[53]]);
    let scale = f16_to_f32(scale_bits);

    (values, scale)
}

/// Unpack TQ1_0 tensor data into a `TernaryTensor` and averaged weight scale.
fn unpack_tq1_0(data: &[u8], n_elements: u64) -> (TernaryTensor, f32) {
    let n = n_elements as usize;
    let n_blocks = n.div_ceil(TQ1_BLOCK_ELEMENTS);

    let mut all_values = Vec::with_capacity(n);
    let mut scale_sum = 0.0f64;

    for block_idx in 0..n_blocks {
        let start = block_idx * TQ1_BLOCK_BYTES;
        let block = &data[start..start + TQ1_BLOCK_BYTES];
        let (values, scale) = unpack_tq1_0_block(block);

        let remaining = n - block_idx * TQ1_BLOCK_ELEMENTS;
        let count = remaining.min(TQ1_BLOCK_ELEMENTS);
        all_values.extend_from_slice(&values[..count]);
        scale_sum += scale.abs() as f64;
    }

    let avg_scale = (scale_sum / n_blocks as f64) as f32;

    let packed_len = n.div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    for (i, &v) in all_values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= (v as u8) << bit_offset;
    }

    (TernaryTensor::from_packed(packed, 1, n), avg_scale)
}

// ---------------------------------------------------------------------------
// Float tensor loading
// ---------------------------------------------------------------------------

/// Load f32/f16/bf16 tensor data and convert to f32 vec.
fn load_float_data(data: &[u8], ggml_type: GgmlType, n_elements: u64) -> Vec<f32> {
    let n = n_elements as usize;
    match ggml_type {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let offset = i * 4;
                let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
                out.push(f32::from_le_bytes(bytes));
            }
            out
        }
        GgmlType::F16 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let offset = i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                out.push(f16_to_f32(bits));
            }
            out
        }
        GgmlType::BF16 => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let offset = i * 2;
                let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                out.push(bf16_to_f32(bits));
            }
            out
        }
        _ => panic!("load_float_data called with non-float type: {:?}", ggml_type),
    }
}

// ---------------------------------------------------------------------------
// GgufFile — main public API
// ---------------------------------------------------------------------------

/// A parsed GGUF file handle. Parses the header cheaply; tensor data is
/// loaded on demand via `load_ternary()` / `load_float()`.
pub struct GgufFile {
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorInfo>,
    tensor_data_offset: u64,
    alignment: usize,
    path: PathBuf,
}

impl GgufFile {
    /// Open and parse a GGUF file from disk.
    ///
    /// This reads only the header, metadata, and tensor info — no tensor data
    /// is loaded until `load_ternary()` or `load_float()` is called.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let mut gguf = Self::open_reader(reader)?;
        gguf.path = path;
        Ok(gguf)
    }

    /// Parse GGUF from any `Read + Seek` source (enables in-memory testing).
    pub fn open_reader<R: Read + Seek>(reader: R) -> Result<Self> {
        let mut r = GgufReader::new(reader);

        // Header
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = r.read_u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = r.read_u64()?;
        let metadata_count = r.read_u64()?;

        debug!(
            tensor_count,
            metadata_count, "parsing GGUF v{version} header"
        );

        // Metadata
        let mut metadata = HashMap::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = r.read_gguf_string()?;
            let type_id = r.read_u32()?;
            let value = r.read_metadata_value(type_id)?;
            metadata.insert(key, value);
        }

        // Alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_ALIGNMENT);

        // Tensor infos
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = r.read_gguf_string()?;
            let n_dims = r.read_u32()? as usize;

            // GGUF stores dimensions innermost-first; we reverse to outermost-first
            let mut shape_reversed = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape_reversed.push(r.read_u64()? as usize);
            }
            shape_reversed.reverse();

            let type_code = r.read_u32()?;
            let ggml_type = GgmlType::try_from(type_code)?;
            let offset = r.read_u64()?;

            let n_elements: u64 = shape_reversed.iter().map(|&d| d as u64).product();

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape: shape_reversed,
                    ggml_type,
                    offset,
                    n_elements,
                },
            );
        }

        // Compute tensor data start: align current position to alignment boundary
        let header_end = r.stream_position()?;
        let tensor_data_offset = align_offset(header_end, alignment);

        info!(
            tensor_count,
            metadata_count,
            tensor_data_offset,
            alignment,
            "GGUF header parsed"
        );

        Ok(Self {
            metadata,
            tensors,
            tensor_data_offset,
            alignment,
            path: PathBuf::new(),
        })
    }

    /// All metadata key-value pairs.
    pub fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.metadata
    }

    /// Get a specific metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Extract LLaMA model hyperparameters from metadata.
    pub fn model_config(&self) -> Result<ModelConfig> {
        // Detect architecture prefix: "llama" or "bitnet-b1.58" etc.
        let arch = self
            .get_metadata("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let get_u32 = |suffix: &str| -> Result<u32> {
            let key = format!("{arch}.{suffix}");
            self.metadata
                .get(&key)
                .ok_or_else(|| GgufError::MissingMetadata(key.clone()))?
                .as_u32()
                .ok_or(GgufError::MetadataTypeMismatch {
                    key,
                    expected: "u32",
                })
        };

        let get_f32 = |suffix: &str| -> Result<f32> {
            let key = format!("{arch}.{suffix}");
            self.metadata
                .get(&key)
                .ok_or_else(|| GgufError::MissingMetadata(key.clone()))?
                .as_f32()
                .ok_or(GgufError::MetadataTypeMismatch {
                    key,
                    expected: "f32",
                })
        };

        // rope_type: optional, defaults to 0 (interleaved/NORM) if absent.
        // In llama.cpp: 0=NORM (interleaved), 2=NEOX (halved), -1=unset.
        let rope_type = {
            let key = format!("{arch}.rope.scaling.type");
            self.metadata
                .get(&key)
                .and_then(|v| v.as_u32().map(|u| u as i32))
                .unwrap_or(0)
        };

        // Hidden activation: check general.hidden_act or {arch}.hidden_act, default to "silu"
        let hidden_act = self
            .get_metadata("general.hidden_act")
            .or_else(|| {
                let key = format!("{arch}.hidden_act");
                self.metadata.get(&key)
            })
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string();

        let model_name = self
            .get_metadata("general.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ModelConfig {
            vocab_size: get_u32("vocab_size")?,
            embedding_dim: get_u32("embedding_length")?,
            n_layers: get_u32("block_count")?,
            n_heads: get_u32("attention.head_count")?,
            n_kv_heads: get_u32("attention.head_count_kv")?,
            context_length: get_u32("context_length")?,
            intermediate_size: get_u32("feed_forward_length")?,
            rope_theta: get_f32("rope.freq_base")?,
            rms_norm_eps: get_f32("attention.layer_norm_rms_epsilon")?,
            rope_type,
            hidden_act,
            model_name,
        })
    }

    /// All tensor infos.
    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Load a ternary tensor (TQ1_0 or TQ2_0) by name.
    ///
    /// Returns `(TernaryTensor, weight_scale)` ready for `BitLinear::new()`.
    /// The tensor is reshaped to (rows, cols) from the stored shape.
    pub fn load_ternary(&self, name: &str) -> Result<(TernaryTensor, f32)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

        let data = self.read_tensor_data(info)?;

        let (mut tensor, scale) = match info.ggml_type {
            GgmlType::TQ2_0 => unpack_tq2_0(&data, info.n_elements),
            GgmlType::TQ1_0 => unpack_tq1_0(&data, info.n_elements),
            GgmlType::I2S => unpack_i2s(&data, info.n_elements),
            other => {
                return Err(GgufError::UnsupportedTensorType(other as u32));
            }
        };

        // Reshape: if 2D, apply (rows, cols)
        if info.shape.len() == 2 {
            tensor = TernaryTensor::from_packed(
                tensor.packed_data().to_vec(),
                info.shape[0],
                info.shape[1],
            );
        }

        debug!(
            name,
            ?info.ggml_type,
            scale,
            rows = tensor.rows(),
            cols = tensor.cols(),
            "loaded ternary tensor"
        );

        Ok((tensor, scale))
    }

    /// Load a float tensor (F32, F16, or BF16) by name.
    pub fn load_float(&self, name: &str) -> Result<FloatTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

        let data = self.read_tensor_data(info)?;
        let float_data = load_float_data(&data, info.ggml_type, info.n_elements);

        debug!(
            name,
            ?info.ggml_type,
            n_elements = info.n_elements,
            "loaded float tensor"
        );

        Ok(FloatTensor::new(float_data, info.shape.clone()))
    }

    /// Load the first `n` raw bytes of a tensor (for diagnostics).
    pub fn load_raw_bytes(&self, name: &str, n: usize) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

        let byte_size = tensor_byte_size(info.ggml_type, info.n_elements, self.alignment);
        let read_len = n.min(byte_size);

        let abs_offset = self.tensor_data_offset + info.offset;
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(abs_offset))?;

        let mut buf = vec![0u8; read_len];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read raw tensor bytes from the file.
    fn read_tensor_data(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let byte_size = tensor_byte_size(info.ggml_type, info.n_elements, self.alignment);

        let abs_offset = self.tensor_data_offset + info.offset;
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(abs_offset))?;

        let mut buf = vec![0u8; byte_size];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read raw tensor bytes (for diagnostic tools that need to try different interpretations).
    pub fn read_tensor_data_pub(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        self.read_tensor_data(info)
    }

    /// Load a ternary tensor from an in-memory reader source.
    /// Used for testing when there's no file on disk.
    pub fn load_ternary_from_reader<R: Read + Seek>(
        &self,
        name: &str,
        mut reader: R,
    ) -> Result<(TernaryTensor, f32)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

        let byte_size = tensor_byte_size(info.ggml_type, info.n_elements, self.alignment);
        let abs_offset = self.tensor_data_offset + info.offset;
        reader.seek(SeekFrom::Start(abs_offset))?;

        let mut buf = vec![0u8; byte_size];
        reader.read_exact(&mut buf)?;

        let (mut tensor, scale) = match info.ggml_type {
            GgmlType::TQ2_0 => unpack_tq2_0(&buf, info.n_elements),
            GgmlType::TQ1_0 => unpack_tq1_0(&buf, info.n_elements),
            other => return Err(GgufError::UnsupportedTensorType(other as u32)),
        };

        if info.shape.len() == 2 {
            tensor = TernaryTensor::from_packed(
                tensor.packed_data().to_vec(),
                info.shape[0],
                info.shape[1],
            );
        }

        Ok((tensor, scale))
    }

    /// Load a float tensor from an in-memory reader source.
    pub fn load_float_from_reader<R: Read + Seek>(
        &self,
        name: &str,
        mut reader: R,
    ) -> Result<FloatTensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingMetadata(name.to_string()))?;

        let byte_size = tensor_byte_size(info.ggml_type, info.n_elements, self.alignment);
        let abs_offset = self.tensor_data_offset + info.offset;
        reader.seek(SeekFrom::Start(abs_offset))?;

        let mut buf = vec![0u8; byte_size];
        reader.read_exact(&mut buf)?;

        let float_data = load_float_data(&buf, info.ggml_type, info.n_elements);
        Ok(FloatTensor::new(float_data, info.shape.clone()))
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GgufFile({}, {} metadata, {} tensors)",
            self.path.display(),
            self.metadata.len(),
            self.tensors.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Align an offset up to the given alignment boundary.
fn align_offset(offset: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    offset.div_ceil(a) * a
}

/// Compute the byte size needed for a tensor's data.
fn tensor_byte_size(ggml_type: GgmlType, n_elements: u64, _alignment: usize) -> usize {
    let n = n_elements as usize;
    match ggml_type {
        GgmlType::F32 => n * 4,
        GgmlType::F16 | GgmlType::BF16 => n * 2,
        GgmlType::TQ2_0 => {
            let n_blocks = n.div_ceil(TQ2_BLOCK_ELEMENTS);
            n_blocks * TQ2_BLOCK_BYTES
        }
        GgmlType::TQ1_0 => {
            let n_blocks = n.div_ceil(TQ1_BLOCK_ELEMENTS);
            n_blocks * TQ1_BLOCK_BYTES
        }
        GgmlType::I2S => {
            // packed 2-bit data + 4-byte float scale
            n.div_ceil(4) + 4
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -- Helper: build a synthetic GGUF file in memory --

    struct GgufBuilder {
        metadata: Vec<(String, u32, Vec<u8>)>, // key, type_id, encoded value
        tensor_infos: Vec<(String, Vec<u64>, u32, Vec<u8>)>, // name, shape, type, data
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                metadata: Vec::new(),
                tensor_infos: Vec::new(),
            }
        }

        fn add_metadata_u32(&mut self, key: &str, val: u32) {
            self.metadata
                .push((key.to_string(), 4, val.to_le_bytes().to_vec()));
        }

        fn add_metadata_f32(&mut self, key: &str, val: f32) {
            self.metadata
                .push((key.to_string(), 6, val.to_le_bytes().to_vec()));
        }

        fn add_metadata_string(&mut self, key: &str, val: &str) {
            let mut encoded = Vec::new();
            encoded.extend_from_slice(&(val.len() as u64).to_le_bytes());
            encoded.extend_from_slice(val.as_bytes());
            self.metadata.push((key.to_string(), 8, encoded));
        }

        fn add_metadata_bool(&mut self, key: &str, val: bool) {
            self.metadata
                .push((key.to_string(), 7, vec![val as u8]));
        }

        fn add_metadata_array_u32(&mut self, key: &str, vals: &[u32]) {
            let mut encoded = Vec::new();
            encoded.extend_from_slice(&4u32.to_le_bytes()); // elem type = u32
            encoded.extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for &v in vals {
                encoded.extend_from_slice(&v.to_le_bytes());
            }
            self.metadata.push((key.to_string(), 9, encoded));
        }

        fn add_tensor(&mut self, name: &str, shape: &[u64], type_code: u32, data: Vec<u8>) {
            // Shape stored innermost-first in GGUF, so reverse our outermost-first
            let mut gguf_shape = shape.to_vec();
            gguf_shape.reverse();
            self.tensor_infos
                .push((name.to_string(), gguf_shape, type_code, data));
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            let alignment = DEFAULT_ALIGNMENT;

            // Magic + version
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&GGUF_VERSION.to_le_bytes());

            // Tensor count + metadata count
            out.extend_from_slice(&(self.tensor_infos.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

            // Metadata
            for (key, type_id, encoded) in &self.metadata {
                // Key: u64 len + bytes
                out.extend_from_slice(&(key.len() as u64).to_le_bytes());
                out.extend_from_slice(key.as_bytes());
                // Type ID
                out.extend_from_slice(&type_id.to_le_bytes());
                // Value
                out.extend_from_slice(encoded);
            }

            // Tensor infos — compute offsets after we know the header size
            // First pass: write tensor info headers, use placeholder offsets
            let mut tensor_data_parts: Vec<&[u8]> = Vec::new();
            let mut offsets: Vec<u64> = Vec::new();
            let mut current_offset: u64 = 0;

            for (name, shape, type_code, data) in &self.tensor_infos {
                // Tensor name
                out.extend_from_slice(&(name.len() as u64).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                // n_dims
                out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
                // Shape (already in GGUF order)
                for &dim in shape {
                    out.extend_from_slice(&dim.to_le_bytes());
                }
                // Type
                out.extend_from_slice(&type_code.to_le_bytes());
                // Offset (relative to tensor data start)
                offsets.push(current_offset);
                out.extend_from_slice(&current_offset.to_le_bytes());

                // Align next tensor
                current_offset += data.len() as u64;
                let aligned = align_offset(current_offset, alignment);
                current_offset = aligned;

                tensor_data_parts.push(data.as_slice());
            }

            // Pad to alignment for tensor data start
            let header_end = out.len();
            let data_start = align_offset(header_end as u64, alignment) as usize;
            out.resize(data_start, 0);

            // Write tensor data with alignment padding
            for data in tensor_data_parts.iter() {
                // Should already be at the right position for tensor i
                out.extend_from_slice(data);
                // Pad to alignment
                let current = out.len() - data_start;
                let next_aligned = align_offset(current as u64, alignment) as usize;
                out.resize(data_start + next_aligned, 0);
            }

            out
        }
    }

    // -- f16 conversion tests --

    #[test]
    fn f16_normal_values() {
        // f16 1.0 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < f32::EPSILON);
        // f16 -1.0 = 0xBC00
        assert!((f16_to_f32(0xBC00) - -1.0).abs() < f32::EPSILON);
        // f16 0.5 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // Negative zero
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn f16_infinity_and_nan() {
        let inf = f16_to_f32(0x7C00);
        assert!(inf.is_infinite() && inf > 0.0);

        let neg_inf = f16_to_f32(0xFC00);
        assert!(neg_inf.is_infinite() && neg_inf < 0.0);

        let nan = f16_to_f32(0x7C01);
        assert!(nan.is_nan());
    }

    #[test]
    fn f16_subnormals() {
        // Smallest positive subnormal: 0x0001 = 2^(-14) * (1/1024)
        let val = f16_to_f32(0x0001);
        let expected = 5.960464e-8; // 2^(-14) / 1024
        assert!((val - expected).abs() < 1e-12);
        assert!(val > 0.0);
    }

    #[test]
    fn bf16_conversion() {
        // BF16 1.0 = 0x3F80 (top 16 bits of f32 1.0 = 0x3F800000)
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < f32::EPSILON);
        assert!((bf16_to_f32(0xBF80) - -1.0).abs() < f32::EPSILON);
    }

    // -- TQ2_0 tests --

    #[test]
    fn tq2_remap_correctness() {
        assert_eq!(Ternary::from_bits(TQ2_REMAP[0]), Ternary::Neg);
        assert_eq!(Ternary::from_bits(TQ2_REMAP[1]), Ternary::Zero);
        assert_eq!(Ternary::from_bits(TQ2_REMAP[2]), Ternary::Pos);
        assert_eq!(Ternary::from_bits(TQ2_REMAP[3]), Ternary::Zero); // unused
    }

    #[test]
    fn tq2_unpack_basic() {
        // Build a TQ2_0 block: 256 values, all -1 (GGUF code 0)
        let mut block = vec![0u8; TQ2_BLOCK_BYTES];
        // All data bytes are 0 → all values are GGUF 0 → our Neg
        // Set scale to f16 1.0
        let scale_bits: u16 = 0x3C00;
        block[64] = scale_bits.to_le_bytes()[0];
        block[65] = scale_bits.to_le_bytes()[1];

        let (values, scale) = unpack_tq2_0_block(&block);
        assert!((scale - 1.0).abs() < f32::EPSILON);
        for v in &values {
            assert_eq!(*v, Ternary::Neg);
        }
    }

    #[test]
    fn tq2_unpack_mixed() {
        let mut block = vec![0u8; TQ2_BLOCK_BYTES];
        // First byte: GGUF values [0,1,2,1] = [Neg,Zero,Pos,Zero]
        // Encoding: 0|(1<<2)|(2<<4)|(1<<6) = 0 + 4 + 32 + 64 = 100 = 0x64
        block[0] = 0x64;
        // Scale = f16 0.5 = 0x3800
        block[64] = 0x00;
        block[65] = 0x38;

        let (values, scale) = unpack_tq2_0_block(&block);
        assert!((scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(values[0], Ternary::Neg);
        assert_eq!(values[1], Ternary::Zero);
        assert_eq!(values[2], Ternary::Pos);
        assert_eq!(values[3], Ternary::Zero);
    }

    // -- TQ1_0 / base-3 tests --

    #[test]
    fn decode_base3_all_neg() {
        // All -1: each rem = 0, so byte = 0*3^4 + 0*3^3 + 0*3^2 + 0*3 + 0 = 0
        let result = decode_base3(0);
        for v in &result {
            assert_eq!(*v, Ternary::Neg);
        }
    }

    #[test]
    fn decode_base3_all_zero() {
        // All 0 (ternary code 1): 1 + 3 + 9 + 27 + 81 = 121
        let result = decode_base3(121);
        for v in &result {
            assert_eq!(*v, Ternary::Zero);
        }
    }

    #[test]
    fn decode_base3_all_pos() {
        // All +1 (ternary code 2): 2 + 6 + 18 + 54 + 162 = 242
        let result = decode_base3(242);
        for v in &result {
            assert_eq!(*v, Ternary::Pos);
        }
    }

    #[test]
    fn decode_base3_mixed() {
        // [Neg, Zero, Pos, Neg, Zero] = [0, 1, 2, 0, 1]
        // byte = 0 + 1*3 + 2*9 + 0*27 + 1*81 = 0 + 3 + 18 + 0 + 81 = 102
        let result = decode_base3(102);
        assert_eq!(result[0], Ternary::Neg);
        assert_eq!(result[1], Ternary::Zero);
        assert_eq!(result[2], Ternary::Pos);
        assert_eq!(result[3], Ternary::Neg);
        assert_eq!(result[4], Ternary::Zero);
    }

    #[test]
    fn tq1_unpack_block() {
        // Build a TQ1_0 block with known values
        let mut block = vec![0u8; TQ1_BLOCK_BYTES];

        // Byte 0: all Neg (base3 = 0)
        block[0] = 0;
        // Byte 1: all Zero (base3 = 121)
        block[1] = 121;
        // Bytes 2..48: all Pos (base3 = 242)
        for i in 2..48 {
            block[i] = 242;
        }
        // Tail (bytes 48..52): all GGUF 2 = Pos
        // 2|(2<<2)|(2<<4)|(2<<6) = 2+8+32+128 = 0xAA
        for i in 48..52 {
            block[i] = 0xAA;
        }
        // Scale: f16 1.0
        block[52] = 0x00;
        block[53] = 0x3C;

        let (values, scale) = unpack_tq1_0_block(&block);
        assert!((scale - 1.0).abs() < f32::EPSILON);

        // First 5 values: all Neg
        for i in 0..5 {
            assert_eq!(values[i], Ternary::Neg, "index {i}");
        }
        // Next 5: all Zero
        for i in 5..10 {
            assert_eq!(values[i], Ternary::Zero, "index {i}");
        }
        // 10..240: all Pos
        for i in 10..240 {
            assert_eq!(values[i], Ternary::Pos, "index {i}");
        }
        // 240..256: all Pos (from tail)
        for i in 240..256 {
            assert_eq!(values[i], Ternary::Pos, "index {i}");
        }
    }

    // -- Header parsing tests --

    #[test]
    fn parse_valid_header() {
        let mut b = GgufBuilder::new();
        b.add_metadata_string("general.name", "test-model");
        b.add_metadata_u32("general.alignment", DEFAULT_ALIGNMENT as u32);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        assert_eq!(gguf.metadata.len(), 2);
        assert_eq!(
            gguf.get_metadata("general.name").unwrap().as_str().unwrap(),
            "test-model"
        );
        assert_eq!(gguf.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = vec![0u8; 24];
        // Wrong magic
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data[4..8].copy_from_slice(&GGUF_VERSION.to_le_bytes());

        let err = GgufFile::open_reader(Cursor::new(data)).unwrap_err();
        assert!(matches!(err, GgufError::BadMagic(0xDEADBEEF)));
    }

    #[test]
    fn bad_version_rejected() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        data[8..16].copy_from_slice(&0u64.to_le_bytes()); // tensor count
        data[16..24].copy_from_slice(&0u64.to_le_bytes()); // metadata count

        let err = GgufFile::open_reader(Cursor::new(data)).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99)));
    }

    // -- Metadata roundtrip tests --

    #[test]
    fn metadata_all_scalar_types() {
        let mut b = GgufBuilder::new();
        b.add_metadata_u32("test.u32", 42);
        b.add_metadata_f32("test.f32", 3.14);
        b.add_metadata_string("test.str", "hello");
        b.add_metadata_bool("test.bool", true);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        assert_eq!(gguf.get_metadata("test.u32").unwrap().as_u32().unwrap(), 42);
        assert!((gguf.get_metadata("test.f32").unwrap().as_f32().unwrap() - 3.14).abs() < 0.001);
        assert_eq!(
            gguf.get_metadata("test.str").unwrap().as_str().unwrap(),
            "hello"
        );
        assert_eq!(
            gguf.get_metadata("test.bool").unwrap().as_bool().unwrap(),
            true
        );
    }

    #[test]
    fn metadata_array_roundtrip() {
        let mut b = GgufBuilder::new();
        b.add_metadata_array_u32("test.arr", &[10, 20, 30]);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        let arr = gguf
            .get_metadata("test.arr")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_u32().unwrap(), 10);
        assert_eq!(arr[1].as_u32().unwrap(), 20);
        assert_eq!(arr[2].as_u32().unwrap(), 30);
    }

    // -- Tensor info parsing --

    #[test]
    fn tensor_info_shape_reversed() {
        let mut b = GgufBuilder::new();
        // Add a small F32 tensor: shape (2, 4) → stored as [4, 2] in GGUF
        let tensor_data = vec![0u8; 2 * 4 * 4]; // 8 f32 values
        b.add_tensor("test.weight", &[2, 4], GgmlType::F32 as u32, tensor_data);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        let info = gguf.tensor_info("test.weight").unwrap();
        // Shape should be reversed back to outermost-first
        assert_eq!(info.shape, vec![2, 4]);
        assert_eq!(info.n_elements, 8);
        assert_eq!(info.ggml_type, GgmlType::F32);
    }

    #[test]
    fn tensor_type_parsing() {
        assert_eq!(GgmlType::try_from(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::try_from(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::try_from(30).unwrap(), GgmlType::BF16);
        assert_eq!(GgmlType::try_from(34).unwrap(), GgmlType::TQ1_0);
        assert_eq!(GgmlType::try_from(35).unwrap(), GgmlType::TQ2_0);
        assert!(GgmlType::try_from(99).is_err());
    }

    // -- ModelConfig extraction --

    #[test]
    fn model_config_extraction() {
        let mut b = GgufBuilder::new();
        b.add_metadata_u32("llama.vocab_size", 32000);
        b.add_metadata_u32("llama.embedding_length", 2048);
        b.add_metadata_u32("llama.block_count", 22);
        b.add_metadata_u32("llama.attention.head_count", 32);
        b.add_metadata_u32("llama.attention.head_count_kv", 4);
        b.add_metadata_u32("llama.context_length", 4096);
        b.add_metadata_u32("llama.feed_forward_length", 5632);
        b.add_metadata_f32("llama.rope.freq_base", 10000.0);
        b.add_metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        let config = gguf.model_config().unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.embedding_dim, 2048);
        assert_eq!(config.n_layers, 22);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 4);
        assert_eq!(config.context_length, 4096);
        assert_eq!(config.intermediate_size, 5632);
        assert!((config.rope_theta - 10000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn model_config_missing_key() {
        let b = GgufBuilder::new();
        let data = b.build();
        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        let err = gguf.model_config().unwrap_err();
        assert!(matches!(err, GgufError::MissingMetadata(_)));
    }

    // -- End-to-end: synthetic GGUF with real tensor data --

    fn make_f16_bytes(val: f32) -> [u8; 2] {
        // Simple f32→f16 for test values (only handles normal values)
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3FF;
        let f16 = if exp <= 0 {
            (sign << 15) as u16
        } else if exp >= 31 {
            ((sign << 15) | (0x1F << 10)) as u16
        } else {
            ((sign << 15) | ((exp as u32) << 10) | mant) as u16
        };
        f16.to_le_bytes()
    }

    #[test]
    fn end_to_end_f32_tensor() {
        let mut b = GgufBuilder::new();
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor_data = Vec::new();
        for &v in &values {
            tensor_data.extend_from_slice(&v.to_le_bytes());
        }
        b.add_tensor("embed.weight", &[2, 2], GgmlType::F32 as u32, tensor_data);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data.clone())).unwrap();
        let ft = gguf
            .load_float_from_reader("embed.weight", Cursor::new(data))
            .unwrap();
        assert_eq!(ft.shape(), &[2, 2]);
        assert_eq!(ft.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn end_to_end_f16_tensor() {
        let mut b = GgufBuilder::new();
        let mut tensor_data = Vec::new();
        // f16 values: 1.0, -1.0
        tensor_data.extend_from_slice(&make_f16_bytes(1.0));
        tensor_data.extend_from_slice(&make_f16_bytes(-1.0));
        b.add_tensor("norm.weight", &[2], GgmlType::F16 as u32, tensor_data);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data.clone())).unwrap();
        let ft = gguf
            .load_float_from_reader("norm.weight", Cursor::new(data))
            .unwrap();
        assert_eq!(ft.shape(), &[2]);
        assert!((ft.data()[0] - 1.0).abs() < f32::EPSILON);
        assert!((ft.data()[1] - -1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn end_to_end_tq2_tensor() {
        let mut b = GgufBuilder::new();

        // Build one TQ2_0 block: 256 elements
        let mut block = vec![0u8; TQ2_BLOCK_BYTES];
        // First 64 bytes: all GGUF code 2 (= +1) → 0xAA per byte
        // GGUF 2 in each 2-bit slot: 2|(2<<2)|(2<<4)|(2<<6) = 0xAA
        for i in 0..64 {
            block[i] = 0xAA;
        }
        // Scale: f16 0.5
        block[64] = 0x00;
        block[65] = 0x38;

        b.add_tensor(
            "blk.0.attn_q.weight",
            &[16, 16],
            GgmlType::TQ2_0 as u32,
            block,
        );
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data.clone())).unwrap();
        let (tensor, scale) = gguf
            .load_ternary_from_reader("blk.0.attn_q.weight", Cursor::new(data))
            .unwrap();

        assert_eq!(tensor.rows(), 16);
        assert_eq!(tensor.cols(), 16);
        assert!((scale - 0.5).abs() < f32::EPSILON);

        // All values should be Pos
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(tensor.get(r, c), Ternary::Pos, "({r},{c})");
            }
        }
    }

    #[test]
    fn end_to_end_tq1_tensor() {
        let mut b = GgufBuilder::new();

        // Build one TQ1_0 block: all Zero (base3 = 121 for each byte)
        let mut block = vec![0u8; TQ1_BLOCK_BYTES];
        for i in 0..48 {
            block[i] = 121; // all Zero in base-3
        }
        // Tail: GGUF code 1 = Zero → 1|(1<<2)|(1<<4)|(1<<6) = 0x55
        for i in 48..52 {
            block[i] = 0x55;
        }
        // Scale: f16 1.0
        block[52] = 0x00;
        block[53] = 0x3C;

        b.add_tensor(
            "blk.0.ffn.weight",
            &[16, 16],
            GgmlType::TQ1_0 as u32,
            block,
        );
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data.clone())).unwrap();
        let (tensor, scale) = gguf
            .load_ternary_from_reader("blk.0.ffn.weight", Cursor::new(data))
            .unwrap();

        assert_eq!(tensor.rows(), 16);
        assert_eq!(tensor.cols(), 16);
        assert!((scale - 1.0).abs() < f32::EPSILON);

        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(tensor.get(r, c), Ternary::Zero, "({r},{c})");
            }
        }
    }

    #[test]
    fn alignment_padding() {
        // Verify that tensor data starts at an aligned offset
        let mut b = GgufBuilder::new();
        b.add_metadata_string("general.name", "x");
        let tensor_data = vec![0u8; 4 * 4]; // 4 f32 values
        b.add_tensor("t", &[4], GgmlType::F32 as u32, tensor_data);
        let data = b.build();

        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        assert_eq!(gguf.tensor_data_offset % DEFAULT_ALIGNMENT as u64, 0);
    }

    #[test]
    fn end_to_end_mixed_tensors() {
        let mut b = GgufBuilder::new();

        // F32 embedding
        let embed_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b.add_tensor("embed", &[2, 2], GgmlType::F32 as u32, embed_data);

        // TQ2_0 weight
        let mut tq2_block = vec![0u8; TQ2_BLOCK_BYTES];
        for i in 0..64 {
            tq2_block[i] = 0xAA; // all Pos
        }
        tq2_block[64] = 0x00;
        tq2_block[65] = 0x3C; // scale = 1.0
        b.add_tensor("weight", &[16, 16], GgmlType::TQ2_0 as u32, tq2_block);

        let data = b.build();
        let gguf = GgufFile::open_reader(Cursor::new(data.clone())).unwrap();

        // Load both
        let ft = gguf
            .load_float_from_reader("embed", Cursor::new(data.clone()))
            .unwrap();
        assert_eq!(ft.data(), &[1.0, 2.0, 3.0, 4.0]);

        let (tt, scale) = gguf
            .load_ternary_from_reader("weight", Cursor::new(data))
            .unwrap();
        assert_eq!(tt.rows(), 16);
        assert!((scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_alignment_when_missing() {
        let b = GgufBuilder::new();
        let data = b.build();
        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        assert_eq!(gguf.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn debug_format() {
        let b = GgufBuilder::new();
        let data = b.build();
        let gguf = GgufFile::open_reader(Cursor::new(data)).unwrap();
        let debug = format!("{:?}", gguf);
        assert!(debug.contains("GgufFile"));
        assert!(debug.contains("0 metadata"));
        assert!(debug.contains("0 tensors"));
    }
}
