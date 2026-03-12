//! Computational kernels for 1.58-bit inference.
//!
//! Two kernel strategies:
//!
//! - **I2S** (`matmul`): Unpack 2-bit weights, conditional add/sub/skip.
//!   Simple, vectorization-friendly, good baseline.
//!
//! - **LUT** (`lut`): Group 2 weights → 4-bit index into 9-entry precomputed
//!   table. Replaces all arithmetic with table lookups.
//!
//! - **Quantize** (`quantize`): Absmax 8-bit activation quantization.

pub mod matmul;
pub mod lut;
pub mod quantize;
