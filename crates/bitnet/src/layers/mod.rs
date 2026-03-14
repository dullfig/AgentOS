//! Neural network layers for 1.58-bit transformer inference.
//!
//! Each layer operates on the tensor types from `tensor.rs` using the
//! kernels from `ops/`. The layers compose to build a full transformer
//! forward pass in `transformer.rs`.

pub mod attention;
pub mod bitlinear;
pub mod kv_cache;
pub mod model;
pub mod rmsnorm;
pub mod rope;
pub mod sampler;
pub mod swiglu;
pub mod transformer;
