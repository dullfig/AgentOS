//! Neural network layers for 1.58-bit transformer inference.
//!
//! Each layer operates on the tensor types from `tensor.rs` using the
//! kernels from `ops/`. The layers compose to build a full transformer
//! forward pass in `transformer.rs`.

pub mod bitlinear;
pub mod rmsnorm;
