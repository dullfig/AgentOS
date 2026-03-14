//! # agentos-bitnet
//!
//! Pure-Rust 1.58-bit inference engine for ternary LLMs.
//!
//! Ternary weights ∈ {-1, 0, +1} eliminate floating-point multiplication entirely.
//! The matmul hot path reduces to conditional addition: add, subtract, or skip.
//!
//! ## Kernel strategies
//!
//! - **I2S**: 2-bit packed weights, scalar add/sub/skip loop with SIMD.
//! - **LUT (TL1)**: Group 2 weights → 4-bit index into 9-entry precomputed table.
//!   Replaces arithmetic with table lookup.
//!
//! ## Design constraints
//!
//! - Zero `unsafe` (SIMD via safe abstractions, with unsafe gated behind cfg)
//! - No external ML framework dependencies
//! - GGUF model loading (same format as BitNet.cpp / llama.cpp)

pub mod tensor;
pub mod ops;
pub mod compute;
pub mod layers;
pub mod gguf;
pub mod tokenizer;
pub mod loader;

pub use tensor::{TernaryTensor, ActivationTensor, Ternary};
pub use gguf::{GgufFile, GgufError, GgmlType, TensorInfo, ModelConfig, MetadataValue};
pub use tokenizer::Tokenizer;
pub use loader::{load_model, LoadedModel};
