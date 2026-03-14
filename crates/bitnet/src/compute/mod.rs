//! Compute backend abstraction for ternary inference.
//!
//! Provides a `ComputeBackend` trait that abstracts over hardware-specific
//! implementations (scalar, AVX2, future GPU). The `detect()` function
//! auto-selects the fastest backend for the current CPU.
//!
//! ## Architecture
//!
//! ```text
//!   ComputeBackend (trait)
//!       ├── ScalarBackend     — portable, no SIMD
//!       ├── Avx2Backend       — x86-64 AVX2 (32-wide i8 ops)
//!       └── WgpuBackend       — GPU via wgpu/Vulkan/DX12/Metal (future)
//! ```

pub mod scalar;
pub mod device;
#[cfg(target_arch = "x86_64")]
pub mod avx2;

use crate::tensor::TernaryTensor;

/// Hardware-independent compute interface for ternary inference kernels.
///
/// Each method corresponds to a hot-path operation in the transformer forward
/// pass. Backends implement whichever operations they can accelerate; the
/// trait provides default implementations that fall back to scalar code.
pub trait ComputeBackend: Send + Sync + std::fmt::Debug {
    /// Human-readable name for logging (e.g., "scalar", "avx2").
    fn name(&self) -> &str;

    /// Ternary matrix-vector product: y = W · x.
    ///
    /// `weights`: packed ternary (out_features × in_features).
    /// `input`: quantized 8-bit activations (in_features), must be in [-127, 127].
    ///          (Absmax quantization guarantees this. The -128 value is excluded
    ///          because SIMD `sign_epi8` can't negate it without overflow.)
    /// Returns: i32 accumulator per output feature.
    fn ternary_matvec(&self, weights: &TernaryTensor, input: &[i8]) -> Vec<i32>;

    /// RMS normalization: x_i * (w_i / rms), where rms = sqrt(mean(x²) + eps).
    fn rmsnorm(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        input.iter().zip(weight).map(|(&x, &w)| x * inv_rms * w).collect()
    }

    /// Softmax over a slice (in-place would be ideal, but returning Vec is fine
    /// for now — the bottleneck is matmul, not softmax).
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Element-wise multiply: out[i] = a[i] * b[i].
    fn elementwise_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(&x, &y)| x * y).collect()
    }
}

/// Capability flags detected at runtime.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features using `std::arch::is_x86_feature_detected` (x86)
    /// or compile-time target detection (ARM).
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: true, // NEON is mandatory on aarch64
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: false,
            }
        }
    }
}

/// Auto-detect the fastest available backend for this CPU.
///
/// Returns an `Arc<dyn ComputeBackend>` ready to be shared across layers.
pub fn detect() -> std::sync::Arc<dyn ComputeBackend> {
    let features = CpuFeatures::detect();
    tracing::info!(?features, "detecting compute backend");

    #[cfg(target_arch = "x86_64")]
    if features.avx2 {
        tracing::info!("using AVX2 compute backend");
        return std::sync::Arc::new(avx2::Avx2Backend);
    }

    tracing::info!("using scalar compute backend");
    std::sync::Arc::new(scalar::ScalarBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_backend() {
        let backend = detect();
        let name = backend.name();
        assert!(!name.is_empty());
        eprintln!("detected backend: {name}");
    }

    #[test]
    fn cpu_features_detect() {
        let f = CpuFeatures::detect();
        eprintln!("CPU features: avx2={}, avx512f={}, neon={}", f.avx2, f.avx512f, f.neon);
    }

    #[test]
    fn default_rmsnorm() {
        let backend = scalar::ScalarBackend;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let result = backend.rmsnorm(&input, &weight, 1e-6);
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-6).sqrt();
        for (i, &v) in result.iter().enumerate() {
            let expected = input[i] / rms;
            assert!((v - expected).abs() < 1e-5, "rmsnorm[{i}]: expected {expected}, got {v}");
        }
    }

    #[test]
    fn default_softmax() {
        let backend = scalar::ScalarBackend;
        let result = backend.softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }
}
