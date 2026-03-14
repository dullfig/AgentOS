//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! Used in LLaMA-style architectures (and BitNet b1.58) instead of LayerNorm.
//! Simpler and faster: no mean subtraction, no bias.
//!
//!   RMSNorm(x) = (x / √(mean(x²) + ε)) * γ
//!
//! Where γ is a learnable per-feature scale parameter.

/// RMSNorm layer with learnable scale weights.
#[derive(Clone)]
pub struct RmsNorm {
    /// Per-feature scale weights γ.
    weight: Vec<f32>,
    /// Small epsilon for numerical stability.
    eps: f32,
}

impl RmsNorm {
    /// Create an RMSNorm layer with the given weights and epsilon.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Hidden dimension.
    pub fn dim(&self) -> usize { self.weight.len() }

    /// Forward pass: normalize and scale.
    ///
    /// Input and output are both f32 vectors of length `dim`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.weight.len(), "dimension mismatch");

        // Compute mean(x²).
        let n = input.len() as f32;
        let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / n;

        // Compute 1/√(mean(x²) + ε).
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();

        // Normalize and scale.
        input.iter()
            .zip(self.weight.iter())
            .map(|(&x, &w)| x * inv_rms * w)
            .collect()
    }

    /// Forward pass, writing into an existing output buffer.
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.weight.len());
        assert_eq!(output.len(), self.weight.len());

        let n = input.len() as f32;
        let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / n;
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();

        for ((x, w), o) in input.iter().zip(self.weight.iter()).zip(output.iter_mut()) {
            *o = x * inv_rms * w;
        }
    }

    /// Access the weight vector.
    pub fn weight(&self) -> &[f32] { &self.weight }
}

impl std::fmt::Debug for RmsNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RmsNorm(dim={}, eps={})", self.weight.len(), self.eps)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_weights_normalize() {
        let norm = RmsNorm::new(vec![1.0; 4], 1e-6);
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let output = norm.forward(&input);

        // RMS of [2,2,2,2] = 2.0, so normalized = [1,1,1,1].
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn scale_weights_applied() {
        let norm = RmsNorm::new(vec![2.0; 4], 1e-6);
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let output = norm.forward(&input);

        // Normalized = 1.0, then scaled by 2.0.
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn mixed_input() {
        let norm = RmsNorm::new(vec![1.0, 1.0], 1e-6);
        let input = vec![3.0, 4.0];
        let output = norm.forward(&input);

        // mean(x²) = (9+16)/2 = 12.5
        // rms = √12.5 ≈ 3.5355
        let rms = 12.5f32.sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-5);
        assert!((output[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn zero_input_with_eps() {
        let norm = RmsNorm::new(vec![1.0; 4], 1e-6);
        let input = vec![0.0; 4];
        let output = norm.forward(&input);

        // All zeros: mean_sq = 0, rms = √(0 + eps), output ≈ 0.
        for &v in &output {
            assert!(v.abs() < 1e-3, "expected ~0, got {v}");
        }
    }

    #[test]
    fn forward_into_matches_forward() {
        let norm = RmsNorm::new(vec![0.5, 1.5, 2.0], 1e-6);
        let input = vec![1.0, -2.0, 3.0];

        let expected = norm.forward(&input);
        let mut actual = vec![0.0f32; 3];
        norm.forward_into(&input, &mut actual);

        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-6);
        }
    }
}
