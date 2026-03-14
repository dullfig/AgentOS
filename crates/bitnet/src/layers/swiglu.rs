//! SwiGLU Feed-Forward Network — the MLP block in LLaMA transformers.
//!
//! SwiGLU splits the traditional FFN into a gated architecture:
//!
//!   FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
//!
//! Where:
//! - `gate_proj`: embed_dim → intermediate_size (ternary BitLinear)
//! - `up_proj`:   embed_dim → intermediate_size (ternary BitLinear)
//! - `down_proj`: intermediate_size → embed_dim (ternary BitLinear)
//! - SiLU(x) = x · σ(x) = x / (1 + exp(-x))
//! - ⊙ = element-wise multiplication

use crate::layers::bitlinear::BitLinear;
use crate::layers::rmsnorm::RmsNorm;

/// Gated feed-forward network with ternary projections.
///
/// Supports both SwiGLU (SiLU activation, standard LLaMA) and
/// BitNet-style (squared ReLU activation).
pub struct SwiGLU {
    /// Gate projection (embed_dim → intermediate_size).
    gate_proj: BitLinear,
    /// Up projection (embed_dim → intermediate_size).
    up_proj: BitLinear,
    /// Down projection (intermediate_size → embed_dim).
    down_proj: BitLinear,
    /// Optional sub-normalization applied to intermediate (before down projection).
    /// BitNet b1.58 uses this — shape is [intermediate_size].
    sub_norm: Option<RmsNorm>,
    /// Activation function for the gate path.
    activation: GateActivation,
}

impl SwiGLU {
    /// Create a gated FFN with SiLU activation (standard SwiGLU).
    pub fn new(gate_proj: BitLinear, up_proj: BitLinear, down_proj: BitLinear) -> Self {
        Self::with_activation(gate_proj, up_proj, down_proj, GateActivation::SiLU)
    }

    /// Create a gated FFN with a specific activation function.
    pub fn with_activation(
        gate_proj: BitLinear,
        up_proj: BitLinear,
        down_proj: BitLinear,
        activation: GateActivation,
    ) -> Self {
        Self::check_dims(&gate_proj, &up_proj, &down_proj);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            sub_norm: None,
            activation,
        }
    }

    /// Create a gated FFN with sub-normalization before the down projection.
    pub fn with_sub_norm(
        gate_proj: BitLinear,
        up_proj: BitLinear,
        down_proj: BitLinear,
        sub_norm: RmsNorm,
    ) -> Self {
        Self::with_sub_norm_and_activation(gate_proj, up_proj, down_proj, sub_norm, GateActivation::SiLU)
    }

    /// Create a gated FFN with sub-norm and a specific activation function.
    pub fn with_sub_norm_and_activation(
        gate_proj: BitLinear,
        up_proj: BitLinear,
        down_proj: BitLinear,
        sub_norm: RmsNorm,
        activation: GateActivation,
    ) -> Self {
        Self::check_dims(&gate_proj, &up_proj, &down_proj);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            sub_norm: Some(sub_norm),
            activation,
        }
    }

    fn check_dims(gate_proj: &BitLinear, up_proj: &BitLinear, down_proj: &BitLinear) {
        assert_eq!(
            gate_proj.in_features(),
            up_proj.in_features(),
            "gate and up projections must have same input dim"
        );
        assert_eq!(
            gate_proj.out_features(),
            up_proj.out_features(),
            "gate and up projections must have same output dim"
        );
        assert_eq!(
            gate_proj.out_features(),
            down_proj.in_features(),
            "gate output must match down projection input"
        );
    }

    /// Input dimension (embed_dim).
    pub fn in_features(&self) -> usize {
        self.gate_proj.in_features()
    }

    /// Intermediate dimension.
    pub fn intermediate_size(&self) -> usize {
        self.gate_proj.out_features()
    }

    /// Output dimension (embed_dim).
    pub fn out_features(&self) -> usize {
        self.down_proj.out_features()
    }

    /// Forward pass for a single vector.
    ///
    /// Input: f32 slice of length `embed_dim`.
    /// Output: f32 vec of length `embed_dim`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features(), "input dimension mismatch");

        // Parallel projections
        let gate = self.gate_proj.forward(input);
        let up = self.up_proj.forward(input);

        // activation(gate) ⊙ up
        let act = self.activation;
        let mut intermediate: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| apply_activation(g, act) * u)
            .collect();

        // Optional sub-normalization before down projection (BitNet b1.58)
        if let Some(ref norm) = self.sub_norm {
            intermediate = norm.forward(&intermediate);
        }

        // Down projection
        self.down_proj.forward(&intermediate)
    }

    /// Forward pass over a sequence of tokens.
    ///
    /// `input`: flat f32 slice of shape `[seq_len, embed_dim]`.
    /// Returns: flat f32 vec of shape `[seq_len, embed_dim]`.
    pub fn forward_sequence(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let embed_dim = self.in_features();
        assert_eq!(
            input.len(),
            seq_len * embed_dim,
            "input shape mismatch"
        );

        let mut output = Vec::with_capacity(seq_len * self.out_features());
        for t in 0..seq_len {
            let start = t * embed_dim;
            let token = &input[start..start + embed_dim];
            output.extend_from_slice(&self.forward(token));
        }
        output
    }
}

impl std::fmt::Debug for SwiGLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let act_name = match self.activation {
            GateActivation::SiLU => "SiLU",
            GateActivation::ReLU2 => "ReLU²",
        };
        write!(
            f,
            "SwiGLU({}→{}→{}, {})",
            self.in_features(),
            self.intermediate_size(),
            self.out_features(),
            act_name,
        )
    }
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Which activation function to use in the gated FFN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateActivation {
    /// SiLU (Sigmoid Linear Unit): x · σ(x) = x / (1 + exp(-x)).
    /// Used in standard LLaMA SwiGLU.
    SiLU,
    /// Squared ReLU: max(0, x)². Used in BitNet b1.58.
    ReLU2,
}

/// SiLU (Sigmoid Linear Unit): x · σ(x) = x / (1 + exp(-x)).
///
/// Also known as the Swish activation function.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Squared ReLU: max(0, x)².
///
/// Used in BitNet b1.58 models instead of SiLU.
#[inline]
fn relu_squared(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

/// Apply the selected activation function.
#[inline]
fn apply_activation(x: f32, act: GateActivation) -> f32 {
    match act {
        GateActivation::SiLU => silu(x),
        GateActivation::ReLU2 => relu_squared(x),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

    fn make_bitlinear(weights: &[i8], rows: usize, cols: usize, scale: f32) -> BitLinear {
        let ternary: Vec<Ternary> = weights
            .iter()
            .map(|&v| match v {
                -1 => Ternary::Neg,
                0 => Ternary::Zero,
                1 => Ternary::Pos,
                _ => panic!("not ternary"),
            })
            .collect();
        BitLinear::new(TernaryTensor::pack(&ternary, rows, cols), scale)
    }

    fn make_identity_proj(out_dim: usize, in_dim: usize) -> BitLinear {
        let mut weights = vec![0i8; out_dim * in_dim];
        for i in 0..out_dim.min(in_dim) {
            weights[i * in_dim + i] = 1;
        }
        make_bitlinear(&weights, out_dim, in_dim, 1.0)
    }

    fn make_test_swiglu(embed_dim: usize, intermediate: usize) -> SwiGLU {
        let gate = make_identity_proj(intermediate, embed_dim);
        let up = make_identity_proj(intermediate, embed_dim);
        let down = make_identity_proj(embed_dim, intermediate);
        SwiGLU::new(gate, up, down)
    }

    // -- SiLU tests --

    #[test]
    fn silu_zero() {
        // silu(0) = 0 / (1 + 1) = 0
        assert!((silu(0.0)).abs() < 1e-7);
    }

    #[test]
    fn silu_positive() {
        // silu(x) > 0 for x > 0
        assert!(silu(1.0) > 0.0);
        assert!(silu(5.0) > 0.0);
    }

    #[test]
    fn silu_negative() {
        // silu(x) < 0 for x < 0 (slight negative dip)
        assert!(silu(-1.0) < 0.0);
    }

    #[test]
    fn silu_large_positive() {
        // silu(x) ≈ x for large positive x (σ(x) → 1)
        let x = 10.0;
        assert!((silu(x) - x).abs() < 0.01);
    }

    #[test]
    fn silu_large_negative() {
        // silu(x) ≈ 0 for large negative x (σ(x) → 0)
        assert!(silu(-10.0).abs() < 0.01);
    }

    #[test]
    fn silu_known_value() {
        // silu(1) = 1 / (1 + exp(-1)) = 1 / (1 + 0.3679) ≈ 0.7311
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((silu(1.0) - expected).abs() < 1e-6);
    }

    // -- Construction tests --

    #[test]
    fn construction() {
        let ffn = make_test_swiglu(8, 16);
        assert_eq!(ffn.in_features(), 8);
        assert_eq!(ffn.intermediate_size(), 16);
        assert_eq!(ffn.out_features(), 8);
    }

    #[test]
    #[should_panic(expected = "same input dim")]
    fn mismatched_gate_up_input_panics() {
        let gate = make_identity_proj(8, 4);
        let up = make_identity_proj(8, 6); // different input dim
        let down = make_identity_proj(4, 8);
        SwiGLU::new(gate, up, down);
    }

    #[test]
    #[should_panic(expected = "same output dim")]
    fn mismatched_gate_up_output_panics() {
        let gate = make_identity_proj(8, 4);
        let up = make_identity_proj(10, 4); // different output dim
        let down = make_identity_proj(4, 8);
        SwiGLU::new(gate, up, down);
    }

    // -- Forward tests --

    #[test]
    fn forward_output_shape() {
        let ffn = make_test_swiglu(8, 16);
        let input = vec![1.0f32; 8];
        let output = ffn.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn forward_finite() {
        let ffn = make_test_swiglu(8, 16);
        let input = vec![0.5f32; 8];
        let output = ffn.forward(&input);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_zero_input() {
        let ffn = make_test_swiglu(4, 8);
        let input = vec![0.0f32; 4];
        let output = ffn.forward(&input);
        // silu(0) = 0, so gate output is all zero, so intermediate is all zero
        for &v in &output {
            assert!(
                v.abs() < 0.1,
                "zero input should produce near-zero output, got {v}"
            );
        }
    }

    #[test]
    fn forward_different_inputs() {
        let ffn = make_test_swiglu(8, 16);
        let input_a = vec![1.0f32; 8];
        let input_b = vec![-1.0f32; 8];
        let out_a = ffn.forward(&input_a);
        let out_b = ffn.forward(&input_b);

        assert!(
            out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "different inputs should produce different outputs"
        );
    }

    #[test]
    fn forward_sequence_shape() {
        let ffn = make_test_swiglu(8, 16);
        let seq_len = 3;
        let input = vec![0.5f32; seq_len * 8];
        let output = ffn.forward_sequence(&input, seq_len);
        assert_eq!(output.len(), seq_len * 8);
    }

    #[test]
    fn forward_sequence_matches_individual() {
        let ffn = make_test_swiglu(4, 8);
        let tok0 = vec![1.0f32, 0.5, -0.3, 0.8];
        let tok1 = vec![-0.2f32, 0.7, 0.1, -0.5];

        let out0 = ffn.forward(&tok0);
        let out1 = ffn.forward(&tok1);

        let mut seq_input = tok0.clone();
        seq_input.extend_from_slice(&tok1);
        let seq_output = ffn.forward_sequence(&seq_input, 2);

        for (a, b) in out0.iter().chain(out1.iter()).zip(seq_output.iter()) {
            assert!((a - b).abs() < 1e-7, "sequence should match individual: {a} vs {b}");
        }
    }

    #[test]
    fn debug_format() {
        let ffn = make_test_swiglu(8, 16);
        let debug = format!("{:?}", ffn);
        assert!(debug.contains("SwiGLU"));
        assert!(debug.contains("8→16→8"));
    }

    // -- ReLU² tests --

    #[test]
    fn relu_squared_zero() {
        assert_eq!(relu_squared(0.0), 0.0);
    }

    #[test]
    fn relu_squared_positive() {
        assert!((relu_squared(2.0) - 4.0).abs() < 1e-7);
        assert!((relu_squared(3.0) - 9.0).abs() < 1e-7);
    }

    #[test]
    fn relu_squared_negative() {
        assert_eq!(relu_squared(-1.0), 0.0);
        assert_eq!(relu_squared(-100.0), 0.0);
    }

    #[test]
    fn relu2_activation_differs_from_silu() {
        // For negative inputs, ReLU² = 0 but SiLU < 0
        assert!(silu(-1.0) < 0.0);
        assert_eq!(relu_squared(-1.0), 0.0);

        // For positive inputs, ReLU²(x) = x² > SiLU(x) ≈ x*sigmoid(x) for x>0
        assert!(relu_squared(2.0) > silu(2.0));
    }

    #[test]
    fn ffn_relu2_output_shape() {
        let gate = make_identity_proj(16, 8);
        let up = make_identity_proj(16, 8);
        let down = make_identity_proj(8, 16);
        let ffn = SwiGLU::with_activation(gate, up, down, GateActivation::ReLU2);

        let input = vec![1.0f32; 8];
        let output = ffn.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn ffn_relu2_zero_for_negative_input() {
        // With identity-ish weights and all-negative input,
        // ReLU²(gate) = 0, so intermediate = 0, so output ≈ 0
        let gate = make_identity_proj(8, 4);
        let up = make_identity_proj(8, 4);
        let down = make_identity_proj(4, 8);
        let ffn = SwiGLU::with_activation(gate, up, down, GateActivation::ReLU2);

        let input = vec![-1.0f32; 4];
        let output = ffn.forward(&input);
        for &v in &output {
            assert!(v.abs() < 0.1, "relu2 should zero out negative gate: got {v}");
        }
    }
}
