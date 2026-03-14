//! BitLinear — the ternary linear layer.
//!
//! Replaces `nn.Linear` in a standard transformer. Weights are ternary
//! {-1, 0, +1} packed as 2 bits each. No bias term.
//!
//! Forward pass:
//!   1. Quantize input activations to 8-bit (absmax per-token).
//!   2. Ternary matmul: y_int = W_ternary · x_int8 (integer accumulation).
//!   3. Rescale: y_float = y_int * activation_scale * weight_scale.
//!
//! The weight_scale (γ) is the absmean of the original float weights,
//! stored alongside the packed ternary data.

use std::sync::Arc;

use crate::tensor::{TernaryTensor, FloatTensor};
use crate::ops::matmul::ternary_matvec;
use crate::ops::quantize::quantize_absmax;
use crate::compute::ComputeBackend;

/// A ternary linear layer: y = W · x (no bias).
pub struct BitLinear {
    /// Packed ternary weights (out_features × in_features).
    weights: TernaryTensor,
    /// Absmean scale factor γ from the original float weights.
    weight_scale: f32,
    /// Optional compute backend for accelerated matmul.
    backend: Option<Arc<dyn ComputeBackend>>,
}

impl BitLinear {
    /// Create a BitLinear layer from packed weights and scale.
    ///
    /// Uses the scalar kernel by default. Call `with_backend` to use SIMD.
    pub fn new(weights: TernaryTensor, weight_scale: f32) -> Self {
        Self { weights, weight_scale, backend: None }
    }

    /// Create a BitLinear layer with a specific compute backend.
    pub fn with_backend(weights: TernaryTensor, weight_scale: f32, backend: Arc<dyn ComputeBackend>) -> Self {
        Self { weights, weight_scale, backend: Some(backend) }
    }

    /// Set the compute backend (for post-construction wiring).
    pub fn set_backend(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backend = Some(backend);
    }

    /// Output features (rows).
    pub fn out_features(&self) -> usize { self.weights.rows() }

    /// Input features (columns).
    pub fn in_features(&self) -> usize { self.weights.cols() }

    /// Forward pass for a single vector.
    ///
    /// Input: f32 vector of length `in_features`.
    /// Output: f32 vector of length `out_features`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features(), "input dimension mismatch");

        // Step 1: Quantize activations to 8-bit.
        let (x_quant, act_scale) = quantize_absmax(input);

        // Step 2: Ternary matmul (pure integer accumulation).
        let y_int = match &self.backend {
            Some(backend) => backend.ternary_matvec(&self.weights, &x_quant),
            None => ternary_matvec(&self.weights, &x_quant),
        };

        // Step 3: Rescale to f32.
        let combined_scale = act_scale * self.weight_scale;
        y_int.iter().map(|&acc| acc as f32 * combined_scale).collect()
    }

    /// Forward pass returning a FloatTensor.
    pub fn forward_tensor(&self, input: &FloatTensor) -> FloatTensor {
        assert_eq!(input.shape().len(), 1, "expected 1D input");
        let out = self.forward(input.data());
        FloatTensor::new(out, vec![self.out_features()])
    }

    /// Batch forward pass for multiple vectors.
    ///
    /// Input: f32 slice of shape (batch_size × in_features).
    /// Returns: Vec of f32 output vectors.
    pub fn forward_batch(&self, inputs: &[f32], batch_size: usize) -> Vec<Vec<f32>> {
        let in_features = self.in_features();
        assert_eq!(inputs.len(), batch_size * in_features);

        (0..batch_size)
            .map(|b| {
                let start = b * in_features;
                let end = start + in_features;
                self.forward(&inputs[start..end])
            })
            .collect()
    }

    /// Access the underlying weights (for inspection/testing).
    pub fn weights(&self) -> &TernaryTensor { &self.weights }

    /// The weight scale factor γ.
    pub fn weight_scale(&self) -> f32 { self.weight_scale }
}

impl std::fmt::Debug for BitLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BitLinear({}→{}, γ={:.6})",
            self.in_features(),
            self.out_features(),
            self.weight_scale,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Ternary;

    fn make_bitlinear(weights: &[i8], rows: usize, cols: usize, scale: f32) -> BitLinear {
        let ternary: Vec<Ternary> = weights.iter().map(|&v| match v {
            -1 => Ternary::Neg,
             0 => Ternary::Zero,
             1 => Ternary::Pos,
             _ => panic!("not ternary"),
        }).collect();
        BitLinear::new(TernaryTensor::pack(&ternary, rows, cols), scale)
    }

    #[test]
    fn identity_forward() {
        let layer = make_bitlinear(&[1, 0, 0, 1], 2, 2, 1.0);
        let input = vec![1.0f32, 0.5];
        let output = layer.forward(&input);

        assert!((output[0] - 1.0).abs() < 0.02);
        assert!((output[1] - 0.5).abs() < 0.02);
    }

    #[test]
    fn negation_forward() {
        let layer = make_bitlinear(&[-1, 0, 0, -1], 2, 2, 1.0);
        let input = vec![1.0f32, 0.5];
        let output = layer.forward(&input);

        assert!((output[0] - -1.0).abs() < 0.02);
        assert!((output[1] - -0.5).abs() < 0.02);
    }

    #[test]
    fn scale_factor_applied() {
        let layer = make_bitlinear(&[1, 1, 1, 1], 1, 4, 0.1);
        let input = vec![1.0f32; 4];
        let output = layer.forward(&input);
        assert!((output[0] - 0.4).abs() < 0.05, "got {}", output[0]);
    }

    #[test]
    fn batch_forward() {
        let layer = make_bitlinear(&[1, -1], 1, 2, 1.0);
        let inputs = vec![
            1.0f32, 0.5,   // batch 0
            0.0, 1.0,      // batch 1
        ];
        let outputs = layer.forward_batch(&inputs, 2);
        assert_eq!(outputs.len(), 2);
        assert!((outputs[0][0] - 0.5).abs() < 0.05);
        assert!((outputs[1][0] - -1.0).abs() < 0.05);
    }

    #[test]
    fn debug_format() {
        let layer = make_bitlinear(&[1, 0, 0, 1], 2, 2, 0.5);
        let debug = format!("{:?}", layer);
        assert!(debug.contains("BitLinear"));
        assert!(debug.contains("2→2"));
    }
}
