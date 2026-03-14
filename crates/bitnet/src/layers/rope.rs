//! Rotary Position Embeddings (RoPE) — position encoding for attention.
//!
//! Encodes absolute position by rotating pairs of dimensions in the Q/K
//! vectors. Each pair is rotated by an angle proportional to the sequence
//! position, with frequency determined by the dimension index.
//!
//! θ_i = 1 / (base^(2i / dim))
//!
//! Two layouts are supported:
//!
//! **Interleaved** (llama.cpp default, original RoPE paper):
//!   Pairs: (0,1), (2,3), (4,5), ...
//!   x'_{2i}   = x_{2i} · cos(pos · θ_i) − x_{2i+1} · sin(pos · θ_i)
//!   x'_{2i+1} = x_{2i} · sin(pos · θ_i) + x_{2i+1} · cos(pos · θ_i)
//!
//! **Halved** (GPT-NeoX / HuggingFace Transformers):
//!   Pairs: (0, dim/2), (1, dim/2+1), (2, dim/2+2), ...
//!   x'_i        = x_i · cos(pos · θ_i) − x_{i+half} · sin(pos · θ_i)
//!   x'_{i+half} = x_i · sin(pos · θ_i) + x_{i+half} · cos(pos · θ_i)
//!
//! Applied independently to each attention head's Q and K vectors.

use crate::tensor::FloatTensor;

/// Layout convention for RoPE dimension pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoPELayout {
    /// Interleaved pairs: (0,1), (2,3), ... — llama.cpp convention.
    Interleaved,
    /// Halved pairs: (i, i+dim/2) — HuggingFace/GPT-NeoX convention.
    Halved,
}

/// Rotary Position Embedding layer.
///
/// Precomputes the inverse-frequency table at construction time.
/// The forward pass applies the rotation for a given sequence position.
pub struct RoPE {
    /// Dimension of the vectors being rotated (must be even).
    dim: usize,
    /// Precomputed inverse frequencies: θ_i = 1 / (base^(2i/dim)).
    /// Length = dim / 2.
    inv_freq: Vec<f32>,
    /// Dimension pairing layout.
    layout: RoPELayout,
}

impl RoPE {
    /// Create a RoPE layer with interleaved layout (llama.cpp convention).
    ///
    /// `dim` must be even (Q/K head dimension). `base` is the frequency
    /// base, typically 10000.0 (from `llama.rope.freq_base`).
    pub fn new(dim: usize, base: f32) -> Self {
        Self::with_layout(dim, base, RoPELayout::Interleaved)
    }

    /// Create a RoPE layer with the specified layout.
    pub fn with_layout(dim: usize, base: f32, layout: RoPELayout) -> Self {
        assert!(dim > 0 && dim.is_multiple_of(2), "RoPE dim must be positive and even");
        assert!(base > 0.0, "RoPE base must be positive");

        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        Self { dim, inv_freq, layout }
    }

    /// Dimension this RoPE layer was created for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Layout convention.
    pub fn layout(&self) -> RoPELayout {
        self.layout
    }

    /// Apply RoPE to a single vector at the given sequence position.
    ///
    /// `input` must have length == `dim`. Returns the rotated vector.
    pub fn forward(&self, input: &[f32], seq_pos: usize) -> Vec<f32> {
        assert_eq!(input.len(), self.dim, "input length must equal RoPE dim");
        let mut output = vec![0.0f32; self.dim];
        self.forward_into(input, seq_pos, &mut output);
        output
    }

    /// Apply RoPE in-place into a pre-allocated output buffer.
    pub fn forward_into(&self, input: &[f32], seq_pos: usize, output: &mut [f32]) {
        assert_eq!(input.len(), self.dim);
        assert_eq!(output.len(), self.dim);

        match self.layout {
            RoPELayout::Interleaved => self.forward_interleaved(input, seq_pos, output),
            RoPELayout::Halved => self.forward_halved(input, seq_pos, output),
        }
    }

    /// Interleaved layout: pairs (2i, 2i+1).
    fn forward_interleaved(&self, input: &[f32], seq_pos: usize, output: &mut [f32]) {
        let pos = seq_pos as f32;
        for (i, &freq) in self.inv_freq.iter().enumerate() {
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let x0 = input[2 * i];
            let x1 = input[2 * i + 1];

            output[2 * i] = x0 * cos_a - x1 * sin_a;
            output[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }

    /// Halved layout: pairs (i, i + dim/2).
    fn forward_halved(&self, input: &[f32], seq_pos: usize, output: &mut [f32]) {
        let half = self.dim / 2;
        let pos = seq_pos as f32;

        // Copy input first (in case output aliases input)
        output.copy_from_slice(input);

        for (i, &freq) in self.inv_freq.iter().enumerate() {
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let x0 = input[i];
            let x1 = input[i + half];

            output[i] = x0 * cos_a - x1 * sin_a;
            output[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }

    /// Apply RoPE to a FloatTensor (1D vector at a single position).
    pub fn forward_tensor(&self, input: &FloatTensor, seq_pos: usize) -> FloatTensor {
        assert_eq!(input.shape().len(), 1, "expected 1D input");
        let out = self.forward(input.data(), seq_pos);
        FloatTensor::new(out, vec![self.dim])
    }

    /// Apply RoPE to multiple heads packed in a single vector.
    ///
    /// `input` has shape `n_heads * head_dim` where `head_dim == self.dim`.
    /// Each head's slice is rotated independently at the same position.
    pub fn forward_heads(&self, input: &[f32], n_heads: usize, seq_pos: usize) -> Vec<f32> {
        let total = n_heads * self.dim;
        assert_eq!(input.len(), total, "input length must equal n_heads * dim");

        let mut output = vec![0.0f32; total];
        for h in 0..n_heads {
            let start = h * self.dim;
            let end = start + self.dim;
            self.forward_into(&input[start..end], seq_pos, &mut output[start..end]);
        }
        output
    }

    /// Apply RoPE to a sequence of vectors (one per position).
    ///
    /// `input` is a flat slice of `seq_len * dim` values.
    /// Position `i` gets `start_pos + i` as its sequence position.
    /// Returns the rotated sequence as a flat Vec.
    pub fn forward_sequence(
        &self,
        input: &[f32],
        seq_len: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let total = seq_len * self.dim;
        assert_eq!(input.len(), total, "input length must equal seq_len * dim");

        let mut output = vec![0.0f32; total];
        for t in 0..seq_len {
            let start = t * self.dim;
            let end = start + self.dim;
            self.forward_into(
                &input[start..end],
                start_pos + t,
                &mut output[start..end],
            );
        }
        output
    }

    /// Precomputed inverse frequencies (exposed for debugging/testing).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }
}

impl std::fmt::Debug for RoPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RoPE(dim={}, n_freqs={}, {:?})", self.dim, self.inv_freq.len(), self.layout)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let rope = RoPE::new(64, 10000.0);
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.inv_freq().len(), 32);
    }

    #[test]
    fn inv_freq_values() {
        let rope = RoPE::new(4, 10000.0);
        // θ_0 = 1 / (10000^(0/4)) = 1.0
        // θ_1 = 1 / (10000^(2/4)) = 1/100 = 0.01
        assert!((rope.inv_freq()[0] - 1.0).abs() < 1e-6);
        assert!((rope.inv_freq()[1] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn position_zero_is_identity() {
        // At position 0, all angles are 0, cos=1, sin=0 → no rotation.
        let rope = RoPE::new(4, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = rope.forward(&input, 0);

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-6,
                "pos 0 should be identity, idx {i}: {inp} vs {out}"
            );
        }
    }

    #[test]
    fn rotation_preserves_magnitude() {
        // Rotation should preserve the L2 norm of each pair.
        let rope = RoPE::new(4, 10000.0);
        let input = vec![3.0, 4.0, 1.0, 2.0];

        for pos in [1, 5, 50, 500] {
            let output = rope.forward(&input, pos);

            // Check pair magnitudes
            let mag_in_0 = (input[0] * input[0] + input[1] * input[1]).sqrt();
            let mag_out_0 = (output[0] * output[0] + output[1] * output[1]).sqrt();
            assert!(
                (mag_in_0 - mag_out_0).abs() < 1e-5,
                "pair 0 magnitude changed at pos {pos}"
            );

            let mag_in_1 = (input[2] * input[2] + input[3] * input[3]).sqrt();
            let mag_out_1 = (output[2] * output[2] + output[3] * output[3]).sqrt();
            assert!(
                (mag_in_1 - mag_out_1).abs() < 1e-5,
                "pair 1 magnitude changed at pos {pos}"
            );
        }
    }

    #[test]
    fn known_rotation() {
        // dim=2, base=1.0 → θ_0 = 1.0, so angle at pos p = p radians.
        // At pos = π/2: cos=0, sin=1 → (x,y) → (-y, x)
        let rope = RoPE::new(2, 1.0);
        let input = vec![1.0, 0.0];
        let output = rope.forward(&input, 0); // pos=0, should be identity

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 0.0).abs() < 1e-6);

        // Manually compute at a known angle
        let angle = 1.0f32; // pos=1, freq=1.0
        let expected_x = 1.0 * angle.cos() - 0.0 * angle.sin();
        let expected_y = 1.0 * angle.sin() + 0.0 * angle.cos();
        let output1 = rope.forward(&input, 1);
        assert!((output1[0] - expected_x).abs() < 1e-6);
        assert!((output1[1] - expected_y).abs() < 1e-6);
    }

    #[test]
    fn quarter_turn() {
        // base=1.0, dim=2 → freq=1.0. At pos = π/2: (1,0) → (0,1)
        let rope = RoPE::new(2, 1.0);
        let input = vec![1.0, 0.0];

        // We can't pass a float position, but we can construct the expected
        // result for integer position and verify the rotation formula.
        // pos=1 → angle=1 rad → cos(1)≈0.5403, sin(1)≈0.8415
        let out = rope.forward(&input, 1);
        assert!((out[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((out[1] - 1.0f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn different_positions_differ() {
        let rope = RoPE::new(8, 10000.0);
        let input = vec![1.0; 8];

        let out0 = rope.forward(&input, 0);
        let out1 = rope.forward(&input, 1);
        let out100 = rope.forward(&input, 100);

        // Outputs at different positions must differ
        assert!(
            out0.iter().zip(out1.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "pos 0 and pos 1 should differ"
        );
        assert!(
            out1.iter().zip(out100.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "pos 1 and pos 100 should differ"
        );
    }

    #[test]
    fn forward_into_matches_forward() {
        let rope = RoPE::new(8, 10000.0);
        let input = vec![1.0, -0.5, 0.3, 2.1, -1.0, 0.7, 0.0, 3.0];

        let allocated = rope.forward(&input, 42);
        let mut inplace = vec![0.0f32; 8];
        rope.forward_into(&input, 42, &mut inplace);

        for (a, b) in allocated.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn forward_tensor_works() {
        let rope = RoPE::new(4, 10000.0);
        let input = FloatTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = rope.forward_tensor(&input, 5);
        assert_eq!(output.shape(), &[4]);
        assert_eq!(output.len(), 4);

        // Should match the slice version
        let expected = rope.forward(&[1.0, 2.0, 3.0, 4.0], 5);
        for (a, b) in output.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn forward_heads() {
        let rope = RoPE::new(4, 10000.0);
        // 2 heads, each dim=4
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = rope.forward_heads(&input, 2, 10);

        // Each head rotated independently at same position
        let head0 = rope.forward(&input[0..4], 10);
        let head1 = rope.forward(&input[4..8], 10);

        for (a, b) in output[0..4].iter().zip(head0.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
        for (a, b) in output[4..8].iter().zip(head1.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn forward_sequence() {
        let rope = RoPE::new(4, 10000.0);
        // 3 positions starting at pos 5
        let input = vec![1.0; 12]; // 3 * 4
        let output = rope.forward_sequence(&input, 3, 5);
        assert_eq!(output.len(), 12);

        // Each position should be rotated at 5, 6, 7
        for t in 0..3 {
            let expected = rope.forward(&input[0..4], 5 + t);
            let start = t * 4;
            for (a, b) in output[start..start + 4].iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn relative_position_property() {
        // RoPE's key property: dot product between rotated vectors depends
        // on relative position, not absolute position.
        // q at pos p, k at pos p+d should have the same dot product
        // regardless of p.
        let rope = RoPE::new(8, 10000.0);
        let q = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.7, 1.1, 0.4];
        let k = vec![0.3, -0.2, 0.6, 1.0, -0.5, 0.9, 0.1, -0.8];

        let delta = 7; // relative distance

        // Compute dot product at (pos=10, pos=10+delta)
        let q_rot_a = rope.forward(&q, 10);
        let k_rot_a = rope.forward(&k, 10 + delta);
        let dot_a: f32 = q_rot_a.iter().zip(k_rot_a.iter()).map(|(a, b)| a * b).sum();

        // Compute dot product at (pos=500, pos=500+delta)
        let q_rot_b = rope.forward(&q, 500);
        let k_rot_b = rope.forward(&k, 500 + delta);
        let dot_b: f32 = q_rot_b.iter().zip(k_rot_b.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_a - dot_b).abs() < 1e-3,
            "relative position property violated: {dot_a} vs {dot_b}"
        );
    }

    #[test]
    fn higher_dims_rotate_slower() {
        // Higher dimension pairs should rotate more slowly (lower frequency).
        let rope = RoPE::new(8, 10000.0);
        let input = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let out = rope.forward(&input, 1);

        // Pair 0 rotates at freq θ_0, pair 3 at freq θ_3.
        // θ_0 > θ_1 > θ_2 > θ_3, so pair 0 should deviate more from (1,0).
        let deviation = |pair: usize| -> f32 {
            let x = out[2 * pair];
            let y = out[2 * pair + 1];
            // Distance from (1, 0)
            ((x - 1.0) * (x - 1.0) + y * y).sqrt()
        };

        assert!(
            deviation(0) > deviation(1),
            "pair 0 should rotate more than pair 1"
        );
        assert!(
            deviation(1) > deviation(2),
            "pair 1 should rotate more than pair 2"
        );
        assert!(
            deviation(2) > deviation(3),
            "pair 2 should rotate more than pair 3"
        );
    }

    #[test]
    fn debug_format() {
        let rope = RoPE::new(64, 10000.0);
        let debug = format!("{:?}", rope);
        assert!(debug.contains("RoPE"));
        assert!(debug.contains("dim=64"));
        assert!(debug.contains("n_freqs=32"));
    }

    #[test]
    #[should_panic(expected = "even")]
    fn odd_dim_panics() {
        RoPE::new(3, 10000.0);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn zero_dim_panics() {
        RoPE::new(0, 10000.0);
    }

    // -- Halved layout tests --

    #[test]
    fn halved_position_zero_is_identity() {
        let rope = RoPE::with_layout(4, 10000.0, RoPELayout::Halved);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = rope.forward(&input, 0);
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-6,
                "halved pos 0 should be identity, idx {i}: {inp} vs {out}"
            );
        }
    }

    #[test]
    fn halved_preserves_magnitude() {
        let rope = RoPE::with_layout(4, 10000.0, RoPELayout::Halved);
        let input = vec![3.0, 4.0, 1.0, 2.0];
        for pos in [1, 5, 50, 500] {
            let output = rope.forward(&input, pos);
            // Pair (0,2) and pair (1,3) magnitudes should be preserved
            let mag_in_0 = (input[0] * input[0] + input[2] * input[2]).sqrt();
            let mag_out_0 = (output[0] * output[0] + output[2] * output[2]).sqrt();
            assert!(
                (mag_in_0 - mag_out_0).abs() < 1e-5,
                "halved pair (0,2) magnitude changed at pos {pos}"
            );
            let mag_in_1 = (input[1] * input[1] + input[3] * input[3]).sqrt();
            let mag_out_1 = (output[1] * output[1] + output[3] * output[3]).sqrt();
            assert!(
                (mag_in_1 - mag_out_1).abs() < 1e-5,
                "halved pair (1,3) magnitude changed at pos {pos}"
            );
        }
    }

    #[test]
    fn halved_different_positions_differ() {
        let rope = RoPE::with_layout(8, 10000.0, RoPELayout::Halved);
        let input = vec![1.0; 8];
        let out0 = rope.forward(&input, 0);
        let out1 = rope.forward(&input, 1);
        assert!(
            out0.iter().zip(out1.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "halved: pos 0 and pos 1 should differ"
        );
    }

    #[test]
    fn halved_relative_position_property() {
        let rope = RoPE::with_layout(8, 10000.0, RoPELayout::Halved);
        let q = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.7, 1.1, 0.4];
        let k = vec![0.3, -0.2, 0.6, 1.0, -0.5, 0.9, 0.1, -0.8];
        let delta = 7;

        let q_rot_a = rope.forward(&q, 10);
        let k_rot_a = rope.forward(&k, 10 + delta);
        let dot_a: f32 = q_rot_a.iter().zip(k_rot_a.iter()).map(|(a, b)| a * b).sum();

        let q_rot_b = rope.forward(&q, 500);
        let k_rot_b = rope.forward(&k, 500 + delta);
        let dot_b: f32 = q_rot_b.iter().zip(k_rot_b.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_a - dot_b).abs() < 1e-3,
            "halved: relative position property violated: {dot_a} vs {dot_b}"
        );
    }

    #[test]
    fn halved_vs_interleaved_differ() {
        // The two layouts should produce different results for the same input
        // (unless the input happens to have a special structure).
        let inter = RoPE::with_layout(8, 10000.0, RoPELayout::Interleaved);
        let halved = RoPE::with_layout(8, 10000.0, RoPELayout::Halved);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let out_inter = inter.forward(&input, 5);
        let out_halved = halved.forward(&input, 5);

        assert!(
            out_inter.iter().zip(out_halved.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "interleaved and halved should produce different outputs"
        );
    }

    #[test]
    fn halved_known_rotation() {
        // dim=4, base=1.0, halved layout
        // Pairs: (0,2) with freq θ_0=1.0, (1,3) with freq θ_1=1/1^(2/4)=1.0
        // Wait, base=1.0: θ_0 = 1/1^(0/4) = 1.0, θ_1 = 1/1^(2/4) = 1.0
        // At pos=1: angle=1.0 for both pairs
        // (x0,x2) → (x0*cos(1) - x2*sin(1), x0*sin(1) + x2*cos(1))
        let rope = RoPE::with_layout(4, 1.0, RoPELayout::Halved);
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = rope.forward(&input, 1);
        // Pair (0,2): x0=1.0, x1=0.0 → (cos(1), sin(1))
        assert!((output[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((output[2] - 1.0f32.sin()).abs() < 1e-6);
        // Pair (1,3): x0=0.0, x1=0.0 → (0, 0)
        assert!(output[1].abs() < 1e-6);
        assert!(output[3].abs() < 1e-6);
    }
}
