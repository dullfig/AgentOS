//! Multi-Head Attention with ternary Q/K/V projections.
//!
//! Implements Grouped Query Attention (GQA) as used in LLaMA 2/3 and
//! BitNet b1.58. Q has `n_heads` heads while K/V share `n_kv_heads`
//! (where `n_kv_heads` divides `n_heads`). This reduces KV memory and
//! compute while preserving model quality.
//!
//! Forward pass for a sequence of length `seq_len`:
//!   1. Project input through ternary Q, K, V, O projections (BitLinear)
//!   2. Apply RoPE to Q and K
//!   3. Reshape into heads
//!   4. Expand KV heads to match Q head count (GQA repeat)
//!   5. Scaled dot-product attention with causal mask
//!   6. Concatenate heads and project through O

use crate::layers::bitlinear::BitLinear;
use crate::layers::kv_cache::KvCache;
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::{RoPE, RoPELayout};

/// Multi-head attention with ternary weight projections.
pub struct MultiHeadAttention {
    /// Query projection (embed_dim → n_heads * head_dim).
    q_proj: BitLinear,
    /// Key projection (embed_dim → n_kv_heads * head_dim).
    k_proj: BitLinear,
    /// Value projection (embed_dim → n_kv_heads * head_dim).
    v_proj: BitLinear,
    /// Output projection (n_heads * head_dim → embed_dim).
    o_proj: BitLinear,
    /// Rotary position embeddings for Q and K.
    rope: RoPE,
    /// Optional sub-normalization applied to concatenated heads before O projection.
    /// This is the BitNet b1.58 "attn_sub_norm" — normalizes activations before
    /// quantization in the output projection.
    o_sub_norm: Option<RmsNorm>,
    /// Total number of query heads.
    n_heads: usize,
    /// Number of key/value heads (GQA: n_kv_heads ≤ n_heads).
    n_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Create a multi-head attention layer with interleaved RoPE (default).
    ///
    /// - `n_heads`: number of query heads
    /// - `n_kv_heads`: number of KV heads (must divide `n_heads`)
    /// - `head_dim`: dimension per head
    /// - `rope_base`: frequency base for RoPE (typically 10000.0)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Self {
        Self::with_rope_layout(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, rope_base,
            RoPELayout::Interleaved,
        )
    }

    /// Create a multi-head attention layer with explicit RoPE layout.
    #[allow(clippy::too_many_arguments)]
    pub fn with_rope_layout(
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        rope_layout: RoPELayout,
    ) -> Self {
        assert!(n_heads > 0 && n_kv_heads > 0, "head counts must be positive");
        assert!(
            n_heads.is_multiple_of(n_kv_heads),
            "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        );

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        assert_eq!(q_proj.out_features(), q_dim, "q_proj output must be n_heads * head_dim");
        assert_eq!(k_proj.out_features(), kv_dim, "k_proj output must be n_kv_heads * head_dim");
        assert_eq!(v_proj.out_features(), kv_dim, "v_proj output must be n_kv_heads * head_dim");
        assert_eq!(o_proj.in_features(), q_dim, "o_proj input must be n_heads * head_dim");
        assert_eq!(
            q_proj.in_features(),
            o_proj.out_features(),
            "q_proj input and o_proj output must both be embed_dim"
        );

        let rope = RoPE::with_layout(head_dim, rope_base, rope_layout);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            o_sub_norm: None,
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// Set the sub-normalization applied before the O projection (BitNet b1.58).
    ///
    /// This normalizes the concatenated head outputs before they enter the
    /// output projection's quantization step, improving quantization fidelity.
    pub fn set_o_sub_norm(&mut self, norm: RmsNorm) {
        self.o_sub_norm = Some(norm);
    }

    /// Embedding dimension (input/output size).
    pub fn embed_dim(&self) -> usize {
        self.q_proj.in_features()
    }

    /// Number of query heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of Q heads per KV head group.
    fn heads_per_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Forward pass over a sequence.
    ///
    /// `input`: flat f32 slice of shape `[seq_len, embed_dim]`.
    /// `start_pos`: sequence position of the first token (for RoPE).
    /// Returns: flat f32 vec of shape `[seq_len, embed_dim]`.
    pub fn forward(&self, input: &[f32], seq_len: usize, start_pos: usize) -> Vec<f32> {
        let embed_dim = self.embed_dim();
        assert_eq!(input.len(), seq_len * embed_dim, "input shape mismatch");

        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;

        // 1. Project each token through Q, K, V
        let mut q_all = Vec::with_capacity(seq_len * q_dim);
        let mut k_all = Vec::with_capacity(seq_len * kv_dim);
        let mut v_all = Vec::with_capacity(seq_len * kv_dim);

        for t in 0..seq_len {
            let token = &input[t * embed_dim..(t + 1) * embed_dim];
            q_all.extend_from_slice(&self.q_proj.forward(token));
            k_all.extend_from_slice(&self.k_proj.forward(token));
            v_all.extend_from_slice(&self.v_proj.forward(token));
        }

        // 2. Apply RoPE to Q and K (per-head, per-position)
        for t in 0..seq_len {
            let pos = start_pos + t;
            // Q: n_heads heads
            for h in 0..self.n_heads {
                let offset = t * q_dim + h * self.head_dim;
                let slice = &q_all[offset..offset + self.head_dim].to_vec();
                let rotated = self.rope.forward(slice, pos);
                q_all[offset..offset + self.head_dim].copy_from_slice(&rotated);
            }
            // K: n_kv_heads heads
            for h in 0..self.n_kv_heads {
                let offset = t * kv_dim + h * self.head_dim;
                let slice = &k_all[offset..offset + self.head_dim].to_vec();
                let rotated = self.rope.forward(slice, pos);
                k_all[offset..offset + self.head_dim].copy_from_slice(&rotated);
            }
        }

        // 3. Compute attention per Q-head
        let mut output = vec![0.0f32; seq_len * q_dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let hpg = self.heads_per_group();

        for qh in 0..self.n_heads {
            let kv_h = qh / hpg; // which KV head this Q head reads from

            for t in 0..seq_len {
                // Compute attention scores: Q[t,qh] · K[s,kv_h] for s=0..=t
                let q_off = t * q_dim + qh * self.head_dim;
                let q_vec = &q_all[q_off..q_off + self.head_dim];

                // Collect scores with causal mask (only attend to s <= t)
                let attend_len = t + 1;
                let mut scores = Vec::with_capacity(attend_len);

                for s in 0..attend_len {
                    let k_off = s * kv_dim + kv_h * self.head_dim;
                    let k_vec = &k_all[k_off..k_off + self.head_dim];
                    let dot: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores.push(dot * scale);
                }

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted sum of V vectors
                let out_off = t * q_dim + qh * self.head_dim;
                for (s, &weight) in scores.iter().enumerate() {
                    let v_off = s * kv_dim + kv_h * self.head_dim;
                    for d in 0..self.head_dim {
                        output[out_off + d] += weight * v_all[v_off + d];
                    }
                }
            }
        }

        // 4. Output projection: concatenated heads → embed_dim per token
        //    Apply sub-norm before O projection if present (BitNet b1.58:
        //    normalizes activations before quantization in the output projection).
        let mut final_output = Vec::with_capacity(seq_len * embed_dim);
        for t in 0..seq_len {
            let head_concat = &output[t * q_dim..(t + 1) * q_dim];
            let normed = match &self.o_sub_norm {
                Some(norm) => norm.forward(head_concat),
                None => head_concat.to_vec(),
            };
            final_output.extend_from_slice(&self.o_proj.forward(&normed));
        }

        final_output
    }

    /// Forward pass with KV cache for incremental generation.
    ///
    /// During prefill, `seq_len` equals the prompt length and the cache starts empty.
    /// During decode, `seq_len` is 1 (single new token) and the cache holds all
    /// prior positions.
    ///
    /// New K/V projections are appended to the cache. Q attends to the full
    /// cached K/V sequence (all prior + current positions).
    ///
    /// `input`: flat f32 of shape `[seq_len, embed_dim]` (the new tokens only).
    /// `cache`: mutable KV cache for this layer.
    /// Returns: flat f32 vec of shape `[seq_len, embed_dim]`.
    pub fn forward_cached(
        &self,
        input: &[f32],
        seq_len: usize,
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let embed_dim = self.embed_dim();
        assert_eq!(input.len(), seq_len * embed_dim, "input shape mismatch");

        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let start_pos = cache.len();

        // 1. Project new tokens through Q, K, V
        let mut q_all = Vec::with_capacity(seq_len * q_dim);
        let mut k_new = Vec::with_capacity(seq_len * kv_dim);
        let mut v_new = Vec::with_capacity(seq_len * kv_dim);

        for t in 0..seq_len {
            let token = &input[t * embed_dim..(t + 1) * embed_dim];
            q_all.extend_from_slice(&self.q_proj.forward(token));
            k_new.extend_from_slice(&self.k_proj.forward(token));
            v_new.extend_from_slice(&self.v_proj.forward(token));
        }

        // 2. Apply RoPE to Q and new K
        for t in 0..seq_len {
            let pos = start_pos + t;
            for h in 0..self.n_heads {
                let offset = t * q_dim + h * self.head_dim;
                let slice = q_all[offset..offset + self.head_dim].to_vec();
                let rotated = self.rope.forward(&slice, pos);
                q_all[offset..offset + self.head_dim].copy_from_slice(&rotated);
            }
            for h in 0..self.n_kv_heads {
                let offset = t * kv_dim + h * self.head_dim;
                let slice = k_new[offset..offset + self.head_dim].to_vec();
                let rotated = self.rope.forward(&slice, pos);
                k_new[offset..offset + self.head_dim].copy_from_slice(&rotated);
            }
        }

        // 3. Append new K/V to cache (already RoPE'd)
        cache.append(&k_new, &v_new);

        // 4. Compute attention: Q attends to full cached K/V
        let total_seq = cache.len(); // all positions including new ones
        let mut output = vec![0.0f32; seq_len * q_dim];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let hpg = self.heads_per_group();

        for qh in 0..self.n_heads {
            let kv_h = qh / hpg;

            for t in 0..seq_len {
                let abs_pos = start_pos + t;
                let q_off = t * q_dim + qh * self.head_dim;
                let q_vec = &q_all[q_off..q_off + self.head_dim];

                // Causal: attend to positions 0..=abs_pos
                let attend_len = (abs_pos + 1).min(total_seq);
                let mut scores = Vec::with_capacity(attend_len);

                for s in 0..attend_len {
                    let k_vec = cache.key_at(s, kv_h);
                    let dot: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores.push(dot * scale);
                }

                softmax_inplace(&mut scores);

                let out_off = t * q_dim + qh * self.head_dim;
                for (s, &weight) in scores.iter().enumerate() {
                    let v_vec = cache.value_at(s, kv_h);
                    for d in 0..self.head_dim {
                        output[out_off + d] += weight * v_vec[d];
                    }
                }
            }
        }

        // 5. Output projection (with optional sub-norm before O)
        let mut final_output = Vec::with_capacity(seq_len * embed_dim);
        for t in 0..seq_len {
            let head_concat = &output[t * q_dim..(t + 1) * q_dim];
            let normed = match &self.o_sub_norm {
                Some(norm) => norm.forward(head_concat),
                None => head_concat.to_vec(),
            };
            final_output.extend_from_slice(&self.o_proj.forward(&normed));
        }

        final_output
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultiHeadAttention(embed={}, heads={}, kv_heads={}, head_dim={})",
            self.embed_dim(),
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        )
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// In-place softmax with numerical stability (subtract max first).
fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    if values.len() == 1 {
        values[0] = 1.0;
        return;
    }

    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in values.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Ternary, TernaryTensor};

    // Helper: create a BitLinear with specific ternary weights.
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

    // Helper: identity-ish BitLinear (diagonal of +1, rest 0).
    fn make_identity_proj(dim: usize) -> BitLinear {
        let mut weights = vec![0i8; dim * dim];
        for i in 0..dim {
            weights[i * dim + i] = 1;
        }
        make_bitlinear(&weights, dim, dim, 1.0)
    }

    // Helper: build a basic MHA for testing.
    // Uses embed_dim == n_heads * head_dim with identity-ish projections.
    fn make_test_mha(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> MultiHeadAttention {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let embed_dim = q_dim; // keep it simple for tests

        // Q: embed_dim → q_dim (identity-ish when dims match)
        let q_proj = make_identity_proj(embed_dim);
        // K, V: embed_dim → kv_dim (truncated identity)
        let k_weights = {
            let mut w = vec![0i8; kv_dim * embed_dim];
            for i in 0..kv_dim.min(embed_dim) {
                w[i * embed_dim + i] = 1;
            }
            w
        };
        let k_proj = make_bitlinear(&k_weights, kv_dim, embed_dim, 1.0);
        let v_proj = make_bitlinear(&k_weights, kv_dim, embed_dim, 1.0);
        // O: q_dim → embed_dim
        let o_proj = make_identity_proj(embed_dim);

        MultiHeadAttention::new(q_proj, k_proj, v_proj, o_proj, n_heads, n_kv_heads, head_dim, 10000.0)
    }

    // -- Softmax tests --

    #[test]
    fn softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);

        // Sum should be 1.0
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Monotonically increasing
        assert!(v[0] < v[1]);
        assert!(v[1] < v[2]);
    }

    #[test]
    fn softmax_single() {
        let mut v = vec![42.0];
        softmax_inplace(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn softmax_equal_inputs() {
        let mut v = vec![1.0; 4];
        softmax_inplace(&mut v);
        for &val in &v {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let mut v = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(v[2] > v[1]);
        assert!(v[1] > v[0]);
    }

    // -- Construction tests --

    #[test]
    fn construction_mha() {
        let mha = make_test_mha(4, 4, 8);
        assert_eq!(mha.embed_dim(), 32);
        assert_eq!(mha.n_heads(), 4);
        assert_eq!(mha.n_kv_heads(), 4);
        assert_eq!(mha.head_dim(), 8);
    }

    #[test]
    fn construction_gqa() {
        let mha = make_test_mha(4, 2, 8);
        assert_eq!(mha.n_heads(), 4);
        assert_eq!(mha.n_kv_heads(), 2);
        assert_eq!(mha.heads_per_group(), 2);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn gqa_indivisible_panics() {
        make_test_mha(5, 3, 8); // 5 % 3 != 0
    }

    // -- Forward pass tests --

    #[test]
    fn forward_output_shape() {
        let mha = make_test_mha(2, 2, 4);
        let embed_dim = mha.embed_dim(); // 8
        let input = vec![1.0f32; embed_dim]; // single token
        let output = mha.forward(&input, 1, 0);
        assert_eq!(output.len(), embed_dim);
    }

    #[test]
    fn forward_sequence_shape() {
        let mha = make_test_mha(2, 2, 4);
        let embed_dim = mha.embed_dim(); // 8
        let seq_len = 3;
        let input = vec![0.5f32; seq_len * embed_dim];
        let output = mha.forward(&input, seq_len, 0);
        assert_eq!(output.len(), seq_len * embed_dim);
    }

    #[test]
    fn forward_single_token_finite() {
        let mha = make_test_mha(2, 2, 4);
        let embed_dim = mha.embed_dim();
        let input = vec![1.0f32; embed_dim];
        let output = mha.forward(&input, 1, 0);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn forward_gqa_output_shape() {
        // GQA: 4 Q heads, 2 KV heads
        let mha = make_test_mha(4, 2, 4);
        let embed_dim = mha.embed_dim(); // 16
        let input = vec![0.3f32; embed_dim];
        let output = mha.forward(&input, 1, 0);
        assert_eq!(output.len(), embed_dim);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "gqa output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn causal_mask_first_token() {
        // First token can only attend to itself → attention weight must be 1.0
        // for that single position. With identity-ish projections, the output
        // should approximate the value projection of the input.
        let mha = make_test_mha(2, 2, 4);
        let embed_dim = mha.embed_dim();
        let seq_len = 3;
        let mut input = vec![0.0f32; seq_len * embed_dim];
        // Make first token distinctive
        for i in 0..embed_dim {
            input[i] = (i as f32 + 1.0) * 0.1;
        }
        let output = mha.forward(&input, seq_len, 0);
        // First token's output should be finite and non-zero
        let first_tok: &[f32] = &output[0..embed_dim];
        assert!(
            first_tok.iter().any(|&v| v.abs() > 1e-6),
            "first token output should be non-zero"
        );
    }

    #[test]
    fn different_inputs_different_output() {
        let mha = make_test_mha(2, 2, 4);
        let embed_dim = mha.embed_dim();

        let input_a: Vec<f32> = (0..embed_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let input_b: Vec<f32> = (0..embed_dim).map(|i| (i as f32 + 1.0) * -0.2).collect();

        let out_a = mha.forward(&input_a, 1, 0);
        let out_b = mha.forward(&input_b, 1, 0);

        assert!(
            out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "different inputs should produce different outputs"
        );
    }

    #[test]
    fn debug_format() {
        let mha = make_test_mha(4, 2, 8);
        let debug = format!("{:?}", mha);
        assert!(debug.contains("MultiHeadAttention"));
        assert!(debug.contains("heads=4"));
        assert!(debug.contains("kv_heads=2"));
        assert!(debug.contains("head_dim=8"));
    }
}
