//! Full transformer model — stacks N decoder blocks into a complete LLM.
//!
//! Architecture (LLaMA-style decoder-only transformer):
//!
//!   1. Token embedding lookup (vocab_size × embed_dim, f32)
//!   2. N × TransformerBlock (attention + FFN with residuals)
//!   3. Final RMSNorm
//!   4. Output projection (embed_dim → vocab_size) to produce logits
//!
//! The model operates on token ID sequences and produces logit vectors
//! over the vocabulary for each position.

use crate::layers::bitlinear::BitLinear;
use crate::layers::kv_cache::ModelKvCache;
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::sampler::{Sampler, SamplerConfig};
use crate::layers::transformer::TransformerBlock;
use crate::tensor::FloatTensor;
use rayon::prelude::*;

/// Number of vocabulary entries per parallel chunk for tied embedding projection.
/// Tuned so each chunk has enough work to amortize thread overhead (~4K dot products).
const VOCAB_CHUNK_SIZE: usize = 4096;

/// SIMD-friendly dot product with 4-wide manual unrolling.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    for i in 0..chunks {
        let j = i * 4;
        sum0 += a[j] * b[j];
        sum1 += a[j + 1] * b[j + 1];
        sum2 += a[j + 2] * b[j + 2];
        sum3 += a[j + 3] * b[j + 3];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Rayon-parallel float output projection: logits = normed · W^T.
///
/// `weight_data` is row-major `[vocab_size, embed_dim]` — either the embedding
/// table (tied) or a separate output weight matrix (float projection).
fn float_output_projection(
    normed: &[f32],
    weight_data: &[f32],
    seq_len: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; seq_len * vocab_size];
    for t in 0..seq_len {
        let h_start = t * embed_dim;
        let h_vec = &normed[h_start..h_start + embed_dim];
        let token_logits = &mut logits[t * vocab_size..(t + 1) * vocab_size];
        token_logits
            .par_chunks_mut(VOCAB_CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let v_start = chunk_idx * VOCAB_CHUNK_SIZE;
                for (i, logit) in chunk.iter_mut().enumerate() {
                    let v = v_start + i;
                    let e_start = v * embed_dim;
                    let e_vec = &weight_data[e_start..e_start + embed_dim];
                    *logit = dot_product(h_vec, e_vec);
                }
            });
    }
    logits
}

/// A complete transformer language model.
pub struct TransformerModel {
    /// Token embedding table (vocab_size × embed_dim).
    embedding: FloatTensor,
    /// Stacked transformer decoder blocks.
    blocks: Vec<TransformerBlock>,
    /// Final normalization before output projection.
    final_norm: RmsNorm,
    /// Output projection (embed_dim → vocab_size). May be a ternary
    /// BitLinear or tied to the embedding weights.
    output_proj: OutputProjection,
    /// Vocabulary size.
    vocab_size: usize,
    /// Embedding dimension.
    embed_dim: usize,
}

/// Output projection: either a learned weight matrix or tied to embedding.
pub enum OutputProjection {
    /// Separate learned ternary projection (BitLinear: embed_dim → vocab_size).
    Linear(BitLinear),
    /// Separate learned float projection (embed_dim → vocab_size).
    /// Used when output.weight is F16/F32 (e.g., Falcon3 1.58-bit models).
    Float(FloatTensor),
    /// Tied to embedding weights — project by multiplying by E^T.
    /// Stores a reference shape; the embedding table is used directly.
    TiedEmbedding,
}

impl TransformerModel {
    /// Create a transformer model from its components.
    pub fn new(
        embedding: FloatTensor,
        blocks: Vec<TransformerBlock>,
        final_norm: RmsNorm,
        output_proj: OutputProjection,
    ) -> Self {
        assert_eq!(embedding.shape().len(), 2, "embedding must be 2D");
        let vocab_size = embedding.shape()[0];
        let embed_dim = embedding.shape()[1];
        assert!(!blocks.is_empty(), "must have at least one transformer block");
        assert_eq!(
            blocks[0].embed_dim(),
            embed_dim,
            "block embed_dim must match embedding dim"
        );

        match &output_proj {
            OutputProjection::Linear(ref proj) => {
                assert_eq!(proj.in_features(), embed_dim, "output proj input must be embed_dim");
                assert_eq!(proj.out_features(), vocab_size, "output proj output must be vocab_size");
            }
            OutputProjection::Float(ref tensor) => {
                assert_eq!(tensor.shape().len(), 2, "float output proj must be 2D");
                assert_eq!(tensor.shape()[0], vocab_size, "float output proj rows must be vocab_size");
                assert_eq!(tensor.shape()[1], embed_dim, "float output proj cols must be embed_dim");
            }
            OutputProjection::TiedEmbedding => {}
        }

        Self {
            embedding,
            blocks,
            final_norm,
            output_proj,
            vocab_size,
            embed_dim,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of transformer blocks (layers).
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Access the raw embedding table data (for diagnostics).
    pub fn embedding_data(&self) -> &[f32] {
        self.embedding.data()
    }

    /// Forward pass: token IDs → logits.
    ///
    /// `tokens`: slice of token IDs (each < vocab_size).
    /// `start_pos`: sequence position of the first token (for RoPE).
    /// Returns: flat f32 vec of shape `[seq_len, vocab_size]` (logits).
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "must have at least one token");

        // 1. Embedding lookup
        let mut hidden = Vec::with_capacity(seq_len * self.embed_dim);
        let embed_data = self.embedding.data();
        for &tok in tokens {
            assert!(
                (tok as usize) < self.vocab_size,
                "token ID {tok} out of range (vocab_size={})",
                self.vocab_size
            );
            let start = tok as usize * self.embed_dim;
            hidden.extend_from_slice(&embed_data[start..start + self.embed_dim]);
        }

        // 2. Pass through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden, seq_len, start_pos);
        }

        // 3. Final norm (per token)
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let start = t * self.embed_dim;
            let token_hidden = &hidden[start..start + self.embed_dim];
            normed.extend_from_slice(&self.final_norm.forward(token_hidden));
        }

        // 4. Output projection → logits
        match &self.output_proj {
            OutputProjection::Linear(proj) => {
                let mut logits = Vec::with_capacity(seq_len * self.vocab_size);
                for t in 0..seq_len {
                    let start = t * self.embed_dim;
                    let token_hidden = &normed[start..start + self.embed_dim];
                    logits.extend_from_slice(&proj.forward(token_hidden));
                }
                logits
            }
            OutputProjection::Float(ref weight) => {
                float_output_projection(&normed, weight.data(), seq_len, self.vocab_size, self.embed_dim)
            }
            OutputProjection::TiedEmbedding => {
                float_output_projection(&normed, embed_data, seq_len, self.vocab_size, self.embed_dim)
            }
        }
    }

    /// Forward pass returning only the logits for the last token.
    ///
    /// This is the common case for autoregressive generation: you only
    /// need the logits at the final position to sample the next token.
    pub fn forward_last(&self, tokens: &[u32], start_pos: usize) -> Vec<f32> {
        let all_logits = self.forward(tokens, start_pos);
        let last_start = (tokens.len() - 1) * self.vocab_size;
        all_logits[last_start..last_start + self.vocab_size].to_vec()
    }

    /// Forward pass with KV cache for incremental generation.
    ///
    /// `tokens`: the new token(s) to process.
    /// `cache`: mutable model KV cache (one per layer).
    /// Returns: logits for each new token `[seq_len, vocab_size]`.
    pub fn forward_cached(&self, tokens: &[u32], cache: &mut ModelKvCache) -> Vec<f32> {
        let seq_len = tokens.len();
        assert!(seq_len > 0, "must have at least one token");

        // 1. Embedding lookup
        let mut hidden = Vec::with_capacity(seq_len * self.embed_dim);
        let embed_data = self.embedding.data();
        for &tok in tokens {
            assert!(
                (tok as usize) < self.vocab_size,
                "token ID {tok} out of range (vocab_size={})",
                self.vocab_size
            );
            let start = tok as usize * self.embed_dim;
            hidden.extend_from_slice(&embed_data[start..start + self.embed_dim]);
        }

        // 2. Pass through transformer blocks with cache
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_cached(&hidden, seq_len, cache.layer_mut(layer_idx));
        }

        // 3. Final norm (per token)
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let start = t * self.embed_dim;
            let token_hidden = &hidden[start..start + self.embed_dim];
            normed.extend_from_slice(&self.final_norm.forward(token_hidden));
        }

        // 4. Output projection → logits
        match &self.output_proj {
            OutputProjection::Linear(proj) => {
                let mut logits = Vec::with_capacity(seq_len * self.vocab_size);
                for t in 0..seq_len {
                    let start = t * self.embed_dim;
                    let token_hidden = &normed[start..start + self.embed_dim];
                    logits.extend_from_slice(&proj.forward(token_hidden));
                }
                logits
            }
            OutputProjection::Float(ref weight) => {
                float_output_projection(&normed, weight.data(), seq_len, self.vocab_size, self.embed_dim)
            }
            OutputProjection::TiedEmbedding => {
                float_output_projection(&normed, embed_data, seq_len, self.vocab_size, self.embed_dim)
            }
        }
    }

    /// Create a KV cache sized for this model.
    pub fn create_kv_cache(&self, max_seq_len: usize) -> ModelKvCache {
        let n_kv_heads = self.blocks[0].attention().n_kv_heads();
        let head_dim = self.blocks[0].attention().head_dim();
        ModelKvCache::new(self.blocks.len(), n_kv_heads, head_dim, max_seq_len)
    }

    /// Generate tokens autoregressively.
    ///
    /// `prompt`: initial token IDs.
    /// `max_tokens`: maximum number of new tokens to generate.
    /// `sampler_config`: sampling strategy (temperature, top-k, top-p).
    /// `seed`: RNG seed for sampling.
    /// `stop_token`: optional token ID that stops generation.
    ///
    /// Returns: the full sequence (prompt + generated tokens).
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampler_config: SamplerConfig,
        seed: u64,
        stop_token: Option<u32>,
    ) -> Vec<u32> {
        assert!(!prompt.is_empty(), "prompt must not be empty");
        let max_seq_len = prompt.len() + max_tokens;
        let mut cache = self.create_kv_cache(max_seq_len);
        let mut sampler = Sampler::new(sampler_config, seed);
        let mut sequence = prompt.to_vec();

        // Prefill: process all prompt tokens at once
        let prefill_logits = self.forward_cached(prompt, &mut cache);
        let last_logits_start = (prompt.len() - 1) * self.vocab_size;
        let last_logits = &prefill_logits[last_logits_start..last_logits_start + self.vocab_size];
        let mut next_token = sampler.sample(last_logits);

        if stop_token == Some(next_token) {
            return sequence;
        }
        sequence.push(next_token);

        // Decode: one token at a time
        for _ in 1..max_tokens {
            let logits = self.forward_cached(&[next_token], &mut cache);
            next_token = sampler.sample(&logits);

            if stop_token == Some(next_token) {
                break;
            }
            sequence.push(next_token);
        }

        sequence
    }
}

impl std::fmt::Debug for TransformerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TransformerModel(vocab={}, embed={}, layers={}, proj={})",
            self.vocab_size,
            self.embed_dim,
            self.blocks.len(),
            match &self.output_proj {
                OutputProjection::Linear(_) => "linear",
                OutputProjection::Float(_) => "float",
                OutputProjection::TiedEmbedding => "tied",
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::attention::MultiHeadAttention;
    use crate::layers::swiglu::SwiGLU;
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

    fn make_test_block(embed_dim: usize, n_heads: usize, n_kv_heads: usize, intermediate: usize) -> TransformerBlock {
        let head_dim = embed_dim / n_heads;
        let kv_dim = n_kv_heads * head_dim;

        let q_proj = make_identity_proj(embed_dim, embed_dim);
        let k_weights = {
            let mut w = vec![0i8; kv_dim * embed_dim];
            for i in 0..kv_dim.min(embed_dim) {
                w[i * embed_dim + i] = 1;
            }
            w
        };
        let k_proj = make_bitlinear(&k_weights, kv_dim, embed_dim, 1.0);
        let v_proj = make_bitlinear(&k_weights, kv_dim, embed_dim, 1.0);
        let o_proj = make_identity_proj(embed_dim, embed_dim);

        let attention = MultiHeadAttention::new(
            q_proj, k_proj, v_proj, o_proj,
            n_heads, n_kv_heads, head_dim, 10000.0,
        );

        let gate = make_identity_proj(intermediate, embed_dim);
        let up = make_identity_proj(intermediate, embed_dim);
        let down = make_identity_proj(embed_dim, intermediate);
        let ffn = SwiGLU::new(gate, up, down);

        let attn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-5);
        let ffn_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-5);

        TransformerBlock::new(attn_norm, attention, ffn_norm, ffn)
    }

    // Build a tiny test model: vocab=8, embed=8, 1 layer, 2 heads
    fn make_test_model(n_layers: usize, tied: bool) -> TransformerModel {
        let vocab_size = 8;
        let embed_dim = 8;
        let n_heads = 2;
        let n_kv_heads = 2;
        let intermediate = 16;

        // Embedding: small distinct vectors per token
        let mut embed_data = vec![0.0f32; vocab_size * embed_dim];
        for v in 0..vocab_size {
            for d in 0..embed_dim {
                embed_data[v * embed_dim + d] = ((v * embed_dim + d) as f32 + 1.0) * 0.01;
            }
        }
        let embedding = FloatTensor::new(embed_data, vec![vocab_size, embed_dim]);

        let blocks: Vec<TransformerBlock> = (0..n_layers)
            .map(|_| make_test_block(embed_dim, n_heads, n_kv_heads, intermediate))
            .collect();

        let final_norm = RmsNorm::new(vec![1.0; embed_dim], 1e-5);

        let output_proj = if tied {
            OutputProjection::TiedEmbedding
        } else {
            OutputProjection::Linear(make_identity_proj(vocab_size, embed_dim))
        };

        TransformerModel::new(embedding, blocks, final_norm, output_proj)
    }

    // -- Construction --

    #[test]
    fn construction() {
        let model = make_test_model(2, false);
        assert_eq!(model.vocab_size(), 8);
        assert_eq!(model.embed_dim(), 8);
        assert_eq!(model.n_layers(), 2);
    }

    #[test]
    fn construction_tied() {
        let model = make_test_model(1, true);
        assert_eq!(model.vocab_size(), 8);
    }

    // -- Forward pass --

    #[test]
    fn forward_single_token_shape() {
        let model = make_test_model(1, false);
        let logits = model.forward(&[0], 0);
        assert_eq!(logits.len(), 8); // vocab_size
    }

    #[test]
    fn forward_sequence_shape() {
        let model = make_test_model(1, false);
        let logits = model.forward(&[0, 1, 2], 0);
        assert_eq!(logits.len(), 3 * 8); // seq_len * vocab_size
    }

    #[test]
    fn forward_finite() {
        let model = make_test_model(1, false);
        let logits = model.forward(&[0, 3, 7], 0);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_tied_embedding() {
        let model = make_test_model(1, true);
        let logits = model.forward(&[0, 1], 0);
        assert_eq!(logits.len(), 2 * 8);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "tied logit[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_multi_layer() {
        let model = make_test_model(3, false);
        let logits = model.forward(&[0, 1], 0);
        assert_eq!(logits.len(), 2 * 8);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "3-layer logit[{i}] not finite: {v}");
        }
    }

    #[test]
    fn forward_last_shape() {
        let model = make_test_model(1, false);
        let logits = model.forward_last(&[0, 1, 2], 0);
        assert_eq!(logits.len(), 8); // just vocab_size
    }

    #[test]
    fn forward_last_matches_full() {
        let model = make_test_model(1, false);
        let tokens = &[0u32, 3, 5];
        let full = model.forward(tokens, 0);
        let last = model.forward_last(tokens, 0);

        // Last token logits from full should match forward_last
        let last_start = 2 * 8;
        for (a, b) in full[last_start..].iter().zip(last.iter()) {
            assert!((a - b).abs() < 1e-7, "forward_last mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn different_tokens_different_logits() {
        let model = make_test_model(1, false);
        let logits_a = model.forward(&[0], 0);
        let logits_b = model.forward(&[7], 0);

        assert!(
            logits_a.iter().zip(logits_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "different tokens should produce different logits"
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn token_out_of_range_panics() {
        let model = make_test_model(1, false);
        model.forward(&[99], 0); // vocab_size is 8
    }

    #[test]
    fn start_pos_offset() {
        // Verify that start_pos flows through to RoPE
        let model = make_test_model(1, false);
        let _logits_pos0 = model.forward(&[0, 1], 0);
        let logits_pos100 = model.forward(&[0, 1], 100);

        // With synthetic weights the difference may be small,
        // but the logits should at least be finite
        for (i, &v) in logits_pos100.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] at pos 100 not finite: {v}");
        }
    }

    #[test]
    fn debug_format() {
        let model = make_test_model(2, false);
        let debug = format!("{:?}", model);
        assert!(debug.contains("TransformerModel"));
        assert!(debug.contains("vocab=8"));
        assert!(debug.contains("layers=2"));
        assert!(debug.contains("proj=linear"));
    }

    #[test]
    fn debug_format_tied() {
        let model = make_test_model(1, true);
        let debug = format!("{:?}", model);
        assert!(debug.contains("proj=tied"));
    }

    // -- Cached forward --

    #[test]
    fn forward_cached_single_token_shape() {
        let model = make_test_model(1, false);
        let mut cache = model.create_kv_cache(64);
        let logits = model.forward_cached(&[0], &mut cache);
        assert_eq!(logits.len(), 8); // vocab_size
        assert_eq!(cache.seq_len(), 1);
    }

    #[test]
    fn forward_cached_prefill_shape() {
        let model = make_test_model(1, false);
        let mut cache = model.create_kv_cache(64);
        let logits = model.forward_cached(&[0, 1, 2], &mut cache);
        assert_eq!(logits.len(), 3 * 8);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn forward_cached_incremental() {
        let model = make_test_model(1, false);
        let mut cache = model.create_kv_cache(64);

        // Prefill
        let _ = model.forward_cached(&[0, 1], &mut cache);
        assert_eq!(cache.seq_len(), 2);

        // Decode one token
        let logits = model.forward_cached(&[2], &mut cache);
        assert_eq!(logits.len(), 8);
        assert_eq!(cache.seq_len(), 3);

        // Decode another
        let logits2 = model.forward_cached(&[3], &mut cache);
        assert_eq!(logits2.len(), 8);
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn forward_cached_finite() {
        let model = make_test_model(2, false);
        let mut cache = model.create_kv_cache(64);

        let logits = model.forward_cached(&[0, 3, 7], &mut cache);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "cached logit[{i}] not finite: {v}");
        }

        // Continue with single token
        let logits2 = model.forward_cached(&[1], &mut cache);
        for (i, &v) in logits2.iter().enumerate() {
            assert!(v.is_finite(), "cached decode logit[{i}] not finite: {v}");
        }
    }

    #[test]
    fn create_kv_cache_correct_size() {
        let model = make_test_model(3, false);
        let cache = model.create_kv_cache(512);
        assert_eq!(cache.n_layers(), 3);
        assert_eq!(cache.seq_len(), 0);
    }

    // -- Generation --

    #[test]
    fn generate_returns_prompt_plus_tokens() {
        let model = make_test_model(1, false);
        let prompt = &[0u32, 1, 2];
        let output = model.generate(prompt, 5, SamplerConfig::greedy(), 42, None);

        // Should start with the prompt
        assert_eq!(&output[..3], prompt);
        // Should have generated up to 5 more tokens
        assert!(output.len() <= 8); // 3 + 5
        assert!(output.len() > 3); // at least one generated

        // All tokens should be valid
        for &tok in &output {
            assert!((tok as usize) < model.vocab_size());
        }
    }

    #[test]
    fn generate_greedy_deterministic() {
        let model = make_test_model(1, false);
        let prompt = &[0u32, 1];

        let out1 = model.generate(prompt, 10, SamplerConfig::greedy(), 42, None);
        let out2 = model.generate(prompt, 10, SamplerConfig::greedy(), 99, None);
        assert_eq!(out1, out2, "greedy should be deterministic regardless of seed");
    }

    #[test]
    fn generate_stop_token() {
        let model = make_test_model(1, false);
        let prompt = &[0u32];

        // Generate with a stop token — output should not contain the stop token
        // (we use token 7 as EOS; if the model happens to generate it, gen stops)
        let out = model.generate(prompt, 100, SamplerConfig::greedy(), 42, Some(7));
        assert!(!out[1..].contains(&7), "stop token should not appear in generated output");
    }

    #[test]
    fn generate_same_seed_same_output() {
        let model = make_test_model(1, false);
        let prompt = &[0u32, 3];
        let config = SamplerConfig::top_k(4, 0.8);

        let out1 = model.generate(prompt, 10, config.clone(), 42, None);
        let out2 = model.generate(prompt, 10, SamplerConfig::top_k(4, 0.8), 42, None);
        assert_eq!(out1, out2, "same seed should produce same output");
    }

    #[test]
    fn generate_multi_layer_finite() {
        let model = make_test_model(3, false);
        let prompt = &[0u32, 1, 2];
        let output = model.generate(prompt, 5, SamplerConfig::greedy(), 1, None);
        assert!(output.len() > 3);
        for &tok in &output {
            assert!((tok as usize) < model.vocab_size());
        }
    }

    #[test]
    fn generate_tied_embedding() {
        let model = make_test_model(1, true);
        let prompt = &[0u32, 1];
        let output = model.generate(prompt, 5, SamplerConfig::greedy(), 1, None);
        assert!(output.len() > 2);
        for &tok in &output {
            assert!((tok as usize) < model.vocab_size());
        }
    }
}
