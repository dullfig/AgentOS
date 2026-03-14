//! Token sampler — selects the next token from logit distributions.
//!
//! Supports three sampling strategies:
//! - **Greedy** (argmax): deterministic, always picks the highest logit
//! - **Top-k**: sample from the k highest-probability tokens
//! - **Top-p** (nucleus): sample from the smallest set whose cumulative
//!   probability exceeds p
//!
//! Temperature scaling is applied before any filtering. Temperature < 1.0
//! sharpens the distribution (more deterministic), > 1.0 flattens it
//! (more random).

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for logit scaling. 1.0 = no change.
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus): keep tokens until cumulative prob exceeds p. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty: logits for recently generated tokens are divided by
    /// this value (if > 1.0). 1.0 = disabled. Typical: 1.1–1.3.
    pub repetition_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty.
    /// 0 = disabled (no penalty window).
    pub repetition_window: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
        }
    }
}

impl SamplerConfig {
    /// Greedy (argmax) sampling.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Top-k sampling with temperature.
    pub fn top_k(k: usize, temperature: f32) -> Self {
        Self {
            temperature,
            top_k: k,
            ..Default::default()
        }
    }

    /// Top-p (nucleus) sampling with temperature.
    pub fn top_p(p: f32, temperature: f32) -> Self {
        Self {
            temperature,
            top_p: p,
            ..Default::default()
        }
    }
}

/// A token sampler that converts logits into token IDs.
pub struct Sampler {
    config: SamplerConfig,
    /// Simple xorshift64 RNG state.
    rng_state: u64,
    /// Ring buffer of recently generated token IDs for repetition penalty.
    recent_tokens: Vec<u32>,
}

impl Sampler {
    /// Create a sampler with the given config and RNG seed.
    pub fn new(config: SamplerConfig, seed: u64) -> Self {
        Self {
            config,
            rng_state: if seed == 0 { 1 } else { seed },
            recent_tokens: Vec::new(),
        }
    }

    /// Create a greedy (deterministic) sampler.
    pub fn greedy() -> Self {
        Self::new(SamplerConfig::greedy(), 1)
    }

    /// Sample a token ID from the logit vector.
    ///
    /// `logits`: raw logit scores, one per vocabulary entry.
    /// Returns the selected token ID.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        assert!(!logits.is_empty(), "logits must not be empty");

        // 0. Apply repetition penalty
        let logits = if self.config.repetition_penalty > 1.0 && !self.recent_tokens.is_empty() {
            let mut penalized = logits.to_vec();
            for &tok in &self.recent_tokens {
                let idx = tok as usize;
                if idx < penalized.len() {
                    if penalized[idx] > 0.0 {
                        penalized[idx] /= self.config.repetition_penalty;
                    } else {
                        penalized[idx] *= self.config.repetition_penalty;
                    }
                }
            }
            penalized
        } else {
            logits.to_vec()
        };
        let logits = &logits;

        // Temperature 0 or top_k=1: pure greedy
        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            let token = argmax(logits) as u32;
            self.record_token(token);
            return token;
        }

        // 1. Apply temperature
        let mut scaled: Vec<f32> = logits.iter().map(|&l| l / self.config.temperature).collect();

        // 2. Convert to probabilities via softmax
        softmax_inplace(&mut scaled);

        // 3. Build sorted index (descending probability)
        let mut indices: Vec<usize> = (0..scaled.len()).collect();
        indices.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());

        // 4. Apply top-k filter
        let mut candidates = indices.len();
        if self.config.top_k > 0 && self.config.top_k < candidates {
            candidates = self.config.top_k;
        }

        // 5. Apply top-p (nucleus) filter
        if self.config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            for i in 0..candidates {
                cumulative += scaled[indices[i]];
                if cumulative >= self.config.top_p {
                    candidates = i + 1;
                    break;
                }
            }
        }

        // 6. Renormalize the candidate probabilities
        let total: f32 = indices[..candidates].iter().map(|&i| scaled[i]).sum();
        let inv_total = if total > 0.0 { 1.0 / total } else { 1.0 };

        // 7. Sample from the filtered distribution
        let r = self.next_f32();
        let mut cumulative = 0.0f32;
        for &idx in &indices[..candidates] {
            cumulative += scaled[idx] * inv_total;
            if r < cumulative {
                self.record_token(idx as u32);
                return idx as u32;
            }
        }

        // Fallback (floating point edge case)
        let token = indices[candidates - 1] as u32;
        self.record_token(token);
        token
    }

    /// Record a token in the recent tokens ring buffer.
    fn record_token(&mut self, token: u32) {
        if self.config.repetition_window > 0 {
            self.recent_tokens.push(token);
            if self.recent_tokens.len() > self.config.repetition_window {
                self.recent_tokens.remove(0);
            }
        }
    }

    /// Sample and also return the probability of the chosen token.
    pub fn sample_with_prob(&mut self, logits: &[f32]) -> (u32, f32) {
        assert!(!logits.is_empty());

        if self.config.temperature <= 0.0 || self.config.top_k == 1 {
            let idx = argmax(logits);
            let mut probs = logits.to_vec();
            softmax_inplace(&mut probs);
            return (idx as u32, probs[idx]);
        }

        let mut scaled: Vec<f32> = logits.iter().map(|&l| l / self.config.temperature).collect();
        softmax_inplace(&mut scaled);

        let token = self.sample_from_probs(&scaled);
        (token, scaled[token as usize])
    }

    /// Internal: sample from a probability distribution (already softmaxed).
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut candidates = indices.len();
        if self.config.top_k > 0 && self.config.top_k < candidates {
            candidates = self.config.top_k;
        }

        if self.config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            for i in 0..candidates {
                cumulative += probs[indices[i]];
                if cumulative >= self.config.top_p {
                    candidates = i + 1;
                    break;
                }
            }
        }

        let total: f32 = indices[..candidates].iter().map(|&i| probs[i]).sum();
        let inv_total = if total > 0.0 { 1.0 / total } else { 1.0 };

        let r = self.next_f32();
        let mut cumulative = 0.0f32;
        for &idx in &indices[..candidates] {
            cumulative += probs[idx] * inv_total;
            if r < cumulative {
                return idx as u32;
            }
        }

        indices[candidates - 1] as u32
    }

    /// Xorshift64 PRNG — returns next u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Returns a random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sampler({:?})", self.config)
    }
}

/// Argmax: index of the largest value.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// In-place softmax with numerical stability.
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

    // -- Argmax --

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn argmax_negative() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    // -- SamplerConfig --

    #[test]
    fn config_default() {
        let c = SamplerConfig::default();
        assert_eq!(c.temperature, 1.0);
        assert_eq!(c.top_k, 0);
        assert_eq!(c.top_p, 1.0);
    }

    #[test]
    fn config_greedy() {
        let c = SamplerConfig::greedy();
        assert_eq!(c.top_k, 1);
    }

    // -- Greedy sampling --

    #[test]
    fn greedy_always_picks_max() {
        let mut sampler = Sampler::greedy();
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        for _ in 0..10 {
            assert_eq!(sampler.sample(&logits), 1);
        }
    }

    #[test]
    fn greedy_zero_temperature() {
        let mut sampler = Sampler::new(
            SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                ..Default::default()
            },
            42,
        );
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        assert_eq!(sampler.sample(&logits), 2);
    }

    // -- Top-k sampling --

    #[test]
    fn top_k_restricts_candidates() {
        let config = SamplerConfig::top_k(2, 1.0);
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 10.0, 9.0, 0.1];

        // With top_k=2, only tokens 1 and 2 should ever be sampled
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(sampler.sample(&logits));
        }
        assert!(
            seen.iter().all(|&t| t == 1 || t == 2),
            "top_k=2 should only sample from top 2 tokens, got: {:?}",
            seen
        );
    }

    #[test]
    fn top_k_1_is_greedy() {
        let config = SamplerConfig::top_k(1, 1.0);
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 5.0, 3.0];
        for _ in 0..10 {
            assert_eq!(sampler.sample(&logits), 1);
        }
    }

    // -- Top-p (nucleus) sampling --

    #[test]
    fn top_p_restricts_distribution() {
        // With very peaked logits and low top_p, should mostly pick the top token
        let config = SamplerConfig::top_p(0.5, 1.0);
        let mut sampler = Sampler::new(config, 42);
        // Token 2 has logit 100, so after softmax it has >99% probability
        let logits = vec![1.0, 2.0, 100.0, 1.0];
        for _ in 0..100 {
            assert_eq!(sampler.sample(&logits), 2);
        }
    }

    #[test]
    fn top_p_1_allows_all() {
        let config = SamplerConfig::top_p(1.0, 1.0);
        let mut sampler = Sampler::new(config, 42);
        // Equal logits — all tokens should eventually appear
        let logits = vec![0.0; 4];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sampler.sample(&logits));
        }
        assert_eq!(seen.len(), 4, "top_p=1.0 with equal logits should sample all tokens");
    }

    // -- Temperature --

    #[test]
    fn low_temperature_sharpens() {
        // With very low temperature, should act like greedy
        let config = SamplerConfig {
            temperature: 0.01,
            top_k: 0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits), 1);
        }
    }

    #[test]
    fn high_temperature_flattens() {
        // High temperature with equal-ish logits should produce variety
        let config = SamplerConfig {
            temperature: 10.0,
            top_k: 0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 1.1, 1.2, 1.3];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sampler.sample(&logits));
        }
        assert!(seen.len() >= 3, "high temperature should produce variety");
    }

    // -- Combined top-k + top-p --

    #[test]
    fn top_k_and_top_p_combined() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.9,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![10.0, 9.0, 8.0, 1.0, 0.5];

        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(sampler.sample(&logits));
        }
        // Should only see tokens 0, 1, 2 (top-k=3 and they dominate probability)
        assert!(
            seen.iter().all(|&t| t <= 2),
            "combined filter should restrict to top tokens, got: {:?}",
            seen
        );
    }

    // -- sample_with_prob --

    #[test]
    fn sample_with_prob_greedy() {
        let mut sampler = Sampler::greedy();
        let logits = vec![1.0, 10.0, 3.0];
        let (token, prob) = sampler.sample_with_prob(&logits);
        assert_eq!(token, 1);
        assert!(prob > 0.9, "top token should have high probability");
    }

    #[test]
    fn sample_with_prob_returns_valid_prob() {
        let config = SamplerConfig::top_k(4, 1.0);
        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let (_token, prob) = sampler.sample_with_prob(&logits);
        assert!(prob > 0.0 && prob <= 1.0, "probability should be in (0, 1]");
    }

    // -- RNG reproducibility --

    #[test]
    fn same_seed_same_sequence() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            ..Default::default()
        };

        let mut s1 = Sampler::new(config.clone(), 12345);
        let mut s2 = Sampler::new(config, 12345);

        let seq1: Vec<u32> = (0..20).map(|_| s1.sample(&logits)).collect();
        let seq2: Vec<u32> = (0..20).map(|_| s2.sample(&logits)).collect();
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn different_seed_different_sequence() {
        let logits = vec![1.0, 1.0, 1.0, 1.0]; // uniform
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            ..Default::default()
        };

        let mut s1 = Sampler::new(config.clone(), 111);
        let mut s2 = Sampler::new(config, 222);

        let seq1: Vec<u32> = (0..20).map(|_| s1.sample(&logits)).collect();
        let seq2: Vec<u32> = (0..20).map(|_| s2.sample(&logits)).collect();
        // Extremely unlikely to be identical with different seeds on uniform dist
        assert_ne!(seq1, seq2);
    }

    // -- Edge cases --

    #[test]
    fn single_token_vocab() {
        let mut sampler = Sampler::new(SamplerConfig::default(), 42);
        assert_eq!(sampler.sample(&[5.0]), 0);
    }

    #[test]
    fn large_logit_range() {
        // Test numerical stability with huge logit differences
        let mut sampler = Sampler::greedy();
        let logits = vec![-1000.0, 1000.0, -1000.0];
        assert_eq!(sampler.sample(&logits), 1);
    }

    // -- Debug format --

    #[test]
    fn debug_format() {
        let sampler = Sampler::greedy();
        let debug = format!("{:?}", sampler);
        assert!(debug.contains("Sampler"));
    }
}
