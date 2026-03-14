//! KV Cache — stores key/value projections for autoregressive generation.
//!
//! During generation, each new token only needs to compute its own Q/K/V.
//! The K and V from all prior positions are cached so we don't recompute
//! them. The cache is pre-allocated for the maximum sequence length.
//!
//! Each transformer layer gets its own `KvCache` instance. The full model
//! holds a `Vec<KvCache>`, one per layer.
//!
//! Layout: K and V are stored as flat f32 arrays of shape
//! `[max_seq_len, n_kv_heads, head_dim]`, filled incrementally as tokens
//! are generated.

/// Pre-allocated key/value cache for one transformer layer.
pub struct KvCache {
    /// Cached key projections (already RoPE'd).
    /// Shape: `[max_seq_len, n_kv_heads * head_dim]` stored flat.
    k_cache: Vec<f32>,
    /// Cached value projections.
    /// Shape: `[max_seq_len, n_kv_heads * head_dim]` stored flat.
    v_cache: Vec<f32>,
    /// Number of KV heads.
    n_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Maximum sequence length (pre-allocated capacity).
    max_seq_len: usize,
    /// Current number of cached positions (next write position).
    len: usize,
}

impl KvCache {
    /// Create a new cache pre-allocated for `max_seq_len` positions.
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            k_cache: vec![0.0f32; max_seq_len * kv_dim],
            v_cache: vec![0.0f32; max_seq_len * kv_dim],
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum sequence length this cache can hold.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// KV dimension per position (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append K and V vectors for one or more new positions.
    ///
    /// `keys`: flat slice of `[n_new, n_kv_heads * head_dim]`.
    /// `values`: flat slice of `[n_new, n_kv_heads * head_dim]`.
    ///
    /// These should already have RoPE applied (for keys).
    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        let kv_dim = self.kv_dim();
        assert_eq!(keys.len() % kv_dim, 0, "keys length must be multiple of kv_dim");
        assert_eq!(keys.len(), values.len(), "keys and values must have same length");

        let n_new = keys.len() / kv_dim;
        assert!(
            self.len + n_new <= self.max_seq_len,
            "KV cache overflow: {} + {} > {}",
            self.len,
            n_new,
            self.max_seq_len
        );

        let start = self.len * kv_dim;
        let end = start + n_new * kv_dim;
        self.k_cache[start..end].copy_from_slice(keys);
        self.v_cache[start..end].copy_from_slice(values);
        self.len += n_new;
    }

    /// Get the cached key vector for a specific position and KV head.
    ///
    /// Returns a slice of length `head_dim`.
    pub fn key_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);
        let kv_dim = self.kv_dim();
        let offset = pos * kv_dim + kv_head * self.head_dim;
        &self.k_cache[offset..offset + self.head_dim]
    }

    /// Get the cached value vector for a specific position and KV head.
    ///
    /// Returns a slice of length `head_dim`.
    pub fn value_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);
        let kv_dim = self.kv_dim();
        let offset = pos * kv_dim + kv_head * self.head_dim;
        &self.v_cache[offset..offset + self.head_dim]
    }

    /// Get all cached keys as a flat slice up to current length.
    ///
    /// Shape: `[len, n_kv_heads * head_dim]`.
    pub fn keys(&self) -> &[f32] {
        let end = self.len * self.kv_dim();
        &self.k_cache[..end]
    }

    /// Get all cached values as a flat slice up to current length.
    ///
    /// Shape: `[len, n_kv_heads * head_dim]`.
    pub fn values(&self) -> &[f32] {
        let end = self.len * self.kv_dim();
        &self.v_cache[..end]
    }

    /// Reset the cache (reuse the allocation).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k_cache.len() * 4 + self.v_cache.len() * 4
    }
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KvCache(kv_heads={}, head_dim={}, len={}/{}, {:.1}KB)",
            self.n_kv_heads,
            self.head_dim,
            self.len,
            self.max_seq_len,
            self.memory_bytes() as f64 / 1024.0,
        )
    }
}

/// A complete KV cache for all layers of a transformer model.
pub struct ModelKvCache {
    /// Per-layer KV caches.
    layers: Vec<KvCache>,
}

impl ModelKvCache {
    /// Create caches for all layers.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| KvCache::new(n_kv_heads, head_dim, max_seq_len))
            .collect();
        Self { layers }
    }

    /// Get the cache for a specific layer.
    pub fn layer(&self, idx: usize) -> &KvCache {
        &self.layers[idx]
    }

    /// Get the mutable cache for a specific layer.
    pub fn layer_mut(&mut self, idx: usize) -> &mut KvCache {
        &mut self.layers[idx]
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Current cached sequence length (same across all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Reset all layer caches.
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }

    /// Total memory usage across all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|c| c.memory_bytes()).sum()
    }
}

impl std::fmt::Debug for ModelKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ModelKvCache(layers={}, seq={}/{}, {:.1}MB)",
            self.layers.len(),
            self.seq_len(),
            self.layers.first().map(|c| c.max_seq_len()).unwrap_or(0),
            self.memory_bytes() as f64 / (1024.0 * 1024.0),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_empty() {
        let cache = KvCache::new(4, 64, 2048);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_seq_len(), 2048);
        assert_eq!(cache.kv_dim(), 256); // 4 * 64
    }

    #[test]
    fn append_single_position() {
        let mut cache = KvCache::new(2, 4, 100);
        let kv_dim = cache.kv_dim(); // 8
        let keys = vec![1.0f32; kv_dim];
        let values = vec![2.0f32; kv_dim];

        cache.append(&keys, &values);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn append_multiple_positions() {
        let mut cache = KvCache::new(2, 4, 100);
        let kv_dim = cache.kv_dim();

        // Append 3 positions at once
        let keys = vec![1.0f32; 3 * kv_dim];
        let values = vec![2.0f32; 3 * kv_dim];
        cache.append(&keys, &values);
        assert_eq!(cache.len(), 3);

        // Append 2 more
        let keys2 = vec![3.0f32; 2 * kv_dim];
        let values2 = vec![4.0f32; 2 * kv_dim];
        cache.append(&keys2, &values2);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn key_value_at_retrieval() {
        let mut cache = KvCache::new(2, 4, 100);

        // Position 0: keys = [1,2,3,4, 5,6,7,8] (head0=[1,2,3,4], head1=[5,6,7,8])
        let keys0: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let values0: Vec<f32> = (11..=18).map(|x| x as f32).collect();
        cache.append(&keys0, &values0);

        // Position 1
        let keys1: Vec<f32> = (21..=28).map(|x| x as f32).collect();
        let values1: Vec<f32> = (31..=38).map(|x| x as f32).collect();
        cache.append(&keys1, &values1);

        // Check pos 0, head 0
        assert_eq!(cache.key_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        // Check pos 0, head 1
        assert_eq!(cache.key_at(0, 1), &[5.0, 6.0, 7.0, 8.0]);
        // Check pos 1, head 0
        assert_eq!(cache.key_at(1, 0), &[21.0, 22.0, 23.0, 24.0]);

        // Values
        assert_eq!(cache.value_at(0, 0), &[11.0, 12.0, 13.0, 14.0]);
        assert_eq!(cache.value_at(1, 1), &[35.0, 36.0, 37.0, 38.0]);
    }

    #[test]
    fn keys_values_slices() {
        let mut cache = KvCache::new(1, 2, 100);
        let keys = vec![1.0, 2.0, 3.0, 4.0]; // 2 positions × kv_dim=2
        let values = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&keys, &values);

        assert_eq!(cache.keys(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.values(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn clear_resets() {
        let mut cache = KvCache::new(2, 4, 100);
        let kv_dim = cache.kv_dim();
        cache.append(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Can reuse after clear
        cache.append(&vec![3.0; kv_dim], &vec![4.0; kv_dim]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn overflow_panics() {
        let mut cache = KvCache::new(1, 2, 3); // max 3 positions
        let kv_dim = cache.kv_dim();
        cache.append(&vec![1.0; 4 * kv_dim], &vec![2.0; 4 * kv_dim]); // 4 > 3
    }

    #[test]
    fn memory_bytes() {
        let cache = KvCache::new(4, 64, 2048);
        // 2 buffers × 2048 × 256 × 4 bytes = 4,194,304
        assert_eq!(cache.memory_bytes(), 2 * 2048 * 256 * 4);
    }

    // -- ModelKvCache tests --

    #[test]
    fn model_cache_construction() {
        let mc = ModelKvCache::new(22, 4, 64, 4096);
        assert_eq!(mc.n_layers(), 22);
        assert_eq!(mc.seq_len(), 0);
    }

    #[test]
    fn model_cache_per_layer() {
        let mut mc = ModelKvCache::new(3, 2, 4, 100);
        let kv_dim = mc.layer(0).kv_dim();

        // Append to layer 0 only
        mc.layer_mut(0).append(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        assert_eq!(mc.layer(0).len(), 1);
        assert_eq!(mc.layer(1).len(), 0);
        assert_eq!(mc.layer(2).len(), 0);
    }

    #[test]
    fn model_cache_clear() {
        let mut mc = ModelKvCache::new(3, 2, 4, 100);
        let kv_dim = mc.layer(0).kv_dim();

        for i in 0..3 {
            mc.layer_mut(i).append(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        }
        assert_eq!(mc.layer(0).len(), 1);

        mc.clear();
        for i in 0..3 {
            assert_eq!(mc.layer(i).len(), 0);
        }
    }

    #[test]
    fn model_cache_memory() {
        let mc = ModelKvCache::new(22, 4, 64, 2048);
        // 22 layers × 2 buffers × 2048 × 256 × 4 bytes
        let expected = 22 * 2 * 2048 * 256 * 4;
        assert_eq!(mc.memory_bytes(), expected);
        // ~92MB for a 2B model at 2K context — reasonable
    }

    #[test]
    fn debug_format_cache() {
        let cache = KvCache::new(4, 64, 2048);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("KvCache"));
        assert!(debug.contains("0/2048"));
    }

    #[test]
    fn debug_format_model_cache() {
        let mc = ModelKvCache::new(22, 4, 64, 2048);
        let debug = format!("{:?}", mc);
        assert!(debug.contains("ModelKvCache"));
        assert!(debug.contains("layers=22"));
    }

    #[test]
    fn incremental_generation_pattern() {
        // Simulate the actual generation pattern:
        // 1. Prefill: append all prompt tokens at once
        // 2. Decode: append one token at a time
        let mut cache = KvCache::new(2, 4, 100);
        let kv_dim = cache.kv_dim();

        // Prefill 5 prompt tokens
        let prompt_keys = vec![1.0f32; 5 * kv_dim];
        let prompt_values = vec![2.0f32; 5 * kv_dim];
        cache.append(&prompt_keys, &prompt_values);
        assert_eq!(cache.len(), 5);

        // Generate 3 tokens one at a time
        for step in 0..3 {
            let new_key = vec![(10 + step) as f32; kv_dim];
            let new_value = vec![(20 + step) as f32; kv_dim];
            cache.append(&new_key, &new_value);
        }
        assert_eq!(cache.len(), 8); // 5 prompt + 3 generated

        // Can retrieve any position
        assert_eq!(cache.key_at(0, 0), &[1.0; 4]); // prompt
        assert_eq!(cache.key_at(5, 0), &[10.0; 4]); // first generated
        assert_eq!(cache.key_at(7, 0), &[12.0; 4]); // third generated
    }
}
