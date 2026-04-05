//! KV cache store — sled-backed per-user persistence.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// Errors from KV cache store operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sled error: {0}")]
    Sled(#[from] sled::Error),

    #[error("serialization error: {0}")]
    Serde(String),

    #[error("user not found: {0}")]
    UserNotFound(String),
}

/// A single KV cache entry — one layer's worth of compressed key/value tensors
/// for a range of sequence positions.
///
/// This is the unit of storage. Engram produces these (via TurboQuant compression)
/// and the store persists them keyed by (user_id, layer, sequence_start).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheEntry {
    /// Transformer layer index (0..n_layers).
    pub layer: u32,
    /// Starting sequence position for this entry.
    pub seq_start: u64,
    /// Number of tokens covered by this entry.
    pub seq_len: u32,
    /// Compressed key tensor bytes (TurboQuant or PolarQuant format).
    pub key_data: Vec<u8>,
    /// Compressed value tensor bytes.
    pub value_data: Vec<u8>,
    /// Compression format tag (for forward compatibility).
    pub format: CacheFormat,
    /// Timestamp when this entry was created (unix millis).
    pub created_at: u64,
}

/// Compression format for stored KV cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheFormat {
    /// TurboQuant 3-bit (12x compression, from engram).
    TurboQuant,
    /// PolarQuant 3-bit (engram's current implementation).
    PolarQuant,
    /// Raw float32 (uncompressed, for debugging).
    Float32,
}

/// Per-user statistics.
#[derive(Debug, Clone)]
pub struct KvCacheStats {
    /// Total entries stored for this user.
    pub entry_count: u64,
    /// Total bytes on disk.
    pub bytes_on_disk: u64,
    /// Sequence range: (min_seq_start, max_seq_start + seq_len).
    pub seq_range: (u64, u64),
    /// Number of layers with data.
    pub layers: u32,
}

/// Per-user KV cache store backed by sled.
///
/// Key layout in sled:
///   `{user_id}:{layer:04}:{seq_start:012}` → serialized KvCacheEntry
///
/// This gives us:
/// - All entries for a user are contiguous (prefix scan by user_id)
/// - Within a user, entries are ordered by layer then sequence position
/// - Range queries are efficient (sled's B-tree ordering)
pub struct KvCacheStore {
    db: sled::Db,
}

impl KvCacheStore {
    /// Open or create a KV cache store at the given path.
    pub fn open(path: &std::path::Path) -> Result<Self, StoreError> {
        let db = sled::open(path)?;
        debug!("KV cache store opened at {}", path.display());
        Ok(Self { db })
    }

    /// Open an in-memory store (for testing).
    pub fn in_memory() -> Result<Self, StoreError> {
        let config = sled::Config::new().temporary(true);
        let db = config.open()?;
        Ok(Self { db })
    }

    // ── Write operations ──

    /// Append new KV cache entries for a user.
    /// Typically called after each inference turn.
    pub fn append(&self, user_id: &str, entries: &[KvCacheEntry]) -> Result<(), StoreError> {
        let batch = entries.iter().try_fold(sled::Batch::default(), |mut batch, entry| {
            let key = format_key(user_id, entry.layer, entry.seq_start);
            let value = bincode_serialize(entry)?;
            batch.insert(key.as_bytes(), value);
            Ok::<_, StoreError>(batch)
        })?;

        self.db.apply_batch(batch)?;
        debug!("Appended {} entries for user '{user_id}'", entries.len());
        Ok(())
    }

    /// Flush all pending writes to disk. Call when a user goes idle.
    pub fn flush(&self, _user_id: &str) -> Result<(), StoreError> {
        self.db.flush()?;
        Ok(())
    }

    // ── Read operations ──

    /// Load the most recent N entries for a user, across all layers.
    /// Returns entries ordered by (layer, seq_start).
    pub fn load_recent(
        &self,
        user_id: &str,
        max_entries: usize,
    ) -> Result<Vec<KvCacheEntry>, StoreError> {
        let prefix = format!("{user_id}:");
        let entries: Vec<KvCacheEntry> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .rev() // most recent first (highest seq_start)
            .take(max_entries)
            .filter_map(|r| r.ok())
            .filter_map(|(_k, v)| bincode_deserialize(&v).ok())
            .collect();

        debug!("Loaded {} entries for user '{user_id}'", entries.len());
        Ok(entries)
    }

    /// Load all entries for a user for a specific layer.
    pub fn load_layer(
        &self,
        user_id: &str,
        layer: u32,
    ) -> Result<Vec<KvCacheEntry>, StoreError> {
        let prefix = format!("{user_id}:{layer:04}:");
        let entries: Vec<KvCacheEntry> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|r| r.ok())
            .filter_map(|(_k, v)| bincode_deserialize(&v).ok())
            .collect();

        Ok(entries)
    }

    /// Load entries for a user within a sequence range, across all layers.
    pub fn load_range(
        &self,
        user_id: &str,
        seq_from: u64,
        seq_to: u64,
    ) -> Result<Vec<KvCacheEntry>, StoreError> {
        let prefix = format!("{user_id}:");
        let entries: Vec<KvCacheEntry> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|r| r.ok())
            .filter_map(|(_k, v)| bincode_deserialize(&v).ok())
            .filter(|e: &KvCacheEntry| e.seq_start >= seq_from && e.seq_start < seq_to)
            .collect();

        Ok(entries)
    }

    // ── Lifecycle operations ──

    /// Consolidate old entries for a user — merge adjacent entries within
    /// each layer to reduce entry count. Called during engram's L2→L3 compaction.
    ///
    /// `keep_recent` entries are left untouched (still being attended to).
    /// Older entries are merged layer-by-layer.
    pub fn consolidate(
        &self,
        user_id: &str,
        keep_recent: usize,
    ) -> Result<u64, StoreError> {
        let prefix = format!("{user_id}:");
        let all_keys: Vec<sled::IVec> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|r| r.ok())
            .map(|(k, _)| k)
            .collect();

        if all_keys.len() <= keep_recent {
            return Ok(0);
        }

        // Remove the oldest entries beyond keep_recent
        let to_remove = all_keys.len() - keep_recent;
        let mut removed = 0u64;
        for key in all_keys.iter().take(to_remove) {
            self.db.remove(key)?;
            removed += 1;
        }

        debug!("Consolidated user '{user_id}': removed {removed} old entries, kept {keep_recent}");
        Ok(removed)
    }

    /// Evict a user's entire cache. Called when the user is permanently removed
    /// or when storage pressure requires it.
    pub fn evict(&self, user_id: &str) -> Result<u64, StoreError> {
        let prefix = format!("{user_id}:");
        let keys: Vec<sled::IVec> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|r| r.ok())
            .map(|(k, _)| k)
            .collect();

        let count = keys.len() as u64;
        let mut batch = sled::Batch::default();
        for key in &keys {
            batch.remove(key.clone());
        }
        self.db.apply_batch(batch)?;

        debug!("Evicted {count} entries for user '{user_id}'");
        Ok(count)
    }

    /// Get statistics for a user's cache.
    pub fn stats(&self, user_id: &str) -> Result<KvCacheStats, StoreError> {
        let prefix = format!("{user_id}:");
        let mut entry_count = 0u64;
        let mut bytes_on_disk = 0u64;
        let mut min_seq = u64::MAX;
        let mut max_seq = 0u64;
        let mut layers = std::collections::HashSet::new();

        for item in self.db.scan_prefix(prefix.as_bytes()) {
            let (_k, v) = item?;
            entry_count += 1;
            bytes_on_disk += v.len() as u64;

            if let Ok(entry) = bincode_deserialize::<KvCacheEntry>(&v) {
                min_seq = min_seq.min(entry.seq_start);
                max_seq = max_seq.max(entry.seq_start + entry.seq_len as u64);
                layers.insert(entry.layer);
            }
        }

        Ok(KvCacheStats {
            entry_count,
            bytes_on_disk,
            seq_range: if entry_count > 0 {
                (min_seq, max_seq)
            } else {
                (0, 0)
            },
            layers: layers.len() as u32,
        })
    }

    /// List all user IDs that have cached data.
    pub fn list_users(&self) -> Result<Vec<String>, StoreError> {
        let mut users = std::collections::BTreeSet::new();
        for item in self.db.iter() {
            let (k, _) = item?;
            if let Ok(key_str) = std::str::from_utf8(&k) {
                if let Some(user_id) = key_str.split(':').next() {
                    users.insert(user_id.to_string());
                }
            }
        }
        Ok(users.into_iter().collect())
    }

    /// Total size of the store on disk (approximate).
    pub fn total_size(&self) -> u64 {
        self.db.size_on_disk().unwrap_or(0)
    }
}

// ── Key formatting ──

/// Format a sled key: `{user_id}:{layer:04}:{seq_start:012}`
fn format_key(user_id: &str, layer: u32, seq_start: u64) -> String {
    format!("{user_id}:{layer:04}:{seq_start:012}")
}

// ── Serialization (using serde_json for now, swap to bincode for production) ──

fn bincode_serialize<T: Serialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    serde_json::to_vec(value).map_err(|e| StoreError::Serde(e.to_string()))
}

fn bincode_deserialize<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, StoreError> {
    serde_json::from_slice(bytes).map_err(|e| StoreError::Serde(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(layer: u32, seq_start: u64) -> KvCacheEntry {
        KvCacheEntry {
            layer,
            seq_start,
            seq_len: 128,
            key_data: vec![0xAA; 64],   // fake compressed KV data
            value_data: vec![0xBB; 64],
            format: CacheFormat::TurboQuant,
            created_at: 1712300000,
        }
    }

    #[test]
    fn open_in_memory() {
        let store = KvCacheStore::in_memory().unwrap();
        assert_eq!(store.list_users().unwrap().len(), 0);
    }

    #[test]
    fn append_and_load() {
        let store = KvCacheStore::in_memory().unwrap();

        let entries = vec![
            sample_entry(0, 0),
            sample_entry(0, 128),
            sample_entry(1, 0),
            sample_entry(1, 128),
        ];

        store.append("user-alice", &entries).unwrap();

        let loaded = store.load_recent("user-alice", 100).unwrap();
        assert_eq!(loaded.len(), 4);
    }

    #[test]
    fn load_layer_filters() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-bob", &[
            sample_entry(0, 0),
            sample_entry(0, 128),
            sample_entry(1, 0),
            sample_entry(2, 0),
        ]).unwrap();

        let layer0 = store.load_layer("user-bob", 0).unwrap();
        assert_eq!(layer0.len(), 2);

        let layer1 = store.load_layer("user-bob", 1).unwrap();
        assert_eq!(layer1.len(), 1);
    }

    #[test]
    fn load_range() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-carol", &[
            sample_entry(0, 0),
            sample_entry(0, 128),
            sample_entry(0, 256),
            sample_entry(0, 384),
        ]).unwrap();

        let range = store.load_range("user-carol", 128, 384).unwrap();
        assert_eq!(range.len(), 2); // seq_start 128 and 256
    }

    #[test]
    fn multi_user_isolation() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-alice", &[sample_entry(0, 0)]).unwrap();
        store.append("user-bob", &[sample_entry(0, 0), sample_entry(0, 128)]).unwrap();

        let alice = store.load_recent("user-alice", 100).unwrap();
        let bob = store.load_recent("user-bob", 100).unwrap();

        assert_eq!(alice.len(), 1);
        assert_eq!(bob.len(), 2);
    }

    #[test]
    fn evict_removes_all() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-alice", &[
            sample_entry(0, 0),
            sample_entry(1, 0),
        ]).unwrap();

        let removed = store.evict("user-alice").unwrap();
        assert_eq!(removed, 2);

        let loaded = store.load_recent("user-alice", 100).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn consolidate_keeps_recent() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-dave", &[
            sample_entry(0, 0),
            sample_entry(0, 128),
            sample_entry(0, 256),
            sample_entry(0, 384),
            sample_entry(0, 512),
        ]).unwrap();

        let removed = store.consolidate("user-dave", 2).unwrap();
        assert_eq!(removed, 3);

        let remaining = store.load_recent("user-dave", 100).unwrap();
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn stats_reports_correctly() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("user-eve", &[
            sample_entry(0, 0),
            sample_entry(0, 128),
            sample_entry(1, 0),
        ]).unwrap();

        let stats = store.stats("user-eve").unwrap();
        assert_eq!(stats.entry_count, 3);
        assert_eq!(stats.layers, 2);
        assert_eq!(stats.seq_range, (0, 256)); // 128 + seq_len(128)
        assert!(stats.bytes_on_disk > 0);
    }

    #[test]
    fn list_users() {
        let store = KvCacheStore::in_memory().unwrap();

        store.append("alice", &[sample_entry(0, 0)]).unwrap();
        store.append("bob", &[sample_entry(0, 0)]).unwrap();
        store.append("carol", &[sample_entry(0, 0)]).unwrap();

        let users = store.list_users().unwrap();
        assert_eq!(users, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn empty_user_returns_empty() {
        let store = KvCacheStore::in_memory().unwrap();
        let loaded = store.load_recent("nobody", 100).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn key_format() {
        assert_eq!(format_key("alice", 0, 0), "alice:0000:000000000000");
        assert_eq!(format_key("alice", 31, 1024), "alice:0031:000000001024");
        assert_eq!(format_key("bob", 5, 999999), "bob:0005:000000999999");
    }

    #[test]
    fn format_roundtrip() {
        let entry = sample_entry(7, 42);
        assert_eq!(entry.format, CacheFormat::TurboQuant);
        assert_eq!(entry.key_data.len(), 64);
        assert_eq!(entry.value_data.len(), 64);
    }
}
