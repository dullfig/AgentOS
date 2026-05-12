//! In-memory LRU idempotency cache for `POST /v1/messages`.
//!
//! Per the v1 API contract (`project_agentos_api_contract.md`):
//!
//! > `idempotency_key` is required. Clients generate a UUID per user
//! > message. AgentOS maintains a 24h cache keyed by
//! > `(service_token, idempotency_key)`. Same key + same body →
//! > replay cached SSE stream. Same key + different body →
//! > `409 idempotency_conflict`. Storage: in-memory LRU is fine for
//! > v1 since retries beyond 24h are rare.
//!
//! Why this exists: network retries on a slow LLM call are common
//! (axum connection drops, mid-stream client disconnects, transient
//! errors). Without idempotency, a retry produces a second Bob
//! response: double cost, double display, possibly conflicting
//! state if Bob's reply is non-deterministic.
//!
//! Design decisions:
//!
//! - **In-flight concurrent same-key retries** → reject as
//!   `409 idempotency_conflict` saying "in-flight". The contract is
//!   silent on this case; the common case (sequential retry after a
//!   timeout) hits the *completed* entry and replays cleanly, so
//!   fail-loud on concurrent retries is safe and simple.
//! - **Body hash for conflict detection** → SHA-256 of the parsed
//!   request body re-serialized via `serde_json::to_vec`. Serde is
//!   deterministic for ordered structs, so two equal
//!   `PostMessagesRequest` values produce equal bytes (and thus
//!   equal hashes).
//! - **Storage** → `DashMap<Key, CachedEntry>`. Concurrent-friendly,
//!   no Mutex contention.
//! - **Token storage** → bearer token gets SHA-256'd to a `u64`;
//!   the cache key carries the hash rather than the full string.
//! - **TTL** → 24h per contract. Background tokio task sweeps every
//!   5min by default; configurable for tests.
//! - **No LRU cap** → TTL alone bounds size at realistic traffic
//!   levels. A cap can be added without breaking the API.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use sha2::{Digest, Sha256};

use crate::sse::{AckPayload, DonePayload};

/// Default time-to-live: 24h per the API contract.
pub const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Default sweep interval. Five minutes balances "memory frees up
/// reasonably fast" against "background task overhead is trivial."
pub const DEFAULT_SWEEP_INTERVAL: Duration = Duration::from_secs(300);

/// Cache lookup key: `(service_token_hash, idempotency_key)`.
///
/// Token hash rather than raw token: avoids carrying bearer strings
/// in the cache, and u64 is cheap to hash/compare. Two distinct
/// tokens with the same SHA-256 prefix collision are astronomically
/// unlikely; for v1 this is fine.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Key {
    pub token_hash: u64,
    pub idempotency_key: String,
}

/// One cache entry. Holds a body-hash for conflict detection plus
/// either the in-flight marker or the completed event payloads.
#[derive(Clone, Debug)]
pub struct CachedEntry {
    pub body_hash: [u8; 32],
    pub state: EntryState,
    pub created_at: Instant,
}

#[derive(Clone, Debug)]
pub enum EntryState {
    /// Stream is in progress; another caller is producing the
    /// events. Concurrent same-key requests see this and 409
    /// "in-flight".
    InFlight,
    /// Stream completed; replayable.
    Completed {
        ack: AckPayload,
        chunks: Vec<String>,
        done: DonePayload,
    },
}

/// Result of probing the cache. `Miss` means the caller claimed the
/// slot (an `InFlight` entry was inserted) and must subsequently
/// call `commit` or `release`. `Replay` means the caller should
/// emit the stored events instead of running the agent.
pub enum LookupResult {
    /// No prior entry; slot claimed; proceed with normal handling.
    Miss,
    /// Prior matching entry; replay these events.
    Replay {
        ack: AckPayload,
        chunks: Vec<String>,
        done: DonePayload,
    },
    /// Same key, different body. Return 409 idempotency_conflict.
    Conflict,
    /// Same key, in-flight. Return 409 idempotency_conflict (with
    /// in-flight messaging).
    InFlight,
}

/// 24h LRU-by-TTL idempotency cache.
///
/// One instance per running server; held by `ServerState` as an
/// `Arc<IdempotencyCache>`.
pub struct IdempotencyCache {
    entries: DashMap<Key, CachedEntry>,
    ttl: Duration,
}

impl IdempotencyCache {
    /// Create a new cache with the contract-mandated 24h TTL.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            entries: DashMap::new(),
            ttl: DEFAULT_TTL,
        })
    }

    /// Create a cache with a custom TTL (testing).
    pub fn with_ttl(ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            entries: DashMap::new(),
            ttl,
        })
    }

    /// Build a `Key` from a service token + idempotency_key.
    pub fn key(auth_token: &str, idempotency_key: &str) -> Key {
        Key {
            token_hash: hash_token(auth_token),
            idempotency_key: idempotency_key.to_string(),
        }
    }

    /// SHA-256 of arbitrary bytes. Used to fingerprint request bodies
    /// for conflict detection.
    pub fn body_hash(bytes: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().into()
    }

    /// Probe the cache; if no entry exists for this key, insert an
    /// `InFlight` placeholder atomically and return `Miss`. Otherwise
    /// return the appropriate result (Replay / Conflict / InFlight).
    ///
    /// The atomicity comes from `DashMap::entry`: the in-flight
    /// insertion happens inside the same locked-slot operation as
    /// the probe, so two concurrent same-key requests can't both
    /// see `Miss`.
    pub fn lookup_or_claim(&self, key: Key, body_hash: [u8; 32]) -> LookupResult {
        use dashmap::mapref::entry::Entry;

        match self.entries.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(CachedEntry {
                    body_hash,
                    state: EntryState::InFlight,
                    created_at: Instant::now(),
                });
                LookupResult::Miss
            }
            Entry::Occupied(slot) => {
                let entry = slot.get();
                // Expired? Pretend the entry doesn't exist. Slot will
                // be claimed on this path; the next sweep cleans up
                // anything we leave behind on the unhappy code paths.
                if entry.created_at.elapsed() > self.ttl {
                    slot.replace_entry(CachedEntry {
                        body_hash,
                        state: EntryState::InFlight,
                        created_at: Instant::now(),
                    });
                    return LookupResult::Miss;
                }
                if entry.body_hash != body_hash {
                    return LookupResult::Conflict;
                }
                match &entry.state {
                    EntryState::InFlight => LookupResult::InFlight,
                    EntryState::Completed { ack, chunks, done } => LookupResult::Replay {
                        ack: ack.clone(),
                        chunks: chunks.clone(),
                        done: done.clone(),
                    },
                }
            }
        }
    }

    /// Replace an in-flight entry with the completed payloads.
    /// Called by the handler when the SSE stream finishes successfully.
    pub fn commit(
        &self,
        key: &Key,
        ack: AckPayload,
        chunks: Vec<String>,
        done: DonePayload,
    ) {
        if let Some(mut entry) = self.entries.get_mut(key) {
            entry.state = EntryState::Completed { ack, chunks, done };
        }
        // If the entry isn't there (it shouldn't happen, but defense
        // in depth) we drop the commit silently. Next retry will
        // re-execute, which is the safe failure mode.
    }

    /// Drop an in-flight entry so subsequent retries can proceed.
    /// Called by the handler on stream timeout or unrecoverable error.
    pub fn release(&self, key: &Key) {
        self.entries.remove(key);
    }

    /// Number of entries currently in the cache. Useful for tests
    /// and observability.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all entries older than the TTL. Returns the count
    /// removed.
    pub fn sweep(&self) -> usize {
        let ttl = self.ttl;
        let now = Instant::now();
        let mut victims = Vec::new();
        for entry in self.entries.iter() {
            if now.duration_since(entry.value().created_at) > ttl {
                victims.push(entry.key().clone());
            }
        }
        let n = victims.len();
        for k in victims {
            self.entries.remove(&k);
        }
        n
    }

    /// Spawn a tokio task that periodically calls `sweep`. Drop the
    /// returned `JoinHandle` to stop sweeping; the cache itself
    /// remains usable.
    pub fn spawn_sweeper(self: Arc<Self>, interval: Duration) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // First tick fires immediately; skip it so we don't sweep
            // an empty cache at startup.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let removed = self.sweep();
                if removed > 0 {
                    tracing::debug!(removed, "idempotency cache sweep");
                }
            }
        })
    }
}

fn hash_token(token: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    let digest = h.finalize();
    u64::from_be_bytes(digest[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::DoneMetadata;

    fn ack(req_id: &str, conv_id: &str) -> AckPayload {
        AckPayload {
            request_id: req_id.into(),
            conversation_id: conv_id.into(),
        }
    }

    fn done(req_id: &str, conv_id: &str) -> DonePayload {
        DonePayload {
            conversation_id: conv_id.into(),
            turn_id: "turn-1".into(),
            request_id: req_id.into(),
            silent: false,
            metadata: DoneMetadata::default(),
        }
    }

    #[test]
    fn miss_then_replay() {
        let cache = IdempotencyCache::new();
        let k = IdempotencyCache::key("token", "idem-1");
        let body = IdempotencyCache::body_hash(b"hello");

        // First lookup: Miss + slot claimed.
        match cache.lookup_or_claim(k.clone(), body) {
            LookupResult::Miss => {}
            other => panic!("expected Miss, got {other:?}"),
        }
        assert_eq!(cache.len(), 1);

        // Commit some events.
        cache.commit(
            &k,
            ack("r1", "c1"),
            vec!["hello".into(), " world".into()],
            done("r1", "c1"),
        );

        // Second lookup with same body: Replay.
        match cache.lookup_or_claim(k.clone(), body) {
            LookupResult::Replay { ack: a, chunks, done: d } => {
                assert_eq!(a.request_id, "r1");
                assert_eq!(chunks, vec!["hello", " world"]);
                assert_eq!(d.request_id, "r1");
            }
            other => panic!("expected Replay, got {other:?}"),
        }
    }

    #[test]
    fn conflict_on_different_body() {
        let cache = IdempotencyCache::new();
        let k = IdempotencyCache::key("token", "idem-2");
        let body_a = IdempotencyCache::body_hash(b"hello");
        let body_b = IdempotencyCache::body_hash(b"different");

        cache.lookup_or_claim(k.clone(), body_a);
        cache.commit(&k, ack("r1", "c1"), vec![], done("r1", "c1"));

        match cache.lookup_or_claim(k, body_b) {
            LookupResult::Conflict => {}
            other => panic!("expected Conflict, got {other:?}"),
        }
    }

    #[test]
    fn inflight_on_repeat_before_commit() {
        let cache = IdempotencyCache::new();
        let k = IdempotencyCache::key("token", "idem-3");
        let body = IdempotencyCache::body_hash(b"hello");

        cache.lookup_or_claim(k.clone(), body); // claim
        match cache.lookup_or_claim(k, body) {
            LookupResult::InFlight => {}
            other => panic!("expected InFlight, got {other:?}"),
        }
    }

    #[test]
    fn release_lets_retry_proceed() {
        let cache = IdempotencyCache::new();
        let k = IdempotencyCache::key("token", "idem-4");
        let body = IdempotencyCache::body_hash(b"hello");

        cache.lookup_or_claim(k.clone(), body);
        cache.release(&k);

        match cache.lookup_or_claim(k, body) {
            LookupResult::Miss => {}
            other => panic!("expected Miss after release, got {other:?}"),
        }
    }

    #[test]
    fn sweep_drops_expired_entries() {
        let cache = IdempotencyCache::with_ttl(Duration::from_millis(50));
        let k = IdempotencyCache::key("token", "idem-5");
        let body = IdempotencyCache::body_hash(b"hello");
        cache.lookup_or_claim(k, body);
        assert_eq!(cache.len(), 1);

        std::thread::sleep(Duration::from_millis(80));
        let removed = cache.sweep();
        assert_eq!(removed, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn expired_entry_reclaimed_on_next_lookup() {
        let cache = IdempotencyCache::with_ttl(Duration::from_millis(50));
        let k = IdempotencyCache::key("token", "idem-6");
        let body_a = IdempotencyCache::body_hash(b"hello");
        let body_b = IdempotencyCache::body_hash(b"different");

        cache.lookup_or_claim(k.clone(), body_a);
        cache.commit(&k, ack("r1", "c1"), vec![], done("r1", "c1"));

        std::thread::sleep(Duration::from_millis(80));

        // After TTL, a same-key request with a different body should
        // succeed (the expired entry is reclaimed), not raise Conflict.
        match cache.lookup_or_claim(k, body_b) {
            LookupResult::Miss => {}
            other => panic!("expected Miss (expired reclaim), got {other:?}"),
        }
    }

    #[test]
    fn different_tokens_dont_collide() {
        let cache = IdempotencyCache::new();
        let body = IdempotencyCache::body_hash(b"hello");

        // Same idempotency_key, different bearer tokens → different slots.
        let k_a = IdempotencyCache::key("token-a", "idem-7");
        let k_b = IdempotencyCache::key("token-b", "idem-7");

        cache.lookup_or_claim(k_a, body);
        match cache.lookup_or_claim(k_b, body) {
            LookupResult::Miss => {}
            other => panic!("expected Miss (different token), got {other:?}"),
        }
    }

    impl std::fmt::Debug for LookupResult {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                LookupResult::Miss => write!(f, "Miss"),
                LookupResult::Replay { .. } => write!(f, "Replay"),
                LookupResult::Conflict => write!(f, "Conflict"),
                LookupResult::InFlight => write!(f, "InFlight"),
            }
        }
    }
}
