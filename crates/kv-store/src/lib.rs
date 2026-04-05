//! agentos-kv-store — Per-user KV cache persistence for multi-tenant inference.
//!
//! # Purpose
//!
//! Bridges the multi-tenant pipeline (14K+ users) to engram's single-user
//! memory engine. Each user's KV cache (compressed attention state from cortex)
//! is persisted to disk via sled and loaded on demand when their dispatch
//! context becomes active.
//!
//! # Architecture
//!
//! ```text
//! Pipeline dispatch (user X arrives)
//!     → KvCacheStore.load("user-x")         ← pull from sled
//!     → hand to engram as working cache
//!     → cortex runs inference
//!     → engram updates cache
//!     → KvCacheStore.append("user-x", ...)  ← persist new entries
//!     → user goes idle
//!     → KvCacheStore.flush("user-x")        ← ensure durability
//!     → engram releases memory
//! ```
//!
//! # Key Design Points
//!
//! - Engram is agnostic — it doesn't know or care where KV bytes come from
//! - This crate handles the multiplexing: one store, many users
//! - sled backend: embedded, append-optimized, concurrent readers/writers
//! - Hot cache in memory for active users, cold users on disk only
//! - VMM-style tiering: Active (in memory) → Shelved (compressed) → Folded (disk only)

mod store;

pub use store::{KvCacheStore, KvCacheEntry, KvCacheStats, StoreError};
