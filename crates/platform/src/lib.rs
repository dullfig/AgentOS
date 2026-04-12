//! agentos-platform — Agent orchestration platform.
//!
//! # Purpose
//!
//! The multi-tenant agent runtime layer. Provides:
//! - **Hierarchical addressing** (`bob[alice].calendar`) — agents as parameterized instances
//! - **Instance registry** — lazy materialization on first message, VMM-style eviction
//! - **Materialization-on-routing** — `send_to(address)` creates anything missing along the path
//! - **Lifetime policies** — UntilIdle, UntilTaskComplete, Pinned, Ephemeral
//! - **Namespace security** — agents can only reach addresses in their own namespace
//!
//! # Architecture
//!
//! The platform sits between the kernel (durable state) and the frontends (TUI, GUI, server).
//! It does NOT contain the pipeline orchestrator (which remains in the root crate for now
//! due to circular dependencies with the buffer system). Instead, it provides the addressing
//! and instance management primitives that the pipeline imports.
//!
//! The [`router::Runtime`] trait is the seam: the pipeline implements it, the platform calls it.
//!
//! # What's Built (41 tests)
//!
//! - [`address::Address`] — full hierarchical grammar with bracket keys, cache composition (`+`), ephemeral detection
//! - [`registry::InstanceRegistry`] — VMM-tiered instance tracking, lifetime policies, idle eviction, parent-child
//! - [`router::Router`] — `send_to` with materialization-on-routing, namespace enforcement, shard pattern expansion
//! - [`router::Runtime`] trait — decouples platform from pipeline
//!
//! # Missing Pieces (TODO)
//!
//! ## Wiring (connects platform to the existing pipeline)
//! - [ ] `Runtime` trait impl on `AgentPipeline` — the actual bridge, makes `send_to` real
//! - [ ] Trigger `send_to:` / `message:` YAML fields + template variable expansion from events
//! - [ ] KV shard loading at materialization time (calls memex/cortex to load shards)
//!
//! ## Concurrency (production-grade)
//! - [ ] `Router` behind `Arc<Mutex<>>` or concurrent map — parallel message routing
//! - [ ] Per-address materialization mutex — prevent double-materialize on simultaneous first-messages
//!
//! ## Lifecycle
//! - [ ] Periodic eviction task — background tokio timer calling `router.evict_idle()`
//! - [ ] Parent-child cascade on kill (optional, policy-driven)
//! - [ ] Registry persistence — survive process restart, replay from sled/WAL
//!
//! ## Buffers
//! - [ ] Buffer creation/routing within instances — `envelope.buffer` field routed to per-channel buffers
//!
//! ## Observability
//! - [ ] Pipeline event emission — `instance.spawned`, `instance.evicted`, `instance.killed`
//! - [ ] Admin commands — `/instances` in TUI, list/inspect/kill/force-materialize
//! - [ ] Metrics — instance count gauges, materialization latency, eviction counters
//!
//! ## Performance
//! - [ ] Organism template cache — don't re-read YAML from disk on every materialization
//!
//! ## Testing
//! - [ ] Integration tests against real kernel + pipeline (not just MockRuntime)

pub mod address;
pub mod buffers;
pub mod concurrent;
pub mod events;
pub mod registry;
pub mod router;
pub mod template;
