//! agentos-platform — Agent orchestration platform.
//!
//! # Purpose
//!
//! The multi-tenant agent runtime layer. Provides:
//! - **Hierarchical addressing** (`bob[alice].calendar`) — agents as parameterized instances
//! - **Instance registry** — lazy materialization on first message, VMM-style eviction
//! - **Materialization-on-routing** — `send_to(address)` creates anything missing along the path
//! - **Lifetime policies** — UntilIdle, UntilTaskComplete, Pinned, Ephemeral
//!
//! # Architecture
//!
//! The platform sits between the kernel (durable state) and the frontends (TUI, GUI, server).
//! It does NOT contain the pipeline orchestrator (which remains in the root crate for now
//! due to circular dependencies with the buffer system). Instead, it provides the addressing
//! and instance management primitives that the pipeline imports.
//!
//! Future work: extract the pipeline orchestrator into this crate once the buffer↔pipeline
//! circular dependency is resolved.
//!
//! # Key Types
//!
//! - [`Address`] — hierarchical agent address with `organism[key].buffer` syntax
//! - `InstanceRegistry` — maps addresses to live agent instances (planned)
//! - `Lifetime` — eviction policy for instances (planned)

pub mod address;
pub mod registry;
