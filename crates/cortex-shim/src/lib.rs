//! Cortex shim management client.
//!
//! Talks to cortex's shim-specific HTTP surface — the registry CRUD
//! endpoints and the standalone classification endpoint. This crate
//! does NOT handle `/v1/chat/completions` (that lives in `agentos-llm`'s
//! OpenAI-compat path; the chat-completion request gains shim fields
//! via `agentos_llm::types::ShimAttachment`). The split:
//!
//! - **`agentos-llm`** — fused gate-and-generate via `/v1/chat/completions`.
//!   Where Bob's per-call shim attachment lives.
//! - **`agentos-cortex-shim`** — registry CRUD + standalone classification
//!   via `/v1/shims/...`. Where the shim-expert agent (Step 5) will
//!   register, validate, and retire shims.
//!
//! Wire spec: see `project_cortex_v1_shim_api.md` in the integration
//! memory.

pub mod client;
pub mod embed;
pub mod error;
pub mod manifest;

pub use client::CortexShimClient;
pub use embed::{EmbedClient, EmbedRequest, EmbedResponse, KnownPooling, Pooling};
pub use error::ShimClientError;
pub use manifest::{
    Attachment, InputShape, OutputShape, ShimDecision, ShimManifest, ShimPhase, ShimSummary,
};
