//! agentos-cloud — Cloud GPU provisioning with Rhai-scriptable provider adapters.
//!
//! # Architecture
//!
//! ```text
//! cloud-expert (agent)
//!       │
//!       ▼
//! CloudProvider trait ─── search / provision / status / teardown
//!       │
//!       ▼
//! ScriptedProvider ────── Rhai runtime with http_post/http_get/http_delete
//!       │
//!       ▼
//! Provider scripts ────── runpod.rhai, lambda.rhai, vastai.rhai ...
//! ```
//!
//! Provider scripts live in `~/.agentos/cloud/providers/`.
//! API keys live in `~/.agentos/cloud/keys.yaml`.
//!
//! New providers can be added by dropping a `.rhai` script into the providers
//! directory — no Rust recompilation needed. The `api-expert` agent can write
//! these scripts at runtime.

pub mod provider;
pub mod register;
pub mod runtime;
pub mod types;

#[cfg(test)]
mod tests;

pub use provider::CloudProvider;
pub use register::{register_cloud_endpoint, deregister_cloud_endpoint, RegisteredEndpoint};
pub use runtime::{ProviderRegistry, ScriptedProvider};
pub use types::*;

/// Errors from cloud operations.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    #[error("script error: {0}")]
    ScriptError(String),

    #[error("provision failed: {0}")]
    ProvisionFailed(String),

    #[error("API error: {0}")]
    ApiError(String),
}

/// Default directory for cloud provider scripts.
pub fn providers_dir() -> std::path::PathBuf {
    let base = if cfg!(windows) {
        std::env::var("USERPROFILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
    } else {
        std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
    };
    base.join(".agentos").join("cloud").join("providers")
}

/// Default path for cloud API keys.
pub fn keys_path() -> std::path::PathBuf {
    let base = if cfg!(windows) {
        std::env::var("USERPROFILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
    } else {
        std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
    };
    base.join(".agentos").join("cloud").join("keys.yaml")
}
