//! AgentOS — kernel infrastructure for AI coding agents.
//!
//! Builds on rust-pipeline to add durable state (WAL + mmap),
//! security profiles, and organism configuration.

pub mod agent;
pub mod buffer;
pub use agentos_config as config;
pub use agentos_embedding as embedding;
pub use agentos_kernel as kernel;
pub mod librarian;
pub mod lsp;
pub mod llm;
pub use agentos_organism as organism;
pub mod pipeline;
pub mod ports;
pub use agentos_routing as routing;
pub mod security;
pub mod tools;
pub mod treesitter;
pub mod tui;
pub mod vdrive;
pub use agentos_wasm as wasm;
pub mod wit;
