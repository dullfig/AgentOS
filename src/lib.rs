//! AgentOS — kernel infrastructure for AI coding agents.
//!
//! Builds on rust-pipeline to add durable state (WAL + mmap),
//! security profiles, and organism configuration.

pub use agentos_agent as agent;
pub use agentos_pipeline::buffer;
pub use agentos_config as config;
pub use agentos_embedding as embedding;
pub use agentos_kernel as kernel;
pub use agentos_librarian as librarian;
pub mod lsp;
pub use agentos_llm as llm;
pub use agentos_organism as organism;
pub use agentos_pipeline as pipeline;
pub use agentos_ports as ports;
pub use agentos_routing as routing;
pub use agentos_security as security;
pub use agentos_wit as wit;

// Tools: re-export the crate; test_organism now lives in agentos-pipeline
// (it depends on AgentPipelineBuilder, so it had to move with pipeline).
pub mod tools {
    pub use agentos_tools::*;
    pub use agentos_pipeline::test_organism;
}
pub use agentos_treesitter as treesitter;
pub mod tui;
pub mod vdrive;
pub use agentos_wasm as wasm;
pub use agentos_platform as platform;
pub use agentos_pipeline::runtime_impl;
