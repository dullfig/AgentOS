//! AgentOS — the platform: kernel + agent runtime + multi-tenant
//! primitives + WASM/WIT extension mechanism + HTTP+SSE API.
//!
//! Apps (TUI, RingHub, future siblings) consume this library. The
//! bundled IDE — `agentos-tui` — is a separate crate apps can opt
//! into. Per `project_agentos_topology.md`.

pub use agentos_agent as agent;
pub use agentos_config as config;
pub use agentos_embedding as embedding;
pub use agentos_kernel as kernel;
pub use agentos_librarian as librarian;
pub use agentos_llm as llm;
pub use agentos_organism as organism;
pub use agentos_pipeline as pipeline;
pub use agentos_pipeline::buffer;
pub use agentos_pipeline::runtime_impl;
pub use agentos_platform as platform;
pub use agentos_ports as ports;
pub use agentos_routing as routing;
pub use agentos_security as security;
pub use agentos_treesitter as treesitter;
pub use agentos_wasm as wasm;
pub use agentos_wit as wit;

// Tools: re-export the crate; test_organism still lives in
// agentos-pipeline (it depends on AgentPipelineBuilder).
pub mod tools {
    pub use agentos_pipeline::test_organism;
    pub use agentos_tools::*;
}

// Bundled IDE — apps that want a terminal UI re-export from here so
// they don't have to spell `agentos_tui::*` directly.
pub use agentos_tui as tui;
