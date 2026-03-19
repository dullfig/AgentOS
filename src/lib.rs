//! AgentOS — kernel infrastructure for AI coding agents.
//!
//! Builds on rust-pipeline to add durable state (WAL + mmap),
//! security profiles, and organism configuration.

pub use agentos_agent as agent;
pub mod buffer;
pub use agentos_config as config;
pub use agentos_embedding as embedding;
pub use agentos_kernel as kernel;
pub use agentos_librarian as librarian;
pub mod lsp;
pub mod llm;
pub use agentos_organism as organism;
pub mod pipeline;
pub mod ports;
pub use agentos_routing as routing;
pub mod security;
pub use agentos_wit as wit;

// Tools: re-export from crate, keep test_organism locally (needs pipeline)
pub mod tools {
    pub use agentos_tools::*;
    pub mod test_organism;
}
pub mod treesitter;
pub mod tui;
pub mod vdrive;
pub use agentos_wasm as wasm;
