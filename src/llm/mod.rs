//! LLM module — re-exports from agentos-llm crate + local handler.
//!
//! The pool, client, and types live in the agentos-llm crate.
//! The handler (pipeline listener) stays here because it depends on Librarian.

pub mod handler;

// Re-export everything from the crate
pub use agentos_llm::*;
