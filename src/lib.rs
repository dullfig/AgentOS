//! BestCode — kernel infrastructure for AgentOS.
//!
//! Builds on rust-pipeline to add durable state (WAL + mmap),
//! security profiles, and organism configuration.

pub mod agent;
pub mod embedding;
pub mod kernel;
pub mod librarian;
pub mod llm;
pub mod organism;
pub mod pipeline;
pub mod ports;
pub mod routing;
pub mod security;
pub mod tools;
pub mod treesitter;
pub mod tui;
pub mod wasm;
