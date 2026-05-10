//! AgentOS bundled IDE — terminal UI + LSP integration.
//!
//! Apps that need a developer-facing IDE on top of the AgentOS
//! platform depend on this crate. Apps that don't (RingHub, headless
//! server deployments) skip it. The root `agentos` library re-exports
//! `agentos-tui::runner::run_tui` so `cargo run` from the workspace
//! root continues to launch the TUI unchanged.
//!
//! Per `project_agentos_topology.md` (2026-05-10), this crate's role
//! is structurally both *an app on the platform* and *the platform's
//! universal toolchain for building extensions* (the IDE).

pub mod lsp;
pub mod tui;
pub mod vdrive;

// Convenience re-exports so callers can `use agentos_tui::run_tui;`
// without spelling the sub-module path.
pub use tui::runner::run_tui;
