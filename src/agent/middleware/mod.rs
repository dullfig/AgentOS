//! Pipeline middleware for cross-cutting agent concerns.
//!
//! Extracted from CodingAgentHandler — these were handler-level checks
//! that belong at the pipeline level as composable middleware.

pub mod debug_gate;
pub mod loop_guard;
pub mod permission_gate;
