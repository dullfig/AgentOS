//! WASM tool runtime — sandboxed user tools via WebAssembly components.
//!
//! Phase 5: The Immune System. Tools can't do harm because they literally
//! can't express harmful operations — missing WASI capabilities mean the
//! import doesn't exist, not that a policy check blocks it.
//!
//! Architecture:
//! - `runtime.rs` — WasmRuntime engine, component loading, metadata extraction
//! - `error.rs` — WasmError types
//! - `peer.rs` — WasmToolPeer: Handler + ToolPeer bridge (M2)
//! - `capabilities.rs` — WASI capability grants (M3)
//! - `definitions.rs` — WasmToolRegistry: auto-generated ToolDefinitions (M4)
//! - `python_runtime.rs` — PythonRuntime + PythonToolPeer: pure .py tools via shared interpreter

pub mod capabilities;
pub mod definitions;
pub mod error;
pub mod peer;
pub mod python_runtime;
pub mod runtime;
