//! Coding Agent — the one thing that thinks.
//!
//! Wires Opus into the pipeline as a stateful coding agent. Receives tasks,
//! calls tools through the pipeline, produces results. The agentic loop.
//!
//! ## Architecture
//!
//! - `tools`: ToolPeer → ToolDefinition bridge (JSON schemas for Anthropic API)
//! - `translate`: JSON ↔ XML translation for tool calls/responses
//! - `state`: Per-thread state machine (Ready → AwaitingTools → ...)
//! - `handler`: CodingAgentHandler — the stateful Handler impl
//! - `prompts`: System prompt templates
//! - `ralph`: Ralph Method story decomposition

pub mod handler;
pub mod prompts;
pub mod ralph;
pub mod state;
pub mod tools;
pub mod translate;
