//! The Control Room — ratatui TUI presentation layer.
//!
//! Renders pipeline activity as a terminal dashboard. Same data as the
//! pipeline, different view — dual rendering. Read-only: the TUI never
//! mutates pipeline state. No LLM needed — pure Rust rendering.
//!
//! ## Architecture (TEA)
//!
//! Model (`TuiApp`) + Update (message handler) + View (render).
//! Immediate mode, no retained widget state. View models decouple
//! kernel from ratatui — lightweight copies, no kernel references
//! held across frames.

pub mod app;
pub mod commands;
pub mod context_tree;
pub mod dashboard;
pub mod event;
pub mod input;
pub mod layout;
pub mod markdown;
pub mod render;
pub mod runner;
pub mod segment_detail;
