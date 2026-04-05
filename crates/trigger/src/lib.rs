//! agentos-trigger — Event-driven dispatch for autonomous agent behavior.
//!
//! # Architecture
//!
//! ```text
//! Organism YAML
//!   trigger:
//!     type: file_watch | timer | cron | event | rhai
//!     target: some-agent
//!                │
//!                ▼
//! TriggerRuntime (one per pipeline)
//! ├── FileWatcher   ── notify crate, glob patterns, debounced
//! ├── Timer         ── tokio::time::interval
//! ├── Cron          ── cron expression → next fire time
//! ├── EventBus      ── pipeline broadcast subscriber, filtered
//! └── Rhai          ── script returns fire/no-fire on schedule
//!                │
//!                ▼
//!        dispatch_tx.send(TriggerEvent)
//!                │
//!                ▼
//!        Pipeline routes to target listener
//! ```
//!
//! Triggers are listeners that *generate* messages rather than *handle* them.
//! The runtime spawns one tokio task per trigger and feeds fired events
//! through a channel that the pipeline consumes.

pub mod runtime;
mod sources;

pub use runtime::{TriggerRuntime, TriggerEvent};

/// Errors from trigger operations.
#[derive(Debug, thiserror::Error)]
pub enum TriggerError {
    #[error("file watch error: {0}")]
    FileWatch(String),

    #[error("cron parse error: {0}")]
    CronParse(String),

    #[error("rhai script error: {0}")]
    ScriptError(String),

    #[error("trigger setup error: {0}")]
    Setup(String),
}
