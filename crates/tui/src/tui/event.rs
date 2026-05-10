//! TUI event loop — merges input, pipeline, tick, and render events.
//!
//! Spawns a tokio task that multiplexes:
//! - crossterm keyboard/mouse events
//! - pipeline broadcast events
//! - tick interval (4Hz — refresh from kernel)
//! - render interval (30fps — draw frame)
//!
//! All events flow through a single mpsc channel as TuiMessages.

use crossterm::event::KeyEvent;
use agentos_pipeline::events::PipelineEvent;

/// Messages that drive the TUI update loop.
#[derive(Debug, Clone)]
pub enum TuiMessage {
    /// Keyboard input.
    Input(KeyEvent),
    /// Pipeline event (message injected, security blocked, etc.).
    Pipeline(PipelineEvent),
    /// Tick: refresh view models from kernel state.
    Tick,
    /// Render: draw a frame.
    Render,
    /// Quit the TUI.
    Quit,
    /// User submitted a task from the input bar.
    SubmitTask(String),
}
