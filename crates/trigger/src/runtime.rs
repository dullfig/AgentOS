//! Trigger runtime — spawns and manages trigger tasks.

use agentos_events::PipelineEvent;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, warn};

use agentos_organism::{ListenerDef, TriggerSource};

use crate::sources;
use crate::TriggerError;

/// A fired trigger event — sent to the pipeline for dispatch.
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// Name of the trigger listener that fired.
    pub trigger_name: String,
    /// Target listener to receive the generated message.
    pub target: String,
    /// Payload — context about why the trigger fired.
    pub payload: TriggerPayload,
}

/// Context about what caused the trigger to fire.
#[derive(Debug, Clone)]
pub enum TriggerPayload {
    /// File changed — includes the path(s).
    FileChanged { paths: Vec<String> },
    /// Timer or cron tick.
    Tick,
    /// Pipeline event matched.
    Event { event_name: String, from: Option<String> },
    /// Rhai script returned a value.
    Rhai { result: String },
}

/// The trigger runtime — owns all trigger tasks, provides a channel for fired events.
pub struct TriggerRuntime {
    /// Channel for fired trigger events — pipeline reads from rx.
    dispatch_tx: mpsc::Sender<TriggerEvent>,
    dispatch_rx: Option<mpsc::Receiver<TriggerEvent>>,
    /// Handles for spawned tasks (for shutdown).
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl TriggerRuntime {
    /// Create a new runtime. Call `register()` for each trigger, then `take_receiver()`
    /// to get the event channel for the pipeline.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(256);
        Self {
            dispatch_tx: tx,
            dispatch_rx: Some(rx),
            tasks: Vec::new(),
        }
    }

    /// Register a trigger from an organism listener definition.
    /// Spawns the appropriate background task.
    ///
    /// `event_rx` is needed for EventBus triggers — pass the pipeline's broadcast subscriber.
    pub fn register(
        &mut self,
        listener: &ListenerDef,
        event_rx: Option<broadcast::Receiver<PipelineEvent>>,
    ) -> Result<(), TriggerError> {
        let config = listener.trigger.as_ref().ok_or_else(|| {
            TriggerError::Setup(format!("Listener '{}' has no trigger config", listener.name))
        })?;

        let name = listener.name.clone();
        let target = config.target.clone();
        let tx = self.dispatch_tx.clone();

        let task = match &config.source {
            TriggerSource::FileWatch { pattern, debounce_ms } => {
                let pattern = pattern.clone();
                let debounce = *debounce_ms;
                tokio::spawn(async move {
                    if let Err(e) = sources::file_watch::run(name, target, pattern, debounce, tx).await {
                        error!("FileWatch trigger failed: {e}");
                    }
                })
            }

            TriggerSource::Timer { interval_secs } => {
                let interval = *interval_secs;
                tokio::spawn(async move {
                    sources::timer::run(name, target, interval, tx).await;
                })
            }

            TriggerSource::Cron { expression } => {
                let expr = expression.clone();
                tokio::spawn(async move {
                    if let Err(e) = sources::cron::run(name, target, expr, tx).await {
                        error!("Cron trigger failed: {e}");
                    }
                })
            }

            TriggerSource::Event { event_name, from } => {
                let event_name = event_name.clone();
                let from = from.clone();
                let rx = event_rx.ok_or_else(|| {
                    TriggerError::Setup("EventBus trigger requires pipeline event receiver".into())
                })?;
                tokio::spawn(async move {
                    sources::event_bus::run(name, target, event_name, from, rx, tx).await;
                })
            }

            TriggerSource::Webhook { path } => {
                let path = path.clone();
                warn!("Webhook trigger '{}' registered but web server not yet available (path: {})", name, path);
                // Placeholder — will activate when web UI lands
                tokio::spawn(async move {
                    // No-op until web server exists
                    let _ = (name, target, path, tx);
                    tokio::signal::ctrl_c().await.ok();
                })
            }

            TriggerSource::Custom { poll_secs } => {
                let interval = *poll_secs;
                // Custom triggers currently just tick — Rhai variant below adds the script
                tokio::spawn(async move {
                    sources::timer::run(name, target, interval, tx).await;
                })
            }

            TriggerSource::Rhai { script, poll_secs } => {
                let script = script.clone();
                let interval = *poll_secs;
                tokio::spawn(async move {
                    if let Err(e) = sources::rhai_trigger::run(name, target, script, interval, tx).await {
                        error!("Rhai trigger failed: {e}");
                    }
                })
            }
        };

        self.tasks.push(task);
        Ok(())
    }

    /// Take the receiver end of the dispatch channel.
    /// The pipeline calls this once at startup and polls for fired triggers.
    pub fn take_receiver(&mut self) -> Option<mpsc::Receiver<TriggerEvent>> {
        self.dispatch_rx.take()
    }

    /// Number of registered triggers.
    pub fn trigger_count(&self) -> usize {
        self.tasks.len()
    }

    /// Shutdown all trigger tasks.
    pub fn shutdown(&mut self) {
        for task in self.tasks.drain(..) {
            task.abort();
        }
    }
}

impl Drop for TriggerRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}
