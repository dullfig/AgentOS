//! Event bus trigger — fires when a matching pipeline event occurs.

use agentos_events::PipelineEvent;
use tokio::sync::{broadcast, mpsc};
use tracing::debug;

use crate::runtime::{TriggerEvent, TriggerPayload};

/// Run an event bus trigger. Subscribes to the pipeline broadcast channel
/// and fires when events match `event_name` (and optionally `from`).
pub async fn run(
    name: String,
    target: String,
    event_name: String,
    from: Option<String>,
    mut event_rx: broadcast::Receiver<PipelineEvent>,
    tx: mpsc::Sender<TriggerEvent>,
) {
    debug!("EventBus trigger '{name}' listening for '{event_name}' from {:?}", from);

    loop {
        match event_rx.recv().await {
            Ok(event) => {
                let event_str = format!("{event:?}");

                // Match on event name
                if !event_str.contains(&event_name) {
                    continue;
                }

                // Optional sender filter
                if let Some(ref sender) = from {
                    if !event_str.contains(sender) {
                        continue;
                    }
                }

                debug!("EventBus trigger '{name}' matched: {event_str}");

                let trigger_event = TriggerEvent {
                    trigger_name: name.clone(),
                    target: target.clone(),
                    payload: TriggerPayload::Event {
                        event_name: event_name.clone(),
                        from: from.clone(),
                    },
                };

                if tx.send(trigger_event).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                debug!("EventBus trigger '{name}' lagged, missed {n} events");
            }
            Err(broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }
}
