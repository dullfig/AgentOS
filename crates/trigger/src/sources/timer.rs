//! Timer trigger — fires on a fixed interval.

use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::debug;

use crate::runtime::{TriggerEvent, TriggerPayload};

/// Run a timer trigger. Fires every `interval_secs` seconds, forever.
pub async fn run(
    name: String,
    target: String,
    interval_secs: u64,
    tx: mpsc::Sender<TriggerEvent>,
) {
    let mut tick = interval(Duration::from_secs(interval_secs));
    // Skip the immediate first tick
    tick.tick().await;

    loop {
        tick.tick().await;
        debug!("Timer trigger '{name}' fired");

        let event = TriggerEvent {
            trigger_name: name.clone(),
            target: target.clone(),
            payload: TriggerPayload::Tick,
        };

        if tx.send(event).await.is_err() {
            // Channel closed — pipeline shutting down
            break;
        }
    }
}
