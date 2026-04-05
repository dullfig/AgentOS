//! Cron trigger — fires according to a cron expression.

use chrono::Utc;
use cron::Schedule;
use std::str::FromStr;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tracing::debug;

use crate::runtime::{TriggerEvent, TriggerPayload};
use crate::TriggerError;

/// Run a cron trigger. Parses the expression, sleeps until the next fire time, repeats.
pub async fn run(
    name: String,
    target: String,
    expression: String,
    tx: mpsc::Sender<TriggerEvent>,
) -> Result<(), TriggerError> {
    // cron crate expects 6-field expressions (sec min hr dom mon dow)
    // Users write 5-field (min hr dom mon dow) — prepend "0" for seconds
    let full_expr = if expression.split_whitespace().count() == 5 {
        format!("0 {expression}")
    } else {
        expression.clone()
    };

    let schedule = Schedule::from_str(&full_expr)
        .map_err(|e| TriggerError::CronParse(format!("Invalid cron '{expression}': {e}")))?;

    debug!("Cron trigger '{name}' scheduled: {expression}");

    loop {
        let now = Utc::now();
        let next = schedule.upcoming(Utc).next();

        let Some(next_fire) = next else {
            // No future fire times — schedule is exhausted (shouldn't happen with standard crons)
            break;
        };

        let wait = (next_fire - now).to_std().unwrap_or(Duration::from_secs(1));
        sleep(wait).await;

        debug!("Cron trigger '{name}' fired at {}", Utc::now());

        let event = TriggerEvent {
            trigger_name: name.clone(),
            target: target.clone(),
            payload: TriggerPayload::Tick,
        };

        if tx.send(event).await.is_err() {
            break;
        }
    }

    Ok(())
}
