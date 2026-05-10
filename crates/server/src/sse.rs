//! SSE event helpers.
//!
//! The contract requires three event types: `ack`, `text`, `done`.
//! Each carries a JSON `data:` payload. This module produces axum
//! [`Event`]s with the right `event:` line and serialized JSON body.

use axum::response::sse::Event;
use serde::Serialize;

/// `event: ack` — sent immediately on request receipt.
#[derive(Serialize)]
pub struct AckPayload {
    pub request_id: String,
    pub conversation_id: String,
}

/// `event: text` — incremental chunk of Bob's reply.
#[derive(Serialize)]
pub struct TextPayload {
    pub chunk: String,
}

/// `event: done` — terminal event with metadata.
#[derive(Serialize)]
pub struct DonePayload {
    pub conversation_id: String,
    pub turn_id: String,
    pub request_id: String,
    pub silent: bool,
    pub metadata: DoneMetadata,
}

/// Observability metadata in the `done` event. Fields beyond the
/// minimum are emitted as-is so RingHub keeps a stable shape; cortex
/// integration will populate them with real values later.
#[derive(Serialize, Default)]
pub struct DoneMetadata {
    pub generation_ms: u64,
    pub model: String,
    /// Memex corpora queried (memex-level identifiers only — never
    /// shard paths containing user IDs; see contract privacy invariant).
    pub memex_corpora_queried: Vec<String>,
}

/// Build an SSE [`Event`] with the given event name and a JSON body.
fn event_with_json<T: Serialize>(name: &str, payload: &T) -> Result<Event, serde_json::Error> {
    let json = serde_json::to_string(payload)?;
    Ok(Event::default().event(name).data(json))
}

pub fn ack_event(payload: &AckPayload) -> Result<Event, serde_json::Error> {
    event_with_json("ack", payload)
}

pub fn text_event(chunk: &str) -> Result<Event, serde_json::Error> {
    event_with_json("text", &TextPayload { chunk: chunk.to_string() })
}

pub fn done_event(payload: &DonePayload) -> Result<Event, serde_json::Error> {
    event_with_json("done", payload)
}
