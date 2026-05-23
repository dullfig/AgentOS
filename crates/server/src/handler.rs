//! `POST /v1/messages` — chat-bubble Bob endpoint.
//!
//! Wires:
//!  1. Bearer token check
//!  2. Request validation (reject `user_tier="anon"`)
//!  3. Subscribe to pipeline events *before* sending so we don't miss
//!     a fast-returning AgentResponse
//!  4. Materialize-and-deliver via the platform router
//!  5. Look up the buffer thread_id (the platform creates one per
//!     instance.default-buffer pair); filter the event stream on it
//!  6. Stream SSE: `ack` immediately, then a single `text` chunk per
//!     `AgentResponse` event for that thread, terminated by `done`

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agentos_events::PipelineEvent;
use agentos_platform::address::Address;
use agentos_platform::buffers::BufferId;
use agentos_platform::router::Envelope;

use crate::idempotency::{IdempotencyCache, LookupResult};
use crate::metrics;

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;

use futures_core::Stream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sse::{ack_event, done_event, text_event, AckPayload, DoneMetadata, DonePayload};
use crate::state::ServerState;

/// JSON request body per the v1 contract.
///
/// Unknown fields are accepted (forward-compat) — serde's default is
/// to ignore them since `deny_unknown_fields` is not set.
#[derive(Debug, Deserialize, Serialize)]
pub struct PostMessagesRequest {
    pub user_id: String,
    pub user_tier: String,
    pub text: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    pub idempotency_key: String,
}

/// Error envelope per the contract: `{ "error": { code, message, request_id } }`.
#[derive(Serialize)]
struct ErrorBody {
    error: ErrorPayload,
}

#[derive(Serialize)]
struct ErrorPayload {
    code: &'static str,
    message: String,
    request_id: String,
}

/// Errors that prevent the SSE stream from starting. Map to HTTP error
/// responses with a JSON body.
pub struct PreStreamError {
    status: StatusCode,
    code: &'static str,
    message: String,
    request_id: String,
}

impl PreStreamError {
    /// Build an error and record the request-duration metric in one
    /// step. Used at every error-return site in `post_messages` so the
    /// metric is captured regardless of the failure mode. The status
    /// class (4xx → `client_error`, 5xx → `server_error`) is derived
    /// from `status` so callers don't have to think about it.
    fn record(
        started: Instant,
        status: StatusCode,
        code: &'static str,
        message: impl Into<String>,
        request_id: &str,
    ) -> Self {
        let class = if status.is_client_error() {
            metrics::STATUS_CLIENT_ERROR
        } else {
            metrics::STATUS_SERVER_ERROR
        };
        metrics::record_request(class, started.elapsed());
        Self {
            status,
            code,
            message: message.into(),
            request_id: request_id.to_string(),
        }
    }
}

impl IntoResponse for PreStreamError {
    fn into_response(self) -> axum::response::Response {
        let body = ErrorBody {
            error: ErrorPayload {
                code: self.code,
                message: self.message,
                request_id: self.request_id,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

/// Pull the bearer token out of the `Authorization` header.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
}

/// Echo `X-Request-Id` if the client supplied one, otherwise mint a fresh UUID.
fn request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

/// Boxed-and-pinned stream type returned by `post_messages`. Both the
/// live path (agent emits events as it responds) and the replay path
/// (cached events re-emitted) produce `async_stream::stream!` blocks
/// with anonymous types; boxing erases the difference so both paths
/// can return through the same function signature.
pub type EventStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// `POST /v1/messages` handler.
pub async fn post_messages(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<PostMessagesRequest>,
) -> Result<Sse<EventStream>, PreStreamError> {
    let request_id = request_id_from_headers(&headers);
    // Single time-origin used by every error-return and the success path.
    // Records into `agentos_request_duration_seconds` histogram on completion.
    let started = Instant::now();

    // 1. Auth. Constant-time comparison via SHA-256 of both sides
    //    (defeats remote byte-by-byte timing oracles; the hash also
    //    erases length-difference leakage that a raw ct_eq on the raw
    //    bytes would still have via its short-circuit length check).
    match bearer_token(&headers) {
        Some(t) if ct_eq_token(t, &state.auth_token) => {}
        Some(_) => {
            return Err(PreStreamError::record(
                started,
                StatusCode::FORBIDDEN,
                "unauthorized",
                "bearer token did not match",
                &request_id,
            ));
        }
        None => {
            return Err(PreStreamError::record(
                started,
                StatusCode::UNAUTHORIZED,
                "unauthenticated",
                "missing or malformed Authorization header",
                &request_id,
            ));
        }
    }

    // 2. Validate. The contract says "anon" never reaches this endpoint —
    //    AgentOS MUST 400 it.
    if req.user_tier == "anon" {
        return Err(PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "user_tier=anon is not accepted; anon traffic is handled client-side",
            &request_id,
        ));
    }
    if !matches!(req.user_tier.as_str(), "warm" | "member") {
        return Err(PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "user_tier must be one of: warm, member",
            &request_id,
        ));
    }
    if req.text.trim().is_empty() {
        return Err(PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "text must be non-empty",
            &request_id,
        ));
    }
    if req.user_id.trim().is_empty() {
        return Err(PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "user_id must be non-empty",
            &request_id,
        ));
    }
    // user_id must NOT contain address-grammar characters. Without this
    // check `user_id="alice].dm[evil"` produces the address string
    // `bob[alice].dm[evil]` which `Address::parse` accepts as the
    // instance `bob[alice]` plus a buffer segment `dm[evil]` — i.e., a
    // confused-deputy that routes attacker traffic into a different
    // buffer in another user's instance with a different channel type.
    // `+` is also reserved for the cache-composition operator (see
    // [[multi-tier-cache-composition]] memory note).
    //
    // Restrictive positive allowlist: ASCII alphanumeric + `-`, `_`,
    // `:`. Matches what realistic upstream IDs use (UUIDs, integers,
    // namespaced keys) without admitting any address syntax.
    if !is_valid_user_id(&req.user_id) {
        return Err(PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "user_id must be 1-128 chars of [A-Za-z0-9_-:]",
            &request_id,
        ));
    }

    // 2.5. Idempotency. Probe the cache; if the same (token, key) was
    //      seen before, either replay the cached SSE stream (same body)
    //      or 409 (different body / in-flight). Per the v1 API contract.
    let body_hash = match serde_json::to_vec(&req) {
        Ok(bytes) => IdempotencyCache::body_hash(&bytes),
        Err(_) => {
            // Re-serializing the parsed body should not fail, but if it
            // did the safe move is to skip idempotency rather than 500.
            // Zero hash is unique-by-construction (no real body produces
            // it) so a conflict can't false-positive here.
            [0u8; 32]
        }
    };
    let cache_key = IdempotencyCache::key(&state.auth_token, &req.idempotency_key);
    let replay_data = match state.idempotency.lookup_or_claim(cache_key.clone(), body_hash) {
        LookupResult::Miss => {
            metrics::record_idempotency_lookup(metrics::RESULT_MISS);
            None
        }
        LookupResult::Replay { ack, chunks, done } => {
            metrics::record_idempotency_lookup(metrics::RESULT_REPLAY);
            Some((ack, chunks, done))
        }
        LookupResult::Conflict => {
            metrics::record_idempotency_lookup(metrics::RESULT_CONFLICT);
            return Err(PreStreamError::record(
                started,
                StatusCode::CONFLICT,
                "idempotency_conflict",
                "idempotency_key reused with a different request body",
                &request_id,
            ));
        }
        LookupResult::InFlight => {
            metrics::record_idempotency_lookup(metrics::RESULT_INFLIGHT);
            return Err(PreStreamError::record(
                started,
                StatusCode::CONFLICT,
                "idempotency_conflict",
                "idempotency_key is in-flight; retry after the prior request completes",
                &request_id,
            ));
        }
    };

    // Replay path: cached payloads are the answer. Don't materialize,
    // don't send, don't subscribe — just yield ack/text/done from cache.
    if let Some((ack, chunks, done)) = replay_data {
        metrics::inc_active_sse_streams();
        let stream = async_stream::stream! {
            if let Ok(ev) = ack_event(&ack) {
                yield Ok(ev);
            }
            for chunk in chunks {
                if let Ok(ev) = text_event(&chunk) {
                    yield Ok(ev);
                }
            }
            if let Ok(ev) = done_event(&done) {
                yield Ok(ev);
            }
            metrics::record_request(metrics::STATUS_OK, started.elapsed());
            metrics::dec_active_sse_streams();
        };
        let boxed: EventStream = Box::pin(stream);
        return Ok(Sse::new(boxed).keep_alive(KeepAlive::default()));
    }

    // 3. IDs. conversation_id resume is punted to Step 3.5 (depends on
    //    registry persistence). For now every request opens a fresh
    //    conversation; the returned id lets RingHub continue the thread
    //    in-process, which is fine for v1's single-host deployment.
    let conversation_id = req.conversation_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let turn_id = Uuid::new_v4().to_string();

    // 4. Subscribe BEFORE send so we never miss a fast AgentResponse.
    let mut events = state.events.subscribe();

    // 5. Build address. Chat-bubble Bob lives at `<agent_name>[user_id]`;
    //    address parsing rejects keys with characters that break the
    //    grammar so this also doubles as a sanity check on user_id shape.
    let address_str = format!("{}[{}]", state.agent_name, req.user_id);
    let address = Address::parse(&address_str).map_err(|e| {
        PreStreamError::record(
            started,
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("user_id produced invalid address: {e}"),
            &request_id,
        )
    })?;

    // 6. Wrap the user text in the listener's payload XML. Bob's
    //    payload_class produces a tag like `AgentTask`; we write the
    //    user message as `<task>` inside it, matching the existing
    //    dispatch + TUI conventions.
    let payload_tag = state
        .organism
        .get_listener(&state.agent_name)
        .map(|l| l.payload_tag.clone())
        .ok_or_else(|| {
            PreStreamError::record(
                started,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("agent listener '{}' not in organism", state.agent_name),
                &request_id,
            )
        })?;

    let body_xml = format!(
        "<{tag}><task>{text}</task></{tag}>",
        tag = payload_tag,
        text = xml_escape(&req.text),
    );

    let envelope = Envelope {
        to: address,
        from: None,
        body: body_xml.into_bytes(),
        buffer: None,
    };

    // 7. Send. After this, the instance is materialized and the message
    //    is in flight. Errors here haven't reached "ack sent" yet so we
    //    surface as HTTP errors per the contract.
    state.router.send_to(&envelope).await.map_err(|e| {
        PreStreamError::record(
            started,
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("router send failed: {e}"),
            &request_id,
        )
    })?;

    // 8. Look up the buffer thread_id we just delivered to. The router
    //    creates a buffer per (instance, default-buffer) pair on the
    //    first message; the handler context's thread_id matches the
    //    buffer's, so that's what AgentResponse events carry.
    let buffer_thread_id = state
        .router
        .list()
        .await
        .into_iter()
        .find(|info| info.address.raw() == address_str)
        .and_then(|info| {
            info.buffers
                .get(&BufferId::default_buffer())
                .map(|b| b.thread_id.clone())
        })
        .ok_or_else(|| {
            PreStreamError::record(
                started,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "could not resolve buffer thread_id after materialization",
                &request_id,
            )
        })?;

    // 9. Build the SSE stream. ack first, then filter events for our
    //    buffer thread, emit text per AgentResponse, terminate on done.
    //    Along the way, accumulate the emitted payloads into the
    //    idempotency cache so subsequent retries with the same key
    //    replay these exact events.
    metrics::inc_active_sse_streams();
    let conv_for_done = conversation_id.clone();
    let req_for_done = request_id.clone();
    let cached_ack = AckPayload {
        request_id: request_id.clone(),
        conversation_id: conversation_id.clone(),
    };
    let idempotency = state.idempotency.clone();
    let cache_key_for_stream = cache_key.clone();

    let stream = async_stream::stream! {
        // --- ack ---
        match ack_event(&cached_ack) {
            Ok(ev) => yield Ok(ev),
            Err(_) => {
                // Serialization can't realistically fail on these types,
                // but if it does, terminate without ack — the connection
                // closes and the client observes an empty stream. Release
                // the cache slot so a retry can proceed.
                idempotency.release(&cache_key_for_stream);
                return;
            }
        }

        let mut silent = true;
        let mut cached_chunks: Vec<String> = Vec::new();
        let mut shim_decisions: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut active_steers: Vec<String> = Vec::new();
        let mut signals: Vec<String> = Vec::new();

        // Cap how long we wait for Bob to respond. v1 contract says nothing
        // about timeout but we don't want a stuck instance to hold an HTTP
        // connection forever. 60s matches typical LLM long-tail latency.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);

        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => break,
                msg = events.recv() => {
                    match msg {
                        Ok(PipelineEvent::AgentResponse { thread_id, text, shim_report, .. })
                            if thread_id == buffer_thread_id =>
                        {
                            // Shim outcomes (if any) drive both the silence
                            // decision and the done-event metadata. Cortex's
                            // `gate_decisions` surface as `shim_decisions` per
                            // the public contract.
                            let force_silent = shim_report.as_ref().map(|r| r.silent).unwrap_or(false);
                            if let Some(report) = shim_report {
                                shim_decisions = report.gate_decisions;
                                active_steers = report.active_steers;
                                signals = report.signals;
                            }

                            if force_silent {
                                // Silence-as-first-class: zero text events,
                                // done with silent=true. Empty `text` from a
                                // cortex silent path matches this branch.
                                silent = true;
                            } else {
                                silent = false;
                                if let Ok(ev) = text_event(&text) {
                                    yield Ok(ev);
                                    cached_chunks.push(text);
                                }
                            }
                            break;
                        }
                        Ok(_) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                            metrics::record_broadcast_lag();
                            continue;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }

        // --- done ---
        let done = DonePayload {
            conversation_id: conv_for_done,
            turn_id: turn_id.clone(),
            request_id: req_for_done,
            silent,
            metadata: DoneMetadata {
                generation_ms: started.elapsed().as_millis() as u64,
                model: String::new(),
                memex_corpora_queried: vec![],
                shim_decisions,
                active_steers,
                signals,
            },
        };
        let done_for_commit = done.clone();
        if let Ok(ev) = done_event(&done) {
            yield Ok(ev);
        }

        // Commit cached payloads to the idempotency cache so a retry
        // with the same (token, idempotency_key, body) replays this
        // exact stream. Timeouts and broadcast-closed paths also
        // commit — by contract, the key identifies the request, not
        // the outcome; clients use a fresh key when they want to
        // "try again with hope of a different result."
        idempotency.commit(
            &cache_key_for_stream,
            cached_ack,
            cached_chunks,
            done_for_commit,
        );

        // Stream complete — record the success metric and release the
        // active-stream gauge slot. Done event was the terminal yield,
        // so this fires after the client has received everything.
        metrics::record_request(metrics::STATUS_OK, started.elapsed());
        metrics::dec_active_sse_streams();
    };

    let boxed: EventStream = Box::pin(stream);
    Ok(Sse::new(boxed).keep_alive(KeepAlive::default()))
}

/// XML-escape user input before embedding it in a payload tag. Matches
/// the conventions in `agentos_tools::xml_escape`; duplicated here so
/// the server crate doesn't pull in the tool layer just for this.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Reject `user_id`s that could escape the `bob[user_id]` address
/// formatter or collide with reserved grammar (`+` for the cache-
/// composition operator). 128-char cap bounds key length in the
/// idempotency cache and any downstream identifier surfaces.
fn is_valid_user_id(s: &str) -> bool {
    let len = s.len();
    if !(1..=128).contains(&len) {
        return false;
    }
    s.bytes().all(|b| {
        b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b':'
    })
}

/// Constant-time bearer-token check.
///
/// `PartialEq` on `str` / `String` short-circuits at the first
/// mismatching byte → enables remote byte-by-byte timing oracle. Both
/// tokens go through SHA-256 (fixed 32-byte output) and the digests
/// are compared with `subtle::ConstantTimeEq`. Hashing makes the
/// comparison length-independent — supplying a 5-byte vs 500-byte
/// token both pay the same SHA-256 + 32-byte compare cost.
fn ct_eq_token(supplied: &str, expected: &str) -> bool {
    let s = Sha256::digest(supplied.as_bytes());
    let e = Sha256::digest(expected.as_bytes());
    s.ct_eq(&e).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_id_accepts_typical_identifiers() {
        // UUID, integer, namespaced — all real-world shapes.
        assert!(is_valid_user_id("alice"));
        assert!(is_valid_user_id("42"));
        assert!(is_valid_user_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_valid_user_id("org:1234"));
        assert!(is_valid_user_id("user_with_underscore"));
        assert!(is_valid_user_id("A"));
    }

    #[test]
    fn user_id_rejects_address_grammar_characters() {
        // These are the ones that would escape `bob[user_id]` formatting.
        assert!(!is_valid_user_id("alice]"));
        assert!(!is_valid_user_id("alice[evil"));
        assert!(!is_valid_user_id("alice].dm[evil"));
        assert!(!is_valid_user_id("alice.admin"));
        // Reserved cache-composition operator.
        assert!(!is_valid_user_id("alice+admin"));
    }

    #[test]
    fn user_id_rejects_whitespace_and_control() {
        assert!(!is_valid_user_id("alice bob"));
        assert!(!is_valid_user_id("alice\tbob"));
        assert!(!is_valid_user_id("alice\nbob"));
        assert!(!is_valid_user_id("alice\0bob"));
    }

    #[test]
    fn user_id_rejects_empty_and_oversize() {
        assert!(!is_valid_user_id(""));
        let huge: String = "a".repeat(129);
        assert!(!is_valid_user_id(&huge));
        let just_right: String = "a".repeat(128);
        assert!(is_valid_user_id(&just_right));
    }

    #[test]
    fn user_id_rejects_non_ascii() {
        // Unicode look-alikes are a classic homoglyph attack vector.
        // Strict ASCII allowlist sidesteps the whole category.
        assert!(!is_valid_user_id("álice"));
        assert!(!is_valid_user_id("аlice")); // Cyrillic 'а'
    }
}