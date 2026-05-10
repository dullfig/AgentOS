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
#[derive(Debug, Deserialize)]
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

/// `POST /v1/messages` handler.
pub async fn post_messages(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(req): Json<PostMessagesRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, PreStreamError> {
    let request_id = request_id_from_headers(&headers);

    // 1. Auth
    match bearer_token(&headers) {
        Some(t) if t == state.auth_token => {}
        Some(_) => {
            return Err(PreStreamError {
                status: StatusCode::FORBIDDEN,
                code: "unauthorized",
                message: "bearer token did not match".into(),
                request_id,
            });
        }
        None => {
            return Err(PreStreamError {
                status: StatusCode::UNAUTHORIZED,
                code: "unauthenticated",
                message: "missing or malformed Authorization header".into(),
                request_id,
            });
        }
    }

    // 2. Validate. The contract says "anon" never reaches this endpoint —
    //    AgentOS MUST 400 it.
    if req.user_tier == "anon" {
        return Err(PreStreamError {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message: "user_tier=anon is not accepted; anon traffic is handled client-side".into(),
            request_id,
        });
    }
    if !matches!(req.user_tier.as_str(), "warm" | "member") {
        return Err(PreStreamError {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message: "user_tier must be one of: warm, member".into(),
            request_id,
        });
    }
    if req.text.trim().is_empty() {
        return Err(PreStreamError {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message: "text must be non-empty".into(),
            request_id,
        });
    }
    if req.user_id.trim().is_empty() {
        return Err(PreStreamError {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message: "user_id must be non-empty".into(),
            request_id,
        });
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
    let address = Address::parse(&address_str).map_err(|e| PreStreamError {
        status: StatusCode::BAD_REQUEST,
        code: "invalid_request",
        message: format!("user_id produced invalid address: {e}"),
        request_id: request_id.clone(),
    })?;

    // 6. Wrap the user text in the listener's payload XML. Bob's
    //    payload_class produces a tag like `AgentTask`; we write the
    //    user message as `<task>` inside it, matching the existing
    //    dispatch + TUI conventions.
    let payload_tag = state
        .organism
        .get_listener(&state.agent_name)
        .map(|l| l.payload_tag.clone())
        .ok_or_else(|| PreStreamError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: format!("agent listener '{}' not in organism", state.agent_name),
            request_id: request_id.clone(),
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
    state.router.send_to(&envelope).await.map_err(|e| PreStreamError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        code: "internal_error",
        message: format!("router send failed: {e}"),
        request_id: request_id.clone(),
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
        .ok_or_else(|| PreStreamError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: "could not resolve buffer thread_id after materialization".into(),
            request_id: request_id.clone(),
        })?;

    // 9. Build the SSE stream. ack first, then filter events for our
    //    buffer thread, emit text per AgentResponse, terminate on done.
    let started = Instant::now();
    let conv_for_done = conversation_id.clone();
    let req_for_done = request_id.clone();

    let stream = async_stream::stream! {
        // --- ack ---
        match ack_event(&AckPayload {
            request_id: request_id.clone(),
            conversation_id: conversation_id.clone(),
        }) {
            Ok(ev) => yield Ok(ev),
            Err(_) => {
                // Serialization can't realistically fail on these types,
                // but if it does, terminate without ack — the connection
                // closes and the client observes an empty stream.
                return;
            }
        }

        let mut silent = true;

        // Cap how long we wait for Bob to respond. v1 contract says nothing
        // about timeout but we don't want a stuck instance to hold an HTTP
        // connection forever. 60s matches typical LLM long-tail latency.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);

        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => break,
                msg = events.recv() => {
                    match msg {
                        Ok(PipelineEvent::AgentResponse { thread_id, text, .. })
                            if thread_id == buffer_thread_id =>
                        {
                            silent = false;
                            if let Ok(ev) = text_event(&text) {
                                yield Ok(ev);
                            }
                            break;
                        }
                        Ok(_) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
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
            },
        };
        if let Ok(ev) = done_event(&done) {
            yield Ok(ev);
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
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