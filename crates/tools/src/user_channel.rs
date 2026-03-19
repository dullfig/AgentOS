//! User channel — pipeline listener that bridges agents to the TUI.
//!
//! Two modes:
//! - **Display** (`<UserDisplay>`) — fire-and-forget, agent narrates progress
//! - **Query** (`<UserQuery>`) — blocks until the user types a response
//!
//! The handler emits PipelineEvents for the TUI to render, and for queries,
//! sends a request through a channel and awaits the user's response via oneshot.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::{broadcast, mpsc, oneshot};

use agentos_events::PipelineEvent;

use super::{extract_tag, ToolPeer, ToolResponse};

/// A query request sent to the TUI, awaiting user response.
pub struct UserQueryRequest {
    /// The agent asking the question.
    pub agent_name: String,
    /// The question text.
    pub question: String,
    /// Thread the agent is working on.
    pub thread_id: String,
    /// Oneshot channel to send the user's answer back.
    pub response_tx: oneshot::Sender<String>,
}

/// The user channel handler — registered as the "user" listener.
pub struct UserChannelHandler {
    /// Broadcast sender for display events (TUI subscribes).
    event_tx: broadcast::Sender<PipelineEvent>,
    /// Channel to send query requests to TUI.
    query_tx: mpsc::Sender<UserQueryRequest>,
}

impl UserChannelHandler {
    pub fn new(
        event_tx: broadcast::Sender<PipelineEvent>,
        query_tx: mpsc::Sender<UserQueryRequest>,
    ) -> Self {
        Self { event_tx, query_tx }
    }
}

#[async_trait]
impl Handler for UserChannelHandler {
    async fn handle(&self, payload: ValidatedPayload, ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        if let Some(text) = extract_tag(&xml_str, "text") {
            // Display mode: emit event, ack immediately
            let _ = self.event_tx.send(PipelineEvent::UserDisplay {
                thread_id: ctx.thread_id.clone(),
                agent_name: ctx.from.clone(),
                text: text.clone(),
            });
            Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok("displayed"),
            })
        } else if let Some(question) = extract_tag(&xml_str, "question") {
            // Query mode: send to TUI, block until user responds
            let (response_tx, response_rx) = oneshot::channel();

            let _ = self.event_tx.send(PipelineEvent::UserQuery {
                thread_id: ctx.thread_id.clone(),
                agent_name: ctx.from.clone(),
                question: question.clone(),
            });

            let send_result = self.query_tx.send(UserQueryRequest {
                agent_name: ctx.from.clone(),
                question,
                thread_id: ctx.thread_id.clone(),
                response_tx,
            }).await;

            if send_result.is_err() {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("TUI is not available to receive queries"),
                });
            }

            // Block until user responds
            match response_rx.await {
                Ok(answer) => Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&answer),
                }),
                Err(_) => Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("user declined to answer"),
                }),
            }
        } else {
            Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "provide <text> for display or <question> to ask the user",
                ),
            })
        }
    }
}

#[async_trait]
impl ToolPeer for UserChannelHandler {
    fn name(&self) -> &str {
        "user"
    }

    fn wit(&self) -> &str {
        r#"
/// Communicate with the user. Use <text> to display a status message (no response), or <question> to ask the user something and wait for their answer.
interface user {
    record request {
        /// Display-only message (fire and forget). Use for progress updates.
        text: option<string>,
        /// Question for the user (blocks until they respond). Use when you need clarification.
        question: option<string>,
    }
    ask: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(from: &str) -> HandlerContext {
        HandlerContext {
            from: from.into(),
            own_name: "user".into(),
            thread_id: "test-thread".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "UserRequest".into(),
        }
    }

    #[tokio::test]
    async fn display_acks_immediately() {
        let (event_tx, mut event_rx) = broadcast::channel(16);
        let (query_tx, _query_rx) = mpsc::channel(1);
        let handler = UserChannelHandler::new(event_tx, query_tx);

        let xml = "<UserRequest><text>Surveying codebase...</text></UserRequest>";
        let result = handler.handle(make_payload(xml), make_ctx("plan-expert")).await.unwrap();

        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("displayed"), "got: {s}");
            }
            _ => panic!("expected Reply"),
        }

        // Event should have been emitted
        let event = event_rx.recv().await.unwrap();
        match event {
            PipelineEvent::UserDisplay { text, agent_name, .. } => {
                assert_eq!(text, "Surveying codebase...");
                assert_eq!(agent_name, "plan-expert");
            }
            _ => panic!("expected UserDisplay event"),
        }
    }

    #[tokio::test]
    async fn query_blocks_until_response() {
        let (event_tx, _event_rx) = broadcast::channel(16);
        let (query_tx, mut query_rx) = mpsc::channel(1);
        let handler = UserChannelHandler::new(event_tx, query_tx);

        let xml = "<UserRequest><question>Split into separate crate?</question></UserRequest>";

        // Spawn the handler (it will block)
        let handle = tokio::spawn(async move {
            handler.handle(make_payload(xml), make_ctx("plan-expert")).await
        });

        // Simulate TUI receiving and responding
        let request = query_rx.recv().await.unwrap();
        assert_eq!(request.question, "Split into separate crate?");
        assert_eq!(request.agent_name, "plan-expert");
        let _ = request.response_tx.send("Yes, do it".into());

        // Handler should now return with the answer
        let result = handle.await.unwrap().unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("Yes, do it"), "got: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn query_returns_error_on_cancel() {
        let (event_tx, _event_rx) = broadcast::channel(16);
        let (query_tx, mut query_rx) = mpsc::channel(1);
        let handler = UserChannelHandler::new(event_tx, query_tx);

        let xml = "<UserRequest><question>Continue?</question></UserRequest>";

        let handle = tokio::spawn(async move {
            handler.handle(make_payload(xml), make_ctx("plan-expert")).await
        });

        // Simulate TUI dropping the oneshot (user pressed Esc)
        let request = query_rx.recv().await.unwrap();
        drop(request.response_tx);

        let result = handle.await.unwrap().unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("declined"), "got: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_both_fields_returns_error() {
        let (event_tx, _event_rx) = broadcast::channel(16);
        let (query_tx, _query_rx) = mpsc::channel(1);
        let handler = UserChannelHandler::new(event_tx, query_tx);

        let xml = "<UserRequest></UserRequest>";
        let result = handler.handle(make_payload(xml), make_ctx("test")).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("false"), "expected error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }
}
