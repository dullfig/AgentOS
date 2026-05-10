//! End-to-end round-trip test: HTTP POST → router → pipeline handler →
//! AgentResponse event → SSE stream → client parses event sequence.
//!
//! Uses a stub `bob` listener that emits a known `AgentResponse` so the
//! test doesn't need an LLM API key. Verifies the contract's
//! `ack → text → done` ordering and payload shapes.

use std::sync::Arc;
use std::time::Duration;

use agentos_events::PipelineEvent;
use agentos_organism::parser::parse_organism;
use agentos_pipeline::AgentPipelineBuilder;
use agentos_server::{build_router, ServerState};

use rust_pipeline::prelude::{FnHandler, HandlerContext, HandlerResponse, ValidatedPayload};
use tempfile::TempDir;
use tokio::net::TcpListener;

const BOB_REPLY: &str = "Hi from the stub Bob!";

fn organism_yaml() -> &'static str {
    // Minimal organism: one agent listener `bob` that the test handler
    // owns. is_agent=true is required because resolve_organism in
    // PipelineRuntime filters on it before materializing.
    r#"
organism:
  name: server-roundtrip-test

listeners:
  - name: bob
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Stub Bob"
    agent:
      prompt: "stub"

profiles:
  default:
    linux_user: agentos
    listeners: [bob]
    journal: retain_forever
"#
}

#[tokio::test]
async fn post_messages_round_trip() {
    let org = parse_organism(organism_yaml()).unwrap();
    let dir = TempDir::new().unwrap();

    // Build the pipeline. The stub handler captures event_tx so it can
    // emit `AgentResponse` — that's what the SSE stream waits for.
    let mut builder = AgentPipelineBuilder::new(org, &dir.path().join("data"));
    let event_tx = builder.event_sender();

    let event_tx_for_handler = event_tx.clone();
    let bob_handler = FnHandler(move |p: ValidatedPayload, ctx: HandlerContext| {
        let event_tx = event_tx_for_handler.clone();
        Box::pin(async move {
            let _ = event_tx.send(PipelineEvent::AgentResponse {
                thread_id: ctx.thread_id.clone(),
                agent_name: "bob".to_string(),
                text: BOB_REPLY.to_string(),
            });
            Ok(HandlerResponse::Reply { payload_xml: p.xml })
        })
    });

    let mut pipeline = builder.register("bob", bob_handler).unwrap().build().unwrap();

    pipeline
        .initialize_root("server-roundtrip-test", "default")
        .await
        .unwrap();
    pipeline.run();

    // SharedRouter on top of the running pipeline.
    let shared_router = Arc::new(pipeline.shared_router(0, Duration::from_secs(60)));

    // ServerState — note `events` is the SAME Sender the handler emits on.
    let state = Arc::new(ServerState {
        router: shared_router,
        events: event_tx,
        organism: Arc::new(pipeline.organism().clone()),
        agent_name: "bob".to_string(),
        auth_token: "test-token".to_string(),
    });

    // Bind the server on an ephemeral port so the test never collides.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give axum a tick to start serving.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // POST /v1/messages
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .header("X-Request-Id", "rid-123")
        .json(&serde_json::json!({
            "user_id": "alice",
            "user_tier": "warm",
            "text": "hello bob",
            "idempotency_key": "idem-1"
        }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "expected 200, got {}", resp.status());
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "text/event-stream"
    );

    // Parse SSE: read body as text and split on blank-line event boundaries.
    // axum's SSE writer emits `event: <name>\ndata: <json>\n\n`. With the
    // body fully buffered, that's enough to verify the ordering.
    let body = resp.text().await.unwrap();
    let events: Vec<&str> = body.split("\n\n").filter(|s| !s.trim().is_empty()).collect();

    assert!(
        events.len() >= 3,
        "expected at least 3 SSE events (ack, text, done), got {}: {:?}",
        events.len(),
        events
    );

    // First event must be ack with our request_id and a non-empty conversation_id.
    let ack = events[0];
    assert!(ack.contains("event: ack"), "first event was not ack: {ack}");
    assert!(ack.contains("\"request_id\":\"rid-123\""), "ack missing request_id: {ack}");
    assert!(ack.contains("\"conversation_id\""), "ack missing conversation_id: {ack}");

    // Some text event must carry our reply.
    let text_events: Vec<&&str> = events.iter().filter(|e| e.contains("event: text")).collect();
    assert!(
        !text_events.is_empty(),
        "expected at least one text event, got events: {events:?}"
    );
    let text = text_events[0];
    assert!(text.contains(BOB_REPLY), "text event did not contain Bob's reply: {text}");

    // Last event must be done with silent=false.
    let last = events.last().unwrap();
    assert!(last.contains("event: done"), "last event was not done: {last}");
    assert!(last.contains("\"silent\":false"), "done.silent should be false: {last}");
    assert!(last.contains("\"request_id\":\"rid-123\""), "done missing request_id: {last}");
}

#[tokio::test]
async fn post_messages_rejects_anon_tier() {
    let org = parse_organism(organism_yaml()).unwrap();
    let dir = TempDir::new().unwrap();

    let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"));
    let event_tx = builder.event_sender();
    let bob = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
        Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
    });
    let mut pipeline = builder.register("bob", bob).unwrap().build().unwrap();
    pipeline
        .initialize_root("server-roundtrip-test", "default")
        .await
        .unwrap();
    pipeline.run();

    let shared_router = Arc::new(pipeline.shared_router(0, Duration::from_secs(60)));
    let state = Arc::new(ServerState {
        router: shared_router,
        events: event_tx,
        organism: Arc::new(pipeline.organism().clone()),
        agent_name: "bob".to_string(),
        auth_token: "test-token".to_string(),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&serde_json::json!({
            "user_id": "alice",
            "user_tier": "anon",
            "text": "hi",
            "idempotency_key": "idem-anon"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn post_messages_rejects_missing_bearer() {
    let org = parse_organism(organism_yaml()).unwrap();
    let dir = TempDir::new().unwrap();
    let builder = AgentPipelineBuilder::new(org, &dir.path().join("data"));
    let event_tx = builder.event_sender();
    let bob = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
        Box::pin(async move { Ok(HandlerResponse::Reply { payload_xml: p.xml }) })
    });
    let mut pipeline = builder.register("bob", bob).unwrap().build().unwrap();
    pipeline
        .initialize_root("server-roundtrip-test", "default")
        .await
        .unwrap();
    pipeline.run();

    let shared_router = Arc::new(pipeline.shared_router(0, Duration::from_secs(60)));
    let state = Arc::new(ServerState {
        router: shared_router,
        events: event_tx,
        organism: Arc::new(pipeline.organism().clone()),
        agent_name: "bob".to_string(),
        auth_token: "test-token".to_string(),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .json(&serde_json::json!({
            "user_id": "alice",
            "user_tier": "warm",
            "text": "hi",
            "idempotency_key": "idem-noauth"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "unauthenticated");
}
