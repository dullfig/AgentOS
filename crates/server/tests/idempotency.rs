//! HTTP-layer integration tests for the idempotency cache.
//!
//! Verifies the contract's behavior end-to-end via `POST /v1/messages`:
//!
//! - Repeating a key with the same body replays the cached SSE stream
//!   without re-invoking the agent.
//! - Repeating a key with a different body returns
//!   `409 idempotency_conflict`.
//! - First-time keys go through the live agent path.
//!
//! Uses a stub `bob` listener (same fixture as `round_trip.rs`) so the
//! tests don't need an LLM. The stub increments a counter each time
//! it's invoked, letting us assert agent re-invocation count from the
//! test.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agentos_events::PipelineEvent;
use agentos_organism::parser::parse_organism;
use agentos_pipeline::AgentPipelineBuilder;
use agentos_server::{build_router, ServerState};

use rust_pipeline::prelude::{FnHandler, HandlerContext, HandlerResponse, ValidatedPayload};
use tempfile::TempDir;
use tokio::net::TcpListener;

const BOB_REPLY: &str = "Hi from the idempotency stub.";

fn organism_yaml() -> &'static str {
    r#"
organism:
  name: idempotency-test

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

/// Boots a server with a stub Bob listener that counts how many times
/// it's invoked. Returns `(addr, counter, _dir, _pipeline)`; the
/// TempDir AND the AgentPipeline must be held by the caller — dropping
/// either closes the ingress channel and the live agent path breaks.
async fn boot_server() -> (
    std::net::SocketAddr,
    Arc<AtomicUsize>,
    TempDir,
    agentos_pipeline::AgentPipeline,
) {
    let org = parse_organism(organism_yaml()).unwrap();
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path().join("data");

    let counter = Arc::new(AtomicUsize::new(0));

    let builder = AgentPipelineBuilder::new(org, &data_dir);
    let event_tx = builder.event_sender();

    let event_tx_for_handler = event_tx.clone();
    let counter_for_handler = counter.clone();
    let bob_handler = FnHandler(move |p: ValidatedPayload, ctx: HandlerContext| {
        let event_tx = event_tx_for_handler.clone();
        let counter = counter_for_handler.clone();
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
            let _ = event_tx.send(PipelineEvent::AgentResponse {
                thread_id: ctx.thread_id.clone(),
                agent_name: "bob".to_string(),
                text: BOB_REPLY.to_string(),
                shim_report: None,
            });
            Ok(HandlerResponse::Reply { payload_xml: p.xml })
        })
    });

    let mut pipeline = builder.register("bob", bob_handler).unwrap().build().unwrap();
    pipeline
        .initialize_root("idempotency-test", "default")
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
        idempotency: agentos_server::idempotency::IdempotencyCache::new(),
    });

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build_router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, counter, dir, pipeline)
}

fn post_body(text: &str, idem_key: &str) -> serde_json::Value {
    serde_json::json!({
        "user_id": "alice",
        "user_tier": "warm",
        "text": text,
        "idempotency_key": idem_key,
    })
}

async fn drain_sse(resp: reqwest::Response) -> String {
    resp.text().await.unwrap()
}

#[tokio::test]
async fn first_request_invokes_agent_and_populates_cache() {
    let (addr, counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "idem-001"))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body = drain_sse(resp).await;
    assert!(
        status.is_success(),
        "expected 2xx, got {status}: {body}"
    );
    assert!(body.contains(BOB_REPLY), "first response should carry agent reply");
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "agent should be invoked exactly once"
    );
}

#[tokio::test]
async fn replay_same_key_same_body_does_not_reinvoke_agent() {
    let (addr, counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    let first = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .header("X-Request-Id", "rid-replay")
        .json(&post_body("hello world", "idem-002"))
        .send()
        .await
        .unwrap();
    let first_body = drain_sse(first).await;

    let second = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello world", "idem-002"))
        .send()
        .await
        .unwrap();
    assert!(second.status().is_success());
    let second_body = drain_sse(second).await;

    // Same SSE payload bytes — replay is byte-identical.
    assert_eq!(
        first_body, second_body,
        "replay should emit the exact same SSE stream"
    );

    // Agent invoked only once across both HTTP requests.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "second request must NOT re-invoke the agent (it must replay from cache)"
    );

    // Replay preserves request_id from the first response, not from
    // any header on the second request.
    assert!(
        second_body.contains("\"request_id\":\"rid-replay\""),
        "replayed events should carry the original request_id"
    );
}

#[tokio::test]
async fn same_key_different_body_returns_409_conflict() {
    let (addr, counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    let first = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "idem-003"))
        .send()
        .await
        .unwrap();
    assert!(first.status().is_success());
    let _ = drain_sse(first).await;

    let second = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("DIFFERENT text", "idem-003"))
        .send()
        .await
        .unwrap();

    assert_eq!(second.status().as_u16(), 409);
    let err: serde_json::Value = second.json().await.unwrap();
    assert_eq!(err["error"]["code"], "idempotency_conflict");

    // Agent only ran for the first request; conflict short-circuits.
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn different_keys_both_invoke_agent_independently() {
    let (addr, counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    let r1 = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "idem-004-a"))
        .send()
        .await
        .unwrap();
    assert!(r1.status().is_success());
    let _ = drain_sse(r1).await;

    let r2 = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "idem-004-b"))
        .send()
        .await
        .unwrap();
    assert!(r2.status().is_success());
    let _ = drain_sse(r2).await;

    // Two distinct keys → agent ran twice, no replay.
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "distinct idempotency_keys must each invoke the agent"
    );
}
