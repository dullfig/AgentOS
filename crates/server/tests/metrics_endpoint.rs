//! HTTP-layer integration tests for the `GET /metrics` endpoint.
//!
//! Verifies:
//! - `/metrics` is reachable without auth and returns Prometheus exposition format.
//! - After a `POST /v1/messages` round-trip, the request-duration histogram
//!   and request counter show up in the output.
//! - Idempotency lookup counters increment on miss + replay.
//! - Active-stream gauge appears in output (we don't assert exact value
//!   since timing makes it racy, but the metric must be declared).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agentos_events::PipelineEvent;
use agentos_organism::parser::parse_organism;
use agentos_pipeline::AgentPipelineBuilder;
use agentos_server::{build_router, metrics, ServerState};

use rust_pipeline::prelude::{FnHandler, HandlerContext, HandlerResponse, ValidatedPayload};
use tempfile::TempDir;
use tokio::net::TcpListener;

const BOB_REPLY: &str = "Hi from the metrics test.";

fn organism_yaml() -> &'static str {
    r#"
organism:
  name: metrics-test

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

/// Mirrors `boot_server` from idempotency.rs — same shape so test
/// failures look familiar. Returns pipeline + tempdir so the caller
/// keeps them alive for the duration of the test.
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
        .initialize_root("metrics-test", "default")
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

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let (addr, _counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "GET /metrics should succeed");

    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/plain"),
        "expected Prometheus text content-type, got {ct}"
    );

    let body = resp.text().await.unwrap();
    // Even before any request fires, the describe_* calls in metrics::init
    // emit HELP/TYPE headers. Body shouldn't be empty.
    assert!(
        body.contains("# HELP") || body.contains("# TYPE"),
        "metrics body should contain at least one HELP or TYPE header; got: {body:?}"
    );
}

#[tokio::test]
async fn request_counter_and_histogram_present_after_round_trip() {
    let (addr, _counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    // Drive one successful request through the live agent path.
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "metrics-idem-1"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    // Drain the SSE body so the stream's terminal recording fires.
    let _ = resp.text().await.unwrap();

    // Same idempotency key + same body → replay path. Exercises the
    // replay branch's success recording too.
    let replay = client
        .post(format!("http://{addr}/v1/messages"))
        .bearer_auth("test-token")
        .json(&post_body("hello", "metrics-idem-1"))
        .send()
        .await
        .unwrap();
    assert!(replay.status().is_success());
    let _ = replay.text().await.unwrap();

    // Now scrape /metrics.
    let scrape = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    let body = scrape.text().await.unwrap();

    // Every metric the handler touches should appear in the output by
    // name. The exact numeric values are timing-dependent; we don't
    // assert them, just that the metrics are declared and visible.
    for required in [
        metrics::REQUEST_DURATION_SECONDS,
        metrics::REQUESTS_TOTAL,
        metrics::IDEMPOTENCY_LOOKUPS_TOTAL,
        metrics::ACTIVE_SSE_STREAMS,
    ] {
        assert!(
            body.contains(required),
            "expected /metrics body to contain {required}; got:\n{body}"
        );
    }

    // Status label on the success path should be present.
    assert!(
        body.contains(&format!(r#"status="{}""#, metrics::STATUS_OK)),
        "expected status=ok label after successful round-trip; got:\n{body}"
    );

    // Both lookup results should show up — one miss (first request)
    // and one replay (second request).
    assert!(
        body.contains(&format!(r#"result="{}""#, metrics::RESULT_MISS)),
        "expected result=miss label after first request"
    );
    assert!(
        body.contains(&format!(r#"result="{}""#, metrics::RESULT_REPLAY)),
        "expected result=replay label after second request"
    );
}

#[tokio::test]
async fn client_error_status_recorded_for_invalid_request() {
    let (addr, _counter, _dir, _pipeline) = boot_server().await;
    let client = reqwest::Client::new();

    // Missing bearer → 401 → status=client_error.
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .json(&post_body("hello", "metrics-idem-2"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);

    let scrape = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    let body = scrape.text().await.unwrap();

    assert!(
        body.contains(&format!(r#"status="{}""#, metrics::STATUS_CLIENT_ERROR)),
        "expected status=client_error label after 401; got:\n{body}"
    );
}
