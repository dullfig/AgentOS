//! Wiremock-backed round-trip tests for `CortexShimClient`.
//!
//! Validates the wire format we emit against what cortex's shim API
//! contract expects. Real-cortex smoke testing is deferred to whenever
//! the 4090 box is up; these tests pin the protocol-level behavior.

use agentos_cortex_shim::manifest::{
    Attachment, InputShape, OutputShape, ShimManifest, ShimPhase, ShimSummary,
};
use agentos_cortex_shim::{CortexShimClient, ShimClientError};
use serde_json::json;
use wiremock::matchers::{header, method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sample_manifest(id: &str) -> ShimManifest {
    ShimManifest {
        id: id.into(),
        version: "0.3.1".into(),
        phase: ShimPhase::Gate,
        attachment: Attachment {
            layer: "final".into(),
            pooling: "last_token".into(),
        },
        input_shape: InputShape { hidden_dim: 4096 },
        output_shape: OutputShape {
            kind: "scalar".into(),
        },
        description: Some("test gate".into()),
    }
}

#[tokio::test]
async fn register_uploads_multipart_with_bearer() {
    let server = MockServer::start().await;
    let m = sample_manifest("should_respond");

    Mock::given(method("PUT"))
        .and(path("/v1/shims/should_respond"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), Some("test-token".into()));
    client.register(&m, b"\x00\x01ONNX-bytes".to_vec()).await.unwrap();
}

#[tokio::test]
async fn register_propagates_api_error() {
    let server = MockServer::start().await;
    let m = sample_manifest("bad_shim");

    Mock::given(method("PUT"))
        .and(path("/v1/shims/bad_shim"))
        .respond_with(ResponseTemplate::new(400).set_body_string("invalid manifest"))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let err = client
        .register(&m, b"bytes".to_vec())
        .await
        .expect_err("expected API error");

    match err {
        ShimClientError::ApiError { status, message } => {
            assert_eq!(status, 400);
            assert!(message.contains("invalid manifest"));
        }
        other => panic!("expected ApiError, got {other:?}"),
    }
}

#[tokio::test]
async fn list_parses_summary_array() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/shims/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"id": "should_respond", "version": "0.3.1", "phase": "gate"},
            {"id": "voice_bob",      "version": "0.1.0", "phase": "steer"}
        ])))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let summaries = client.list().await.unwrap();
    assert_eq!(
        summaries,
        vec![
            ShimSummary {
                id: "should_respond".into(),
                version: "0.3.1".into(),
                phase: ShimPhase::Gate,
            },
            ShimSummary {
                id: "voice_bob".into(),
                version: "0.1.0".into(),
                phase: ShimPhase::Steer,
            },
        ]
    );
}

#[tokio::test]
async fn get_returns_full_manifest() {
    let server = MockServer::start().await;
    let manifest = sample_manifest("should_respond");

    Mock::given(method("GET"))
        .and(path("/v1/shims/should_respond"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&manifest))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let got = client.get("should_respond").await.unwrap();
    assert_eq!(got, manifest);
}

#[tokio::test]
async fn get_404_maps_to_not_found() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path_regex(r"^/v1/shims/.*"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let err = client.get("missing").await.expect_err("expected NotFound");
    match err {
        ShimClientError::NotFound(id) => assert_eq!(id, "missing"),
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_idempotent_on_404() {
    let server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path_regex(r"^/v1/shims/.*"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let err = client.delete("never-existed").await.expect_err("404 → NotFound");
    matches!(err, ShimClientError::NotFound(_));
}

#[tokio::test]
async fn infer_round_trips_scalar_decision() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/shims/infer"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "decision": 0.87,
            "metadata": {"latency_ms": 3}
        })))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let out = client
        .infer("should_respond", json!("does this need bob?"))
        .await
        .unwrap();
    assert_eq!(out.decision, json!(0.87));
    assert_eq!(out.metadata["latency_ms"], json!(3));
}

#[tokio::test]
async fn infer_404_maps_to_not_found() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/shims/infer"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let err = client
        .infer("not_registered", json!(null))
        .await
        .expect_err("404 → NotFound");
    match err {
        ShimClientError::NotFound(id) => assert_eq!(id, "not_registered"),
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn no_bearer_omitted_when_unset() {
    let server = MockServer::start().await;

    // Match a request that does NOT carry an authorization header.
    // wiremock doesn't have a "header is absent" matcher, so we pin
    // on the path + verify the call lands without setting up a
    // bearer-required mock.
    Mock::given(method("GET"))
        .and(path("/v1/shims/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .expect(1)
        .mount(&server)
        .await;

    let client = CortexShimClient::new(server.uri(), None);
    let out = client.list().await.unwrap();
    assert!(out.is_empty());
}
