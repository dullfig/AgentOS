//! Wiremock-backed tests for `EmbedClient` against the proposed
//! `POST /v1/embed` cortex endpoint. Pin the wire format so when
//! cortex ships the real endpoint we know which side to reconcile.

use agentos_cortex_shim::{
    EmbedClient, EmbedRequest, EmbedResponse, Pooling, ShimClientError,
};
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn embed_round_trips_text_to_vector() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embed"))
        .and(header("authorization", "Bearer test"))
        .and(body_partial_json(json!({
            "context": "hello world",
            "layer": "final",
            "pooling": "last_token"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vector": [0.1, -0.2, 0.3, 0.4],
            "dim": 4
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), Some("test".into()));
    let resp = client
        .embed(&EmbedRequest {
            context: "hello world".into(),
            layer: "final".into(),
            pooling: Pooling::LAST_TOKEN,
        })
        .await
        .unwrap();

    assert_eq!(resp.dim, 4);
    assert_eq!(resp.vector, vec![0.1, -0.2, 0.3, 0.4]);
}

#[tokio::test]
async fn embed_text_helper_uses_correct_path() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dim": 8
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), None);
    let resp = client
        .embed_text("hi", "entrance:5", Pooling::MEAN)
        .await
        .unwrap();
    assert_eq!(resp.dim, 8);
}

#[tokio::test]
async fn embed_propagates_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embed"))
        .respond_with(ResponseTemplate::new(503).set_body_string("model loading"))
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), None);
    let err = client
        .embed_text("hi", "final", Pooling::LAST_TOKEN)
        .await
        .expect_err("expected ApiError");

    match err {
        ShimClientError::ApiError { status, message } => {
            assert_eq!(status, 503);
            assert!(message.contains("model loading"));
        }
        other => panic!("expected ApiError, got {other:?}"),
    }
}

#[tokio::test]
async fn embed_propagates_invalid_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embed"))
        // 200 with garbage JSON: parse step should fail with InvalidResponse.
        .respond_with(ResponseTemplate::new(200).set_body_string("{not really json"))
        .mount(&server)
        .await;

    let client = EmbedClient::new(server.uri(), None);
    let err = client
        .embed_text("hi", "final", Pooling::LAST_TOKEN)
        .await
        .expect_err("expected InvalidResponse");
    matches!(err, ShimClientError::InvalidResponse(_));
}

#[tokio::test]
async fn embed_response_deserializes_minimal_shape() {
    // Sanity check: the type itself round-trips even outside an HTTP call.
    let body = r#"{"vector": [0.0, 1.0, 2.0], "dim": 3}"#;
    let resp: EmbedResponse = serde_json::from_str(body).unwrap();
    assert_eq!(resp.dim, 3);
    assert_eq!(resp.vector.len(), 3);
}
