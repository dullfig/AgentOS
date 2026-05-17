//! agentos-server — HTTP+SSE frontend per the AgentOS API contract (v1).
//!
//! See `project_agentos_api_contract.md` for the wire spec. This crate
//! exposes [`build_router`] which mounts `POST /v1/messages` over an
//! axum [`Router`], plus a [`ServerState`] handle that the bin (or
//! integration tests) wire up to an [`AgentPipeline`] + platform router.

pub mod handler;
pub mod idempotency;
pub mod metrics;
pub mod sse;
pub mod state;

pub use state::ServerState;

use axum::response::IntoResponse;
use axum::Router;
use std::sync::Arc;

/// Build the axum router for the v1 contract.
///
/// All routes mount under `/v1/*` except `GET /metrics` which lives at
/// the root (standard Prometheus convention). The shared state carries
/// the platform router (for materializing instances) and the auth token.
///
/// Side effect: installs the global Prometheus recorder on first call
/// (subsequent calls are no-ops). Tests don't need to do this manually.
pub fn build_router(state: Arc<ServerState>) -> Router {
    metrics::init();
    Router::new()
        .route(
            "/v1/messages",
            axum::routing::post(handler::post_messages),
        )
        .route("/metrics", axum::routing::get(metrics_handler))
        .with_state(state)
}

/// `GET /metrics` — Prometheus exposition format.
///
/// Unauthenticated. Production deployments should either firewall this
/// route to the scraper's network or run a reverse proxy that adds auth.
/// Per the standard Prometheus pattern, the metrics endpoint itself is
/// read-only and has no bearer-token convention.
async fn metrics_handler() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics::render(),
    )
}
