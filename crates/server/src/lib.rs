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

use axum::extract::DefaultBodyLimit;
use axum::response::IntoResponse;
use axum::Router;
use std::sync::Arc;
use tower::limit::ConcurrencyLimitLayer;

/// Hard cap on request body size. JSON payload is small (text + a
/// few short fields); 64 KiB is generous and bounds memory
/// amplification from large-body attacks. axum returns 413 Payload
/// Too Large when the limit is hit. Applies to all routes; `/metrics`
/// is GET (no body) so the cap is a no-op there.
const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024;

/// Hard cap on concurrent in-flight requests (each `/v1/messages`
/// holds an SSE stream open). Sized for the projected 2-3k DAU
/// single-host deployment with ~50-100 concurrent streams in normal
/// operation; gives 2-3× headroom before backpressure kicks in.
/// Above this, tower queues incoming requests until a slot frees up.
///
/// `/metrics` shares the pool but Prometheus scrapes return in
/// milliseconds — they won't compete with SSE streams meaningfully.
const MAX_CONCURRENT_REQUESTS: usize = 256;

/// Build the axum router for the v1 contract.
///
/// All routes mount under `/v1/*` except `GET /metrics` which lives at
/// the root (standard Prometheus convention). The shared state carries
/// the platform router (for materializing instances) and the auth token.
///
/// Side effect: installs the global Prometheus recorder on first call
/// (subsequent calls are no-ops). Tests don't need to do this manually.
///
/// **Defense layers** (security audit H2):
/// - body-size cap (via `DefaultBodyLimit`) at the router level
/// - concurrency cap on the whole router so an SSE flood can't
///   stack up unbounded
/// - per-IP rate-limiting is queued as a follow-up (needs
///   tower-governor or equivalent)
pub fn build_router(state: Arc<ServerState>) -> Router {
    metrics::init();
    Router::new()
        .route(
            "/v1/messages",
            axum::routing::post(handler::post_messages),
        )
        .route("/metrics", axum::routing::get(metrics_handler))
        .with_state(state)
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(ConcurrencyLimitLayer::new(MAX_CONCURRENT_REQUESTS))
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
