//! agentos-server — HTTP+SSE frontend per the AgentOS API contract (v1).
//!
//! See `project_agentos_api_contract.md` for the wire spec. This crate
//! exposes [`build_router`] which mounts `POST /v1/messages` over an
//! axum [`Router`], plus a [`ServerState`] handle that the bin (or
//! integration tests) wire up to an [`AgentPipeline`] + platform router.

pub mod handler;
pub mod idempotency;
pub mod sse;
pub mod state;

pub use state::ServerState;

use axum::Router;
use std::sync::Arc;

/// Build the axum router for the v1 contract.
///
/// All routes mount under `/v1/*`. The shared state carries the platform
/// router (for materializing instances) and the auth token.
pub fn build_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route(
            "/v1/messages",
            axum::routing::post(handler::post_messages),
        )
        .with_state(state)
}
