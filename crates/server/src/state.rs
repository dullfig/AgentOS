//! Shared server state — the handle held by every request handler.

use std::sync::Arc;

use agentos_events::PipelineEvent;
use agentos_organism::Organism;
use agentos_pipeline::runtime_impl::PipelineRuntime;
use agentos_platform::concurrent::SharedRouter;
use tokio::sync::broadcast;

use crate::idempotency::IdempotencyCache;

/// State shared across all axum handlers.
///
/// Holds the platform router (for routing chat messages to materialized
/// agent instances), a broadcast subscription factory for streaming
/// agent events back over SSE, the organism (so handlers can look up
/// the chat agent's payload tag), and the static bearer token for auth.
pub struct ServerState {
    /// The platform router. Concurrent — clones cheaply.
    pub router: Arc<SharedRouter<PipelineRuntime>>,
    /// Broadcast sender for pipeline events. Handlers `subscribe()` to
    /// receive events for a specific thread_id.
    pub events: broadcast::Sender<PipelineEvent>,
    /// The loaded organism. Handlers consult it to resolve the chat
    /// agent's payload tag for envelope wrapping.
    pub organism: Arc<Organism>,
    /// Listener that handles `/v1/messages` traffic. Default `"bob"`
    /// per the contract; overridable for tests + alternate deployments.
    pub agent_name: String,
    /// Static bearer token. Requests must present `Authorization: Bearer <token>`.
    pub auth_token: String,
    /// 24h in-memory idempotency cache. Same (service_token,
    /// idempotency_key) → replay cached SSE stream. See
    /// `crate::idempotency` for the design.
    pub idempotency: Arc<IdempotencyCache>,
}
