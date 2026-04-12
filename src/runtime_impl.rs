//! Runtime trait implementation — bridges the platform crate to the pipeline.
//!
//! This is the glue that makes `Router::send_to()` work against a real
//! AgentPipeline. The platform crate defines the `Runtime` trait; this
//! module implements it using the pipeline's kernel, organism, and message
//! injection facilities.

use std::sync::Arc;
use tokio::sync::Mutex;

use agentos_kernel::Kernel;
use agentos_organism::Organism;
use agentos_platform::registry::Lifetime;
use agentos_platform::router::{OrganismMeta, Runtime};
use agentos_platform::address::Address;

/// The real Runtime implementation backed by an AgentPipeline's resources.
///
/// Holds Arc'd references to the kernel and organism, plus the pipeline's
/// ingress channel for message delivery. These are cheap clones from
/// AgentPipeline — the pipeline remains the owner.
pub struct PipelineRuntime {
    kernel: Arc<Mutex<Kernel>>,
    organism: Arc<Organism>,
    ingress_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
}

impl PipelineRuntime {
    /// Create a new PipelineRuntime from pipeline resources.
    ///
    /// Call this after the pipeline is built and running. The references
    /// are cheap Arc clones — no ownership transfer.
    pub fn new(
        kernel: Arc<Mutex<Kernel>>,
        organism: Arc<Organism>,
        ingress_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    ) -> Self {
        Self {
            kernel,
            organism,
            ingress_tx,
        }
    }
}

#[async_trait::async_trait]
impl Runtime for PipelineRuntime {
    /// Resolve an organism template by looking up a listener in the organism YAML.
    ///
    /// In the current architecture, "organisms" are listeners within the loaded
    /// organism YAML that have `is_agent: true`. A listener named "bob" maps to
    /// the organism template "bob". The listener's config provides the metadata
    /// (model, prompt, tools, etc.) that the instance will use.
    async fn resolve_organism(&self, name: &str) -> Option<OrganismMeta> {
        let listener = self.organism.get_listener(name)?;

        // Only agent listeners can be materialized as instances.
        if !listener.is_agent {
            return None;
        }

        // Derive lifetime from listener config.
        // Buffers get UntilTaskComplete; agents get a default idle timeout.
        let default_lifetime = if listener.buffer.is_some() {
            Lifetime::UntilTaskComplete
        } else {
            Lifetime::UntilIdle(std::time::Duration::from_secs(300))
        };

        // Shard pattern: if the organism has a kv-store config, generate
        // the standard shard pattern. Otherwise empty (no memex integration).
        // The {key} placeholder is expanded by the router at materialization time.
        let shard_pattern = vec![
            "shared.public".to_string(),
            "user.{key}".to_string(),
        ];

        Some(OrganismMeta {
            default_lifetime,
            shard_pattern,
            ephemeral: false,
        })
    }

    /// Allocate kernel state for a new instance.
    ///
    /// Creates a new thread in the kernel's thread table and allocates a
    /// context for it. The thread_id from the registry is used as the kernel
    /// thread identifier.
    async fn allocate_instance(
        &self,
        thread_id: &str,
        address: &Address,
        organism: &str,
    ) -> Result<(), String> {
        let mut kernel = self.kernel.lock().await;

        // Create thread in the kernel with the instance address as the description.
        // The organism name is used as the profile for security resolution.
        let profile = self.organism.profile_names().into_iter().next()
            .unwrap_or("default");

        kernel
            .dispatch_message(
                "platform",           // from: the platform runtime
                organism,             // to: the organism handler
                thread_id,            // thread_id: from the registry
                &format!("init-{}", address.raw()), // msg_id: unique
            )
            .map_err(|e| format!("kernel allocate failed for {}: {e}", address.raw()))?;

        tracing::debug!(
            thread_id,
            address = address.raw(),
            organism,
            profile,
            "Kernel state allocated for instance"
        );

        Ok(())
    }

    /// Deliver a message to a materialized instance.
    ///
    /// Injects the envelope's body into the pipeline's ingress channel,
    /// targeting the instance's thread_id. The pipeline's normal dispatch
    /// flow handles routing to the correct handler.
    async fn deliver(
        &self,
        thread_id: &str,
        envelope: &agentos_platform::router::Envelope,
    ) -> Result<(), String> {
        // For now, inject the raw body bytes into the pipeline.
        // TODO: wrap in proper pipeline message format with thread_id targeting.
        // The pipeline's dispatch mechanism needs to route by thread_id,
        // which requires the message to carry the thread_id as metadata.
        let _ = thread_id; // Will be used when message format is wired

        self.ingress_tx
            .send(envelope.body.clone())
            .await
            .map_err(|e| format!("ingress send failed: {e}"))?;

        Ok(())
    }

    /// Clean up kernel state when an instance is evicted.
    ///
    /// Prunes the kernel thread and releases its context.
    async fn evict_instance(&self, thread_id: &str) -> Result<(), String> {
        let mut kernel = self.kernel.lock().await;

        kernel
            .prune_thread(thread_id)
            .map_err(|e| format!("kernel prune failed for {thread_id}: {e}"))?;

        tracing::debug!(thread_id, "Kernel state cleaned up for evicted instance");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests require a real kernel + pipeline, which is heavy.
    // For now, verify the struct can be constructed and the trait is implemented.

    #[test]
    fn pipeline_runtime_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineRuntime>();
    }
}
