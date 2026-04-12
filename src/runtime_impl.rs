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

// ── Trigger → Router integration ──

use std::collections::HashMap;
use agentos_platform::address::Address as PlatformAddress;
use agentos_platform::concurrent::SharedRouter;
use agentos_platform::router::Envelope;
use agentos_platform::template;
use agentos_trigger::TriggerEvent;
use agentos_trigger::runtime::TriggerPayload;

/// Build a variable map from a TriggerEvent for template expansion.
fn trigger_vars(event: &TriggerEvent) -> HashMap<String, String> {
    let mut vars = HashMap::new();
    vars.insert("trigger.name".to_string(), event.trigger_name.clone());
    vars.insert("trigger.target".to_string(), event.target.clone());

    match &event.payload {
        TriggerPayload::FileChanged { paths } => {
            if let Some(first) = paths.first() {
                vars.insert("event.path".to_string(), first.clone());
            }
            vars.insert("event.paths".to_string(), paths.join(","));
        }
        TriggerPayload::Tick => {
            vars.insert("event.type".to_string(), "tick".to_string());
        }
        TriggerPayload::Event { event_name, from } => {
            vars.insert("event.event_name".to_string(), event_name.clone());
            if let Some(f) = from {
                vars.insert("event.from".to_string(), f.to_string());
            }
        }
        TriggerPayload::Rhai { result } => {
            vars.insert("event.result".to_string(), result.clone());
        }
    }

    vars
}

/// Convert a TriggerEvent into an Envelope for the platform router.
///
/// Uses the TriggerConfig's `send_to` as the target address template,
/// and `message` as the body template. Both are expanded with variables
/// from the trigger event.
///
/// Returns None if the trigger config doesn't have `send_to` (old-style
/// trigger that uses raw pipeline dispatch instead of the router).
pub fn trigger_to_envelope(
    event: &TriggerEvent,
    send_to_template: &str,
    message_template: Option<&str>,
    source_namespace: Option<&str>,
) -> Option<Envelope> {
    let vars = trigger_vars(event);

    let address_str = template::expand(send_to_template, &vars);
    let address = match PlatformAddress::parse(&address_str) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(
                trigger = &event.trigger_name,
                address = &address_str,
                error = %e,
                "Trigger send_to address failed to parse after expansion"
            );
            return None;
        }
    };

    let body = message_template
        .map(|t| template::expand(t, &vars))
        .unwrap_or_else(|| format!("Trigger {} fired", event.trigger_name));

    let from = source_namespace.and_then(|ns| {
        PlatformAddress::parse(&format!("{ns}.trigger.{}", event.trigger_name)).ok()
    });

    Some(Envelope {
        to: address,
        from,
        body: body.into_bytes(),
        buffer: None,
    })
}

/// Process trigger events in a loop, routing through the SharedRouter.
///
/// This is the bridge between the trigger runtime (which produces TriggerEvents)
/// and the platform router (which materializes instances and delivers messages).
///
/// Spawn this as a tokio task after the pipeline is running:
///
/// ```ignore
/// let trigger_rx = trigger_runtime.take_receiver().unwrap();
/// tokio::spawn(process_trigger_events(trigger_rx, organism, shared_router));
/// ```
pub async fn process_trigger_events<R: Runtime + 'static>(
    mut trigger_rx: tokio::sync::mpsc::Receiver<TriggerEvent>,
    organism: Arc<Organism>,
    shared_router: Arc<SharedRouter<R>>,
) {
    while let Some(event) = trigger_rx.recv().await {
        // Look up the trigger's config from the organism to get send_to/message.
        let config = organism
            .get_listener(&event.trigger_name)
            .and_then(|l| l.trigger.as_ref());

        let (send_to, message, source_ns) = match config {
            Some(cfg) => (
                cfg.send_to.as_deref(),
                cfg.message.as_deref(),
                cfg.source_namespace.as_deref(),
            ),
            None => {
                tracing::warn!(
                    trigger = &event.trigger_name,
                    "Trigger fired but no config found in organism"
                );
                continue;
            }
        };

        // If send_to is present, route through the platform router.
        if let Some(send_to_tmpl) = send_to {
            if let Some(envelope) = trigger_to_envelope(&event, send_to_tmpl, message, source_ns) {
                match shared_router.send_to(&envelope).await {
                    Ok(()) => {
                        tracing::info!(
                            trigger = &event.trigger_name,
                            address = envelope.to.raw(),
                            "Trigger routed through platform"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            trigger = &event.trigger_name,
                            address = envelope.to.raw(),
                            error = %e,
                            "Trigger routing failed"
                        );
                    }
                }
            }
        } else {
            // Old-style trigger: no send_to, uses raw pipeline dispatch.
            // TODO: inject into pipeline ingress channel as before.
            tracing::debug!(
                trigger = &event.trigger_name,
                target = &event.target,
                "Trigger fired (legacy dispatch, not routed through platform)"
            );
        }
    }

    tracing::info!("Trigger event processing loop ended");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_runtime_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PipelineRuntime>();
    }

    #[test]
    fn trigger_vars_tick() {
        let event = TriggerEvent {
            trigger_name: "morning_digest".into(),
            target: "bob".into(),
            payload: TriggerPayload::Tick,
        };
        let vars = trigger_vars(&event);
        assert_eq!(vars["trigger.name"], "morning_digest");
        assert_eq!(vars["event.type"], "tick");
    }

    #[test]
    fn trigger_vars_event() {
        let event = TriggerEvent {
            trigger_name: "handle_dm".into(),
            target: "concierge".into(),
            payload: TriggerPayload::Event {
                event_name: "dm.received".into(),
                from: Some("ringhub".into()),
            },
        };
        let vars = trigger_vars(&event);
        assert_eq!(vars["event.event_name"], "dm.received");
        assert_eq!(vars["event.from"], "ringhub");
    }

    #[test]
    fn trigger_to_envelope_expands_templates() {
        let event = TriggerEvent {
            trigger_name: "handle_dm".into(),
            target: "concierge".into(),
            payload: TriggerPayload::Rhai {
                result: "alice".into(),
            },
        };

        let envelope = trigger_to_envelope(
            &event,
            "ringhub.concierge[{event.result}].dm",
            Some("Hello from trigger {trigger.name}"),
            None,
        )
        .unwrap();

        assert_eq!(envelope.to.raw(), "ringhub.concierge[alice].dm");
        assert_eq!(
            String::from_utf8(envelope.body).unwrap(),
            "Hello from trigger handle_dm"
        );
        assert!(envelope.from.is_none()); // no source namespace
    }

    #[test]
    fn trigger_to_envelope_with_namespace() {
        let event = TriggerEvent {
            trigger_name: "my_reminder".into(),
            target: "reminder-bot".into(),
            payload: TriggerPayload::Tick,
        };

        let envelope = trigger_to_envelope(
            &event,
            "user.alice.reminder-bot[hourly]",
            None,
            Some("user.alice"),
        )
        .unwrap();

        assert_eq!(envelope.to.raw(), "user.alice.reminder-bot[hourly]");
        // from address includes the source namespace + trigger name
        let from = envelope.from.unwrap();
        assert!(from.raw().starts_with("user.alice.trigger."));
    }

    #[test]
    fn trigger_to_envelope_bad_address() {
        let event = TriggerEvent {
            trigger_name: "broken".into(),
            target: "x".into(),
            payload: TriggerPayload::Tick,
        };

        // Empty address after expansion → parse fails → returns None
        let result = trigger_to_envelope(&event, "", None, None);
        assert!(result.is_none());
    }
}
