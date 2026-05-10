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
    /// The platform mints `thread_id` registry-side; the kernel records
    /// it durably via `register_platform_thread`, which writes a single
    /// WAL batch (ThreadCreate + ContextAllocate). On crash recovery the
    /// thread + context replay back under the same id, matching what the
    /// platform registry's snapshot remembers.
    ///
    /// Profile inheritance: uses the organism's first declared profile.
    /// In single-tenant deployments there is exactly one profile; in
    /// multi-tenant deployments this will be replaced by per-namespace
    /// profile resolution at materialization time.
    async fn allocate_instance(
        &self,
        thread_id: &str,
        address: &Address,
        organism: &str,
    ) -> Result<(), String> {
        let profile = self
            .organism
            .profile_names()
            .into_iter()
            .next()
            .unwrap_or("default")
            .to_string();

        let mut kernel = self.kernel.lock().await;
        kernel
            .register_platform_thread(thread_id, organism, &profile)
            .map_err(|e| format!("kernel allocate failed for {thread_id}: {e}"))?;

        tracing::debug!(
            thread_id,
            address = address.raw(),
            organism,
            profile = %profile,
            "Kernel state allocated for instance"
        );

        Ok(())
    }

    /// Deliver a message to a materialized instance.
    ///
    /// Wraps the envelope body in a rust-pipeline envelope (carrying
    /// `from`/`to`/`thread`) and injects it into the pipeline ingress.
    /// The pipeline's normal parse → validate → route → dispatch flow
    /// handles delivery to the listener handler.
    ///
    /// `to` is derived from the address's organism segment — that's the
    /// listener name the pipeline will route to. The platform's address
    /// hierarchy (namespaces, instance keys, buffers) is collapsed at
    /// this seam to a single listener name; the registry preserves the
    /// rest of the addressing structure for materialization.
    async fn deliver(
        &self,
        thread_id: &str,
        envelope: &agentos_platform::router::Envelope,
    ) -> Result<(), String> {
        let to = envelope.to.organism();
        let from = envelope
            .from
            .as_ref()
            .map(|a| a.raw())
            .unwrap_or("platform");

        let raw = rust_pipeline::prelude::build_envelope(from, to, thread_id, &envelope.body)
            .map_err(|e| format!("envelope build failed: {e}"))?;

        self.ingress_tx
            .send(raw)
            .await
            .map_err(|e| format!("ingress send failed: {e}"))?;

        Ok(())
    }

    /// Clean up kernel state when an instance is evicted.
    ///
    /// `evict_platform_thread` writes ThreadCleanup + ContextRelease as
    /// a single WAL batch so a crash mid-eviction either replays both
    /// or neither — matches the durability story of allocation.
    async fn evict_instance(&self, thread_id: &str) -> Result<(), String> {
        let mut kernel = self.kernel.lock().await;
        kernel
            .evict_platform_thread(thread_id)
            .map_err(|e| format!("kernel evict failed for {thread_id}: {e}"))?;

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

    // ── Integration: SharedRouter → AgentPipeline round-trip ──

    use agentos_organism::parser::parse_organism;
    use crate::AgentPipelineBuilder;
    use agentos_platform::address::Address as PlatformAddress;
    use rust_pipeline::prelude::{
        FnHandler, HandlerContext, HandlerResponse, ValidatedPayload,
    };
    use std::sync::Arc as StdArc;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    use tempfile::TempDir;

    /// End-to-end test: a message sent via `SharedRouter::send_to` must
    /// reach a real listener handler inside `AgentPipeline`. This is the
    /// proof that the Runtime trait is actually wired — every piece in
    /// between (registry materialize → kernel allocate → envelope build
    /// → pipeline ingress → parse/validate/route/dispatch) has to work.
    #[tokio::test]
    async fn send_to_routes_into_real_pipeline() {
        let yaml = r#"
organism:
  name: router-rt-test

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"
    agent:
      prompt: "stub"

profiles:
  default:
    linux_user: agentos
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = TempDir::new().unwrap();

        // Recorder captures whatever payload the handler received.
        let recorder: StdArc<StdMutex<Vec<String>>> = StdArc::new(StdMutex::new(vec![]));
        let recorder_clone = recorder.clone();

        let echo = FnHandler(move |p: ValidatedPayload, _ctx: HandlerContext| {
            let recorder = recorder_clone.clone();
            Box::pin(async move {
                recorder.lock().unwrap().push(String::from_utf8_lossy(&p.xml).into_owned());
                Ok(HandlerResponse::None)
            })
        });

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .build()
            .unwrap();

        // Initialize root, then start the pipeline so ingress_tx is live.
        let profile = pipeline
            .organism()
            .profile_names()
            .into_iter()
            .next()
            .unwrap_or("default")
            .to_string();
        pipeline.initialize_root("router-rt-test", &profile).await.unwrap();
        pipeline.run();

        // Build the SharedRouter over the running pipeline.
        let router = pipeline.shared_router(0, Duration::from_secs(60));

        // Send a routable message. Body is a valid payload XML that the
        // pipeline's parse + validate stages will accept; the handler
        // sees `p.xml` as that payload string.
        let envelope = agentos_platform::router::Envelope {
            to: PlatformAddress::parse("echo[alice]").unwrap(),
            from: None,
            body: b"<Greeting><text>hi</text></Greeting>".to_vec(),
            buffer: None,
        };
        router.send_to(&envelope).await.expect("send_to failed");

        // Pipeline dispatch is asynchronous — give it a moment to run.
        for _ in 0..50 {
            if !recorder.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let received = recorder.lock().unwrap().clone();
        assert_eq!(received.len(), 1, "handler should have received one message");
        assert!(
            received[0].contains("<text>hi</text>"),
            "handler payload should contain the original body, got: {}",
            received[0]
        );

        // Registry must have materialized echo[alice] and stored a thread_id.
        let info = router
            .list()
            .await
            .into_iter()
            .find(|i| i.address.raw() == "echo[alice]")
            .expect("echo[alice] not in registry");
        assert_eq!(info.organism, "echo");
        assert!(!info.thread_id.is_empty());

        pipeline.shutdown().await;
    }

    /// Second send to the same address must reuse the materialized
    /// instance and not double-allocate kernel state.
    #[tokio::test]
    async fn second_send_reuses_instance() {
        let yaml = r#"
organism:
  name: router-rt-test

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"
    agent:
      prompt: "stub"

profiles:
  default:
    linux_user: agentos
    listeners: [echo]
    journal: retain_forever
"#;
        let org = parse_organism(yaml).unwrap();
        let dir = TempDir::new().unwrap();

        let recorder: StdArc<StdMutex<Vec<String>>> = StdArc::new(StdMutex::new(vec![]));
        let recorder_clone = recorder.clone();
        let echo = FnHandler(move |p: ValidatedPayload, _ctx: HandlerContext| {
            let recorder = recorder_clone.clone();
            Box::pin(async move {
                let xml = String::from_utf8_lossy(&p.xml).into_owned();
                recorder.lock().unwrap().push(xml);
                Ok(HandlerResponse::None)
            })
        });

        let mut pipeline = AgentPipelineBuilder::new(org, &dir.path().join("data"))
            .register("echo", echo)
            .unwrap()
            .build()
            .unwrap();

        pipeline.initialize_root("router-rt-test", "default").await.unwrap();
        pipeline.run();

        let router = pipeline.shared_router(0, Duration::from_secs(60));

        let envelope = agentos_platform::router::Envelope {
            to: PlatformAddress::parse("echo[alice]").unwrap(),
            from: None,
            body: b"<Greeting><text>one</text></Greeting>".to_vec(),
            buffer: None,
        };
        router.send_to(&envelope).await.unwrap();

        let envelope2 = agentos_platform::router::Envelope {
            to: PlatformAddress::parse("echo[alice]").unwrap(),
            from: None,
            body: b"<Greeting><text>two</text></Greeting>".to_vec(),
            buffer: None,
        };
        router.send_to(&envelope2).await.unwrap();

        for _ in 0..50 {
            if recorder.lock().unwrap().len() >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        assert_eq!(recorder.lock().unwrap().len(), 2);
        assert_eq!(router.count().await, 1, "should be one instance, not two");

        pipeline.shutdown().await;
    }

    /// Restart-survives test: materialize an instance, shut down the
    /// pipeline, rebuild against the same data dir, and verify both
    /// the registry (snapshot) and the kernel (WAL) replay the same
    /// `bob[alice]` thread_id. Buffer thread_ids are deterministic, so
    /// a fresh message routes to the same buffer thread the first
    /// run created.
    #[tokio::test]
    async fn instance_survives_pipeline_restart() {
        let yaml = r#"
organism:
  name: restart-test

listeners:
  - name: echo
    payload_class: handlers.echo.Greeting
    handler: handlers.echo.handle
    description: "Echo handler"
    agent:
      prompt: "stub"

profiles:
  default:
    linux_user: agentos
    listeners: [echo]
    journal: retain_forever
"#;
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        // ── Round 1: build, materialize echo[alice], shut down ──
        let (instance_thread_id, buffer_thread_id) = {
            let org = parse_organism(yaml).unwrap();
            let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
                Box::pin(async move {
                    Ok(HandlerResponse::Reply { payload_xml: p.xml })
                })
            });
            let mut pipeline = AgentPipelineBuilder::new(org, &data_dir)
                .register("echo", echo)
                .unwrap()
                .build()
                .unwrap();
            pipeline.initialize_root("restart-test", "default").await.unwrap();
            pipeline.run();

            let router = pipeline.shared_router(0, Duration::from_secs(60));
            let envelope = agentos_platform::router::Envelope {
                to: PlatformAddress::parse("echo[alice]").unwrap(),
                from: None,
                body: b"<Greeting><text>one</text></Greeting>".to_vec(),
                buffer: None,
            };
            router.send_to(&envelope).await.unwrap();

            let info = router
                .list()
                .await
                .into_iter()
                .find(|i| i.address.raw() == "echo[alice]")
                .expect("echo[alice] must materialize");
            let buf_thread = info
                .buffers
                .get(&agentos_platform::buffers::BufferId::default_buffer())
                .expect("default buffer must exist after first send")
                .thread_id
                .clone();
            (info.thread_id.clone(), buf_thread)
        }; // pipeline + router drop here; tokio tasks finish

        // Confirm the snapshot file exists where shared_router placed it.
        assert!(
            data_dir.join("platform_registry.json").exists(),
            "registry snapshot should be on disk"
        );

        // ── Round 2: rebuild against the same data dir ──
        let org = parse_organism(yaml).unwrap();
        let echo = FnHandler(|p: ValidatedPayload, _ctx: HandlerContext| {
            Box::pin(async move {
                Ok(HandlerResponse::Reply { payload_xml: p.xml })
            })
        });
        let mut pipeline = AgentPipelineBuilder::new(org, &data_dir)
            .register("echo", echo)
            .unwrap()
            .build()
            .unwrap();
        pipeline.run();
        let router = pipeline.shared_router(0, Duration::from_secs(60));

        // Registry: bob[alice] is back at the same thread_id, no fresh send needed.
        let info = router
            .list()
            .await
            .into_iter()
            .find(|i| i.address.raw() == "echo[alice]")
            .expect("echo[alice] must replay from snapshot");
        assert_eq!(
            info.thread_id, instance_thread_id,
            "instance thread_id should survive restart"
        );

        // Kernel: the platform thread is back from WAL replay.
        {
            let kernel = pipeline.kernel();
            let kernel = kernel.lock().await;
            assert!(
                kernel.threads().lookup(&instance_thread_id).is_some(),
                "kernel WAL replay must restore the platform thread"
            );
            assert!(
                kernel.contexts().exists(&instance_thread_id),
                "kernel WAL replay must restore the platform context"
            );
        }

        // Send another message — buffer thread_id is deterministic so it
        // matches the original. WAL'd context segments under that thread
        // light back up.
        let envelope = agentos_platform::router::Envelope {
            to: PlatformAddress::parse("echo[alice]").unwrap(),
            from: None,
            body: b"<Greeting><text>still here?</text></Greeting>".to_vec(),
            buffer: None,
        };
        router.send_to(&envelope).await.unwrap();

        let info = router
            .list()
            .await
            .into_iter()
            .find(|i| i.address.raw() == "echo[alice]")
            .unwrap();
        let buf_thread_2 = info
            .buffers
            .get(&agentos_platform::buffers::BufferId::default_buffer())
            .unwrap()
            .thread_id
            .clone();
        assert_eq!(
            buf_thread_2, buffer_thread_id,
            "buffer thread_id must be deterministic across restart"
        );

        pipeline.shutdown().await;
    }
}
