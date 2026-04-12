//! Message router — materialization-on-routing.
//!
//! The router is the single entry point for delivering messages to agent instances.
//! When a message is addressed to an instance that doesn't exist, the router
//! materializes it (loads the organism template, allocates kernel state, loads
//! KV shards) before delivering the message. This is the "send_to creates
//! anything missing along the path" semantic.
//!
//! # Design
//!
//! The router defines a [`Runtime`] trait that the pipeline (or any other
//! host) implements. The trait provides the operations the router needs
//! (resolve organism, allocate context, deliver message) without the router
//! depending on the pipeline's concrete types. This avoids the circular
//! dependency between platform and pipeline.
//!
//! # Usage
//!
//! ```ignore
//! let result = router.send_to(address, message, &mut runtime).await;
//! ```

use crate::address::{Address, AddressError};
use crate::registry::{
    InstanceInfo, InstanceRegistry, Lifetime, MaterializeOpts, RegistryError,
};

/// A message to be delivered to an agent instance.
///
/// Kept minimal and format-agnostic — the router doesn't parse message content.
/// The `body` is opaque bytes that the receiving instance interprets.
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Target address.
    pub to: Address,
    /// Source address (who sent this). None for external/trigger sources.
    pub from: Option<Address>,
    /// Opaque message body. Could be JSON, could be pipeline message bytes.
    pub body: Vec<u8>,
    /// Optional buffer name to deliver into. If None, uses the default buffer.
    pub buffer: Option<String>,
}

/// Errors from the routing layer.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("address error: {0}")]
    Address(#[from] AddressError),

    #[error("registry error: {0}")]
    Registry(#[from] RegistryError),

    #[error("organism not found: {0}")]
    OrganismNotFound(String),

    #[error("materialization failed for {address}: {reason}")]
    MaterializationFailed { address: String, reason: String },

    #[error("delivery failed for {address}: {reason}")]
    DeliveryFailed { address: String, reason: String },

    #[error("namespace violation: {0} cannot reach {1}")]
    NamespaceViolation(String, String),
}

/// Trait that the pipeline (or any host) implements to provide the operations
/// the router needs. This is the seam between the platform crate and the
/// pipeline — the platform defines the interface, the pipeline implements it.
///
/// All methods are async because they may involve I/O (loading organisms from
/// disk, loading KV shards from sled, etc.).
#[async_trait::async_trait]
pub trait Runtime: Send + Sync {
    /// Resolve an organism template by name. Returns the organism's default
    /// lifetime and cache shard pattern, or None if the organism doesn't exist.
    async fn resolve_organism(&self, name: &str) -> Option<OrganismMeta>;

    /// Allocate kernel state for a new instance (thread, context, etc.).
    /// The thread_id from the registry is passed for coordination.
    async fn allocate_instance(
        &self,
        thread_id: &str,
        address: &Address,
        organism: &str,
    ) -> Result<(), String>;

    /// Deliver a message to a materialized instance.
    async fn deliver(&self, thread_id: &str, envelope: &Envelope) -> Result<(), String>;

    /// Called when an instance is evicted (idle timeout, explicit kill).
    /// The runtime should clean up kernel state, flush KV, etc.
    async fn evict_instance(&self, thread_id: &str) -> Result<(), String>;
}

/// Metadata about an organism template, returned by [`Runtime::resolve_organism`].
#[derive(Debug, Clone)]
pub struct OrganismMeta {
    /// The organism's declared default lifetime for instances.
    pub default_lifetime: Lifetime,
    /// Default cache shard pattern. `{key}` is replaced with the instance key.
    /// Example: `["shared.public", "shared.wiki", "user.{key}"]`
    pub shard_pattern: Vec<String>,
    /// Whether this organism's instances are ephemeral by default.
    pub ephemeral: bool,
}

/// The message router — send_to with materialization-on-routing.
pub struct Router {
    registry: InstanceRegistry,
}

impl Router {
    /// Create a new router with the given registry.
    pub fn new(registry: InstanceRegistry) -> Self {
        Self { registry }
    }

    /// Access the underlying registry (for inspection, admin tools, etc.).
    pub fn registry(&self) -> &InstanceRegistry {
        &self.registry
    }

    /// Mutable access to the registry.
    pub fn registry_mut(&mut self) -> &mut InstanceRegistry {
        &mut self.registry
    }

    /// Send a message to an address, materializing the target if needed.
    ///
    /// This is the core routing primitive. The entire agent orchestration
    /// model collapses to this one method:
    ///
    /// 1. Parse and validate the address
    /// 2. Check namespace boundaries (if `from` is set)
    /// 3. Look up the instance in the registry
    /// 4. If not found → materialize (resolve organism, allocate, register)
    /// 5. Touch the instance (update timestamp, promote if Shelved)
    /// 6. Deliver the message
    pub async fn send_to(
        &mut self,
        envelope: &Envelope,
        runtime: &dyn Runtime,
    ) -> Result<(), RouterError> {
        let address = &envelope.to;

        // Namespace check: if we know the sender, verify they can reach the target.
        if let Some(ref from) = envelope.from {
            self.check_namespace(from, address)?;
        }

        // Materialize if needed.
        if !self.registry.is_materialized(address) {
            self.materialize_for(address, runtime).await?;
        }

        // Touch — update timestamp, promote Shelved → Active.
        self.registry.touch(address).map_err(RouterError::Registry)?;

        // Deliver.
        let info = self
            .registry
            .lookup(address)
            .ok_or_else(|| RouterError::DeliveryFailed {
                address: address.raw().to_string(),
                reason: "instance vanished between materialize and deliver".into(),
            })?;

        runtime
            .deliver(&info.thread_id, envelope)
            .await
            .map_err(|reason| RouterError::DeliveryFailed {
                address: address.raw().to_string(),
                reason,
            })?;

        Ok(())
    }

    /// Materialize an instance for the given address.
    async fn materialize_for(
        &mut self,
        address: &Address,
        runtime: &dyn Runtime,
    ) -> Result<(), RouterError> {
        let organism_name = address.organism();

        // Resolve the organism template.
        let meta = runtime
            .resolve_organism(organism_name)
            .await
            .ok_or_else(|| RouterError::OrganismNotFound(organism_name.to_string()))?;

        // Build cache shard names from the pattern.
        let instance_key = address.instance_key().unwrap_or("<global>");
        let cache_shards: Vec<String> = meta
            .shard_pattern
            .iter()
            .map(|pat| pat.replace("{key}", instance_key))
            .collect();

        // Determine lifetime.
        let lifetime = if meta.ephemeral || address.is_ephemeral() {
            Lifetime::Ephemeral
        } else {
            meta.default_lifetime
        };

        // Register in the registry.
        let opts = MaterializeOpts {
            organism: organism_name.to_string(),
            lifetime,
            parent: None, // TODO: derive from address hierarchy or envelope.from
            cache_shards,
        };

        let thread_id = self
            .registry
            .materialize(address.clone(), opts)
            .map_err(RouterError::Registry)?;

        // Allocate kernel state via the runtime.
        runtime
            .allocate_instance(&thread_id, address, organism_name)
            .await
            .map_err(|reason| RouterError::MaterializationFailed {
                address: address.raw().to_string(),
                reason,
            })?;

        tracing::info!(
            address = address.raw(),
            organism = organism_name,
            thread_id = &thread_id,
            "Instance materialized"
        );

        Ok(())
    }

    /// Run idle eviction across all instances.
    /// Returns addresses that were evicted, after calling runtime.evict_instance for each.
    pub async fn evict_idle(&mut self, runtime: &dyn Runtime) -> Vec<Address> {
        let evicted = self.registry.evict_idle();

        // Best-effort cleanup — if evict_instance fails, the instance is still
        // removed from the registry (it timed out, we're not going to keep it).
        for addr in &evicted {
            if let Some(info) = self.registry.lookup(addr) {
                let _ = runtime.evict_instance(&info.thread_id).await;
            }
        }

        evicted
    }

    /// Kill a specific instance and clean up via the runtime.
    pub async fn kill(
        &mut self,
        address: &Address,
        runtime: &dyn Runtime,
    ) -> Result<InstanceInfo, RouterError> {
        let info = self.registry.kill(address).map_err(RouterError::Registry)?;

        runtime
            .evict_instance(&info.thread_id)
            .await
            .map_err(|reason| RouterError::DeliveryFailed {
                address: address.raw().to_string(),
                reason,
            })?;

        tracing::info!(
            address = address.raw(),
            thread_id = &info.thread_id,
            "Instance killed"
        );

        Ok(info)
    }

    /// Check namespace boundary: source must share a namespace prefix with target,
    /// OR source must be in a parent namespace (admin can reach into child namespaces).
    fn check_namespace(&self, from: &Address, to: &Address) -> Result<(), RouterError> {
        let from_ns = from.namespace();
        let to_ns = to.namespace();

        match (from_ns, to_ns) {
            // Both un-namespaced — no boundary to enforce.
            (None, None) => Ok(()),
            // Source is un-namespaced (root/admin) — can reach anything.
            (None, Some(_)) => Ok(()),
            // Source is namespaced, target is un-namespaced — blocked.
            (Some(_), None) => Err(RouterError::NamespaceViolation(
                from.raw().to_string(),
                to.raw().to_string(),
            )),
            // Both namespaced — must match.
            (Some(s), Some(t)) => {
                if s == t {
                    Ok(())
                } else {
                    Err(RouterError::NamespaceViolation(
                        from.raw().to_string(),
                        to.raw().to_string(),
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A mock runtime for testing the router without a real pipeline.
    struct MockRuntime {
        organisms: HashMap<String, OrganismMeta>,
        allocated: Mutex<Vec<String>>,   // thread_ids that were allocated
        delivered: Mutex<Vec<String>>,    // thread_ids that received messages
        evicted: Mutex<Vec<String>>,      // thread_ids that were evicted
    }

    use std::collections::HashMap;

    impl MockRuntime {
        fn new() -> Self {
            let mut organisms = HashMap::new();
            organisms.insert(
                "concierge".to_string(),
                OrganismMeta {
                    default_lifetime: Lifetime::UntilIdle(std::time::Duration::from_secs(300)),
                    shard_pattern: vec![
                        "shared.public".into(),
                        "shared.wiki".into(),
                        "user.{key}".into(),
                    ],
                    ephemeral: false,
                },
            );
            organisms.insert(
                "scratch-bot".to_string(),
                OrganismMeta {
                    default_lifetime: Lifetime::Ephemeral,
                    shard_pattern: vec![],
                    ephemeral: true,
                },
            );

            Self {
                organisms,
                allocated: Mutex::new(vec![]),
                delivered: Mutex::new(vec![]),
                evicted: Mutex::new(vec![]),
            }
        }
    }

    #[async_trait::async_trait]
    impl Runtime for MockRuntime {
        async fn resolve_organism(&self, name: &str) -> Option<OrganismMeta> {
            self.organisms.get(name).cloned()
        }

        async fn allocate_instance(
            &self,
            thread_id: &str,
            _address: &Address,
            _organism: &str,
        ) -> Result<(), String> {
            self.allocated.lock().unwrap().push(thread_id.to_string());
            Ok(())
        }

        async fn deliver(&self, thread_id: &str, _envelope: &Envelope) -> Result<(), String> {
            self.delivered.lock().unwrap().push(thread_id.to_string());
            Ok(())
        }

        async fn evict_instance(&self, thread_id: &str) -> Result<(), String> {
            self.evicted.lock().unwrap().push(thread_id.to_string());
            Ok(())
        }
    }

    fn envelope(to: &str) -> Envelope {
        Envelope {
            to: Address::parse(to).unwrap(),
            from: None,
            body: b"hello".to_vec(),
            buffer: None,
        }
    }

    fn envelope_from(from: &str, to: &str) -> Envelope {
        Envelope {
            to: Address::parse(to).unwrap(),
            from: Some(Address::parse(from).unwrap()),
            body: b"hello".to_vec(),
            buffer: None,
        }
    }

    #[tokio::test]
    async fn send_materializes_on_first_message() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        // First message to bob[alice] — should materialize
        router.send_to(&envelope("concierge[alice]"), &runtime).await.unwrap();

        assert!(router.registry().is_materialized(&Address::parse("concierge[alice]").unwrap()));
        assert_eq!(runtime.allocated.lock().unwrap().len(), 1);
        assert_eq!(runtime.delivered.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn second_message_reuses_instance() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        router.send_to(&envelope("concierge[alice]"), &runtime).await.unwrap();
        router.send_to(&envelope("concierge[alice]"), &runtime).await.unwrap();

        // Only one allocation, two deliveries
        assert_eq!(runtime.allocated.lock().unwrap().len(), 1);
        assert_eq!(runtime.delivered.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn unknown_organism_fails() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        let result = router.send_to(&envelope("nonexistent[alice]"), &runtime).await;
        assert!(matches!(result, Err(RouterError::OrganismNotFound(_))));
    }

    #[tokio::test]
    async fn cache_shards_expanded_from_pattern() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        router.send_to(&envelope("concierge[alice]"), &runtime).await.unwrap();

        let info = router
            .registry()
            .lookup(&Address::parse("concierge[alice]").unwrap())
            .unwrap();

        // Pattern was ["shared.public", "shared.wiki", "user.{key}"]
        // Key is "alice", so shards should be:
        assert_eq!(
            info.cache_shards,
            vec!["shared.public", "shared.wiki", "user.alice"]
        );
    }

    #[tokio::test]
    async fn ephemeral_from_organism() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        router
            .send_to(&envelope("scratch-bot[query-1]"), &runtime)
            .await
            .unwrap();

        let info = router
            .registry()
            .lookup(&Address::parse("scratch-bot[query-1]").unwrap())
            .unwrap();

        assert_eq!(info.lifetime, Lifetime::Ephemeral);
    }

    #[tokio::test]
    async fn ephemeral_from_address() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        // "scratch" in the address path makes it ephemeral regardless of organism
        router
            .send_to(&envelope("scratch.concierge[query-1]"), &runtime)
            .await
            .unwrap();

        let info = router
            .registry()
            .lookup(&Address::parse("scratch.concierge[query-1]").unwrap())
            .unwrap();

        assert_eq!(info.lifetime, Lifetime::Ephemeral);
    }

    #[tokio::test]
    async fn namespace_same_namespace_allowed() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        let env = envelope_from("ringhub.concierge[alice]", "ringhub.concierge[bob]");
        router.send_to(&env, &runtime).await.unwrap();
    }

    #[tokio::test]
    async fn namespace_cross_namespace_blocked() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        let env = envelope_from("ringhub.concierge[alice]", "chorus.concierge[sarah]");
        let result = router.send_to(&env, &runtime).await;
        assert!(matches!(result, Err(RouterError::NamespaceViolation(_, _))));
    }

    #[tokio::test]
    async fn namespace_root_can_reach_namespaced() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        // Un-namespaced source (root/admin) can reach namespaced target
        let env = envelope_from("admin", "ringhub.concierge[alice]");
        router.send_to(&env, &runtime).await.unwrap();
    }

    #[tokio::test]
    async fn namespace_namespaced_cannot_reach_root() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        let env = envelope_from("ringhub.concierge[alice]", "admin");
        // "admin" is not a known organism but the namespace check happens first
        let result = router.send_to(&env, &runtime).await;
        assert!(matches!(result, Err(RouterError::NamespaceViolation(_, _))));
    }

    #[tokio::test]
    async fn kill_cleans_up() {
        let reg = InstanceRegistry::new(0);
        let mut router = Router::new(reg);
        let runtime = MockRuntime::new();

        router.send_to(&envelope("concierge[alice]"), &runtime).await.unwrap();

        let addr = Address::parse("concierge[alice]").unwrap();
        let killed = router.kill(&addr, &runtime).await.unwrap();

        assert_eq!(killed.organism, "concierge");
        assert!(!router.registry().is_materialized(&addr));
        assert_eq!(runtime.evicted.lock().unwrap().len(), 1);
    }
}
