//! Instance registry — maps addresses to live agent instances.
//!
//! The registry is a thin in-memory index over kernel contexts. It tracks
//! which agent instances are materialized, their lifecycle state, and their
//! metadata. The kernel owns the actual durable state; the registry provides
//! the addressing and lifecycle layer on top.
//!
//! # Materialization
//!
//! Instances are materialized lazily on first access via [`InstanceRegistry::materialize`].
//! The registry enforces a per-address mutex to prevent double-materialization
//! when concurrent messages arrive for the same address.
//!
//! # VMM Tiering
//!
//! Instances follow the kernel's VMM metaphor:
//! - **Active** — in working memory, processing messages
//! - **Shelved** — idle, compressed, can be reactivated quickly
//! - **Folded** — on disk only, requires full reload
//!
//! Transitions are driven by idle timeouts and memory pressure.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::address::Address;
use crate::buffers::BufferStore;

/// Lifecycle policy for an agent instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lifetime {
    /// Evict after being idle for the given duration.
    UntilIdle(Duration),
    /// Stay alive until explicitly killed (e.g., a task that runs until done).
    UntilTaskComplete,
    /// Never evict — pinned in memory. Used for shared shards and critical services.
    Pinned,
    /// Never persist to disk — exists only in memory, dies with the process.
    /// Used for scratch pads, latent loops, one-shot queries.
    Ephemeral,
}

impl Default for Lifetime {
    fn default() -> Self {
        Lifetime::UntilIdle(Duration::from_secs(300)) // 5 minutes
    }
}

/// Current tier of an instance in the VMM hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// In working memory, actively processing or recently active.
    Active,
    /// Idle, compressed in memory, can be reactivated quickly.
    Shelved,
    /// On disk only, requires full reload to become Active.
    Folded,
}

/// Metadata about a live agent instance.
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// The full address of this instance.
    pub address: Address,
    /// The organism template this instance was materialized from.
    pub organism: String,
    /// The kernel thread_id assigned to this instance.
    pub thread_id: String,
    /// Lifecycle policy.
    pub lifetime: Lifetime,
    /// Current VMM tier.
    pub tier: Tier,
    /// Parent instance address, if spawned by another instance.
    pub parent: Option<Address>,
    /// Cache shard names for memex integration (empty if no memex).
    pub cache_shards: Vec<String>,
    /// When this instance was first materialized.
    pub created_at: Instant,
    /// When this instance was last accessed (message received or sent).
    pub last_accessed: Instant,
    /// Per-channel buffer store — isolates DM, public, help, task conversations.
    pub buffers: BufferStore,
}

/// Options for materializing a new instance.
#[derive(Debug, Clone)]
pub struct MaterializeOpts {
    /// Organism template to use.
    pub organism: String,
    /// Lifecycle policy.
    pub lifetime: Lifetime,
    /// Parent instance, if this is a child.
    pub parent: Option<Address>,
    /// Cache shard names for memex retrieval context.
    pub cache_shards: Vec<String>,
}

/// Errors from the instance registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("instance already exists: {0}")]
    AlreadyExists(String),

    #[error("instance not found: {0}")]
    NotFound(String),

    #[error("organism template not found: {0}")]
    OrganismNotFound(String),

    #[error("instance limit reached ({0} instances)")]
    LimitReached(usize),
}

/// The instance registry — maps addresses to live agent instances.
///
/// The registry is an in-memory index. It does NOT own kernel state — it
/// holds metadata about which addresses are materialized and their lifecycle.
/// The kernel owns the actual thread contexts and durable state.
pub struct InstanceRegistry {
    /// Address (as raw string) → instance info.
    instances: HashMap<String, InstanceInfo>,
    /// Maximum number of concurrent instances (0 = unlimited).
    max_instances: usize,
    /// Counter for generating unique thread_ids.
    next_thread_id: u64,
}

impl InstanceRegistry {
    /// Create a new empty registry.
    pub fn new(max_instances: usize) -> Self {
        Self {
            instances: HashMap::new(),
            max_instances,
            next_thread_id: 1,
        }
    }

    /// Look up an instance by address. Returns None if not materialized.
    pub fn lookup(&self, address: &Address) -> Option<&InstanceInfo> {
        self.instances.get(address.raw())
    }

    /// Look up a mutable instance by address.
    pub fn lookup_mut(&mut self, address: &Address) -> Option<&mut InstanceInfo> {
        self.instances.get_mut(address.raw())
    }

    /// Check whether an address is currently materialized.
    pub fn is_materialized(&self, address: &Address) -> bool {
        self.instances.contains_key(address.raw())
    }

    /// Materialize a new instance at the given address.
    ///
    /// Returns the assigned thread_id. Fails if the address is already
    /// materialized or the instance limit is reached.
    pub fn materialize(
        &mut self,
        address: Address,
        opts: MaterializeOpts,
    ) -> Result<String, RegistryError> {
        let key = address.raw().to_string();

        if self.instances.contains_key(&key) {
            return Err(RegistryError::AlreadyExists(key));
        }

        if self.max_instances > 0 && self.instances.len() >= self.max_instances {
            return Err(RegistryError::LimitReached(self.max_instances));
        }

        let thread_id = format!("inst-{:06}", self.next_thread_id);
        self.next_thread_id += 1;

        let now = Instant::now();
        let info = InstanceInfo {
            address,
            organism: opts.organism,
            thread_id: thread_id.clone(),
            lifetime: opts.lifetime,
            tier: Tier::Active,
            parent: opts.parent,
            cache_shards: opts.cache_shards,
            created_at: now,
            last_accessed: now,
            buffers: BufferStore::new(),
        };

        self.instances.insert(key, info);
        Ok(thread_id)
    }

    /// Touch an instance — update its last_accessed timestamp.
    /// Also promotes Shelved instances back to Active.
    pub fn touch(&mut self, address: &Address) -> Result<(), RegistryError> {
        let info = self
            .instances
            .get_mut(address.raw())
            .ok_or_else(|| RegistryError::NotFound(address.raw().to_string()))?;

        info.last_accessed = Instant::now();
        if info.tier == Tier::Shelved {
            info.tier = Tier::Active;
        }
        Ok(())
    }

    /// Shelve an instance — transition Active → Shelved.
    pub fn shelve(&mut self, address: &Address) -> Result<(), RegistryError> {
        let info = self
            .instances
            .get_mut(address.raw())
            .ok_or_else(|| RegistryError::NotFound(address.raw().to_string()))?;

        if info.tier == Tier::Active {
            info.tier = Tier::Shelved;
        }
        Ok(())
    }

    /// Fold an instance — transition to Folded (disk only).
    pub fn fold(&mut self, address: &Address) -> Result<(), RegistryError> {
        let info = self
            .instances
            .get_mut(address.raw())
            .ok_or_else(|| RegistryError::NotFound(address.raw().to_string()))?;

        info.tier = Tier::Folded;
        Ok(())
    }

    /// Kill an instance — remove it from the registry entirely.
    /// Returns the removed instance info.
    pub fn kill(&mut self, address: &Address) -> Result<InstanceInfo, RegistryError> {
        self.instances
            .remove(address.raw())
            .ok_or_else(|| RegistryError::NotFound(address.raw().to_string()))
    }

    /// List all materialized instances.
    pub fn list(&self) -> Vec<&InstanceInfo> {
        self.instances.values().collect()
    }

    /// List instances filtered by tier.
    pub fn list_by_tier(&self, tier: Tier) -> Vec<&InstanceInfo> {
        self.instances
            .values()
            .filter(|i| i.tier == tier)
            .collect()
    }

    /// List instances filtered by organism template.
    pub fn list_by_organism(&self, organism: &str) -> Vec<&InstanceInfo> {
        self.instances
            .values()
            .filter(|i| i.organism == organism)
            .collect()
    }

    /// List children of a given parent address.
    pub fn children_of(&self, parent: &Address) -> Vec<&InstanceInfo> {
        let parent_raw = parent.raw();
        self.instances
            .values()
            .filter(|i| i.parent.as_ref().map(|p| p.raw()) == Some(parent_raw))
            .collect()
    }

    /// Evict idle instances based on their lifetime policy.
    /// Returns the addresses of evicted instances.
    pub fn evict_idle(&mut self) -> Vec<Address> {
        let now = Instant::now();
        let mut to_evict = Vec::new();

        for (key, info) in &self.instances {
            if let Lifetime::UntilIdle(timeout) = &info.lifetime {
                if now.duration_since(info.last_accessed) > *timeout {
                    to_evict.push(key.clone());
                }
            }
        }

        to_evict
            .iter()
            .filter_map(|key| {
                self.instances
                    .remove(key)
                    .map(|info| info.address)
            })
            .collect()
    }

    /// Total number of materialized instances.
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Count by tier.
    pub fn count_by_tier(&self) -> (usize, usize, usize) {
        let mut active = 0;
        let mut shelved = 0;
        let mut folded = 0;
        for info in self.instances.values() {
            match info.tier {
                Tier::Active => active += 1,
                Tier::Shelved => shelved += 1,
                Tier::Folded => folded += 1,
            }
        }
        (active, shelved, folded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> Address {
        Address::parse(s).unwrap()
    }

    fn default_opts(organism: &str) -> MaterializeOpts {
        MaterializeOpts {
            organism: organism.to_string(),
            lifetime: Lifetime::default(),
            parent: None,
            cache_shards: vec![],
        }
    }

    #[test]
    fn materialize_and_lookup() {
        let mut reg = InstanceRegistry::new(0);
        let a = addr("bob[alice]");

        let tid = reg.materialize(a.clone(), default_opts("concierge")).unwrap();
        assert!(tid.starts_with("inst-"));

        let info = reg.lookup(&a).unwrap();
        assert_eq!(info.organism, "concierge");
        assert_eq!(info.tier, Tier::Active);
        assert_eq!(info.address, a);
    }

    #[test]
    fn duplicate_materialize_fails() {
        let mut reg = InstanceRegistry::new(0);
        let a = addr("bob[alice]");

        reg.materialize(a.clone(), default_opts("concierge")).unwrap();
        let err = reg.materialize(a, default_opts("concierge")).unwrap_err();
        assert!(matches!(err, RegistryError::AlreadyExists(_)));
    }

    #[test]
    fn instance_limit() {
        let mut reg = InstanceRegistry::new(2);

        reg.materialize(addr("bob[a]"), default_opts("x")).unwrap();
        reg.materialize(addr("bob[b]"), default_opts("x")).unwrap();
        let err = reg.materialize(addr("bob[c]"), default_opts("x")).unwrap_err();
        assert!(matches!(err, RegistryError::LimitReached(2)));
    }

    #[test]
    fn kill_removes() {
        let mut reg = InstanceRegistry::new(0);
        let a = addr("bob[alice]");

        reg.materialize(a.clone(), default_opts("concierge")).unwrap();
        assert!(reg.is_materialized(&a));

        let killed = reg.kill(&a).unwrap();
        assert_eq!(killed.organism, "concierge");
        assert!(!reg.is_materialized(&a));
    }

    #[test]
    fn kill_nonexistent_fails() {
        let mut reg = InstanceRegistry::new(0);
        let err = reg.kill(&addr("nobody")).unwrap_err();
        assert!(matches!(err, RegistryError::NotFound(_)));
    }

    #[test]
    fn tier_transitions() {
        let mut reg = InstanceRegistry::new(0);
        let a = addr("bob[alice]");

        reg.materialize(a.clone(), default_opts("concierge")).unwrap();
        assert_eq!(reg.lookup(&a).unwrap().tier, Tier::Active);

        reg.shelve(&a).unwrap();
        assert_eq!(reg.lookup(&a).unwrap().tier, Tier::Shelved);

        // Touch promotes Shelved → Active
        reg.touch(&a).unwrap();
        assert_eq!(reg.lookup(&a).unwrap().tier, Tier::Active);

        reg.fold(&a).unwrap();
        assert_eq!(reg.lookup(&a).unwrap().tier, Tier::Folded);
    }

    #[test]
    fn list_by_tier() {
        let mut reg = InstanceRegistry::new(0);

        reg.materialize(addr("bob[a]"), default_opts("concierge")).unwrap();
        reg.materialize(addr("bob[b]"), default_opts("concierge")).unwrap();
        reg.materialize(addr("bob[c]"), default_opts("concierge")).unwrap();

        reg.shelve(&addr("bob[b]")).unwrap();
        reg.fold(&addr("bob[c]")).unwrap();

        assert_eq!(reg.list_by_tier(Tier::Active).len(), 1);
        assert_eq!(reg.list_by_tier(Tier::Shelved).len(), 1);
        assert_eq!(reg.list_by_tier(Tier::Folded).len(), 1);

        let (a, s, f) = reg.count_by_tier();
        assert_eq!((a, s, f), (1, 1, 1));
    }

    #[test]
    fn list_by_organism() {
        let mut reg = InstanceRegistry::new(0);

        reg.materialize(addr("bob[a]"), default_opts("concierge")).unwrap();
        reg.materialize(addr("planner[x]"), default_opts("planner")).unwrap();
        reg.materialize(addr("bob[b]"), default_opts("concierge")).unwrap();

        assert_eq!(reg.list_by_organism("concierge").len(), 2);
        assert_eq!(reg.list_by_organism("planner").len(), 1);
        assert_eq!(reg.list_by_organism("nobody").len(), 0);
    }

    #[test]
    fn parent_child() {
        let mut reg = InstanceRegistry::new(0);
        let parent = addr("bob[alice]");

        reg.materialize(parent.clone(), default_opts("concierge")).unwrap();

        let child_opts = MaterializeOpts {
            organism: "hotel-booker".to_string(),
            lifetime: Lifetime::UntilTaskComplete,
            parent: Some(parent.clone()),
            cache_shards: vec![],
        };
        reg.materialize(addr("hotel-booker[winter-2027:alice]"), child_opts).unwrap();

        let children = reg.children_of(&parent);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].organism, "hotel-booker");
    }

    #[test]
    fn cache_shards_stored() {
        let mut reg = InstanceRegistry::new(0);

        let opts = MaterializeOpts {
            organism: "concierge".to_string(),
            lifetime: Lifetime::default(),
            parent: None,
            cache_shards: vec!["shared.public".into(), "shared.wiki".into(), "user.alice".into()],
        };
        reg.materialize(addr("bob[alice]"), opts).unwrap();

        let info = reg.lookup(&addr("bob[alice]")).unwrap();
        assert_eq!(info.cache_shards, vec!["shared.public", "shared.wiki", "user.alice"]);
    }

    #[test]
    fn ephemeral_lifetime() {
        let mut reg = InstanceRegistry::new(0);

        let opts = MaterializeOpts {
            organism: "scratch-bot".to_string(),
            lifetime: Lifetime::Ephemeral,
            parent: None,
            cache_shards: vec![],
        };
        reg.materialize(addr("scratch.bot[query-123]"), opts).unwrap();

        let info = reg.lookup(&addr("scratch.bot[query-123]")).unwrap();
        assert_eq!(info.lifetime, Lifetime::Ephemeral);
    }

    #[test]
    fn evict_idle() {
        let mut reg = InstanceRegistry::new(0);

        // Materialize with a 0-second timeout (immediately idle)
        let opts = MaterializeOpts {
            organism: "concierge".to_string(),
            lifetime: Lifetime::UntilIdle(Duration::from_secs(0)),
            parent: None,
            cache_shards: vec![],
        };
        reg.materialize(addr("bob[alice]"), opts).unwrap();

        // Also materialize a pinned instance (should NOT be evicted)
        let pinned_opts = MaterializeOpts {
            organism: "concierge".to_string(),
            lifetime: Lifetime::Pinned,
            parent: None,
            cache_shards: vec![],
        };
        reg.materialize(addr("bob[global]"), pinned_opts).unwrap();

        // Small sleep to ensure the idle timeout passes
        std::thread::sleep(Duration::from_millis(10));

        let evicted = reg.evict_idle();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].raw(), "bob[alice]");

        // Pinned instance survives
        assert!(reg.is_materialized(&addr("bob[global]")));
        assert!(!reg.is_materialized(&addr("bob[alice]")));
    }

    #[test]
    fn unique_thread_ids() {
        let mut reg = InstanceRegistry::new(0);

        let t1 = reg.materialize(addr("a"), default_opts("x")).unwrap();
        let t2 = reg.materialize(addr("b"), default_opts("x")).unwrap();
        let t3 = reg.materialize(addr("c"), default_opts("x")).unwrap();

        assert_ne!(t1, t2);
        assert_ne!(t2, t3);
        assert_ne!(t1, t3);
    }

    #[test]
    fn namespaced_instances() {
        let mut reg = InstanceRegistry::new(0);

        reg.materialize(addr("ringhub.bob[alice]"), default_opts("concierge")).unwrap();
        reg.materialize(addr("ringhub.bob[david]"), default_opts("concierge")).unwrap();
        reg.materialize(addr("chorus.bob[sarah]"), default_opts("concierge")).unwrap();

        assert_eq!(reg.count(), 3);
        assert_eq!(reg.list_by_organism("concierge").len(), 3);

        // Each is independently addressable
        assert!(reg.is_materialized(&addr("ringhub.bob[alice]")));
        assert!(reg.is_materialized(&addr("chorus.bob[sarah]")));
        assert!(!reg.is_materialized(&addr("ringhub.bob[sarah]")));
    }
}
