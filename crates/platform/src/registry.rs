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
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::address::Address;
use crate::buffers::BufferStore;
use crate::snapshot::{self, InstanceRecord, RegistrySnapshot};

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
    /// Optional snapshot path. When `Some`, materialize / kill / evict
    /// flush the in-memory state to disk so it survives restart.
    snapshot_path: Option<PathBuf>,
}

impl InstanceRegistry {
    /// Create a new empty in-memory registry. Use [`Self::open`] for
    /// the persistent variant.
    pub fn new(max_instances: usize) -> Self {
        Self {
            instances: HashMap::new(),
            max_instances,
            next_thread_id: 1,
            snapshot_path: None,
        }
    }

    /// Open a registry persisted at `snapshot_path`. If the file exists
    /// and parses, instances are restored. Missing or corrupt snapshots
    /// produce an empty registry (treated as first boot).
    ///
    /// `next_thread_id` is reseeded from the highest restored
    /// `inst-NNNNNN` so freshly-materialized instances don't collide
    /// with replayed ones.
    pub fn open(snapshot_path: PathBuf, max_instances: usize) -> Self {
        let mut instances: HashMap<String, InstanceInfo> = HashMap::new();
        let mut max_seen: u64 = 0;

        if let Ok(Some(snap)) = snapshot::read(&snapshot_path) {
            for rec in snap.instances {
                let address = match Address::parse(&rec.address_raw) {
                    Ok(a) => a,
                    Err(e) => {
                        tracing::warn!(
                            address = %rec.address_raw,
                            error = %e,
                            "skipping unparseable address in registry snapshot"
                        );
                        continue;
                    }
                };
                let parent = rec
                    .parent_raw
                    .as_deref()
                    .and_then(|s| Address::parse(s).ok());
                if let Some(n) = thread_id_suffix(&rec.thread_id) {
                    if n > max_seen {
                        max_seen = n;
                    }
                }
                let now = Instant::now();
                let info = InstanceInfo {
                    address,
                    organism: rec.organism,
                    thread_id: rec.thread_id,
                    lifetime: rec.lifetime.into(),
                    tier: Tier::Active,
                    parent,
                    cache_shards: rec.cache_shards,
                    created_at: now,
                    last_accessed: now,
                    buffers: BufferStore::new(),
                };
                instances.insert(rec.address_raw, info);
            }
        }

        Self {
            instances,
            max_instances,
            next_thread_id: max_seen + 1,
            snapshot_path: Some(snapshot_path),
        }
    }

    /// Force a snapshot write. Returns `Ok(())` when no snapshot path
    /// is configured (in-memory mode). Errors propagate up so callers
    /// can decide whether to fail-loud or log-and-continue.
    pub fn flush(&self) -> std::io::Result<()> {
        let Some(path) = self.snapshot_path.as_ref() else {
            return Ok(());
        };
        let records: Vec<InstanceRecord> = self
            .instances
            .values()
            .map(|info| InstanceRecord {
                address_raw: info.address.raw().to_string(),
                organism: info.organism.clone(),
                thread_id: info.thread_id.clone(),
                lifetime: (&info.lifetime).into(),
                parent_raw: info.parent.as_ref().map(|p| p.raw().to_string()),
                cache_shards: info.cache_shards.clone(),
            })
            .collect();
        snapshot::write_atomic(path, &RegistrySnapshot::new(records))
    }

    /// Best-effort flush — logs but doesn't propagate IO errors.
    /// Used after mutations where a snapshot failure shouldn't crash
    /// the request path; the next mutation retries.
    fn flush_quiet(&self) {
        if let Err(e) = self.flush() {
            tracing::warn!(
                path = ?self.snapshot_path,
                error = %e,
                "platform registry snapshot write failed; in-memory state intact"
            );
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
        self.flush_quiet();
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
        let info = self
            .instances
            .remove(address.raw())
            .ok_or_else(|| RegistryError::NotFound(address.raw().to_string()))?;
        self.flush_quiet();
        Ok(info)
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

        let evicted: Vec<Address> = to_evict
            .iter()
            .filter_map(|key| {
                self.instances
                    .remove(key)
                    .map(|info| info.address)
            })
            .collect();
        if !evicted.is_empty() {
            self.flush_quiet();
        }
        evicted
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

/// Parse the trailing number out of an `inst-NNNNNN` thread_id, used
/// to reseed the registry's counter on `open` so we don't reissue an
/// id that's already on disk.
fn thread_id_suffix(thread_id: &str) -> Option<u64> {
    thread_id
        .strip_prefix("inst-")
        .and_then(|s| s.parse::<u64>().ok())
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

    // ── Snapshot-backed persistence ──

    #[test]
    fn open_with_no_snapshot_starts_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("registry.json");
        let reg = InstanceRegistry::open(path, 0);
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn materialize_writes_snapshot() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("registry.json");
        let mut reg = InstanceRegistry::open(path.clone(), 0);
        reg.materialize(addr("bob[alice]"), default_opts("bob")).unwrap();

        assert!(path.exists(), "snapshot file should be written on materialize");
        let snap = crate::snapshot::read(&path).unwrap().expect("snapshot must parse");
        assert_eq!(snap.instances.len(), 1);
        assert_eq!(snap.instances[0].address_raw, "bob[alice]");
    }

    #[test]
    fn instance_survives_registry_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("registry.json");

        let original_thread_id = {
            let mut reg = InstanceRegistry::open(path.clone(), 0);
            let tid = reg.materialize(addr("bob[alice]"), default_opts("bob")).unwrap();
            tid
        };

        let reg = InstanceRegistry::open(path, 0);
        let info = reg
            .lookup(&addr("bob[alice]"))
            .expect("bob[alice] must replay from snapshot");
        assert_eq!(info.thread_id, original_thread_id);
        assert_eq!(info.organism, "bob");
        // Tier resets to Active on replay (idle-eviction will reshelve naturally).
        assert_eq!(info.tier, Tier::Active);
    }

    #[test]
    fn next_thread_id_advances_past_replayed() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("registry.json");

        // First boot: materialize three instances → inst-000001..3.
        {
            let mut reg = InstanceRegistry::open(path.clone(), 0);
            reg.materialize(addr("bob[a]"), default_opts("bob")).unwrap();
            reg.materialize(addr("bob[b]"), default_opts("bob")).unwrap();
            reg.materialize(addr("bob[c]"), default_opts("bob")).unwrap();
        }

        // Second boot: a fresh materialize must skip past 1..3.
        let mut reg = InstanceRegistry::open(path, 0);
        let tid = reg
            .materialize(addr("bob[d]"), default_opts("bob"))
            .unwrap();
        assert_eq!(tid, "inst-000004");
    }

    #[test]
    fn kill_persists() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("registry.json");

        let mut reg = InstanceRegistry::open(path.clone(), 0);
        reg.materialize(addr("bob[alice]"), default_opts("bob")).unwrap();
        reg.materialize(addr("bob[bob]"), default_opts("bob")).unwrap();
        reg.kill(&addr("bob[alice]")).unwrap();

        // Reopen — only bob[bob] should remain.
        let reg2 = InstanceRegistry::open(path, 0);
        assert!(reg2.lookup(&addr("bob[alice]")).is_none());
        assert!(reg2.lookup(&addr("bob[bob]")).is_some());
    }
}
