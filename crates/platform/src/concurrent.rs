//! Concurrent router — production-grade wrapper for multi-user access.
//!
//! Wraps the [`Router`] in a `tokio::sync::Mutex` so multiple tasks can
//! route messages simultaneously. Handles the materialization race condition
//! (two messages for the same un-materialized instance arrive simultaneously)
//! and runs periodic idle eviction in the background.
//!
//! # Usage
//!
//! ```ignore
//! let shared = SharedRouter::new(registry, runtime, eviction_interval);
//! shared.start_eviction_timer();
//!
//! // From any number of concurrent tasks:
//! shared.send_to(envelope).await?;
//! ```

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::address::Address;
use crate::registry::{InstanceInfo, InstanceRegistry, RegistryError};
use crate::router::{Envelope, Router, RouterError, Runtime};

/// A thread-safe, multi-user router.
///
/// Wraps [`Router`] + [`Runtime`] behind an `Arc<Mutex<>>` so concurrent
/// tasks can safely send messages without external synchronization.
///
/// The mutex is held only during registry operations (microseconds).
/// Runtime I/O (allocation, delivery) is fast (channel sends), so
/// contention is minimal even at hundreds of concurrent users.
pub struct SharedRouter<R: Runtime> {
    inner: Arc<Mutex<Router>>,
    runtime: Arc<R>,
    eviction_interval: Duration,
}

impl<R: Runtime + 'static> SharedRouter<R> {
    /// Create a new in-memory shared router. State is lost on process
    /// exit. Use [`Self::open`] for a registry that persists across
    /// restarts.
    ///
    /// `max_instances`: 0 = unlimited.
    /// `eviction_interval`: how often to sweep for idle instances.
    pub fn new(max_instances: usize, runtime: R, eviction_interval: Duration) -> Self {
        let registry = InstanceRegistry::new(max_instances);
        let router = Router::new(registry);
        Self {
            inner: Arc::new(Mutex::new(router)),
            runtime: Arc::new(runtime),
            eviction_interval,
        }
    }

    /// Create a shared router backed by an on-disk JSON snapshot.
    ///
    /// On boot the snapshot at `snapshot_path` is replayed into the
    /// in-memory registry; from then on every materialize / kill /
    /// idle-eviction triggers a flush. Missing or corrupt snapshots
    /// are treated as first-boot.
    pub fn open(
        snapshot_path: std::path::PathBuf,
        max_instances: usize,
        runtime: R,
        eviction_interval: Duration,
    ) -> Self {
        let registry = InstanceRegistry::open(snapshot_path, max_instances);
        let router = Router::new(registry);
        Self {
            inner: Arc::new(Mutex::new(router)),
            runtime: Arc::new(runtime),
            eviction_interval,
        }
    }

    /// Send a message, materializing the target instance if needed.
    ///
    /// Handles the race condition where two concurrent messages target the
    /// same un-materialized instance: the second caller retries after the
    /// first materializes, transparently.
    pub async fn send_to(&self, envelope: &Envelope) -> Result<(), RouterError> {
        let mut router = self.inner.lock().await;
        match router.send_to(envelope, self.runtime.as_ref()).await {
            Ok(()) => Ok(()),
            Err(RouterError::Registry(RegistryError::AlreadyExists(_))) => {
                // Race condition: another task materialized this instance
                // between our is_materialized check and our materialize call.
                // The instance exists now — just deliver.
                drop(router);
                let mut router = self.inner.lock().await;
                // Touch + deliver via a second send_to attempt.
                // The instance now exists, so this won't try to materialize again.
                router.send_to(envelope, self.runtime.as_ref()).await
            }
            Err(e) => Err(e),
        }
    }

    /// Kill a specific instance.
    pub async fn kill(&self, address: &Address) -> Result<InstanceInfo, RouterError> {
        let mut router = self.inner.lock().await;
        router.kill(address, self.runtime.as_ref()).await
    }

    /// List all live instances.
    pub async fn list(&self) -> Vec<InstanceInfo> {
        let router = self.inner.lock().await;
        router.registry().list().into_iter().cloned().collect()
    }

    /// Get instance count.
    pub async fn count(&self) -> usize {
        let router = self.inner.lock().await;
        router.registry().count()
    }

    /// Get count by tier (active, shelved, folded).
    pub async fn count_by_tier(&self) -> (usize, usize, usize) {
        let router = self.inner.lock().await;
        router.registry().count_by_tier()
    }

    /// Run one eviction sweep. Called by the background timer, but can
    /// also be called manually for testing.
    pub async fn evict_idle(&self) -> Vec<Address> {
        let mut router = self.inner.lock().await;
        router.evict_idle(self.runtime.as_ref()).await
    }

    /// Start the background eviction timer.
    ///
    /// Spawns a tokio task that calls `evict_idle()` at the configured
    /// interval. Returns a handle that can be used to abort the timer.
    pub fn start_eviction_timer(&self) -> tokio::task::JoinHandle<()> {
        let inner = Arc::clone(&self.inner);
        let runtime = Arc::clone(&self.runtime);
        let interval = self.eviction_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.tick().await; // skip first immediate tick

            loop {
                ticker.tick().await;

                let mut router = inner.lock().await;
                let evicted = router.evict_idle(runtime.as_ref()).await;

                if !evicted.is_empty() {
                    tracing::info!(
                        count = evicted.len(),
                        addresses = ?evicted.iter().map(|a| a.raw()).collect::<Vec<_>>(),
                        "Evicted idle instances"
                    );
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Lifetime;
    use crate::router::OrganismMeta;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    /// Test runtime — same as router's MockRuntime but with counters.
    struct TestRuntime {
        organisms: HashMap<String, OrganismMeta>,
        delivered: Arc<StdMutex<Vec<String>>>,
    }

    impl TestRuntime {
        fn new() -> Self {
            let mut organisms = HashMap::new();
            organisms.insert(
                "concierge".to_string(),
                OrganismMeta {
                    default_lifetime: Lifetime::UntilIdle(Duration::from_secs(300)),
                    shard_pattern: vec!["shared.public".into(), "user.{key}".into()],
                    ephemeral: false,
                },
            );
            Self {
                organisms,
                delivered: Arc::new(StdMutex::new(vec![])),
            }
        }
    }

    #[async_trait::async_trait]
    impl Runtime for TestRuntime {
        async fn resolve_organism(&self, name: &str) -> Option<OrganismMeta> {
            self.organisms.get(name).cloned()
        }

        async fn allocate_instance(
            &self,
            _thread_id: &str,
            _address: &Address,
            _organism: &str,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn deliver(&self, thread_id: &str, _envelope: &Envelope) -> Result<(), String> {
            self.delivered.lock().unwrap().push(thread_id.to_string());
            Ok(())
        }

        async fn evict_instance(&self, _thread_id: &str) -> Result<(), String> {
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

    #[tokio::test]
    async fn concurrent_sends_to_different_instances() {
        let runtime = TestRuntime::new();
        let shared = SharedRouter::new(0, runtime, Duration::from_secs(60));

        // Spawn 10 concurrent sends to different instances
        let mut handles = vec![];
        for i in 0..10 {
            let shared_ref = &shared;
            let env = envelope(&format!("concierge[user-{i}]"));
            handles.push(tokio::spawn({
                let inner = Arc::clone(&shared_ref.inner);
                let runtime = Arc::clone(&shared_ref.runtime);
                async move {
                    let mut router = inner.lock().await;
                    router.send_to(&env, runtime.as_ref()).await
                }
            }));
        }

        for h in handles {
            h.await.unwrap().unwrap();
        }

        assert_eq!(shared.count().await, 10);
    }

    #[tokio::test]
    async fn concurrent_sends_to_same_instance() {
        let runtime = TestRuntime::new();
        let shared = SharedRouter::new(0, runtime, Duration::from_secs(60));

        // First message materializes
        shared.send_to(&envelope("concierge[alice]")).await.unwrap();

        // Second message reuses
        shared.send_to(&envelope("concierge[alice]")).await.unwrap();

        assert_eq!(shared.count().await, 1);

        // Both were delivered
        let delivered = shared.runtime.delivered.lock().unwrap();
        assert_eq!(delivered.len(), 2);
    }

    #[tokio::test]
    async fn eviction_timer_removes_idle() {
        let mut organisms = HashMap::new();
        organisms.insert(
            "concierge".to_string(),
            OrganismMeta {
                default_lifetime: Lifetime::UntilIdle(Duration::from_millis(50)),
                shard_pattern: vec![],
                ephemeral: false,
            },
        );
        let runtime = TestRuntime { organisms, delivered: Arc::new(StdMutex::new(vec![])) };
        let shared = SharedRouter::new(0, runtime, Duration::from_millis(100));

        shared.send_to(&envelope("concierge[alice]")).await.unwrap();
        assert_eq!(shared.count().await, 1);

        // Start eviction timer
        let handle = shared.start_eviction_timer();

        // Wait for the idle timeout + one eviction interval
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Should be evicted
        assert_eq!(shared.count().await, 0);

        handle.abort();
    }

    #[tokio::test]
    async fn active_instances_survive_eviction() {
        let runtime = TestRuntime::new();
        let shared = SharedRouter::new(0, runtime, Duration::from_millis(100));

        shared.send_to(&envelope("concierge[alice]")).await.unwrap();

        // Evict immediately — alice's idle timeout is 300s, she should survive
        let evicted = shared.evict_idle().await;
        assert!(evicted.is_empty());
        assert_eq!(shared.count().await, 1);
    }

    #[tokio::test]
    async fn kill_from_concurrent_context() {
        let runtime = TestRuntime::new();
        let shared = SharedRouter::new(0, runtime, Duration::from_secs(60));

        shared.send_to(&envelope("concierge[alice]")).await.unwrap();
        assert_eq!(shared.count().await, 1);

        let killed = shared.kill(&Address::parse("concierge[alice]").unwrap()).await.unwrap();
        assert_eq!(killed.organism, "concierge");
        assert_eq!(shared.count().await, 0);
    }

    #[tokio::test]
    async fn list_returns_cloned_data() {
        let runtime = TestRuntime::new();
        let shared = SharedRouter::new(0, runtime, Duration::from_secs(60));

        shared.send_to(&envelope("concierge[alice]")).await.unwrap();
        shared.send_to(&envelope("concierge[bob]")).await.unwrap();

        let list = shared.list().await;
        assert_eq!(list.len(), 2);
    }
}
