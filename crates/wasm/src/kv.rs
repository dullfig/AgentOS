//! Host-side KV store for WASM tools.
//!
//! Each tool gets a `KvNamespace` that maps its flat key space to one or more
//! physical namespaces in the shared `KvStore`. Namespace resolution is
//! transparent — tools call `get("price:AAPL")` and the host prepends the
//! resolved namespace prefix.
//!
//! The `KvStore` is shared across all tools in a pipeline instance.
//! Concurrency is handled by the store's internal locking.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Physical backing store — all namespaces in one map, keyed by `ns:key`.
#[derive(Debug, Clone)]
pub struct KvStore {
    data: Arc<Mutex<HashMap<String, String>>>,
}

impl KvStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a value from a specific namespace.
    pub fn get(&self, namespace: &str, key: &str) -> Option<String> {
        let full_key = format!("{namespace}\x00{key}");
        self.data.lock().unwrap().get(&full_key).cloned()
    }

    /// Put a value into a specific namespace.
    pub fn put(&self, namespace: &str, key: &str, value: String) {
        let full_key = format!("{namespace}\x00{key}");
        self.data.lock().unwrap().insert(full_key, value);
    }

    /// Delete a key from a specific namespace. Returns true if the key existed.
    pub fn delete(&self, namespace: &str, key: &str) -> bool {
        let full_key = format!("{namespace}\x00{key}");
        self.data.lock().unwrap().remove(&full_key).is_some()
    }

    /// List keys in a namespace matching a prefix.
    pub fn list_keys(&self, namespace: &str, prefix: &str) -> Vec<String> {
        let ns_prefix = format!("{namespace}\x00{prefix}");
        let strip_prefix = format!("{namespace}\x00");
        self.data
            .lock()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with(&ns_prefix))
            .map(|k| k.strip_prefix(&strip_prefix).unwrap_or(k).to_string())
            .collect()
    }
}

/// A tool's view of the KV store — maps operations to resolved namespaces.
///
/// - `own_namespace`: private namespace (always read-write)
/// - `read_namespaces`: additional namespaces the tool can read
/// - `write_namespaces`: additional namespaces the tool can write
///
/// On `get`, the tool's own namespace is checked first, then read grants.
/// On `put`/`delete`, only own + write-granted namespaces are allowed.
#[derive(Debug, Clone)]
pub struct KvGrants {
    /// The tool's private namespace (e.g., "tool:stock-tracker").
    pub own_namespace: String,
    /// Namespaces this tool can read (in addition to own).
    pub read_namespaces: Vec<String>,
    /// Namespaces this tool can write (in addition to own).
    pub write_namespaces: Vec<String>,
}

impl KvGrants {
    /// Create grants for a tool with only its private namespace.
    pub fn private_only(tool_name: &str) -> Self {
        Self {
            own_namespace: format!("tool:{tool_name}"),
            read_namespaces: Vec::new(),
            write_namespaces: Vec::new(),
        }
    }
}

/// A scoped KV view for a single tool invocation.
///
/// This is what gets embedded in the WASM `ToolState` and called
/// by the host-function implementations.
#[derive(Debug, Clone)]
pub struct KvScope {
    pub store: KvStore,
    pub grants: KvGrants,
}

impl KvScope {
    pub fn new(store: KvStore, grants: KvGrants) -> Self {
        Self { store, grants }
    }

    /// Get: check own namespace first, then read-granted namespaces.
    pub fn get(&self, key: &str) -> Option<String> {
        // Own namespace first
        if let Some(val) = self.store.get(&self.grants.own_namespace, key) {
            return Some(val);
        }
        // Then read grants
        for ns in &self.grants.read_namespaces {
            if let Some(val) = self.store.get(ns, key) {
                return Some(val);
            }
        }
        None
    }

    /// Put: own namespace, or write-granted namespace if key is prefixed.
    /// For now, always writes to own namespace.
    pub fn put(&self, key: &str, value: String) -> Result<(), String> {
        self.store.put(&self.grants.own_namespace, key, value);
        Ok(())
    }

    /// Put to a specific shared namespace (must be write-granted).
    pub fn put_shared(&self, namespace: &str, key: &str, value: String) -> Result<(), String> {
        if namespace == self.grants.own_namespace
            || self.grants.write_namespaces.contains(&namespace.to_string())
        {
            self.store.put(namespace, key, value);
            Ok(())
        } else {
            Err(format!("no write access to namespace '{namespace}'"))
        }
    }

    /// Delete from own namespace.
    pub fn delete(&self, key: &str) -> Result<(), String> {
        if self.store.delete(&self.grants.own_namespace, key) {
            Ok(())
        } else {
            Err(format!("key '{key}' not found"))
        }
    }

    /// List keys: merges own namespace + read-granted namespaces.
    pub fn list_keys(&self, prefix: &str) -> Vec<String> {
        let mut keys = self.store.list_keys(&self.grants.own_namespace, prefix);
        for ns in &self.grants.read_namespaces {
            keys.extend(self.store.list_keys(ns, prefix));
        }
        keys.sort();
        keys.dedup();
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_namespace_isolation() {
        let store = KvStore::new();
        let scope_a = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));
        let scope_b = KvScope::new(store.clone(), KvGrants::private_only("tool-b"));

        scope_a.put("key1", "value-a".into()).unwrap();
        scope_b.put("key1", "value-b".into()).unwrap();

        assert_eq!(scope_a.get("key1"), Some("value-a".into()));
        assert_eq!(scope_b.get("key1"), Some("value-b".into()));
    }

    #[test]
    fn read_grant_sees_other_namespace() {
        let store = KvStore::new();
        let writer = KvScope::new(store.clone(), KvGrants::private_only("stock-tracker"));
        let reader = KvScope::new(
            store.clone(),
            KvGrants {
                own_namespace: "tool:portfolio".into(),
                read_namespaces: vec!["tool:stock-tracker".into()],
                write_namespaces: Vec::new(),
            },
        );

        writer.put("price:AAPL", "198.50".into()).unwrap();

        // Reader can see it via read grant
        assert_eq!(reader.get("price:AAPL"), Some("198.50".into()));
    }

    #[test]
    fn no_grant_no_access() {
        let store = KvStore::new();
        let writer = KvScope::new(store.clone(), KvGrants::private_only("stock-tracker"));
        let stranger = KvScope::new(store.clone(), KvGrants::private_only("unrelated-tool"));

        writer.put("secret", "data".into()).unwrap();

        // Stranger can't see it
        assert_eq!(stranger.get("secret"), None);
    }

    #[test]
    fn write_grant_to_shared_namespace() {
        let store = KvStore::new();
        let scope = KvScope::new(
            store.clone(),
            KvGrants {
                own_namespace: "tool:producer".into(),
                read_namespaces: Vec::new(),
                write_namespaces: vec!["shared:market".into()],
            },
        );

        // Can write to shared namespace
        assert!(scope.put_shared("shared:market", "index", "5200".into()).is_ok());

        // Can read back from shared via store directly
        assert_eq!(store.get("shared:market", "index"), Some("5200".into()));
    }

    #[test]
    fn write_denied_without_grant() {
        let store = KvStore::new();
        let scope = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));

        let result = scope.put_shared("shared:market", "index", "5200".into());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no write access"));
    }

    #[test]
    fn delete_existing_key() {
        let store = KvStore::new();
        let scope = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));

        scope.put("key1", "val".into()).unwrap();
        assert!(scope.delete("key1").is_ok());
        assert_eq!(scope.get("key1"), None);
    }

    #[test]
    fn delete_missing_key_errors() {
        let store = KvStore::new();
        let scope = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));

        let result = scope.delete("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn list_keys_own_namespace() {
        let store = KvStore::new();
        let scope = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));

        scope.put("price:AAPL", "198".into()).unwrap();
        scope.put("price:GOOG", "175".into()).unwrap();
        scope.put("volume:AAPL", "1000000".into()).unwrap();

        let keys = scope.list_keys("price:");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"price:AAPL".to_string()));
        assert!(keys.contains(&"price:GOOG".to_string()));
    }

    #[test]
    fn list_keys_merges_read_grants() {
        let store = KvStore::new();
        let writer = KvScope::new(store.clone(), KvGrants::private_only("tracker"));
        let reader = KvScope::new(
            store.clone(),
            KvGrants {
                own_namespace: "tool:reader".into(),
                read_namespaces: vec!["tool:tracker".into()],
                write_namespaces: Vec::new(),
            },
        );

        writer.put("price:AAPL", "198".into()).unwrap();
        reader.put("price:MSFT", "420".into()).unwrap();

        let keys = reader.list_keys("price:");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"price:AAPL".to_string()));
        assert!(keys.contains(&"price:MSFT".to_string()));
    }

    #[test]
    fn own_namespace_takes_priority() {
        let store = KvStore::new();
        // Both write the same key in their own namespace
        let tracker = KvScope::new(store.clone(), KvGrants::private_only("tracker"));
        tracker.put("price:AAPL", "old-price".into()).unwrap();

        let reader = KvScope::new(
            store.clone(),
            KvGrants {
                own_namespace: "tool:reader".into(),
                read_namespaces: vec!["tool:tracker".into()],
                write_namespaces: Vec::new(),
            },
        );
        reader.put("price:AAPL", "my-price".into()).unwrap();

        // Own namespace wins
        assert_eq!(reader.get("price:AAPL"), Some("my-price".into()));
    }

    #[test]
    fn store_is_shared_across_clones() {
        let store = KvStore::new();
        let store2 = store.clone();

        store.put("ns", "key", "value".into());
        assert_eq!(store2.get("ns", "key"), Some("value".into()));
    }

    #[test]
    fn empty_prefix_lists_all() {
        let store = KvStore::new();
        let scope = KvScope::new(store.clone(), KvGrants::private_only("tool-a"));

        scope.put("a", "1".into()).unwrap();
        scope.put("b", "2".into()).unwrap();
        scope.put("c", "3".into()).unwrap();

        let keys = scope.list_keys("");
        assert_eq!(keys.len(), 3);
    }
}
