//! Durable thread table — WAL-backed upgrade of rust-pipeline's ThreadRegistry.
//!
//! Same API as ThreadRegistry but with:
//! - Profile field on each thread record
//! - All mutations flow through the WAL
//! - State persisted to disk (HashMap-based, rebuilt from WAL on recovery)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use uuid::Uuid;

use crate::wal::{EntryType, WalEntry};

/// Result of pruning a thread chain for a response.
#[derive(Debug, PartialEq)]
pub struct PruneResult {
    /// The target agent (new last segment after pruning).
    pub target: String,
    /// The UUID for the pruned chain.
    pub thread_id: String,
}

/// A thread record stored in the table.
#[derive(Debug, Clone)]
pub struct ThreadRecord {
    /// UUID for this thread.
    pub uuid: String,
    /// Dot-separated call chain.
    pub chain: String,
    /// Security profile name (inherited from thread root).
    pub profile: String,
    /// Creation timestamp (unix epoch millis).
    pub created_at: u64,
}

/// Durable thread table.
#[allow(dead_code)]
pub struct ThreadTable {
    /// chain → UUID
    chain_to_uuid: HashMap<String, String>,
    /// UUID → ThreadRecord
    records: HashMap<String, ThreadRecord>,
    /// Root thread UUID
    root_uuid: Option<String>,
    /// Root chain string
    root_chain: String,
    /// Path for persistence
    path: PathBuf,
}

impl ThreadTable {
    /// Open or create the thread table.
    pub fn open(path: &Path) -> crate::error::KernelResult<Self> {
        Ok(Self {
            chain_to_uuid: HashMap::new(),
            records: HashMap::new(),
            root_uuid: None,
            root_chain: "system".into(),
            path: path.to_path_buf(),
        })
    }

    /// Apply a WAL entry during replay.
    pub fn apply_wal_entry(&mut self, entry: &WalEntry) {
        match entry.entry_type {
            EntryType::ThreadCreate => {
                // Payload: uuid\0chain\0profile
                if let Some((uuid, chain, profile)) = Self::parse_create_payload(&entry.payload) {
                    self.insert_record(uuid, chain, profile);
                }
            }
            EntryType::ThreadExtend => {
                // Payload: current_uuid\0next_hop
                if let Some((current_uuid, next_hop)) = Self::parse_extend_payload(&entry.payload) {
                    self.extend_chain(&current_uuid, &next_hop);
                }
            }
            EntryType::ThreadPrune => {
                // Payload: thread_id
                let thread_id = String::from_utf8_lossy(&entry.payload).to_string();
                self.prune_for_response(&thread_id);
            }
            EntryType::ThreadCleanup => {
                let thread_id = String::from_utf8_lossy(&entry.payload).to_string();
                self.cleanup(&thread_id);
            }
            _ => {} // not a thread op
        }
    }

    /// Initialize the root thread at boot time.
    pub fn initialize_root(&mut self, organism_name: &str, profile: &str) -> String {
        if let Some(ref uuid) = self.root_uuid {
            return uuid.clone();
        }

        self.root_chain = format!("system.{organism_name}");
        let uuid = Uuid::new_v4().to_string();

        let record = ThreadRecord {
            uuid: uuid.clone(),
            chain: self.root_chain.clone(),
            profile: profile.to_string(),
            created_at: now_millis(),
        };

        self.chain_to_uuid
            .insert(self.root_chain.clone(), uuid.clone());
        self.records.insert(uuid.clone(), record);
        self.root_uuid = Some(uuid.clone());
        uuid
    }

    /// Build a WAL entry for root initialization.
    pub fn wal_entry_initialize_root(
        &self,
        uuid: &str,
        organism_name: &str,
        profile: &str,
    ) -> WalEntry {
        let chain = format!("system.{organism_name}");
        let mut payload = Vec::new();
        payload.extend_from_slice(uuid.as_bytes());
        payload.push(0);
        payload.extend_from_slice(chain.as_bytes());
        payload.push(0);
        payload.extend_from_slice(profile.as_bytes());
        WalEntry::new(EntryType::ThreadCreate, payload)
    }

    /// Get the root thread UUID.
    pub fn root_uuid(&self) -> Option<&str> {
        self.root_uuid.as_deref()
    }

    /// Look up chain for a UUID.
    pub fn lookup(&self, thread_id: &str) -> Option<&str> {
        self.records.get(thread_id).map(|r| r.chain.as_str())
    }

    /// Get the security profile for a thread (walks to root).
    pub fn get_profile(&self, thread_id: &str) -> Option<&str> {
        let record = self.records.get(thread_id)?;

        // If this thread has its own profile, use it
        if !record.profile.is_empty() {
            return Some(&record.profile);
        }

        // Otherwise walk up: find the shortest chain prefix that has a profile
        let parts: Vec<&str> = record.chain.split('.').collect();
        for i in 1..=parts.len() {
            let prefix = parts[..i].join(".");
            if let Some(uuid) = self.chain_to_uuid.get(&prefix) {
                if let Some(r) = self.records.get(uuid) {
                    if !r.profile.is_empty() {
                        return Some(self.records.get(uuid).unwrap().profile.as_str());
                    }
                }
            }
        }

        // Fallback: check root
        if let Some(ref root_uuid) = self.root_uuid {
            if let Some(root) = self.records.get(root_uuid) {
                return Some(&root.profile);
            }
        }

        None
    }

    /// Register an external thread (existing UUID, new chain).
    pub fn register_thread(
        &mut self,
        thread_id: &str,
        initiator: &str,
        target: &str,
        profile: &str,
    ) -> String {
        if self.records.contains_key(thread_id) {
            return thread_id.to_string();
        }

        let chain = if self.root_uuid.is_some() {
            format!("{}.{initiator}.{target}", self.root_chain)
        } else {
            format!("{initiator}.{target}")
        };

        if let Some(existing) = self.chain_to_uuid.get(&chain) {
            return existing.clone();
        }

        self.insert_record(thread_id.to_string(), chain, profile.to_string());
        thread_id.to_string()
    }

    /// Extend a chain with a new hop. Returns UUID for the extended chain.
    pub fn extend_chain(&mut self, current_uuid: &str, next_hop: &str) -> String {
        let current_chain = self
            .records
            .get(current_uuid)
            .map(|r| r.chain.clone())
            .unwrap_or_default();

        let new_chain = if current_chain.is_empty() {
            next_hop.to_string()
        } else {
            format!("{current_chain}.{next_hop}")
        };

        if let Some(uuid) = self.chain_to_uuid.get(&new_chain) {
            return uuid.clone();
        }

        // Inherit profile from current thread
        let profile = self
            .records
            .get(current_uuid)
            .map(|r| r.profile.clone())
            .unwrap_or_default();

        let uuid = Uuid::new_v4().to_string();
        self.insert_record(uuid.clone(), new_chain, profile);
        uuid
    }

    /// Build a WAL entry for extend_chain.
    pub fn wal_entry_extend(&self, current_uuid: &str, next_hop: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(current_uuid.as_bytes());
        payload.push(0);
        payload.extend_from_slice(next_hop.as_bytes());
        WalEntry::new(EntryType::ThreadExtend, payload)
    }

    /// Peek at what prune would do without actually pruning. Used by Kernel
    /// to check before building the WAL batch.
    pub fn peek_prune(&self, thread_id: &str) -> Option<PruneResult> {
        let record = self.records.get(thread_id)?;
        let parts: Vec<&str> = record.chain.split('.').collect();
        if parts.len() <= 1 {
            return None;
        }
        let pruned_parts = &parts[..parts.len() - 1];
        let target = pruned_parts.last().unwrap().to_string();
        let pruned_chain = pruned_parts.join(".");
        let new_uuid = self
            .chain_to_uuid
            .get(&pruned_chain)
            .cloned()
            .unwrap_or_else(|| "pending".to_string());
        Some(PruneResult {
            target,
            thread_id: new_uuid,
        })
    }

    /// Prune chain for a response. Returns the target and new UUID.
    pub fn prune_for_response(&mut self, thread_id: &str) -> Option<PruneResult> {
        let chain = self.records.get(thread_id)?.chain.clone();

        let parts: Vec<&str> = chain.split('.').collect();
        if parts.len() <= 1 {
            self.cleanup(thread_id);
            return None;
        }

        let pruned_parts = &parts[..parts.len() - 1];
        let target = pruned_parts.last().unwrap().to_string();
        let pruned_chain = pruned_parts.join(".");

        let new_uuid = if let Some(uuid) = self.chain_to_uuid.get(&pruned_chain) {
            uuid.clone()
        } else {
            // Create record for pruned chain with inherited profile
            let profile = self
                .records
                .get(thread_id)
                .map(|r| r.profile.clone())
                .unwrap_or_default();
            let uuid = Uuid::new_v4().to_string();
            self.insert_record(uuid.clone(), pruned_chain, profile);
            uuid
        };

        Some(PruneResult {
            target,
            thread_id: new_uuid,
        })
    }

    /// Clean up a thread record.
    pub fn cleanup(&mut self, thread_id: &str) {
        if let Some(record) = self.records.remove(thread_id) {
            self.chain_to_uuid.remove(&record.chain);
        }
    }

    /// Get a thread record by UUID.
    pub fn get_record(&self, thread_id: &str) -> Option<&ThreadRecord> {
        self.records.get(thread_id)
    }

    /// Iterate over all thread records.
    pub fn all_records(&self) -> impl Iterator<Item = &ThreadRecord> {
        self.records.values()
    }

    /// Number of active thread records.
    pub fn count(&self) -> usize {
        self.records.len()
    }

    // ── Internal helpers ──

    fn insert_record(&mut self, uuid: String, chain: String, profile: String) {
        // Check if this is a root chain before we move values
        let is_root = self.root_uuid.is_none()
            && chain.starts_with("system.")
            && chain.matches('.').count() == 1;

        let record = ThreadRecord {
            uuid: uuid.clone(),
            chain: chain.clone(),
            profile,
            created_at: now_millis(),
        };
        self.chain_to_uuid.insert(chain, uuid.clone());
        self.records.insert(uuid.clone(), record);

        if is_root {
            self.root_chain = self.records.get(&uuid).unwrap().chain.clone();
            self.root_uuid = Some(uuid);
        }
    }

    fn parse_create_payload(payload: &[u8]) -> Option<(String, String, String)> {
        let s = String::from_utf8_lossy(payload);
        let parts: Vec<&str> = s.splitn(3, '\0').collect();
        if parts.len() >= 3 {
            Some((
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
            ))
        } else if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string(), String::new()))
        } else {
            None
        }
    }

    fn parse_extend_payload(payload: &[u8]) -> Option<(String, String)> {
        let s = String::from_utf8_lossy(payload);
        let parts: Vec<&str> = s.splitn(2, '\0').collect();
        if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else {
            None
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn initialize_root() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        let uuid = table.initialize_root("hello-world", "root");
        assert!(!uuid.is_empty());
        assert_eq!(table.lookup(&uuid), Some("system.hello-world"));
        assert_eq!(table.get_profile(&uuid), Some("root"));
    }

    #[test]
    fn extend_and_prune() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        let root = table.initialize_root("org", "admin");

        let t1 = table.extend_chain(&root, "handler");
        assert_eq!(table.lookup(&t1), Some("system.org.handler"));

        // Profile inherited from root
        assert_eq!(table.get_profile(&t1), Some("admin"));

        let t2 = table.extend_chain(&t1, "subhandler");
        assert_eq!(table.lookup(&t2), Some("system.org.handler.subhandler"));

        // Prune subhandler → back to handler
        let result = table.prune_for_response(&t2).unwrap();
        assert_eq!(result.target, "handler");
    }

    #[test]
    fn register_external_thread() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        table.initialize_root("org", "root");

        let ext_uuid = "ext-uuid-123";
        let registered = table.register_thread(ext_uuid, "console", "router", "public");
        assert_eq!(registered, ext_uuid);
        assert_eq!(table.lookup(ext_uuid), Some("system.org.console.router"));
        assert_eq!(table.get_profile(ext_uuid), Some("public"));
    }

    #[test]
    fn prune_exhausted() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();

        // Single-segment chain with manual insert
        let uuid = Uuid::new_v4().to_string();
        table.insert_record(uuid.clone(), "single".into(), "root".into());
        assert!(table.prune_for_response(&uuid).is_none());
    }

    #[test]
    fn cleanup_removes_record() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        let root = table.initialize_root("org", "root");
        let t1 = table.extend_chain(&root, "handler");

        assert!(table.lookup(&t1).is_some());
        table.cleanup(&t1);
        assert!(table.lookup(&t1).is_none());
    }

    #[test]
    fn wal_replay_recovers_state() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();

        // Simulate WAL replay of a create entry
        let entry = WalEntry::new(
            EntryType::ThreadCreate,
            b"uuid-1\0system.org\0root".to_vec(),
        );
        table.apply_wal_entry(&entry);

        assert_eq!(table.lookup("uuid-1"), Some("system.org"));
        assert_eq!(table.get_profile("uuid-1"), Some("root"));
    }

    #[test]
    fn thread_table_all_records() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        let root = table.initialize_root("org", "admin");
        let _t1 = table.extend_chain(&root, "handler");

        let records: Vec<&ThreadRecord> = table.all_records().collect();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn thread_table_all_records_empty() {
        let dir = TempDir::new().unwrap();
        let table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        assert_eq!(table.all_records().count(), 0);
    }

    #[test]
    fn profile_propagation() {
        let dir = TempDir::new().unwrap();
        let mut table = ThreadTable::open(&dir.path().join("threads.bin")).unwrap();
        let root = table.initialize_root("org", "admin");

        let t1 = table.extend_chain(&root, "a");
        let t2 = table.extend_chain(&t1, "b");
        let t3 = table.extend_chain(&t2, "c");

        // All inherit admin profile from root
        assert_eq!(table.get_profile(&t1), Some("admin"));
        assert_eq!(table.get_profile(&t2), Some("admin"));
        assert_eq!(table.get_profile(&t3), Some("admin"));
    }
}
