//! Kernel — durable state for AgentOS.
//!
//! Three pieces of nuclear-proof state:
//! - Thread table (call stack)
//! - Context store (VMM)
//! - Message journal (audit/tape)
//!
//! One WAL, atomic ops. Everything else is ephemeral userspace.

pub mod context_store;
pub mod error;
pub mod journal;
pub mod thread_table;
pub mod wal;

use std::path::{Path, PathBuf};

use context_store::ContextStore;
use error::KernelResult;
use journal::Journal;
use thread_table::ThreadTable;
use wal::Wal;

/// The kernel: wraps all three stores and provides atomic cross-store operations.
pub struct Kernel {
    pub wal: Wal,
    pub threads: ThreadTable,
    pub contexts: ContextStore,
    pub journal: Journal,
    data_dir: PathBuf,
}

impl Kernel {
    /// Open or create the kernel at the given data directory.
    /// Replays the WAL to recover any uncommitted state.
    pub fn open(data_dir: &Path) -> KernelResult<Self> {
        std::fs::create_dir_all(data_dir)?;

        let wal = Wal::open(&data_dir.join("kernel.wal"))?;
        let mut threads = ThreadTable::open(&data_dir.join("threads.bin"))?;
        let mut contexts = ContextStore::open(&data_dir.join("contexts"))?;
        let mut journal = Journal::open(&data_dir.join("journal.bin"))?;

        // Replay WAL and apply any entries not yet reflected in state
        let entries = wal.replay()?;
        for entry in &entries {
            threads.apply_wal_entry(entry);
            contexts.apply_wal_entry(entry);
            journal.apply_wal_entry(entry);
        }

        Ok(Self {
            wal,
            threads,
            contexts,
            journal,
            data_dir: data_dir.to_path_buf(),
        })
    }

    /// Initialize the root thread with WAL logging.
    pub fn initialize_root(&mut self, organism_name: &str, profile: &str) -> KernelResult<String> {
        let uuid = self.threads.initialize_root(organism_name, profile);
        let entry = self
            .threads
            .wal_entry_initialize_root(&uuid, organism_name, profile);
        self.wal.append(&entry)?;
        Ok(uuid)
    }

    /// Atomic prune: thread pruned + context released + journal updated.
    pub fn prune_thread(
        &mut self,
        thread_id: &str,
    ) -> KernelResult<Option<thread_table::PruneResult>> {
        // Look up what we'll prune before writing WAL
        let prune_result = self.threads.peek_prune(thread_id);
        if prune_result.is_none() {
            return Ok(None);
        }

        // Build batch
        let batch = vec![
            wal::WalEntry::new(wal::EntryType::ThreadPrune, thread_id.as_bytes().to_vec()),
            wal::WalEntry::new(
                wal::EntryType::ContextRelease,
                thread_id.as_bytes().to_vec(),
            ),
            wal::WalEntry::new(
                wal::EntryType::JournalDelivered,
                thread_id.as_bytes().to_vec(),
            ),
        ];

        // WAL first, then apply to state
        self.wal.append_batch(&batch)?;
        let result = self.threads.prune_for_response(thread_id);
        self.contexts.release(thread_id)?;
        self.journal.mark_delivered_by_thread(thread_id);

        Ok(result)
    }

    /// Atomic fold: thread pruned + context folded (summary in parent) + journal updated.
    /// Alternative to `prune_thread()` — compresses instead of destroying.
    /// The `summary` is inserted as a fold segment in the parent's context.
    pub fn fold_thread(
        &mut self,
        thread_id: &str,
        summary: &[u8],
    ) -> KernelResult<Option<thread_table::PruneResult>> {
        // Look up what we'll prune before writing WAL
        let prune_result = self.threads.peek_prune(thread_id);
        if prune_result.is_none() {
            return Ok(None);
        }

        // Stash child segment contents in fold_store before releasing
        let fold_thread_ref = format!("fold-thread-{}", thread_id);
        let mut has_content = false;
        if let Some(ctx) = self.contexts.get(thread_id) {
            let mut combined_content = Vec::new();
            for seg in ctx.segments.values() {
                combined_content.extend_from_slice(&seg.content);
                combined_content.push(b'\n');
            }
            if !combined_content.is_empty() {
                self.contexts.fold_store.insert(fold_thread_ref.clone(), combined_content);
                has_content = true;
            }
        }

        // Build WAL batch: prune + release child context + journal delivered
        let batch = vec![
            wal::WalEntry::new(
                wal::EntryType::ThreadPrune,
                thread_id.as_bytes().to_vec(),
            ),
            wal::WalEntry::new(
                wal::EntryType::ContextRelease,
                thread_id.as_bytes().to_vec(),
            ),
            wal::WalEntry::new(
                wal::EntryType::JournalDelivered,
                thread_id.as_bytes().to_vec(),
            ),
        ];

        // WAL first, then apply to state
        self.wal.append_batch(&batch)?;

        let result = self.threads.prune_for_response(thread_id);
        self.contexts.release(thread_id)?;
        self.journal.mark_delivered_by_thread(thread_id);

        // Add summary segment to parent's context (if parent exists)
        // PruneResult.thread_id is the parent's UUID after pruning
        if let Some(ref pr) = result {
            let parent_id = &pr.thread_id;
            if self.contexts.exists(parent_id) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let fold_seg = context_store::ContextSegment {
                    id: format!("fold:{}", thread_id),
                    tag: "fold-summary".into(),
                    content: summary.to_vec(),
                    status: context_store::SegmentStatus::Folded,
                    relevance: 0.5,
                    created_at: now,
                    fold_ref: if has_content {
                        Some(fold_thread_ref)
                    } else {
                        None
                    },
                };
                let _ = self.contexts.add_segment(parent_id, fold_seg);
            }
        }

        Ok(result)
    }

    /// Atomic dispatch: extend thread + allocate context + log journal entry.
    /// Returns the new thread UUID.
    pub fn dispatch_message(
        &mut self,
        from: &str,
        to: &str,
        thread_id: &str,
        message_id: &str,
    ) -> KernelResult<String> {
        // Build batch payload
        let mut dispatch_payload = Vec::new();
        dispatch_payload.extend_from_slice(thread_id.as_bytes());
        dispatch_payload.push(0); // null separator
        dispatch_payload.extend_from_slice(to.as_bytes());

        let mut journal_payload = Vec::new();
        journal_payload.extend_from_slice(message_id.as_bytes());
        journal_payload.push(0);
        journal_payload.extend_from_slice(thread_id.as_bytes());
        journal_payload.push(0);
        journal_payload.extend_from_slice(from.as_bytes());
        journal_payload.push(0);
        journal_payload.extend_from_slice(to.as_bytes());

        let batch = vec![
            wal::WalEntry::new(wal::EntryType::ThreadExtend, dispatch_payload),
            wal::WalEntry::new(
                wal::EntryType::ContextAllocate,
                thread_id.as_bytes().to_vec(),
            ),
            wal::WalEntry::new(wal::EntryType::JournalDispatched, journal_payload),
        ];

        self.wal.append_batch(&batch)?;

        let new_uuid = self.threads.extend_chain(thread_id, to);
        self.contexts.create(thread_id)?;
        self.journal
            .log_dispatch_simple(message_id, thread_id, from, to);

        Ok(new_uuid)
    }

    /// Get a reference to the thread table.
    pub fn threads(&self) -> &ThreadTable {
        &self.threads
    }

    /// Get a mutable reference to the thread table.
    pub fn threads_mut(&mut self) -> &mut ThreadTable {
        &mut self.threads
    }

    /// Get a reference to the context store.
    pub fn contexts(&self) -> &ContextStore {
        &self.contexts
    }

    /// Get a mutable reference to the context store.
    pub fn contexts_mut(&mut self) -> &mut ContextStore {
        &mut self.contexts
    }

    /// Get a reference to the journal.
    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// Get a reference to the WAL.
    pub fn wal(&self) -> &Wal {
        &self.wal
    }

    /// Data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn kernel_opens_and_creates_dirs() {
        let dir = TempDir::new().unwrap();
        let _kernel = Kernel::open(&dir.path().join("data")).unwrap();
        assert!(dir.path().join("data").exists());
        assert!(dir.path().join("data/kernel.wal").exists());
    }

    #[test]
    fn kernel_dispatch_and_prune_lifecycle() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();

        // Initialize root thread
        let root = kernel.initialize_root("test", "root").unwrap();

        // Dispatch: extends chain root → handler
        let new_uuid = kernel
            .dispatch_message("console", "handler", &root, "msg-001")
            .unwrap();

        assert!(kernel.threads().lookup(&new_uuid).is_some());
        assert!(kernel.contexts().exists(&root));

        // Prune: handler responds
        let prune = kernel.prune_thread(&new_uuid).unwrap();
        assert!(prune.is_some());
    }

    #[test]
    fn kernel_crash_recovery() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        // First session: create some state
        {
            let mut kernel = Kernel::open(&data_dir).unwrap();
            kernel.initialize_root("test", "root").unwrap();
        }

        // Second session: reopen — WAL replay should recover state
        let kernel = Kernel::open(&data_dir).unwrap();
        // The root should exist (either from mmap or WAL replay)
        assert!(kernel.threads().root_uuid().is_some());
    }

    #[test]
    fn dispatch_verifies_all_three_stores() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();

        let root = kernel.initialize_root("org", "admin").unwrap();

        let new_uuid = kernel
            .dispatch_message("console", "handler", &root, "msg-100")
            .unwrap();

        // Thread table: new chain exists
        let chain = kernel.threads().lookup(&new_uuid);
        assert!(chain.is_some());
        assert!(chain.unwrap().contains("handler"));

        // Context store: context allocated for the source thread
        assert!(kernel.contexts().exists(&root));

        // Journal: dispatch entry recorded
        let entry = kernel.journal().get("msg-100");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.from, "console");
        assert_eq!(entry.to, "handler");
        assert_eq!(entry.status, journal::MessageStatus::Dispatched);
    }

    #[test]
    fn crash_mid_dispatch_recovery() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        {
            let mut kernel = Kernel::open(&data_dir).unwrap();
            let root = kernel.initialize_root("org", "admin").unwrap();

            let mut dispatch_payload = Vec::new();
            dispatch_payload.extend_from_slice(root.as_bytes());
            dispatch_payload.push(0);
            dispatch_payload.extend_from_slice(b"handler");

            let mut journal_payload = Vec::new();
            journal_payload.extend_from_slice(b"crash-msg");
            journal_payload.push(0);
            journal_payload.extend_from_slice(root.as_bytes());
            journal_payload.push(0);
            journal_payload.extend_from_slice(b"console");
            journal_payload.push(0);
            journal_payload.extend_from_slice(b"handler");

            let batch = vec![
                wal::WalEntry::new(wal::EntryType::ThreadExtend, dispatch_payload),
                wal::WalEntry::new(wal::EntryType::ContextAllocate, root.as_bytes().to_vec()),
                wal::WalEntry::new(wal::EntryType::JournalDispatched, journal_payload),
            ];

            kernel.wal.append_batch(&batch).unwrap();
        }

        let kernel = Kernel::open(&data_dir).unwrap();
        assert!(kernel.threads().root_uuid().is_some());

        let entry = kernel.journal().get("crash-msg");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().status, journal::MessageStatus::Dispatched);

        let root_uuid = kernel.threads().root_uuid().unwrap().to_string();
        assert!(kernel.contexts().exists(&root_uuid));
    }

    #[test]
    fn crash_mid_prune_recovery() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        let child_uuid;
        {
            let mut kernel = Kernel::open(&data_dir).unwrap();
            let root = kernel.initialize_root("org", "admin").unwrap();

            child_uuid = kernel
                .dispatch_message("console", "handler", &root, "msg-prune")
                .unwrap();

            let batch = vec![
                wal::WalEntry::new(wal::EntryType::ThreadPrune, child_uuid.as_bytes().to_vec()),
                wal::WalEntry::new(
                    wal::EntryType::ContextRelease,
                    child_uuid.as_bytes().to_vec(),
                ),
                wal::WalEntry::new(
                    wal::EntryType::JournalDelivered,
                    child_uuid.as_bytes().to_vec(),
                ),
            ];
            kernel.wal.append_batch(&batch).unwrap();
        }

        let kernel = Kernel::open(&data_dir).unwrap();
        assert!(!kernel.contexts().exists(&child_uuid));
    }

    #[test]
    fn undelivered_messages_found_after_crash() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        {
            let mut kernel = Kernel::open(&data_dir).unwrap();
            let root = kernel.initialize_root("org", "admin").unwrap();

            kernel
                .dispatch_message("console", "handler-a", &root, "msg-a")
                .unwrap();
            kernel
                .dispatch_message("console", "handler-b", &root, "msg-b")
                .unwrap();
            kernel
                .dispatch_message("console", "handler-c", &root, "msg-c")
                .unwrap();
        }

        let kernel = Kernel::open(&data_dir).unwrap();

        let undelivered = kernel.journal().find_undelivered();
        assert_eq!(undelivered.len(), 3);

        let ids: Vec<&str> = undelivered.iter().map(|e| e.message_id.as_str()).collect();
        assert!(ids.contains(&"msg-a"));
        assert!(ids.contains(&"msg-b"));
        assert!(ids.contains(&"msg-c"));
    }

    #[test]
    fn full_lifecycle_all_stores_consistent() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();

        let root = kernel.initialize_root("org", "admin").unwrap();
        assert!(kernel.threads().lookup(&root).is_some());
        assert_eq!(kernel.journal().count(), 0);

        let child = kernel
            .dispatch_message("console", "worker", &root, "msg-lifecycle")
            .unwrap();
        assert!(kernel.threads().lookup(&child).is_some());
        assert!(kernel.contexts().exists(&root));
        assert_eq!(kernel.journal().count(), 1);
        assert_eq!(
            kernel.journal().get("msg-lifecycle").unwrap().status,
            journal::MessageStatus::Dispatched
        );

        let prune = kernel.prune_thread(&child).unwrap();
        assert!(prune.is_some());
        let prune = prune.unwrap();
        assert_eq!(prune.target, "org");

        assert!(!kernel.contexts().exists(&child));
    }

    #[test]
    fn fold_thread_basic() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();

        let child = kernel
            .dispatch_message("console", "handler", &root, "msg-fold")
            .unwrap();

        kernel.contexts_mut().add_segment(
            &root,
            context_store::ContextSegment {
                id: "work".into(),
                tag: "code".into(),
                content: b"fn handler() { /* work */ }".to_vec(),
                status: context_store::SegmentStatus::Active,
                relevance: 0.8,
                created_at: 0,
                fold_ref: None,
            },
        ).unwrap();

        let result = kernel.fold_thread(&child, b"[handler completed work]").unwrap();
        assert!(result.is_some());
        assert!(!kernel.contexts().exists(&child));
    }

    #[test]
    fn fold_thread_preserves_parent() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();

        kernel.contexts_mut().create(&root).unwrap();
        kernel.contexts_mut().add_segment(
            &root,
            context_store::ContextSegment {
                id: "parent-data".into(),
                tag: "msg".into(),
                content: b"parent context data".to_vec(),
                status: context_store::SegmentStatus::Active,
                relevance: 0.9,
                created_at: 0,
                fold_ref: None,
            },
        ).unwrap();

        let child = kernel
            .dispatch_message("console", "handler", &root, "msg-fp")
            .unwrap();

        kernel.fold_thread(&child, b"[summary]").unwrap();

        let parent_seg = kernel.contexts().get_segment(&root, "parent-data").unwrap();
        assert_eq!(parent_seg.content, b"parent context data");
    }

    #[test]
    fn fold_thread_wal_batch() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();
        let child = kernel
            .dispatch_message("console", "handler", &root, "msg-wb")
            .unwrap();

        kernel.fold_thread(&child, b"[folded]").unwrap();
        assert!(kernel.wal().size().unwrap() > 0);
    }

    #[test]
    fn fold_thread_crash_recovery() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");

        let child_uuid;
        {
            let mut kernel = Kernel::open(&data_dir).unwrap();
            let root = kernel.initialize_root("org", "admin").unwrap();
            child_uuid = kernel
                .dispatch_message("console", "handler", &root, "msg-cr")
                .unwrap();
            kernel.fold_thread(&child_uuid, b"[recovered fold]").unwrap();
        }

        let kernel = Kernel::open(&data_dir).unwrap();
        assert!(!kernel.contexts().exists(&child_uuid));
    }

    #[test]
    fn fold_thread_nonexistent_fails() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        kernel.initialize_root("org", "admin").unwrap();

        let result = kernel.fold_thread("nonexistent-uuid", b"[summary]").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn prune_thread_still_works() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();
        let child = kernel
            .dispatch_message("console", "handler", &root, "msg-pt")
            .unwrap();

        let result = kernel.prune_thread(&child).unwrap();
        assert!(result.is_some());
        assert!(!kernel.contexts().exists(&child));
    }

    #[test]
    fn fold_thread_child_context_released() {
        let dir = TempDir::new().unwrap();
        let mut kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let root = kernel.initialize_root("org", "admin").unwrap();
        let child = kernel
            .dispatch_message("console", "handler", &root, "msg-ccr")
            .unwrap();

        assert!(kernel.contexts().exists(&root));

        kernel.fold_thread(&child, b"[done]").unwrap();
        assert!(!kernel.contexts().exists(&child));
    }
}
