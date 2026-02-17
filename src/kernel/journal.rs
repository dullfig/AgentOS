//! Message journal — audit trail for every dispatched message.
//!
//! Tracks dispatch/deliver/fail lifecycle. Supports retention policies
//! for cleanup (retain_forever, prune_on_delivery, retain_days).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::error::KernelResult;
use super::wal::{EntryType, WalEntry};

/// Retention policy for journal entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Keep forever (coding agent).
    Forever,
    /// Delete after delivery confirmation.
    PruneOnDelivery,
    /// Keep for N days.
    RetainDays(u16),
}

/// Status of a journal entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageStatus {
    Dispatched,
    Delivered,
    Failed,
}

/// A single journal entry.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    pub message_id: String,
    pub thread_id: String,
    pub from: String,
    pub to: String,
    pub status: MessageStatus,
    pub dispatched_at: u64,
    pub delivered_at: u64,
    pub retention: RetentionPolicy,
    pub failure_reason: Option<String>,
}

/// The message journal.
#[allow(dead_code)]
pub struct Journal {
    /// message_id → JournalEntry
    entries: HashMap<String, JournalEntry>,
    /// Path for persistence
    path: PathBuf,
}

impl Journal {
    /// Open or create the journal.
    pub fn open(path: &Path) -> KernelResult<Self> {
        Ok(Self {
            entries: HashMap::new(),
            path: path.to_path_buf(),
        })
    }

    /// Apply a WAL entry during replay.
    pub fn apply_wal_entry(&mut self, entry: &WalEntry) {
        match entry.entry_type {
            EntryType::JournalDispatched => {
                // Payload: message_id\0thread_id\0from\0to
                if let Some(je) = Self::parse_dispatch_payload(&entry.payload) {
                    self.entries.insert(je.message_id.clone(), je);
                }
            }
            EntryType::JournalDelivered => {
                let key = String::from_utf8_lossy(&entry.payload).to_string();
                // Could be a message_id or thread_id
                if let Some(e) = self.entries.get_mut(&key) {
                    e.status = MessageStatus::Delivered;
                    e.delivered_at = now_millis();
                }
            }
            EntryType::JournalFailed => {
                // Payload: message_id\0reason
                if let Some((id, reason)) = Self::parse_fail_payload(&entry.payload) {
                    if let Some(e) = self.entries.get_mut(&id) {
                        e.status = MessageStatus::Failed;
                        e.failure_reason = Some(reason);
                    }
                }
            }
            _ => {} // not a journal op
        }
    }

    /// Log a message dispatch.
    pub fn log_dispatch(&mut self, entry: JournalEntry) {
        self.entries.insert(entry.message_id.clone(), entry);
    }

    /// Simplified dispatch logging (used by Kernel).
    pub fn log_dispatch_simple(&mut self, message_id: &str, thread_id: &str, from: &str, to: &str) {
        let entry = JournalEntry {
            message_id: message_id.to_string(),
            thread_id: thread_id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            status: MessageStatus::Dispatched,
            dispatched_at: now_millis(),
            delivered_at: 0,
            retention: RetentionPolicy::Forever,
            failure_reason: None,
        };
        self.entries.insert(message_id.to_string(), entry);
    }

    /// Build a WAL entry for dispatch.
    pub fn wal_entry_dispatch(message_id: &str, thread_id: &str, from: &str, to: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(message_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(from.as_bytes());
        payload.push(0);
        payload.extend_from_slice(to.as_bytes());
        WalEntry::new(EntryType::JournalDispatched, payload)
    }

    /// Mark a message as delivered.
    pub fn mark_delivered(&mut self, message_id: &str) {
        if let Some(entry) = self.entries.get_mut(message_id) {
            entry.status = MessageStatus::Delivered;
            entry.delivered_at = now_millis();
        }
    }

    /// Mark all messages for a thread as delivered.
    pub fn mark_delivered_by_thread(&mut self, thread_id: &str) {
        for entry in self.entries.values_mut() {
            if entry.thread_id == thread_id && entry.status == MessageStatus::Dispatched {
                entry.status = MessageStatus::Delivered;
                entry.delivered_at = now_millis();
            }
        }
    }

    /// Mark a message as failed.
    pub fn mark_failed(&mut self, message_id: &str, reason: &str) {
        if let Some(entry) = self.entries.get_mut(message_id) {
            entry.status = MessageStatus::Failed;
            entry.failure_reason = Some(reason.to_string());
        }
    }

    /// Find all undelivered (dispatched but not delivered/failed) messages.
    pub fn find_undelivered(&self) -> Vec<&JournalEntry> {
        self.entries
            .values()
            .filter(|e| e.status == MessageStatus::Dispatched)
            .collect()
    }

    /// Sweep entries according to retention policy.
    /// Returns number of entries removed.
    pub fn sweep(&mut self, now: u64) -> usize {
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| match e.retention {
                RetentionPolicy::PruneOnDelivery => e.status == MessageStatus::Delivered,
                RetentionPolicy::RetainDays(days) => {
                    let age_millis = now.saturating_sub(e.dispatched_at);
                    let day_millis = days as u64 * 24 * 60 * 60 * 1000;
                    age_millis > day_millis
                }
                RetentionPolicy::Forever => false,
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.entries.remove(&id);
        }
        count
    }

    /// Get a journal entry by message ID.
    pub fn get(&self, message_id: &str) -> Option<&JournalEntry> {
        self.entries.get(message_id)
    }

    /// Number of entries.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    fn parse_dispatch_payload(payload: &[u8]) -> Option<JournalEntry> {
        let s = String::from_utf8_lossy(payload);
        let parts: Vec<&str> = s.splitn(4, '\0').collect();
        if parts.len() >= 4 {
            Some(JournalEntry {
                message_id: parts[0].to_string(),
                thread_id: parts[1].to_string(),
                from: parts[2].to_string(),
                to: parts[3].to_string(),
                status: MessageStatus::Dispatched,
                dispatched_at: now_millis(),
                delivered_at: 0,
                retention: RetentionPolicy::Forever,
                failure_reason: None,
            })
        } else {
            None
        }
    }

    fn parse_fail_payload(payload: &[u8]) -> Option<(String, String)> {
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
    fn dispatch_and_deliver_lifecycle() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        journal.log_dispatch_simple("msg-1", "thread-1", "alice", "bob");
        assert_eq!(journal.count(), 1);

        let entry = journal.get("msg-1").unwrap();
        assert_eq!(entry.status, MessageStatus::Dispatched);

        let undelivered = journal.find_undelivered();
        assert_eq!(undelivered.len(), 1);

        journal.mark_delivered("msg-1");
        let entry = journal.get("msg-1").unwrap();
        assert_eq!(entry.status, MessageStatus::Delivered);
        assert!(entry.delivered_at > 0);

        let undelivered = journal.find_undelivered();
        assert!(undelivered.is_empty());
    }

    #[test]
    fn mark_failed() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        journal.log_dispatch_simple("msg-1", "thread-1", "alice", "bob");
        journal.mark_failed("msg-1", "handler panicked");

        let entry = journal.get("msg-1").unwrap();
        assert_eq!(entry.status, MessageStatus::Failed);
        assert_eq!(entry.failure_reason.as_deref(), Some("handler panicked"));
    }

    #[test]
    fn find_undelivered_after_crash() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        // Simulate: 3 messages dispatched, only 1 delivered
        journal.log_dispatch_simple("msg-1", "t1", "a", "b");
        journal.log_dispatch_simple("msg-2", "t2", "a", "c");
        journal.log_dispatch_simple("msg-3", "t3", "a", "d");
        journal.mark_delivered("msg-2");

        let undelivered = journal.find_undelivered();
        assert_eq!(undelivered.len(), 2);
    }

    #[test]
    fn retention_sweep_prune_on_delivery() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        let mut entry = JournalEntry {
            message_id: "msg-1".into(),
            thread_id: "t1".into(),
            from: "a".into(),
            to: "b".into(),
            status: MessageStatus::Delivered,
            dispatched_at: now_millis(),
            delivered_at: now_millis(),
            retention: RetentionPolicy::PruneOnDelivery,
            failure_reason: None,
        };
        journal.log_dispatch(entry.clone());

        entry.message_id = "msg-2".into();
        entry.retention = RetentionPolicy::Forever;
        journal.log_dispatch(entry);

        let removed = journal.sweep(now_millis());
        assert_eq!(removed, 1);
        assert_eq!(journal.count(), 1);
        assert!(journal.get("msg-2").is_some()); // forever entry kept
    }

    #[test]
    fn retention_sweep_retain_days_expired() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        // Entry dispatched 10 days ago with retain_days: 7
        let ten_days_ago = now_millis() - (10 * 24 * 60 * 60 * 1000);
        journal.log_dispatch(JournalEntry {
            message_id: "old-msg".into(),
            thread_id: "t1".into(),
            from: "a".into(),
            to: "b".into(),
            status: MessageStatus::Delivered,
            dispatched_at: ten_days_ago,
            delivered_at: ten_days_ago + 1000,
            retention: RetentionPolicy::RetainDays(7),
            failure_reason: None,
        });

        // Entry dispatched 3 days ago with retain_days: 7 — should survive
        let three_days_ago = now_millis() - (3 * 24 * 60 * 60 * 1000);
        journal.log_dispatch(JournalEntry {
            message_id: "recent-msg".into(),
            thread_id: "t2".into(),
            from: "a".into(),
            to: "c".into(),
            status: MessageStatus::Delivered,
            dispatched_at: three_days_ago,
            delivered_at: three_days_ago + 1000,
            retention: RetentionPolicy::RetainDays(7),
            failure_reason: None,
        });

        // Forever entry — should always survive
        journal.log_dispatch(JournalEntry {
            message_id: "forever-msg".into(),
            thread_id: "t3".into(),
            from: "a".into(),
            to: "d".into(),
            status: MessageStatus::Delivered,
            dispatched_at: ten_days_ago,
            delivered_at: ten_days_ago + 1000,
            retention: RetentionPolicy::Forever,
            failure_reason: None,
        });

        let removed = journal.sweep(now_millis());
        assert_eq!(removed, 1); // only old-msg expired
        assert_eq!(journal.count(), 2);
        assert!(journal.get("old-msg").is_none());
        assert!(journal.get("recent-msg").is_some());
        assert!(journal.get("forever-msg").is_some());
    }

    #[test]
    fn wal_replay_recovers_journal() {
        let dir = TempDir::new().unwrap();
        let mut journal = Journal::open(&dir.path().join("journal.bin")).unwrap();

        // Simulate WAL replay
        let dispatch_entry = WalEntry::new(
            EntryType::JournalDispatched,
            b"msg-1\0thread-1\0alice\0bob".to_vec(),
        );
        journal.apply_wal_entry(&dispatch_entry);

        assert_eq!(journal.count(), 1);
        let entry = journal.get("msg-1").unwrap();
        assert_eq!(entry.from, "alice");
        assert_eq!(entry.to, "bob");
        assert_eq!(entry.status, MessageStatus::Dispatched);
    }
}
