# agentos-kernel API

## Kernel

```rust
pub struct Kernel { ... }

impl Kernel {
    pub fn open(data_dir: &Path) -> KernelResult<Self>;
    pub fn initialize_root(&mut self, organism_name: &str, profile: &str) -> KernelResult<String>;
    pub fn dispatch_message(&mut self, from: &str, to: &str, thread_id: &str, message_id: &str) -> KernelResult<String>;
    pub fn prune_thread(&mut self, thread_id: &str) -> KernelResult<Option<PruneResult>>;
    pub fn fold_thread(&mut self, thread_id: &str, summary: &[u8]) -> KernelResult<Option<PruneResult>>;
    pub fn threads(&self) -> &ThreadTable;
    pub fn threads_mut(&mut self) -> &mut ThreadTable;
    pub fn contexts(&self) -> &ContextStore;
    pub fn contexts_mut(&mut self) -> &mut ContextStore;
    pub fn journal(&self) -> &Journal;
    pub fn wal(&self) -> &Wal;
    pub fn data_dir(&self) -> &Path;
}
```

## ThreadTable

```rust
pub struct PruneResult { pub target: String, pub thread_id: String }
pub struct ThreadRecord { pub uuid: String, pub chain: String, pub profile: String, pub created_at: u64 }

impl ThreadTable {
    pub fn open(path: &Path) -> KernelResult<Self>;
    pub fn initialize_root(&mut self, organism_name: &str, profile: &str) -> String;
    pub fn root_uuid(&self) -> Option<&str>;
    pub fn lookup(&self, thread_id: &str) -> Option<&str>;
    pub fn get_profile(&self, thread_id: &str) -> Option<&str>;
    pub fn register_thread(&mut self, thread_id: &str, initiator: &str, target: &str, profile: &str) -> String;
    pub fn extend_chain(&mut self, current_uuid: &str, next_hop: &str) -> String;
    pub fn prune_for_response(&mut self, thread_id: &str) -> Option<PruneResult>;
    pub fn peek_prune(&self, thread_id: &str) -> Option<PruneResult>;
    pub fn cleanup(&mut self, thread_id: &str);
    pub fn get_record(&self, thread_id: &str) -> Option<&ThreadRecord>;
    pub fn all_records(&self) -> impl Iterator<Item = &ThreadRecord>;
    pub fn count(&self) -> usize;
}
```

## ContextStore

```rust
pub enum SegmentStatus { Active, Shelved, Folded }
pub struct ContextSegment { pub id: String, pub tag: String, pub content: Vec<u8>, pub status: SegmentStatus, pub relevance: f32, pub created_at: u64, pub fold_ref: Option<String> }
pub struct SegmentMeta { pub id: String, pub tag: String, pub size: usize, pub status: SegmentStatus, pub relevance: f32, pub created_at: u64 }
pub struct ContextInventory { pub thread_id: String, pub total_segments: usize, pub active_count: usize, pub shelved_count: usize, pub folded_count: usize, pub total_bytes: usize, pub segments: Vec<SegmentMeta> }

impl ContextStore {
    pub fn open(path: &Path) -> KernelResult<Self>;
    pub fn create(&mut self, thread_id: &str) -> KernelResult<()>;
    pub fn exists(&self, thread_id: &str) -> bool;
    pub fn get(&self, thread_id: &str) -> Option<&ThreadContext>;
    pub fn release(&mut self, thread_id: &str) -> KernelResult<()>;
    pub fn add_segment(&mut self, thread_id: &str, segment: ContextSegment) -> KernelResult<()>;
    pub fn remove_segment(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()>;
    pub fn get_segment(&self, thread_id: &str, segment_id: &str) -> Option<&ContextSegment>;
    pub fn page_in(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()>;
    pub fn page_out(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()>;
    pub fn update_relevance(&mut self, thread_id: &str, segment_id: &str, score: f32) -> KernelResult<()>;
    pub fn inventory(&self, thread_id: &str) -> Option<ContextInventory>;
    pub fn working_set(&self, thread_id: &str) -> Vec<&ContextSegment>;
    pub fn fold(&mut self, thread_id: &str, segment_id: &str, summary: Vec<u8>) -> KernelResult<()>;
    pub fn unfold(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<Option<Vec<u8>>>;
}
```

## Journal

```rust
pub enum MessageStatus { Dispatched, Delivered, Failed }
pub enum RetentionPolicy { Forever, PruneOnDelivery, RetainDays(u16) }
pub struct JournalEntry { pub message_id: String, pub thread_id: String, pub from: String, pub to: String, pub status: MessageStatus, pub dispatched_at: u64, pub delivered_at: u64, pub retention: RetentionPolicy, pub failure_reason: Option<String> }

impl Journal {
    pub fn open(path: &Path) -> KernelResult<Self>;
    pub fn log_dispatch(&mut self, entry: JournalEntry);
    pub fn log_dispatch_simple(&mut self, message_id: &str, thread_id: &str, from: &str, to: &str);
    pub fn mark_delivered(&mut self, message_id: &str);
    pub fn mark_delivered_by_thread(&mut self, thread_id: &str);
    pub fn mark_failed(&mut self, message_id: &str, reason: &str);
    pub fn find_undelivered(&self) -> Vec<&JournalEntry>;
    pub fn sweep(&mut self, now: u64) -> usize;
    pub fn get(&self, message_id: &str) -> Option<&JournalEntry>;
    pub fn all_entries(&self) -> impl Iterator<Item = &JournalEntry>;
    pub fn count(&self) -> usize;
}
```

## Error Types

```rust
pub enum KernelError { Wal(String), WalCorrupted { offset, reason }, ThreadNotFound(String), ThreadTableFull(u32), ContextNotFound(String), JournalEntryNotFound(String), Io(io::Error), InvalidData(String) }
pub type KernelResult<T> = Result<T, KernelError>;
```
