//! Buffer management within instances — per-channel conversation isolation.
//!
//! Each agent instance can have multiple active buffers (conversation threads),
//! one per channel-interaction. Buffers share the instance's identity, persona,
//! and KV cache, but have independent message histories.
//!
//! # Examples
//!
//! ```text
//! concierge[alice]              ← the instance
//! ├── dm                        ← Alice's private DM thread
//! ├── help[email-issue]         ← active help session
//! ├── public[thread-9281]       ← Alice's reply in a public thread
//! └── hotel-coordination        ← spawned task buffer, still alive
//! ```
//!
//! Buffers are created lazily on first message (same materialization-on-routing
//! principle as instances). They track metadata but don't own message content —
//! the kernel's ThreadContext holds the actual conversation state.

use std::collections::HashMap;
use std::time::Instant;

use crate::address::Address;

/// Identifies a buffer within an instance.
///
/// Derived from the address: for `concierge[alice].help[email-issue]`,
/// the buffer id is `BufferId { name: "help", key: Some("email-issue") }`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferId {
    /// Buffer name (the segment name after the organism, e.g., "dm", "help", "public").
    pub name: String,
    /// Optional sub-key (e.g., "email-issue" in `help[email-issue]`).
    pub key: Option<String>,
}

impl BufferId {
    /// The default buffer for instances with no explicit buffer segment.
    pub fn default_buffer() -> Self {
        Self {
            name: "default".to_string(),
            key: None,
        }
    }

    /// Create from an address's buffer segment.
    pub fn from_address(address: &Address) -> Self {
        match address.buffer() {
            Some(segment) => Self {
                name: segment.name().to_string(),
                key: segment.key().map(|k| k.to_string()),
            },
            None => Self::default_buffer(),
        }
    }

    /// Canonical string form for display and kernel thread naming.
    pub fn canonical(&self) -> String {
        match &self.key {
            Some(k) => format!("{}[{}]", self.name, k),
            None => self.name.clone(),
        }
    }
}

impl std::fmt::Display for BufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.canonical())
    }
}

/// The channel type determines tone, topic scope, and behavior constraints.
/// Derived from the buffer name or explicitly set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Private direct message — warm, conversational, longer-form.
    Dm,
    /// Public thread reply — brief, in-lane, don't monopolize.
    Public,
    /// In-app help widget — friendly, troubleshooty, expanded topic scope.
    Help,
    /// Long-running task (hotel coordination, event planning) — focused, persistent.
    Task,
    /// Default channel when no specific type is detected.
    Default,
}

impl ChannelType {
    /// Infer channel type from buffer name.
    pub fn from_name(name: &str) -> Self {
        match name {
            "dm" | "direct" | "message" => ChannelType::Dm,
            "public" | "thread" | "feed" => ChannelType::Public,
            "help" | "support" => ChannelType::Help,
            "task" | "coordination" | "planning" => ChannelType::Task,
            "default" => ChannelType::Default,
            _ => ChannelType::Default,
        }
    }

    /// The channel-aware sub-prompt block to inject into the system prompt.
    /// Returns None for Default (no special instructions).
    pub fn sub_prompt(&self) -> Option<&'static str> {
        match self {
            ChannelType::Dm => Some(
                "You're in a private direct message. Tone: warm, conversational, \
                 personal. You can be longer-form. Reference shared history freely."
            ),
            ChannelType::Public => Some(
                "You're replying in a public thread visible to all members. \
                 Tone: brief, helpful, personable but professional. Stay rigorously \
                 in your lane — if asked anything off-topic, decline gracefully \
                 rather than redirecting. Don't monopolize the thread."
            ),
            ChannelType::Help => Some(
                "You're in the in-app help widget. The user needs platform \
                 assistance. Tone: friendly, troubleshooty, concise. You can answer \
                 questions about platform features, navigation, account settings, \
                 and troubleshooting — these ARE in your lane here. Keep replies short."
            ),
            ChannelType::Task => Some(
                "You're managing a long-running task. Stay focused on the task \
                 objective. Track progress, report status, coordinate with other \
                 agents if needed. Be thorough but concise in updates."
            ),
            ChannelType::Default => None,
        }
    }
}

/// Metadata about an active buffer within an instance.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Buffer identifier.
    pub id: BufferId,
    /// Inferred channel type.
    pub channel: ChannelType,
    /// Kernel thread_id for this buffer's conversation state.
    /// Distinct from the instance's thread_id — each buffer gets its own thread.
    pub thread_id: String,
    /// When this buffer was created.
    pub created_at: Instant,
    /// When this buffer last received or sent a message.
    pub last_accessed: Instant,
    /// Number of messages delivered to this buffer.
    pub message_count: u64,
}

/// Per-instance buffer store.
///
/// Tracks all active buffers for a single agent instance. Keyed by BufferId.
/// The store is owned by the instance — when the instance is evicted, all
/// its buffers go with it.
#[derive(Debug, Clone, Default)]
pub struct BufferStore {
    buffers: HashMap<BufferId, BufferInfo>,
    next_thread_suffix: u64,
}

impl BufferStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a buffer. Returns (buffer_info, was_created).
    ///
    /// If the buffer already exists, updates last_accessed and returns it.
    /// If not, creates it with a new thread_id derived from the instance's
    /// thread_id + a suffix.
    pub fn get_or_create(
        &mut self,
        id: BufferId,
        instance_thread_id: &str,
    ) -> (&BufferInfo, bool) {
        let channel = ChannelType::from_name(&id.name);

        if self.buffers.contains_key(&id) {
            let info = self.buffers.get_mut(&id).unwrap();
            info.last_accessed = Instant::now();
            (info, false)
        } else {
            self.next_thread_suffix += 1;
            let thread_id = format!(
                "{}/buf-{:03}",
                instance_thread_id, self.next_thread_suffix
            );

            let now = Instant::now();
            let info = BufferInfo {
                id: id.clone(),
                channel,
                thread_id,
                created_at: now,
                last_accessed: now,
                message_count: 0,
            };

            self.buffers.insert(id.clone(), info);
            (self.buffers.get(&id).unwrap(), true)
        }
    }

    /// Look up a buffer by id.
    pub fn get(&self, id: &BufferId) -> Option<&BufferInfo> {
        self.buffers.get(id)
    }

    /// Record a message delivery to a buffer.
    pub fn record_message(&mut self, id: &BufferId) {
        if let Some(info) = self.buffers.get_mut(id) {
            info.message_count += 1;
            info.last_accessed = Instant::now();
        }
    }

    /// List all active buffers.
    pub fn list(&self) -> Vec<&BufferInfo> {
        self.buffers.values().collect()
    }

    /// Number of active buffers.
    pub fn count(&self) -> usize {
        self.buffers.len()
    }

    /// Remove a buffer. Returns the removed info if it existed.
    pub fn remove(&mut self, id: &BufferId) -> Option<BufferInfo> {
        self.buffers.remove(id)
    }

    /// Remove all buffers. Returns the count removed.
    pub fn clear(&mut self) -> usize {
        let count = self.buffers.len();
        self.buffers.clear();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_id_from_address_with_buffer() {
        let addr = Address::parse("concierge[alice].dm").unwrap();
        let id = BufferId::from_address(&addr);
        assert_eq!(id.name, "dm");
        assert_eq!(id.key, None);
        assert_eq!(id.canonical(), "dm");
    }

    #[test]
    fn buffer_id_from_address_with_subkey() {
        let addr = Address::parse("concierge[alice].help[email-issue]").unwrap();
        let id = BufferId::from_address(&addr);
        assert_eq!(id.name, "help");
        assert_eq!(id.key, Some("email-issue".to_string()));
        assert_eq!(id.canonical(), "help[email-issue]");
    }

    #[test]
    fn buffer_id_from_address_without_buffer() {
        let addr = Address::parse("concierge[alice]").unwrap();
        let id = BufferId::from_address(&addr);
        assert_eq!(id.name, "default");
        assert_eq!(id.key, None);
    }

    #[test]
    fn channel_type_inference() {
        assert_eq!(ChannelType::from_name("dm"), ChannelType::Dm);
        assert_eq!(ChannelType::from_name("direct"), ChannelType::Dm);
        assert_eq!(ChannelType::from_name("public"), ChannelType::Public);
        assert_eq!(ChannelType::from_name("thread"), ChannelType::Public);
        assert_eq!(ChannelType::from_name("help"), ChannelType::Help);
        assert_eq!(ChannelType::from_name("support"), ChannelType::Help);
        assert_eq!(ChannelType::from_name("task"), ChannelType::Task);
        assert_eq!(ChannelType::from_name("unknown"), ChannelType::Default);
    }

    #[test]
    fn sub_prompts_exist_for_typed_channels() {
        assert!(ChannelType::Dm.sub_prompt().is_some());
        assert!(ChannelType::Public.sub_prompt().is_some());
        assert!(ChannelType::Help.sub_prompt().is_some());
        assert!(ChannelType::Task.sub_prompt().is_some());
        assert!(ChannelType::Default.sub_prompt().is_none());
    }

    #[test]
    fn get_or_create_creates_on_first_access() {
        let mut store = BufferStore::new();
        let id = BufferId { name: "dm".into(), key: None };

        let (info, created) = store.get_or_create(id.clone(), "inst-001");
        assert!(created);
        assert_eq!(info.channel, ChannelType::Dm);
        assert!(info.thread_id.starts_with("inst-001/buf-"));
        assert_eq!(info.message_count, 0);
    }

    #[test]
    fn get_or_create_reuses_on_second_access() {
        let mut store = BufferStore::new();
        let id = BufferId { name: "dm".into(), key: None };

        let (_, created1) = store.get_or_create(id.clone(), "inst-001");
        assert!(created1);

        let (_, created2) = store.get_or_create(id, "inst-001");
        assert!(!created2);
    }

    #[test]
    fn unique_thread_ids_per_buffer() {
        let mut store = BufferStore::new();

        let (b1, _) = store.get_or_create(
            BufferId { name: "dm".into(), key: None },
            "inst-001",
        );
        let t1 = b1.thread_id.clone();

        let (b2, _) = store.get_or_create(
            BufferId { name: "help".into(), key: Some("issue-1".into()) },
            "inst-001",
        );
        let t2 = b2.thread_id.clone();

        assert_ne!(t1, t2);
        assert!(t1.starts_with("inst-001/buf-"));
        assert!(t2.starts_with("inst-001/buf-"));
    }

    #[test]
    fn record_message_increments() {
        let mut store = BufferStore::new();
        let id = BufferId { name: "dm".into(), key: None };

        store.get_or_create(id.clone(), "inst-001");
        store.record_message(&id);
        store.record_message(&id);
        store.record_message(&id);

        assert_eq!(store.get(&id).unwrap().message_count, 3);
    }

    #[test]
    fn list_and_count() {
        let mut store = BufferStore::new();
        store.get_or_create(BufferId { name: "dm".into(), key: None }, "inst-001");
        store.get_or_create(BufferId { name: "help".into(), key: None }, "inst-001");
        store.get_or_create(BufferId { name: "public".into(), key: Some("t-1".into()) }, "inst-001");

        assert_eq!(store.count(), 3);
        assert_eq!(store.list().len(), 3);
    }

    #[test]
    fn remove_buffer() {
        let mut store = BufferStore::new();
        let id = BufferId { name: "dm".into(), key: None };
        store.get_or_create(id.clone(), "inst-001");

        let removed = store.remove(&id);
        assert!(removed.is_some());
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn clear_all() {
        let mut store = BufferStore::new();
        store.get_or_create(BufferId { name: "dm".into(), key: None }, "inst-001");
        store.get_or_create(BufferId { name: "help".into(), key: None }, "inst-001");

        let cleared = store.clear();
        assert_eq!(cleared, 2);
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn subkeyed_buffers_are_distinct() {
        let mut store = BufferStore::new();

        store.get_or_create(
            BufferId { name: "help".into(), key: Some("issue-1".into()) },
            "inst-001",
        );
        store.get_or_create(
            BufferId { name: "help".into(), key: Some("issue-2".into()) },
            "inst-001",
        );

        // Two distinct buffers despite same name
        assert_eq!(store.count(), 2);
    }
}
