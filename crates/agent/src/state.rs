//! Agent state machine — per-thread conversation state.
//!
//! Each thread tracked by the CodingAgent has its own state machine:
//! Ready → AwaitingTools → Ready (loop until end_turn).
//!
//! Message history is bounded by a sliding window to prevent unbounded
//! memory growth. The first message (original task) is pinned, and a
//! synthetic summary is injected when older messages are pruned.

use agentos_events::{ContentBlock, Message, ToolResultBlock};

/// Default maximum number of messages to retain in a thread.
/// ~30 agentic turns (each turn ≈ 2-3 messages: user/assistant/tool_result).
const DEFAULT_MAX_MESSAGES: usize = 80;

/// Per-thread conversation state.
pub struct AgentThread {
    /// Conversation history, bounded by `max_messages`.
    pub messages: Vec<Message>,
    /// Current state in the agentic loop.
    pub state: AgentState,
    /// Maximum messages to retain. When exceeded, the oldest messages
    /// (except the first) are pruned and replaced with a summary.
    max_messages: usize,
    /// Number of messages that have been pruned over the thread's lifetime.
    pruned_count: usize,
}

/// State machine for the agentic loop.
pub enum AgentState {
    /// Ready for a new task or tool response.
    Ready,
    /// Waiting for tool results. Processing them one at a time (sequential).
    AwaitingTools {
        /// The assistant's content blocks (preserved for conversation history).
        assistant_blocks: Vec<ContentBlock>,
        /// The tool_use blocks to process (in order).
        pending: Vec<PendingToolCall>,
        /// Collected results so far.
        collected: Vec<ToolResultBlock>,
        /// Index of the tool call currently being dispatched.
        current_index: usize,
    },
}

/// A pending tool call extracted from an Opus response.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
}

impl Default for AgentThread {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            state: AgentState::Ready,
            max_messages: DEFAULT_MAX_MESSAGES,
            pruned_count: 0,
        }
    }
}

impl AgentThread {
    /// Create a new thread with initial system state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a thread with a custom message limit.
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            max_messages: max_messages.max(4), // minimum viable window
            ..Self::default()
        }
    }

    /// Add a user message to the conversation.
    pub fn push_user_message(&mut self, content: &str) {
        self.messages.push(Message::text("user", content));
        self.maybe_prune();
    }

    /// Add the assistant's response to the conversation history.
    pub fn push_assistant_blocks(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message::assistant_blocks(blocks));
        self.maybe_prune();
    }

    /// Add tool results to the conversation as a user message.
    pub fn push_tool_results(&mut self, results: Vec<ToolResultBlock>) {
        self.messages.push(Message::tool_results(results));
        self.maybe_prune();
    }

    /// Number of messages pruned over this thread's lifetime.
    pub fn pruned_count(&self) -> usize {
        self.pruned_count
    }

    /// Current message window limit.
    pub fn max_messages(&self) -> usize {
        self.max_messages
    }

    /// Prune old messages if we've exceeded the window limit.
    ///
    /// Strategy:
    /// - Keep message[0]: the original task (always role: "user")
    /// - Keep the most recent (max_messages - 2) messages
    /// - Insert a synthetic assistant summary at position 1 to bridge the gap
    ///   and maintain the required user/assistant alternation
    ///
    /// The result is always: [original_task, summary, ...recent_messages]
    /// which satisfies the API's "start with user, alternate roles" constraint
    /// regardless of what the first recent message's role is, because the
    /// summary (assistant) sits between the pinned user message and the window.
    fn maybe_prune(&mut self) {
        if self.messages.len() <= self.max_messages {
            return;
        }

        // Need at least: pinned[0] + summary + 2 recent messages
        if self.messages.len() < 4 {
            return;
        }

        let keep_recent = self.max_messages.saturating_sub(2); // room for pinned + summary
        let drop_start = 1; // after the pinned first message
        let drop_end = self.messages.len() - keep_recent;

        if drop_end <= drop_start {
            return;
        }

        let n_dropping = drop_end - drop_start;
        self.pruned_count += n_dropping;

        // Check if there's already a summary at position 1 from a prior prune
        let has_existing_summary = self.messages.len() > 1
            && self.messages[1].role == "assistant"
            && self.messages[1].content.text()
                .map(|t| t.starts_with("[Earlier conversation pruned"))
                .unwrap_or(false);

        let summary = Message::text(
            "assistant",
            &format!(
                "[Earlier conversation pruned: {} messages removed, {} total pruned. \
                 The original task is preserved above. Continuing from recent context.]",
                n_dropping, self.pruned_count
            ),
        );

        // Remove the old middle section
        self.messages.drain(drop_start..drop_end);

        // Insert or replace the summary at position 1
        if has_existing_summary && self.messages.len() > 1 {
            self.messages[1] = summary;
        } else {
            self.messages.insert(1, summary);
        }
    }
}

impl AgentState {
    /// Get the next pending tool call, if any.
    pub fn next_pending(&self) -> Option<&PendingToolCall> {
        match self {
            AgentState::AwaitingTools {
                pending,
                current_index,
                ..
            } => pending.get(*current_index),
            _ => None,
        }
    }

    /// Check if all tool results have been collected.
    pub fn all_collected(&self) -> bool {
        match self {
            AgentState::AwaitingTools {
                pending,
                collected,
                ..
            } => collected.len() >= pending.len(),
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_thread_is_ready() {
        let thread = AgentThread::new();
        assert!(thread.messages.is_empty());
        assert!(matches!(thread.state, AgentState::Ready));
    }

    #[test]
    fn push_user_message() {
        let mut thread = AgentThread::new();
        thread.push_user_message("Hello");
        assert_eq!(thread.messages.len(), 1);
        assert_eq!(thread.messages[0].role, "user");
    }

    #[test]
    fn push_assistant_blocks() {
        let mut thread = AgentThread::new();
        thread.push_assistant_blocks(vec![ContentBlock::Text {
            text: "Hi!".into(),
        }]);
        assert_eq!(thread.messages.len(), 1);
        assert_eq!(thread.messages[0].role, "assistant");
    }

    #[test]
    fn push_tool_results() {
        let mut thread = AgentThread::new();
        thread.push_tool_results(vec![ToolResultBlock {
            tool_use_id: "t1".into(),
            content: "42".into(),
            is_error: false,
        }]);
        assert_eq!(thread.messages.len(), 1);
        assert_eq!(thread.messages[0].role, "user");
    }

    #[test]
    fn awaiting_tools_next_pending() {
        let state = AgentState::AwaitingTools {
            assistant_blocks: vec![],
            pending: vec![
                PendingToolCall {
                    tool_use_id: "t1".into(),
                    tool_name: "shell".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
                PendingToolCall {
                    tool_use_id: "t2".into(),
                    tool_name: "file-ops".into(),
                    input: serde_json::json!({"action": "read", "path": "foo.rs"}),
                },
            ],
            collected: vec![],
            current_index: 0,
        };

        let next = state.next_pending().unwrap();
        assert_eq!(next.tool_name, "shell");
    }

    #[test]
    fn awaiting_tools_all_collected() {
        let state = AgentState::AwaitingTools {
            assistant_blocks: vec![],
            pending: vec![PendingToolCall {
                tool_use_id: "t1".into(),
                tool_name: "shell".into(),
                input: serde_json::json!({}),
            }],
            collected: vec![ToolResultBlock {
                tool_use_id: "t1".into(),
                content: "ok".into(),
                is_error: false,
            }],
            current_index: 1,
        };
        assert!(state.all_collected());
    }

    #[test]
    fn ready_state_all_collected() {
        let state = AgentState::Ready;
        assert!(state.all_collected());
    }

    #[test]
    fn default_max_messages() {
        let thread = AgentThread::new();
        assert_eq!(thread.max_messages(), DEFAULT_MAX_MESSAGES);
        assert_eq!(thread.pruned_count(), 0);
    }

    #[test]
    fn custom_max_messages() {
        let thread = AgentThread::with_max_messages(20);
        assert_eq!(thread.max_messages(), 20);
    }

    #[test]
    fn min_max_messages_clamped() {
        let thread = AgentThread::with_max_messages(1);
        assert_eq!(thread.max_messages(), 4); // clamped to minimum
    }

    #[test]
    fn no_pruning_under_limit() {
        let mut thread = AgentThread::with_max_messages(10);
        for i in 0..8 {
            if i % 2 == 0 {
                thread.push_user_message(&format!("msg {i}"));
            } else {
                thread.push_assistant_blocks(vec![ContentBlock::Text {
                    text: format!("reply {i}"),
                }]);
            }
        }
        assert_eq!(thread.messages.len(), 8);
        assert_eq!(thread.pruned_count(), 0);
    }

    #[test]
    fn pruning_triggers_at_limit() {
        let mut thread = AgentThread::with_max_messages(6);
        // Push 7 messages to trigger pruning
        thread.push_user_message("original task"); // pinned
        thread.push_assistant_blocks(vec![ContentBlock::Text { text: "r1".into() }]);
        thread.push_user_message("u2");
        thread.push_assistant_blocks(vec![ContentBlock::Text { text: "r2".into() }]);
        thread.push_user_message("u3");
        thread.push_assistant_blocks(vec![ContentBlock::Text { text: "r3".into() }]);
        thread.push_user_message("u4"); // 7th message → triggers prune

        // Should be: pinned + summary + 4 recent = 6
        assert!(thread.messages.len() <= 6, "len={}", thread.messages.len());
        assert!(thread.pruned_count() > 0);

        // First message preserved
        assert_eq!(thread.messages[0].role, "user");
        assert_eq!(
            thread.messages[0].content.text().unwrap(),
            "original task"
        );

        // Summary at position 1
        assert_eq!(thread.messages[1].role, "assistant");
        assert!(thread.messages[1].content.text().unwrap().contains("pruned"));
    }

    #[test]
    fn repeated_pruning_accumulates_count() {
        let mut thread = AgentThread::with_max_messages(6);
        thread.push_user_message("task");
        // Push enough to trigger multiple prunes
        for i in 0..20 {
            if i % 2 == 0 {
                thread.push_assistant_blocks(vec![ContentBlock::Text {
                    text: format!("r{i}"),
                }]);
            } else {
                thread.push_user_message(&format!("u{i}"));
            }
        }
        assert!(thread.messages.len() <= 6);
        assert!(thread.pruned_count() > 1);

        // Original task still pinned
        assert_eq!(
            thread.messages[0].content.text().unwrap(),
            "task"
        );
    }

    #[test]
    fn alternation_preserved_after_prune() {
        let mut thread = AgentThread::with_max_messages(6);
        thread.push_user_message("task");
        for i in 0..10 {
            if i % 2 == 0 {
                thread.push_assistant_blocks(vec![ContentBlock::Text {
                    text: format!("a{i}"),
                }]);
            } else {
                thread.push_user_message(&format!("u{i}"));
            }
        }

        // First message is always user
        assert_eq!(thread.messages[0].role, "user");
        // Second is always assistant (the summary)
        assert_eq!(thread.messages[1].role, "assistant");
    }
}
