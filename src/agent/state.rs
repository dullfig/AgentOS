//! Agent state machine — per-thread conversation state.
//!
//! Each thread tracked by the CodingAgent has its own state machine:
//! Ready → AwaitingTools → Ready (loop until end_turn).

use crate::llm::types::{ContentBlock, Message, ToolResultBlock};

/// Per-thread conversation state.
pub struct AgentThread {
    /// Full conversation history for this thread.
    pub messages: Vec<Message>,
    /// Current state in the agentic loop.
    pub state: AgentState,
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
        }
    }
}

impl AgentThread {
    /// Create a new thread with initial system state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a user message to the conversation.
    pub fn push_user_message(&mut self, content: &str) {
        self.messages.push(Message::text("user", content));
    }

    /// Add the assistant's response to the conversation history.
    pub fn push_assistant_blocks(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message::assistant_blocks(blocks));
    }

    /// Add tool results to the conversation as a user message.
    pub fn push_tool_results(&mut self, results: Vec<ToolResultBlock>) {
        self.messages.push(Message::tool_results(results));
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
}
