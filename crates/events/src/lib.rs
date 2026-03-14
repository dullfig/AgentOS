//! Pipeline events — broadcast channel for TUI and observers.
//!
//! Best-effort delivery: if a subscriber falls behind, `Lagged` errors
//! skip events. The TUI refreshes from kernel truth on the next tick.

/// Events emitted by the pipeline for observation.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// A message was successfully injected.
    MessageInjected {
        thread_id: String,
        target: String,
        profile: String,
    },
    /// A message was blocked by security policy.
    SecurityBlocked {
        profile: String,
        target: String,
    },
    /// Token usage from an LLM API call.
    TokenUsage {
        thread_id: String,
        input_tokens: u32,
        output_tokens: u32,
    },
    /// A kernel-level operation occurred.
    KernelOp {
        op: KernelOpType,
        thread_id: String,
    },
    /// Semantic router matched a tool.
    SemanticMatch {
        thread_id: String,
        tool_name: String,
        score: f32,
    },
    /// Form-filler attempt.
    FormFillAttempt {
        thread_id: String,
        tool_name: String,
        model: String,
        success: bool,
    },
    /// Coding agent produced a final response.
    AgentResponse {
        thread_id: String,
        agent_name: String,
        text: String,
    },
    /// Agent is about to call the LLM (thinking).
    AgentThinking {
        thread_id: String,
        agent_name: String,
    },
    /// A tool call has been dispatched.
    ToolDispatched {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        detail: String,
    },
    /// A tool call completed (result received).
    ToolCompleted {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        success: bool,
        detail: String,
    },
    /// Conversation state sync — full conversation for a thread (for TUI display).
    ConversationSync {
        thread_id: String,
        agent_name: String,
        entries: Vec<ConversationEntry>,
    },
    /// Tool permission check result (for activity trace).
    ToolApproval {
        thread_id: String,
        agent_name: String,
        tool_name: String,
        verdict: String, // "approved", "denied", "auto", "denied_by_policy"
    },
    /// Suspected prompt injection detected in tool output.
    InjectionDetected {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// User allowed suspected injection through (quarantined).
    InjectionAllowed {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// User blocked suspected injection — sanitized output sent to agent.
    InjectionBlocked {
        thread_id: String,
        tool_name: String,
        agent_name: String,
    },
    /// Agent sent a display-only message to the user (no response expected).
    UserDisplay {
        thread_id: String,
        agent_name: String,
        text: String,
    },
    /// Agent is asking the user a question (response expected).
    UserQuery {
        thread_id: String,
        agent_name: String,
        question: String,
    },
    /// Interactive buffer child wants TUI focus (child agent taking over).
    FocusAcquire {
        agent_name: String,
        parent_agent: String,
    },
    /// Interactive buffer child released TUI focus (returning to parent).
    FocusRelease {
        agent_name: String,
        parent_agent: String,
    },
}

/// A conversation entry for TUI display (lightweight, no raw API content).
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    /// "user", "assistant", or "tool_result"
    pub role: String,
    /// Truncated text or tool description.
    pub summary: String,
    /// Was this a tool_use block?
    pub is_tool_use: bool,
    /// Tool name if this was a tool_use or tool_result.
    pub tool_name: Option<String>,
    /// Whether this entry represents an error.
    pub is_error: bool,
}

/// Kernel operation types for event reporting.
#[derive(Debug, Clone)]
pub enum KernelOpType {
    ThreadCreated,
    ThreadPruned,
    ContextAllocated,
    ContextReleased,
    ContextFolded,
}
