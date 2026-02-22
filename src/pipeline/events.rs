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
        text: String,
    },
    /// Agent is about to call the LLM (thinking).
    AgentThinking {
        thread_id: String,
    },
    /// A tool call has been dispatched.
    ToolDispatched {
        thread_id: String,
        tool_name: String,
        detail: String,
    },
    /// A tool call completed (result received).
    ToolCompleted {
        thread_id: String,
        tool_name: String,
        success: bool,
        detail: String,
    },
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
