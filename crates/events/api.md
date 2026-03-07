# agentos-events API

```rust
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    MessageInjected { thread_id: String, target: String, profile: String },
    SecurityBlocked { profile: String, target: String },
    TokenUsage { thread_id: String, input_tokens: u32, output_tokens: u32 },
    KernelOp { op: KernelOpType, thread_id: String },
    SemanticMatch { thread_id: String, tool_name: String, score: f32 },
    FormFillAttempt { thread_id: String, tool_name: String, model: String, success: bool },
    AgentResponse { thread_id: String, agent_name: String, text: String },
    AgentThinking { thread_id: String, agent_name: String },
    ToolDispatched { thread_id: String, agent_name: String, tool_name: String, detail: String },
    ToolCompleted { thread_id: String, agent_name: String, tool_name: String, success: bool, detail: String },
    ConversationSync { thread_id: String, agent_name: String, entries: Vec<ConversationEntry> },
    ToolApproval { thread_id: String, agent_name: String, tool_name: String, verdict: String },
}

#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub role: String,
    pub summary: String,
    pub is_tool_use: bool,
    pub tool_name: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub enum KernelOpType {
    ThreadCreated,
    ThreadPruned,
    ContextAllocated,
    ContextReleased,
    ContextFolded,
}
```
