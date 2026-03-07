# agentos-events

Shared event types for the AgentOS pipeline broadcast channel.

## Purpose

Zero-dependency crate containing the event types that flow between pipeline, agents, TUI, and observers. Extracted because these types are imported by 6+ modules across the codebase.

## Types

- **PipelineEvent** — Main enum: MessageInjected, SecurityBlocked, TokenUsage, KernelOp, SemanticMatch, FormFillAttempt, AgentResponse, AgentThinking, ToolDispatched, ToolCompleted, ConversationSync, ToolApproval
- **ConversationEntry** — Lightweight conversation display record (role, summary, tool info)
- **KernelOpType** — ThreadCreated, ThreadPruned, ContextAllocated, ContextReleased, ContextFolded

## Consumers

- `agent/handler.rs` — emits AgentResponse, AgentThinking, ToolDispatched, ToolCompleted, ConversationSync
- `agent/middleware/debug_gate.rs` — emits debug events
- `agent/middleware/permission_gate.rs` — emits ToolApproval
- `buffer/mod.rs` — forwards child pipeline events
- `tui/app.rs` — displays all event types
- `tui/event.rs` — wraps PipelineEvent in TUI event loop
- `tui/runner.rs` — subscribes to event stream
- `pipeline/mod.rs` — emits MessageInjected, SecurityBlocked
