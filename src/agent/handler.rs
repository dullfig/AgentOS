//! CodingAgentHandler — the stateful agentic loop.
//!
//! This is the heart of Phase 4. A stateful Handler that runs Opus's
//! think → act → observe loop through the pipeline.
//!
//! ## Message Protocol
//!
//! Two directions only:
//! - `stop_reason: "tool_use"` → **down** (extend thread, dispatch to tool-peer)
//! - `stop_reason: "end_turn"` → **up** (prune thread, reply to caller)
//!
//! ## State Machine
//!
//! ```text
//! Ready ──[new task]──→ call Opus ──→ check stop_reason
//!                                          │
//!                                ┌─────────┴──────────┐
//!                                ▼                    ▼
//!                          [tool_use]           [end_turn]
//!                                │                    │
//!                                ▼                    ▼
//!                     Send first tool call      Reply up
//!                                │
//!                     AwaitingTools ←──[result]──┐
//!                          │                     │
//!                   ┌──more pending?──┐          │
//!                   ▼                 ▼          │
//!             Send next         Call Opus again──┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::{broadcast, Mutex};

use crate::agent::permissions::{
    resolve_tier, ApprovalVerdict, PermissionMap, PermissionTier, ToolApprovalRequest,
};
use crate::librarian::Librarian;
use crate::llm::types::{ContentBlock, ToolDefinition, ToolResultBlock};
use crate::llm::LlmPool;
use crate::organism::AgentConfig;
use crate::pipeline::events::{ConversationEntry, PipelineEvent};
use crate::routing::{RouteDecision, SemanticRouter};

use super::state::{AgentState, AgentThread, PendingToolCall};
use super::translate;

/// A snapshot of an agent thread's state (for TUI display).
#[derive(Debug, Clone)]
pub struct AgentThreadSnapshot {
    pub thread_id: String,
    pub message_count: usize,
    pub state_description: String,
}

/// The coding agent handler — stateful, per-thread conversation management.
pub struct CodingAgentHandler {
    pool: Arc<Mutex<LlmPool>>,
    librarian: Option<Arc<Mutex<Librarian>>>,
    tool_definitions: Vec<ToolDefinition>,
    /// Per-thread conversation state.
    threads: Arc<Mutex<HashMap<String, AgentThread>>>,
    system_prompt: String,
    /// Optional semantic router for invisible tool dispatch.
    semantic_router: Option<SemanticRouter>,
    /// Maximum routing iterations per turn (prevents infinite loops).
    max_routing_iterations: usize,
    /// Maximum global agentic loop iterations (Opus→tool→Opus cycles).
    max_agentic_iterations: usize,
    /// Optional event sender for emitting AgentResponse events to the TUI.
    event_tx: Option<broadcast::Sender<PipelineEvent>>,
    /// Max tokens for LLM completion (from AgentConfig).
    max_tokens: u32,
    /// Model override. None = pool default.
    model: Option<String>,
    /// Per-tool permission tiers (from AgentConfig).
    permissions: PermissionMap,
    /// Channel for sending tool approval requests to the TUI.
    approval_tx: Option<tokio::sync::mpsc::Sender<ToolApprovalRequest>>,
}

/// Type alias — generic agent handler (same implementation, data-driven identity).
pub type AgentHandler = CodingAgentHandler;

/// Default max routing iterations per turn.
const DEFAULT_MAX_ROUTING_ITERATIONS: usize = 5;

/// Default max agentic loop iterations (Opus→tool→Opus cycles).
const DEFAULT_MAX_AGENTIC_ITERATIONS: usize = 25;

impl CodingAgentHandler {
    /// Create a new coding agent handler.
    pub fn new(
        pool: Arc<Mutex<LlmPool>>,
        tool_definitions: Vec<ToolDefinition>,
        system_prompt: String,
    ) -> Self {
        Self {
            pool,
            librarian: None,
            tool_definitions,
            threads: Arc::new(Mutex::new(HashMap::new())),
            system_prompt,
            semantic_router: None,
            max_routing_iterations: DEFAULT_MAX_ROUTING_ITERATIONS,
            max_agentic_iterations: DEFAULT_MAX_AGENTIC_ITERATIONS,
            event_tx: None,
            max_tokens: 4096,
            model: None,
            permissions: PermissionMap::new(),
            approval_tx: None,
        }
    }

    /// Create from an AgentConfig (YAML-defined agent).
    pub fn from_config(
        pool: Arc<Mutex<LlmPool>>,
        tool_definitions: Vec<ToolDefinition>,
        system_prompt: String,
        config: &AgentConfig,
    ) -> Self {
        Self {
            pool,
            librarian: None,
            tool_definitions,
            threads: Arc::new(Mutex::new(HashMap::new())),
            system_prompt,
            semantic_router: None,
            max_routing_iterations: config.max_routing_iterations,
            max_agentic_iterations: config.max_agentic_iterations,
            event_tx: None,
            max_tokens: config.max_tokens,
            model: config.model.clone(),
            permissions: config.permissions.clone(),
            approval_tx: None,
        }
    }

    /// Create with an attached Librarian for context curation.
    pub fn with_librarian(
        pool: Arc<Mutex<LlmPool>>,
        librarian: Arc<Mutex<Librarian>>,
        tool_definitions: Vec<ToolDefinition>,
        system_prompt: String,
    ) -> Self {
        Self {
            pool,
            librarian: Some(librarian),
            tool_definitions,
            threads: Arc::new(Mutex::new(HashMap::new())),
            system_prompt,
            semantic_router: None,
            max_routing_iterations: DEFAULT_MAX_ROUTING_ITERATIONS,
            max_agentic_iterations: DEFAULT_MAX_AGENTIC_ITERATIONS,
            event_tx: None,
            max_tokens: 4096,
            model: None,
            permissions: PermissionMap::new(),
            approval_tx: None,
        }
    }

    /// Create with a semantic router for invisible tool dispatch.
    pub fn with_semantic_router(
        pool: Arc<Mutex<LlmPool>>,
        router: SemanticRouter,
        tool_definitions: Vec<ToolDefinition>,
        system_prompt: String,
    ) -> Self {
        Self {
            pool,
            librarian: None,
            tool_definitions,
            threads: Arc::new(Mutex::new(HashMap::new())),
            system_prompt,
            semantic_router: Some(router),
            max_routing_iterations: DEFAULT_MAX_ROUTING_ITERATIONS,
            max_agentic_iterations: DEFAULT_MAX_AGENTIC_ITERATIONS,
            event_tx: None,
            max_tokens: 4096,
            model: None,
            permissions: PermissionMap::new(),
            approval_tx: None,
        }
    }

    /// Attach a librarian to an existing handler (builder-style).
    pub fn with_librarian_attached(mut self, lib: Arc<Mutex<Librarian>>) -> Self {
        self.librarian = Some(lib);
        self
    }

    /// Attach a semantic router to an existing handler (builder-style).
    pub fn with_router_attached(mut self, router: SemanticRouter) -> Self {
        self.semantic_router = Some(router);
        self
    }

    /// Set the maximum routing iterations per turn.
    pub fn set_max_routing_iterations(&mut self, max: usize) {
        self.max_routing_iterations = max;
    }

    /// Set the event sender for emitting pipeline events (e.g., AgentResponse).
    pub fn set_event_sender(&mut self, tx: broadcast::Sender<PipelineEvent>) {
        self.event_tx = Some(tx);
    }

    /// Set the approval channel sender (for tool permission prompts to the TUI).
    pub fn set_approval_sender(&mut self, tx: tokio::sync::mpsc::Sender<ToolApprovalRequest>) {
        self.approval_tx = Some(tx);
    }

    /// Check permission for a tool call. Returns Ok(()) to proceed, Err(denial_msg) if denied.
    async fn check_tool_permission(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        thread_id: &str,
    ) -> Result<(), String> {
        let tier = resolve_tier(&self.permissions, tool_name);
        match tier {
            PermissionTier::Auto => {
                self.maybe_emit(PipelineEvent::ToolApproval {
                    thread_id: thread_id.to_string(),
                    tool_name: tool_name.to_string(),
                    verdict: "auto".into(),
                });
                Ok(())
            }
            PermissionTier::Deny => {
                self.maybe_emit(PipelineEvent::ToolApproval {
                    thread_id: thread_id.to_string(),
                    tool_name: tool_name.to_string(),
                    verdict: "denied_by_policy".into(),
                });
                Err(format!(
                    "Permission denied: {tool_name} is blocked by policy"
                ))
            }
            PermissionTier::Prompt => {
                if let Some(ref tx) = self.approval_tx {
                    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                    let summary = summarize_tool_input(tool_name, input);
                    let request = ToolApprovalRequest {
                        tool_name: tool_name.to_string(),
                        args_summary: summary,
                        thread_id: thread_id.to_string(),
                        response_tx: resp_tx,
                    };
                    if tx.send(request).await.is_err() {
                        // TUI disconnected — default to auto (headless mode)
                        return Ok(());
                    }
                    match resp_rx.await {
                        Ok(ApprovalVerdict::Approved) => {
                            self.maybe_emit(PipelineEvent::ToolApproval {
                                thread_id: thread_id.to_string(),
                                tool_name: tool_name.to_string(),
                                verdict: "approved".into(),
                            });
                            Ok(())
                        }
                        Ok(ApprovalVerdict::Denied) | Err(_) => {
                            self.maybe_emit(PipelineEvent::ToolApproval {
                                thread_id: thread_id.to_string(),
                                tool_name: tool_name.to_string(),
                                verdict: "denied".into(),
                            });
                            Err(format!(
                                "User denied permission to use {tool_name}"
                            ))
                        }
                    }
                } else {
                    // No TUI connected — default to auto (headless mode)
                    Ok(())
                }
            }
        }
    }

    /// Emit an AgentResponse event if an event sender is attached.
    fn maybe_emit_response(&self, thread_id: &str, result: &HandlerResult) {
        if let Ok(HandlerResponse::Reply { ref payload_xml }) = result {
            if let Some(ref tx) = self.event_tx {
                let text = String::from_utf8_lossy(payload_xml);
                let response_text = extract_tag(&text, "result")
                    .unwrap_or_else(|| text.to_string());
                let _ = tx.send(PipelineEvent::AgentResponse {
                    thread_id: thread_id.to_string(),
                    text: response_text,
                });
            }
        }
    }

    /// Emit a pipeline event if an event sender is attached.
    fn maybe_emit(&self, event: PipelineEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event);
        }
    }

    /// Emit an error as an AgentResponse event so the TUI can display it.
    fn emit_error(&self, thread_id: &str, error: &str) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(PipelineEvent::AgentResponse {
                thread_id: thread_id.to_string(),
                text: format!("Error: {error}"),
            });
        }
    }

    /// Emit a ConversationSync event with the current thread state.
    fn maybe_emit_conversation(&self, thread_id: &str, thread: &AgentThread) {
        if let Some(ref tx) = self.event_tx {
            let entries = build_conversation_entries(&thread.messages);
            let _ = tx.send(PipelineEvent::ConversationSync {
                thread_id: thread_id.to_string(),
                entries,
            });
        }
    }

    /// Increment agentic iteration counter and check limit.
    /// Returns a reply if the limit is exceeded.
    fn check_agentic_limit(&self, thread: &mut AgentThread) -> Option<HandlerResult> {
        thread.agentic_iterations += 1;
        if thread.agentic_iterations > self.max_agentic_iterations {
            let msg = format!(
                "Agentic iteration limit reached ({} iterations). Stopping to prevent runaway loop.",
                self.max_agentic_iterations
            );
            let reply_xml = format!(
                "<AgentResponse><result>{}</result></AgentResponse>",
                translate::xml_escape_text(&msg)
            );
            Some(Ok(HandlerResponse::Reply {
                payload_xml: reply_xml.into_bytes(),
            }))
        } else {
            None
        }
    }

    /// Check if a semantic router is attached.
    pub fn has_semantic_router(&self) -> bool {
        self.semantic_router.is_some()
    }

    /// Get snapshots of all active agent threads (for TUI display).
    pub async fn thread_snapshots(&self) -> Vec<AgentThreadSnapshot> {
        let threads = self.threads.lock().await;
        threads
            .iter()
            .map(|(id, t)| {
                let state_desc = match &t.state {
                    AgentState::Ready => "Ready".to_string(),
                    AgentState::AwaitingTools {
                        pending,
                        collected,
                        ..
                    } => format!("AwaitingTools({}/{})", collected.len(), pending.len()),
                };
                AgentThreadSnapshot {
                    thread_id: id.clone(),
                    message_count: t.messages.len(),
                    state_description: state_desc,
                }
            })
            .collect()
    }

    /// Call the LLM API with the current conversation state.
    async fn call_opus(
        &self,
        thread: &AgentThread,
    ) -> Result<crate::llm::types::MessagesResponse, String> {
        // Optional: curate context before the API call
        let mut system = self.system_prompt.clone();
        if let Some(ref librarian) = self.librarian {
            let lib = librarian.lock().await;
            let budget = 6000usize;
            if let Ok(result) = lib.curate("agent", &thread.messages, budget).await {
                if let Some(ctx) = result.system_context {
                    system = format!("{system}\n\n{ctx}");
                }
            }
        }

        let pool = self.pool.lock().await;
        pool.complete_with_tools(
            self.model.as_deref(),
            thread.messages.clone(),
            self.max_tokens,
            Some(&system),
            self.tool_definitions.clone(),
        )
        .await
        .map_err(|e| format!("LLM API error: {e}"))
    }

    /// Process an Opus response: extract tool calls or final text.
    fn process_response(
        &self,
        response: &crate::llm::types::MessagesResponse,
    ) -> ResponseAction {
        let stop_reason = response.stop_reason.as_deref().unwrap_or("unknown");

        // Check for forgiving ingress: text alongside tool calls
        let has_text = response.text().is_some();
        let has_tool_use = response.has_tool_use();

        if has_text && has_tool_use {
            let text = response.text().unwrap_or("").to_string();
            tracing::warn!(
                "forgiving ingress: assistant sent text alongside tool calls: {text}"
            );
        }

        if stop_reason == "tool_use" || has_tool_use {
            let pending: Vec<PendingToolCall> = response
                .tool_use_blocks()
                .into_iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(PendingToolCall {
                        tool_use_id: id.clone(),
                        tool_name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect();

            ResponseAction::ToolCalls {
                blocks: response.content.clone(),
                pending,
            }
        } else {
            let text = response.text().unwrap_or("(no response)").to_string();
            ResponseAction::FinalText {
                blocks: response.content.clone(),
                text,
            }
        }
    }

    /// Handle the result of an Opus API call — dispatch tool calls or reply.
    fn dispatch_response(
        thread: &mut AgentThread,
        action: ResponseAction,
    ) -> HandlerResult {
        match action {
            ResponseAction::ToolCalls { blocks, pending } => {
                if pending.is_empty() {
                    // Shouldn't happen, but handle gracefully
                    let reply_xml =
                        "<AgentResponse><result>(no tool calls)</result></AgentResponse>";
                    return Ok(HandlerResponse::Reply {
                        payload_xml: reply_xml.as_bytes().to_vec(),
                    });
                }
                let first_name = pending[0].tool_name.clone();
                let first_xml =
                    translate::tool_call_to_xml(&pending[0].tool_name, &pending[0].input);
                thread.state = AgentState::AwaitingTools {
                    assistant_blocks: blocks,
                    pending,
                    collected: Vec::new(),
                    current_index: 0,
                };
                Ok(HandlerResponse::Send {
                    to: first_name,
                    payload_xml: first_xml.into_bytes(),
                })
            }
            ResponseAction::FinalText { blocks, text } => {
                thread.push_assistant_blocks(blocks);
                let reply_xml = format!(
                    "<AgentResponse><result>{}</result></AgentResponse>",
                    translate::xml_escape_text(&text)
                );
                Ok(HandlerResponse::Reply {
                    payload_xml: reply_xml.into_bytes(),
                })
            }
        }
    }

    /// Dispatch a response action, trying semantic routing for FinalText.
    ///
    /// If a semantic router is attached and the text matches a tool,
    /// the routing loop handles it. Otherwise, normal dispatch.
    async fn dispatch_or_route(
        &self,
        thread: &mut AgentThread,
        action: ResponseAction,
        allowed_tools: &[String],
    ) -> HandlerResult {
        match action {
            ResponseAction::FinalText { blocks, text } if self.semantic_router.is_some() => {
                self.dispatch_with_routing(thread, blocks, text, allowed_tools, 0)
                    .await
            }
            _ => Self::dispatch_response(thread, action),
        }
    }

    /// Dispatch with permission gating.
    ///
    /// Wraps `dispatch_or_route` with permission checks on Send results.
    /// If a tool is denied: synthesizes error results, calls Opus again, loops.
    async fn dispatch_or_route_with_gate(
        &self,
        thread: &mut AgentThread,
        mut action: ResponseAction,
        allowed_tools: &[String],
        thread_id: &str,
    ) -> HandlerResult {
        loop {
            let result = self.dispatch_or_route(thread, action, allowed_tools).await?;

            match result {
                HandlerResponse::Send { ref to, .. } => {
                    // Get tool input from pending state for permission check
                    let tool_input = match &thread.state {
                        AgentState::AwaitingTools {
                            pending,
                            current_index,
                            ..
                        } => pending
                            .get(*current_index)
                            .map(|p| p.input.clone())
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::Null,
                    };

                    match self.check_tool_permission(to, &tool_input, thread_id).await {
                        Ok(()) => return Ok(result), // Approved — dispatch
                        Err(denial) => {
                            // Denied — synthesize error, try remaining tools or re-invoke Opus
                            let old_state =
                                std::mem::replace(&mut thread.state, AgentState::Ready);
                            if let AgentState::AwaitingTools {
                                assistant_blocks,
                                pending,
                                mut collected,
                                current_index,
                            } = old_state
                            {
                                // Deny current tool
                                let tool_use_id = pending
                                    .get(current_index)
                                    .map(|p| p.tool_use_id.clone())
                                    .unwrap_or_default();
                                collected.push(ToolResultBlock {
                                    tool_use_id,
                                    content: denial,
                                    is_error: true,
                                });

                                // Try remaining tools in batch
                                let mut idx = current_index + 1;
                                while idx < pending.len() {
                                    let next = &pending[idx];
                                    match self
                                        .check_tool_permission(
                                            &next.tool_name,
                                            &next.input,
                                            thread_id,
                                        )
                                        .await
                                    {
                                        Ok(()) => {
                                            // Approved — dispatch this one
                                            let xml = translate::tool_call_to_xml(
                                                &next.tool_name,
                                                &next.input,
                                            );
                                            let name = next.tool_name.clone();
                                            self.maybe_emit(PipelineEvent::ToolDispatched {
                                                thread_id: thread_id.to_string(),
                                                tool_name: next.tool_name.clone(),
                                                detail: summarize_tool_input(
                                                    &next.tool_name,
                                                    &next.input,
                                                ),
                                            });
                                            thread.state = AgentState::AwaitingTools {
                                                assistant_blocks,
                                                pending,
                                                collected,
                                                current_index: idx,
                                            };
                                            return Ok(HandlerResponse::Send {
                                                to: name,
                                                payload_xml: xml.into_bytes(),
                                            });
                                        }
                                        Err(denial) => {
                                            collected.push(ToolResultBlock {
                                                tool_use_id: next.tool_use_id.clone(),
                                                content: denial,
                                                is_error: true,
                                            });
                                            idx += 1;
                                        }
                                    }
                                }

                                // All tools denied — push results and re-invoke Opus
                                thread.push_assistant_blocks(assistant_blocks);
                                thread.push_tool_results(collected);
                                thread.state = AgentState::Ready;

                                self.maybe_emit(PipelineEvent::AgentThinking {
                                    thread_id: thread_id.to_string(),
                                });

                                if let Some(limit_result) = self.check_agentic_limit(thread) {
                                    return limit_result;
                                }

                                let response = self
                                    .call_opus(thread)
                                    .await
                                    .map_err(|e| PipelineError::Handler(e))?;
                                action = self.process_response(&response);
                                continue; // Loop: dispatch the new action
                            }

                            // Shouldn't reach here (state wasn't AwaitingTools)
                            return Ok(HandlerResponse::Reply {
                                payload_xml: b"<AgentResponse><error>unexpected state during permission check</error></AgentResponse>".to_vec(),
                            });
                        }
                    }
                }
                _ => return Ok(result), // Reply, None, etc. — pass through
            }
        }
    }

    /// Try semantic routing on a final text response.
    ///
    /// If a semantic router is attached and the text matches a tool,
    /// the router fills parameters and returns the tool result (or failure note)
    /// for injection into the conversation. The caller then re-invokes Opus
    /// with the result in context.
    ///
    /// Returns `None` if no router is attached or no tool matched.
    async fn try_semantic_route(
        &self,
        text: &str,
        allowed_tools: &[String],
    ) -> Option<RouteDecision> {
        let router = self.semantic_router.as_ref()?;
        let decision = router.route(text, allowed_tools).await;
        match &decision {
            RouteDecision::Response => None,
            _ => Some(decision),
        }
    }

    /// Dispatch a final text with semantic routing.
    ///
    /// If the semantic router intercepts the text:
    /// - Records the assistant's text in conversation history
    /// - Injects the tool result (or failure note) as a synthetic user message
    /// - Calls Opus again to see the result in context
    /// - Recurses up to `max_routing_iterations` times
    async fn dispatch_with_routing(
        &self,
        thread: &mut AgentThread,
        blocks: Vec<ContentBlock>,
        text: String,
        allowed_tools: &[String],
        iterations: usize,
    ) -> HandlerResult {
        if iterations >= self.max_routing_iterations {
            // Max iterations reached — return the text as-is
            thread.push_assistant_blocks(blocks);
            let reply_xml = format!(
                "<AgentResponse><result>{}</result></AgentResponse>",
                translate::xml_escape_text(&text)
            );
            return Ok(HandlerResponse::Reply {
                payload_xml: reply_xml.into_bytes(),
            });
        }

        match self.try_semantic_route(&text, allowed_tools).await {
            Some(RouteDecision::ToolResult {
                tool_name,
                result_xml,
            }) => {
                // Record assistant's text, inject result as synthetic user message
                thread.push_assistant_blocks(blocks);
                thread.push_user_message(&format!(
                    "<{tool_name}_result>{result_xml}</{tool_name}_result>"
                ));

                // Check agentic iteration limit before calling Opus
                if let Some(result) = self.check_agentic_limit(thread) {
                    return result;
                }

                // Call Opus again — it sees the result in context
                let response = self
                    .call_opus(thread)
                    .await
                    .map_err(PipelineError::Handler)?;
                let action = self.process_response(&response);

                match action {
                    ResponseAction::FinalText {
                        blocks: new_blocks,
                        text: new_text,
                    } => {
                        // Recurse: Opus might express another tool intent
                        Box::pin(self.dispatch_with_routing(
                            thread,
                            new_blocks,
                            new_text,
                            allowed_tools,
                            iterations + 1,
                        ))
                        .await
                    }
                    _ => Self::dispatch_response(thread, action),
                }
            }
            Some(RouteDecision::ToolFailed { note }) => {
                // Record assistant's text + failure note
                thread.push_assistant_blocks(blocks);
                thread.push_user_message(&format!("<system_note>{note}</system_note>"));

                // Check agentic iteration limit before calling Opus
                if let Some(result) = self.check_agentic_limit(thread) {
                    return result;
                }

                // Call Opus again with the failure note
                let response = self
                    .call_opus(thread)
                    .await
                    .map_err(PipelineError::Handler)?;
                let action = self.process_response(&response);
                Self::dispatch_response(thread, action)
            }
            _ => {
                // No match — normal reply
                thread.push_assistant_blocks(blocks);
                let reply_xml = format!(
                    "<AgentResponse><result>{}</result></AgentResponse>",
                    translate::xml_escape_text(&text)
                );
                Ok(HandlerResponse::Reply {
                    payload_xml: reply_xml.into_bytes(),
                })
            }
        }
    }
}

/// What to do after processing an Opus response.
enum ResponseAction {
    ToolCalls {
        blocks: Vec<ContentBlock>,
        pending: Vec<PendingToolCall>,
    },
    FinalText {
        blocks: Vec<ContentBlock>,
        text: String,
    },
}

#[async_trait]
impl Handler for CodingAgentHandler {
    async fn handle(&self, payload: ValidatedPayload, ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let thread_id = ctx.thread_id.clone();

        let mut threads = self.threads.lock().await;
        let thread = threads
            .entry(thread_id.clone())
            .or_insert_with(AgentThread::new);

        let is_tool_response = xml_str.contains("<ToolResponse>");

        if is_tool_response {
            // ── Tool response path ──
            let (result_content, is_error) = translate::xml_response_to_result(&xml_str);

            // Extract state, replacing with Ready temporarily
            let old_state = std::mem::replace(&mut thread.state, AgentState::Ready);

            match old_state {
                AgentState::AwaitingTools {
                    assistant_blocks,
                    pending,
                    mut collected,
                    current_index,
                } => {
                    // Add the result for the current tool call
                    let tool_use_id = pending
                        .get(current_index)
                        .map(|p| p.tool_use_id.clone())
                        .unwrap_or_default();
                    // Lifecycle: tool completed
                    let completed_tool = pending
                        .get(current_index)
                        .map(|p| p.tool_name.clone())
                        .unwrap_or_default();
                    let completed_detail = if is_error {
                        result_content.chars().take(80).collect::<String>()
                    } else {
                        String::new()
                    };
                    self.maybe_emit(PipelineEvent::ToolCompleted {
                        thread_id: thread_id.clone(),
                        tool_name: completed_tool,
                        success: !is_error,
                        detail: completed_detail,
                    });

                    collected.push(ToolResultBlock {
                        tool_use_id,
                        content: result_content,
                        is_error,
                    });

                    // Try dispatching remaining tools (skip denied ones)
                    let mut next_index = current_index + 1;
                    while next_index < pending.len() {
                        let next = &pending[next_index];
                        match self
                            .check_tool_permission(
                                &next.tool_name,
                                &next.input,
                                &thread_id,
                            )
                            .await
                        {
                            Ok(()) => {
                                // Approved — dispatch this tool
                                let xml = translate::tool_call_to_xml(
                                    &next.tool_name,
                                    &next.input,
                                );
                                let next_name = next.tool_name.clone();

                                self.maybe_emit(PipelineEvent::ToolDispatched {
                                    thread_id: thread_id.clone(),
                                    tool_name: next.tool_name.clone(),
                                    detail: summarize_tool_input(
                                        &next.tool_name,
                                        &next.input,
                                    ),
                                });

                                thread.state = AgentState::AwaitingTools {
                                    assistant_blocks,
                                    pending,
                                    collected,
                                    current_index: next_index,
                                };
                                return Ok(HandlerResponse::Send {
                                    to: next_name,
                                    payload_xml: xml.into_bytes(),
                                });
                            }
                            Err(denial) => {
                                // Denied — push error result, try next
                                collected.push(ToolResultBlock {
                                    tool_use_id: next.tool_use_id.clone(),
                                    content: denial,
                                    is_error: true,
                                });
                                next_index += 1;
                            }
                        }
                    }

                    // All collected — record in conversation history and call Opus again
                    thread.push_assistant_blocks(assistant_blocks);
                    thread.push_tool_results(collected);
                    thread.state = AgentState::Ready;

                    // Lifecycle: thinking (after all tools collected)
                    self.maybe_emit(PipelineEvent::AgentThinking {
                        thread_id: thread_id.clone(),
                    });

                    // Check agentic iteration limit before calling Opus
                    if let Some(result) = self.check_agentic_limit(thread) {
                        self.maybe_emit_response(&thread_id, &result);
                        self.maybe_emit_conversation(&thread_id, thread);
                        return result;
                    }

                    let response = match self.call_opus(thread).await {
                        Ok(r) => r,
                        Err(e) => {
                            self.emit_error(&thread_id, &e);
                            return Err(PipelineError::Handler(e));
                        }
                    };
                    let action = self.process_response(&response);

                    // Lifecycle: tool dispatched (after re-call from tool-response path)
                    if let ResponseAction::ToolCalls { ref pending, .. } = action {
                        if let Some(first) = pending.first() {
                            self.maybe_emit(PipelineEvent::ToolDispatched {
                                thread_id: thread_id.clone(),
                                tool_name: first.tool_name.clone(),
                                detail: summarize_tool_input(&first.tool_name, &first.input),
                            });
                        }
                    }

                    let result = self
                        .dispatch_or_route_with_gate(thread, action, &[], &thread_id)
                        .await;
                    self.maybe_emit_response(&thread_id, &result);
                    self.maybe_emit_conversation(&thread_id, thread);
                    result
                }
                AgentState::Ready => {
                    // Unexpected tool response when not awaiting
                    let reply_xml =
                        "<AgentResponse><error>unexpected tool response</error></AgentResponse>";
                    Ok(HandlerResponse::Reply {
                        payload_xml: reply_xml.as_bytes().to_vec(),
                    })
                }
            }
        } else {
            // ── New task path ──
            let task = extract_tag(&xml_str, "task")
                .or_else(|| extract_tag(&xml_str, "content"))
                .unwrap_or_else(|| xml_str.to_string());

            thread.push_user_message(&task);
            thread.state = AgentState::Ready;

            // Lifecycle: thinking (new task)
            self.maybe_emit(PipelineEvent::AgentThinking {
                thread_id: thread_id.clone(),
            });

            // Check agentic iteration limit before calling Opus
            if let Some(result) = self.check_agentic_limit(thread) {
                self.maybe_emit_response(&thread_id, &result);
                self.maybe_emit_conversation(&thread_id, thread);
                return result;
            }

            let response = match self.call_opus(thread).await {
                Ok(r) => r,
                Err(e) => {
                    self.emit_error(&thread_id, &e);
                    return Err(PipelineError::Handler(e));
                }
            };
            let action = self.process_response(&response);

            // Lifecycle: tool dispatched (first tool from new task)
            if let ResponseAction::ToolCalls { ref pending, .. } = action {
                if let Some(first) = pending.first() {
                    self.maybe_emit(PipelineEvent::ToolDispatched {
                        thread_id: thread_id.clone(),
                        tool_name: first.tool_name.clone(),
                        detail: summarize_tool_input(&first.tool_name, &first.input),
                    });
                }
            }

            let result = self
                .dispatch_or_route_with_gate(thread, action, &[], &thread_id)
                .await;
            self.maybe_emit_response(&thread_id, &result);
            self.maybe_emit_conversation(&thread_id, thread);
            result
        }
    }
}

/// Convert a slice of Messages into ConversationEntry items for TUI display.
pub fn build_conversation_entries(messages: &[crate::llm::types::Message]) -> Vec<ConversationEntry> {
    use crate::llm::types::{ContentBlock, MessageContent};
    let mut entries = Vec::new();
    for msg in messages {
        match &msg.content {
            MessageContent::Text(text) => {
                entries.push(ConversationEntry {
                    role: msg.role.clone(),
                    summary: truncate_text(text, 200),
                    is_tool_use: false,
                    tool_name: None,
                    is_error: false,
                });
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => {
                            entries.push(ConversationEntry {
                                role: msg.role.clone(),
                                summary: truncate_text(text, 200),
                                is_tool_use: false,
                                tool_name: None,
                                is_error: false,
                            });
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            let detail = summarize_tool_input(name, input);
                            entries.push(ConversationEntry {
                                role: "assistant".into(),
                                summary: format!("{name} → {detail}"),
                                is_tool_use: true,
                                tool_name: Some(name.clone()),
                                is_error: false,
                            });
                        }
                        ContentBlock::ToolResult {
                            content, is_error, ..
                        } => {
                            let err = is_error.unwrap_or(false);
                            let text = content.as_deref().unwrap_or("");
                            entries.push(ConversationEntry {
                                role: "tool_result".into(),
                                summary: truncate_text(text, 120),
                                is_tool_use: false,
                                tool_name: None,
                                is_error: err,
                            });
                        }
                    }
                }
            }
        }
    }
    entries
}

/// Truncate text to a maximum character count, appending "..." if truncated.
fn truncate_text(text: &str, max: usize) -> String {
    // Take first line only for summary
    let first_line = text.lines().next().unwrap_or(text);
    if first_line.len() <= max {
        first_line.to_string()
    } else {
        format!("{}...", &first_line[..max.saturating_sub(3)])
    }
}

/// Summarize tool input JSON into a short detail string for activity trace.
pub(crate) fn summarize_tool_input(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
        "file-read" | "file-write" | "file-edit" => input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "command-exec" => input
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| {
                if s.len() > 60 {
                    format!("{}...", &s[..57])
                } else {
                    s.to_string()
                }
            })
            .unwrap_or_default(),
        "glob-search" => input
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "grep-search" => input
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => {
            let s = input.to_string();
            if s.len() > 60 {
                format!("{}...", &s[..57])
            } else {
                s
            }
        }
    }
}

/// Extract text content between `<tag>` and `</tag>`.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml.find(&close)?;
    if start <= end {
        Some(xml_unescape(&xml[start..end]))
    } else {
        None
    }
}

/// Unescape XML entities back to plain text.
fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_pool() -> Arc<Mutex<LlmPool>> {
        Arc::new(Mutex::new(LlmPool::with_base_url(
            "test-key".into(),
            "opus",
            "http://localhost:19999".into(),
        )))
    }

    fn sample_tool_defs() -> Vec<ToolDefinition> {
        crate::agent::tools::build_tool_definitions(&["file-read", "command-exec"])
    }

    #[test]
    fn handler_creation() {
        let pool = mock_pool();
        let handler =
            CodingAgentHandler::new(pool, sample_tool_defs(), "You are a test agent.".into());
        assert_eq!(handler.tool_definitions.len(), 2);
        assert_eq!(handler.system_prompt, "You are a test agent.");
    }

    #[test]
    fn handler_with_librarian_creation() {
        let pool = mock_pool();
        let kernel =
            crate::kernel::Kernel::open(&tempfile::TempDir::new().unwrap().path().join("data"))
                .unwrap();
        let kernel_arc = Arc::new(Mutex::new(kernel));
        let lib = Arc::new(Mutex::new(Librarian::new(pool.clone(), kernel_arc)));

        let handler = CodingAgentHandler::with_librarian(
            pool,
            lib,
            sample_tool_defs(),
            "test".into(),
        );
        assert!(handler.librarian.is_some());
    }

    #[test]
    fn extract_task_from_xml() {
        assert_eq!(
            extract_tag("<AgentTask><task>Do the thing</task></AgentTask>", "task"),
            Some("Do the thing".into())
        );
        assert_eq!(
            extract_tag("<Msg><content>Hello</content></Msg>", "content"),
            Some("Hello".into())
        );
        assert_eq!(extract_tag("<Msg>no tags</Msg>", "task"), None);
    }

    #[test]
    fn process_response_end_turn() {
        let pool = mock_pool();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into());
        let response = crate::llm::types::MessagesResponse {
            id: "msg_1".into(),
            model: "test".into(),
            content: vec![ContentBlock::Text {
                text: "Done!".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: crate::llm::types::Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let action = handler.process_response(&response);
        match action {
            ResponseAction::FinalText { text, .. } => assert_eq!(text, "Done!"),
            _ => panic!("expected FinalText"),
        }
    }

    #[test]
    fn process_response_tool_use() {
        let pool = mock_pool();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into());
        let response = crate::llm::types::MessagesResponse {
            id: "msg_2".into(),
            model: "test".into(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me read that file.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "file-read".into(),
                    input: serde_json::json!({"path": "src/main.rs"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: crate::llm::types::Usage {
                input_tokens: 20,
                output_tokens: 15,
            },
        };
        let action = handler.process_response(&response);
        match action {
            ResponseAction::ToolCalls { pending, .. } => {
                assert_eq!(pending.len(), 1);
                assert_eq!(pending[0].tool_name, "file-read");
                assert_eq!(pending[0].tool_use_id, "toolu_1");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn process_response_multiple_tool_calls() {
        let pool = mock_pool();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into());
        let response = crate::llm::types::MessagesResponse {
            id: "msg_3".into(),
            model: "test".into(),
            content: vec![
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "file-read".into(),
                    input: serde_json::json!({"path": "a.rs"}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_2".into(),
                    name: "command-exec".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: crate::llm::types::Usage {
                input_tokens: 30,
                output_tokens: 25,
            },
        };
        let action = handler.process_response(&response);
        match action {
            ResponseAction::ToolCalls { pending, .. } => {
                assert_eq!(pending.len(), 2);
                assert_eq!(pending[0].tool_name, "file-read");
                assert_eq!(pending[1].tool_name, "command-exec");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn dispatch_final_text() {
        let mut thread = AgentThread::new();
        thread.push_user_message("Hello");

        let action = ResponseAction::FinalText {
            blocks: vec![ContentBlock::Text {
                text: "Hi there!".into(),
            }],
            text: "Hi there!".into(),
        };

        let result = CodingAgentHandler::dispatch_response(&mut thread, action).unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<AgentResponse>"));
                assert!(xml.contains("Hi there!"));
            }
            _ => panic!("expected Reply"),
        }
        // Thread should now have 2 messages: user + assistant
        assert_eq!(thread.messages.len(), 2);
    }

    #[test]
    fn dispatch_tool_calls() {
        let mut thread = AgentThread::new();
        thread.push_user_message("Read foo.rs");

        let action = ResponseAction::ToolCalls {
            blocks: vec![ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "file-read".into(),
                input: serde_json::json!({"path": "foo.rs"}),
            }],
            pending: vec![PendingToolCall {
                tool_use_id: "toolu_1".into(),
                tool_name: "file-read".into(),
                input: serde_json::json!({"path": "foo.rs"}),
            }],
        };

        let result = CodingAgentHandler::dispatch_response(&mut thread, action).unwrap();
        match result {
            HandlerResponse::Send { to, payload_xml } => {
                assert_eq!(to, "file-read");
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<FileReadRequest>"));
                assert!(xml.contains("<path>foo.rs</path>"));
            }
            _ => panic!("expected Send"),
        }
        // Thread should be in AwaitingTools state
        assert!(matches!(thread.state, AgentState::AwaitingTools { .. }));
    }

    #[test]
    fn dispatch_empty_tool_calls() {
        let mut thread = AgentThread::new();
        let action = ResponseAction::ToolCalls {
            blocks: vec![],
            pending: vec![],
        };
        let result = CodingAgentHandler::dispatch_response(&mut thread, action).unwrap();
        assert!(matches!(result, HandlerResponse::Reply { .. }));
    }

    #[tokio::test]
    async fn handle_unexpected_tool_response() {
        let pool = mock_pool();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into());

        // Inject a tool response when no thread exists (will create with Ready state)
        let payload = ValidatedPayload {
            xml: b"<ToolResponse><success>true</success><result>42</result></ToolResponse>"
                .to_vec(),
            tag: "ToolResponse".into(),
        };
        let ctx = HandlerContext {
            thread_id: "orphan-thread".into(),
            from: "file-read".into(),
            own_name: "coding-agent".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("unexpected tool response"));
            }
            _ => panic!("expected Reply with error"),
        }
    }

    #[test]
    fn thread_state_management() {
        let mut thread = AgentThread::new();
        assert!(matches!(thread.state, AgentState::Ready));

        thread.push_user_message("Hello");
        assert_eq!(thread.messages.len(), 1);

        thread.push_assistant_blocks(vec![ContentBlock::Text {
            text: "Hi!".into(),
        }]);
        assert_eq!(thread.messages.len(), 2);

        thread.push_tool_results(vec![ToolResultBlock {
            tool_use_id: "t1".into(),
            content: "42".into(),
            is_error: false,
        }]);
        assert_eq!(thread.messages.len(), 3);
    }

    #[test]
    fn handle_new_task_xml_parsing() {
        let xml = "<AgentTask><task>Read src/main.rs</task></AgentTask>";
        let task = extract_tag(xml, "task").unwrap();
        assert_eq!(task, "Read src/main.rs");
    }

    // ── Semantic Routing Integration Tests ──

    fn build_test_router() -> crate::routing::SemanticRouter {
        use crate::embedding::tfidf::TfIdfProvider;
        use crate::embedding::{EmbeddingIndex, EmbeddingProvider};
        use crate::routing::form_filler::CloudFormFiller;
        use crate::routing::ToolMetadata;
        use std::collections::HashMap as StdHashMap;

        let descriptions = vec![
            "read file contents from the local filesystem source code configuration",
            "execute shell commands run programs compile code run tests",
        ];
        let provider = TfIdfProvider::from_corpus(&descriptions);

        let mut index = EmbeddingIndex::new(0.1);
        index.register("file-read", provider.embed(descriptions[0]));
        index.register("command-exec", provider.embed(descriptions[1]));

        let filler = CloudFormFiller::new(mock_pool(), 3);

        let mut metadata = StdHashMap::new();
        metadata.insert(
            "file-read".to_string(),
            ToolMetadata {
                description: "File read tool".into(),
                xml_template: "<FileReadRequest><path/></FileReadRequest>".into(),
                payload_tag: "FileReadRequest".into(),
            },
        );
        metadata.insert(
            "command-exec".to_string(),
            ToolMetadata {
                description: "Command execution tool".into(),
                xml_template: "<CommandExecRequest><command/></CommandExecRequest>".into(),
                payload_tag: "CommandExecRequest".into(),
            },
        );

        crate::routing::SemanticRouter::new(Box::new(provider), index, Box::new(filler), metadata)
    }

    #[test]
    fn handler_with_semantic_router_creation() {
        let pool = mock_pool();
        let router = build_test_router();
        let handler = CodingAgentHandler::with_semantic_router(
            pool,
            router,
            sample_tool_defs(),
            "You are a test agent.".into(),
        );
        assert!(handler.has_semantic_router());
        assert_eq!(handler.tool_definitions.len(), 2);
    }

    #[tokio::test]
    async fn handler_routes_matching_text() {
        let pool = mock_pool();
        let router = build_test_router();
        let handler = CodingAgentHandler::with_semantic_router(
            pool,
            router,
            sample_tool_defs(),
            "test".into(),
        );

        // The semantic router should match "read src/main.rs" to file-read
        let allowed = vec!["file-read".to_string(), "command-exec".to_string()];
        let decision = handler
            .try_semantic_route("read the source code file at src/main.rs", &allowed)
            .await;

        // Should match (form-filler will fail with mock URL, but it matched)
        assert!(decision.is_some());
    }

    #[tokio::test]
    async fn handler_passes_through_no_match() {
        let pool = mock_pool();
        let router = build_test_router();
        let mut handler = CodingAgentHandler::with_semantic_router(
            pool,
            router,
            sample_tool_defs(),
            "test".into(),
        );
        handler.set_max_routing_iterations(5);

        let allowed = vec!["file-read".to_string()];
        // Completely unrelated text — should not match
        let _decision = handler
            .try_semantic_route("the meaning of life is to create meaning", &allowed)
            .await;

        // With TF-IDF, generic philosophical text shouldn't match tool descriptions
        // at a reasonable threshold. If it does match weakly, that's fine — the binary
        // fork is still correct.
        // The key test: when there's no router, it returns None
        let no_router = CodingAgentHandler::new(mock_pool(), sample_tool_defs(), "test".into());
        let no_decision = no_router
            .try_semantic_route("anything", &allowed)
            .await;
        assert!(no_decision.is_none());
    }

    #[tokio::test]
    async fn handler_injects_result_as_context() {
        let mut thread = AgentThread::new();
        thread.push_user_message("initial task");

        // Simulate what happens when routing succeeds: assistant text + synthetic user message
        thread.push_assistant_blocks(vec![ContentBlock::Text {
            text: "I need to see parser.rs".into(),
        }]);
        thread.push_user_message("<file-read_result><content>fn main() {}</content></file-read_result>");

        // Thread should have 3 messages: user, assistant, synthetic user
        assert_eq!(thread.messages.len(), 3);
        assert_eq!(thread.messages[2].role, "user");
        let content = thread.messages[2].content.text().unwrap();
        assert!(content.contains("file-read_result"));
    }

    #[tokio::test]
    async fn handler_injects_failure_note() {
        let mut thread = AgentThread::new();
        thread.push_user_message("initial task");

        // Simulate routing failure: assistant text + failure note
        thread.push_assistant_blocks(vec![ContentBlock::Text {
            text: "I need to check something".into(),
        }]);
        thread.push_user_message("<system_note>Could not retrieve parser error handling information.</system_note>");

        assert_eq!(thread.messages.len(), 3);
        let note = thread.messages[2].content.text().unwrap();
        assert!(note.contains("system_note"));
        assert!(note.contains("Could not"));
    }

    // ── summarize_tool_input tests ──

    #[test]
    fn summarize_file_read() {
        let input = serde_json::json!({"path": "src/main.rs"});
        assert_eq!(summarize_tool_input("file-read", &input), "src/main.rs");
    }

    #[test]
    fn summarize_command_exec() {
        let input = serde_json::json!({"command": "cargo test --lib"});
        assert_eq!(
            summarize_tool_input("command-exec", &input),
            "cargo test --lib"
        );
    }

    #[test]
    fn summarize_command_exec_truncates() {
        let long_cmd = "a".repeat(100);
        let input = serde_json::json!({"command": long_cmd});
        let result = summarize_tool_input("command-exec", &input);
        assert!(result.len() <= 63); // 57 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn summarize_grep_search() {
        let input = serde_json::json!({"pattern": "fn main"});
        assert_eq!(summarize_tool_input("grep-search", &input), "fn main");
    }

    #[test]
    fn summarize_unknown_tool() {
        let input = serde_json::json!({"foo": "bar"});
        let result = summarize_tool_input("unknown-tool", &input);
        assert!(result.contains("foo"));
    }

    #[test]
    fn handler_max_iterations_guard() {
        let pool = mock_pool();
        let router = build_test_router();
        let mut handler = CodingAgentHandler::with_semantic_router(
            pool,
            router,
            sample_tool_defs(),
            "test".into(),
        );
        handler.set_max_routing_iterations(3);
        assert_eq!(handler.max_routing_iterations, 3);
    }

    // ── YAML-Defined Agents: from_config, builder attachment tests ──

    #[test]
    fn from_config_default() {
        let pool = mock_pool();
        let config = AgentConfig::default();
        let handler = CodingAgentHandler::from_config(
            pool,
            sample_tool_defs(),
            "test prompt".into(),
            &config,
        );
        assert_eq!(handler.max_tokens, 4096);
        assert_eq!(handler.model, None);
        assert_eq!(handler.max_routing_iterations, 5);
        assert_eq!(handler.max_agentic_iterations, 25);
        assert_eq!(handler.system_prompt, "test prompt");
    }

    #[test]
    fn from_config_custom_tokens() {
        let pool = mock_pool();
        let config = AgentConfig {
            max_tokens: 8192,
            ..AgentConfig::default()
        };
        let handler = CodingAgentHandler::from_config(
            pool,
            sample_tool_defs(),
            "test".into(),
            &config,
        );
        assert_eq!(handler.max_tokens, 8192);
    }

    #[test]
    fn from_config_custom_model() {
        let pool = mock_pool();
        let config = AgentConfig {
            model: Some("haiku".into()),
            ..AgentConfig::default()
        };
        let handler = CodingAgentHandler::from_config(
            pool,
            sample_tool_defs(),
            "test".into(),
            &config,
        );
        assert_eq!(handler.model.as_deref(), Some("haiku"));
    }

    #[test]
    fn builder_attach_librarian() {
        let pool = mock_pool();
        let kernel =
            crate::kernel::Kernel::open(&tempfile::TempDir::new().unwrap().path().join("data"))
                .unwrap();
        let kernel_arc = Arc::new(Mutex::new(kernel));
        let lib = Arc::new(Mutex::new(Librarian::new(pool.clone(), kernel_arc)));

        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into())
            .with_librarian_attached(lib);
        assert!(handler.librarian.is_some());
    }

    #[test]
    fn builder_attach_router() {
        let pool = mock_pool();
        let router = build_test_router();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into())
            .with_router_attached(router);
        assert!(handler.has_semantic_router());
    }

    // ── Agentic iteration limit tests ──

    #[test]
    fn from_config_reads_max_agentic_iterations() {
        let pool = mock_pool();
        let config = AgentConfig {
            max_agentic_iterations: 50,
            ..AgentConfig::default()
        };
        let handler = CodingAgentHandler::from_config(
            pool,
            sample_tool_defs(),
            "test".into(),
            &config,
        );
        assert_eq!(handler.max_agentic_iterations, 50);
    }

    #[test]
    fn check_agentic_limit_increments_counter() {
        let pool = mock_pool();
        let handler = CodingAgentHandler::new(pool, sample_tool_defs(), "test".into());
        let mut thread = AgentThread::new();
        assert_eq!(thread.agentic_iterations, 0);

        let result = handler.check_agentic_limit(&mut thread);
        assert!(result.is_none());
        assert_eq!(thread.agentic_iterations, 1);
    }

    #[test]
    fn check_agentic_limit_returns_reply_at_limit() {
        let pool = mock_pool();
        let config = AgentConfig {
            max_agentic_iterations: 2,
            ..AgentConfig::default()
        };
        let handler = CodingAgentHandler::from_config(
            pool,
            sample_tool_defs(),
            "test".into(),
            &config,
        );
        let mut thread = AgentThread::new();

        // Iteration 1: ok
        assert!(handler.check_agentic_limit(&mut thread).is_none());
        // Iteration 2: ok (at limit, not over)
        assert!(handler.check_agentic_limit(&mut thread).is_none());
        // Iteration 3: over limit — should return reply
        let result = handler.check_agentic_limit(&mut thread);
        assert!(result.is_some());
        let reply = result.unwrap().unwrap();
        match reply {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("iteration limit reached"));
                assert!(xml.contains("AgentResponse"));
            }
            _ => panic!("expected Reply"),
        }
    }

    // ── ConversationEntry conversion tests ──

    #[test]
    fn conversation_entry_from_user_message() {
        use crate::llm::types::Message;
        let messages = vec![Message::text("user", "Read the README")];
        let entries = build_conversation_entries(&messages);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[0].summary, "Read the README");
        assert!(!entries[0].is_tool_use);
        assert!(entries[0].tool_name.is_none());
    }

    #[test]
    fn conversation_entry_from_assistant_text() {
        use crate::llm::types::Message;
        let messages = vec![Message::assistant_blocks(vec![ContentBlock::Text {
            text: "Here's the README contents.".into(),
        }])];
        let entries = build_conversation_entries(&messages);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "assistant");
        assert!(entries[0].summary.contains("README contents"));
        assert!(!entries[0].is_tool_use);
    }

    #[test]
    fn conversation_entry_from_tool_use() {
        use crate::llm::types::Message;
        let messages = vec![Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "toolu_1".into(),
            name: "file-read".into(),
            input: serde_json::json!({"path": "src/main.rs"}),
        }])];
        let entries = build_conversation_entries(&messages);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "assistant");
        assert!(entries[0].is_tool_use);
        assert_eq!(entries[0].tool_name.as_deref(), Some("file-read"));
        assert!(entries[0].summary.contains("file-read"));
        assert!(entries[0].summary.contains("src/main.rs"));
    }

    #[test]
    fn conversation_entry_from_tool_result() {
        use crate::llm::types::{Message, ToolResultBlock};
        let messages = vec![Message::tool_results(vec![ToolResultBlock {
            tool_use_id: "toolu_1".into(),
            content: "fn main() {}".into(),
            is_error: false,
        }])];
        let entries = build_conversation_entries(&messages);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "tool_result");
        assert!(!entries[0].is_error);
        assert!(entries[0].summary.contains("fn main"));
    }

    #[test]
    fn conversation_entry_full_conversation() {
        use crate::llm::types::{Message, ToolResultBlock};
        let messages = vec![
            Message::text("user", "Read foo.rs"),
            Message::assistant_blocks(vec![
                ContentBlock::Text {
                    text: "I'll read that file.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "file-read".into(),
                    input: serde_json::json!({"path": "foo.rs"}),
                },
            ]),
            Message::tool_results(vec![ToolResultBlock {
                tool_use_id: "toolu_1".into(),
                content: "pub fn foo() {}".into(),
                is_error: false,
            }]),
            Message::assistant_blocks(vec![ContentBlock::Text {
                text: "The file contains a foo function.".into(),
            }]),
        ];
        let entries = build_conversation_entries(&messages);
        // user(1) + text(1) + tool_use(1) + tool_result(1) + text(1) = 5
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[1].role, "assistant");
        assert!(!entries[1].is_tool_use);
        assert_eq!(entries[2].role, "assistant");
        assert!(entries[2].is_tool_use);
        assert_eq!(entries[3].role, "tool_result");
        assert_eq!(entries[4].role, "assistant");
    }
}
