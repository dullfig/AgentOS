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
use tokio::sync::Mutex;

use crate::librarian::Librarian;
use crate::llm::types::{ContentBlock, ToolDefinition, ToolResultBlock};
use crate::llm::LlmPool;

use super::state::{AgentState, AgentThread, PendingToolCall};
use super::translate;

/// The coding agent handler — stateful, per-thread conversation management.
pub struct CodingAgentHandler {
    pool: Arc<Mutex<LlmPool>>,
    librarian: Option<Arc<Mutex<Librarian>>>,
    tool_definitions: Vec<ToolDefinition>,
    /// Per-thread conversation state.
    threads: Arc<Mutex<HashMap<String, AgentThread>>>,
    system_prompt: String,
}

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
        }
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
            Some("opus"),
            thread.messages.clone(),
            4096,
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
                    collected.push(ToolResultBlock {
                        tool_use_id,
                        content: result_content,
                        is_error,
                    });

                    let next_index = current_index + 1;

                    if next_index < pending.len() {
                        // More tool calls to dispatch
                        let next = &pending[next_index];
                        let xml =
                            translate::tool_call_to_xml(&next.tool_name, &next.input);
                        let next_name = next.tool_name.clone();
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

                    // All collected — record in conversation history and call Opus again
                    thread.push_assistant_blocks(assistant_blocks);
                    thread.push_tool_results(collected);
                    thread.state = AgentState::Ready;

                    let response = self
                        .call_opus(thread)
                        .await
                        .map_err(PipelineError::Handler)?;
                    let action = self.process_response(&response);
                    Self::dispatch_response(thread, action)
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

            let response = self
                .call_opus(thread)
                .await
                .map_err(PipelineError::Handler)?;
            let action = self.process_response(&response);
            Self::dispatch_response(thread, action)
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
        Some(xml[start..end].to_string())
    } else {
        None
    }
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
        crate::agent::tools::build_tool_definitions(&["file-ops", "shell"])
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
                    name: "file-ops".into(),
                    input: serde_json::json!({"action": "read", "path": "src/main.rs"}),
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
                assert_eq!(pending[0].tool_name, "file-ops");
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
                    name: "file-ops".into(),
                    input: serde_json::json!({"action": "read", "path": "a.rs"}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_2".into(),
                    name: "shell".into(),
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
                assert_eq!(pending[0].tool_name, "file-ops");
                assert_eq!(pending[1].tool_name, "shell");
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
                name: "file-ops".into(),
                input: serde_json::json!({"action": "read", "path": "foo.rs"}),
            }],
            pending: vec![PendingToolCall {
                tool_use_id: "toolu_1".into(),
                tool_name: "file-ops".into(),
                input: serde_json::json!({"action": "read", "path": "foo.rs"}),
            }],
        };

        let result = CodingAgentHandler::dispatch_response(&mut thread, action).unwrap();
        match result {
            HandlerResponse::Send { to, payload_xml } => {
                assert_eq!(to, "file-ops");
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<FileOpsRequest>"));
                assert!(xml.contains("<action>read</action>"));
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
            from: "file-ops".into(),
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
}
