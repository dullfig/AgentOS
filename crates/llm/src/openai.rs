//! OpenAI-compatible HTTP client.
//!
//! Speaks the `/v1/chat/completions` wire format used by OpenAI, cortex,
//! vLLM, llama.cpp server, and most local inference engines.
//!
//! Translates between AgentOS internal types (`Message`, `ContentBlock`,
//! `MessagesRequest`, `MessagesResponse`) and the OpenAI JSON format so the
//! rest of the pipeline doesn't need to know which wire protocol is in use.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::client::LlmError;
use super::types::{
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, Usage,
};

/// HTTP request timeout (5 minutes, matching AnthropicClient).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// OpenAI-compatible HTTP client.
#[derive(Debug)]
pub struct OpenAiClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            api_key,
            base_url,
        }
    }

    /// Send a chat completions request, translating to/from AgentOS types.
    pub async fn messages(&self, request: &MessagesRequest) -> Result<MessagesResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let oai_request = to_openai_request(request);

        let response = self
            .http
            .post(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&oai_request)
            .send()
            .await?;

        let status = response.status().as_u16();

        if status == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            return Err(LlmError::RateLimited { retry_after });
        }

        if status >= 400 {
            let body = response.text().await.unwrap_or_else(|_| "(no body)".into());
            return Err(LlmError::ApiError {
                status,
                message: body,
            });
        }

        let oai_resp: OaiChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(format!("failed to parse response: {e}")))?;

        Ok(from_openai_response(oai_resp))
    }
}

// ── OpenAI wire format types (private) ─────────────────────────────

#[derive(Debug, Serialize)]
struct OaiChatRequest {
    model: String,
    messages: Vec<OaiMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OaiTool>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OaiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiToolCall>>,
    /// For tool result messages (role = "tool")
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OaiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OaiFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OaiFunction {
    name: String,
    arguments: String, // JSON string
}

#[derive(Debug, Serialize)]
struct OaiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OaiToolDef,
}

#[derive(Debug, Serialize)]
struct OaiToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OaiChatResponse {
    id: String,
    model: String,
    choices: Vec<OaiChoice>,
    usage: OaiUsage,
}

#[derive(Debug, Deserialize)]
struct OaiChoice {
    message: OaiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ── Translation functions ──────────────────────────────────────────

fn to_openai_request(req: &MessagesRequest) -> OaiChatRequest {
    let mut messages = Vec::new();

    // System prompt becomes a system message
    if let Some(ref system) = req.system {
        messages.push(OaiMessage {
            role: "system".into(),
            content: Some(system.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert each AgentOS message
    for msg in &req.messages {
        match &msg.content {
            MessageContent::Text(text) => {
                messages.push(OaiMessage {
                    role: msg.role.clone(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            MessageContent::Blocks(blocks) => {
                // Assistant messages with tool_use blocks
                if msg.role == "assistant" {
                    let text = blocks.iter().find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    });
                    let tool_calls: Vec<OaiToolCall> = blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse { id, name, input } => Some(OaiToolCall {
                                id: id.clone(),
                                call_type: "function".into(),
                                function: OaiFunction {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_default(),
                                },
                            }),
                            _ => None,
                        })
                        .collect();

                    messages.push(OaiMessage {
                        role: "assistant".into(),
                        content: text,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                        tool_call_id: None,
                    });
                } else {
                    // User messages with tool_result blocks → individual "tool" messages
                    for block in blocks {
                        match block {
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => {
                                messages.push(OaiMessage {
                                    role: "tool".into(),
                                    content: content.clone(),
                                    tool_calls: None,
                                    tool_call_id: Some(tool_use_id.clone()),
                                });
                            }
                            ContentBlock::Text { text } => {
                                messages.push(OaiMessage {
                                    role: msg.role.clone(),
                                    content: Some(text.clone()),
                                    tool_calls: None,
                                    tool_call_id: None,
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Convert tool definitions
    let tools = req.tools.as_ref().map(|defs| {
        defs.iter()
            .map(|d| OaiTool {
                tool_type: "function".into(),
                function: OaiToolDef {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    parameters: d.input_schema.clone(),
                },
            })
            .collect()
    });

    OaiChatRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        tools,
    }
}

fn from_openai_response(resp: OaiChatResponse) -> MessagesResponse {
    let choice = resp.choices.into_iter().next();

    let (content, stop_reason) = match choice {
        Some(c) => {
            let mut blocks = Vec::new();

            // Text content
            if let Some(text) = c.message.content {
                if !text.is_empty() {
                    blocks.push(ContentBlock::Text { text });
                }
            }

            // Tool calls → ToolUse blocks
            if let Some(tool_calls) = c.message.tool_calls {
                for tc in tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
            }

            // Map finish_reason: "tool_calls" → "tool_use", "stop" → "end_turn"
            let reason = c.finish_reason.map(|r| match r.as_str() {
                "tool_calls" => "tool_use".into(),
                "stop" => "end_turn".into(),
                other => other.to_string(),
            });

            (blocks, reason)
        }
        None => (vec![], None),
    };

    MessagesResponse {
        id: resp.id,
        model: resp.model,
        content,
        stop_reason,
        usage: Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ToolDefinition};

    #[test]
    fn request_translation_basic() {
        let req = MessagesRequest {
            model: "qwen-30b".into(),
            max_tokens: 4096,
            messages: vec![Message::text("user", "Hello")],
            system: Some("You are helpful.".into()),
            temperature: None,
            tools: None,
        };

        let oai = to_openai_request(&req);
        assert_eq!(oai.model, "qwen-30b");
        assert_eq!(oai.messages.len(), 2); // system + user
        assert_eq!(oai.messages[0].role, "system");
        assert_eq!(oai.messages[0].content.as_deref(), Some("You are helpful."));
        assert_eq!(oai.messages[1].role, "user");
        assert_eq!(oai.messages[1].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn request_translation_with_tools() {
        let req = MessagesRequest {
            model: "qwen-30b".into(),
            max_tokens: 1024,
            messages: vec![Message::text("user", "calc 2+2")],
            system: None,
            temperature: None,
            tools: Some(vec![ToolDefinition {
                name: "calculator".into(),
                description: "A calculator".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            }]),
        };

        let oai = to_openai_request(&req);
        assert!(oai.tools.is_some());
        let tools = oai.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "calculator");
    }

    #[test]
    fn request_translation_tool_use_roundtrip() {
        // Simulate: assistant made a tool call, user returns result
        let req = MessagesRequest {
            model: "qwen-30b".into(),
            max_tokens: 1024,
            messages: vec![
                Message::text("user", "What is 2+2?"),
                Message::assistant_blocks(vec![
                    ContentBlock::Text { text: "Let me calculate.".into() },
                    ContentBlock::ToolUse {
                        id: "call_123".into(),
                        name: "calculator".into(),
                        input: serde_json::json!({"expr": "2+2"}),
                    },
                ]),
                Message {
                    role: "user".into(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: "call_123".into(),
                        content: Some("4".into()),
                        is_error: None,
                    }]),
                },
            ],
            system: None,
            temperature: None,
            tools: None,
        };

        let oai = to_openai_request(&req);
        // user + assistant (with tool_calls) + tool result
        assert_eq!(oai.messages.len(), 3);

        // Assistant message should have tool_calls
        let assistant = &oai.messages[1];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content.as_deref(), Some("Let me calculate."));
        let tool_calls = assistant.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "calculator");

        // Tool result should be role "tool" with tool_call_id
        let tool_msg = &oai.messages[2];
        assert_eq!(tool_msg.role, "tool");
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(tool_msg.content.as_deref(), Some("4"));
    }

    #[test]
    fn response_translation_text_only() {
        let oai_resp = OaiChatResponse {
            id: "chatcmpl-abc".into(),
            model: "qwen-30b".into(),
            choices: vec![OaiChoice {
                message: OaiMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: OaiUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let resp = from_openai_response(oai_resp);
        assert_eq!(resp.id, "chatcmpl-abc");
        assert_eq!(resp.text(), Some("Hello!"));
        assert!(!resp.has_tool_use());
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
    }

    #[test]
    fn response_translation_with_tool_calls() {
        let oai_resp = OaiChatResponse {
            id: "chatcmpl-xyz".into(),
            model: "qwen-30b".into(),
            choices: vec![OaiChoice {
                message: OaiMessage {
                    role: "assistant".into(),
                    content: Some("I'll calculate.".into()),
                    tool_calls: Some(vec![OaiToolCall {
                        id: "call_456".into(),
                        call_type: "function".into(),
                        function: OaiFunction {
                            name: "calculator".into(),
                            arguments: r#"{"expr":"2+2"}"#.into(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: OaiUsage {
                prompt_tokens: 20,
                completion_tokens: 15,
            },
        };

        let resp = from_openai_response(oai_resp);
        assert_eq!(resp.text(), Some("I'll calculate."));
        assert!(resp.has_tool_use());
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));

        let tool_blocks = resp.tool_use_blocks();
        assert_eq!(tool_blocks.len(), 1);
        match tool_blocks[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_456");
                assert_eq!(name, "calculator");
                assert_eq!(input["expr"], "2+2");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn response_translation_empty_choices() {
        let oai_resp = OaiChatResponse {
            id: "chatcmpl-empty".into(),
            model: "test".into(),
            choices: vec![],
            usage: OaiUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
            },
        };

        let resp = from_openai_response(oai_resp);
        assert!(resp.content.is_empty());
        assert!(resp.stop_reason.is_none());
    }
}
