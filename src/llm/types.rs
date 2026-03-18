//! Rust types for the Anthropic Messages API.
//!
//! Core message types (`Message`, `ContentBlock`, `MessageContent`, `ToolDefinition`,
//! `ToolResultBlock`) live in `agentos-events` for cross-crate access.
//! Re-exported here for convenience.
//!
//! API-specific types (`MessagesRequest`, `MessagesResponse`, `Usage`) and
//! model resolution functions stay here.

use serde::{Deserialize, Serialize};

// Re-export shared types from events crate
pub use agentos_events::{
    ContentBlock, Message, MessageContent, ToolDefinition, ToolResultBlock,
};

/// Resolve model aliases to full Anthropic model IDs.
/// This is the hardcoded fallback — prefer `resolve_model_from_config` when config is available.
pub fn resolve_model(alias: &str) -> &str {
    match alias {
        "opus" => "claude-opus-4-6",
        "sonnet" => "claude-sonnet-4-6",
        "sonnet-4.5" => "claude-sonnet-4-5-20250929",
        "haiku" => "claude-haiku-4-5-20251001",
        _ => alias, // pass through full model IDs
    }
}

/// Resolve a model alias using the config first, falling back to hardcoded aliases.
/// Returns the full model ID string.
pub fn resolve_model_from_config(config: &crate::config::ModelsConfig, alias: &str) -> String {
    if let Some(resolved) = config.resolve(alias) {
        resolved.model_id
    } else {
        resolve_model(alias).to_string()
    }
}

// ── Request / Response (API-specific, stays in llm) ──

/// Request body for the Anthropic Messages API.
#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Response from the Anthropic Messages API.
#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

/// Token usage from the API response.
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl MessagesResponse {
    /// Extract the text content from the first text block, if any.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Check if the response contains tool_use blocks.
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }

    /// Extract all tool_use blocks from the response.
    pub fn tool_use_blocks(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_aliases() {
        assert_eq!(resolve_model("opus"), "claude-opus-4-6");
        assert_eq!(resolve_model("sonnet"), "claude-sonnet-4-6");
        assert_eq!(resolve_model("sonnet-4.5"), "claude-sonnet-4-5-20250929");
        assert_eq!(resolve_model("haiku"), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn resolve_model_passthrough() {
        assert_eq!(
            resolve_model("claude-opus-4-20250514"),
            "claude-opus-4-20250514"
        );
        assert_eq!(resolve_model("custom-model-id"), "custom-model-id");
    }

    #[test]
    fn request_serializes_to_json() {
        let req = MessagesRequest {
            model: "claude-opus-4-20250514".into(),
            max_tokens: 4096,
            messages: vec![Message::text("user", "Hello")],
            system: Some("You are helpful.".into()),
            temperature: None,
            tools: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"claude-opus-4-20250514\""));
        assert!(json.contains("\"max_tokens\":4096"));
        assert!(json.contains("\"system\":\"You are helpful.\""));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("tools"));
    }

    #[test]
    fn request_with_tools_serializes() {
        let req = MessagesRequest {
            model: "haiku".into(),
            max_tokens: 1024,
            messages: vec![Message::text("user", "What is 2+2?")],
            system: None,
            temperature: None,
            tools: Some(vec![ToolDefinition {
                name: "calculator".into(),
                description: "A simple calculator".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }),
            }]),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("calculator"));
        assert!(json.contains("input_schema"));
    }

    #[test]
    fn response_deserializes_text_only() {
        let json = r#"{
            "id": "msg_123",
            "model": "claude-opus-4-20250514",
            "content": [
                {"type": "text", "text": "Hello back!"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.text(), Some("Hello back!"));
        assert!(!resp.has_tool_use());
        assert!(resp.tool_use_blocks().is_empty());
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn response_deserializes_tool_use() {
        let json = r#"{
            "id": "msg_456",
            "model": "claude-haiku-4-5-20251001",
            "content": [
                {"type": "text", "text": "I'll calculate that."},
                {"type": "tool_use", "id": "toolu_123", "name": "calculator", "input": {"expression": "2+2"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text(), Some("I'll calculate that."));
        assert!(resp.has_tool_use());
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));

        let tool_blocks = resp.tool_use_blocks();
        assert_eq!(tool_blocks.len(), 1);
        match tool_blocks[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "calculator");
                assert_eq!(input["expression"], "2+2");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn response_deserializes_multiple_tool_use() {
        let json = r#"{
            "id": "msg_789",
            "model": "claude-opus-4-20250514",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "file-ops", "input": {"action": "read", "path": "a.rs"}},
                {"type": "tool_use", "id": "toolu_2", "name": "shell", "input": {"command": "ls"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 25}
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(resp.has_tool_use());
        assert_eq!(resp.tool_use_blocks().len(), 2);
        assert!(resp.text().is_none());
    }

    #[test]
    fn message_text_helper() {
        let msg = Message::text("user", "What is 2+2?");
        assert_eq!(msg.role, "user");
        match &msg.content {
            MessageContent::Text(s) => assert_eq!(s, "What is 2+2?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn message_text_serializes_as_string() {
        let msg = Message::text("user", "Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn message_blocks_serialize_as_array() {
        let msg = Message::tool_results(vec![ToolResultBlock {
            tool_use_id: "toolu_123".into(),
            content: "4".into(),
            is_error: false,
        }]);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"tool_use_id\":\"toolu_123\""));
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(!json.contains("is_error"));
    }

    #[test]
    fn message_tool_result_with_error() {
        let msg = Message::tool_results(vec![ToolResultBlock {
            tool_use_id: "toolu_err".into(),
            content: "file not found".into(),
            is_error: true,
        }]);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"is_error\":true"));
    }

    #[test]
    fn message_content_deserializes_from_string() {
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        match &msg.content {
            MessageContent::Text(s) => assert_eq!(s, "Hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn message_content_deserializes_from_array() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me help."},
                {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 1}}
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        match &msg.content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(&blocks[0], ContentBlock::Text { .. }));
                assert!(matches!(&blocks[1], ContentBlock::ToolUse { .. }));
            }
            _ => panic!("expected Blocks"),
        }
    }

    #[test]
    fn message_content_text_helper() {
        let text_content = MessageContent::Text("hello".into());
        assert_eq!(text_content.text(), Some("hello".into()));
        assert!(!text_content.has_tool_use());

        let blocks_content = MessageContent::Blocks(vec![
            ContentBlock::Text { text: "part 1 ".into() },
            ContentBlock::Text { text: "part 2".into() },
        ]);
        assert_eq!(blocks_content.text(), Some("part 1 part 2".into()));

        let tool_only = MessageContent::Blocks(vec![ContentBlock::ToolUse {
            id: "t1".into(),
            name: "calc".into(),
            input: serde_json::json!({}),
        }]);
        assert!(tool_only.text().is_none());
        assert!(tool_only.has_tool_use());
    }

    #[test]
    fn message_roundtrip_text() {
        let msg = Message::text("user", "What is 2+2?");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content.text(), Some("What is 2+2?".into()));
    }

    #[test]
    fn message_roundtrip_blocks() {
        let msg = Message::assistant_blocks(vec![
            ContentBlock::Text { text: "I'll help.".into() },
            ContentBlock::ToolUse {
                id: "toolu_abc".into(),
                name: "file-ops".into(),
                input: serde_json::json!({"action": "read", "path": "foo.rs"}),
            },
        ]);
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "assistant");
        assert!(back.content.has_tool_use());
        assert_eq!(back.content.tool_use_blocks().len(), 1);
    }

    #[test]
    fn tool_definition_roundtrip() {
        let def = ToolDefinition {
            name: "file-ops".into(),
            description: "File operations".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "list"]},
                    "path": {"type": "string"}
                },
                "required": ["action", "path"]
            }),
        };

        let json = serde_json::to_string(&def).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "file-ops");
        assert_eq!(back.input_schema["type"], "object");
    }

    #[test]
    fn content_block_text_roundtrip() {
        let block = ContentBlock::Text { text: "Hello world".into() };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        match back {
            ContentBlock::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn content_block_tool_use_roundtrip() {
        let block = ContentBlock::ToolUse {
            id: "toolu_x".into(),
            name: "shell".into(),
            input: serde_json::json!({"command": "echo hi"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        match back {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_x");
                assert_eq!(name, "shell");
                assert_eq!(input["command"], "echo hi");
            }
            _ => panic!("expected ToolUse"),
        }
    }
}
