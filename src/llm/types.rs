//! Rust types for the Anthropic Messages API.
//!
//! Serde-serializable to JSON for HTTP calls. Internal types stay Rust-native.

use serde::{Deserialize, Serialize};

/// Resolve model aliases to full Anthropic model IDs.
pub fn resolve_model(alias: &str) -> &str {
    match alias {
        "opus" => "claude-opus-4-20250514",
        "sonnet" => "claude-sonnet-4-5-20250514",
        "haiku" => "claude-haiku-4-5-20251001",
        _ => alias, // pass through full model IDs
    }
}

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
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
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

/// A content block in the response.
#[derive(Debug, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
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
        self.content
            .iter()
            .find(|b| b.content_type == "text")
            .and_then(|b| b.text.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_aliases() {
        assert_eq!(resolve_model("opus"), "claude-opus-4-20250514");
        assert_eq!(resolve_model("sonnet"), "claude-sonnet-4-5-20250514");
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
            messages: vec![Message {
                role: "user".into(),
                content: "Hello".into(),
            }],
            system: Some("You are helpful.".into()),
            temperature: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"claude-opus-4-20250514\""));
        assert!(json.contains("\"max_tokens\":4096"));
        assert!(json.contains("\"system\":\"You are helpful.\""));
        // temperature is None → should be skipped
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn response_deserializes_from_json() {
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
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn message_roundtrip() {
        let msg = Message {
            role: "user".into(),
            content: "What is 2+2?".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content, "What is 2+2?");
    }
}
