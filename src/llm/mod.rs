//! LLM Pool — model routing and connection management for Anthropic API.
//!
//! Wraps AnthropicClient with model aliasing and default model selection.
//! The `llm-pool` listener in the pipeline uses this for inference.

pub mod client;
pub mod handler;
pub mod types;

use client::{AnthropicClient, LlmError};
use types::{resolve_model, Message, MessagesRequest, MessagesResponse};

/// LLM connection pool with model routing.
#[derive(Debug)]
pub struct LlmPool {
    client: AnthropicClient,
    default_model: String,
}

impl LlmPool {
    /// Create a pool with an explicit API key and default model.
    pub fn new(api_key: String, default_model: &str) -> Self {
        Self {
            client: AnthropicClient::new(api_key),
            default_model: resolve_model(default_model).to_string(),
        }
    }

    /// Create a pool reading ANTHROPIC_API_KEY from the environment.
    pub fn from_env(default_model: &str) -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            LlmError::MissingApiKey("ANTHROPIC_API_KEY environment variable not set".into())
        })?;
        Ok(Self::new(api_key, default_model))
    }

    /// Create a pool with a custom base URL (for testing).
    pub fn with_base_url(api_key: String, default_model: &str, base_url: String) -> Self {
        Self {
            client: AnthropicClient::with_base_url(api_key, base_url),
            default_model: resolve_model(default_model).to_string(),
        }
    }

    /// Send a completion request.
    ///
    /// - `model`: None means use default model, Some("alias") resolves aliases.
    /// - `messages`: Conversation history.
    /// - `max_tokens`: Maximum tokens to generate.
    /// - `system`: Optional system prompt.
    pub async fn complete(
        &self,
        model: Option<&str>,
        messages: Vec<Message>,
        max_tokens: u32,
        system: Option<&str>,
    ) -> Result<MessagesResponse, LlmError> {
        let resolved_model = model
            .map(|m| resolve_model(m).to_string())
            .unwrap_or_else(|| self.default_model.clone());

        let request = MessagesRequest {
            model: resolved_model,
            max_tokens,
            messages,
            system: system.map(|s| s.to_string()),
            temperature: None,
            tools: None,
        };

        self.client.messages(&request).await
    }

    /// Send a completion request with tool definitions.
    pub async fn complete_with_tools(
        &self,
        model: Option<&str>,
        messages: Vec<Message>,
        max_tokens: u32,
        system: Option<&str>,
        tools: Vec<types::ToolDefinition>,
    ) -> Result<MessagesResponse, LlmError> {
        let resolved_model = model
            .map(|m| resolve_model(m).to_string())
            .unwrap_or_else(|| self.default_model.clone());

        let request = MessagesRequest {
            model: resolved_model,
            max_tokens,
            messages,
            system: system.map(|s| s.to_string()),
            temperature: None,
            tools: if tools.is_empty() { None } else { Some(tools) },
        };

        self.client.messages(&request).await
    }

    /// Change the default model at runtime (e.g. from `/model` command).
    pub fn set_default_model(&mut self, alias: &str) {
        self.default_model = resolve_model(alias).to_string();
    }

    /// Get the default model (resolved to full ID).
    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creation() {
        let pool = LlmPool::new("test-key".into(), "opus");
        assert_eq!(pool.default_model(), "claude-opus-4-20250514");
    }

    #[test]
    fn pool_creation_full_model_id() {
        let pool = LlmPool::new("test-key".into(), "claude-sonnet-4-5-20250514");
        assert_eq!(pool.default_model(), "claude-sonnet-4-5-20250514");
    }

    #[test]
    fn from_env_missing_key() {
        // Temporarily ensure the env var is not set
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = LlmPool::from_env("opus");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn pool_with_custom_base_url() {
        let pool = LlmPool::with_base_url("key".into(), "haiku", "http://localhost:9999".into());
        assert_eq!(pool.default_model(), "claude-haiku-4-5-20251001");
    }
}
