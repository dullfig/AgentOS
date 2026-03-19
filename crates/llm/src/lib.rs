//! LLM Pool — model routing and connection management for Anthropic API.
//!
//! Wraps AnthropicClient with model aliasing and default model selection.
//! The `llm-pool` listener in the pipeline uses this for inference.

pub mod client;
pub mod types;

pub use client::{AnthropicClient, LlmError, ModelInfo};
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

    /// Create a pool from a ModelsConfig.
    /// Resolves the default model and gets API key + base_url from the provider.
    pub fn from_config(config: &agentos_config::ModelsConfig) -> Result<Self, LlmError> {
        let default_alias = config
            .default
            .as_deref()
            .unwrap_or("sonnet");

        let resolved = config.resolve_or_fallback(default_alias).ok_or_else(|| {
            LlmError::MissingApiKey("No models configured and ANTHROPIC_API_KEY not set".into())
        })?;

        let api_key = resolved.api_key.ok_or_else(|| {
            LlmError::MissingApiKey(format!(
                "No API key for provider '{}'. Set it via /models add or ANTHROPIC_API_KEY env var.",
                resolved.provider
            ))
        })?;

        let client = if let Some(base_url) = resolved.base_url {
            AnthropicClient::with_base_url(api_key, base_url)
        } else {
            AnthropicClient::new(api_key)
        };

        Ok(Self {
            client,
            default_model: resolved.model_id,
        })
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

    /// Change the default model using config resolution first.
    pub fn set_default_model_from_config(&mut self, config: &agentos_config::ModelsConfig, alias: &str) {
        self.default_model = crate::types::resolve_model_from_config(config, alias);
    }

    /// Rebuild the pool from a ModelsConfig — replaces client, api_key, base_url, default model.
    /// Used after `/models add`, `/models update`, or `/model <alias>` to hot-swap credentials.
    pub fn rebuild_from_config(&mut self, config: &agentos_config::ModelsConfig) -> Result<(), LlmError> {
        let default_alias = config.default.as_deref().unwrap_or("sonnet");
        let resolved = config.resolve_or_fallback(default_alias).ok_or_else(|| {
            LlmError::MissingApiKey("No models configured".into())
        })?;
        let api_key = resolved.api_key.ok_or_else(|| {
            LlmError::MissingApiKey(format!(
                "No API key for provider '{}'. Use /models update {} to set it.",
                resolved.provider, resolved.provider
            ))
        })?;
        self.client = if let Some(base_url) = resolved.base_url {
            AnthropicClient::with_base_url(api_key, base_url)
        } else {
            AnthropicClient::new(api_key)
        };
        self.default_model = resolved.model_id;
        Ok(())
    }

    /// Rebuild targeting a specific alias (for `/model <alias>` cross-provider switch).
    /// If the alias resolves in config (with key), rebuilds the full client.
    /// If not in config, falls back to just changing the default model ID (keeps existing client).
    pub fn rebuild_for_alias(&mut self, config: &agentos_config::ModelsConfig, alias: &str) -> Result<(), LlmError> {
        if let Some(resolved) = config.resolve_or_fallback(alias) {
            if let Some(api_key) = resolved.api_key {
                // Full rebuild — new provider/key
                self.client = if let Some(base_url) = resolved.base_url {
                    AnthropicClient::with_base_url(api_key, base_url)
                } else {
                    AnthropicClient::new(api_key)
                };
                self.default_model = resolved.model_id;
            } else {
                // Config knows the model but no key — just change model ID, keep existing client
                self.default_model = resolved.model_id;
            }
        } else {
            // Not in config at all — resolve alias via hardcoded table, keep existing client
            self.default_model = resolve_model(alias).to_string();
        }
        Ok(())
    }

    /// Get the default model (resolved to full ID).
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// List models available from the API.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.client.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_creation() {
        let pool = LlmPool::new("test-key".into(), "opus");
        assert_eq!(pool.default_model(), "claude-opus-4-6");
    }

    #[test]
    fn pool_creation_full_model_id() {
        let pool = LlmPool::new("test-key".into(), "claude-sonnet-4-5-20250929");
        assert_eq!(pool.default_model(), "claude-sonnet-4-5-20250929");
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

    #[test]
    fn pool_from_config() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "opus", "claude-opus-4-6", Some("sk-test".into()), None);
        config.set_default("opus");

        let pool = LlmPool::from_config(&config).unwrap();
        assert_eq!(pool.default_model(), "claude-opus-4-6");
    }

    #[test]
    fn pool_from_config_no_key_errors() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "opus", "claude-opus-4-6", None, None);
        config.set_default("opus");

        let result = LlmPool::from_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
    }

    #[test]
    fn pool_from_config_with_base_url() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("openai", "gpt4", "gpt-4o", Some("sk-test".into()), Some("https://api.openai.com/v1".into()));
        config.set_default("gpt4");

        let pool = LlmPool::from_config(&config).unwrap();
        assert_eq!(pool.default_model(), "gpt-4o");
    }

    #[test]
    fn set_default_model_from_config() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "opus", "claude-opus-4-6", Some("key".into()), None);
        config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", None, None);

        let mut pool = LlmPool::new("key".into(), "opus");
        pool.set_default_model_from_config(&config, "haiku");
        assert_eq!(pool.default_model(), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn rebuild_from_config_updates_model() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", Some("new-key".into()), None);
        config.set_default("haiku");

        let mut pool = LlmPool::new("old-key".into(), "opus");
        assert_eq!(pool.default_model(), "claude-opus-4-6");

        pool.rebuild_from_config(&config).unwrap();
        assert_eq!(pool.default_model(), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn rebuild_from_config_no_key_errors() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", None, None);
        config.set_default("haiku");

        let mut pool = LlmPool::new("key".into(), "opus");
        let result = pool.rebuild_from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn rebuild_for_alias_with_config() {
        let mut config = agentos_config::ModelsConfig::default();
        config.add_model("anthropic", "haiku", "claude-haiku-4-5-20251001", Some("key".into()), None);
        config.add_model("openai", "gpt4", "gpt-4o", Some("sk-openai".into()), Some("https://api.openai.com/v1".into()));

        let mut pool = LlmPool::new("key".into(), "haiku");
        // Switch to a different provider
        pool.rebuild_for_alias(&config, "gpt4").unwrap();
        assert_eq!(pool.default_model(), "gpt-4o");
    }

    #[test]
    fn rebuild_for_alias_fallback_no_config() {
        let config = agentos_config::ModelsConfig::default(); // empty
        let mut pool = LlmPool::new("key".into(), "opus");

        // Alias not in config — falls back to hardcoded resolve, keeps client
        pool.rebuild_for_alias(&config, "haiku").unwrap();
        assert_eq!(pool.default_model(), "claude-haiku-4-5-20251001");
    }
}
