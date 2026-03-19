//! Raw HTTP client for the Anthropic Messages API.
//!
//! No pipeline awareness — just makes API calls via reqwest.

use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;

use super::types::{MessagesRequest, MessagesResponse};

/// HTTP request timeout for LLM API calls (5 minutes).
/// Large contexts can take a while; this prevents infinite hangs.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// A model returned by the List Models API.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    /// Model ID (e.g. "claude-sonnet-4-6").
    pub id: String,
    /// Human-readable name (e.g. "Claude Sonnet 4.6").
    pub display_name: String,
    /// ISO 8601 creation/release date.
    pub created_at: String,
}

/// Response from GET /v1/models.
#[derive(Debug, Deserialize)]
struct ModelsListResponse {
    data: Vec<ModelInfo>,
    has_more: bool,
    #[allow(dead_code)]
    last_id: Option<String>,
}

/// Errors from LLM operations.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("rate limited (retry after {retry_after:?}s)")]
    RateLimited { retry_after: Option<u64> },

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("missing API key: {0}")]
    MissingApiKey(String),
}

/// Raw HTTP client for the Anthropic Messages API.
#[derive(Debug)]
pub struct AnthropicClient {
    http: Client,
    api_key: String,
    base_url: String,
    api_version: String,
}

impl AnthropicClient {
    /// Create a client with default base URL (https://api.anthropic.com).
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, "https://api.anthropic.com".into())
    }

    /// Create a client with a custom base URL (for testing with mock servers).
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            http,
            api_key,
            base_url,
            api_version: "2023-06-01".into(),
        }
    }

    /// Send a messages request to the Anthropic API.
    pub async fn messages(&self, request: &MessagesRequest) -> Result<MessagesResponse, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .json(request)
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

        let resp: MessagesResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(format!("failed to parse response: {e}")))?;

        Ok(resp)
    }

    /// List available models from the API.
    /// Paginates automatically to fetch all models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let mut url = format!("{}/v1/models?limit=100", self.base_url);
            if let Some(ref cursor) = after_id {
                url.push_str(&format!("&after_id={cursor}"));
            }

            let response = self
                .http
                .get(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", &self.api_version)
                .send()
                .await?;

            let status = response.status().as_u16();
            if status >= 400 {
                let body = response.text().await.unwrap_or_else(|_| "(no body)".into());
                return Err(LlmError::ApiError {
                    status,
                    message: body,
                });
            }

            let page: ModelsListResponse = response
                .json()
                .await
                .map_err(|e| LlmError::InvalidResponse(format!("failed to parse models list: {e}")))?;

            let has_more = page.has_more;
            let last = page.data.last().map(|m| m.id.clone());
            all_models.extend(page.data);

            if has_more {
                after_id = last;
            } else {
                break;
            }
        }

        Ok(all_models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn client_creation() {
        let client = AnthropicClient::new("test-key".into());
        assert_eq!(client.base_url, "https://api.anthropic.com");
        assert_eq!(client.api_version, "2023-06-01");
    }

    #[test]
    fn client_custom_base_url() {
        let client =
            AnthropicClient::with_base_url("test-key".into(), "http://localhost:8080".into());
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn request_builds_correctly() {
        let req = MessagesRequest {
            model: "claude-opus-4-20250514".into(),
            max_tokens: 1024,
            messages: vec![Message::text("user", "Hello")],
            system: None,
            temperature: Some(0.7),
            tools: None,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-opus-4-20250514");
        assert_eq!(json["max_tokens"], 1024);
        // f32 precision: 0.7f32 round-trips through JSON as ~0.699999988
        let temp = json["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001);
        assert!(json.get("system").is_none());
    }

    #[test]
    fn error_display() {
        let err = LlmError::ApiError {
            status: 401,
            message: "invalid api key".into(),
        };
        assert!(err.to_string().contains("401"));
        assert!(err.to_string().contains("invalid api key"));

        let err = LlmError::RateLimited {
            retry_after: Some(30),
        };
        assert!(err.to_string().contains("rate limited"));

        let err = LlmError::MissingApiKey("ANTHROPIC_API_KEY not set".into());
        assert!(err.to_string().contains("missing API key"));
    }
}
