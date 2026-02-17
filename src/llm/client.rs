//! Raw HTTP client for the Anthropic Messages API.
//!
//! No pipeline awareness — just makes API calls via reqwest.

use reqwest::Client;

use super::types::{MessagesRequest, MessagesResponse};

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
        Self {
            http: Client::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;

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
            messages: vec![Message {
                role: "user".into(),
                content: "Hello".into(),
            }],
            system: None,
            temperature: Some(0.7),
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
