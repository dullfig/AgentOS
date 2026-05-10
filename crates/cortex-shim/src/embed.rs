//! Cortex hidden-state extraction client.
//!
//! Talks to cortex's `POST /v1/embed` endpoint to turn a text context
//! into a hidden-state vector at a specified layer + pooling. Required
//! by the shim-expert agent's training pipeline: each `(text, label)`
//! example is embedded against the same model the shim will eventually
//! attach to, producing the `(hidden_vector, label)` pairs the FFN
//! trainer consumes.
//!
//! # Wire format (proposed; final naming pending cortex-claude review)
//!
//! ```text
//! POST /v1/embed
//! {
//!   "context": "...",
//!   "layer":   "final" | "entrance:N",
//!   "pooling": "last_token" | "mean" | "attention" | "none"
//! }
//! → 200 {"vector": [..f32..], "dim": <int>}
//! ```
//!
//! `layer` and `pooling` mirror the shim manifest's `attachment.layer`
//! and `attachment.pooling` fields so the embedded vectors are
//! distributionally identical to what cortex will feed into the shim
//! at inference time.
//!
//! # Status
//!
//! v1 of cortex's shim API does NOT document this endpoint. AgentOS
//! Step 5 builds against this shape using wiremock for tests; production
//! deployment is gated on cortex-claude shipping the endpoint.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::client::check_status;
use crate::error::ShimClientError;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Pooling strategy a shim-trainer requests when extracting hidden
/// states. Mirrors the shim manifest's `attachment.pooling` vocabulary.
///
/// Kept as a thin enum (rather than free-form `String`) so callers
/// can't typo the wire token, and so the cortex-side dispatcher has a
/// closed set to match against. `Custom(String)` escape hatch carries
/// any future value cortex defines without breaking the API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", untagged)]
pub enum Pooling {
    Known(KnownPooling),
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownPooling {
    LastToken,
    Mean,
    Attention,
    None,
}

impl Pooling {
    pub const LAST_TOKEN: Self = Self::Known(KnownPooling::LastToken);
    pub const MEAN: Self = Self::Known(KnownPooling::Mean);
    pub const ATTENTION: Self = Self::Known(KnownPooling::Attention);
    pub const NONE: Self = Self::Known(KnownPooling::None);
}

/// One embedding request.
#[derive(Debug, Clone, Serialize)]
pub struct EmbedRequest {
    /// Text context to embed. Future: token-id input via a different
    /// field (`tokens`) once cortex supports it; for v1 only `context`
    /// (free-form text) is on the wire.
    pub context: String,
    /// Which layer's hidden state to capture. Matches the shim
    /// manifest's `attachment.layer` exactly so trainer and inference
    /// see the same distribution.
    pub layer: String,
    /// How to pool the hidden state into a single vector.
    pub pooling: Pooling,
}

/// Cortex's response to one embed call.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EmbedResponse {
    pub vector: Vec<f32>,
    pub dim: u32,
}

/// HTTP client for cortex's hidden-state extraction surface.
///
/// Holds the same shape as [`crate::CortexShimClient`] (base url +
/// optional bearer + reqwest client). Cheap to clone.
#[derive(Debug, Clone)]
pub struct EmbedClient {
    http: Client,
    base_url: String,
    bearer: Option<String>,
}

impl EmbedClient {
    /// Build a client targeting `base_url` (e.g. `http://cortex:8080`).
    /// Trailing slashes are normalized.
    pub fn new(base_url: impl Into<String>, bearer: Option<String>) -> Self {
        let http = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        Self {
            http,
            base_url: url,
            bearer,
        }
    }

    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.bearer {
            Some(t) => rb.header("authorization", format!("Bearer {t}")),
            None => rb,
        }
    }

    /// Embed one text context into a hidden-state vector.
    ///
    /// `layer` strings come from the shim manifest's `attachment.layer`
    /// vocabulary (`"final"`, `"entrance:N"`); `pooling` likewise.
    pub async fn embed(&self, request: &EmbedRequest) -> Result<EmbedResponse, ShimClientError> {
        let url = format!("{}/v1/embed", self.base_url);

        let resp = self
            .auth(self.http.post(&url))
            .json(request)
            .send()
            .await?;

        let resp = check_status(resp.status(), resp).await?;

        resp.json::<EmbedResponse>()
            .await
            .map_err(|e| ShimClientError::InvalidResponse(format!("embed parse: {e}")))
    }

    /// Convenience: build the request inline.
    pub async fn embed_text(
        &self,
        context: impl Into<String>,
        layer: impl Into<String>,
        pooling: Pooling,
    ) -> Result<EmbedResponse, ShimClientError> {
        self.embed(&EmbedRequest {
            context: context.into(),
            layer: layer.into(),
            pooling,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_serializes_snake_case_for_known_variants() {
        assert_eq!(
            serde_json::to_string(&Pooling::LAST_TOKEN).unwrap(),
            "\"last_token\""
        );
        assert_eq!(serde_json::to_string(&Pooling::MEAN).unwrap(), "\"mean\"");
        assert_eq!(
            serde_json::to_string(&Pooling::ATTENTION).unwrap(),
            "\"attention\""
        );
        assert_eq!(serde_json::to_string(&Pooling::NONE).unwrap(), "\"none\"");
    }

    #[test]
    fn pooling_custom_round_trips() {
        let p = Pooling::Custom("first_token".into());
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"first_token\"");
        let back: Pooling = serde_json::from_str(&json).unwrap();
        match back {
            Pooling::Known(KnownPooling::None) => {} // deserialized as known none for "none"
            Pooling::Custom(s) => assert_eq!(s, "first_token"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn embed_request_serializes() {
        let req = EmbedRequest {
            context: "hello".into(),
            layer: "final".into(),
            pooling: Pooling::LAST_TOKEN,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(json["context"], "hello");
        assert_eq!(json["layer"], "final");
        assert_eq!(json["pooling"], "last_token");
    }

    #[test]
    fn embed_response_deserializes() {
        let body = r#"{"vector": [0.1, 0.2, 0.3], "dim": 3}"#;
        let resp: EmbedResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.dim, 3);
    }

    #[test]
    fn base_url_strips_trailing_slash() {
        let c = EmbedClient::new("http://cortex/", None);
        assert_eq!(c.base_url, "http://cortex");
        let c = EmbedClient::new("http://cortex///", None);
        assert_eq!(c.base_url, "http://cortex");
    }
}
