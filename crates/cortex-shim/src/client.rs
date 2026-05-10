//! HTTP client for cortex's shim-management surface.
//!
//! Endpoints (per `project_cortex_v1_shim_api.md`):
//!
//! | Method | Path                | Purpose                                  |
//! |--------|---------------------|------------------------------------------|
//! | `PUT`  | `/v1/shims/{id}`    | Upload manifest + ONNX bytes             |
//! | `GET`  | `/v1/shims/`        | List registered shims                    |
//! | `GET`  | `/v1/shims/{id}`    | Fetch one manifest                       |
//! | `DELETE` | `/v1/shims/{id}`  | Unregister                               |
//! | `POST` | `/v1/shims/infer`   | Standalone classification (no generation) |
//!
//! `PUT` uses `multipart/form-data` with a `manifest` JSON part and an
//! `onnx` binary part. This matches cortex's own loader contract while
//! keeping the wire format human-debuggable from `curl`.

use std::time::Duration;

use reqwest::{multipart, Client, StatusCode};
use serde::Serialize;

use crate::error::ShimClientError;
use crate::manifest::{ShimDecision, ShimManifest, ShimSummary};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// HTTP client for cortex's shim registry + standalone classifier.
///
/// Cheap to clone (`reqwest::Client` is internally reference-counted).
/// Holds a single base URL + bearer; multi-tenant usage that needs
/// per-tenant credentials should hold one client per tenant.
#[derive(Debug, Clone)]
pub struct CortexShimClient {
    http: Client,
    base_url: String,
    bearer: Option<String>,
}

impl CortexShimClient {
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

    /// Apply bearer auth to a request builder, when configured.
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.bearer {
            Some(t) => rb.header("authorization", format!("Bearer {t}")),
            None => rb,
        }
    }

    /// Register (or replace) a shim. Cortex hot-loads it on success;
    /// subsequent chat-completion calls referencing the shim id will
    /// pick it up.
    pub async fn register(
        &self,
        manifest: &ShimManifest,
        onnx: Vec<u8>,
    ) -> Result<(), ShimClientError> {
        let url = format!("{}/v1/shims/{}", self.base_url, manifest.id);

        let manifest_json = serde_json::to_string(manifest)
            .map_err(|e| ShimClientError::InvalidManifest(e.to_string()))?;

        let form = multipart::Form::new()
            .text("manifest", manifest_json)
            .part(
                "onnx",
                multipart::Part::bytes(onnx)
                    .file_name(format!("{}.onnx", manifest.id))
                    .mime_str("application/octet-stream")
                    .map_err(|e| ShimClientError::InvalidResponse(e.to_string()))?,
            );

        let resp = self
            .auth(self.http.put(&url))
            .multipart(form)
            .send()
            .await?;

        check_status(resp.status(), resp).await.map(|_| ())
    }

    /// List all registered shim manifests.
    pub async fn list(&self) -> Result<Vec<ShimSummary>, ShimClientError> {
        let url = format!("{}/v1/shims/", self.base_url);
        let resp = self.auth(self.http.get(&url)).send().await?;
        let resp = check_status(resp.status(), resp).await?;
        resp.json::<Vec<ShimSummary>>()
            .await
            .map_err(|e| ShimClientError::InvalidResponse(format!("list parse: {e}")))
    }

    /// Fetch one shim's manifest by id.
    pub async fn get(&self, id: &str) -> Result<ShimManifest, ShimClientError> {
        let url = format!("{}/v1/shims/{}", self.base_url, id);
        let resp = self.auth(self.http.get(&url)).send().await?;

        if resp.status() == StatusCode::NOT_FOUND {
            return Err(ShimClientError::NotFound(id.to_string()));
        }
        let resp = check_status(resp.status(), resp).await?;

        resp.json::<ShimManifest>()
            .await
            .map_err(|e| ShimClientError::InvalidResponse(format!("manifest parse: {e}")))
    }

    /// Unregister a shim. Idempotent: 404 maps to `NotFound`, callers
    /// can choose to treat that as success.
    pub async fn delete(&self, id: &str) -> Result<(), ShimClientError> {
        let url = format!("{}/v1/shims/{}", self.base_url, id);
        let resp = self.auth(self.http.delete(&url)).send().await?;

        if resp.status() == StatusCode::NOT_FOUND {
            return Err(ShimClientError::NotFound(id.to_string()));
        }
        check_status(resp.status(), resp).await.map(|_| ())
    }

    /// Run a shim against a free-form context, without generation.
    ///
    /// Used for shims that gate decisions outside the chat-completion
    /// flow — e.g. an ambient-listener pass deciding which posts are
    /// archive-worthy. The body shape follows the cortex spec:
    ///
    /// ```json
    /// {"shim_id": "...", "context": <opaque>}
    /// ```
    ///
    /// `context` is opaque on the wire — typically a string, but may
    /// be tokens or an embedding depending on the shim's input_shape.
    pub async fn infer(
        &self,
        shim_id: &str,
        context: serde_json::Value,
    ) -> Result<ShimDecision, ShimClientError> {
        let url = format!("{}/v1/shims/infer", self.base_url);

        #[derive(Serialize)]
        struct InferRequest<'a> {
            shim_id: &'a str,
            context: serde_json::Value,
        }

        let resp = self
            .auth(self.http.post(&url))
            .json(&InferRequest { shim_id, context })
            .send()
            .await?;

        if resp.status() == StatusCode::NOT_FOUND {
            return Err(ShimClientError::NotFound(shim_id.to_string()));
        }
        let resp = check_status(resp.status(), resp).await?;

        resp.json::<ShimDecision>()
            .await
            .map_err(|e| ShimClientError::InvalidResponse(format!("infer parse: {e}")))
    }
}

/// Map HTTP errors into `ShimClientError`. Returns the response when OK.
async fn check_status(
    status: StatusCode,
    resp: reqwest::Response,
) -> Result<reqwest::Response, ShimClientError> {
    if status.is_success() {
        return Ok(resp);
    }
    let code = status.as_u16();
    let body = resp.text().await.unwrap_or_else(|_| "(no body)".into());
    Err(ShimClientError::ApiError {
        status: code,
        message: body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_strips_trailing_slash() {
        let c = CortexShimClient::new("http://cortex/", None);
        assert_eq!(c.base_url, "http://cortex");
        let c = CortexShimClient::new("http://cortex///", None);
        assert_eq!(c.base_url, "http://cortex");
        let c = CortexShimClient::new("http://cortex", None);
        assert_eq!(c.base_url, "http://cortex");
    }
}
