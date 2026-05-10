//! CortexShimTool — registry CRUD + standalone classification against
//! cortex's `/v1/shims/...` endpoints.
//!
//! Single tool exposing all shim-management operations dispatched by an
//! `<action>` tag. Mirrors the codebase-index pattern (one tool, multiple
//! actions) so the LLM has a clear surface for the lifecycle work the
//! shim-expert agent does.
//!
//! Configuration: pass a configured `CortexShimClient` at construction
//! (the AgentPipelineBuilder does this at register-tool time, sourcing
//! `base_url` + bearer from deployment config).

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde_json::json;

use agentos_cortex_shim::{CortexShimClient, ShimClientError, ShimManifest};

use super::{extract_tag, ToolPeer, ToolResponse};

/// Tool wrapper around `CortexShimClient`.
#[derive(Clone)]
pub struct CortexShimTool {
    client: Arc<CortexShimClient>,
}

impl CortexShimTool {
    /// Construct from a pre-configured client. The client holds the
    /// cortex base URL and bearer; AgentPipelineBuilder sets these
    /// once per deployment.
    pub fn new(client: CortexShimClient) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    async fn handle_register(&self, xml_str: &str) -> Result<String, String> {
        let manifest_json = extract_tag(xml_str, "manifest")
            .ok_or_else(|| "missing required <manifest>".to_string())?;
        let onnx_path = extract_tag(xml_str, "onnx_path")
            .ok_or_else(|| "missing required <onnx_path>".to_string())?;

        let manifest: ShimManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| format!("manifest is not valid JSON: {e}"))?;

        let bytes = std::fs::read(PathBuf::from(&onnx_path))
            .map_err(|e| format!("read {onnx_path}: {e}"))?;

        match self.client.register(&manifest, bytes).await {
            Ok(()) => Ok(json!({
                "registered": manifest.id,
                "version": manifest.version,
            })
            .to_string()),
            Err(e) => Err(format!("register failed: {e}")),
        }
    }

    async fn handle_list(&self) -> Result<String, String> {
        match self.client.list().await {
            Ok(summaries) => serde_json::to_string(&summaries)
                .map_err(|e| format!("serialize list: {e}")),
            Err(e) => Err(format!("list failed: {e}")),
        }
    }

    async fn handle_get(&self, xml_str: &str) -> Result<String, String> {
        let id = extract_tag(xml_str, "id")
            .ok_or_else(|| "missing required <id>".to_string())?;
        match self.client.get(&id).await {
            Ok(manifest) => serde_json::to_string(&manifest)
                .map_err(|e| format!("serialize manifest: {e}")),
            Err(ShimClientError::NotFound(_)) => Err(format!("shim not found: {id}")),
            Err(e) => Err(format!("get failed: {e}")),
        }
    }

    async fn handle_delete(&self, xml_str: &str) -> Result<String, String> {
        let id = extract_tag(xml_str, "id")
            .ok_or_else(|| "missing required <id>".to_string())?;
        match self.client.delete(&id).await {
            Ok(()) => Ok(json!({"deleted": id}).to_string()),
            Err(ShimClientError::NotFound(_)) => Err(format!("shim not found: {id}")),
            Err(e) => Err(format!("delete failed: {e}")),
        }
    }

    async fn handle_infer(&self, xml_str: &str) -> Result<String, String> {
        let id = extract_tag(xml_str, "id")
            .ok_or_else(|| "missing required <id>".to_string())?;
        let context_raw = extract_tag(xml_str, "context")
            .ok_or_else(|| "missing required <context>".to_string())?;

        // Context is a JSON value: parse it. Wraps strings/numbers/objects
        // — the shim's manifest dictates the shape cortex accepts.
        let context: serde_json::Value = serde_json::from_str(&context_raw)
            .unwrap_or_else(|_| serde_json::Value::String(context_raw));

        match self.client.infer(&id, context).await {
            Ok(decision) => serde_json::to_string(&decision)
                .map_err(|e| format!("serialize decision: {e}")),
            Err(ShimClientError::NotFound(_)) => Err(format!("shim not found: {id}")),
            Err(e) => Err(format!("infer failed: {e}")),
        }
    }
}

#[async_trait]
impl Handler for CortexShimTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let action = extract_tag(&xml_str, "action").unwrap_or_default();
        let result: Result<String, String> = match action.as_str() {
            "register" => self.handle_register(&xml_str).await,
            "list" => self.handle_list().await,
            "get" => self.handle_get(&xml_str).await,
            "delete" => self.handle_delete(&xml_str).await,
            "infer" => self.handle_infer(&xml_str).await,
            "" => Err("missing required <action>".into()),
            other => Err(format!(
                "unknown action: {other} (allowed: register|list|get|delete|infer)"
            )),
        };

        let payload_xml = match result {
            Ok(body) => ToolResponse::ok(&body),
            Err(msg) => ToolResponse::err(&msg),
        };
        Ok(HandlerResponse::Reply { payload_xml })
    }
}

#[async_trait]
impl ToolPeer for CortexShimTool {
    fn name(&self) -> &str {
        "cortex-shim"
    }

    fn wit(&self) -> &str {
        r#"
/// Manage cortex shims: register a new shim, list/get/delete existing
/// ones, or run a registered shim against a free-form context for
/// standalone classification (no generation).
interface cortex-shim {
    record request {
        /// register | list | get | delete | infer
        action: string,
        /// shim id (required for get/delete/infer/register's response handling)
        id: option<string>,
        /// JSON-serialized ShimManifest (required for register)
        manifest: option<string>,
        /// path to the ONNX file on disk (required for register)
        onnx-path: option<string>,
        /// JSON-serialized context value (required for infer)
        context: option<string>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "shim-expert".into(),
            own_name: "cortex-shim".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "CortexShim".into(),
        }
    }

    fn parse_response(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let body = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, body)
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn list_returns_summaries() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/shims/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([
                {"id": "should_respond", "version": "0.3.1", "phase": "gate"}
            ])))
            .mount(&server)
            .await;

        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = "<CortexShim><action>list</action></CortexShim>";
        let resp = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        let (ok, body) = parse_response(resp);
        assert!(ok, "expected success, got: {body}");
        assert!(body.contains("should_respond"), "body: {body}");
    }

    #[tokio::test]
    async fn missing_action_errors() {
        let server = MockServer::start().await;
        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = "<CortexShim></CortexShim>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("missing required <action>"));
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let server = MockServer::start().await;
        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = "<CortexShim><action>noop</action></CortexShim>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("unknown action"));
    }

    #[tokio::test]
    async fn get_404_maps_to_not_found_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/shims/missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = "<CortexShim><action>get</action><id>missing</id></CortexShim>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("shim not found"));
        assert!(msg.contains("missing"));
    }

    #[tokio::test]
    async fn delete_round_trips() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/shims/voice_bob"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = "<CortexShim><action>delete</action><id>voice_bob</id></CortexShim>";
        let (ok, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("voice_bob"));
    }

    #[tokio::test]
    async fn infer_passes_context_through() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/shims/infer"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "decision": 0.91,
                "metadata": {}
            })))
            .mount(&server)
            .await;

        let tool = CortexShimTool::new(CortexShimClient::new(server.uri(), None));
        let xml = r#"<CortexShim>
            <action>infer</action>
            <id>should_respond</id>
            <context>"is bob there?"</context>
        </CortexShim>"#;
        let (ok, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("0.91"));
    }
}
