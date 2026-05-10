//! CortexEmbedTool — turn a text context into a hidden-state vector
//! via cortex's `POST /v1/embed`.
//!
//! Used by the shim-expert agent during training-data preparation:
//! each `(text, label)` example gets embedded against the same model
//! the shim will eventually attach to, so the trained FFN sees the
//! same vector distribution at inference time.

use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde_json::json;

use agentos_cortex_shim::{EmbedClient, Pooling};

use super::{extract_tag, ToolPeer, ToolResponse};

#[derive(Clone)]
pub struct CortexEmbedTool {
    client: Arc<EmbedClient>,
}

impl CortexEmbedTool {
    pub fn new(client: EmbedClient) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    fn parse_pooling(raw: &str) -> Pooling {
        match raw {
            "last_token" => Pooling::LAST_TOKEN,
            "mean" => Pooling::MEAN,
            "attention" => Pooling::ATTENTION,
            "none" => Pooling::NONE,
            other => Pooling::Custom(other.to_string()),
        }
    }
}

#[async_trait]
impl Handler for CortexEmbedTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let context = match extract_tag(&xml_str, "context") {
            Some(c) if !c.is_empty() => c,
            _ => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("missing required <context>"),
                });
            }
        };
        let layer = extract_tag(&xml_str, "layer").unwrap_or_else(|| "final".to_string());
        let pooling_raw =
            extract_tag(&xml_str, "pooling").unwrap_or_else(|| "last_token".to_string());
        let pooling = Self::parse_pooling(&pooling_raw);

        match self.client.embed_text(context, layer, pooling).await {
            Ok(resp) => {
                let body = json!({
                    "vector": resp.vector,
                    "dim": resp.dim,
                })
                .to_string();
                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&body),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("embed failed: {e}")),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for CortexEmbedTool {
    fn name(&self) -> &str {
        "cortex-embed"
    }

    fn wit(&self) -> &str {
        r#"
/// Embed a text context into a hidden-state vector via cortex.
/// Layer + pooling must match the shim manifest's attachment so the
/// trainer and inference see the same distribution.
interface cortex-embed {
    record request {
        /// Free-form text to embed.
        context: string,
        /// "final" | "entrance:N"; defaults to "final" if omitted.
        layer: option<string>,
        /// "last_token" | "mean" | "attention" | "none"; defaults to "last_token".
        pooling: option<string>,
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
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "shim-expert".into(),
            own_name: "cortex-embed".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "CortexEmbed".into(),
        }
    }

    fn parse(resp: HandlerResponse) -> (bool, String) {
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
    async fn embed_round_trips_through_tool() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embed"))
            .and(body_partial_json(json!({
                "context": "hello",
                "layer": "final",
                "pooling": "last_token"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "vector": [0.1, 0.2, 0.3],
                "dim": 3
            })))
            .mount(&server)
            .await;

        let tool = CortexEmbedTool::new(EmbedClient::new(server.uri(), None));
        let xml = "<CortexEmbed><context>hello</context></CortexEmbed>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("\"dim\":3"), "body: {body}");
        assert!(body.contains("0.1"), "body: {body}");
    }

    #[tokio::test]
    async fn missing_context_errors() {
        let server = MockServer::start().await;
        let tool = CortexEmbedTool::new(EmbedClient::new(server.uri(), None));
        let xml = "<CortexEmbed></CortexEmbed>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<context>"));
    }

    #[tokio::test]
    async fn explicit_layer_and_pooling_pass_through() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embed"))
            .and(body_partial_json(json!({
                "context": "hi",
                "layer": "entrance:3",
                "pooling": "mean"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "vector": [0.0, 0.0],
                "dim": 2
            })))
            .expect(1)
            .mount(&server)
            .await;

        let tool = CortexEmbedTool::new(EmbedClient::new(server.uri(), None));
        let xml = "<CortexEmbed><context>hi</context><layer>entrance:3</layer><pooling>mean</pooling></CortexEmbed>";
        let (ok, _) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok);
    }

    #[tokio::test]
    async fn cortex_error_propagates_as_tool_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embed"))
            .respond_with(ResponseTemplate::new(503).set_body_string("loading"))
            .mount(&server)
            .await;

        let tool = CortexEmbedTool::new(EmbedClient::new(server.uri(), None));
        let xml = "<CortexEmbed><context>x</context></CortexEmbed>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("embed failed"));
    }
}
