//! HttpRequestTool — issue HTTP/HTTPS requests, return status + headers + body.
//!
//! Built for QA-expert (Track 3 of the roadmap) — scraping `/metrics`
//! endpoints on cortex, agentos, memex. Also general-purpose: any
//! organism that needs to call out to an HTTP service.
//!
//! ## Scope (v1)
//! - Methods: GET, HEAD, POST, PUT, DELETE, PATCH.
//! - Headers: optional JSON object encoded in `<headers>`.
//! - Body: optional text in `<body>`. Ignored for GET/HEAD.
//! - Timeout: `<timeout_secs>`, default 30.
//! - Response: JSON `{"status": int, "headers": {...}, "body": string}`
//!   wrapped in the standard `<ToolResponse>` envelope.
//! - Response body cap: 1 MiB. Larger responses error rather than truncate
//!   (silent truncation hides bugs; explicit failure prompts the agent
//!   to use streaming or pagination).
//!
//! ## Sandboxing
//! The tool itself accepts any URL with `http://` or `https://` scheme.
//! Sandboxing is delegated to the YAML port declarations on the listener
//! (see existing pattern: `ports: [{port: 80, direction: outbound,
//! hosts: [cortex.local]}]`). If we need belt-and-suspenders at the
//! tool layer later, add an allowlist field to the constructor.
//!
//! ## TLS
//! reqwest is configured with the workspace's `rustls-tls` feature.
//! Server certificate verification is always on — no insecure-mode knob.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Method;
use rust_pipeline::prelude::*;
use serde_json::{json, Value};

use super::{extract_tag, ToolPeer, ToolResponse};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_RESPONSE_BYTES: usize = 1_048_576; // 1 MiB

/// Stateless HTTP client tool. Wraps a shared `reqwest::Client` so
/// connection pooling kicks in across invocations.
#[derive(Clone)]
pub struct HttpRequestTool {
    client: Arc<reqwest::Client>,
}

impl HttpRequestTool {
    /// Construct with a default reqwest client. Idiomatic in tests and
    /// in deployments that don't need custom TLS / proxy config.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest client build");
        Self {
            client: Arc::new(client),
        }
    }

    /// Construct from a pre-configured client. For deployments that
    /// need custom TLS roots, proxies, etc.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    async fn execute(&self, xml: &str) -> Result<String, String> {
        let url = extract_tag(xml, "url")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing required <url>".to_string())?;

        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(format!(
                "url scheme must be http:// or https://: {url}"
            ));
        }

        let method_str = extract_tag(xml, "method").unwrap_or_else(|| "GET".to_string());
        let method = parse_method(&method_str)?;

        let timeout_secs = extract_tag(xml, "timeout_secs")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let headers = match extract_tag(xml, "headers") {
            Some(s) if !s.is_empty() => parse_headers(&s)?,
            _ => HeaderMap::new(),
        };

        let body = extract_tag(xml, "body").unwrap_or_default();
        let body_allowed = !matches!(method, Method::GET | Method::HEAD);

        let mut req = self
            .client
            .request(method.clone(), &url)
            .timeout(Duration::from_secs(timeout_secs));
        if !headers.is_empty() {
            req = req.headers(headers);
        }
        if !body.is_empty() && body_allowed {
            req = req.body(body);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status().as_u16();
        let resp_headers: serde_json::Map<String, Value> = resp
            .headers()
            .iter()
            .map(|(k, v)| {
                let val = v.to_str().unwrap_or("").to_string();
                (k.as_str().to_lowercase(), Value::String(val))
            })
            .collect();

        let body_bytes = read_bounded(resp).await?;
        let body_text = String::from_utf8_lossy(&body_bytes).to_string();

        let payload = json!({
            "status": status,
            "headers": resp_headers,
            "body": body_text,
        });

        Ok(payload.to_string())
    }
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_method(s: &str) -> Result<Method, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "GET" => Ok(Method::GET),
        "HEAD" => Ok(Method::HEAD),
        "POST" => Ok(Method::POST),
        "PUT" => Ok(Method::PUT),
        "DELETE" => Ok(Method::DELETE),
        "PATCH" => Ok(Method::PATCH),
        other => Err(format!("unsupported HTTP method: {other}")),
    }
}

fn parse_headers(json_str: &str) -> Result<HeaderMap, String> {
    let v: Value = serde_json::from_str(json_str)
        .map_err(|e| format!("<headers> must be a JSON object: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| "<headers> must be a JSON object".to_string())?;
    let mut map = HeaderMap::new();
    for (k, v) in obj {
        let val = v
            .as_str()
            .ok_or_else(|| format!("header value for '{k}' must be a string"))?;
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| format!("invalid header name '{k}': {e}"))?;
        let value = HeaderValue::from_str(val)
            .map_err(|e| format!("invalid header value for '{k}': {e}"))?;
        map.insert(name, value);
    }
    Ok(map)
}

/// Stream the response body with a hard byte cap. Fails fast on cap
/// breach so the agent doesn't silently get truncated metrics.
async fn read_bounded(resp: reqwest::Response) -> Result<Vec<u8>, String> {
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| format!("body read: {e}"))?;
        if buf.len() + bytes.len() > MAX_RESPONSE_BYTES {
            return Err(format!(
                "response body exceeds {MAX_RESPONSE_BYTES}-byte cap"
            ));
        }
        buf.extend_from_slice(&bytes);
    }
    Ok(buf)
}

#[async_trait]
impl Handler for HttpRequestTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml = String::from_utf8_lossy(&payload.xml);
        match self.execute(&xml).await {
            Ok(body) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&body),
            }),
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for HttpRequestTool {
    fn name(&self) -> &str {
        "http-request"
    }

    fn wit(&self) -> &str {
        r#"
/// Issue an HTTP/HTTPS request and return status, headers, and body.
///
/// Returns JSON: {"status": int, "headers": {...}, "body": string}.
/// Response body capped at 1 MiB. Default timeout 30s. GET and HEAD
/// ignore the body field. Methods supported: GET, HEAD, POST, PUT,
/// DELETE, PATCH.
interface http-request {
    record request {
        /// Absolute URL with http:// or https:// scheme.
        url: string,
        /// HTTP method. Defaults to GET.
        method: option<string>,
        /// JSON object of request headers, e.g. {"Accept": "text/plain"}.
        headers: option<string>,
        /// Request body. Ignored for GET and HEAD.
        body: option<string>,
        /// Per-request timeout in seconds. Defaults to 30.
        timeout-secs: option<u64>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use wiremock::matchers::{body_string, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "http-request".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "HttpRequest".into(),
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

    /// Parse a successful tool response body as JSON. Panics on failure
    /// — the response shape is part of the contract, so an unparseable
    /// body is a bug.
    fn parse_ok_json(body: &str) -> Value {
        serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("response body is not JSON: {e}\nbody: {body}"))
    }

    #[tokio::test]
    async fn get_returns_status_headers_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/metrics"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/plain; version=0.0.4")
                    .set_body_string(
                        "# HELP agentos_requests_total ok\nagentos_requests_total 42\n",
                    ),
            )
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!(
            "<HttpRequest><url>{}/metrics</url></HttpRequest>",
            server.uri()
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "expected success; got: {body}");

        let v = parse_ok_json(&body);
        assert_eq!(v["status"], 200);
        assert!(
            v["headers"]["content-type"]
                .as_str()
                .unwrap_or("")
                .starts_with("text/plain"),
            "content-type missing: {v}"
        );
        assert!(
            v["body"].as_str().unwrap().contains("agentos_requests_total 42"),
            "body missing metric: {v}"
        );
    }

    #[tokio::test]
    async fn method_defaults_to_get() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!("<HttpRequest><url>{}/</url></HttpRequest>", server.uri());
        let (ok, _) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
    }

    #[tokio::test]
    async fn post_with_body_and_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/echo"))
            .and(header("content-type", "application/json"))
            .and(body_string(r#"{"hello":"world"}"#))
            .respond_with(ResponseTemplate::new(201).set_body_string("created"))
            .expect(1)
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!(
            r#"<HttpRequest><url>{}/echo</url><method>POST</method><headers>{{"Content-Type": "application/json"}}</headers><body>{{"hello":"world"}}</body></HttpRequest>"#,
            server.uri()
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        let v = parse_ok_json(&body);
        assert_eq!(v["status"], 201);
        assert_eq!(v["body"], "created");
    }

    #[tokio::test]
    async fn http_error_status_is_surfaced_not_failed() {
        // 4xx/5xx responses are NOT tool failures — the agent might
        // want to react to a 503, retry, etc. Surface the status and
        // body via the success path.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!(
            "<HttpRequest><url>{}/missing</url></HttpRequest>",
            server.uri()
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "tool should not error on HTTP 4xx; got: {body}");
        let v = parse_ok_json(&body);
        assert_eq!(v["status"], 404);
        assert_eq!(v["body"], "not found");
    }

    #[tokio::test]
    async fn missing_url_errors() {
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<url>"), "got: {msg}");
    }

    #[tokio::test]
    async fn invalid_scheme_errors() {
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest><url>file:///etc/passwd</url></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("scheme"), "got: {msg}");
    }

    #[tokio::test]
    async fn invalid_method_errors() {
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest><url>http://x/</url><method>TEAPOT</method></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("unsupported HTTP method"), "got: {msg}");
    }

    #[tokio::test]
    async fn malformed_headers_json_errors() {
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest><url>http://x/</url><headers>not-json</headers></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("JSON"), "got: {msg}");
    }

    #[tokio::test]
    async fn oversized_response_body_errors() {
        // wiremock can set a body of arbitrary size; 2 MiB exceeds the cap.
        let big = "x".repeat(2 * 1024 * 1024);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(ResponseTemplate::new(200).set_body_string(big))
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!(
            "<HttpRequest><url>{}/big</url></HttpRequest>",
            server.uri()
        );
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("cap"), "got: {msg}");
    }

    #[tokio::test]
    async fn body_ignored_on_get() {
        // GET requests with a `<body>` shouldn't send a body — wiremock
        // should match a body-less GET. We assert this by setting up
        // the mock with a body matcher and checking the empty case.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .and(body_string(""))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!(
            "<HttpRequest><url>{}/</url><body>this should be dropped</body></HttpRequest>",
            server.uri()
        );
        let (ok, _) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
    }

    #[tokio::test]
    async fn connection_refused_is_a_tool_error_not_panic() {
        // Port 1 typically isn't listening; the connect should fail
        // fast and produce a tool error rather than panic.
        let tool = HttpRequestTool::new();
        let xml =
            "<HttpRequest><url>http://127.0.0.1:1/</url><timeout_secs>1</timeout_secs></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(
            msg.contains("request failed"),
            "expected request failed error; got: {msg}"
        );
    }
}
