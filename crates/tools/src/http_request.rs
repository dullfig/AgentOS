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
/// Hard upper bound on `<timeout_secs>`. Caps the unbounded-u64 DoS
/// (security audit H2): an attacker setting timeout_secs = u64::MAX
/// would otherwise hang the tool indefinitely while holding tokio
/// resources and inflating connection-pool pressure.
const MAX_TIMEOUT_SECS: u64 = 120;
const MAX_RESPONSE_BYTES: usize = 1_048_576; // 1 MiB
/// Symmetric request-body cap (security audit H3). Prevents an agent
/// from POSTing a 2 GiB body and tying up tokio + internal services.
const MAX_REQUEST_BYTES: usize = 1_048_576; // 1 MiB

/// Stateless HTTP client tool. Wraps a shared `reqwest::Client` so
/// connection pooling kicks in across invocations.
///
/// ## SSRF protection (security audit B4)
///
/// - **Redirects are disabled** at the client level
///   (`Policy::none()`). A 3xx response is returned verbatim to the
///   agent; if it wants to follow, it must issue a second tool call
///   with the new URL. This eliminates the
///   `attacker.com → 302 → 169.254.169.254` IMDS smuggling path.
/// - **Per-registration host allowlist.** If `allowed_hosts` is
///   non-empty, the URL's host must exactly match one of the entries
///   (literal string compare — no wildcards or CIDR for v1). Empty =
///   no host restriction (back-compat for tests; not for production).
/// - The structural sandbox model intended for v1: each organism
///   that uses http-request registers it with its own allowlist.
///   QA-expert gets `["cortex.local", "agentos.local", "memex.local",
///   "127.0.0.1"]`; a future organism that talks to an external API
///   registers a separate http-request listener with its own list.
#[derive(Clone)]
pub struct HttpRequestTool {
    client: Arc<reqwest::Client>,
    allowed_hosts: Vec<String>,
}

impl HttpRequestTool {
    /// Construct with a default reqwest client (redirects disabled).
    /// Idiomatic in tests and in deployments that don't need custom
    /// TLS / proxy config.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("reqwest client build");
        Self {
            client: Arc::new(client),
            allowed_hosts: Vec::new(),
        }
    }

    /// Construct from a pre-configured client. For deployments that
    /// need custom TLS roots, proxies, etc. **Callers MUST set
    /// `redirect(Policy::none())` on the builder** — otherwise the
    /// SSRF protection from disabling redirects is lost.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client: Arc::new(client),
            allowed_hosts: Vec::new(),
        }
    }

    /// Restrict outbound URLs to this exact-host allowlist. Hosts are
    /// matched literally against `url.host_str()` (post-parse, post-
    /// scheme-strip). No wildcards or CIDR — v1 keeps the surface
    /// small. Set at registration time; matches v1's per-organism
    /// http-request registration pattern.
    pub fn with_allowed_hosts<I, S>(mut self, hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_hosts = hosts.into_iter().map(Into::into).collect();
        self
    }

    fn host_allowed(&self, host: &str) -> bool {
        self.allowed_hosts.is_empty()
            || self.allowed_hosts.iter().any(|h| h == host)
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

        // Host allowlist check. Parse the URL (cheap) to extract the
        // host, then compare literally. Done BEFORE the request fires
        // so blocked hosts never get a connect-attempt.
        let parsed = reqwest::Url::parse(&url)
            .map_err(|e| format!("invalid url `{url}`: {e}"))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| format!("url has no host: {url}"))?;
        if !self.host_allowed(host) {
            return Err(format!(
                "host `{host}` not in allowed-hosts list ({})",
                if self.allowed_hosts.is_empty() {
                    "(empty)".to_string()
                } else {
                    self.allowed_hosts.join(", ")
                }
            ));
        }

        let method_str = extract_tag(xml, "method").unwrap_or_else(|| "GET".to_string());
        let method = parse_method(&method_str)?;

        let timeout_secs = extract_tag(xml, "timeout_secs")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        let headers = match extract_tag(xml, "headers") {
            Some(s) if !s.is_empty() => parse_headers(&s)?,
            _ => HeaderMap::new(),
        };

        let body = extract_tag(xml, "body").unwrap_or_default();
        if body.len() > MAX_REQUEST_BYTES {
            return Err(format!(
                "request body exceeds {MAX_REQUEST_BYTES}-byte cap (got {} bytes)",
                body.len()
            ));
        }
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

    // ── B4 (SSRF) ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn empty_allowed_hosts_accepts_anything() {
        // Back-compat: tools constructed without an allowlist (the
        // default) reach any host. Tests + sandbox demos rely on this.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new();
        let xml = format!("<HttpRequest><url>{}/</url></HttpRequest>", server.uri());
        let (ok, _) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
    }

    #[tokio::test]
    async fn allowed_host_passes() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        // Extract the host (e.g., "127.0.0.1") from the mock URI.
        let uri = server.uri();
        let url = reqwest::Url::parse(&uri).unwrap();
        let host = url.host_str().unwrap().to_string();

        let tool = HttpRequestTool::new().with_allowed_hosts([host]);
        let xml = format!("<HttpRequest><url>{}/</url></HttpRequest>", server.uri());
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "expected success; got: {body}");
    }

    #[tokio::test]
    async fn disallowed_host_rejected_before_request_fires() {
        // The key SSRF property: with an allowlist set, an attacker's
        // attempt to reach a non-listed host must error WITHOUT firing
        // a network request. The mock is set with .expect(0) — any
        // request to it = test failure.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let tool = HttpRequestTool::new()
            .with_allowed_hosts(["cortex.local", "memex.local"]);
        let xml = format!("<HttpRequest><url>{}/</url></HttpRequest>", server.uri());
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok, "disallowed host must be refused: {msg}");
        assert!(msg.contains("not in allowed-hosts"), "got: {msg}");
    }

    #[tokio::test]
    async fn aws_metadata_host_blocked_by_default_allowlist() {
        // QA-expert-style: allowlist is cortex/memex/agentos. AWS IMDS
        // is not in it. Attempt to reach 169.254.169.254 (link-local
        // IMDS) must be refused at the tool layer — no DNS, no
        // connect, no bytes returned.
        let tool = HttpRequestTool::new()
            .with_allowed_hosts(["cortex.local", "memex.local"]);
        let xml = "<HttpRequest><url>http://169.254.169.254/latest/meta-data/iam/security-credentials/</url><timeout_secs>1</timeout_secs></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("not in allowed-hosts"), "got: {msg}");
        // The error must surface BEFORE timeout — i.e., the tool
        // didn't actually try to connect. We can't easily assert
        // "fast" but the message check above is the structural proof.
    }

    #[tokio::test]
    async fn invalid_url_returns_clean_error() {
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest><url>http://[not a url</url></HttpRequest>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("invalid url"), "got: {msg}");
    }

    #[tokio::test]
    async fn timeout_secs_clamped_to_max() {
        // H2 regression: an attacker setting a giant timeout_secs
        // shouldn't be able to hold the tool indefinitely. The actual
        // request to localhost:1 should fail within MAX_TIMEOUT_SECS,
        // not after the requested u64::MAX seconds.
        //
        // We don't wait 120s in the test — we just confirm the request
        // fails (connection refused) within a reasonable wall clock.
        // The cap is structural; the test proves it doesn't panic and
        // doesn't hang forever.
        let tool = HttpRequestTool::new();
        let xml = "<HttpRequest><url>http://127.0.0.1:1/</url>\
                   <timeout_secs>18446744073709551615</timeout_secs></HttpRequest>";
        let start = std::time::Instant::now();
        let (ok, _) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        let elapsed = start.elapsed();
        assert!(!ok);
        // Connection refused returns immediately on most systems;
        // worst case is the bound itself. 130s gives slack.
        assert!(
            elapsed < std::time::Duration::from_secs(130),
            "timeout_secs cap should bound wall clock; elapsed: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn oversized_request_body_rejected() {
        // H3 regression: request body capped at MAX_REQUEST_BYTES.
        // No need for a mock — rejection happens before send.
        let tool = HttpRequestTool::new();
        let big = "x".repeat(2 * 1024 * 1024); // 2 MiB
        let xml = format!(
            "<HttpRequest><url>http://example.test/</url><method>POST</method>\
             <body>{big}</body></HttpRequest>"
        );
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("cap") && msg.contains("body"), "got: {msg}");
    }

    #[tokio::test]
    async fn redirect_not_followed_returns_3xx_verbatim() {
        // The B4 redirect-disable property: a 302 from one host pointing
        // at a different host is NOT followed. The agent sees the 302
        // and the Location header, and must decide what to do.
        //
        // Without this fix:
        //   attacker.com/r → 302 Location: http://169.254.169.254/
        //   reqwest follows → IMDS hit, secrets in response body.
        // With this fix: tool returns the 302 verbatim; reqwest never
        // follows.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/redirect"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", "http://169.254.169.254/secrets"),
            )
            .mount(&server)
            .await;
        // The downstream IMDS path must NEVER be requested by reqwest.
        // We can't mock 169.254 here (it's not on the mock server), so
        // we rely on the redirect-disable property: status==302 means
        // reqwest stopped at the first hop.

        let tool = HttpRequestTool::new();
        let xml = format!(
            "<HttpRequest><url>{}/redirect</url></HttpRequest>",
            server.uri()
        );
        let (ok, body) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "the 302 itself is a successful response; got: {body}");
        let v = parse_ok_json(&body);
        assert_eq!(v["status"], 302, "should surface the 302 verbatim, not follow");
        assert_eq!(
            v["headers"]["location"], "http://169.254.169.254/secrets",
            "location header must be returned for the agent to inspect"
        );
    }
}
