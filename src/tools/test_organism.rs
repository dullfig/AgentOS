//! test-organism tool — smoke-test organism YAML with Haiku and dummy tools.
//!
//! Builds a throwaway pipeline from an organism YAML, replaces all tools with
//! Haiku-backed dummies that generate plausible responses, runs test cases,
//! and reports which tools were called + agent responses + token usage.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::Mutex;
#[allow(unused_imports)]
use rust_pipeline::prelude::build_envelope;

use super::{extract_tag, ToolPeer, ToolResponse};
use super::vdrive_tools::DriveSlot;
use crate::llm::{LlmPool, types::Message};
use crate::organism::parser::parse_organism;
use crate::pipeline::AgentPipelineBuilder;

// ── DummyTool — Haiku-backed tool simulator ──

/// A recorded tool call from the agent under test.
#[derive(Debug, Clone)]
struct DummyCall {
    tool_name: String,
    params_xml: String,
}

/// A tool simulator that records calls and uses Haiku to generate plausible responses.
struct DummyTool {
    tool_name: String,
    description: String,
    pool: Arc<Mutex<LlmPool>>,
    calls: Arc<Mutex<Vec<DummyCall>>>,
}

impl DummyTool {
    fn new(
        tool_name: String,
        description: String,
        pool: Arc<Mutex<LlmPool>>,
        calls: Arc<Mutex<Vec<DummyCall>>>,
    ) -> Self {
        Self { tool_name, description, pool, calls }
    }
}

#[async_trait]
impl Handler for DummyTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let params = String::from_utf8_lossy(&payload.xml).to_string();

        // Record the call
        self.calls.lock().await.push(DummyCall {
            tool_name: self.tool_name.clone(),
            params_xml: params.clone(),
        });

        // Ask Haiku for a plausible response
        let prompt = format!(
            "You are simulating the tool \"{}\" ({}).\n\
             The agent sent this request:\n{}\n\n\
             Return a plausible, short response as if you were this tool. \
             Plain text only, no XML tags, no explanation. 2-5 lines max.",
            self.tool_name, self.description, params
        );

        let response = {
            let pool = self.pool.lock().await;
            pool.complete(
                None, // use default (haiku)
                vec![Message::text("user", &prompt)],
                256,
                Some("You are a tool simulator. Return realistic but brief tool output."),
            ).await
        };

        let result_text = match response {
            Ok(resp) => resp.text().unwrap_or("done").to_string(),
            Err(_) => "done".to_string(), // fallback if Haiku call fails
        };

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&result_text),
        })
    }
}

/// Minimal WIT for a dummy tool — single optional input parameter.
fn make_dummy_wit(tool_name: &str, description: &str) -> String {
    let iface_name = tool_name.replace('-', "_");
    format!(
        r#"/// {description}
interface {iface_name} {{
    record request {{
        /// Input parameters
        input: option<string>,
    }}
    run: func(req: request) -> result<string, string>;
}}"#,
        description = description,
        iface_name = iface_name,
    )
}

#[async_trait]
impl ToolPeer for DummyTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn wit(&self) -> &str {
        // WIT is only used at registration time by the builder.
        // We use a static placeholder — the real WIT is set via DummyToolWithWit wrapper.
        ""
    }
}

/// Wrapper that carries a WIT string alongside the DummyTool.
/// Needed because ToolPeer::wit() returns &str (borrowed from self).
struct DummyToolWithWit {
    inner: DummyTool,
    wit: String,
}

impl DummyToolWithWit {
    fn new(
        tool_name: String,
        description: String,
        pool: Arc<Mutex<LlmPool>>,
        calls: Arc<Mutex<Vec<DummyCall>>>,
    ) -> Self {
        let wit = make_dummy_wit(&tool_name, &description);
        Self {
            inner: DummyTool::new(tool_name, description, pool, calls),
            wit,
        }
    }
}

#[async_trait]
impl Handler for DummyToolWithWit {
    async fn handle(&self, payload: ValidatedPayload, ctx: HandlerContext) -> HandlerResult {
        self.inner.handle(payload, ctx).await
    }
}

#[async_trait]
impl ToolPeer for DummyToolWithWit {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn wit(&self) -> &str {
        &self.wit
    }
}

// ── TestOrganismTool — the main tool ──

pub struct TestOrganismTool {
    slot: DriveSlot,
    /// Parent's LLM pool — reserved for future use (extract API key without env var).
    _parent_pool: Option<Arc<Mutex<LlmPool>>>,
}

impl TestOrganismTool {
    pub fn new(slot: DriveSlot, parent_pool: Option<Arc<Mutex<LlmPool>>>) -> Self {
        Self { slot, _parent_pool: parent_pool }
    }

    /// Create a Haiku-only LLM pool for testing.
    fn make_haiku_pool() -> Result<LlmPool, String> {
        LlmPool::from_env("haiku").map_err(|e| format!("cannot create Haiku pool: {e}"))
    }
}

#[async_trait]
impl Handler for TestOrganismTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Optional: source directory for breadcrumb
        let source_dir = extract_tag(&xml_str, "source_dir");

        // Parse YAML — inline or from file
        let yaml = if let Some(inline) = extract_tag(&xml_str, "yaml") {
            inline
        } else if let Some(path) = extract_tag(&xml_str, "path") {
            let guard = self.slot.read().await;
            match guard.as_ref() {
                Some(drive) => {
                    match drive.read_file(&path, 1, 50_000) {
                        Ok(result) => result.content,
                        Err(e) => return Ok(HandlerResponse::Reply {
                            payload_xml: ToolResponse::err(
                                &format!("failed to read '{}': {}", path, e),
                            ),
                        }),
                    }
                }
                None => return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(
                        "no storage mounted — provide inline <yaml> or mount a workspace",
                    ),
                }),
            }
        } else {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "provide <yaml> (inline) or <path> (file path), plus <test_cases>",
                ),
            });
        };

        // Parse test cases (newline-separated)
        let test_cases_raw = extract_tag(&xml_str, "test_cases")
            .unwrap_or_default();
        let test_cases: Vec<&str> = test_cases_raw
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        if test_cases.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("no <test_cases> provided"),
            });
        }

        // Parse organism
        let organism = match parse_organism(&yaml) {
            Ok(org) => org,
            Err(e) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("organism parse error: {e}")),
            }),
        };

        // Create Haiku pool
        let haiku_pool = match Self::make_haiku_pool() {
            Ok(p) => Arc::new(Mutex::new(p)),
            Err(e) => return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e),
            }),
        };

        // Timeout per test case
        let timeout_secs = extract_tag(&xml_str, "timeout_secs")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60);

        let mut report = Vec::new();
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;

        for (i, test_input) in test_cases.iter().enumerate() {
            report.push(format!("=== Test {} ===", i + 1));
            report.push(format!("Input: \"{}\"", test_input));

            match self.run_test_case(
                &organism,
                &yaml,
                &haiku_pool,
                test_input,
                timeout_secs,
            ).await {
                Ok(result) => {
                    if result.tools_called.is_empty() {
                        report.push("Tools called: (none)".to_string());
                    } else {
                        report.push("Tools called:".to_string());
                        for (j, call) in result.tools_called.iter().enumerate() {
                            // Truncate params for readability
                            let params_short = if call.params_xml.len() > 120 {
                                format!("{}...", &call.params_xml[..120])
                            } else {
                                call.params_xml.clone()
                            };
                            report.push(format!("  {}. {}({})", j + 1, call.tool_name, params_short));
                        }
                    }
                    report.push(format!("Response: {}", result.response));
                    report.push(format!(
                        "Tokens: {} input, {} output",
                        result.input_tokens, result.output_tokens
                    ));
                    report.push(format!("Status: {}", if result.error.is_none() { "OK" } else { "ERROR" }));
                    if let Some(ref err) = result.error {
                        report.push(format!("Error: {}", err));
                    }
                    total_input_tokens += result.input_tokens as u64;
                    total_output_tokens += result.output_tokens as u64;
                }
                Err(e) => {
                    report.push(format!("Status: FAILED"));
                    report.push(format!("Error: {}", e));
                }
            }
            report.push(String::new());
        }

        report.push(format!(
            "Total: {} test(s), {} input tokens, {} output tokens",
            test_cases.len(), total_input_tokens, total_output_tokens
        ));

        // Write breadcrumb if source_dir provided and no failures
        let has_failures = report.iter().any(|line| line.contains("Status: FAILED"));
        if !has_failures {
            if let Some(ref dir) = source_dir {
                let guard = self.slot.read().await;
                if let Some(drive) = guard.as_ref() {
                    let marker_path = format!("{}/.tested", dir);
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let content = format!(
                        "{{\"status\":\"ok\",\"timestamp\":{},\"tests\":{},\"input_tokens\":{},\"output_tokens\":{}}}",
                        timestamp, test_cases.len(), total_input_tokens, total_output_tokens
                    );
                    let _ = drive.write_file(&marker_path, &content);
                }
            }
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&report.join("\n")),
        })
    }
}

/// Result of a single test case.
struct TestResult {
    tools_called: Vec<DummyCall>,
    response: String,
    input_tokens: u32,
    output_tokens: u32,
    error: Option<String>,
}

impl TestOrganismTool {
    /// Run a single test case against the organism.
    async fn run_test_case(
        &self,
        organism: &crate::organism::Organism,
        _yaml: &str,
        haiku_pool: &Arc<Mutex<LlmPool>>,
        test_input: &str,
        timeout_secs: u64,
    ) -> Result<TestResult, String> {
        use agentos_events::PipelineEvent;
        use tokio::sync::broadcast;

        // Create temp directory for ephemeral kernel
        let tempdir = tempfile::TempDir::new()
            .map_err(|e| format!("tempdir: {e}"))?;
        let data_dir = tempdir.path().join("data");

        // Shared call recorder
        let calls: Arc<Mutex<Vec<DummyCall>>> = Arc::new(Mutex::new(Vec::new()));

        // Build child pipeline
        let mut builder = AgentPipelineBuilder::new(organism.clone(), &data_dir);

        // Inject Haiku pool
        builder = builder.with_shared_llm_pool(haiku_pool.clone())?;

        // Register dummy tools for every non-agent, non-llm-pool listener
        let listeners = organism.listeners();
        for (name, def) in listeners {
            if def.is_agent || name == "llm-pool" || name == "librarian" || name == "codebase-index" {
                continue;
            }
            let dummy = DummyToolWithWit::new(
                name.to_string(),
                def.description.clone(),
                haiku_pool.clone(),
                calls.clone(),
            );
            builder = builder.register_tool(name, dummy)?;
        }

        // Wire agents (uses the organism's agent config)
        builder = builder.with_agents()?;

        // Build pipeline
        let mut pipeline = builder.build()?;

        // Subscribe to events
        let mut rx = pipeline.subscribe();

        // Initialize root thread
        let profiles: Vec<String> = pipeline
            .organism()
            .profile_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let profile = profiles
            .first()
            .ok_or("organism has no profiles")?;
        let root_uuid = pipeline.initialize_root("test-harness", profile).await?;

        // Start pipeline
        pipeline.run();

        // Build and inject the test task envelope
        let agent_name = pipeline
            .organism()
            .agent_listeners()
            .into_iter()
            .next()
            .map(|a| a.name.clone())
            .ok_or("organism has no agent listener")?;

        let task_xml = format!(
            "<AgentTask><task>{}</task></AgentTask>",
            crate::tools::xml_escape(test_input)
        );
        let envelope = build_envelope(
            "test-harness",
            &agent_name,
            &root_uuid,
            task_xml.as_bytes(),
        )
        .map_err(|e| format!("envelope: {e}"))?;
        pipeline.inject_raw(envelope).await?;

        // Collect events until AgentResponse or timeout
        let mut response_text = String::new();
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut error = None;

        let timeout = Duration::from_secs(timeout_secs);
        let result = tokio::time::timeout(timeout, async {
            loop {
                match rx.recv().await {
                    Ok(PipelineEvent::AgentResponse { text, .. }) => {
                        response_text = text.clone();
                        if text.starts_with("Error: ") {
                            error = Some(text.clone());
                        }
                        return;
                    }
                    Ok(PipelineEvent::TokenUsage {
                        input_tokens: inp,
                        output_tokens: out,
                        ..
                    }) => {
                        input_tokens += inp;
                        output_tokens += out;
                    }
                    Ok(_) => {} // skip other events
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {
                        error = Some("event channel closed".into());
                        return;
                    }
                }
            }
        })
        .await;

        if result.is_err() {
            error = Some(format!("timeout after {timeout_secs}s"));
        }

        // Truncate response for report
        let response = if response_text.len() > 500 {
            format!("{}...", &response_text[..500])
        } else {
            response_text
        };

        let recorded_calls = calls.lock().await.clone();

        Ok(TestResult {
            tools_called: recorded_calls,
            response,
            input_tokens,
            output_tokens,
            error,
        })
    }
}

#[async_trait]
impl ToolPeer for TestOrganismTool {
    fn name(&self) -> &str {
        "test-organism"
    }

    fn wit(&self) -> &str {
        r#"
/// Smoke-test an organism YAML by running test cases through a throwaway pipeline with Haiku and dummy tools. Returns which tools were called, the agent's response, and token counts. Each dummy tool uses Haiku to generate plausible responses.
interface test_organism {
    record request {
        /// Path to organism YAML file
        path: option<string>,
        /// Inline organism YAML
        yaml: option<string>,
        /// Newline-separated test case inputs to send to the agent
        test-cases: string,
        /// Timeout per test case in seconds (default: 60)
        timeout-secs: option<u32>,
        /// Source directory to write .tested breadcrumb on success
        source-dir: option<string>,
    }
    test: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vdrive_tools::empty_slot;

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "TestOrganismRequest".into(),
        }
    }

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            from: "test".into(),
            own_name: "test-organism".into(),
            thread_id: "test-thread".into(),
        }
    }

    #[tokio::test]
    async fn missing_yaml_and_path_returns_error() {
        let tool = TestOrganismTool::new(empty_slot(), None);
        let xml = "<TestOrganismRequest><test_cases>hello</test_cases></TestOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "expected error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_test_cases_returns_error() {
        let tool = TestOrganismTool::new(empty_slot(), None);
        let yaml = "organism:\n  name: test";
        let xml = format!(
            "<TestOrganismRequest><yaml>{}</yaml></TestOrganismRequest>",
            crate::tools::xml_escape(yaml)
        );
        let result = tool.handle(make_payload(&xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("no &lt;test_cases&gt;"), "expected test_cases error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn invalid_yaml_returns_parse_error() {
        let tool = TestOrganismTool::new(empty_slot(), None);
        let xml = "<TestOrganismRequest>\
            <yaml>not: valid: [}</yaml>\
            <test_cases>hello</test_cases>\
            </TestOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("parse error"), "expected parse error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn dummy_tool_records_calls() {
        let pool = Arc::new(Mutex::new(
            LlmPool::new("test-key".into(), "haiku"),
        ));
        let calls: Arc<Mutex<Vec<DummyCall>>> = Arc::new(Mutex::new(Vec::new()));
        let tool = DummyTool::new(
            "file-read".into(),
            "Read files".into(),
            pool,
            calls.clone(),
        );

        let payload = ValidatedPayload {
            xml: b"<FileReadRequest><path>src/main.rs</path></FileReadRequest>".to_vec(),
            tag: "FileReadRequest".into(),
        };
        let ctx = HandlerContext {
            from: "test-agent".into(),
            own_name: "file-read".into(),
            thread_id: "t1".into(),
        };

        // Will fail the Haiku call (fake key) but should still record + return fallback
        let result = tool.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>true</success>"), "expected success: {s}");
            }
            _ => panic!("expected Reply"),
        }

        let recorded = calls.lock().await;
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].tool_name, "file-read");
        assert!(recorded[0].params_xml.contains("src/main.rs"));
    }

    #[tokio::test]
    async fn make_dummy_wit_valid() {
        let wit = make_dummy_wit("file-read", "Read files from disk");
        assert!(wit.contains("interface file_read"));
        assert!(wit.contains("Read files from disk"));
        assert!(wit.contains("input: option<string>"));
    }
}
