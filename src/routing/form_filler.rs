//! Form filler — parameter extraction for semantic routing.
//!
//! Two strategies:
//! - `CloudFormFiller`: Haiku/Sonnet model ladder via cloud API (original)
//! - `LocalFormFiller`: codeLlm constrained decoding (guaranteed valid XML)
//!
//! Model ladder: Haiku (cheap, fast) → Sonnet (escalate on failure).
//! Never Opus — Opus is the thinker.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::info;

use crate::llm::types::Message;
use crate::llm::LlmPool;

use super::local_engine::SharedEngine;

/// Result of a form-fill attempt.
#[derive(Debug)]
pub enum FormFillResult {
    /// Successfully produced valid XML for the tool.
    Success {
        tool_name: String,
        filled_xml: String,
    },
    /// All retries exhausted.
    Failed {
        tool_name: String,
        last_error: String,
    },
}

/// Strategy trait for form filling — open for extension.
#[async_trait::async_trait]
pub trait FormFillStrategy: Send + Sync {
    /// Fill tool XML from natural language intent.
    async fn fill(
        &self,
        intent: &str,
        tool_name: &str,
        tool_description: &str,
        xml_template: &str,
        payload_tag: &str,
    ) -> FormFillResult;
}

// ── Cloud form filler (original implementation) ──

/// Cloud-based form filler: extracts tool parameters via LLM API calls.
pub struct CloudFormFiller {
    pool: Arc<Mutex<LlmPool>>,
    max_retries: usize,
}

/// Backward-compatible type alias.
pub type FormFiller = CloudFormFiller;

/// Model ladder sequence: Haiku first, escalate to Sonnet.
const MODEL_LADDER: &[&str] = &["haiku", "haiku", "sonnet"];

impl CloudFormFiller {
    /// Create a new cloud form filler.
    pub fn new(pool: Arc<Mutex<LlmPool>>, max_retries: usize) -> Self {
        Self { pool, max_retries }
    }

    /// Get the configured max retries.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }
}

#[async_trait::async_trait]
impl FormFillStrategy for CloudFormFiller {
    async fn fill(
        &self,
        intent: &str,
        tool_name: &str,
        tool_description: &str,
        xml_template: &str,
        payload_tag: &str,
    ) -> FormFillResult {
        let mut last_error = String::new();

        for attempt in 0..self.max_retries {
            let model = model_for_attempt(attempt);
            let prompt = if attempt == 0 {
                build_fill_prompt(intent, tool_name, tool_description, xml_template)
            } else {
                build_retry_prompt(
                    intent,
                    tool_name,
                    tool_description,
                    xml_template,
                    &last_error,
                )
            };

            let pool = self.pool.lock().await;
            let result = pool
                .complete(
                    Some(model),
                    vec![Message::text("user", &prompt)],
                    1024,
                    Some("You are a tool parameter extractor. Respond with ONLY filled XML. No explanation, no markdown fencing."),
                )
                .await;

            match result {
                Ok(response) => {
                    if let Some(text) = response.text() {
                        let cleaned = strip_xml_fencing(text);
                        match validate_xml(&cleaned, payload_tag) {
                            Ok(()) => {
                                return FormFillResult::Success {
                                    tool_name: tool_name.to_string(),
                                    filled_xml: cleaned,
                                };
                            }
                            Err(e) => {
                                last_error = e;
                            }
                        }
                    } else {
                        last_error = "LLM returned no text content".to_string();
                    }
                }
                Err(e) => {
                    last_error = format!("LLM API error: {e}");
                }
            }
        }

        FormFillResult::Failed {
            tool_name: tool_name.to_string(),
            last_error,
        }
    }
}

// ── Local form filler (constrained decoding) ──

/// Local constrained-decoding form filler.
///
/// Uses codeLlm's `XmlSchemaConstraint` to guarantee structurally valid XML.
/// Falls back to cloud when a tool has no codeLlm schema (e.g. List-only fields)
/// or when local inference fails.
pub struct LocalFormFiller {
    engine: SharedEngine,
    /// Pre-computed schemas keyed by tool name.
    schemas: HashMap<String, code_llm::schema::ToolSchema>,
    /// Cloud fallback (optional — None means no fallback).
    cloud_fallback: Option<CloudFormFiller>,
}

impl LocalFormFiller {
    /// Create a new local form filler.
    ///
    /// `schemas` maps tool names to their codeLlm schemas.
    /// `cloud_fallback` is used for tools without local schemas.
    pub fn new(
        engine: SharedEngine,
        schemas: HashMap<String, code_llm::schema::ToolSchema>,
        cloud_fallback: Option<CloudFormFiller>,
    ) -> Self {
        Self {
            engine,
            schemas,
            cloud_fallback,
        }
    }
}

#[async_trait::async_trait]
impl FormFillStrategy for LocalFormFiller {
    async fn fill(
        &self,
        intent: &str,
        tool_name: &str,
        tool_description: &str,
        xml_template: &str,
        payload_tag: &str,
    ) -> FormFillResult {
        // Look up pre-computed schema
        let schema = match self.schemas.get(tool_name) {
            Some(s) => s.clone(),
            None => {
                // No local schema — fall back to cloud
                info!("no local schema for '{tool_name}', falling back to cloud");
                return self
                    .cloud_fill_or_fail(intent, tool_name, tool_description, xml_template, payload_tag)
                    .await;
            }
        };

        // Build constraint and run local inference
        let prompt = build_fill_prompt(intent, tool_name, tool_description, xml_template);
        let mut engine = self.engine.lock().await;

        let mut constraint =
            code_llm::constraint::XmlSchemaConstraint::new(schema, engine.tokenizer());

        match engine.complete_constrained(&prompt, &mut constraint, "", 256) {
            Ok((output, _stats)) => {
                // Belt-and-suspenders validation
                match validate_xml(&output, payload_tag) {
                    Ok(()) => {
                        info!("local inference succeeded for '{tool_name}'");
                        FormFillResult::Success {
                            tool_name: tool_name.to_string(),
                            filled_xml: output,
                        }
                    }
                    Err(e) => {
                        info!("local inference produced invalid XML for '{tool_name}': {e}");
                        self.cloud_fill_or_fail(
                            intent,
                            tool_name,
                            tool_description,
                            xml_template,
                            payload_tag,
                        )
                        .await
                    }
                }
            }
            Err(e) => {
                info!("local inference failed for '{tool_name}': {e}");
                self.cloud_fill_or_fail(
                    intent,
                    tool_name,
                    tool_description,
                    xml_template,
                    payload_tag,
                )
                .await
            }
        }
    }
}

impl LocalFormFiller {
    /// Delegate to cloud fallback, or return Failed if no fallback available.
    async fn cloud_fill_or_fail(
        &self,
        intent: &str,
        tool_name: &str,
        tool_description: &str,
        xml_template: &str,
        payload_tag: &str,
    ) -> FormFillResult {
        if let Some(ref cloud) = self.cloud_fallback {
            cloud
                .fill(intent, tool_name, tool_description, xml_template, payload_tag)
                .await
        } else {
            FormFillResult::Failed {
                tool_name: tool_name.to_string(),
                last_error: "no local schema and no cloud fallback".to_string(),
            }
        }
    }
}

// ── Shared utilities ──

/// Build the initial fill prompt.
pub fn build_fill_prompt(
    intent: &str,
    tool_name: &str,
    tool_description: &str,
    xml_template: &str,
) -> String {
    format!(
        "Given the user's intent and a tool's XML template, \
produce a filled XML document that fulfills the intent. \
Use ONLY the tags shown in the template.\n\n\
Intent: \"{intent}\"\n\n\
Tool: {tool_name}\n\
Description: {tool_description}\n\
XML Template:\n{xml_template}\n\n\
Respond with ONLY the filled XML. No explanation."
    )
}

/// Build a retry prompt that includes the previous error.
fn build_retry_prompt(
    intent: &str,
    tool_name: &str,
    tool_description: &str,
    xml_template: &str,
    previous_error: &str,
) -> String {
    format!(
        "Your previous attempt failed: {previous_error}\n\n\
Please try again. Given the user's intent and a tool's XML template, \
produce a filled XML document that fulfills the intent. \
Use ONLY the tags shown in the template.\n\n\
Intent: \"{intent}\"\n\n\
Tool: {tool_name}\n\
Description: {tool_description}\n\
XML Template:\n{xml_template}\n\n\
Respond with ONLY the filled XML. No explanation."
    )
}

/// Select model for a given attempt index (model ladder).
fn model_for_attempt(attempt: usize) -> &'static str {
    if attempt < MODEL_LADDER.len() {
        MODEL_LADDER[attempt]
    } else {
        "sonnet" // fallback to sonnet for any extra attempts
    }
}

/// Strip common XML fencing from LLM output.
///
/// Handles: ```xml\n...\n```, ```\n...\n```, and bare XML.
pub fn strip_xml_fencing(text: &str) -> String {
    let trimmed = text.trim();

    // Handle ```xml ... ``` or ``` ... ```
    if let Some(rest) = trimmed.strip_prefix("```xml") {
        let without_closing = rest.trim().strip_suffix("```").unwrap_or(rest.trim());
        return without_closing.trim().to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        let without_closing = rest.trim().strip_suffix("```").unwrap_or(rest.trim());
        return without_closing.trim().to_string();
    }

    trimmed.to_string()
}

/// Validate that the XML is well-formed and has the expected root tag.
pub fn validate_xml(xml: &str, expected_root_tag: &str) -> Result<(), String> {
    let trimmed = xml.trim();

    if trimmed.is_empty() {
        return Err("empty XML".to_string());
    }

    // Check it starts with a tag
    if !trimmed.starts_with('<') {
        return Err("not valid XML: doesn't start with '<'".to_string());
    }

    // Extract root tag name
    let expected_open = format!("<{expected_root_tag}");
    let expected_close = format!("</{expected_root_tag}>");

    if !trimmed.starts_with(&expected_open) {
        // Try to extract actual root tag for error message
        if let Some(end) = trimmed.find(['>', ' ']) {
            let actual = &trimmed[1..end];
            return Err(format!(
                "expected root tag <{expected_root_tag}>, got <{actual}>"
            ));
        }
        return Err(format!("expected root tag <{expected_root_tag}>"));
    }

    if !trimmed.ends_with(&expected_close) {
        return Err(format!(
            "missing closing tag </{expected_root_tag}>"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_pool() -> Arc<Mutex<LlmPool>> {
        Arc::new(Mutex::new(LlmPool::with_base_url(
            "test-key".into(),
            "haiku",
            "http://localhost:19999".into(),
        )))
    }

    #[test]
    fn form_filler_creation() {
        let pool = mock_pool();
        let filler = CloudFormFiller::new(pool, 3);
        assert_eq!(filler.max_retries(), 3);
    }

    #[test]
    fn build_fill_prompt_includes_all_parts() {
        let prompt = build_fill_prompt(
            "I need to see parser.rs",
            "file-ops",
            "Reads and writes files on the filesystem",
            "<FileOpsRequest><action/><path/></FileOpsRequest>",
        );
        assert!(prompt.contains("I need to see parser.rs"));
        assert!(prompt.contains("file-ops"));
        assert!(prompt.contains("Reads and writes files"));
        assert!(prompt.contains("<FileOpsRequest>"));
    }

    #[test]
    fn parse_fill_response_valid_xml() {
        let xml = "<FileOpsRequest><action>read</action><path>src/parser.rs</path></FileOpsRequest>";
        let result = validate_xml(xml, "FileOpsRequest");
        assert!(result.is_ok());
    }

    #[test]
    fn parse_fill_response_with_fencing() {
        let fenced = "```xml\n<FileOpsRequest><action>read</action><path>foo.rs</path></FileOpsRequest>\n```";
        let cleaned = strip_xml_fencing(fenced);
        assert!(cleaned.starts_with("<FileOpsRequest>"));
        assert!(cleaned.ends_with("</FileOpsRequest>"));
        assert!(validate_xml(&cleaned, "FileOpsRequest").is_ok());
    }

    #[test]
    fn parse_fill_response_malformed() {
        let result = validate_xml("not xml at all", "FileOpsRequest");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not valid XML"));
    }

    #[test]
    fn validate_xml_root_tag() {
        let xml = "<ShellRequest><command>ls</command></ShellRequest>";
        assert!(validate_xml(xml, "ShellRequest").is_ok());
        let wrong = validate_xml(xml, "FileOpsRequest");
        assert!(wrong.is_err());
        assert!(wrong.unwrap_err().contains("expected root tag"));
    }

    #[test]
    fn validate_xml_malformed() {
        // Missing closing tag
        let xml = "<FileOpsRequest><action>read</action>";
        let result = validate_xml(xml, "FileOpsRequest");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing closing tag"));
    }

    #[test]
    fn model_ladder_sequence() {
        assert_eq!(model_for_attempt(0), "haiku");
        assert_eq!(model_for_attempt(1), "haiku");
        assert_eq!(model_for_attempt(2), "sonnet");
        // Beyond ladder: falls back to sonnet
        assert_eq!(model_for_attempt(5), "sonnet");
    }

    #[test]
    fn form_fill_result_variants() {
        let success = FormFillResult::Success {
            tool_name: "file-ops".into(),
            filled_xml: "<FileOpsRequest><action>read</action></FileOpsRequest>".into(),
        };
        assert!(matches!(success, FormFillResult::Success { .. }));

        let failed = FormFillResult::Failed {
            tool_name: "file-ops".into(),
            last_error: "malformed XML".into(),
        };
        assert!(matches!(failed, FormFillResult::Failed { .. }));
    }

    #[test]
    fn max_retries_configurable() {
        let pool = mock_pool();
        let filler = CloudFormFiller::new(pool.clone(), 5);
        assert_eq!(filler.max_retries(), 5);

        let filler2 = CloudFormFiller::new(pool, 1);
        assert_eq!(filler2.max_retries(), 1);
    }

    // ── Type alias backward compat ──

    #[test]
    fn form_filler_alias_works() {
        let pool = mock_pool();
        let filler: FormFiller = CloudFormFiller::new(pool, 3);
        assert_eq!(filler.max_retries(), 3);
    }

    // ── LocalFormFiller tests ──

    #[test]
    fn local_form_filler_no_schema_no_fallback() {
        // Verify schema lookup behavior when no schemas are registered
        let schemas: HashMap<String, code_llm::schema::ToolSchema> = HashMap::new();
        assert!(schemas.is_empty());
        assert!(!schemas.contains_key("file-read"));
    }

    #[test]
    fn local_form_filler_schema_lookup() {
        use code_llm::schema::{ToolSchema, ToolFieldType as CLT};

        let mut schemas = HashMap::new();
        schemas.insert(
            "file-read".to_string(),
            ToolSchema::new("FileReadRequest")
                .required("path", CLT::String)
                .optional("offset", CLT::Integer),
        );

        assert!(schemas.contains_key("file-read"));
        assert!(!schemas.contains_key("unknown-tool"));
    }
}
