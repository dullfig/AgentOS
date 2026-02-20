//! Form filler — Haiku-based parameter extraction for semantic routing.
//!
//! Takes natural language intent + tool metadata, produces filled XML.
//! Model ladder: Haiku (cheap, fast) → Sonnet (escalate on failure).
//! Never Opus — Opus is the thinker.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::llm::types::Message;
use crate::llm::LlmPool;

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

/// The form filler: extracts tool parameters from natural language via LLM.
pub struct FormFiller {
    pool: Arc<Mutex<LlmPool>>,
    max_retries: usize,
}

/// Model ladder sequence: Haiku first, escalate to Sonnet.
const MODEL_LADDER: &[&str] = &["haiku", "haiku", "sonnet"];

impl FormFiller {
    /// Create a new form filler.
    pub fn new(pool: Arc<Mutex<LlmPool>>, max_retries: usize) -> Self {
        Self { pool, max_retries }
    }

    /// Fill tool XML from natural language intent.
    ///
    /// Tries the model ladder (Haiku → Sonnet) up to `max_retries` times.
    /// Returns `FormFillResult::Success` with filled XML, or `Failed` if
    /// all retries are exhausted.
    pub async fn fill(
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

    /// Get the configured max retries.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }
}

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
    if trimmed.starts_with("```") {
        let without_opening = if trimmed.starts_with("```xml") {
            &trimmed[6..]
        } else {
            &trimmed[3..]
        };
        let without_closing = without_opening
            .trim()
            .strip_suffix("```")
            .unwrap_or(without_opening.trim());
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
        if let Some(end) = trimmed.find(|c: char| c == '>' || c == ' ') {
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
        let filler = FormFiller::new(pool, 3);
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
        let filler = FormFiller::new(pool.clone(), 5);
        assert_eq!(filler.max_retries(), 5);

        let filler2 = FormFiller::new(pool, 1);
        assert_eq!(filler2.max_retries(), 1);
    }
}
