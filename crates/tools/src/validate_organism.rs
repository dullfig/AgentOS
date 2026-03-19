//! validate-organism tool — parse and validate organism YAML.
//!
//! Reads a YAML string (or file path via VDrive), runs parse_organism(),
//! and returns structured diagnostics. Used by agent-expert for
//! build→validate→fix loops.

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::{extract_tag, ToolPeer, ToolResponse};
use super::vdrive_tools::DriveSlot;
use agentos_organism::parser::parse_organism;

pub struct ValidateOrganismTool {
    slot: DriveSlot,
}

impl ValidateOrganismTool {
    pub fn new(slot: DriveSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Handler for ValidateOrganismTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Optional: source directory for breadcrumb
        let source_dir = extract_tag(&xml_str, "source_dir");

        // Accept either inline YAML or a file path
        let yaml = if let Some(inline) = extract_tag(&xml_str, "yaml") {
            inline
        } else if let Some(path) = extract_tag(&xml_str, "path") {
            // Read from VDrive
            let guard = self.slot.read().await;
            match guard.as_ref() {
                Some(drive) => {
                    match drive.read_file(&path, 1, 50_000) {
                        Ok(result) => result.content,
                        Err(e) => {
                            return Ok(HandlerResponse::Reply {
                                payload_xml: ToolResponse::err(
                                    &format!("failed to read '{}': {}", path, e),
                                ),
                            });
                        }
                    }
                }
                None => {
                    return Ok(HandlerResponse::Reply {
                        payload_xml: ToolResponse::err(
                            "no storage mounted — provide inline <yaml> or mount a workspace",
                        ),
                    });
                }
            }
        } else {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(
                    "provide either <yaml> (inline YAML) or <path> (file path to read)",
                ),
            });
        };

        // Parse and validate
        match parse_organism(&yaml) {
            Ok(org) => {
                let mut report = Vec::new();
                report.push(format!("organism: {}", org.name));

                // Summarize listeners
                let listeners = org.listeners();
                report.push(format!("listeners: {}", listeners.len()));
                for (name, def) in listeners {
                    let mut desc = format!("  - {} (handler: {})", name, def.handler);
                    if def.is_agent {
                        desc.push_str(" [agent]");
                    }
                    if def.buffer.is_some() {
                        desc.push_str(" [buffer]");
                    }
                    if def.wasm.is_some() {
                        desc.push_str(" [wasm]");
                    }
                    if def.python.is_some() {
                        desc.push_str(" [python]");
                    }
                    if !def.peers.is_empty() {
                        desc.push_str(&format!(" peers: [{}]", def.peers.join(", ")));
                    }
                    report.push(desc);
                }

                // Summarize profiles
                let profiles = org.profile_names();
                report.push(format!("profiles: {}", profiles.len()));
                for name in &profiles {
                    report.push(format!("  - {}", name));
                }

                // Cross-reference checks
                let mut warnings = Vec::new();

                // Check: peers reference existing listeners
                for (name, def) in listeners {
                    for peer in &def.peers {
                        if !listeners.contains_key(peer.as_str()) {
                            warnings.push(format!(
                                "listener '{}' references peer '{}' which is not declared",
                                name, peer
                            ));
                        }
                    }
                }

                // Check: agent listeners exist
                let agents: Vec<_> = listeners.values().filter(|l| l.is_agent).collect();
                if agents.is_empty() {
                    warnings.push("no agent listeners found".to_string());
                }

                // Check: llm-pool exists if agents are declared
                if !agents.is_empty() && !listeners.contains_key("llm-pool") {
                    warnings.push(
                        "agent listener(s) declared but no 'llm-pool' listener found".to_string(),
                    );
                }

                // Check: buffer organisms reference files (can't check existence from here)
                for (name, def) in listeners {
                    if let Some(ref buf) = def.buffer {
                        if let Some(ref org_path) = buf.organism {
                            report.push(format!(
                                "  note: buffer '{}' references child organism '{}'",
                                name, org_path
                            ));
                        }
                    }
                }

                if warnings.is_empty() {
                    report.push("validation: OK".to_string());

                    // Write breadcrumb if source_dir provided
                    if let Some(ref dir) = source_dir {
                        let guard = self.slot.read().await;
                        if let Some(drive) = guard.as_ref() {
                            let marker_path = format!("{}/.validated", dir);
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0);
                            let content = format!("{{\"status\":\"ok\",\"timestamp\":{}}}", timestamp);
                            let _ = drive.write_file(&marker_path, &content);
                        }
                    }
                } else {
                    report.push(format!("warnings: {}", warnings.len()));
                    for w in &warnings {
                        report.push(format!("  ⚠ {}", w));
                    }
                }

                Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::ok(&report.join("\n")),
                })
            }
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("parse error: {}", e)),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for ValidateOrganismTool {
    fn name(&self) -> &str {
        "validate-organism"
    }

    fn wit(&self) -> &str {
        r#"
/// Validate an organism YAML configuration. Parses the YAML, checks listener/profile cross-references, and returns a diagnostic report. Provide either inline YAML or a file path.
interface validate-organism {
    record request {
        /// Inline YAML to validate (mutually exclusive with path)
        yaml: option<string>,
        /// Path to a YAML file to read and validate (mutually exclusive with yaml)
        path: option<string>,
        /// Source directory to write .validated breadcrumb on success
        source-dir: option<string>,
    }
    validate: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vdrive_tools::empty_slot;

    fn make_tool() -> ValidateOrganismTool {
        ValidateOrganismTool::new(empty_slot())
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "ValidateOrganismRequest".into(),
        }
    }

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            from: "test".into(),
            own_name: "validate-organism".into(),
            thread_id: "test-thread".into(),
        }
    }

    #[tokio::test]
    async fn valid_organism_reports_ok() {
        let tool = make_tool();
        let yaml = r#"
organism:
  name: test
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test agent"
    agent: true
    peers: [llm-pool]
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, llm-pool]
    journal: retain_forever
"#;
        let xml = format!(
            "<ValidateOrganismRequest><yaml>{}</yaml></ValidateOrganismRequest>",
            crate::xml_escape(yaml)
        );
        let result = tool.handle(make_payload(&xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>true</success>"), "expected success: {s}");
                assert!(s.contains("validation: OK"), "expected OK: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn invalid_yaml_returns_error() {
        let tool = make_tool();
        let xml = "<ValidateOrganismRequest><yaml>not: valid: yaml: [}</yaml></ValidateOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "expected error: {s}");
                assert!(s.contains("parse error"), "expected parse error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn warns_on_dangling_peer() {
        let tool = make_tool();
        let yaml = r#"
organism:
  name: test
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test agent"
    agent: true
    peers: [nonexistent-tool]
  - name: llm-pool
    payload_class: llm.LlmRequest
    handler: llm.handle
    description: "LLM pool"
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent, llm-pool]
    journal: retain_forever
"#;
        let xml = format!(
            "<ValidateOrganismRequest><yaml>{}</yaml></ValidateOrganismRequest>",
            crate::xml_escape(yaml)
        );
        let result = tool.handle(make_payload(&xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("nonexistent-tool"), "expected dangling peer warning: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn warns_on_missing_llm_pool() {
        let tool = make_tool();
        let yaml = r#"
organism:
  name: test
listeners:
  - name: my-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Test agent"
    agent: true
profiles:
  default:
    linux_user: agentos
    listeners: [my-agent]
    journal: retain_forever
"#;
        let xml = format!(
            "<ValidateOrganismRequest><yaml>{}</yaml></ValidateOrganismRequest>",
            crate::xml_escape(yaml)
        );
        let result = tool.handle(make_payload(&xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("llm-pool"), "expected missing llm-pool warning: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn missing_both_yaml_and_path() {
        let tool = make_tool();
        let xml = "<ValidateOrganismRequest></ValidateOrganismRequest>";
        let result = tool.handle(make_payload(xml), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("<success>false</success>"), "expected error: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }
}
