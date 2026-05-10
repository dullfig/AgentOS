//! ShimRulesTool — read/update an agent's per-agent shim configuration.
//!
//! Each agent that uses cortex shims has a `shim-rules.json` file under
//! `<rules_dir>/<agent>/shim-rules.json`. The file holds a serialized
//! `agentos_llm::types::ShimAttachment` (gate_shims, steer_shims,
//! inject_shims, shim_rules). At pipeline build time the loader reads
//! this file and installs the parsed value on the target agent's
//! handler; the shim-expert agent edits it via this tool.
//!
//! v1 reload semantics: restart-required. The loader runs once at
//! `AgentPipelineBuilder::build()`. A future revision will likely turn
//! this JSON file into a runtime subsystem analogous to the VMM context
//! manager — versioned, hot-swappable, observable. Out of scope for v1.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use serde_json::json;

use agentos_llm::types::ShimAttachment;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Tool that reads/writes per-agent `shim-rules.json` files.
#[derive(Clone)]
pub struct ShimRulesTool {
    rules_dir: Arc<PathBuf>,
}

impl ShimRulesTool {
    /// Construct rooted at `rules_dir` (typically `<data_dir>/agents/`).
    pub fn new(rules_dir: PathBuf) -> Self {
        Self {
            rules_dir: Arc::new(rules_dir),
        }
    }

    fn path_for(&self, agent: &str) -> PathBuf {
        self.rules_dir.join(agent).join("shim-rules.json")
    }

    fn read(&self, agent: &str) -> Result<String, String> {
        let path = self.path_for(agent);
        if !path.exists() {
            // Empty default — equivalent to "no shims attached".
            return Ok(serde_json::to_string(&ShimAttachment::default())
                .map_err(|e| format!("serialize default: {e}"))?);
        }
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))
    }

    fn update(&self, agent: &str, raw: &str) -> Result<String, String> {
        // Parse to validate structure before touching disk; this also
        // catches typos in shim_rules' if/then/else shape because the
        // ShimAttachment struct's serde gates are strict.
        let parsed: ShimAttachment = serde_json::from_str(raw).map_err(|e| {
            format!("rules JSON does not match ShimAttachment schema: {e}")
        })?;

        let agent_dir = self.rules_dir.join(agent);
        std::fs::create_dir_all(&agent_dir)
            .map_err(|e| format!("create {}: {e}", agent_dir.display()))?;

        let path = agent_dir.join("shim-rules.json");
        // Pretty-print so a human reviewer reading the file sees a
        // diffable structure.
        let pretty = serde_json::to_string_pretty(&parsed)
            .map_err(|e| format!("re-serialize: {e}"))?;
        std::fs::write(&path, &pretty)
            .map_err(|e| format!("write {}: {e}", path.display()))?;

        Ok(json!({
            "agent": agent,
            "path": path,
            "gate_count": parsed.gate_shims.len(),
            "steer_count": parsed.steer_shims.len(),
            "inject_count": parsed.inject_shims.len(),
            "rule_count": parsed.shim_rules.len(),
            "note": "agent restart required for changes to take effect"
        })
        .to_string())
    }
}

#[async_trait]
impl Handler for ShimRulesTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let action = extract_tag(&xml_str, "action").unwrap_or_default();
        let agent = match extract_tag(&xml_str, "agent") {
            Some(a) if !a.is_empty() => a,
            _ => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err("missing required <agent>"),
                });
            }
        };

        let result = match action.as_str() {
            "read" => self.read(&agent),
            "update" => match extract_tag(&xml_str, "rules") {
                Some(r) if !r.is_empty() => self.update(&agent, &r),
                _ => Err("missing required <rules> for update".into()),
            },
            "" => Err("missing required <action>".into()),
            other => Err(format!("unknown action: {other} (allowed: read|update)")),
        };

        let payload_xml = match result {
            Ok(body) => ToolResponse::ok(&body),
            Err(msg) => ToolResponse::err(&msg),
        };
        Ok(HandlerResponse::Reply { payload_xml })
    }
}

#[async_trait]
impl ToolPeer for ShimRulesTool {
    fn name(&self) -> &str {
        "shim-rules"
    }

    fn wit(&self) -> &str {
        r#"
/// Read or update an agent's per-agent shim configuration JSON
/// (gate_shims, steer_shims, inject_shims, shim_rules). Updates
/// require an agent restart to take effect (v1).
interface shim-rules {
    record request {
        /// "read" | "update"
        action: string,
        /// target agent name (e.g. "bob")
        agent: string,
        /// JSON-serialized ShimAttachment (required for update)
        rules: option<string>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "shim-expert".into(),
            own_name: "shim-rules".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "ShimRules".into(),
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
    async fn read_missing_returns_default() {
        let dir = TempDir::new().unwrap();
        let tool = ShimRulesTool::new(dir.path().to_path_buf());

        let xml = "<ShimRules><action>read</action><agent>bob</agent></ShimRules>";
        let (ok, body) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        // Default is an empty ShimAttachment — all four arrays are
        // omitted by serde, leaving an empty object.
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            parsed.as_object().map(|o| o.is_empty()).unwrap_or(false),
            "expected empty object, got {body}"
        );
    }

    #[tokio::test]
    async fn update_then_read_round_trips() {
        let dir = TempDir::new().unwrap();
        let tool = ShimRulesTool::new(dir.path().to_path_buf());

        let rules_json = r#"{
            "gate_shims": ["should_respond"],
            "steer_shims": ["voice_bob"],
            "inject_shims": [],
            "shim_rules": [
                {"if": {"gate": "should_respond", "gt": 0.7},
                 "then": {"activate": ["voice_bob"]}}
            ]
        }"#;

        let update_xml = format!(
            "<ShimRules><action>update</action><agent>bob</agent><rules>{}</rules></ShimRules>",
            agentos_events::xml_escape(rules_json),
        );
        let (ok, body) = parse(tool.handle(make_payload(&update_xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        assert!(body.contains("\"agent\":\"bob\""));
        assert!(body.contains("restart required"));

        // File should exist on disk now.
        assert!(dir.path().join("bob").join("shim-rules.json").exists());

        // Reading should reflect the write.
        let read_xml = "<ShimRules><action>read</action><agent>bob</agent></ShimRules>";
        let (ok, body) = parse(tool.handle(make_payload(read_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let parsed: ShimAttachment = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.gate_shims, vec!["should_respond"]);
        assert_eq!(parsed.steer_shims, vec!["voice_bob"]);
        assert_eq!(parsed.shim_rules.len(), 1);
    }

    #[tokio::test]
    async fn update_with_invalid_json_errors() {
        let dir = TempDir::new().unwrap();
        let tool = ShimRulesTool::new(dir.path().to_path_buf());

        let xml = format!(
            "<ShimRules><action>update</action><agent>bob</agent><rules>{}</rules></ShimRules>",
            agentos_events::xml_escape(r#"{"gate_shims": "not an array"}"#),
        );
        let (ok, msg) = parse(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("ShimAttachment schema"), "msg: {msg}");
    }

    #[tokio::test]
    async fn missing_agent_errors() {
        let dir = TempDir::new().unwrap();
        let tool = ShimRulesTool::new(dir.path().to_path_buf());

        let xml = "<ShimRules><action>read</action></ShimRules>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<agent>"));
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let dir = TempDir::new().unwrap();
        let tool = ShimRulesTool::new(dir.path().to_path_buf());

        let xml = "<ShimRules><action>delete</action><agent>bob</agent></ShimRules>";
        let (ok, msg) = parse(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("unknown action"));
    }
}
