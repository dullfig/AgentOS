//! list-agents tool — enumerate available specialist agents.
//!
//! Zero-input, read-only tool that returns a summary of all agents
//! and buffer nodes in the current organism. Used by the concierge
//! (Bob) to give meaningful routing guidance.

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::{ToolPeer, ToolResponse};
use crate::organism::Organism;

/// A read-only tool that lists registered agents and their capabilities.
///
/// The agent info is captured at build time from the organism config.
pub struct ListAgentsTool {
    info: String,
}

impl ListAgentsTool {
    /// Build from an organism, capturing agent and buffer metadata.
    pub fn from_organism(org: &Organism) -> Self {
        let mut lines = Vec::new();

        // Agent listeners (direct agents like Bob)
        let agents = org.agent_listeners();
        if !agents.is_empty() {
            lines.push("Agents:".to_string());
            for a in &agents {
                let model = a.model.as_deref().unwrap_or("default");
                lines.push(format!("  - {} (model: {}): {}", a.name, model, a.description));
            }
        }

        // Buffer nodes (child pipelines like coder, agent-expert)
        let buffers = org.buffer_listeners();
        if !buffers.is_empty() {
            lines.push("Specialists (buffer nodes):".to_string());
            for b in &buffers {
                let buf = b.buffer.as_ref().unwrap();
                let org_ref = buf.organism.as_deref().unwrap_or("(self-clone)");
                lines.push(format!("  - {}: {}", b.name, buf.description));
                lines.push(format!("    organism: {}, timeout: {}s, max_concurrency: {}",
                    org_ref, buf.timeout_secs, buf.max_concurrency));
                if !buf.requires.is_empty() {
                    lines.push(format!("    tools: [{}]", buf.requires.join(", ")));
                }
            }
        }

        if lines.is_empty() {
            lines.push("No agents or specialists configured.".to_string());
        }

        Self { info: lines.join("\n") }
    }
}

#[async_trait]
impl Handler for ListAgentsTool {
    async fn handle(&self, _payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&self.info),
        })
    }
}

#[async_trait]
impl ToolPeer for ListAgentsTool {
    fn name(&self) -> &str {
        "list-agents"
    }

    fn wit(&self) -> &str {
        r#"
/// List all available specialist agents and their capabilities. No input needed — returns a summary of every agent and buffer node registered in the current organism.
interface list-agents {
    record request {
    }
    list: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::Organism;
    use crate::organism::{ListenerDef, BufferConfig, CallableParam};

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            from: "test".into(),
            own_name: "list-agents".into(),
            thread_id: "test-thread".into(),
        }
    }

    fn make_payload() -> ValidatedPayload {
        ValidatedPayload {
            xml: b"<ListAgentsRequest></ListAgentsRequest>".to_vec(),
            tag: "ListAgentsRequest".into(),
        }
    }

    #[tokio::test]
    async fn lists_agents_and_buffers() {
        let mut org = Organism::new("test");
        org.register_listener(ListenerDef {
            name: "bob".into(),
            payload_tag: "AgentTask".into(),
            handler: "agent.handle".into(),
            description: "Concierge agent".into(),
            is_agent: true,
            peers: vec![],
            model: Some("haiku".into()),
            ports: vec![],
            librarian: false,
            wasm: None,
            python: None,
            semantic_description: None,
            agent_config: None,
            buffer: None,
        }).unwrap();
        org.register_listener(ListenerDef {
            name: "coder".into(),
            payload_tag: "CoderRequest".into(),
            handler: "buffer".into(),
            description: "Coding specialist".into(),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
            librarian: false,
            wasm: None,
            python: None,
            semantic_description: None,
            agent_config: None,
            buffer: Some(BufferConfig {
                description: "Write code, edit files, run tests".into(),
                parameters: vec![CallableParam {
                    name: "task".into(),
                    param_type: "string".into(),
                    description: Some("The coding task".into()),
                    enum_values: None,
                }],
                required: vec!["task".into()],
                requires: vec!["file-read".into(), "file-write".into()],
                organism: Some("organisms/coder.yaml".into()),
                max_concurrency: 1,
                timeout_secs: 600,
                context_visible: false,
            }),
        }).unwrap();

        let tool = ListAgentsTool::from_organism(&org);
        let result = tool.handle(make_payload(), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("bob"), "should list bob: {s}");
                assert!(s.contains("haiku"), "should show model: {s}");
                assert!(s.contains("coder"), "should list coder: {s}");
                assert!(s.contains("Write code"), "should show buffer desc: {s}");
                assert!(s.contains("file-read"), "should show tools: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn empty_organism() {
        let org = Organism::new("empty");
        let tool = ListAgentsTool::from_organism(&org);
        let result = tool.handle(make_payload(), make_ctx()).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let s = String::from_utf8(payload_xml).unwrap();
                assert!(s.contains("No agents"), "should say none: {s}");
            }
            _ => panic!("expected Reply"),
        }
    }
}
