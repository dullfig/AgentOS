//! D2 diagram renderer — parses D2 syntax and renders box-drawing art.
//!
//! LLMs emit D2 in fenced code blocks (`\`\`\`d2`). This module intercepts
//! those blocks and replaces them with deterministic box-drawing renderings.
//! No external dependencies — pure Rust, ratatui-native output.

pub mod parser;
pub mod layout;
pub mod grid;

use ratatui::text::Line;

use crate::organism::{ListenerDef, Organism};

/// Render D2 source text to styled ratatui Lines.
///
/// `d2_source` is the raw D2 text (without the fenced code markers).
/// `max_width` constrains the output to fit the terminal width.
pub fn render_d2(d2_source: &str, max_width: usize) -> Vec<Line<'static>> {
    let graph = parser::parse_d2(d2_source);
    let positioned = layout::layout(&graph, max_width);
    grid::render_to_lines(&positioned, max_width)
}

/// Determine the D2 shape for a listener based on its type.
fn listener_shape(def: &ListenerDef) -> &'static str {
    if def.is_agent {
        "diamond"
    } else if def.buffer.is_some() {
        "hexagon"
    } else if def.wasm.is_some() {
        "cylinder"
    } else {
        "rectangle"
    }
}

/// Convert an organism definition into D2 source text for rendering.
///
/// Walks all listeners, emits node declarations with shapes, and directed
/// edges from agents to their peers. Sorted alphabetically for deterministic layout.
pub fn organism_to_d2(org: &Organism) -> String {
    let mut lines = Vec::new();
    let mut names: Vec<&str> = org.listeners().keys().map(|s| s.as_str()).collect();
    names.sort();

    // Node declarations
    for name in &names {
        if let Some(def) = org.get_listener(name) {
            let shape = listener_shape(def);
            lines.push(format!("{name}: {{ shape: {shape} }}"));
        }
    }

    // Edge declarations (agents → peers)
    for name in &names {
        if let Some(def) = org.get_listener(name) {
            if def.is_agent {
                let mut peers: Vec<&str> = def.peers.iter().map(|s| s.as_str()).collect();
                peers.sort();
                for peer in peers {
                    lines.push(format!("{name} -> {peer}"));
                }
            }
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod organism_tests {
    use super::*;
    use crate::organism::{ListenerDef, Organism};
    use crate::organism::AgentConfig;

    fn sample_listener(name: &str) -> ListenerDef {
        ListenerDef {
            name: name.into(),
            payload_tag: format!("{name}Request"),
            handler: format!("handlers.{name}.handle"),
            description: format!("{name} handler"),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
            librarian: false,
            wasm: None,
            semantic_description: None,
            agent_config: None,
            buffer: None,
        }
    }

    #[test]
    fn empty_organism_produces_empty_d2() {
        let org = Organism::new("test");
        let d2 = organism_to_d2(&org);
        assert!(d2.is_empty());
    }

    #[test]
    fn agent_gets_diamond_shape() {
        let mut org = Organism::new("test");
        let mut agent = sample_listener("coding-agent");
        agent.is_agent = true;
        agent.agent_config = Some(AgentConfig::default());
        org.register_listener(agent).unwrap();

        let d2 = organism_to_d2(&org);
        assert!(d2.contains("coding-agent: { shape: diamond }"));
    }

    #[test]
    fn regular_tool_gets_rectangle() {
        let mut org = Organism::new("test");
        org.register_listener(sample_listener("file-read")).unwrap();

        let d2 = organism_to_d2(&org);
        assert!(d2.contains("file-read: { shape: rectangle }"));
    }

    #[test]
    fn wasm_tool_gets_cylinder() {
        let mut org = Organism::new("test");
        let mut tool = sample_listener("my-wasm");
        tool.wasm = Some(crate::organism::WasmToolConfig {
            path: "tools/my.wasm".into(),
            capabilities: Default::default(),
        });
        org.register_listener(tool).unwrap();

        let d2 = organism_to_d2(&org);
        assert!(d2.contains("my-wasm: { shape: cylinder }"));
    }

    #[test]
    fn buffer_gets_hexagon() {
        let mut org = Organism::new("test");
        let mut tool = sample_listener("sub-agent");
        tool.buffer = Some(crate::organism::BufferConfig {
            description: "A sub-agent".into(),
            parameters: vec![],
            required: vec![],
            requires: vec![],
            organism: "child.yaml".into(),
            max_concurrency: 5,
            timeout_secs: 300,
        });
        org.register_listener(tool).unwrap();

        let d2 = organism_to_d2(&org);
        assert!(d2.contains("sub-agent: { shape: hexagon }"));
    }

    #[test]
    fn agent_peers_produce_edges() {
        let mut org = Organism::new("test");
        let mut agent = sample_listener("agent-1");
        agent.is_agent = true;
        agent.agent_config = Some(AgentConfig::default());
        agent.peers = vec!["file-read".into(), "command-exec".into()];
        org.register_listener(agent).unwrap();
        org.register_listener(sample_listener("file-read")).unwrap();
        org.register_listener(sample_listener("command-exec")).unwrap();

        let d2 = organism_to_d2(&org);
        assert!(d2.contains("agent-1 -> command-exec"));
        assert!(d2.contains("agent-1 -> file-read"));
    }

    #[test]
    fn round_trip_through_render() {
        let mut org = Organism::new("test");
        let mut agent = sample_listener("agent");
        agent.is_agent = true;
        agent.agent_config = Some(AgentConfig::default());
        agent.peers = vec!["tool".into()];
        org.register_listener(agent).unwrap();
        org.register_listener(sample_listener("tool")).unwrap();

        let d2 = organism_to_d2(&org);
        let lines = render_d2(&d2, 80);
        // Should produce some output without panicking
        assert!(!lines.is_empty());
    }
}
