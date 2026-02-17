//! Tool Definition Bridge — ToolPeer → Anthropic ToolDefinition.
//!
//! Generates JSON Schema tool definitions from registered ToolPeers.
//! Hand-written schemas for Phase 4's known tools (file-ops, shell, codebase-index).
//! Phase 5 (WASM+WIT) will auto-generate from WIT definitions.

use crate::llm::types::ToolDefinition;

/// Build a ToolDefinition for the file-ops tool.
pub fn file_ops_definition() -> ToolDefinition {
    ToolDefinition {
        name: "file-ops".into(),
        description: "File operations: read, write, or list files and directories.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "The file operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (only for 'write' action)"
                }
            },
            "required": ["action", "path"]
        }),
    }
}

/// Build a ToolDefinition for the shell tool.
pub fn shell_definition() -> ToolDefinition {
    ToolDefinition {
        name: "shell".into(),
        description: "Execute a shell command and return stdout, stderr, and exit code.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 5000)"
                }
            },
            "required": ["command"]
        }),
    }
}

/// Build a ToolDefinition for the codebase-index tool.
pub fn codebase_index_definition() -> ToolDefinition {
    ToolDefinition {
        name: "codebase-index".into(),
        description: "Tree-sitter code indexing: index files/directories, search symbols, or get a codebase map.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["index_file", "index_directory", "search", "codebase_map"],
                    "description": "The indexing operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (for index_file, index_directory)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)"
                },
                "kind": {
                    "type": "string",
                    "description": "Symbol kind filter (for search action, e.g. 'function', 'struct')"
                }
            },
            "required": ["action"]
        }),
    }
}

/// Build tool definitions for all known tool-peers.
///
/// Returns definitions for tools that are available in the given peer list.
/// Peer names must match the tool-peer's registered name in the organism.
pub fn build_tool_definitions(peer_names: &[&str]) -> Vec<ToolDefinition> {
    let mut defs = Vec::new();
    for name in peer_names {
        if let Some(def) = definition_for_peer(name) {
            defs.push(def);
        }
    }
    defs
}

/// Get the tool definition for a named peer, if known.
pub fn definition_for_peer(name: &str) -> Option<ToolDefinition> {
    match name {
        "file-ops" => Some(file_ops_definition()),
        "shell" => Some(shell_definition()),
        "codebase-index" => Some(codebase_index_definition()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_ops_def_is_valid() {
        let def = file_ops_definition();
        assert_eq!(def.name, "file-ops");
        assert!(def.description.contains("File"));
        assert_eq!(def.input_schema["type"], "object");
        let props = &def.input_schema["properties"];
        assert!(props.get("action").is_some());
        assert!(props.get("path").is_some());
        assert!(props.get("content").is_some());
        let required = def.input_schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("action")));
        assert!(required.contains(&serde_json::json!("path")));
    }

    #[test]
    fn shell_def_is_valid() {
        let def = shell_definition();
        assert_eq!(def.name, "shell");
        let props = &def.input_schema["properties"];
        assert!(props.get("command").is_some());
        assert!(props.get("timeout").is_some());
    }

    #[test]
    fn codebase_index_def_is_valid() {
        let def = codebase_index_definition();
        assert_eq!(def.name, "codebase-index");
        let props = &def.input_schema["properties"];
        assert!(props.get("action").is_some());
        assert!(props.get("query").is_some());
    }

    #[test]
    fn build_definitions_from_peer_names() {
        let defs = build_tool_definitions(&["file-ops", "shell", "codebase-index"]);
        assert_eq!(defs.len(), 3);
        assert_eq!(defs[0].name, "file-ops");
        assert_eq!(defs[1].name, "shell");
        assert_eq!(defs[2].name, "codebase-index");
    }

    #[test]
    fn build_definitions_skips_unknown() {
        let defs = build_tool_definitions(&["file-ops", "unknown-tool", "shell"]);
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "file-ops");
        assert_eq!(defs[1].name, "shell");
    }

    #[test]
    fn build_definitions_empty() {
        let defs = build_tool_definitions(&[]);
        assert!(defs.is_empty());
    }

    #[test]
    fn definition_for_peer_none() {
        assert!(definition_for_peer("nonexistent").is_none());
    }

    #[test]
    fn definitions_serialize_to_valid_json() {
        let defs = build_tool_definitions(&["file-ops", "shell", "codebase-index"]);
        for def in &defs {
            let json = serde_json::to_string(def).unwrap();
            assert!(json.contains(&def.name));
            // Re-parse to ensure valid JSON
            let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        }
    }
}
