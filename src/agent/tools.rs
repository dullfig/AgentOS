//! Tool Definition Bridge — ToolPeer → Anthropic ToolDefinition.
//!
//! Generates JSON Schema tool definitions from registered ToolPeers.
//! Hand-written schemas for the core tool set.
//! WASM tools auto-generate from WIT definitions.

use crate::llm::types::ToolDefinition;
use crate::wasm::definitions::WasmToolRegistry;

/// Build a ToolDefinition for the file-read tool.
pub fn file_read_definition() -> ToolDefinition {
    ToolDefinition {
        name: "file-read".into(),
        description: "Read file contents with line numbers. Supports offset and limit for large files. Detects binary files.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based, default: 1)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum lines to read (default: 2000)"
                }
            },
            "required": ["path"]
        }),
    }
}

/// Build a ToolDefinition for the file-write tool.
pub fn file_write_definition() -> ToolDefinition {
    ToolDefinition {
        name: "file-write".into(),
        description: "Write or create a file. Auto-creates parent directories.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }),
    }
}

/// Build a ToolDefinition for the file-edit tool.
pub fn file_edit_definition() -> ToolDefinition {
    ToolDefinition {
        name: "file-edit".into(),
        description: "Surgical text replacement in a file. Replaces old_string with new_string. The old_string must match exactly once. Returns unified diff.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace (must be unique in the file)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text"
                }
            },
            "required": ["path", "old_string", "new_string"]
        }),
    }
}

/// Build a ToolDefinition for the glob tool.
pub fn glob_definition() -> ToolDefinition {
    ToolDefinition {
        name: "glob".into(),
        description: "Find files matching a glob pattern (e.g. **/*.rs, src/*.txt).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match (e.g. **/*.rs)"
                },
                "base_path": {
                    "type": "string",
                    "description": "Base directory for the pattern (default: current directory)"
                }
            },
            "required": ["pattern"]
        }),
    }
}

/// Build a ToolDefinition for the grep tool.
pub fn grep_definition() -> ToolDefinition {
    ToolDefinition {
        name: "grep".into(),
        description: "Regex search across files. Recursively searches directories, skips binary files.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: current directory)"
                },
                "glob_filter": {
                    "type": "string",
                    "description": "Filter files by glob (e.g. *.rs)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)"
                }
            },
            "required": ["pattern"]
        }),
    }
}

/// Build a ToolDefinition for the bash tool.
pub fn bash_definition() -> ToolDefinition {
    ToolDefinition {
        name: "bash".into(),
        description: "Execute a shell command in the workspace. Only allowed commands can be run (cargo, git, npm, etc). Captures stdout, stderr, and exit code.".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command"
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
        "file-read" => Some(file_read_definition()),
        "file-write" => Some(file_write_definition()),
        "file-edit" => Some(file_edit_definition()),
        "glob" => Some(glob_definition()),
        "grep" => Some(grep_definition()),
        "bash" => Some(bash_definition()),
        "codebase-index" => Some(codebase_index_definition()),
        _ => None,
    }
}

/// Build tool definitions with WASM registry fallback.
///
/// Built-in tools checked first, then WASM registry fallback for unknown peers.
pub fn build_tool_definitions_with_wasm(
    peer_names: &[&str],
    wasm_registry: Option<&WasmToolRegistry>,
) -> Vec<ToolDefinition> {
    let mut defs = Vec::new();
    for name in peer_names {
        if let Some(def) = definition_for_peer(name) {
            defs.push(def);
        } else if let Some(reg) = wasm_registry {
            if let Some(def) = reg.definition_for(name) {
                defs.push(def.clone());
            }
        }
    }
    defs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_read_def_is_valid() {
        let def = file_read_definition();
        assert_eq!(def.name, "file-read");
        assert!(def.description.contains("Read"));
        assert_eq!(def.input_schema["type"], "object");
        let props = &def.input_schema["properties"];
        assert!(props.get("path").is_some());
        assert!(props.get("offset").is_some());
        assert!(props.get("limit").is_some());
        let required = def.input_schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("path")));
    }

    #[test]
    fn file_write_def_is_valid() {
        let def = file_write_definition();
        assert_eq!(def.name, "file-write");
        let props = &def.input_schema["properties"];
        assert!(props.get("path").is_some());
        assert!(props.get("content").is_some());
    }

    #[test]
    fn file_edit_def_is_valid() {
        let def = file_edit_definition();
        assert_eq!(def.name, "file-edit");
        let props = &def.input_schema["properties"];
        assert!(props.get("path").is_some());
        assert!(props.get("old_string").is_some());
        assert!(props.get("new_string").is_some());
    }

    #[test]
    fn glob_def_is_valid() {
        let def = glob_definition();
        assert_eq!(def.name, "glob");
        let props = &def.input_schema["properties"];
        assert!(props.get("pattern").is_some());
    }

    #[test]
    fn grep_def_is_valid() {
        let def = grep_definition();
        assert_eq!(def.name, "grep");
        let props = &def.input_schema["properties"];
        assert!(props.get("pattern").is_some());
        assert!(props.get("path").is_some());
        assert!(props.get("glob_filter").is_some());
        assert!(props.get("case_insensitive").is_some());
    }

    #[test]
    fn bash_def_is_valid() {
        let def = bash_definition();
        assert_eq!(def.name, "bash");
        let props = &def.input_schema["properties"];
        assert!(props.get("command").is_some());
        assert!(props.get("timeout").is_some());
        assert!(props.get("working_dir").is_some());
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
        let defs = build_tool_definitions(&[
            "file-read",
            "file-write",
            "file-edit",
            "glob",
            "grep",
            "bash",
            "codebase-index",
        ]);
        assert_eq!(defs.len(), 7);
        assert_eq!(defs[0].name, "file-read");
        assert_eq!(defs[1].name, "file-write");
        assert_eq!(defs[2].name, "file-edit");
        assert_eq!(defs[3].name, "glob");
        assert_eq!(defs[4].name, "grep");
        assert_eq!(defs[5].name, "bash");
        assert_eq!(defs[6].name, "codebase-index");
    }

    #[test]
    fn build_definitions_skips_unknown() {
        let defs = build_tool_definitions(&["file-read", "unknown-tool", "grep"]);
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "file-read");
        assert_eq!(defs[1].name, "grep");
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
        let defs = build_tool_definitions(&[
            "file-read",
            "file-write",
            "file-edit",
            "glob",
            "grep",
            "command-exec",
            "codebase-index",
        ]);
        for def in &defs {
            let json = serde_json::to_string(def).unwrap();
            assert!(json.contains(&def.name));
            let _: serde_json::Value = serde_json::from_str(&json).unwrap();
        }
    }

    // ── Phase 5: WASM tool definition integration ──

    #[test]
    fn build_with_wasm_fallback() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&crate::wasm::runtime::ToolMetadata {
            name: "echo".into(),
            description: "Echo tool".into(),
            semantic_description: "Echoes".into(),
            request_tag: "EchoRequest".into(),
            request_schema: "".into(),
            response_schema: "".into(),
            input_json_schema: r#"{"type":"object","properties":{"message":{"type":"string"}}}"#
                .into(),
        })
        .unwrap();

        // file-read from built-in, echo from WASM
        let defs = build_tool_definitions_with_wasm(&["file-read", "echo"], Some(&reg));
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "file-read");
        assert_eq!(defs[1].name, "echo");
    }

    #[test]
    fn build_wasm_only() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&crate::wasm::runtime::ToolMetadata {
            name: "echo".into(),
            description: "Echo tool".into(),
            semantic_description: "Echoes".into(),
            request_tag: "EchoRequest".into(),
            request_schema: "".into(),
            response_schema: "".into(),
            input_json_schema: r#"{"type":"object","properties":{"message":{"type":"string"}}}"#
                .into(),
        })
        .unwrap();
        reg.register(&crate::wasm::runtime::ToolMetadata {
            name: "reverse".into(),
            description: "Reverse tool".into(),
            semantic_description: "Reverses".into(),
            request_tag: "ReverseRequest".into(),
            request_schema: "".into(),
            response_schema: "".into(),
            input_json_schema: r#"{"type":"object","properties":{"text":{"type":"string"}}}"#
                .into(),
        })
        .unwrap();

        // All from WASM registry
        let defs = build_tool_definitions_with_wasm(&["echo", "reverse"], Some(&reg));
        assert_eq!(defs.len(), 2);
    }
}
