//! WasmToolRegistry — auto-generated ToolDefinitions from WASM metadata.
//!
//! Replaces the hand-written JSON schemas in agent/tools.rs for WASM tools.
//! Each WASM tool's metadata provides an input_json_schema that gets parsed
//! into a ToolDefinition automatically.

use std::collections::HashMap;

use agentos_events::ToolDefinition;

use super::error::WasmError;
use super::runtime::ToolMetadata;

/// Registry of auto-generated tool definitions from WASM components.
#[derive(Default)]
pub struct WasmToolRegistry {
    tools: HashMap<String, ToolDefinition>,
    tag_map: HashMap<String, String>, // tool name → XML request tag
}

impl WasmToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a WASM tool's metadata, generating a ToolDefinition.
    pub fn register(&mut self, metadata: &ToolMetadata) -> Result<(), WasmError> {
        let schema: serde_json::Value =
            serde_json::from_str(&metadata.input_json_schema).map_err(|e| {
                WasmError::Metadata(format!(
                    "invalid JSON schema for tool '{}': {e}",
                    metadata.name
                ))
            })?;

        self.tools.insert(
            metadata.name.clone(),
            ToolDefinition {
                name: metadata.name.clone(),
                description: metadata.description.clone(),
                input_schema: schema,
            },
        );
        self.tag_map
            .insert(metadata.name.clone(), metadata.request_tag.clone());

        Ok(())
    }

    /// Get the ToolDefinition for a named tool.
    pub fn definition_for(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Get the XML request tag for a named tool.
    pub fn request_tag_for(&self, name: &str) -> Option<&str> {
        self.tag_map.get(name).map(|s| s.as_str())
    }

    /// Get all registered tool definitions.
    pub fn all_definitions(&self) -> Vec<&ToolDefinition> {
        self.tools.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "echo".into(),
            description: "Echo tool".into(),
            semantic_description: "Echoes input back".into(),
            request_tag: "EchoRequest".into(),
            request_schema: "<xs:schema/>".into(),
            response_schema: "<xs:schema/>".into(),
            input_json_schema: r#"{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}"#.into(),
        }
    }

    fn reverse_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "reverse".into(),
            description: "Reverse tool".into(),
            semantic_description: "Reverses input".into(),
            request_tag: "ReverseRequest".into(),
            request_schema: "<xs:schema/>".into(),
            response_schema: "<xs:schema/>".into(),
            input_json_schema: r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#.into(),
        }
    }

    #[test]
    fn empty_registry() {
        let reg = WasmToolRegistry::new();
        assert!(reg.all_definitions().is_empty());
        assert!(reg.definition_for("anything").is_none());
    }

    #[test]
    fn register_tool() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&echo_metadata()).unwrap();
        assert!(reg.definition_for("echo").is_some());
    }

    #[test]
    fn valid_tool_definition() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&echo_metadata()).unwrap();
        let def = reg.definition_for("echo").unwrap();
        assert_eq!(def.name, "echo");
        assert_eq!(def.description, "Echo tool");
        assert_eq!(def.input_schema["type"], "object");
    }

    #[test]
    fn invalid_json_schema_fails() {
        let mut reg = WasmToolRegistry::new();
        let mut bad = echo_metadata();
        bad.input_json_schema = "not valid json {{{".into();
        let result = reg.register(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn request_tag_registered() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&echo_metadata()).unwrap();
        assert_eq!(reg.request_tag_for("echo"), Some("EchoRequest"));
    }

    #[test]
    fn all_definitions() {
        let mut reg = WasmToolRegistry::new();
        reg.register(&echo_metadata()).unwrap();
        reg.register(&reverse_metadata()).unwrap();
        let all = reg.all_definitions();
        assert_eq!(all.len(), 2);
    }
}
