//! Tool-Peer framework — protocol for tools as pipeline listeners.
//!
//! Tools don't think — they execute. Every tool-peer is a Handler,
//! but adds self-documenting metadata (name, description, schemas).

#[cfg(test)]
pub mod command_exec;
pub mod compile_wasm;
pub mod dispatch;
#[cfg(test)]
pub mod file_edit;
#[cfg(test)]
pub mod file_read;
#[cfg(test)]
pub mod file_write;
#[cfg(test)]
pub mod glob_tool;
#[cfg(test)]
pub mod grep;
pub mod list_agents;
pub mod model_download;
pub mod model_search;
pub mod model_verify;
pub mod safe_commands;
pub mod user_channel;
pub mod package_organism;
pub mod validate_organism;
pub mod vdrive_tools;

use std::collections::HashMap;

use rust_pipeline::prelude::*;

// Re-export shared tool types from events crate
pub use agentos_events::{ToolPeer, ToolResponse, extract_tag, xml_escape, xml_unescape};

/// Schema for the shared ToolResponse envelope.
/// Registered at pipeline build time so validate_stage enforces it on re-entry.
pub fn tool_response_schema() -> PayloadSchema {
    let mut fields = HashMap::new();
    fields.insert(
        "success".into(),
        FieldSchema {
            required: true,
            field_type: FieldType::String,
        },
    );
    PayloadSchema {
        root_tag: "ToolResponse".into(),
        fields,
        strict: false, // allows <result> or <error> child
    }
}

/// Schema for the AgentResponse envelope.
/// Registered at pipeline build time so validate_stage enforces it on re-entry.
pub fn agent_response_schema() -> PayloadSchema {
    PayloadSchema {
        root_tag: "AgentResponse".into(),
        fields: HashMap::new(),
        strict: false, // allows <result> or <error> child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_response_schema_validates_ok() {
        let schema = tool_response_schema();
        let xml = b"<ToolResponse><success>true</success><result>done</result></ToolResponse>";
        rust_pipeline::validation::validate_payload(xml, &schema).unwrap();
    }

    #[test]
    fn tool_response_schema_rejects_missing_success() {
        let schema = tool_response_schema();
        let xml = b"<ToolResponse><result>oops</result></ToolResponse>";
        let err = rust_pipeline::validation::validate_payload(xml, &schema);
        assert!(err.is_err(), "should reject ToolResponse without <success>");
    }

    #[test]
    fn agent_response_schema_validates_ok() {
        let schema = agent_response_schema();
        let xml = b"<AgentResponse><result>hello</result></AgentResponse>";
        rust_pipeline::validation::validate_payload(xml, &schema).unwrap();
    }

    #[test]
    fn agent_response_schema_rejects_wrong_root() {
        let schema = agent_response_schema();
        let xml = b"<WrongTag><result>hello</result></WrongTag>";
        let err = rust_pipeline::validation::validate_payload(xml, &schema);
        assert!(err.is_err(), "should reject wrong root tag");
    }

    #[test]
    fn tool_response_ok() {
        let resp = ToolResponse::ok("file contents here");
        let xml = String::from_utf8(resp).unwrap();
        assert!(xml.contains("<success>true</success>"));
        assert!(xml.contains("<result>file contents here</result>"));
    }

    #[test]
    fn tool_response_err() {
        let resp = ToolResponse::err("file not found");
        let xml = String::from_utf8(resp).unwrap();
        assert!(xml.contains("<success>false</success>"));
        assert!(xml.contains("<error>file not found</error>"));
    }

    #[test]
    fn tool_response_escapes_xml() {
        let resp = ToolResponse::ok("a < b & c > d");
        let xml = String::from_utf8(resp).unwrap();
        assert!(xml.contains("a &lt; b &amp; c &gt; d"));
    }

    #[test]
    fn extract_tag_basic() {
        let xml = "<root><name>hello</name></root>";
        assert_eq!(extract_tag(xml, "name"), Some("hello".into()));
    }

    #[test]
    fn extract_tag_with_entities() {
        let xml = "<root><val>a &lt; b</val></root>";
        assert_eq!(extract_tag(xml, "val"), Some("a < b".into()));
    }

    #[test]
    fn extract_tag_missing() {
        let xml = "<root><name>hello</name></root>";
        assert_eq!(extract_tag(xml, "missing"), None);
    }
}
