//! Tool-Peer framework — protocol for tools as pipeline listeners.
//!
//! Tools don't think — they execute. Every tool-peer is a Handler,
//! but adds self-documenting metadata (name, description, schemas).

pub mod command_exec;
pub mod file_edit;
pub mod file_read;
pub mod file_write;
pub mod glob_tool;
pub mod grep;
pub mod vdrive_tools;

use std::collections::HashMap;

use async_trait::async_trait;
use rust_pipeline::prelude::*;

/// Marker trait for tool-peers. All tool-peers are Handlers,
/// but this trait adds tool-specific metadata for self-documentation.
///
/// Each tool declares its interface as a WIT string. The pipeline
/// builder parses it once at registration time to generate
/// PayloadSchema, ToolDefinition, and XML tag mapping.
#[async_trait]
pub trait ToolPeer: Handler {
    /// Tool name (used in routing).
    fn name(&self) -> &str;

    /// WIT interface definition for this tool.
    ///
    /// The pipeline builder parses this at registration time to derive:
    /// - PayloadSchema for pipeline validation
    /// - ToolDefinition (JSON Schema) for the LLM
    /// - XML request tag for routing
    fn wit(&self) -> &str;
}

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

/// Standard tool response envelope.
pub struct ToolResponse;

impl ToolResponse {
    /// Build a success response as XML bytes.
    pub fn ok(result: &str) -> Vec<u8> {
        format!(
            "<ToolResponse><success>true</success><result>{}</result></ToolResponse>",
            xml_escape(result)
        )
        .into_bytes()
    }

    /// Build an error response as XML bytes.
    pub fn err(error: &str) -> Vec<u8> {
        format!(
            "<ToolResponse><success>false</success><error>{}</error></ToolResponse>",
            xml_escape(error)
        )
        .into_bytes()
    }
}

/// Basic XML escaping.
pub fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Extract text content between `<tag>` and `</tag>`.
pub fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml.find(&close)?;
    if start <= end {
        Some(xml_unescape(&xml[start..end]))
    } else {
        None
    }
}

/// Unescape XML entities.
pub fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
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
