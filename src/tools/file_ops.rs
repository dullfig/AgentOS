//! FileOps stub — proves the tool-peer framework wiring.
//!
//! Returns canned responses. Real file operations come in Phase 5 (WASM sandbox).

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::{ToolPeer, ToolResponse};

/// Stub file operations tool.
pub struct FileOpsStub;

#[async_trait]
impl Handler for FileOpsStub {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        // Parse action and path from XML
        let action = extract_tag(&xml_str, "action").unwrap_or_default();
        let path = extract_tag(&xml_str, "path").unwrap_or_default();

        let response = match action.as_str() {
            "read" => ToolResponse::ok(&format!("[stub] contents of {path}")),
            "write" => ToolResponse::ok(&format!("[stub] wrote to {path}")),
            "list" => ToolResponse::ok(&format!("[stub] listing of {path}")),
            _ => ToolResponse::err(&format!("[stub] unknown action: {action}")),
        };

        Ok(HandlerResponse::Reply {
            payload_xml: response,
        })
    }
}

#[async_trait]
impl ToolPeer for FileOpsStub {
    fn name(&self) -> &str {
        "file-ops"
    }

    fn description(&self) -> &str {
        "File operations (read, write, list)"
    }

    fn request_schema(&self) -> &str {
        r#"<xs:schema>
  <xs:element name="FileOpsRequest">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="action" type="xs:string"/>
        <xs:element name="path" type="xs:string"/>
        <xs:element name="content" type="xs:string" minOccurs="0"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>"#
    }

    fn response_schema(&self) -> &str {
        r#"<xs:schema>
  <xs:element name="ToolResponse">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="success" type="xs:boolean"/>
        <xs:element name="result" type="xs:string" minOccurs="0"/>
        <xs:element name="error" type="xs:string" minOccurs="0"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>"#
    }
}

/// Extract text content between `<tag>` and `</tag>`.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml.find(&close)?;
    if start <= end {
        Some(xml[start..end].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_ops_read_stub() {
        let handler = FileOpsStub;
        let payload = ValidatedPayload {
            xml:
                b"<FileOpsRequest><action>read</action><path>/etc/hostname</path></FileOpsRequest>"
                    .to_vec(),
            tag: "FileOpsRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "file-ops".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<success>true</success>"));
                assert!(xml.contains("/etc/hostname"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn file_ops_unknown_action() {
        let handler = FileOpsStub;
        let payload = ValidatedPayload {
            xml: b"<FileOpsRequest><action>delete</action><path>/tmp/x</path></FileOpsRequest>"
                .to_vec(),
            tag: "FileOpsRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "file-ops".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<success>false</success>"));
                assert!(xml.contains("unknown action"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn file_ops_metadata() {
        let tool = FileOpsStub;
        assert_eq!(tool.name(), "file-ops");
        assert!(!tool.description().is_empty());
        assert!(tool.request_schema().contains("FileOpsRequest"));
        assert!(tool.response_schema().contains("ToolResponse"));
    }
}
