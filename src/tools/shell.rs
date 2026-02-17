//! Shell stub — proves the tool-peer framework wiring.
//!
//! Returns canned responses. Real shell execution comes in Phase 5 (WASM sandbox).

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::{ToolPeer, ToolResponse};

/// Stub shell execution tool.
pub struct ShellStub;

#[async_trait]
impl Handler for ShellStub {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let command = extract_tag(&xml_str, "command").unwrap_or_default();
        let _timeout = extract_tag(&xml_str, "timeout")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5000);

        let response = ToolResponse::ok(&format!(
            "[stub] executed: {command}\nexit_code: 0\nstdout: (stub output)"
        ));

        Ok(HandlerResponse::Reply {
            payload_xml: response,
        })
    }
}

#[async_trait]
impl ToolPeer for ShellStub {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Shell command execution"
    }

    fn request_schema(&self) -> &str {
        r#"<xs:schema>
  <xs:element name="ShellRequest">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="command" type="xs:string"/>
        <xs:element name="timeout" type="xs:integer" minOccurs="0"/>
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
    async fn shell_stub_executes() {
        let handler = ShellStub;
        let payload = ValidatedPayload {
            xml:
                b"<ShellRequest><command>echo hello</command><timeout>5000</timeout></ShellRequest>"
                    .to_vec(),
            tag: "ShellRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "shell".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<success>true</success>"));
                assert!(xml.contains("echo hello"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn shell_metadata() {
        let tool = ShellStub;
        assert_eq!(tool.name(), "shell");
        assert!(!tool.description().is_empty());
        assert!(tool.request_schema().contains("ShellRequest"));
    }
}
