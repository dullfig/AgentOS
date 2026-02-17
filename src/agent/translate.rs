//! JSON ↔ XML translation for tool calls and responses.
//!
//! Translates between Opus's JSON tool_use format and the pipeline's XML format.
//! Opus: `{ name: "file-ops", input: { action: "read", path: "src/main.rs" } }`
//! Pipeline: `<FileOpsRequest><action>read</action><path>src/main.rs</path></FileOpsRequest>`

/// Translate a JSON tool call from Opus into pipeline XML.
///
/// The XML tag is derived from the tool name:
/// - "file-ops" → "FileOpsRequest"
/// - "shell" → "ShellRequest"
/// - "codebase-index" → "CodeIndexRequest"
///
/// JSON object properties become child XML elements.
pub fn tool_call_to_xml(tool_name: &str, input: &serde_json::Value) -> String {
    let tag = xml_tag_for_tool(tool_name);
    let mut xml = format!("<{tag}>");

    if let Some(obj) = input.as_object() {
        for (key, value) in obj {
            let text = json_value_to_text(value);
            xml.push_str(&format!("<{key}>{}</{key}>", xml_escape(&text)));
        }
    }

    xml.push_str(&format!("</{tag}>"));
    xml
}

/// Parse a pipeline XML tool response into a plain text result.
///
/// Extracts the `<result>` or `<error>` from a `<ToolResponse>`.
/// Returns (content, is_error).
pub fn xml_response_to_result(xml: &str) -> (String, bool) {
    let success = extract_tag(xml, "success")
        .map(|s| s == "true")
        .unwrap_or(false);

    if success {
        let result = extract_tag(xml, "result").unwrap_or_else(|| "(empty result)".into());
        (result, false)
    } else {
        let error = extract_tag(xml, "error").unwrap_or_else(|| "(unknown error)".into());
        (error, true)
    }
}

/// Get the XML request tag name for a tool.
pub fn xml_tag_for_tool(tool_name: &str) -> &str {
    match tool_name {
        "file-ops" => "FileOpsRequest",
        "shell" => "ShellRequest",
        "codebase-index" => "CodeIndexRequest",
        _ => "UnknownRequest",
    }
}

/// Get the pipeline payload tag (what the listener is registered with) for a tool.
pub fn payload_tag_for_tool(tool_name: &str) -> &str {
    match tool_name {
        "file-ops" => "FileOpsRequest",
        "shell" => "ShellRequest",
        "codebase-index" => "CodeIndexRequest",
        _ => "UnknownRequest",
    }
}

/// Convert a JSON value to its text representation for XML.
fn json_value_to_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        // For objects/arrays, serialize as JSON (nested structures)
        other => other.to_string(),
    }
}

/// Extract text content between `<tag>` and `</tag>`.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
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

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Public XML escape for use by the handler when building response XML.
pub fn xml_escape_text(s: &str) -> String {
    xml_escape(s)
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_file_ops_read() {
        let input = serde_json::json!({
            "action": "read",
            "path": "src/main.rs"
        });
        let xml = tool_call_to_xml("file-ops", &input);
        assert!(xml.starts_with("<FileOpsRequest>"));
        assert!(xml.ends_with("</FileOpsRequest>"));
        assert!(xml.contains("<action>read</action>"));
        assert!(xml.contains("<path>src/main.rs</path>"));
    }

    #[test]
    fn translate_shell_command() {
        let input = serde_json::json!({
            "command": "echo hello",
            "timeout": 5000
        });
        let xml = tool_call_to_xml("shell", &input);
        assert!(xml.starts_with("<ShellRequest>"));
        assert!(xml.contains("<command>echo hello</command>"));
        assert!(xml.contains("<timeout>5000</timeout>"));
    }

    #[test]
    fn translate_codebase_index_search() {
        let input = serde_json::json!({
            "action": "search",
            "query": "parse"
        });
        let xml = tool_call_to_xml("codebase-index", &input);
        assert!(xml.starts_with("<CodeIndexRequest>"));
        assert!(xml.contains("<action>search</action>"));
        assert!(xml.contains("<query>parse</query>"));
    }

    #[test]
    fn translate_escapes_xml_chars() {
        let input = serde_json::json!({
            "command": "echo '<hello>'"
        });
        let xml = tool_call_to_xml("shell", &input);
        assert!(xml.contains("&lt;hello&gt;"));
    }

    #[test]
    fn parse_success_response() {
        let xml = "<ToolResponse><success>true</success><result>file contents here</result></ToolResponse>";
        let (content, is_error) = xml_response_to_result(xml);
        assert_eq!(content, "file contents here");
        assert!(!is_error);
    }

    #[test]
    fn parse_error_response() {
        let xml = "<ToolResponse><success>false</success><error>file not found</error></ToolResponse>";
        let (content, is_error) = xml_response_to_result(xml);
        assert_eq!(content, "file not found");
        assert!(is_error);
    }

    #[test]
    fn parse_response_with_xml_entities() {
        let xml = "<ToolResponse><success>true</success><result>a &lt; b &amp; c</result></ToolResponse>";
        let (content, is_error) = xml_response_to_result(xml);
        assert_eq!(content, "a < b & c");
        assert!(!is_error);
    }

    #[test]
    fn unknown_tool_tag() {
        let input = serde_json::json!({"x": "y"});
        let xml = tool_call_to_xml("unknown-tool", &input);
        assert!(xml.starts_with("<UnknownRequest>"));
    }

    #[test]
    fn xml_tag_mapping() {
        assert_eq!(xml_tag_for_tool("file-ops"), "FileOpsRequest");
        assert_eq!(xml_tag_for_tool("shell"), "ShellRequest");
        assert_eq!(xml_tag_for_tool("codebase-index"), "CodeIndexRequest");
        assert_eq!(xml_tag_for_tool("other"), "UnknownRequest");
    }
}
