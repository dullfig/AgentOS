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
    tool_call_to_xml_with_tag(&tag, input)
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
///
/// Known tools return static strings. Unknown tools get a dynamic PascalCase
/// tag derived from the kebab-case name (e.g., "email-sender" → "EmailSenderRequest").
pub fn xml_tag_for_tool(tool_name: &str) -> String {
    match tool_name {
        "file-read" => "FileReadRequest".to_string(),
        "file-write" => "FileWriteRequest".to_string(),
        "file-edit" => "FileEditRequest".to_string(),
        "glob" => "GlobRequest".to_string(),
        "grep" => "GrepRequest".to_string(),
        "bash" => "BashRequest".to_string(),
        "codebase-index" => "CodeIndexRequest".to_string(),
        _ => to_pascal_case_request(tool_name),
    }
}

/// Get the pipeline payload tag (what the listener is registered with) for a tool.
pub fn payload_tag_for_tool(tool_name: &str) -> String {
    match tool_name {
        "file-read" => "FileReadRequest".to_string(),
        "file-write" => "FileWriteRequest".to_string(),
        "file-edit" => "FileEditRequest".to_string(),
        "glob" => "GlobRequest".to_string(),
        "grep" => "GrepRequest".to_string(),
        "bash" => "BashRequest".to_string(),
        "codebase-index" => "CodeIndexRequest".to_string(),
        _ => to_pascal_case_request(tool_name),
    }
}

/// Convert a kebab-case tool name to PascalCaseRequest.
/// "email-sender" → "EmailSenderRequest"
/// "my-tool" → "MyToolRequest"
fn to_pascal_case_request(name: &str) -> String {
    let pascal: String = name
        .split('-')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(ch) => ch.to_uppercase().to_string() + chars.as_str(),
            }
        })
        .collect();
    format!("{pascal}Request")
}

/// Translate a JSON tool call to XML with a custom tag name.
///
/// Used for WASM tools where the tag comes from the WasmToolRegistry
/// rather than the hardcoded xml_tag_for_tool() map.
pub fn tool_call_to_xml_with_tag(tag: &str, input: &serde_json::Value) -> String {
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
    fn translate_file_read() {
        let input = serde_json::json!({
            "path": "src/main.rs",
            "limit": 100
        });
        let xml = tool_call_to_xml("file-read", &input);
        assert!(xml.starts_with("<FileReadRequest>"));
        assert!(xml.ends_with("</FileReadRequest>"));
        assert!(xml.contains("<path>src/main.rs</path>"));
    }

    #[test]
    fn translate_file_write() {
        let input = serde_json::json!({
            "path": "output.txt",
            "content": "hello world"
        });
        let xml = tool_call_to_xml("file-write", &input);
        assert!(xml.starts_with("<FileWriteRequest>"));
        assert!(xml.contains("<content>hello world</content>"));
    }

    #[test]
    fn translate_file_edit() {
        let input = serde_json::json!({
            "path": "src/main.rs",
            "old_string": "fn old()",
            "new_string": "fn new()"
        });
        let xml = tool_call_to_xml("file-edit", &input);
        assert!(xml.starts_with("<FileEditRequest>"));
        assert!(xml.contains("<old_string>fn old()</old_string>"));
        assert!(xml.contains("<new_string>fn new()</new_string>"));
    }

    #[test]
    fn translate_glob() {
        let input = serde_json::json!({
            "pattern": "**/*.rs"
        });
        let xml = tool_call_to_xml("glob", &input);
        assert!(xml.starts_with("<GlobRequest>"));
        assert!(xml.contains("<pattern>**/*.rs</pattern>"));
    }

    #[test]
    fn translate_grep() {
        let input = serde_json::json!({
            "pattern": "fn\\s+\\w+",
            "path": "src/"
        });
        let xml = tool_call_to_xml("grep", &input);
        assert!(xml.starts_with("<GrepRequest>"));
    }

    #[test]
    fn translate_command_exec() {
        let input = serde_json::json!({
            "command": "cargo test",
            "timeout": 60
        });
        let xml = tool_call_to_xml("command-exec", &input);
        assert!(xml.starts_with("<CommandExecRequest>"));
        assert!(xml.contains("<command>cargo test</command>"));
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
        let xml = tool_call_to_xml("command-exec", &input);
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
    fn unknown_tool_dynamic_pascal_case() {
        let input = serde_json::json!({"x": "y"});
        let xml = tool_call_to_xml("unknown-tool", &input);
        assert!(xml.starts_with("<UnknownToolRequest>"));
        assert!(xml.ends_with("</UnknownToolRequest>"));
    }

    #[test]
    fn xml_tag_mapping() {
        assert_eq!(xml_tag_for_tool("file-read"), "FileReadRequest");
        assert_eq!(xml_tag_for_tool("file-write"), "FileWriteRequest");
        assert_eq!(xml_tag_for_tool("file-edit"), "FileEditRequest");
        assert_eq!(xml_tag_for_tool("glob"), "GlobRequest");
        assert_eq!(xml_tag_for_tool("grep"), "GrepRequest");
        assert_eq!(xml_tag_for_tool("command-exec"), "CommandExecRequest");
        assert_eq!(xml_tag_for_tool("codebase-index"), "CodeIndexRequest");
    }

    #[test]
    fn dynamic_pascal_case_tags() {
        assert_eq!(xml_tag_for_tool("email-sender"), "EmailSenderRequest");
        assert_eq!(xml_tag_for_tool("my-cool-tool"), "MyCoolToolRequest");
        assert_eq!(xml_tag_for_tool("simple"), "SimpleRequest");
    }

    // ── Phase 5: Dynamic XML tag for WASM tools ──

    #[test]
    fn tool_call_to_xml_with_custom_tag() {
        let input = serde_json::json!({
            "message": "hello world"
        });
        let xml = tool_call_to_xml_with_tag("EchoRequest", &input);
        assert!(xml.starts_with("<EchoRequest>"));
        assert!(xml.ends_with("</EchoRequest>"));
        assert!(xml.contains("<message>hello world</message>"));
    }

    #[test]
    fn tool_call_to_xml_with_tag_escapes() {
        let input = serde_json::json!({
            "content": "a < b & c > d"
        });
        let xml = tool_call_to_xml_with_tag("MyToolRequest", &input);
        assert!(xml.contains("a &lt; b &amp; c &gt; d"));
    }
}
