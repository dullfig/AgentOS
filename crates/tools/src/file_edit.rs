//! FileEditTool — surgical old→new text replacement with unified diff output.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use similar::{ChangeTag, TextDiff};
use std::path::Path;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Surgical text replacement in files. Returns unified diff.
pub struct FileEditTool;

#[async_trait]
impl Handler for FileEditTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let path = extract_tag(&xml_str, "path").unwrap_or_default();
        if path.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <path>"),
            });
        }

        let old_string = extract_tag(&xml_str, "old_string").unwrap_or_default();
        if old_string.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <old_string>"),
            });
        }

        let new_string = extract_tag(&xml_str, "new_string").unwrap_or_default();

        let file_path = Path::new(&path);
        if !file_path.exists() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("file not found: {path}")),
            });
        }

        let content = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("read error: {e}")),
                });
            }
        };

        // Count matches
        let match_count = content.matches(&old_string).count();

        if match_count == 0 {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("old_string not found in file"),
            });
        }

        if match_count > 1 {
            // Find line numbers of each occurrence
            let mut line_numbers = Vec::new();
            let mut search_start = 0;
            while let Some(pos) = content[search_start..].find(&old_string) {
                let abs_pos = search_start + pos;
                let line_num = content[..abs_pos].lines().count() + 1;
                line_numbers.push(line_num);
                search_start = abs_pos + 1;
            }
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "old_string has {match_count} matches (must be unique). Found at lines: {:?}",
                    line_numbers
                )),
            });
        }

        // Exactly one match — perform replacement
        let new_content = content.replacen(&old_string, &new_string, 1);

        if let Err(e) = std::fs::write(file_path, &new_content) {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("write error: {e}")),
            });
        }

        // Generate unified diff
        let diff = TextDiff::from_lines(&content, &new_content);
        let mut diff_output = String::new();
        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            diff_output.push_str(&format!("{sign}{change}"));
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&diff_output),
        })
    }
}

#[async_trait]
impl ToolPeer for FileEditTool {
    fn name(&self) -> &str {
        "file-edit"
    }

    fn wit(&self) -> &str {
        r#"
/// Surgical text replacement in a file. Replaces old_string with new_string. The old_string must match exactly once. Returns unified diff.
interface file-edit {
    record request {
        /// The file path to edit
        path: string,
        /// The exact text to find and replace (must be unique in the file)
        old-string: string,
        /// The replacement text
        new-string: string,
    }
    edit: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "file-edit".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "FileEditRequest".into(),
        }
    }

    fn get_result(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let content = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, content)
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn edit_single_match() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "fn hello() {\n    println!(\"greetings\");\n}\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>fn hello()</old_string><new_string>fn world()</new_string></FileEditRequest>"
        );
        let (ok, content) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("-fn hello()"));
        assert!(content.contains("+fn world()"));

        let new = std::fs::read_to_string(&path).unwrap();
        assert!(new.contains("fn world()"));
        // The rest of the file should be unchanged
        assert!(new.contains("println!(\"greetings\")"));
    }

    #[tokio::test]
    async fn edit_no_match() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line one\nline two\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>nonexistent</old_string><new_string>replacement</new_string></FileEditRequest>"
        );
        let (ok, content) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("not found"));
    }

    #[tokio::test]
    async fn edit_multiple_matches_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar\nfoo baz\nfoo qux\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>foo</old_string><new_string>replaced</new_string></FileEditRequest>"
        );
        let (ok, content) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("3 matches"));
        assert!(content.contains("lines:"));
    }

    #[tokio::test]
    async fn edit_multiline_replacement() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("multi.txt");
        std::fs::write(&path, "aaa\nbbb\nccc\nddd\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>bbb\nccc</old_string><new_string>BBB\nCCC\nEEE</new_string></FileEditRequest>"
        );
        let (ok, content) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("-bbb"));
        assert!(content.contains("+BBB"));

        let result = std::fs::read_to_string(&path).unwrap();
        assert_eq!(result, "aaa\nBBB\nCCC\nEEE\nddd\n");
    }

    #[tokio::test]
    async fn edit_missing_file() {
        let xml = "<FileEditRequest><path>/nonexistent/file.txt</path><old_string>x</old_string><new_string>y</new_string></FileEditRequest>";
        let (ok, content) = get_result(FileEditTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("file not found"));
    }

    #[tokio::test]
    async fn edit_missing_old_string() {
        let xml = "<FileEditRequest><path>/tmp/x</path><new_string>y</new_string></FileEditRequest>";
        let (ok, content) = get_result(FileEditTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn edit_delete_text() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("delete.txt");
        std::fs::write(&path, "keep this\nremove this\nkeep too\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>remove this\n</old_string><new_string></new_string></FileEditRequest>"
        );
        let (ok, _) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "keep this\nkeep too\n"
        );
    }

    #[tokio::test]
    async fn edit_with_xml_entities() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("entities.rs");
        std::fs::write(&path, "if a < b {\n    ok()\n}\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>a &lt; b</old_string><new_string>a &gt; b</new_string></FileEditRequest>"
        );
        let (ok, _) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "if a > b {\n    ok()\n}\n"
        );
    }

    #[tokio::test]
    async fn edit_diff_output_format() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("diff.txt");
        std::fs::write(&path, "alpha\nbeta\ngamma\n").unwrap();
        let path_str = path.to_str().unwrap();

        let xml = format!(
            "<FileEditRequest><path>{path_str}</path><old_string>beta</old_string><new_string>BETA</new_string></FileEditRequest>"
        );
        let (ok, diff) = get_result(FileEditTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        // Diff should have context lines (space prefix) and changes (+/-)
        assert!(diff.contains(" alpha\n"));
        assert!(diff.contains("-beta\n"));
        assert!(diff.contains("+BETA\n"));
        assert!(diff.contains(" gamma\n"));
    }

    #[test]
    fn file_edit_metadata() {
        let tool = FileEditTool;
        assert_eq!(tool.name(), "file-edit");
        let iface = agentos_wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "file-edit");
        assert_eq!(iface.request_tag(), "FileEditRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "path"));
    }
}
