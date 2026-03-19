//! GlobTool — find files by glob pattern.

use async_trait::async_trait;
use rust_pipeline::prelude::*;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Find files matching a glob pattern.
pub struct GlobTool;

const MAX_RESULTS: usize = 1000;

#[async_trait]
impl Handler for GlobTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let pattern = extract_tag(&xml_str, "pattern").unwrap_or_default();
        if pattern.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <pattern>"),
            });
        }

        let base_path = extract_tag(&xml_str, "base_path").unwrap_or_default();

        // Build full pattern
        let full_pattern = if base_path.is_empty() {
            pattern.clone()
        } else {
            let base = base_path.trim_end_matches('/').trim_end_matches('\\');
            format!("{base}/{pattern}")
        };

        let entries = match glob::glob(&full_pattern) {
            Ok(paths) => paths,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("invalid glob pattern: {e}")),
                });
            }
        };

        let mut results: Vec<String> = Vec::new();
        let mut total = 0usize;

        for entry in entries {
            match entry {
                Ok(path) => {
                    total += 1;
                    if results.len() < MAX_RESULTS {
                        results.push(path.display().to_string());
                    }
                }
                Err(e) => {
                    // Skip unreadable entries
                    total += 1;
                    if results.len() < MAX_RESULTS {
                        results.push(format!("(error: {e})"));
                    }
                }
            }
        }

        results.sort();

        let mut output = results.join("\n");
        if total > MAX_RESULTS {
            output.push_str(&format!("\n\n... ({total} total, showing first {MAX_RESULTS})"));
        } else {
            output.push_str(&format!("\n\n{total} files matched"));
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&output),
        })
    }
}

#[async_trait]
impl ToolPeer for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn wit(&self) -> &str {
        r#"
/// Find files matching a glob pattern (e.g. **/*.rs, src/*.txt).
interface glob {
    record request {
        /// The glob pattern to match (e.g. **/*.rs)
        pattern: string,
        /// Base directory for the pattern (default: current directory)
        base-path: option<string>,
    }
    search: func(req: request) -> result<string, string>;
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
            own_name: "glob".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "GlobRequest".into(),
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
    async fn glob_finds_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();

        let pattern = dir.path().join("*.rs").to_str().unwrap().to_string();
        let xml = format!("<GlobRequest><pattern>{pattern}</pattern></GlobRequest>");
        let (ok, content) = get_result(GlobTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("a.rs"));
        assert!(content.contains("b.rs"));
        assert!(!content.contains("c.txt"));
        assert!(content.contains("2 files matched"));
    }

    #[tokio::test]
    async fn glob_with_base_path() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "").unwrap();

        let base = dir.path().to_str().unwrap();
        let xml = format!(
            "<GlobRequest><pattern>*.txt</pattern><base_path>{base}</base_path></GlobRequest>"
        );
        let (ok, content) = get_result(GlobTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("test.txt"));
    }

    #[tokio::test]
    async fn glob_no_matches() {
        let dir = TempDir::new().unwrap();
        let pattern = dir.path().join("*.nonexistent").to_str().unwrap().to_string();
        let xml = format!("<GlobRequest><pattern>{pattern}</pattern></GlobRequest>");
        let (ok, content) = get_result(GlobTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("0 files matched"));
    }

    #[tokio::test]
    async fn glob_invalid_pattern() {
        let xml = "<GlobRequest><pattern>[invalid</pattern></GlobRequest>";
        let (ok, content) = get_result(GlobTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("invalid glob"));
    }

    #[tokio::test]
    async fn glob_missing_pattern() {
        let xml = "<GlobRequest></GlobRequest>";
        let (ok, content) = get_result(GlobTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn glob_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.path().join("top.rs"), "").unwrap();
        std::fs::write(sub.join("nested.rs"), "").unwrap();

        let pattern = dir.path().join("**/*.rs").to_str().unwrap().to_string();
        let xml = format!("<GlobRequest><pattern>{pattern}</pattern></GlobRequest>");
        let (ok, content) = get_result(GlobTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("nested.rs"));
    }

    #[test]
    fn glob_metadata() {
        let tool = GlobTool;
        assert_eq!(tool.name(), "glob");
        let iface = agentos_wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "glob");
        assert_eq!(iface.request_tag(), "GlobRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "pattern"));
    }
}
