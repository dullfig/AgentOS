//! GrepTool — regex search across files.

use async_trait::async_trait;
use regex::Regex;
use rust_pipeline::prelude::*;
use std::path::Path;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Regex search across files in a directory tree.
pub struct GrepTool;

const MAX_MATCHES: usize = 500;

impl GrepTool {
    /// Check if a byte slice looks like binary (contains null bytes in first 8KB).
    fn is_binary(data: &[u8]) -> bool {
        let check_len = data.len().min(8192);
        data[..check_len].contains(&0)
    }

    /// Recursively walk a directory, searching each text file.
    fn search_dir(
        dir: &Path,
        re: &Regex,
        glob_filter: Option<&glob::Pattern>,
        results: &mut Vec<String>,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut entries_vec: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        entries_vec.sort_by_key(|e| e.path());

        for entry in entries_vec {
            if results.len() >= MAX_MATCHES {
                return;
            }

            let path = entry.path();

            // Skip hidden directories/files
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') {
                    continue;
                }
            }

            if path.is_dir() {
                Self::search_dir(&path, re, glob_filter, results);
            } else if path.is_file() {
                // Apply glob filter if present
                if let Some(filter) = glob_filter {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if !filter.matches(name) {
                            continue;
                        }
                    }
                }

                Self::search_file(&path, re, results);
            }
        }
    }

    fn search_file(path: &Path, re: &Regex, results: &mut Vec<String>) {
        let raw = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => return,
        };

        if raw.is_empty() || Self::is_binary(&raw) {
            return;
        }

        let content = String::from_utf8_lossy(&raw);
        let display = path.display();

        for (i, line) in content.lines().enumerate() {
            if results.len() >= MAX_MATCHES {
                return;
            }
            if re.is_match(line) {
                results.push(format!("{display}:{}:{line}", i + 1));
            }
        }
    }
}

#[async_trait]
impl Handler for GrepTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let pattern = extract_tag(&xml_str, "pattern").unwrap_or_default();
        if pattern.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <pattern>"),
            });
        }

        let case_insensitive = extract_tag(&xml_str, "case_insensitive")
            .map(|s| s == "true")
            .unwrap_or(false);

        let re = if case_insensitive {
            Regex::new(&format!("(?i){pattern}"))
        } else {
            Regex::new(&pattern)
        };

        let re = match re {
            Ok(r) => r,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("invalid regex: {e}")),
                });
            }
        };

        let search_path = extract_tag(&xml_str, "path").unwrap_or_else(|| ".".into());
        let glob_filter_str = extract_tag(&xml_str, "glob_filter");
        let glob_filter = glob_filter_str
            .as_ref()
            .and_then(|g| glob::Pattern::new(g).ok());

        let search = Path::new(&search_path);
        let mut results = Vec::new();

        if search.is_file() {
            Self::search_file(search, &re, &mut results);
        } else if search.is_dir() {
            Self::search_dir(search, &re, glob_filter.as_ref(), &mut results);
        } else {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("path not found: {search_path}")),
            });
        }

        let total = results.len();
        let mut output = results.join("\n");
        if total >= MAX_MATCHES {
            output.push_str(&format!("\n\n... (truncated at {MAX_MATCHES} matches)"));
        } else {
            output.push_str(&format!("\n\n{total} matches"));
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&output),
        })
    }
}

#[async_trait]
impl ToolPeer for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn wit(&self) -> &str {
        r#"
/// Regex search across files. Recursively searches directories, skips binary files.
interface grep {
    record request {
        /// Regex pattern to search for
        pattern: string,
        /// File or directory to search (default: current directory)
        path: option<string>,
        /// Filter files by glob (e.g. *.rs)
        glob-filter: option<string>,
        /// Case insensitive search (default: false)
        case-insensitive: option<bool>,
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
            own_name: "grep".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "GrepRequest".into(),
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
    async fn grep_single_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.rs");
        std::fs::write(&file, "fn hello() {}\nfn world() {}\nfn other() {}\n").unwrap();

        let path = file.to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>fn \\w+</pattern><path>{path}</path></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("fn hello"));
        assert!(content.contains("fn world"));
        assert!(content.contains("3 matches"));
    }

    #[tokio::test]
    async fn grep_directory_recursive() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("b.rs"), "fn beta() {}\n").unwrap();

        let base = dir.path().to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>fn \\w+</pattern><path>{base}</path></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("alpha"));
        assert!(content.contains("beta"));
    }

    #[tokio::test]
    async fn grep_with_glob_filter() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("code.rs"), "hello\n").unwrap();
        std::fs::write(dir.path().join("readme.md"), "hello\n").unwrap();

        let base = dir.path().to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>hello</pattern><path>{base}</path><glob_filter>*.rs</glob_filter></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("code.rs"));
        assert!(!content.contains("readme.md"));
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "Hello World\nhello world\nHELLO WORLD\n").unwrap();

        let path = file.to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>hello</pattern><path>{path}</path><case_insensitive>true</case_insensitive></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("3 matches"));
    }

    #[tokio::test]
    async fn grep_no_matches() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "alpha beta gamma\n").unwrap();

        let path = file.to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>zzz</pattern><path>{path}</path></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("0 matches"));
    }

    #[tokio::test]
    async fn grep_invalid_regex() {
        let xml = "<GrepRequest><pattern>[invalid</pattern><path>.</path></GrepRequest>";
        let (ok, content) = get_result(GrepTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("invalid regex"));
    }

    #[tokio::test]
    async fn grep_missing_pattern() {
        let xml = "<GrepRequest><path>.</path></GrepRequest>";
        let (ok, content) = get_result(GrepTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("text.txt"), "findme\n").unwrap();
        let binary_path = dir.path().join("binary.bin");
        let mut binary = vec![0u8; 100];
        binary.extend_from_slice(b"findme\n");
        std::fs::write(&binary_path, binary).unwrap();

        let base = dir.path().to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>findme</pattern><path>{base}</path></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("text.txt"));
        assert!(!content.contains("binary.bin"));
    }

    #[tokio::test]
    async fn grep_skips_hidden_dirs() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "findme\n").unwrap();
        let hidden = dir.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("secret.txt"), "findme\n").unwrap();

        let base = dir.path().to_str().unwrap();
        let xml = format!(
            "<GrepRequest><pattern>findme</pattern><path>{base}</path></GrepRequest>"
        );
        let (ok, content) = get_result(GrepTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("visible.txt"));
        assert!(!content.contains("secret.txt"));
    }

    #[test]
    fn grep_metadata() {
        let tool = GrepTool;
        assert_eq!(tool.name(), "grep");
        let iface = agentos_wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "grep");
        assert_eq!(iface.request_tag(), "GrepRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "pattern"));
    }
}
