//! FileReadTool — read file contents with line numbers, offset/limit.

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use std::path::Path;

use super::{extract_tag, ToolPeer, ToolResponse};

/// Read file contents with optional offset and limit.
pub struct FileReadTool;

impl FileReadTool {
    /// Check if a byte slice looks like binary (contains null bytes in first 8KB).
    fn is_binary(data: &[u8]) -> bool {
        let check_len = data.len().min(8192);
        data[..check_len].contains(&0)
    }
}

#[async_trait]
impl Handler for FileReadTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);

        let path = extract_tag(&xml_str, "path").unwrap_or_default();
        if path.is_empty() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err("missing required <path>"),
            });
        }

        let offset = extract_tag(&xml_str, "offset")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1); // 1-based line numbering
        let limit = extract_tag(&xml_str, "limit")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2000);

        let file_path = Path::new(&path);
        if !file_path.exists() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("file not found: {path}")),
            });
        }

        if file_path.is_dir() {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!("path is a directory: {path}")),
            });
        }

        // Read raw bytes for binary detection
        let raw = match std::fs::read(file_path) {
            Ok(data) => data,
            Err(e) => {
                return Ok(HandlerResponse::Reply {
                    payload_xml: ToolResponse::err(&format!("read error: {e}")),
                });
            }
        };

        if Self::is_binary(&raw) {
            return Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&format!(
                    "binary file detected: {path} ({} bytes)",
                    raw.len()
                )),
            });
        }

        let content = String::from_utf8_lossy(&raw);
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // offset is 1-based
        let start = if offset > 0 { offset - 1 } else { 0 };
        let end = (start + limit).min(total_lines);

        let mut output = String::new();
        for (i, line) in lines.iter().enumerate().skip(start).take(end - start) {
            let line_num = i + 1;
            // Truncate long lines at 2000 chars
            if line.len() > 2000 {
                output.push_str(&format!("{line_num}| {}...\n", &line[..2000]));
            } else {
                output.push_str(&format!("{line_num}| {line}\n"));
            }
        }

        if end < total_lines {
            output.push_str(&format!(
                "\n... ({} more lines, {} total)",
                total_lines - end,
                total_lines
            ));
        }

        Ok(HandlerResponse::Reply {
            payload_xml: ToolResponse::ok(&output),
        })
    }
}

#[async_trait]
impl ToolPeer for FileReadTool {
    fn name(&self) -> &str {
        "file-read"
    }

    fn wit(&self) -> &str {
        r#"
/// Read file contents with line numbers. Supports offset and limit for large files. Detects binary files.
interface file-read {
    record request {
        /// The file path to read
        path: string,
        /// Starting line number (1-based, default: 1)
        offset: option<u32>,
        /// Maximum lines to read (default: 2000)
        limit: option<u32>,
    }
    read: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "file-read".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "FileReadRequest".into(),
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
    async fn read_basic_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line one").unwrap();
        writeln!(f, "line two").unwrap();
        writeln!(f, "line three").unwrap();

        let path = f.path().to_str().unwrap();
        let xml = format!("<FileReadRequest><path>{path}</path></FileReadRequest>");
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("1| line one"));
        assert!(content.contains("2| line two"));
        assert!(content.contains("3| line three"));
    }

    #[tokio::test]
    async fn read_with_offset() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let path = f.path().to_str().unwrap();
        let xml = format!(
            "<FileReadRequest><path>{path}</path><offset>5</offset><limit>3</limit></FileReadRequest>"
        );
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("5| line 5"));
        assert!(content.contains("6| line 6"));
        assert!(content.contains("7| line 7"));
        assert!(!content.contains("8| line 8"));
    }

    #[tokio::test]
    async fn read_truncation_note() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=100 {
            writeln!(f, "line {i}").unwrap();
        }

        let path = f.path().to_str().unwrap();
        let xml = format!(
            "<FileReadRequest><path>{path}</path><limit>10</limit></FileReadRequest>"
        );
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("... (90 more lines, 100 total)"));
    }

    #[tokio::test]
    async fn read_missing_file() {
        let xml = "<FileReadRequest><path>/nonexistent/file.txt</path></FileReadRequest>";
        let (ok, content) = get_result(FileReadTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("file not found"));
    }

    #[tokio::test]
    async fn read_missing_path_tag() {
        let xml = "<FileReadRequest></FileReadRequest>";
        let (ok, content) = get_result(FileReadTool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("missing required"));
    }

    #[tokio::test]
    async fn read_directory_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let xml = format!("<FileReadRequest><path>{path}</path></FileReadRequest>");
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("directory"));
    }

    #[tokio::test]
    async fn read_binary_file_rejected() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0x00, 0x01, 0x02, 0xFF]).unwrap();

        let path = f.path().to_str().unwrap();
        let xml = format!("<FileReadRequest><path>{path}</path></FileReadRequest>");
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(content.contains("binary file"));
    }

    #[tokio::test]
    async fn read_empty_file() {
        let f = NamedTempFile::new().unwrap();
        let path = f.path().to_str().unwrap();
        let xml = format!("<FileReadRequest><path>{path}</path></FileReadRequest>");
        let (ok, _content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
    }

    #[tokio::test]
    async fn read_default_limit_2000() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=2500 {
            writeln!(f, "line {i}").unwrap();
        }

        let path = f.path().to_str().unwrap();
        let xml = format!("<FileReadRequest><path>{path}</path></FileReadRequest>");
        let (ok, content) = get_result(FileReadTool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);
        assert!(content.contains("2000| line 2000"));
        assert!(!content.contains("2001| line 2001"));
        assert!(content.contains("500 more lines"));
    }

    #[test]
    fn file_read_metadata() {
        let tool = FileReadTool;
        assert_eq!(tool.name(), "file-read");
        let iface = agentos_wit::parser::parse_wit(tool.wit()).unwrap();
        assert_eq!(iface.name, "file-read");
        assert_eq!(iface.request_tag(), "FileReadRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "path"));
    }
}
