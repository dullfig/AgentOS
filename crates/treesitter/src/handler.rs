//! CodeIndexHandler — ToolPeer wrapping CodeIndex.
//!
//! Receives XML requests for indexing and search operations.

use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::Mutex;

use super::CodeIndex;
use agentos_events::{ToolPeer, ToolResponse};

/// Pipeline handler wrapping a CodeIndex.
pub struct CodeIndexHandler {
    index: Arc<Mutex<CodeIndex>>,
}

impl CodeIndexHandler {
    pub fn new(index: Arc<Mutex<CodeIndex>>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl Handler for CodeIndexHandler {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let action = extract_tag(&xml_str, "action").unwrap_or_default();

        let response = match action.as_str() {
            "index_file" => {
                let path = extract_tag(&xml_str, "path").unwrap_or_default();
                let mut idx = self.index.lock().await;
                match idx.index_file(std::path::Path::new(&path)) {
                    Ok(count) => ToolResponse::ok(&format!("indexed {count} symbols from {path}")),
                    Err(e) => ToolResponse::err(&e),
                }
            }
            "index_directory" => {
                let path = extract_tag(&xml_str, "path").unwrap_or_default();
                let mut idx = self.index.lock().await;
                match idx.index_directory(std::path::Path::new(&path)) {
                    Ok(stats) => ToolResponse::ok(&format!(
                        "indexed {} files ({} symbols), skipped {}",
                        stats.files_indexed, stats.total_symbols, stats.files_skipped
                    )),
                    Err(e) => ToolResponse::err(&e),
                }
            }
            "search" => {
                let query = extract_tag(&xml_str, "query").unwrap_or_default();
                let kind = extract_tag(&xml_str, "kind");
                let idx = self.index.lock().await;
                let results = idx.search(&query, kind.as_deref());
                let xml = results
                    .iter()
                    .map(|(path, sym)| {
                        format!(
                            "<symbol file=\"{}\" kind=\"{}\" line=\"{}\">{}</symbol>",
                            path, sym.kind, sym.start_line, sym.name
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolResponse::ok(&format!(
                    "<symbols count=\"{}\">\n{}\n</symbols>",
                    results.len(),
                    xml
                ))
            }
            "codebase_map" => {
                let idx = self.index.lock().await;
                let map = idx.codebase_map();
                let xml = map
                    .iter()
                    .map(|entry| {
                        let syms = entry
                            .symbols
                            .iter()
                            .map(|s| format!("    <sym kind=\"{}\">{}</sym>", s.kind, s.name))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!(
                            "  <file path=\"{}\" lang=\"{}\">\n{}\n  </file>",
                            entry.path, entry.language, syms
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolResponse::ok(&format!("<codebase_map>\n{}\n</codebase_map>", xml))
            }
            _ => ToolResponse::err(&format!("unknown action: {action}")),
        };

        Ok(HandlerResponse::Reply {
            payload_xml: response,
        })
    }
}

#[async_trait]
impl ToolPeer for CodeIndexHandler {
    fn name(&self) -> &str {
        "codebase-index"
    }

    fn wit(&self) -> &str {
        r#"
/// Tree-sitter code indexing: index files/directories, search symbols, or get a codebase map.
interface codebase-index {
    record request {
        /// The indexing operation to perform
        action: string,
        /// File or directory path (for index_file, index_directory)
        path: option<string>,
        /// Search query (for search action)
        query: option<string>,
        /// Symbol kind filter (for search action, e.g. 'function', 'struct')
        kind: option<string>,
    }
    handle: func(req: request) -> result<string, string>;
}
"#
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
    async fn handler_search() {
        let index = Arc::new(Mutex::new(CodeIndex::new()));
        {
            let mut idx = index.lock().await;
            idx.index_source(
                "test.rs",
                crate::languages::Lang::Rust,
                b"pub fn hello() {} pub struct World {}",
            )
            .unwrap();
        }

        let handler = CodeIndexHandler::new(index);
        let payload = ValidatedPayload {
            xml:
                b"<CodeIndexRequest><action>search</action><query>hello</query></CodeIndexRequest>"
                    .to_vec(),
            tag: "CodeIndexRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "codebase-index".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<success>true</success>"));
                assert!(xml.contains("hello"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handler_unknown_action() {
        let index = Arc::new(Mutex::new(CodeIndex::new()));
        let handler = CodeIndexHandler::new(index);
        let payload = ValidatedPayload {
            xml: b"<CodeIndexRequest><action>bogus</action></CodeIndexRequest>".to_vec(),
            tag: "CodeIndexRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "codebase-index".into(),
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
    fn handler_metadata() {
        let index = Arc::new(tokio::sync::Mutex::new(CodeIndex::new()));
        let handler = CodeIndexHandler::new(index);
        assert_eq!(handler.name(), "codebase-index");
        let iface = agentos_wit::parser::parse_wit(handler.wit()).unwrap();
        assert_eq!(iface.name, "codebase-index");
        assert_eq!(iface.request_tag(), "CodebaseIndexRequest");
        assert!(iface.request.fields.iter().any(|f| f.name == "action"));
    }
}
