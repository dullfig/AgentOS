//! LLM Handler — pipeline Handler wrapping LlmPool.
//!
//! Receives XML `<LlmRequest>` payloads, calls the API, returns `<LlmResponse>`.
//! This is the `llm-pool` listener in the pipeline.

use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::Mutex;

use super::types::Message;
use super::LlmPool;
use crate::librarian::Librarian;

/// Pipeline handler that wraps an LlmPool.
/// Optionally holds a Librarian for auto-curation before API calls.
pub struct LlmHandler {
    pool: Arc<Mutex<LlmPool>>,
    librarian: Option<Arc<Mutex<Librarian>>>,
}

impl LlmHandler {
    pub fn new(pool: Arc<Mutex<LlmPool>>) -> Self {
        Self {
            pool,
            librarian: None,
        }
    }

    /// Create an LlmHandler with auto-curation via the Librarian.
    pub fn with_librarian(pool: Arc<Mutex<LlmPool>>, librarian: Arc<Mutex<Librarian>>) -> Self {
        Self {
            pool,
            librarian: Some(librarian),
        }
    }
}

#[async_trait]
impl Handler for LlmHandler {
    async fn handle(&self, payload: ValidatedPayload, ctx: HandlerContext) -> HandlerResult {
        // Parse XML request
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let mut request = match parse_llm_request(&xml_str) {
            Ok(r) => r,
            Err(e) => {
                let error_xml = format!(
                    "<LlmResponse><error>{}</error></LlmResponse>",
                    xml_escape(&e)
                );
                return Ok(HandlerResponse::Reply {
                    payload_xml: error_xml.into_bytes(),
                });
            }
        };

        // Auto-curation: if librarian is attached, curate context before the API call
        if let Some(ref librarian) = self.librarian {
            let lib = librarian.lock().await;
            let token_budget = request.max_tokens.saturating_sub(1000) as usize;
            let curation = lib
                .curate(&ctx.thread_id, &request.messages, token_budget)
                .await;

            if let Ok(result) = curation {
                if let Some(sys) = result.system_context {
                    request.system = Some(match request.system {
                        Some(existing) => format!("{existing}\n\n{sys}"),
                        None => sys,
                    });
                }
            }
            // If curation fails, proceed without it (graceful degradation)
        }

        // Call the pool
        let pool = self.pool.lock().await;
        let result = pool
            .complete(
                request.model.as_deref(),
                request.messages,
                request.max_tokens,
                request.system.as_deref(),
            )
            .await;

        let response_xml = match result {
            Ok(resp) => {
                let text = resp.text().unwrap_or("");
                format!(
                    "<LlmResponse>\
                       <model>{}</model>\
                       <content>{}</content>\
                       <stop_reason>{}</stop_reason>\
                       <input_tokens>{}</input_tokens>\
                       <output_tokens>{}</output_tokens>\
                     </LlmResponse>",
                    xml_escape(&resp.model),
                    xml_escape(text),
                    xml_escape(resp.stop_reason.as_deref().unwrap_or("unknown")),
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                )
            }
            Err(e) => {
                format!(
                    "<LlmResponse><error>{}</error></LlmResponse>",
                    xml_escape(&e.to_string())
                )
            }
        };

        Ok(HandlerResponse::Reply {
            payload_xml: response_xml.into_bytes(),
        })
    }
}

/// Parsed LLM request from XML.
#[derive(Debug)]
struct ParsedLlmRequest {
    model: Option<String>,
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<String>,
}

/// Parse an `<LlmRequest>` XML payload into a structured request.
fn parse_llm_request(xml: &str) -> Result<ParsedLlmRequest, String> {
    let model = extract_tag(xml, "model");
    let max_tokens_str = extract_tag(xml, "max_tokens").unwrap_or_else(|| "4096".into());
    let max_tokens: u32 = max_tokens_str
        .parse()
        .map_err(|_| format!("invalid max_tokens: {max_tokens_str}"))?;
    let system = extract_tag(xml, "system");

    // Parse messages
    let messages = parse_messages(xml)?;
    if messages.is_empty() {
        return Err("no messages in LlmRequest".into());
    }

    Ok(ParsedLlmRequest {
        model,
        max_tokens,
        messages,
        system,
    })
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

/// Parse `<messages>` block containing `<message role="...">text</message>` entries.
fn parse_messages(xml: &str) -> Result<Vec<Message>, String> {
    let mut messages = Vec::new();

    let mut search_from = 0;
    // Find `<message ` (with space) to avoid matching `<messages>`
    while let Some(pos) = xml[search_from..].find("<message ") {
        let msg_start = search_from + pos;

        // Extract role attribute
        let tag_end = xml[msg_start..]
            .find('>')
            .ok_or("malformed <message> tag")?
            + msg_start;
        let tag_str = &xml[msg_start..=tag_end];

        let role =
            extract_attribute(tag_str, "role").ok_or("missing role attribute on <message>")?;

        // Extract content (between > and </message>)
        let content_start = tag_end + 1;
        let content_end = xml[content_start..]
            .find("</message>")
            .ok_or("missing </message> close tag")?
            + content_start;

        let content = xml_unescape(&xml[content_start..content_end]);

        messages.push(Message { role, content });

        search_from = content_end + "</message>".len();
    }

    Ok(messages)
}

/// Extract an attribute value from a tag string like `<message role="user">`.
fn extract_attribute(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

/// Basic XML escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Basic XML unescaping.
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
    fn parse_llm_request_xml() {
        let xml = r#"<LlmRequest>
  <model>opus</model>
  <max_tokens>4096</max_tokens>
  <system>You are a helpful assistant.</system>
  <messages>
    <message role="user">Hello</message>
    <message role="assistant">Hi there!</message>
    <message role="user">What is 2+2?</message>
  </messages>
</LlmRequest>"#;

        let req = parse_llm_request(xml).unwrap();
        assert_eq!(req.model, Some("opus".into()));
        assert_eq!(req.max_tokens, 4096);
        assert_eq!(req.system, Some("You are a helpful assistant.".into()));
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "Hello");
        assert_eq!(req.messages[1].role, "assistant");
        assert_eq!(req.messages[2].content, "What is 2+2?");
    }

    #[test]
    fn parse_request_no_model() {
        let xml = r#"<LlmRequest>
  <max_tokens>1024</max_tokens>
  <messages>
    <message role="user">Test</message>
  </messages>
</LlmRequest>"#;

        let req = parse_llm_request(xml).unwrap();
        assert!(req.model.is_none());
        assert_eq!(req.max_tokens, 1024);
    }

    #[test]
    fn parse_request_no_messages_fails() {
        let xml = "<LlmRequest><max_tokens>100</max_tokens></LlmRequest>";
        let err = parse_llm_request(xml).unwrap_err();
        assert!(err.contains("no messages"));
    }

    #[test]
    fn xml_escape_roundtrip() {
        let original = "a < b & c > d \"e\"";
        let escaped = xml_escape(original);
        assert_eq!(escaped, "a &lt; b &amp; c &gt; d &quot;e&quot;");
        let back = xml_unescape(&escaped);
        assert_eq!(back, original);
    }

    #[test]
    fn build_response_xml() {
        let text = "Hello & goodbye";
        let model = "claude-opus-4-20250514";
        let response = format!(
            "<LlmResponse>\
               <model>{}</model>\
               <content>{}</content>\
               <stop_reason>end_turn</stop_reason>\
               <input_tokens>10</input_tokens>\
               <output_tokens>5</output_tokens>\
             </LlmResponse>",
            xml_escape(model),
            xml_escape(text),
        );
        assert!(response.contains("Hello &amp; goodbye"));
        assert!(response.contains("<model>claude-opus-4-20250514</model>"));
    }

    #[test]
    fn extract_tag_works() {
        assert_eq!(
            extract_tag("<foo><bar>baz</bar></foo>", "bar"),
            Some("baz".into())
        );
        assert_eq!(extract_tag("<foo>no bar here</foo>", "bar"), None);
    }

    #[test]
    fn extract_attribute_works() {
        assert_eq!(
            extract_attribute(r#"<message role="user">"#, "role"),
            Some("user".into())
        );
        assert_eq!(extract_attribute("<message>", "role"), None);
    }

    #[test]
    fn handler_without_librarian() {
        let pool = Arc::new(Mutex::new(crate::llm::LlmPool::with_base_url(
            "k".into(),
            "opus",
            "http://localhost:1".into(),
        )));
        let handler = LlmHandler::new(pool);
        assert!(handler.librarian.is_none());
    }

    #[test]
    fn handler_with_librarian() {
        let pool = Arc::new(Mutex::new(crate::llm::LlmPool::with_base_url(
            "k".into(),
            "haiku",
            "http://localhost:1".into(),
        )));
        let kernel =
            crate::kernel::Kernel::open(&tempfile::TempDir::new().unwrap().path().join("data"))
                .unwrap();
        let kernel = Arc::new(Mutex::new(kernel));
        let lib = Arc::new(Mutex::new(crate::librarian::Librarian::new(
            pool.clone(),
            kernel,
        )));
        let handler = LlmHandler::with_librarian(pool, lib);
        assert!(handler.librarian.is_some());
    }
}
