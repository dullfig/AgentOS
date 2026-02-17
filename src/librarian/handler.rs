//! LibrarianHandler — pipeline Handler wrapping the Librarian service.
//!
//! Receives XML `<LibrarianRequest>` payloads for explicit curation/scoring requests.

use std::sync::Arc;

use async_trait::async_trait;
use rust_pipeline::prelude::*;
use tokio::sync::Mutex;

use super::Librarian;
use crate::llm::types::Message;

/// Pipeline handler wrapping a Librarian.
pub struct LibrarianHandler {
    librarian: Arc<Mutex<Librarian>>,
}

impl LibrarianHandler {
    pub fn new(librarian: Arc<Mutex<Librarian>>) -> Self {
        Self { librarian }
    }
}

#[async_trait]
impl Handler for LibrarianHandler {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml_str = String::from_utf8_lossy(&payload.xml);
        let action = extract_tag(&xml_str, "action").unwrap_or_default();
        let thread_id = extract_tag(&xml_str, "thread_id").unwrap_or_default();

        let response_xml = match action.as_str() {
            "curate" => {
                let token_budget: usize = extract_tag(&xml_str, "token_budget")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8000);
                let messages = parse_messages(&xml_str);

                let lib = self.librarian.lock().await;
                match lib.curate(&thread_id, &messages, token_budget).await {
                    Ok(result) => format!(
                        "<LibrarianResponse>\
                           <paged_in>{}</paged_in>\
                           <paged_out>{}</paged_out>\
                           <working_set_tokens>{}</working_set_tokens>\
                         </LibrarianResponse>",
                        result.paged_in.join(", "),
                        result.paged_out.join(", "),
                        result.working_set_tokens,
                    ),
                    Err(e) => format!(
                        "<LibrarianResponse><error>{}</error></LibrarianResponse>",
                        xml_escape(&e.to_string())
                    ),
                }
            }
            "score" => {
                let query = extract_tag(&xml_str, "query").unwrap_or_default();
                let lib = self.librarian.lock().await;
                match lib.score_relevance(&thread_id, &query).await {
                    Ok(scores) => {
                        let scores_xml: String = scores
                            .iter()
                            .map(|(id, score)| format!("<score id=\"{id}\" value=\"{score:.2}\"/>"))
                            .collect::<Vec<_>>()
                            .join("");
                        format!(
                            "<LibrarianResponse><scores>{scores_xml}</scores></LibrarianResponse>"
                        )
                    }
                    Err(e) => format!(
                        "<LibrarianResponse><error>{}</error></LibrarianResponse>",
                        xml_escape(&e.to_string())
                    ),
                }
            }
            "inventory" => {
                let lib = self.librarian.lock().await;
                let kernel = lib.kernel.lock().await;
                match kernel.contexts().get_inventory(&thread_id) {
                    Ok(inv) => {
                        let segs_xml: String = inv
                            .segments
                            .iter()
                            .map(|s| {
                                let status = match s.status {
                                    crate::kernel::context_store::SegmentStatus::Active => "active",
                                    crate::kernel::context_store::SegmentStatus::Shelved => "shelved",
                                };
                                format!(
                                    "<segment id=\"{}\" tag=\"{}\" size=\"{}\" status=\"{}\" relevance=\"{:.2}\"/>",
                                    s.id, s.tag, s.size, status, s.relevance
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        format!(
                            "<LibrarianResponse>\
                               <inventory active=\"{}\" shelved=\"{}\" total_bytes=\"{}\">{}</inventory>\
                             </LibrarianResponse>",
                            inv.active_count, inv.shelved_count, inv.total_bytes, segs_xml
                        )
                    }
                    Err(e) => format!(
                        "<LibrarianResponse><error>{}</error></LibrarianResponse>",
                        xml_escape(&e.to_string())
                    ),
                }
            }
            _ => format!(
                "<LibrarianResponse><error>unknown action: {}</error></LibrarianResponse>",
                xml_escape(&action)
            ),
        };

        Ok(HandlerResponse::Reply {
            payload_xml: response_xml.into_bytes(),
        })
    }
}

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

fn parse_messages(xml: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = xml[search_from..].find("<message ") {
        let msg_start = search_from + pos;
        if let Some(tag_end_offset) = xml[msg_start..].find('>') {
            let tag_end = msg_start + tag_end_offset;
            let tag_str = &xml[msg_start..=tag_end];
            let role = extract_attr(tag_str, "role").unwrap_or_else(|| "user".into());
            let content_start = tag_end + 1;
            if let Some(close_offset) = xml[content_start..].find("</message>") {
                let content_end = content_start + close_offset;
                let content = xml[content_start..content_end].to_string();
                messages.push(Message::text(&role, &content));
                search_from = content_end + "</message>".len();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    messages
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::llm::LlmPool;

    #[tokio::test]
    async fn handler_inventory_on_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let pool = LlmPool::with_base_url("test".into(), "haiku", "http://localhost:19999".into());

        let kernel = Arc::new(Mutex::new(kernel));
        let pool = Arc::new(Mutex::new(pool));

        // Create thread context
        {
            let mut k = kernel.lock().await;
            k.contexts_mut().create("t1").unwrap();
        }

        let librarian = Arc::new(Mutex::new(Librarian::new(pool, kernel)));
        let handler = LibrarianHandler::new(librarian);

        let payload = ValidatedPayload {
            xml: b"<LibrarianRequest><action>inventory</action><thread_id>t1</thread_id></LibrarianRequest>"
                .to_vec(),
            tag: "LibrarianRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "librarian".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("<LibrarianResponse>"));
                assert!(xml.contains("active=\"0\""));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[tokio::test]
    async fn handler_unknown_action() {
        let dir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let pool = LlmPool::with_base_url("test".into(), "haiku", "http://localhost:19999".into());

        let kernel = Arc::new(Mutex::new(kernel));
        let pool = Arc::new(Mutex::new(pool));
        let librarian = Arc::new(Mutex::new(Librarian::new(pool, kernel)));
        let handler = LibrarianHandler::new(librarian);

        let payload = ValidatedPayload {
            xml: b"<LibrarianRequest><action>bogus</action><thread_id>t1</thread_id></LibrarianRequest>"
                .to_vec(),
            tag: "LibrarianRequest".into(),
        };
        let ctx = HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "librarian".into(),
        };

        let result = handler.handle(payload, ctx).await.unwrap();
        match result {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                assert!(xml.contains("unknown action"));
            }
            _ => panic!("expected Reply"),
        }
    }

    #[test]
    fn parse_messages_from_xml() {
        let xml = r#"<LibrarianRequest>
            <messages>
                <message role="user">Hello</message>
                <message role="assistant">Hi</message>
            </messages>
        </LibrarianRequest>"#;

        let msgs = parse_messages(xml);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content.text(), Some("Hello".into()));
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.text(), Some("Hi".into()));
    }
}
