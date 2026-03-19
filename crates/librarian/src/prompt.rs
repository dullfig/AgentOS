//! Haiku prompt templates for curation decisions.
//!
//! Builds structured prompts for Haiku to decide what to page in/out.
//! Responses are XML-structured for reliable parsing.

use agentos_kernel::context_store::{ContextInventory, SegmentStatus};
use agentos_events::Message;

/// System prompt for curation requests.
pub const CURATION_SYSTEM: &str = "\
You are a context librarian. Your job is to decide which context segments should be \
active (paged in), shelved (paged out), or folded (compressed to summary) for an upcoming LLM call. \
Folded segments retain a summary visible to the model; the full content can be unfolded on demand. \
Use folding for context that may be needed later but is not immediately relevant. \
Consider the incoming messages and the segment metadata to make your decision. \
Stay within the token budget. Respond ONLY with a CurationDecision XML block.";

/// System prompt for scoring requests.
pub const SCORING_SYSTEM: &str = "\
You are a relevance scorer. Score each context segment from 0.0 to 1.0 based on \
how relevant it is to the given query. Respond ONLY with a ScoringResult XML block.";

/// A parsed curation decision from Haiku.
#[derive(Debug, Clone)]
pub struct CurationDecision {
    pub page_in: Vec<String>,
    pub page_out: Vec<String>,
    pub fold: Vec<String>,
    pub unfold: Vec<String>,
}

/// Build the curation prompt sent to Haiku.
pub fn build_curation_prompt(
    inventory: &ContextInventory,
    incoming_messages: &[Message],
    token_budget: usize,
) -> String {
    let mut prompt = String::new();

    prompt.push_str("<CurationRequest>\n");
    prompt.push_str(&format!("  <token_budget>{token_budget}</token_budget>\n"));

    // Incoming messages
    prompt.push_str("  <incoming_messages>\n");
    for msg in incoming_messages {
        let text = msg.content.text().unwrap_or_default();
        prompt.push_str(&format!(
            "    <message role=\"{}\">{}</message>\n",
            msg.role,
            truncate(&text, 500)
        ));
    }
    prompt.push_str("  </incoming_messages>\n");

    // Context inventory
    prompt.push_str("  <inventory>\n");
    for seg in &inventory.segments {
        let status_str = match seg.status {
            SegmentStatus::Active => "active",
            SegmentStatus::Shelved => "shelved",
            SegmentStatus::Folded => "folded",
        };
        prompt.push_str(&format!(
            "    <segment id=\"{}\" tag=\"{}\" size=\"{}\" status=\"{}\" relevance=\"{:.2}\"/>\n",
            seg.id, seg.tag, seg.size, status_str, seg.relevance
        ));
    }
    prompt.push_str("  </inventory>\n");

    prompt.push_str(&format!(
        "  <summary active=\"{}\" shelved=\"{}\" folded=\"{}\" active_bytes=\"{}\" total_bytes=\"{}\"/>\n",
        inventory.active_count,
        inventory.shelved_count,
        inventory.folded_count,
        inventory.active_bytes,
        inventory.total_bytes
    ));

    prompt.push_str("</CurationRequest>");
    prompt
}

/// Build a scoring prompt for relevance scoring.
pub fn build_scoring_prompt(inventory: &ContextInventory, query: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("<ScoringRequest>\n");
    prompt.push_str(&format!("  <query>{}</query>\n", truncate(query, 500)));
    prompt.push_str("  <segments>\n");
    for seg in &inventory.segments {
        prompt.push_str(&format!(
            "    <segment id=\"{}\" tag=\"{}\" size=\"{}\"/>\n",
            seg.id, seg.tag, seg.size
        ));
    }
    prompt.push_str("  </segments>\n");
    prompt.push_str("</ScoringRequest>");
    prompt
}

/// Parse Haiku's curation response.
pub fn parse_curation_response(response: &str) -> Result<CurationDecision, String> {
    let mut page_in = Vec::new();
    let mut page_out = Vec::new();
    let mut fold = Vec::new();
    let mut unfold = Vec::new();

    // Parse <page_in> section
    if let Some(section) = extract_section(response, "page_in") {
        for id in extract_segment_ids(&section) {
            page_in.push(id);
        }
    }

    // Parse <page_out> section
    if let Some(section) = extract_section(response, "page_out") {
        for id in extract_segment_ids(&section) {
            page_out.push(id);
        }
    }

    // Parse <fold> section
    if let Some(section) = extract_section(response, "fold") {
        for id in extract_segment_ids(&section) {
            fold.push(id);
        }
    }

    // Parse <unfold> section
    if let Some(section) = extract_section(response, "unfold") {
        for id in extract_segment_ids(&section) {
            unfold.push(id);
        }
    }

    Ok(CurationDecision { page_in, page_out, fold, unfold })
}

/// Parse Haiku's scoring response.
pub fn parse_scoring_response(response: &str) -> Result<Vec<(String, f32)>, String> {
    let mut scores = Vec::new();

    // Look for <score id="..." value="..."/> patterns
    let mut search_from = 0;
    while let Some(pos) = response[search_from..].find("<score ") {
        let start = search_from + pos;
        if let Some(end) = response[start..].find("/>") {
            let tag = &response[start..start + end + 2];
            if let (Some(id), Some(value)) = (extract_attr(tag, "id"), extract_attr(tag, "value")) {
                if let Ok(score) = value.parse::<f32>() {
                    scores.push((id, score.clamp(0.0, 1.0)));
                }
            }
            search_from = start + end + 2;
        } else {
            break;
        }
    }

    Ok(scores)
}

// ── Helpers ──

fn extract_section(xml: &str, tag: &str) -> Option<String> {
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

fn extract_segment_ids(section: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = section[search_from..].find("<segment ") {
        let start = search_from + pos;
        if let Some(end) = section[start..].find("/>") {
            let tag = &section[start..start + end + 2];
            if let Some(id) = extract_attr(tag, "id") {
                ids.push(id);
            }
            search_from = start + end + 2;
        } else {
            break;
        }
    }
    ids
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentos_kernel::context_store::SegmentMeta;

    fn sample_inventory() -> ContextInventory {
        ContextInventory {
            thread_id: "t1".into(),
            segments: vec![
                SegmentMeta {
                    id: "code:parser.rs".into(),
                    tag: "code".into(),
                    size: 2000,
                    status: SegmentStatus::Shelved,
                    relevance: 0.3,
                    created_at: 1000,
                },
                SegmentMeta {
                    id: "msg-001".into(),
                    tag: "message".into(),
                    size: 500,
                    status: SegmentStatus::Active,
                    relevance: 0.8,
                    created_at: 2000,
                },
                SegmentMeta {
                    id: "map:crate".into(),
                    tag: "codebase-map".into(),
                    size: 1000,
                    status: SegmentStatus::Shelved,
                    relevance: 0.5,
                    created_at: 500,
                },
            ],
            active_count: 1,
            shelved_count: 2,
            folded_count: 0,
            total_bytes: 3500,
            active_bytes: 500,
        }
    }

    #[test]
    fn build_curation_prompt_includes_inventory() {
        let inv = sample_inventory();
        let msgs = vec![Message::text("user", "What does the parser do?")];

        let prompt = build_curation_prompt(&inv, &msgs, 8000);
        assert!(prompt.contains("<CurationRequest>"));
        assert!(prompt.contains("<token_budget>8000</token_budget>"));
        assert!(prompt.contains("code:parser.rs"));
        assert!(prompt.contains("msg-001"));
        assert!(prompt.contains("map:crate"));
        assert!(prompt.contains("What does the parser do?"));
    }

    #[test]
    fn parse_curation_response_xml() {
        let response = r#"
<CurationDecision>
  <page_in>
    <segment id="code:parser.rs" reason="user asking about parsing"/>
    <segment id="map:crate" reason="need codebase overview"/>
  </page_in>
  <page_out>
    <segment id="msg-old-003" reason="stale conversation"/>
  </page_out>
</CurationDecision>"#;

        let decision = parse_curation_response(response).unwrap();
        assert_eq!(decision.page_in, vec!["code:parser.rs", "map:crate"]);
        assert_eq!(decision.page_out, vec!["msg-old-003"]);
    }

    #[test]
    fn parse_empty_curation_response() {
        let decision = parse_curation_response("no xml here").unwrap();
        assert!(decision.page_in.is_empty());
        assert!(decision.page_out.is_empty());
    }

    #[test]
    fn parse_scoring_response_xml() {
        let response = r#"
<ScoringResult>
  <score id="code:parser.rs" value="0.9"/>
  <score id="msg-001" value="0.3"/>
  <score id="map:crate" value="0.7"/>
</ScoringResult>"#;

        let scores = parse_scoring_response(response).unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].0, "code:parser.rs");
        assert!((scores[0].1 - 0.9).abs() < f32::EPSILON);
        assert_eq!(scores[1].0, "msg-001");
        assert!((scores[1].1 - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn scoring_clamps_to_range() {
        let response = r#"<score id="s1" value="1.5"/><score id="s2" value="-0.3"/>"#;
        let scores = parse_scoring_response(response).unwrap();
        assert!((scores[0].1 - 1.0).abs() < f32::EPSILON);
        assert!((scores[1].1 - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_scoring_prompt_includes_query() {
        let inv = sample_inventory();
        let prompt = build_scoring_prompt(&inv, "parsing");
        assert!(prompt.contains("<ScoringRequest>"));
        assert!(prompt.contains("<query>parsing</query>"));
        assert!(prompt.contains("code:parser.rs"));
    }

    // ── Folding Context tests (Milestone 3) ──

    #[test]
    fn curation_prompt_includes_folded_status() {
        let inv = ContextInventory {
            thread_id: "t1".into(),
            segments: vec![
                SegmentMeta {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Folded,
                    relevance: 0.5,
                    created_at: 0,
                },
            ],
            active_count: 0,
            shelved_count: 0,
            folded_count: 1,
            total_bytes: 100,
            active_bytes: 0,
        };
        let msgs = vec![Message::text("user", "test")];
        let prompt = build_curation_prompt(&inv, &msgs, 8000);
        assert!(prompt.contains("status=\"folded\""));
    }

    #[test]
    fn parse_fold_decision() {
        let response = r#"
<CurationDecision>
  <fold>
    <segment id="code:old.rs" reason="no longer relevant"/>
    <segment id="msg-002" reason="stale"/>
  </fold>
</CurationDecision>"#;

        let decision = parse_curation_response(response).unwrap();
        assert_eq!(decision.fold, vec!["code:old.rs", "msg-002"]);
    }

    #[test]
    fn parse_unfold_decision() {
        let response = r#"
<CurationDecision>
  <unfold>
    <segment id="code:needed.rs" reason="user asks about this"/>
  </unfold>
</CurationDecision>"#;

        let decision = parse_curation_response(response).unwrap();
        assert_eq!(decision.unfold, vec!["code:needed.rs"]);
    }

    #[test]
    fn parse_empty_fold_unfold() {
        let response = r#"
<CurationDecision>
  <page_in>
    <segment id="s1"/>
  </page_in>
</CurationDecision>"#;

        let decision = parse_curation_response(response).unwrap();
        assert!(decision.fold.is_empty());
        assert!(decision.unfold.is_empty());
        assert_eq!(decision.page_in, vec!["s1"]);
    }

    #[test]
    fn curation_decision_has_fold_fields() {
        let decision = CurationDecision {
            page_in: vec!["a".into()],
            page_out: vec!["b".into()],
            fold: vec!["c".into()],
            unfold: vec!["d".into()],
        };
        assert_eq!(decision.fold.len(), 1);
        assert_eq!(decision.unfold.len(), 1);
    }

    #[test]
    fn librarian_system_prompt_mentions_folding() {
        assert!(CURATION_SYSTEM.contains("folded"));
        assert!(CURATION_SYSTEM.contains("summary"));
        assert!(CURATION_SYSTEM.contains("unfold"));
    }

    #[test]
    fn build_curation_prompt_summary_line() {
        let inv = ContextInventory {
            thread_id: "t1".into(),
            segments: vec![],
            active_count: 2,
            shelved_count: 1,
            folded_count: 3,
            total_bytes: 5000,
            active_bytes: 3000,
        };
        let prompt = build_curation_prompt(&inv, &[], 8000);
        assert!(prompt.contains("folded=\"3\""));
    }
}
