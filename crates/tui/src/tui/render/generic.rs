//! Generic XML renderer — fallback for unknown tags.
//!
//! Pretty-prints any XML with tag coloring.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use super::XmlRenderer;

/// Fallback renderer: pretty-prints any XML with tag coloring.
pub struct GenericXmlRenderer;

impl XmlRenderer for GenericXmlRenderer {
    fn tags(&self) -> &[&str] {
        &[] // fallback — handles everything not claimed by others
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        let decoded = decode_xml_entities(xml);
        let mut lines = Vec::new();

        for line in decoded.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with('<') {
                // Tag line — color the tag
                lines.push(Line::from(vec![Span::styled(
                    trimmed.to_string(),
                    Style::default().fg(Color::Blue),
                )]));
            } else {
                // Content line
                lines.push(Line::from(trimmed.to_string()));
            }
        }

        if lines.is_empty() {
            lines.push(Line::from(decoded));
        }

        Text::from(lines)
    }
}

/// Decode common XML entities.
fn decode_xml_entities(xml: &str) -> String {
    xml.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_any_xml() {
        let renderer = GenericXmlRenderer;
        let xml = "<Custom><data>hello world</data></Custom>";
        let text = renderer.render(xml);
        assert!(!text.lines.is_empty());
    }

    #[test]
    fn render_xml_entities() {
        let renderer = GenericXmlRenderer;
        let xml = "<Msg>&lt;b&gt;bold&lt;/b&gt; &amp; more</Msg>";
        let text = renderer.render(xml);
        // Should decode entities
        let combined: String = text.lines.iter().map(|l| l.to_string()).collect();
        assert!(combined.contains("<b>bold</b>"));
        assert!(combined.contains("& more"));
    }

    #[test]
    fn registry_dispatch_known() {
        let registry = super::super::RendererRegistry::new();
        let text = registry.render(
            "ToolResponse",
            "<ToolResponse><success>true</success><result>ok</result></ToolResponse>",
        );
        let combined: String = text.lines.iter().map(|l| l.to_string()).collect();
        assert!(combined.contains("OK"));
    }

    #[test]
    fn registry_dispatch_unknown() {
        let registry = super::super::RendererRegistry::new();
        let text = registry.render("UnknownTag", "<UnknownTag>data</UnknownTag>");
        // Should use fallback renderer
        assert!(!text.lines.is_empty());
    }
}
