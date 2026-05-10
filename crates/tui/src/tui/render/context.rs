//! Context segment renderer — renders by segment tag.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use super::XmlRenderer;

/// Renders context segments based on their tag type.
pub struct ContextSegmentRenderer;

impl XmlRenderer for ContextSegmentRenderer {
    fn tags(&self) -> &[&str] {
        &["ContextSegment"]
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        // Simple rendering: show the tag and content
        Text::from(vec![Line::from(vec![
            Span::styled("segment: ", Style::default().fg(Color::Cyan)),
            Span::raw(truncate(xml, 120)),
        ])])
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}
