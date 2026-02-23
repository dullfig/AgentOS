//! Markdown rendering for the Messages pane.
//!
//! Thin wrapper around `tui-markdown` — converts markdown text to
//! styled ratatui `Line`s. Single point for future post-processing
//! (custom table colors, link handling, etc.).

use ratatui::text::{Line, Span};

/// Parse markdown text and return styled lines suitable for a `Paragraph`.
///
/// All data is owned (`'static` lifetime) so callers can mix freely
/// with other `Line` sources.
pub fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let rendered = tui_markdown::from_str(text);
    rendered
        .lines
        .into_iter()
        .map(|line| {
            let spans: Vec<Span<'static>> = line
                .spans
                .into_iter()
                .map(|span| Span::styled(span.content.into_owned(), span.style))
                .collect();
            Line::from(spans)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_plain_text() {
        let lines = render_markdown("Hello world");
        assert!(!lines.is_empty());
        let text: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn render_table() {
        let md = "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |";
        let lines = render_markdown(md);
        assert!(!lines.is_empty());
        let text: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Col A"));
        assert!(text.contains("1"));
    }

    #[test]
    fn render_code_block() {
        let md = "```rust\nfn main() {}\n```";
        let lines = render_markdown(md);
        assert!(!lines.is_empty());
        let text: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(text.contains("fn main"));
    }

    #[test]
    fn render_heading() {
        let md = "# Big Title\nSome text";
        let lines = render_markdown(md);
        let text: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Big Title"));
        assert!(text.contains("Some text"));
    }

    #[test]
    fn render_mixed() {
        let md = "# Report\n\nSome prose.\n\n| A | B |\n|---|---|\n| x | y |\n\n```\ncode\n```";
        let lines = render_markdown(md);
        let text: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(text.contains("Report"));
        assert!(text.contains("prose"));
        assert!(text.contains("x"));
        assert!(text.contains("code"));
    }

    #[test]
    fn render_empty() {
        let lines = render_markdown("");
        // Should not panic — empty or single blank line is fine
        assert!(lines.len() <= 1);
    }
}
