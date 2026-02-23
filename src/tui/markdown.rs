//! Markdown rendering for the Messages pane.
//!
//! Thin wrapper around `tui-markdown` — converts markdown text to
//! styled ratatui `Line`s. Intercepts ` ```d2 ` fenced code blocks
//! and delegates to the diagram renderer for box-drawing output.

use ratatui::text::{Line, Span};

/// Parse markdown text and return styled lines suitable for a `Paragraph`.
///
/// D2 fenced code blocks are rendered as box-drawing diagrams instead of
/// plain code. All other markdown passes through `tui-markdown` as before.
pub fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let mut result = Vec::new();
    let mut remaining = text;

    while let Some(d2_start) = remaining.find("```d2") {
        // Render markdown before the D2 block
        let before = &remaining[..d2_start];
        if !before.trim().is_empty() {
            result.extend(render_markdown_raw(before));
        }

        // Find the code content start (after the ```d2 line)
        let after_marker = &remaining[d2_start + 5..];
        let code_start = match after_marker.find('\n') {
            Some(i) => d2_start + 5 + i + 1,
            None => break, // no newline after marker, treat as-is
        };

        // Find the closing ```
        let code_end = match remaining[code_start..].find("```") {
            Some(i) => code_start + i,
            None => break, // unclosed block, fall through to render as-is
        };

        let d2_source = &remaining[code_start..code_end];
        result.extend(super::diagram::render_d2(d2_source, 80));

        // Skip past the closing ``` and optional trailing newline
        let after_close = code_end + 3;
        remaining = if after_close < remaining.len() {
            &remaining[after_close..]
        } else {
            ""
        };
    }

    // Render any remaining markdown
    if !remaining.trim().is_empty() {
        result.extend(render_markdown_raw(remaining));
    }
    result
}

/// Render plain markdown via tui-markdown (no D2 interception).
fn render_markdown_raw(text: &str) -> Vec<Line<'static>> {
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

    fn lines_to_text(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

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

    #[test]
    fn render_d2_block() {
        let md = "Here is a diagram:\n\n```d2\na -> b\nb -> c\n```\n\nEnd.";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        // Should contain rendered box art, not raw D2 source
        assert!(text.contains('a'));
        assert!(text.contains('b'));
        // Should contain box-drawing characters from the renderer
        assert!(text.contains('┌') || text.contains('▼'));
    }

    #[test]
    fn render_multiple_d2_blocks() {
        let md = "First:\n\n```d2\na -> b\n```\n\nSecond:\n\n```d2\nx -> y\n```";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains('a'));
        assert!(text.contains('x'));
    }

    #[test]
    fn render_unclosed_d2_block() {
        let md = "```d2\na -> b\nno closing fence";
        let lines = render_markdown(md);
        // Should not panic — falls back to rendering as-is
        assert!(!lines.is_empty());
    }
}
