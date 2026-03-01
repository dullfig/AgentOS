//! Markdown rendering for the Messages pane.
//!
//! Thin wrapper around `tui-markdown` — converts markdown text to
//! styled ratatui `Line`s. Intercepts ` ```d2 ` fenced code blocks
//! and delegates to the diagram renderer for box-drawing output.
//! Also intercepts pipe-delimited tables (which `tui-markdown` does
//! not support) and renders them as box-drawing tables.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

/// A rendered line with a flag indicating whether it should be exempt from word-wrapping
/// (tables, diagrams, code blocks — anything that must preserve exact column layout).
pub struct TaggedLine {
    pub line: Line<'static>,
    pub nowrap: bool,
}

/// Parse markdown text and return tagged styled lines suitable for a `Paragraph`.
///
/// D2 fenced code blocks are rendered as box-drawing diagrams. Other fenced
/// code blocks are rendered via `tui-markdown` but tagged as nowrap. Pipe-
/// delimited tables are rendered as box-drawing tables (also nowrap).
/// All other markdown passes through `tui-markdown` as wrappable prose.
pub fn render_markdown(text: &str) -> Vec<TaggedLine> {
    let mut result = Vec::new();
    let mut remaining = text;

    // Process all fenced code blocks (``` with any language tag, including ```d2)
    while let Some(fence_start) = remaining.find("```") {
        // Render markdown before this fence
        let before = &remaining[..fence_start];
        if !before.trim().is_empty() {
            result.extend(render_with_tables(before));
        }

        // Parse the language tag (everything after ``` until newline)
        let after_fence = &remaining[fence_start + 3..];
        let newline_pos = match after_fence.find('\n') {
            Some(i) => i,
            None => break, // no newline after opening fence
        };
        let lang = after_fence[..newline_pos].trim();
        let code_start = fence_start + 3 + newline_pos + 1;

        // Find the closing ```
        let code_end = match remaining[code_start..].find("```") {
            Some(i) => code_start + i,
            None => break, // unclosed block, fall through to render as-is
        };

        let code_source = &remaining[code_start..code_end];

        if lang == "d2" {
            // D2 diagrams: render as box-drawing art
            for line in super::diagram::render_d2(code_source, 80) {
                result.push(TaggedLine { line, nowrap: true });
            }
        } else {
            // Other code blocks: render via tui-markdown, tag as nowrap
            let fenced = format!("```{lang}\n{code_source}```");
            for line in render_markdown_raw(&fenced) {
                result.push(TaggedLine { line, nowrap: true });
            }
        }

        // Skip past the closing ``` and optional trailing newline
        let after_close = code_end + 3;
        remaining = if after_close < remaining.len() {
            &remaining[after_close..]
        } else {
            ""
        };
    }

    // Render any remaining markdown (prose + tables)
    if !remaining.trim().is_empty() {
        result.extend(render_with_tables(remaining));
    }
    result
}

/// Split text into table and non-table chunks, rendering each appropriately.
/// A table block is a run of consecutive lines that start with `|`.
fn render_with_tables(text: &str) -> Vec<TaggedLine> {
    let mut result = Vec::new();
    let mut non_table = String::new();

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        if is_table_line(lines[i]) {
            // Flush any accumulated non-table text
            if !non_table.trim().is_empty() {
                for line in render_markdown_raw(&non_table) {
                    result.push(TaggedLine { line, nowrap: false });
                }
            }
            non_table.clear();

            // Collect consecutive table lines
            let mut table_lines = Vec::new();
            while i < lines.len() && is_table_line(lines[i]) {
                table_lines.push(lines[i]);
                i += 1;
            }
            // All table lines are nowrap
            for line in render_table_block(&table_lines) {
                result.push(TaggedLine { line, nowrap: true });
            }
        } else {
            non_table.push_str(lines[i]);
            non_table.push('\n');
            i += 1;
        }
    }

    // Flush remaining non-table text
    if !non_table.trim().is_empty() {
        for line in render_markdown_raw(&non_table) {
            result.push(TaggedLine { line, nowrap: false });
        }
    }

    result
}

/// A line belongs to a markdown table if it starts with `|`.
fn is_table_line(line: &str) -> bool {
    line.trim_start().starts_with('|')
}

/// Strip variation selector VS16 (U+FE0F) from text.
///
/// Terminals render `⚠\uFE0F` as a 2-column emoji, but `unicode-width` and
/// ratatui measure `⚠` as 1 column (text presentation). Stripping VS16 forces
/// text presentation so measurement and rendering agree. Without this, table
/// columns misalign whenever cells contain `⚠️`, `☁️`, `❤️`, etc.
fn strip_vs16(s: &str) -> String {
    s.chars().filter(|&c| c != '\u{FE0F}').collect()
}

/// Parse and render a markdown pipe-delimited table as box-drawing art.
fn render_table_block(lines: &[&str]) -> Vec<Line<'static>> {
    if lines.is_empty() {
        return Vec::new();
    }

    // Parse rows: split each line on `|`, trim cells
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut separator_indices: Vec<usize> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Strip leading/trailing pipe
        let inner = trimmed
            .strip_prefix('|')
            .unwrap_or(trimmed);
        let inner = inner.strip_suffix('|').unwrap_or(inner);

        let cells: Vec<String> = inner.split('|').map(|c| c.trim().to_string()).collect();

        // Check if this is a separator row (all cells are dashes/colons)
        let is_sep = cells.iter().all(|c| {
            let stripped = c.trim_matches(|ch: char| ch == '-' || ch == ':' || ch == ' ');
            stripped.is_empty() && !c.is_empty()
        });

        if is_sep {
            separator_indices.push(idx);
        } else {
            rows.push(cells);
        }
    }

    if rows.is_empty() {
        return Vec::new();
    }

    // Normalize cell content: strip VS16 to prevent width mismatch between
    // unicode-width measurement and terminal rendering.
    for row in &mut rows {
        for cell in row.iter_mut() {
            *cell = strip_vs16(cell);
        }
    }

    // Compute column widths using Unicode display width (handles emojis, CJK, etc.)
    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_widths: Vec<usize> = vec![0; col_count];
    for row in &rows {
        for (j, cell) in row.iter().enumerate() {
            if j < col_count {
                col_widths[j] = col_widths[j].max(cell.width());
            }
        }
    }
    // Minimum column width of 3 for aesthetics
    for w in &mut col_widths {
        *w = (*w).max(3);
    }

    let mut result = Vec::new();

    // Top border: ┌───┬───┐
    result.push(Line::from(Span::styled(
        build_border(&col_widths, '┌', '┬', '┐'),
        Style::default().fg(Color::DarkGray),
    )));

    for (i, row) in rows.iter().enumerate() {
        // Data row: │ cell │ cell │
        let mut spans = Vec::new();
        spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
        for (j, width) in col_widths.iter().enumerate() {
            let cell = row.get(j).map(|s| s.as_str()).unwrap_or("");
            // Pad using display width so emojis/CJK align correctly
            let display_w = cell.width();
            let pad = width.saturating_sub(display_w);
            let padded = format!(" {}{} ", cell, " ".repeat(pad));
            let style = if i == 0 {
                // Header row: bold cyan
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            spans.push(Span::styled(padded, style));
            spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
        }
        result.push(Line::from(spans));

        // After header row (row 0), draw a separator: ├───┼───┤
        if i == 0 && rows.len() > 1 {
            result.push(Line::from(Span::styled(
                build_border(&col_widths, '├', '┼', '┤'),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    // Bottom border: └───┴───┘
    result.push(Line::from(Span::styled(
        build_border(&col_widths, '└', '┴', '┘'),
        Style::default().fg(Color::DarkGray),
    )));

    result
}

/// Build a horizontal border line: left + (─×width + 2 padding)... + right
fn build_border(col_widths: &[usize], left: char, mid: char, right: char) -> String {
    let mut s = String::new();
    s.push(left);
    for (i, w) in col_widths.iter().enumerate() {
        // +2 for the padding spaces around the cell content
        for _ in 0..(w + 2) {
            s.push('─');
        }
        if i + 1 < col_widths.len() {
            s.push(mid);
        }
    }
    s.push(right);
    s
}

/// Render plain markdown via tui-markdown (no D2 or table interception).
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

    /// Extract just the Line refs from TaggedLines for text assertions.
    fn lines_to_text(tagged: &[TaggedLine]) -> String {
        tagged
            .iter()
            .map(|t| t.line.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn flat_text(tagged: &[TaggedLine]) -> String {
        tagged.iter().flat_map(|t| t.line.spans.iter()).map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn render_plain_text() {
        let lines = render_markdown("Hello world");
        assert!(!lines.is_empty());
        assert!(flat_text(&lines).contains("Hello world"));
    }

    #[test]
    fn render_table() {
        let md = "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains("Col A"));
        assert!(text.contains("Col B"));
        assert!(text.contains("1"));
        assert!(text.contains("2"));
        assert!(text.contains('┌'));
        assert!(text.contains('│'));
        assert!(text.contains('└'));
        // Table lines should be tagged nowrap
        assert!(lines.iter().any(|t| t.nowrap));
    }

    #[test]
    fn render_table_box_drawing() {
        let md = "| Dimension | Score |\n|---|---|\n| Architecture | 5 |\n| Security | 4 |";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains('├'));
        assert!(text.contains('┼'));
        assert!(text.contains('┤'));
        assert!(text.contains("Architecture"));
        assert!(text.contains("Security"));
    }

    #[test]
    fn render_code_block() {
        let md = "```rust\nfn main() {}\n```";
        let lines = render_markdown(md);
        assert!(!lines.is_empty());
        assert!(flat_text(&lines).contains("fn main"));
    }

    #[test]
    fn render_heading() {
        let md = "# Big Title\nSome text";
        let lines = render_markdown(md);
        let text = flat_text(&lines);
        assert!(text.contains("Big Title"));
        assert!(text.contains("Some text"));
    }

    #[test]
    fn render_mixed() {
        let md = "# Report\n\nSome prose.\n\n| A | B |\n|---|---|\n| x | y |\n\n```\ncode\n```";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains("Report"));
        assert!(text.contains("prose"));
        assert!(text.contains("x"));
        assert!(text.contains("code"));
        assert!(text.contains('┌'));
    }

    #[test]
    fn render_empty() {
        let lines = render_markdown("");
        assert!(lines.len() <= 1);
    }

    #[test]
    fn render_d2_block() {
        let md = "Here is a diagram:\n\n```d2\na -> b\nb -> c\n```\n\nEnd.";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains('a'));
        assert!(text.contains('b'));
        assert!(text.contains('┌') || text.contains('▼'));
        // D2 lines should be tagged nowrap
        assert!(lines.iter().any(|t| t.nowrap));
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
        assert!(!lines.is_empty());
    }

    #[test]
    fn render_table_with_emojis() {
        let md = "| Dim | Score |\n|---|---|\n| Arch | ⭐⭐⭐⭐⭐ Great |";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains("Arch"));
        assert!(text.contains("⭐"));
    }

    #[test]
    fn vs16_stripped_in_tables() {
        // ⚠️ = U+26A0 + U+FE0F — VS16 must be stripped for correct alignment
        let md = "| Status | Note |\n|---|---|\n| ⚠️ Partial | ok |\n| ✅ Full | ok |";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        // VS16 stripped: shows ⚠ not ⚠️, but "Partial" must be intact (not truncated)
        assert!(text.contains("Partial"), "text should not be truncated: {text}");
        // All rows must have the same display width
        let widths: Vec<usize> = lines
            .iter()
            .map(|t| t.line.spans.iter().map(|s| s.content.width()).sum::<usize>())
            .collect();
        let first = widths[0];
        for (i, w) in widths.iter().enumerate() {
            assert_eq!(*w, first, "line {i} has width {w}, expected {first}");
        }
    }

    #[test]
    fn strip_vs16_helper() {
        assert_eq!(strip_vs16("⚠\u{FE0F} Partial"), "⚠ Partial");
        assert_eq!(strip_vs16("✅ Full"), "✅ Full"); // no VS16, unchanged
        assert_eq!(strip_vs16("plain text"), "plain text");
    }

    #[test]
    fn emoji_columns_align() {
        let md = "| Dimension | Assessment |\n|---|---|\n| Architecture | ⭐⭐⭐⭐⭐ Clean |\n| Security | ⭐⭐⭐ OK |";
        let lines = render_markdown(md);
        let widths: Vec<usize> = lines
            .iter()
            .map(|t| {
                t.line.spans.iter().map(|s| s.content.width()).sum::<usize>()
            })
            .collect();
        let first = widths[0];
        for (i, w) in widths.iter().enumerate() {
            assert_eq!(*w, first, "line {i} has width {w}, expected {first}");
        }
    }

    #[test]
    fn render_single_row_table() {
        let md = "| Only | Header |\n|---|---|";
        let lines = render_markdown(md);
        let text = lines_to_text(&lines);
        assert!(text.contains("Only"));
        assert!(text.contains('┌'));
        assert!(text.contains('└'));
    }

    #[test]
    fn shorten_model_id_helper() {
        assert!(is_table_line("| a | b |"));
        assert!(is_table_line("|---|---|"));
        assert!(!is_table_line("not a table"));
        assert!(!is_table_line("some | pipe | in text"));
    }

    #[test]
    fn build_border_works() {
        let b = build_border(&[5, 3], '┌', '┬', '┐');
        assert_eq!(b, "┌───────┬─────┐");
    }

    #[test]
    fn short_header_long_body_aligns() {
        let md = "| A | B |\n|---|---|\n| Architecture | ⭐⭐⭐⭐⭐ Clean modular design |\n| Security | ⭐⭐⭐⭐ Strong |";
        let lines = render_markdown(md);
        let widths: Vec<usize> = lines
            .iter()
            .map(|t| t.line.spans.iter().map(|s| s.content.width()).sum::<usize>())
            .collect();
        let first = widths[0];
        for (i, w) in widths.iter().enumerate() {
            assert_eq!(*w, first, "line {i} has width {w}, expected {first}");
        }
    }

    #[test]
    fn uneven_column_count_aligns() {
        let md = "| A | B |\n|---|---|\n| x | y | extra |";
        let lines = render_markdown(md);
        let widths: Vec<usize> = lines
            .iter()
            .map(|t| t.line.spans.iter().map(|s| s.content.width()).sum::<usize>())
            .collect();
        let first = widths[0];
        for (i, w) in widths.iter().enumerate() {
            assert_eq!(*w, first, "line {i} has width {w}, expected {first}");
        }
    }

    #[test]
    fn d2_lines_all_nowrap() {
        let md = "```d2\na -> b\nb -> c\n```";
        let lines = render_markdown(md);
        assert!(lines.iter().all(|t| t.nowrap), "all D2 lines should be nowrap");
    }

    #[test]
    fn table_lines_all_nowrap() {
        let md = "| A | B |\n|---|---|\n| x | y |";
        let lines = render_markdown(md);
        assert!(lines.iter().all(|t| t.nowrap), "all table lines should be nowrap");
    }

    #[test]
    fn prose_lines_not_nowrap() {
        let md = "Hello world\n\nSome paragraph text.";
        let lines = render_markdown(md);
        assert!(lines.iter().all(|t| !t.nowrap), "prose lines should not be nowrap");
    }
}
