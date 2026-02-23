//! D2 diagram renderer — parses D2 syntax and renders box-drawing art.
//!
//! LLMs emit D2 in fenced code blocks (`\`\`\`d2`). This module intercepts
//! those blocks and replaces them with deterministic box-drawing renderings.
//! No external dependencies — pure Rust, ratatui-native output.

pub mod parser;
pub mod layout;
pub mod grid;

use ratatui::text::Line;

/// Render D2 source text to styled ratatui Lines.
///
/// `d2_source` is the raw D2 text (without the fenced code markers).
/// `max_width` constrains the output to fit the terminal width.
pub fn render_d2(d2_source: &str, max_width: usize) -> Vec<Line<'static>> {
    let graph = parser::parse_d2(d2_source);
    let positioned = layout::layout(&graph, max_width);
    grid::render_to_lines(&positioned, max_width)
}
