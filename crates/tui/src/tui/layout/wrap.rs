//! Cursor positioning and word-wrapping helpers.

use ratatui::text::{Line, Span};

/// Compute cursor (col, row) for plain unwrapped text.
/// Used for the external input bar (single-line, no wrapping).
pub(super) fn plain_cursor_xy(content: &str, cursor_char: usize) -> (u16, u16) {
    use unicode_width::UnicodeWidthChar;
    let col: usize = content
        .chars()
        .take(cursor_char)
        .map(|c| c.width().unwrap_or(1))
        .sum();
    (col as u16, 0)
}

/// Compute cursor (col, row) within text that was wrapped by `wrap_line`.
/// Walks through wrapped lines to find which one contains the cursor.
pub(super) fn wrapped_cursor_xy(wrapped: &[Line], cursor_char: usize) -> (u16, u16) {
    use unicode_width::UnicodeWidthStr;
    let mut chars_so_far: usize = 0;
    for (row, wline) in wrapped.iter().enumerate() {
        let line_text: String = wline.spans.iter().map(|s| s.content.as_ref()).collect();
        let line_char_count = line_text.chars().count();
        if chars_so_far + line_char_count > cursor_char || row == wrapped.len() - 1 {
            let offset = cursor_char.saturating_sub(chars_so_far);
            let prefix: String = line_text.chars().take(offset).collect();
            return (prefix.width() as u16, row as u16);
        }
        chars_so_far += line_char_count;
    }
    (0, 0)
}

/// Compute cursor (col, row) for content that may contain newlines.
/// Splits on `\n`, wraps each line, and maps cursor char offset to visual position.
pub(super) fn multiline_cursor_xy(content: &str, cursor_char: usize, wrap_width: usize) -> (u16, u16) {
    let mut chars_consumed: usize = 0;
    let mut visual_row: u16 = 0;

    for (line_idx, raw_line) in content.split('\n').enumerate() {
        let line_chars = raw_line.chars().count();

        if chars_consumed + line_chars >= cursor_char {
            // Cursor is within this raw line
            let offset_in_line = cursor_char - chars_consumed;
            let wrapped = wrap_line(Line::from(raw_line.to_string()), wrap_width);
            let (cx, cy) = wrapped_cursor_xy(&wrapped, offset_in_line);
            return (cx, visual_row + cy);
        }

        // This line's visual height
        let wrapped = wrap_line(Line::from(raw_line.to_string()), wrap_width);
        visual_row += wrapped.len() as u16;

        // +1 for the \n character
        chars_consumed += line_chars + 1;
        let _ = line_idx;
    }
    (0, visual_row)
}

/// Word-wrap a single `Line` at `max_width` display columns, preserving span styles.
/// Returns the line unchanged if it already fits.
pub(super) fn wrap_line(line: Line<'static>, max_width: usize) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthChar;
    use unicode_width::UnicodeWidthStr;

    if max_width == 0 {
        return vec![line];
    }

    // Fast path: line fits — no wrapping needed
    let total: usize = line.spans.iter().map(|s| s.content.width()).sum();
    if total <= max_width {
        return vec![line];
    }

    let mut result: Vec<Line<'static>> = Vec::new();
    let mut current_spans: Vec<Span<'static>> = Vec::new();
    let mut current_width: usize = 0;

    for span in line.spans {
        let style = span.style;
        let text: String = span.content.into();

        let mut remaining = text.as_str();
        while !remaining.is_empty() {
            let rem_width = remaining.width();

            if current_width + rem_width <= max_width {
                // Entire remaining text fits on current line
                current_spans.push(Span::styled(remaining.to_string(), style));
                current_width += rem_width;
                break;
            }

            // Need to split — walk characters to find break point
            let available = max_width.saturating_sub(current_width);
            let mut col: usize = 0;
            let mut byte_at_limit: usize = 0;
            let mut last_space_byte: Option<usize> = None;

            for (i, ch) in remaining.char_indices() {
                let ch_w = ch.width().unwrap_or(0);
                if col + ch_w > available {
                    break;
                }
                col += ch_w;
                byte_at_limit = i + ch.len_utf8();
                if ch == ' ' {
                    last_space_byte = Some(i + ch.len_utf8());
                }
            }

            let split_at = last_space_byte.unwrap_or(byte_at_limit);

            if split_at == 0 {
                if current_spans.is_empty() {
                    // Can't fit even one char — take one to avoid infinite loop
                    let ch = remaining.chars().next().unwrap();
                    current_spans.push(Span::styled(ch.to_string(), style));
                    remaining = &remaining[ch.len_utf8()..];
                }
                // Flush current line
                result.push(Line::from(std::mem::take(&mut current_spans)));
                current_width = 0;
            } else {
                let (before, after) = remaining.split_at(split_at);
                if !before.is_empty() {
                    current_spans.push(Span::styled(before.to_string(), style));
                }
                result.push(Line::from(std::mem::take(&mut current_spans)));
                current_width = 0;
                remaining = after;
            }
        }
    }

    if !current_spans.is_empty() {
        result.push(Line::from(current_spans));
    }

    if result.is_empty() {
        result.push(Line::from(""));
    }

    result
}
