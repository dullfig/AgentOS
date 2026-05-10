//! Shared unicode-width-aware box-drawing utilities.
//!
//! Used by the markdown table renderer and D2 diagram renderer.
//! Single call site for width measurement — if we need to add VS16
//! stripping or other normalization later, one place to change.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Display width of a string in terminal columns.
///
/// Iterates by grapheme cluster so that multi-codepoint emoji (ZWJ sequences,
/// flags, skin-tone variants) are measured as the terminal renders them —
/// typically 2 columns — rather than summing individual codepoint widths.
pub fn display_width(s: &str) -> usize {
    s.graphemes(true)
        .map(|g| grapheme_width(g))
        .sum()
}

/// Display width of a single grapheme cluster.
///
/// Multi-codepoint clusters (ZWJ emoji, flags, etc.) are 2 columns wide in
/// terminals regardless of how many codepoints compose them.
pub fn grapheme_width(g: &str) -> usize {
    let cp_count = g.chars().count();
    if cp_count > 1 {
        // Multi-codepoint cluster: ZWJ sequence, flag, skin-tone variant → 2 columns
        2
    } else {
        // Single codepoint: defer to unicode-width
        UnicodeWidthStr::width(g)
    }
}

/// Strip Variation Selector VS16 (U+FE0F) from text.
///
/// Terminals render `⚠\uFE0F` as a 2-column emoji, but `unicode-width` and
/// ratatui measure `⚠` as 1 column (text presentation). Stripping VS16 forces
/// text presentation so measurement and rendering agree.
pub fn strip_vs16(s: &str) -> String {
    s.chars().filter(|&c| c != '\u{FE0F}').collect()
}

/// Build a horizontal border line: `left` + (`─` × width+2) for each column + `right`.
///
/// Example: `build_border(&[5, 3], '┌', '┬', '┐')` → `"┌───────┬─────┐"`.
pub fn build_border(col_widths: &[usize], left: char, mid: char, right: char) -> String {
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

/// Pad cell content to `target_width` using display-width-aware spacing.
///
/// Returns `" content<padding> "` with 1-space margins on each side.
pub fn pad_cell(content: &str, target_width: usize) -> String {
    let pad = target_width.saturating_sub(display_width(content));
    format!(" {}{} ", content, " ".repeat(pad))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_width_ascii() {
        assert_eq!(display_width("hello"), 5);
    }

    #[test]
    fn display_width_emoji() {
        // Emoji are 2 columns wide
        assert_eq!(display_width("🚀"), 2);
        assert_eq!(display_width("⭐⭐⭐"), 6);
    }

    #[test]
    fn display_width_cjk() {
        assert_eq!(display_width("漢字"), 4);
    }

    #[test]
    fn strip_vs16_removes_fe0f() {
        assert_eq!(strip_vs16("⚠\u{FE0F} Partial"), "⚠ Partial");
        assert_eq!(strip_vs16("✅ Full"), "✅ Full"); // no VS16, unchanged
        assert_eq!(strip_vs16("plain text"), "plain text");
    }

    #[test]
    fn build_border_works() {
        let b = build_border(&[5, 3], '┌', '┬', '┐');
        assert_eq!(b, "┌───────┬─────┐");
    }

    #[test]
    fn build_border_single_col() {
        let b = build_border(&[4], '└', '┴', '┘');
        assert_eq!(b, "└──────┘");
    }

    #[test]
    fn pad_cell_ascii() {
        assert_eq!(pad_cell("hi", 5), " hi    ");
    }

    #[test]
    fn pad_cell_emoji() {
        // "🚀" is 2 cols wide, target 5 → 3 spaces of padding
        assert_eq!(pad_cell("🚀", 5), " 🚀    ");
    }

    #[test]
    fn pad_cell_exact_width() {
        assert_eq!(pad_cell("abc", 3), " abc ");
    }
}
