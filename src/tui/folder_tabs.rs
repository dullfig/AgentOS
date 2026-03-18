//! Folder-tab bar widget — reusable tab strip with frame integration.
//!
//! Takes two rows of screen space. Active tab connects to the content
//! frame below via a gap in the top border.
//!
//! ```text
//!  ┌─ Bob ─┐ ┌─ Coder ─┐ ┌─ Activity ─┐
//!  │       └──┘         └──┘            └──────────────┐
//! ```
//!
//! Usage:
//! ```ignore
//! let bar = FolderTabBar::new(&labels, active_idx)
//!     .scroll(scroll_offset)
//!     .shortcuts(true);     // show ^1, ^2, ^3 prefixes
//! let regions = bar.render(f, area);
//! ```

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

/// Style configuration for the tab bar.
#[derive(Clone)]
pub struct TabBarStyle {
    pub border: Style,
    pub active_label: Style,
    pub inactive_label: Style,
    pub inactive_border: Style,
}

impl Default for TabBarStyle {
    fn default() -> Self {
        Self {
            border: Style::default().fg(Color::Cyan),
            active_label: Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            inactive_label: Style::default().fg(Color::DarkGray),
            inactive_border: Style::default().fg(Color::DarkGray),
        }
    }
}

/// A click region returned after rendering — (x_start, x_end, tab_index).
pub type TabRegion = (u16, u16, usize);

/// Reusable folder-tab bar widget.
pub struct FolderTabBar<'a> {
    /// Tab labels (display text for each tab).
    labels: &'a [String],
    /// Index of the active tab.
    active: usize,
    /// Scroll offset: index of the first visible tab.
    scroll: usize,
    /// Whether to show shortcut prefixes (^1, ^2, etc.).
    shortcuts: bool,
    /// Visual style.
    style: TabBarStyle,
}

impl<'a> FolderTabBar<'a> {
    /// Create a new tab bar from labels and active index.
    pub fn new(labels: &'a [String], active: usize) -> Self {
        Self {
            labels,
            active,
            scroll: 0,
            shortcuts: true,
            style: TabBarStyle::default(),
        }
    }

    /// Set the scroll offset (first visible tab index).
    pub fn scroll(mut self, offset: usize) -> Self {
        self.scroll = offset;
        self
    }

    /// Show or hide shortcut prefixes (^1, ^2, ...).
    pub fn shortcuts(mut self, show: bool) -> Self {
        self.shortcuts = show;
        self
    }

    /// Set custom styles.
    pub fn style(mut self, style: TabBarStyle) -> Self {
        self.style = style;
        self
    }

    /// Format a tab label, optionally with shortcut prefix.
    fn format_label(&self, idx: usize, label: &str) -> String {
        if self.shortcuts {
            format!("^{} {}", idx + 1, label)
        } else {
            label.to_string()
        }
    }

    /// Render the tab bar into the given area (must be at least 2 rows tall).
    /// Returns click regions for mouse support: (x_start, x_end, tab_index).
    pub fn render(&self, f: &mut Frame, area: Rect) -> Vec<TabRegion> {
        let w = area.width as usize;
        if w < 4 || area.height < 2 || self.labels.is_empty() {
            return Vec::new();
        }

        let mut regions = Vec::new();
        let mut row1 = Vec::new();
        let mut row1_col: usize = 0;
        let mut active_tab_start: usize = 0;
        let mut active_tab_end: usize = 0;
        let mut active_visible = false;

        // Iterate visible tabs starting from scroll offset
        for i in self.scroll..self.labels.len() {
            let is_active = i == self.active;
            let label = self.format_label(i, &self.labels[i]);
            let tab_w = label.len() + 4; // "┌─" + label + "─┐"

            if row1_col + tab_w >= w {
                break; // no room
            }

            let border_style = if is_active {
                self.style.border
            } else {
                self.style.inactive_border
            };
            let label_style = if is_active {
                self.style.active_label
            } else {
                self.style.inactive_label
            };

            if is_active {
                active_tab_start = row1_col;
                active_tab_end = row1_col + tab_w;
                active_visible = true;
            }

            // Click region: label area within the tab
            let x_start = area.x + row1_col as u16 + 2; // after "┌─"
            let x_end = x_start + label.len() as u16;
            regions.push((x_start, x_end, i));

            row1.push(Span::styled("\u{250C}\u{2500}", border_style)); // "┌─"
            row1.push(Span::styled(label, label_style));
            row1.push(Span::styled("\u{2500}\u{2510}", border_style)); // "─┐"
            row1_col += tab_w;

            // Space between tabs
            if row1_col < w {
                row1.push(Span::raw(" "));
                row1_col += 1;
            }
        }

        // Fill rest of row 1
        if row1_col < w {
            row1.push(Span::raw(" ".repeat(w - row1_col)));
        }

        // If active tab scrolled off-screen, draw a plain top border
        if !active_visible {
            active_tab_start = 0;
            active_tab_end = 0;
        }

        // ── Row 2: top border with gap under active tab ──
        let row2 = self.render_border_row(w, active_tab_start, active_tab_end, active_visible);

        // Render both rows
        let row1_area = Rect { x: area.x, y: area.y, width: area.width, height: 1 };
        let row2_area = Rect { x: area.x, y: area.y + 1, width: area.width, height: 1 };
        f.render_widget(Paragraph::new(Line::from(row1)), row1_area);
        f.render_widget(Paragraph::new(Line::from(row2)), row2_area);

        regions
    }

    /// Render the border row (row 2) with gap under the active tab.
    fn render_border_row(
        &self,
        w: usize,
        active_start: usize,
        active_end: usize,
        active_visible: bool,
    ) -> Vec<Span<'static>> {
        let bs = self.style.border;
        let mut row = Vec::new();
        #[allow(unused_assignments)]
        let mut col: usize = 0;

        if !active_visible {
            // No active tab visible — plain top border
            row.push(Span::styled("\u{250C}", bs)); // ┌
            col = 1;
            if col + 1 < w {
                row.push(Span::styled("\u{2500}".repeat(w - col - 1), bs));
            }
            row.push(Span::styled("\u{2510}", bs)); // ┐
            return row;
        }

        if active_start == 0 {
            // Active tab at left edge — left wall IS the frame border
            row.push(Span::styled("\u{2502}", bs)); // │
            col = 1;
            let gap_end = active_end.saturating_sub(1);
            if gap_end > col {
                row.push(Span::raw(" ".repeat(gap_end - col)));
                col = gap_end;
            }
            row.push(Span::styled("\u{2514}", bs)); // └
            col += 1;
        } else {
            // Frame top-left corner
            row.push(Span::styled("\u{250C}", bs)); // ┌
            col = 1;
            if active_start > col {
                row.push(Span::styled("\u{2500}".repeat(active_start - col), bs));
                col = active_start;
            }
            // ┘ under active tab's ┌
            row.push(Span::styled("\u{2518}", bs));
            col += 1;
            let gap_end = active_end.saturating_sub(1);
            if gap_end > col {
                row.push(Span::raw(" ".repeat(gap_end - col)));
                col = gap_end;
            }
            // └ under active tab's ┐
            row.push(Span::styled("\u{2514}", bs));
            col += 1;
        }

        // ─ fills to the end, ┐ closes
        if col + 1 < w {
            row.push(Span::styled("\u{2500}".repeat(w - col - 1), bs));
        }
        row.push(Span::styled("\u{2510}", bs)); // ┐

        row
    }

    /// Compute the scroll offset needed to make a given tab index visible,
    /// given the available width. Call this before render() when the active
    /// tab changes.
    pub fn scroll_to_visible(
        labels: &[String],
        target: usize,
        current_scroll: usize,
        width: usize,
        shortcuts: bool,
    ) -> usize {
        if labels.is_empty() || width < 4 {
            return 0;
        }

        // If target is before current scroll, scroll to it
        if target < current_scroll {
            return target;
        }

        // Walk from current_scroll, see if target fits
        let mut col = 0;
        for i in current_scroll..labels.len() {
            let label_len = if shortcuts {
                format!("^{} {}", i + 1, labels[i]).len()
            } else {
                labels[i].len()
            };
            let tab_w = label_len + 4 + 1; // ┌─ + label + ─┐ + space

            if i == target {
                if col + tab_w <= width {
                    return current_scroll; // already visible
                } else {
                    // Need to scroll right — find new offset
                    break;
                }
            }
            col += tab_w;
        }

        // Scroll right until target fits
        let mut scroll = current_scroll;
        loop {
            let mut col = 0;
            let mut found = false;
            for i in scroll..labels.len() {
                let label_len = if shortcuts {
                    format!("^{} {}", i + 1, labels[i]).len()
                } else {
                    labels[i].len()
                };
                let tab_w = label_len + 4 + 1;
                if i == target && col + tab_w <= width {
                    found = true;
                    break;
                }
                col += tab_w;
            }
            if found || scroll >= target {
                break;
            }
            scroll += 1;
        }
        scroll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn format_label_with_shortcuts() {
        let labs = labels(&["Bob"]);
        let bar = FolderTabBar::new(&labs, 0);
        assert_eq!(bar.format_label(0, "Bob"), "^1 Bob");
        assert_eq!(bar.format_label(2, "Activity"), "^3 Activity");
    }

    #[test]
    fn format_label_without_shortcuts() {
        let labs = labels(&["Bob"]);
        let bar = FolderTabBar::new(&labs, 0).shortcuts(false);
        assert_eq!(bar.format_label(0, "Bob"), "Bob");
    }

    #[test]
    fn scroll_to_visible_no_change_needed() {
        let labs = labels(&["Bob", "Coder", "Activity"]);
        // All fit in 80 cols, target is 1
        let scroll = FolderTabBar::scroll_to_visible(&labs, 1, 0, 80, true);
        assert_eq!(scroll, 0);
    }

    #[test]
    fn scroll_to_visible_target_before_scroll() {
        let labs = labels(&["Bob", "Coder", "Activity"]);
        // Scrolled to 2, target is 0 → scroll back
        let scroll = FolderTabBar::scroll_to_visible(&labs, 0, 2, 80, true);
        assert_eq!(scroll, 0);
    }

    #[test]
    fn scroll_to_visible_narrow_width() {
        // Each tab is ~12 chars ("┌─^N Name─┐ "), so in 25 cols only ~2 fit
        let labs = labels(&["Bob", "Coder", "Activity"]);
        let scroll = FolderTabBar::scroll_to_visible(&labs, 2, 0, 25, true);
        // Should scroll right so Activity is visible
        assert!(scroll > 0);
    }

    #[test]
    fn scroll_to_visible_empty_labels() {
        let labs: Vec<String> = vec![];
        let scroll = FolderTabBar::scroll_to_visible(&labs, 0, 0, 80, true);
        assert_eq!(scroll, 0);
    }

    #[test]
    fn border_row_active_at_start() {
        let labs = labels(&["Bob", "Coder"]);
        let bar = FolderTabBar::new(&labs, 0);
        // Active at position 0: row starts with │ (left wall continues)
        let row = bar.render_border_row(40, 0, 10, true);
        let first_char: String = row[0].content.to_string();
        assert_eq!(first_char, "\u{2502}"); // │
    }

    #[test]
    fn border_row_active_in_middle() {
        let labs = labels(&["Bob", "Coder"]);
        let bar = FolderTabBar::new(&labs, 1);
        // Active not at 0: row starts with ┌ (frame corner)
        let row = bar.render_border_row(40, 10, 20, true);
        let first_char: String = row[0].content.to_string();
        assert_eq!(first_char, "\u{250C}"); // ┌
    }

    #[test]
    fn border_row_no_active_visible() {
        let labs = labels(&["Bob"]);
        let bar = FolderTabBar::new(&labs, 0);
        // No active visible: plain top border ┌──...──┐
        let row = bar.render_border_row(20, 0, 0, false);
        let first_char: String = row[0].content.to_string();
        assert_eq!(first_char, "\u{250C}"); // ┌
        let last_char: String = row.last().unwrap().content.to_string();
        assert_eq!(last_char, "\u{2510}"); // ┐
    }
}
