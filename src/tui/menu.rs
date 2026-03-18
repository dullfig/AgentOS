//! Menu bar widget — declarative Windows-style menu with accelerator keys.
//!
//! Define menus with `&` accelerator markers in labels. The widget handles
//! rendering (with underlined accelerators), keyboard navigation (arrows,
//! Enter, Esc, Alt+letter), and dropdown display.
//!
//! Replaces `tui-menu` with built-in support for accelerator keys.
//!
//! ```text
//!  File  View  Models  Help
//!  ┌──────────────┐
//!  │ New Agent     │
//!  │ Save       ^S │
//!  │ ───────────── │
//!  │ Quit       ^C │
//!  └──────────────┘
//! ```

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Clear, StatefulWidget, Widget};
use std::marker::PhantomData;

// ── Menu item definition ──

/// A menu item — either a leaf action, a separator, or a group with children.
#[derive(Debug, Clone)]
pub struct MenuDef<T: Clone> {
    /// Label with optional `&` accelerator marker (e.g., "&File").
    pub label: String,
    /// Action data for leaf items.
    pub action: Option<T>,
    /// Child items (non-empty = group/submenu).
    pub children: Vec<MenuDef<T>>,
}

impl<T: Clone> MenuDef<T> {
    /// Create a leaf item.
    pub fn item(label: impl Into<String>, action: T) -> Self {
        Self {
            label: label.into(),
            action: Some(action),
            children: Vec::new(),
        }
    }

    /// Create a group (submenu).
    pub fn group(label: impl Into<String>, children: Vec<MenuDef<T>>) -> Self {
        Self {
            label: label.into(),
            action: None,
            children,
        }
    }

    /// Create a separator.
    pub fn separator() -> Self {
        Self {
            label: "---".into(),
            action: None,
            children: Vec::new(),
        }
    }

    /// Is this a separator?
    pub fn is_separator(&self) -> bool {
        self.label == "---"
    }

    /// Is this a group (has children)?
    pub fn is_group(&self) -> bool {
        !self.children.is_empty()
    }

    /// Display label (with `&` stripped).
    pub fn display_label(&self) -> String {
        self.label.replace('&', "")
    }

    /// Accelerator character (lowercase), if any.
    pub fn accel_char(&self) -> Option<char> {
        let mut chars = self.label.chars();
        while let Some(c) = chars.next() {
            if c == '&' {
                return chars.next().map(|a| a.to_ascii_lowercase());
            }
        }
        None
    }

    /// Position of the accelerator character in the display label.
    pub fn accel_pos(&self) -> Option<usize> {
        self.label.find('&')
    }
}

// ── Menu state ──

/// Menu bar navigation state.
#[derive(Debug)]
pub struct MenuBarState<T: Clone> {
    /// Top-level menu definitions.
    pub items: Vec<MenuDef<T>>,
    /// Whether the menu bar is active (has focus).
    pub active: bool,
    /// Index of the highlighted top-level group (-1 = none).
    pub focused_group: Option<usize>,
    /// Whether the dropdown is open.
    pub dropdown_open: bool,
    /// Index of the highlighted item within the open dropdown.
    pub focused_item: Option<usize>,
    /// Pending selected actions (drained by the app each frame).
    pub events: Vec<T>,
}

impl<T: Clone> MenuBarState<T> {
    /// Create from menu definitions.
    pub fn new(items: Vec<MenuDef<T>>) -> Self {
        Self {
            items,
            active: false,
            focused_group: None,
            dropdown_open: false,
            focused_item: None,
            events: Vec::new(),
        }
    }

    /// Replace menu definitions (e.g., after config change).
    pub fn set_items(&mut self, items: Vec<MenuDef<T>>) {
        self.items = items;
        self.reset();
    }

    /// Activate the menu bar (highlight first group).
    pub fn activate(&mut self) {
        self.active = true;
        self.focused_group = Some(0);
        self.dropdown_open = false;
        self.focused_item = None;
    }

    /// Activate and open a specific group by index.
    pub fn open_group(&mut self, index: usize) {
        if index < self.items.len() {
            self.active = true;
            self.focused_group = Some(index);
            self.dropdown_open = true;
            self.focused_item = Some(0);
        }
    }

    /// Try to open a group by accelerator character. Returns true if matched.
    pub fn open_by_accel(&mut self, ch: char) -> bool {
        let ch = ch.to_ascii_lowercase();
        for (i, item) in self.items.iter().enumerate() {
            if item.accel_char() == Some(ch) {
                self.open_group(i);
                return true;
            }
        }
        false
    }

    /// Reset to inactive state.
    pub fn reset(&mut self) {
        self.active = false;
        self.focused_group = None;
        self.dropdown_open = false;
        self.focused_item = None;
    }

    /// Drain pending selection events.
    pub fn drain_events(&mut self) -> impl Iterator<Item = T> + '_ {
        self.events.drain(..)
    }

    /// Move left in the menu bar.
    pub fn left(&mut self) {
        if let Some(idx) = self.focused_group {
            let new = if idx == 0 { self.items.len() - 1 } else { idx - 1 };
            self.focused_group = Some(new);
            if self.dropdown_open {
                self.focused_item = Some(0);
            }
        }
    }

    /// Move right in the menu bar.
    pub fn right(&mut self) {
        if let Some(idx) = self.focused_group {
            let new = (idx + 1) % self.items.len();
            self.focused_group = Some(new);
            if self.dropdown_open {
                self.focused_item = Some(0);
            }
        }
    }

    /// Move up in the dropdown.
    pub fn up(&mut self) {
        if !self.dropdown_open {
            return;
        }
        if let (Some(group_idx), Some(item_idx)) = (self.focused_group, self.focused_item) {
            if let Some(group) = self.items.get(group_idx) {
                if item_idx == 0 {
                    // At top of dropdown — close it
                    self.dropdown_open = false;
                    self.focused_item = None;
                } else {
                    // Skip separators going up
                    let mut new = item_idx - 1;
                    while new > 0 && group.children.get(new).map_or(false, |c| c.is_separator()) {
                        new -= 1;
                    }
                    if !group.children.get(new).map_or(true, |c| c.is_separator()) {
                        self.focused_item = Some(new);
                    }
                }
            }
        }
    }

    /// Move down in the dropdown (or open it).
    pub fn down(&mut self) {
        if !self.dropdown_open {
            self.dropdown_open = true;
            self.focused_item = Some(0);
            // Skip initial separators
            if let Some(group_idx) = self.focused_group {
                if let Some(group) = self.items.get(group_idx) {
                    let mut idx = 0;
                    while idx < group.children.len() && group.children[idx].is_separator() {
                        idx += 1;
                    }
                    self.focused_item = Some(idx);
                }
            }
            return;
        }
        if let (Some(group_idx), Some(item_idx)) = (self.focused_group, self.focused_item) {
            if let Some(group) = self.items.get(group_idx) {
                let max = group.children.len().saturating_sub(1);
                if item_idx < max {
                    // Skip separators going down
                    let mut new = item_idx + 1;
                    while new < max && group.children.get(new).map_or(false, |c| c.is_separator()) {
                        new += 1;
                    }
                    if !group.children.get(new).map_or(true, |c| c.is_separator()) {
                        self.focused_item = Some(new);
                    }
                }
            }
        }
    }

    /// Select the currently highlighted item.
    pub fn select(&mut self) {
        if let (Some(group_idx), Some(item_idx)) = (self.focused_group, self.focused_item) {
            if let Some(group) = self.items.get(group_idx) {
                if let Some(item) = group.children.get(item_idx) {
                    if item.is_group() {
                        // TODO: nested submenus (not needed yet)
                    } else if let Some(action) = &item.action {
                        self.events.push(action.clone());
                        self.reset();
                    }
                }
            }
        } else if self.focused_group.is_some() && !self.dropdown_open {
            // Group highlighted but dropdown not open — open it
            self.down();
        }
    }
}

// ── Menu bar widget ──

/// Menu bar rendering widget.
pub struct MenuBar<T> {
    pub bar_style: Style,
    pub highlight_style: Style,
    pub accel_style: Style,
    pub dropdown_style: Style,
    pub dropdown_highlight: Style,
    pub dropdown_width: u16,
    _marker: PhantomData<T>,
}

impl<T> MenuBar<T> {
    pub fn new() -> Self {
        Self {
            bar_style: Style::default().fg(Color::Black).bg(Color::White),
            highlight_style: Style::default()
                .fg(Color::White)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
            accel_style: Style::default()
                .fg(Color::Black)
                .bg(Color::White)
                .add_modifier(Modifier::UNDERLINED),
            dropdown_style: Style::default().fg(Color::Black).bg(Color::White),
            dropdown_highlight: Style::default()
                .fg(Color::White)
                .bg(Color::DarkGray),
            dropdown_width: 22,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone> StatefulWidget for MenuBar<T> {
    type State = MenuBarState<T>;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let area = area.clamp(*buf.area());
        if area.height < 1 {
            return;
        }

        // Fill bar background
        buf.set_style(area, self.bar_style);

        let mut x = area.x;
        let y = area.y;
        let mut group_positions: Vec<(u16, u16)> = Vec::new(); // (x_start, width)

        // Leading space
        if x < area.right() {
            buf.set_string(x, y, " ", self.bar_style);
            x += 1;
        }

        for (i, item) in state.items.iter().enumerate() {
            let display = item.display_label();
            let padded = format!(" {} ", display);
            let w = padded.len() as u16;
            let is_focused = state.focused_group == Some(i);

            let style = if is_focused {
                self.highlight_style
            } else {
                self.bar_style
            };

            let group_x = x;
            group_positions.push((group_x, w));

            buf.set_string(x, y, &padded, style);

            // Underline accelerator character
            if let Some(accel_pos) = item.accel_pos() {
                let accel_x = x + 1 + accel_pos as u16; // +1 for leading space in padded
                if accel_x < area.right() {
                    let ch = display.chars().nth(accel_pos).unwrap_or('?');
                    let accel_s = if is_focused {
                        // When focused, underline on highlight background
                        Style::default()
                            .fg(Color::White)
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::UNDERLINED | Modifier::BOLD)
                    } else {
                        self.accel_style
                    };
                    buf.set_string(accel_x, y, ch.to_string(), accel_s);
                }
            }

            x += w;
        }

        // Fill remaining bar
        while x < area.right() {
            buf.set_string(x, y, " ", self.bar_style);
            x += 1;
        }

        // Render dropdown if open
        if state.dropdown_open {
            if let Some(group_idx) = state.focused_group {
                if let Some(group) = state.items.get(group_idx) {
                    if let Some(&(gx, _gw)) = group_positions.get(group_idx) {
                        self.render_dropdown(
                            gx,
                            y + 1,
                            &group.children,
                            state.focused_item,
                            buf,
                        );
                    }
                }
            }
        }
    }
}

impl<T> MenuBar<T> {
    fn render_dropdown(
        &self,
        x: u16,
        y: u16,
        items: &[MenuDef<impl Clone>],
        focused: Option<usize>,
        buf: &mut Buffer,
    ) {
        if items.is_empty() {
            return;
        }

        // Compute width from longest item
        let max_label = items
            .iter()
            .map(|item| item.display_label().len())
            .max()
            .unwrap_or(8);
        let inner_w = (max_label + 2).max(self.dropdown_width as usize - 2);
        let outer_w = inner_w + 2; // borders
        let outer_h = items.len() + 2; // borders

        // Clamp to buffer
        let buf_area = *buf.area();
        let dx = x.min(buf_area.right().saturating_sub(outer_w as u16));
        let dropdown_area = Rect::new(
            dx,
            y,
            (outer_w as u16).min(buf_area.width),
            (outer_h as u16).min(buf_area.height.saturating_sub(y - buf_area.y)),
        );

        // Clear area
        Clear.render(dropdown_area, buf);
        buf.set_style(dropdown_area, self.dropdown_style);

        // Draw border
        let right = dropdown_area.right() - 1;
        let bottom = dropdown_area.bottom() - 1;
        buf.set_string(dx, y, "┌", self.dropdown_style);
        buf.set_string(right, y, "┐", self.dropdown_style);
        buf.set_string(dx, bottom, "└", self.dropdown_style);
        buf.set_string(right, bottom, "┘", self.dropdown_style);
        for col in (dx + 1)..right {
            buf.set_string(col, y, "─", self.dropdown_style);
            buf.set_string(col, bottom, "─", self.dropdown_style);
        }
        for row in (y + 1)..bottom {
            buf.set_string(dx, row, "│", self.dropdown_style);
            buf.set_string(right, row, "│", self.dropdown_style);
        }

        // Draw items
        for (i, item) in items.iter().enumerate() {
            let item_y = y + 1 + i as u16;
            if item_y >= bottom {
                break;
            }

            if item.is_separator() {
                // Horizontal separator
                buf.set_string(dx, item_y, "├", self.dropdown_style);
                for col in (dx + 1)..right {
                    buf.set_string(col, item_y, "─", self.dropdown_style);
                }
                buf.set_string(right, item_y, "┤", self.dropdown_style);
            } else {
                let is_focused = focused == Some(i);
                let style = if is_focused {
                    self.dropdown_highlight
                } else {
                    self.dropdown_style
                };

                let display = item.display_label();
                let padded = format!(" {:<width$}", display, width = inner_w - 1);
                buf.set_string(dx + 1, item_y, &padded, style);

                // Submenu indicator
                if item.is_group() {
                    buf.set_string(right - 1, item_y, "▸", style);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accel_char_extraction() {
        let item = MenuDef::<i32>::group("&File", vec![]);
        assert_eq!(item.accel_char(), Some('f'));
    }

    #[test]
    fn accel_char_middle() {
        let item = MenuDef::<i32>::group("Mo&dels", vec![]);
        assert_eq!(item.accel_char(), Some('d'));
    }

    #[test]
    fn accel_char_none() {
        let item = MenuDef::<i32>::group("Help", vec![]);
        assert_eq!(item.accel_char(), None);
    }

    #[test]
    fn accel_pos() {
        let item = MenuDef::<i32>::group("&File", vec![]);
        assert_eq!(item.accel_pos(), Some(0));
        assert_eq!(item.display_label(), "File");
    }

    #[test]
    fn display_label_strips_ampersand() {
        let item = MenuDef::<i32>::group("&Models", vec![]);
        assert_eq!(item.display_label(), "Models");
    }

    #[test]
    fn separator() {
        let sep = MenuDef::<i32>::separator();
        assert!(sep.is_separator());
        assert!(!sep.is_group());
    }

    #[test]
    fn open_by_accel() {
        let items = vec![
            MenuDef::group("&File", vec![MenuDef::item("Quit", 1)]),
            MenuDef::group("&View", vec![MenuDef::item("Threads", 2)]),
            MenuDef::group("&Help", vec![MenuDef::item("About", 3)]),
        ];
        let mut state = MenuBarState::new(items);

        assert!(state.open_by_accel('v'));
        assert!(state.active);
        assert_eq!(state.focused_group, Some(1));
        assert!(state.dropdown_open);
    }

    #[test]
    fn open_by_accel_no_match() {
        let items = vec![
            MenuDef::group("&File", vec![MenuDef::item("Quit", 1)]),
        ];
        let mut state = MenuBarState::new(items);

        assert!(!state.open_by_accel('x'));
        assert!(!state.active);
    }

    #[test]
    fn navigation_left_right() {
        let items: Vec<MenuDef<i32>> = vec![
            MenuDef::group("A", vec![]),
            MenuDef::group("B", vec![]),
            MenuDef::group("C", vec![]),
        ];
        let mut state = MenuBarState::new(items);
        state.activate();
        assert_eq!(state.focused_group, Some(0));

        state.right();
        assert_eq!(state.focused_group, Some(1));

        state.right();
        assert_eq!(state.focused_group, Some(2));

        state.right(); // wraps
        assert_eq!(state.focused_group, Some(0));

        state.left(); // wraps back
        assert_eq!(state.focused_group, Some(2));
    }

    #[test]
    fn down_opens_dropdown() {
        let items = vec![
            MenuDef::group("A", vec![
                MenuDef::item("One", 1),
                MenuDef::item("Two", 2),
            ]),
        ];
        let mut state = MenuBarState::new(items);
        state.activate();
        assert!(!state.dropdown_open);

        state.down();
        assert!(state.dropdown_open);
        assert_eq!(state.focused_item, Some(0));

        state.down();
        assert_eq!(state.focused_item, Some(1));
    }

    #[test]
    fn up_closes_dropdown_at_top() {
        let items = vec![
            MenuDef::group("A", vec![
                MenuDef::item("One", 1),
                MenuDef::item("Two", 2),
            ]),
        ];
        let mut state = MenuBarState::new(items);
        state.open_group(0);
        assert_eq!(state.focused_item, Some(0));

        state.up(); // at top — closes dropdown
        assert!(!state.dropdown_open);
        assert_eq!(state.focused_item, None);
    }

    #[test]
    fn select_emits_event() {
        let items = vec![
            MenuDef::group("A", vec![
                MenuDef::item("One", 42),
            ]),
        ];
        let mut state = MenuBarState::new(items);
        state.open_group(0);
        state.select();

        let events: Vec<i32> = state.drain_events().collect();
        assert_eq!(events, vec![42]);
        assert!(!state.active); // reset after select
    }

    #[test]
    fn separator_skipped_in_navigation() {
        let items = vec![
            MenuDef::group("A", vec![
                MenuDef::item("One", 1),
                MenuDef::separator(),
                MenuDef::item("Two", 2),
            ]),
        ];
        let mut state = MenuBarState::new(items);
        state.open_group(0);
        assert_eq!(state.focused_item, Some(0));

        state.down(); // should skip separator, land on "Two"
        assert_eq!(state.focused_item, Some(2));

        state.up(); // should skip separator, land on "One"
        assert_eq!(state.focused_item, Some(0));
    }

    #[test]
    fn reset_clears_state() {
        let items = vec![
            MenuDef::group("A", vec![MenuDef::item("X", 1)]),
        ];
        let mut state = MenuBarState::new(items);
        state.open_group(0);
        state.reset();

        assert!(!state.active);
        assert_eq!(state.focused_group, None);
        assert!(!state.dropdown_open);
        assert_eq!(state.focused_item, None);
    }

    #[test]
    fn set_items_resets() {
        let items = vec![
            MenuDef::group("A", vec![MenuDef::item("X", 1)]),
        ];
        let mut state = MenuBarState::new(items);
        state.open_group(0);

        state.set_items(vec![
            MenuDef::group("B", vec![MenuDef::item("Y", 2)]),
        ]);
        assert!(!state.active);
        assert_eq!(state.items.len(), 1);
        assert_eq!(state.items[0].display_label(), "B");
    }
}
