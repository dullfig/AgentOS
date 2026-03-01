//! Mouse event handling for the TUI.
//!
//! Provides hit-testing against cached layout regions, scroll wheel dispatch,
//! text selection with clipboard copy, tab bar clicking, and input cursor
//! positioning. Mouse events flow from the crossterm input thread through
//! the runner into `handle_mouse()`.

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;

use super::app::{TabId, MessagesFocus, ThreadsFocus, TuiApp};

/// Cached layout regions for mouse hit-testing. Updated each render frame.
#[derive(Default, Clone, Debug)]
pub struct LayoutAreas {
    pub menu_bar: Rect,
    pub tab_bar: Rect,
    pub content: Rect,
    pub input_bar: Rect,
    pub status_bar: Rect,
    /// Tab label spans within tab_bar: (x_start, x_end, TabId).
    pub tab_regions: Vec<(u16, u16, TabId)>,
    /// Messages content area (inside the border, above embedded input).
    pub messages_content: Rect,
}

/// Line-based text selection in the Messages pane.
#[derive(Default, Clone, Debug)]
pub struct TextSelection {
    pub active: bool,
    /// Absolute line index in rendered content (start of selection).
    pub start_line: usize,
    /// Absolute line index in rendered content (end of selection, inclusive).
    pub end_line: usize,
    /// Where drag started (anchor — start/end swap around this).
    pub anchor_line: usize,
}

/// Handle a mouse event, dispatching to the appropriate handler based on
/// which layout region was clicked.
pub fn handle_mouse(app: &mut TuiApp, event: MouseEvent) {
    let col = event.column;
    let row = event.row;

    match event.kind {
        MouseEventKind::ScrollUp => {
            handle_scroll(app, col, row, true);
        }
        MouseEventKind::ScrollDown => {
            handle_scroll(app, col, row, false);
        }
        MouseEventKind::Down(MouseButton::Left) => {
            handle_left_down(app, col, row);
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            handle_left_drag(app, col, row);
        }
        MouseEventKind::Up(MouseButton::Left) => {
            handle_left_up(app);
        }
        _ => {}
    }
}

/// Scroll wheel: route to the correct pane based on mouse position.
fn handle_scroll(app: &mut TuiApp, col: u16, row: u16, up: bool) {
    let areas = &app.layout_areas;

    // Input area scroll (embedded within agent tab content)
    if app.active_tab.is_agent() && rect_contains(app.input_area, col, row) {
        for _ in 0..3 {
            if up {
                app.input_scroll = app.input_scroll.saturating_sub(1);
            } else {
                app.input_scroll = app.input_scroll.saturating_add(1);
                // Clamping happens at render time (we don't know total wrapped lines here)
            }
        }
        return;
    }

    if rect_contains(areas.content, col, row) {
        match app.active_tab {
            TabId::Agent(_) => {
                for _ in 0..3 {
                    if up {
                        app.scroll_messages_up();
                    } else {
                        app.scroll_messages_down();
                    }
                }
            }
            TabId::Threads => {
                // Route to focused sub-pane
                match app.threads_focus {
                    ThreadsFocus::ThreadList => {
                        if up {
                            app.move_up();
                        } else {
                            app.move_down();
                        }
                    }
                    ThreadsFocus::Conversation => {
                        for _ in 0..3 {
                            if up {
                                app.scroll_conversation_up();
                            } else {
                                app.scroll_conversation_down();
                            }
                        }
                    }
                    ThreadsFocus::ContextTree => {
                        // Tree widget doesn't have scroll methods; use key_up/key_down
                        for _ in 0..3 {
                            if up {
                                app.context_tree_state.key_up();
                            } else {
                                app.context_tree_state.key_down();
                            }
                        }
                    }
                }
            }
            TabId::Graph => {
                for _ in 0..3 {
                    if up {
                        app.scroll_graph_up();
                    } else {
                        app.scroll_graph_down();
                    }
                }
            }
            TabId::Activity => {
                for _ in 0..3 {
                    if up {
                        app.scroll_activity_up();
                    } else {
                        app.scroll_activity_down();
                    }
                }
            }
            TabId::Yaml => {
                // YAML editor handles its own scroll
            }
        }
    }
}

/// Left mouse button pressed.
fn handle_left_down(app: &mut TuiApp, col: u16, row: u16) {
    let areas = app.layout_areas.clone();

    // Tab bar click → switch tab
    if rect_contains(areas.tab_bar, col, row) {
        for (x_start, x_end, tab) in &areas.tab_regions {
            if col >= *x_start && col < *x_end {
                app.active_tab = tab.clone();
                // Clear any active selection when switching tabs
                app.text_selection.active = false;
                return;
            }
        }
        return;
    }

    // Input bar click → position cursor (outer input bar OR embedded input area)
    let input_rect = if app.active_tab.is_agent() {
        app.input_area // embedded inside content
    } else {
        areas.input_bar
    };
    if rect_contains(input_rect, col, row) {
        let char_offset = (col - input_rect.x) as usize;
        app.input_line.set_cursor(char_offset);
        app.messages_focus = MessagesFocus::Input;
        // Clear text selection on input click
        app.text_selection.active = false;
        return;
    }

    // Content area click on agent tab → check code block copy, then text selection
    if rect_contains(areas.content, col, row) && app.active_tab.is_agent() {
        app.messages_focus = MessagesFocus::Messages;
        let msg_content = areas.messages_content;
        if rect_contains(msg_content, col, row) {
            let visual_row = (row - msg_content.y) as usize;
            let abs_line = visual_row + app.rendered_messages_scroll as usize;

            // Check if clicking a code block header line → insert into input area
            if let Some((_, raw_fenced)) = app.code_block_copies.iter().find(|(line, _)| *line == abs_line) {
                let text = raw_fenced.clone();
                // Also copy to system clipboard
                if let Ok(mut clip) = arboard::Clipboard::new() {
                    let _ = clip.set_text(text.clone());
                }
                // Insert directly into input area — no paste step needed
                app.input_line.insert_str(&text);
                return;
            }

            app.text_selection = TextSelection {
                active: true,
                start_line: abs_line,
                end_line: abs_line,
                anchor_line: abs_line,
            };
        }
        return;
    }

    // Click anywhere else clears selection
    app.text_selection.active = false;
}

/// Left mouse drag — extend text selection.
fn handle_left_drag(app: &mut TuiApp, col: u16, row: u16) {
    if !app.text_selection.active {
        return;
    }
    if !app.active_tab.is_agent() {
        return;
    }

    let msg_content = app.layout_areas.messages_content;
    let _ = col; // selection is line-based, column doesn't matter

    // Clamp row to content area
    let clamped_row = row.clamp(msg_content.y, msg_content.y + msg_content.height.saturating_sub(1));
    let visual_row = (clamped_row - msg_content.y) as usize;
    let abs_line = visual_row + app.rendered_messages_scroll as usize;

    let anchor = app.text_selection.anchor_line;
    if abs_line <= anchor {
        app.text_selection.start_line = abs_line;
        app.text_selection.end_line = anchor;
    } else {
        app.text_selection.start_line = anchor;
        app.text_selection.end_line = abs_line;
    }
}

/// Left mouse button released — copy selection to clipboard if multi-line.
fn handle_left_up(app: &mut TuiApp) {
    if !app.text_selection.active {
        return;
    }

    let sel = &app.text_selection;
    if sel.start_line == sel.end_line {
        // Single-line click, not a drag — clear selection
        app.text_selection.active = false;
        return;
    }

    // Copy raw markdown of selected entries to clipboard.
    // Map visual lines → chat_log entry indices, then copy raw source text.
    let start = sel.start_line;
    let end = sel.end_line;
    let map = &app.rendered_messages_entry_map;
    if start < map.len() {
        let clamped_end = end.min(map.len().saturating_sub(1));
        // Collect unique entry indices in order
        let mut entry_indices: Vec<usize> = Vec::new();
        for i in start..=clamped_end {
            if let Some(idx) = map[i] {
                if entry_indices.last() != Some(&idx) {
                    entry_indices.push(idx);
                }
            }
        }
        // Build text from raw chat_log entries
        let parts: Vec<&str> = entry_indices.iter()
            .filter_map(|&idx| app.chat_log.get(idx))
            .map(|e| e.text.as_str())
            .collect();
        if !parts.is_empty() {
            let selected_text = parts.join("\n\n");
            if let Ok(mut clip) = arboard::Clipboard::new() {
                let _ = clip.set_text(selected_text);
            }
        }
    }
    // Keep selection visible until next keystroke/click
}

/// Check if a point (col, row) is inside a Rect.
fn rect_contains(r: Rect, col: u16, row: u16) -> bool {
    col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
}

/// Selection highlight color (muted blue).
pub const SELECTION_BG: ratatui::style::Color = ratatui::style::Color::Rgb(40, 60, 100);

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{MouseEvent, MouseEventKind, MouseButton};

    fn make_app_with_areas() -> TuiApp {
        let mut app = TuiApp::new();
        app.layout_areas = LayoutAreas {
            menu_bar: Rect::new(0, 0, 80, 1),
            tab_bar: Rect::new(0, 1, 80, 1),
            content: Rect::new(0, 2, 80, 20),
            input_bar: Rect::new(0, 22, 80, 3),
            status_bar: Rect::new(0, 25, 80, 1),
            tab_regions: vec![
                (1, 16, TabId::Agent("planner".into())),
                (17, 30, TabId::Threads),
                (31, 42, TabId::Graph),
                (43, 52, TabId::Yaml),
                (53, 66, TabId::Activity),
            ],
            messages_content: Rect::new(1, 3, 78, 16),
        };
        app
    }

    #[test]
    fn scroll_up_in_content_scrolls_messages() {
        let mut app = make_app_with_areas();
        app.message_scroll = 10;
        app.message_auto_scroll = false;

        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 40,
            row: 10,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.message_scroll, 7); // 3 lines
    }

    #[test]
    fn scroll_down_in_content_scrolls_messages() {
        let mut app = make_app_with_areas();
        app.message_scroll = 10;
        app.message_auto_scroll = false;

        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 40,
            row: 10,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.message_scroll, 13);
    }

    #[test]
    fn tab_click_switches_tab() {
        let mut app = make_app_with_areas();
        assert_eq!(app.active_tab, TabId::Agent("planner".into()));

        // Click on Threads tab region (col 20, row 1)
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 20,
            row: 1,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.active_tab, TabId::Threads);
    }

    #[test]
    fn tab_click_on_graph() {
        let mut app = make_app_with_areas();

        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 35,
            row: 1,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.active_tab, TabId::Graph);
    }

    #[test]
    fn selection_single_click_clears() {
        let mut app = make_app_with_areas();
        app.rendered_messages_text = vec!["line 1".into(), "line 2".into(), "line 3".into()];
        app.rendered_messages_scroll = 0;

        // Mouse down in messages content
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 3,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        assert!(app.text_selection.active);

        // Mouse up at same position → single click → clears
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 5,
            row: 3,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        assert!(!app.text_selection.active);
    }

    #[test]
    fn selection_drag_selects_lines() {
        let mut app = make_app_with_areas();
        app.rendered_messages_text = (0..20).map(|i| format!("line {i}")).collect();
        app.rendered_messages_scroll = 0;

        // Mouse down at row 3 (visual row 0 in content)
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 3,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        // Drag to row 6 (visual row 3)
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 5,
            row: 6,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert!(app.text_selection.active);
        assert_eq!(app.text_selection.start_line, 0);
        assert_eq!(app.text_selection.end_line, 3);
    }

    #[test]
    fn scroll_on_activity_tab() {
        let mut app = make_app_with_areas();
        app.active_tab = TabId::Activity;
        app.activity_scroll = 10;

        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 40,
            row: 10,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.activity_scroll, 7);
    }

    #[test]
    fn scroll_on_graph_tab() {
        let mut app = make_app_with_areas();
        app.active_tab = TabId::Graph;
        app.graph_scroll = 10;

        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 40,
            row: 10,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(app.graph_scroll, 13);
    }

    #[test]
    fn rect_contains_works() {
        let r = Rect::new(10, 20, 30, 5);
        assert!(rect_contains(r, 10, 20));
        assert!(rect_contains(r, 39, 24));
        assert!(!rect_contains(r, 40, 20)); // x + width = out
        assert!(!rect_contains(r, 10, 25)); // y + height = out
        assert!(!rect_contains(r, 9, 20));  // before x
    }

    #[test]
    fn input_cursor_positioning() {
        let mut app = make_app_with_areas();
        app.active_tab = TabId::Agent("planner".into());
        // input_area is set by the renderer; simulate it
        app.input_area = Rect::new(3, 22, 75, 1);
        app.input_line.set_content("hello world");

        // Click at column 8 in input area
        handle_mouse(&mut app, MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 8,
            row: 22,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        // Cursor should be at offset 5 (col 8 - input.x 3 = 5)
        assert_eq!(app.input_line.cursor(), 5);
    }
}
