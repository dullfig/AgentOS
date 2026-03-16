//! Tabbed layout with TextArea input bar.
//!
//! ```text
//! ┌─[ Messages ]──[ Threads ]──[ YAML ]──[ WASM ]─┐
//! │                                                 │
//! │  (full-screen content for the active tab)       │
//! │                                                 │
//! ├─────────────────────────────────────────────────┤
//! │ > input bar (tui-textarea)                      │
//! ├─────────────────────────────────────────────────┤
//! │ [idle] [Tokens: 12K/3K] ^1/2/3/4:Tabs          │
//! └─────────────────────────────────────────────────┘
//! ```

mod activity;
mod graph;
mod messages;
mod shared;
mod threads;
mod tool_editor;
pub(crate) mod wrap;
mod yaml;

use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use tui_menu::Menu;

use super::app::{TabId, TuiApp};

/// Subtle dark background for fixed-width blocks (tables, code, diagrams).
/// Works on 256-color and truecolor terminals; falls back gracefully on 16-color.
pub(super) const BLOCK_BG: Color = Color::Rgb(30, 30, 36);


/// Draw the full TUI layout.
pub fn draw(f: &mut Frame, app: &mut TuiApp) {
    // Messages tab: input is embedded inside draw_messages (single outline).
    // YAML tab: input is hidden (editor takes full area).
    // Other tabs: external input bar below content.
    let input_height = match app.active_tab {
        TabId::Agent(_) | TabId::Yaml | TabId::Tool(_) => 0,
        _ => 3,
    };
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                // menu bar
            Constraint::Length(2),                // tab bar (folder tab + top border)
            Constraint::Min(5),                  // content area
            Constraint::Length(input_height),     // input (textarea) — hidden on Messages/YAML
            Constraint::Length(2),                // status bar (2 lines)
        ])
        .split(f.area());

    // Cache layout areas for mouse hit-testing
    app.layout_areas.menu_bar = outer[0];
    app.layout_areas.tab_bar = outer[1];
    app.layout_areas.content = outer[2];
    app.layout_areas.input_bar = outer[3];
    app.layout_areas.status_bar = outer[4];

    draw_tab_bar(f, app, outer[1]);

    match app.active_tab {
        TabId::Agent(_) => messages::draw_messages(f, app, outer[2]),
        TabId::Tool(_) => tool_editor::draw_tool_editor(f, app, outer[2]),
        TabId::Threads => threads::draw_threads(f, app, outer[2]),
        TabId::Yaml => yaml::draw_yaml_editor(f, app, outer[2]),
        TabId::Graph => graph::draw_graph(f, app, outer[2]),
        TabId::Activity => activity::draw_activity(f, app, outer[2]),
    }

    if input_height > 0 {
        // Cache input area for key routing
        app.input_area = outer[3];

        if app.in_wizard() {
            shared::draw_wizard_input(f, app, outer[3]);
        } else {
            // Render the input line with a border overlay
            let input_block = Block::default()
                .title(" Task ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan));
            let input_inner = input_block.inner(outer[3]);
            f.render_widget(input_block, outer[3]);
            let content = app.input_line.content().to_string();
            f.render_widget(Paragraph::new(content.clone()), input_inner);
            // Position cursor
            let (cx, cy) = wrap::plain_cursor_xy(&content, app.input_line.cursor());
            f.set_cursor_position(Position::new(
                input_inner.x + cx,
                input_inner.y + cy,
            ));
            shared::draw_ghost_text(f, app, outer[3]);
            shared::draw_command_popup(f, app, outer[3]);
        }
    }
    shared::draw_status(f, app, outer[4]);

    // Tool approval popup — centered overlay when awaiting user consent
    if app.pending_approval.is_some() {
        shared::draw_approval_popup(f, app, outer[2]);
    }

    // File picker overlay
    if app.file_picker.is_some() {
        shared::draw_file_picker(f, app, outer[2]);
    }

    // Fill the menu bar row with white background before rendering menu items.
    f.render_widget(
        Paragraph::new("").style(Style::default().bg(Color::White)),
        outer[0],
    );

    // Menu bar rendered last — dropdowns overlay tab bar + content below.
    let menu_area = Rect {
        x: outer[0].x,
        y: outer[0].y,
        width: outer[0].width,
        height: outer[0].height + outer[1].height + outer[2].height,
    };
    let menu_widget = Menu::new()
        .default_style(Style::default().fg(Color::Black).bg(Color::White))
        .highlight(
            Style::default()
                .fg(Color::White)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .dropdown_width(20)
        .dropdown_style(Style::default().fg(Color::Black).bg(Color::White));
    f.render_stateful_widget(menu_widget, menu_area, &mut app.menu_state);

    // Underline accelerator letters by overwriting specific cells.
    // Non-debug: File, View, Help — accels F, V, H
    // Debug:     File, View, Debug, Help — accels F, V, D, H
    let accel_style = Style::default()
        .fg(Color::Black)
        .bg(Color::White)
        .add_modifier(Modifier::UNDERLINED);
    let (names, accels): (&[&str], &[char]) = if app.debug_mode {
        (&["File", "View", "Debug", "Help"],
         &['F', 'V', 'D', 'H'])
    } else {
        (&["File", "View", "Help"],
         &['F', 'V', 'H'])
    };
    let mut x = outer[0].x + 1; // skip initial " "
    for (i, name) in names.iter().enumerate() {
        x += 1; // leading space of " name "
        let cell = Rect::new(x, outer[0].y, 1, 1);
        f.render_widget(
            Paragraph::new(Span::styled(accels[i].to_string(), accel_style)),
            cell,
        );
        x += name.len() as u16 + 1; // rest of name + trailing space
    }
}

fn draw_tab_bar(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    // Folder-tab design: active tab protrudes above the top border.
    //
    //  ┌─^1 Bob─┐
    //  │        └─ ^2 Coder  ^3 Activity ─────────┐
    //
    // Row 1 (area.y):     tab top line
    // Row 2 (area.y + 1): left wall of tab + connector + inactive tabs + top border

    let border_style = Style::default().fg(Color::Cyan);
    let active_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let inactive_style = Style::default().fg(Color::DarkGray);
    let w = area.width as usize;

    if w < 4 || area.height < 2 {
        return;
    }

    // Find active tab
    let active_idx = app.open_tabs.iter().position(|t| *t == app.active_tab).unwrap_or(0);
    // active_label no longer needed — tabs are rendered in the loop below.

    // ── Row 1: all tabs on same line, active in cyan, inactive in grey ──
    // "┌─^1 Bob─┐ ┌─^2 Coder─┐ ┌─^3 Activity─┐"
    // Only the active tab connects to the frame below.

    let mut tab_regions = Vec::new();
    let mut row1 = Vec::new();
    let mut row1_col: usize = 0;
    // Track where the active tab starts and ends (column positions) for row 2
    let mut active_tab_start: usize = 0;
    let mut active_tab_end: usize = 0;

    for (i, tab) in app.open_tabs.iter().enumerate() {
        let is_active = i == active_idx;
        let label = format!("^{} {}", i + 1, tab.label());
        let tab_w = label.len() + 4; // "┌─" + label + "─┐"

        if row1_col + tab_w >= w {
            break; // no room
        }

        let style = if is_active { border_style } else { inactive_style };
        let label_style = if is_active { active_style } else { inactive_style };

        if is_active {
            active_tab_start = row1_col;
            active_tab_end = row1_col + tab_w;
        }

        let x_start = area.x + row1_col as u16 + 2; // after "┌─"
        let x_end = x_start + label.len() as u16;
        tab_regions.push((x_start, x_end, tab.clone()));

        row1.push(Span::styled("\u{250C}\u{2500}", style));   // "┌─"
        row1.push(Span::styled(label, label_style));
        row1.push(Span::styled("\u{2500}\u{2510}", style));   // "─┐"
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

    // ── Row 2: top border of content frame, with gap under active tab ──
    //
    // Active tab at col 0:
    //   "│        └──────────────────────────┐"
    //   │ = tab left wall continues as frame left border
    //   └ = under active tab's ┐
    //
    // Active tab at col N:
    //   "┌──────────┘           └────────────┐"
    //   ┌ = frame top-left corner
    //   ┘ = under active tab's ┌ (border turns up into tab)
    //   └ = under active tab's ┐ (border resumes from tab)

    let mut row2 = Vec::new();
    #[allow(unused_assignments)]
    let mut col: usize = 0;

    if active_tab_start == 0 {
        // Active tab at left edge — its left wall IS the frame left border
        row2.push(Span::styled("\u{2502}", border_style)); // "│"
        col = 1;
        // Spaces under the active tab interior
        let gap_end = active_tab_end.saturating_sub(1); // └ aligns under ┐
        if gap_end > col {
            row2.push(Span::raw(" ".repeat(gap_end - col)));
            col = gap_end;
        }
        // └ under active tab's ┐, then ─ continues as top border
        row2.push(Span::styled("\u{2514}", border_style));
        col += 1;
    } else {
        // Frame top-left corner
        row2.push(Span::styled("\u{250C}", border_style)); // "┌"
        col = 1;
        // ─ runs from col 1 to just before the active tab
        if active_tab_start > col {
            let fill = active_tab_start - col;
            row2.push(Span::styled("\u{2500}".repeat(fill), border_style));
            col = active_tab_start;
        }
        // ┘ under active tab's ┌ (border turns up into tab)
        row2.push(Span::styled("\u{2518}", border_style));
        col += 1;
        // Spaces under the active tab interior
        let gap_end = active_tab_end.saturating_sub(1); // └ aligns under ┐
        if gap_end > col {
            row2.push(Span::raw(" ".repeat(gap_end - col)));
            col = gap_end;
        }
        // └ under active tab's ┐ (border resumes)
        row2.push(Span::styled("\u{2514}", border_style));
        col += 1;
    }

    // ─ fills to the end, ┐ closes the frame top-right
    if col + 1 < w {
        let fill = w - col - 1;
        row2.push(Span::styled("\u{2500}".repeat(fill), border_style));
    }
    row2.push(Span::styled("\u{2510}", border_style)); // ┐

    app.layout_areas.tab_regions = tab_regions;

    // Render both rows
    let row1_area = Rect { x: area.x, y: area.y, width: area.width, height: 1 };
    let row2_area = Rect { x: area.x, y: area.y + 1, width: area.width, height: 1 };
    f.render_widget(Paragraph::new(Line::from(row1)), row1_area);
    f.render_widget(Paragraph::new(Line::from(row2)), row2_area);
}
