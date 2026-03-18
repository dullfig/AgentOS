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
use ratatui::text::Span;
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
    // Non-debug: File, View, Models, Help — accels F, V, M, H
    // Debug:     File, View, Models, Debug, Help — accels F, V, M, D, H
    let accel_style = Style::default()
        .fg(Color::Black)
        .bg(Color::White)
        .add_modifier(Modifier::UNDERLINED);
    let (names, accels): (&[&str], &[char]) = if app.debug_mode {
        (&["File", "View", "Models", "Debug", "Help"],
         &['F', 'V', 'M', 'D', 'H'])
    } else {
        (&["File", "View", "Models", "Help"],
         &['F', 'V', 'M', 'H'])
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
    use super::folder_tabs::FolderTabBar;

    let labels: Vec<String> = app.open_tabs.iter().map(|t| t.label().to_string()).collect();
    let active_idx = app.open_tabs.iter().position(|t| *t == app.active_tab).unwrap_or(0);

    // Auto-scroll to keep active tab visible
    app.tab_scroll = FolderTabBar::scroll_to_visible(
        &labels,
        active_idx,
        app.tab_scroll,
        area.width as usize,
        true,
    );

    let bar = FolderTabBar::new(&labels, active_idx).scroll(app.tab_scroll);
    let regions = bar.render(f, area);

    // Map (x_start, x_end, index) back to (x_start, x_end, TabId)
    app.layout_areas.tab_regions = regions
        .into_iter()
        .filter_map(|(x_start, x_end, idx)| {
            app.open_tabs.get(idx).map(|tab| (x_start, x_end, tab.clone()))
        })
        .collect();
}
