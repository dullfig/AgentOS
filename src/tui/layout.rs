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

use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Wrap,
};
use ratatui::Frame;
use tui_menu::Menu;

use super::app::{ActiveTab, AgentStatus, ThreadsFocus, TuiApp};
use super::context_tree;
use super::dashboard;

/// Draw the full TUI layout.
pub fn draw(f: &mut Frame, app: &mut TuiApp) {
    // YAML tab: hide input bar, give all space to editor
    let input_height = if app.active_tab == ActiveTab::Yaml { 0 } else { 3 };
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),            // menu bar
            Constraint::Length(1),            // tab bar
            Constraint::Min(5),              // content area
            Constraint::Length(input_height), // input (textarea) — hidden on YAML tab
            Constraint::Length(1),            // status bar
        ])
        .split(f.area());

    draw_tab_bar(f, app, outer[1]);

    match app.active_tab {
        ActiveTab::Messages => draw_messages(f, app, outer[2]),
        ActiveTab::Threads => draw_threads(f, app, outer[2]),
        ActiveTab::Yaml => draw_yaml_editor(f, app, outer[2]),
        ActiveTab::Wasm => draw_wasm_placeholder(f, outer[2]),
        ActiveTab::Debug => draw_debug(f, app, outer[2]),
    }

    if input_height > 0 {
        // Cache input area for key routing
        app.input_area = outer[3];
        // Render the input editor with a border overlay
        let input_block = Block::default()
            .title(" Task ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));
        let input_inner = input_block.inner(outer[3]);
        f.render_widget(input_block, outer[3]);
        f.render_widget(&app.input_editor, input_inner);
        // Position cursor within the input editor
        if let Some((x, y)) = app.input_editor.get_visible_cursor(&input_inner) {
            f.set_cursor_position(Position::new(x, y));
        }
        draw_ghost_text(f, app, outer[3]);
        draw_command_popup(f, app, outer[3]);
    }
    draw_status(f, app, outer[4]);

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
        .dropdown_width(16)
        .dropdown_style(Style::default().fg(Color::Black).bg(Color::White));
    f.render_stateful_widget(menu_widget, menu_area, &mut app.menu_state);

    // Underline accelerator letters (F, V, M, H) by overwriting specific cells.
    // tui-menu renders items as: " " + " File " + " View " + " Model " + " Help "
    // so each accelerator letter is at: initial_space + leading_space + cumulative widths.
    let accel_style = Style::default()
        .fg(Color::Black)
        .bg(Color::White)
        .add_modifier(Modifier::UNDERLINED);
    let names = ["File", "View", "Model", "Help"];
    let accels = ['F', 'V', 'M', 'H'];
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

fn draw_tab_bar(f: &mut Frame, app: &TuiApp, area: Rect) {
    let mut tabs: Vec<(&str, ActiveTab, &str)> = vec![
        ("Messages", ActiveTab::Messages, "1"),
        ("Threads", ActiveTab::Threads, "2"),
        ("YAML", ActiveTab::Yaml, "3"),
        ("WASM", ActiveTab::Wasm, "4"),
    ];
    if app.debug_mode {
        tabs.push(("Debug", ActiveTab::Debug, "5"));
    }

    let spans: Vec<Span> = tabs
        .iter()
        .flat_map(|(name, tab, num)| {
            let is_active = *tab == app.active_tab;
            let style = if is_active {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            vec![
                Span::raw(" "),
                Span::styled(format!("[^{num} {name}]"), style),
            ]
        })
        .collect();

    let line = Line::from(spans);
    let para = Paragraph::new(line);
    f.render_widget(para, area);
}

/// Render command popup above the input bar when typing `/`.
fn draw_command_popup(f: &mut Frame, app: &TuiApp, input_area: Rect) {
    let input = app.input_text();
    if !input.starts_with('/') || input.contains(' ') {
        return;
    }

    let matches = super::commands::matching_commands(&input);
    if matches.is_empty() {
        return;
    }

    // Popup dimensions
    let popup_width = 44u16.min(input_area.width);
    let popup_height = (matches.len() as u16 + 2).min(10); // +2 for borders
    let popup_x = input_area.x;
    let popup_y = input_area.y.saturating_sub(popup_height);

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear background
    f.render_widget(
        Paragraph::new("").style(Style::default().bg(Color::Black)),
        popup_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .style(Style::default().bg(Color::Black));

    let selected = app.command_popup_index;
    let items: Vec<Line> = matches
        .iter()
        .enumerate()
        .map(|(i, cmd)| {
            let arg_hint = if cmd.has_arg { " <arg>" } else { "" };
            let is_selected = i == selected;
            let style = if is_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let desc_style = if is_selected {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            Line::from(vec![
                Span::styled(format!("{}{}", cmd.name, arg_hint), style),
                Span::styled(format!("  {}", cmd.description), desc_style),
            ])
        })
        .collect();

    let para = Paragraph::new(items).block(block);
    f.render_widget(para, popup_area);
}

/// Render ghost-text autocomplete overlay after the cursor in the input bar.
fn draw_ghost_text(f: &mut Frame, app: &TuiApp, area: Rect) {
    let input = app.input_text();
    if let Some(suffix) = super::commands::ghost_suffix(&input) {
        let inner = Block::default().borders(Borders::ALL).inner(area);
        let cursor_pos = app.input_editor.get_visible_cursor(&inner);
        let (x, y) = match cursor_pos {
            Some((cx, cy)) => (cx, cy),
            None => return,
        };
        let max_width = area.right().saturating_sub(x);
        if max_width > 0 {
            let ghost = Paragraph::new(Span::styled(
                &suffix[..suffix.len().min(max_width as usize)],
                Style::default().fg(Color::DarkGray),
            ));
            let ghost_rect = Rect::new(x, y, max_width.min(suffix.len() as u16), 1);
            f.render_widget(ghost, ghost_rect);
        }
    }
}

fn draw_threads(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    // Two-pane vertical split: threads list, context tree
    let thread_rows = (app.threads.len() as u16 + 2).min(7).max(3);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(thread_rows), // threads
            Constraint::Min(5),             // context tree (gets all remaining space)
        ])
        .split(area);

    // ── Pane 1: Thread list ──
    let thread_border_color = if app.threads_focus == ThreadsFocus::ThreadList {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let thread_block = Block::default()
        .title(" Threads ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(thread_border_color));

    let thread_items: Vec<ListItem> = app
        .threads
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let chain_short = t.chain.split('.').next_back().unwrap_or(&t.chain);
            let is_selected = i == app.selected_thread;
            let style = if is_selected {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            let prefix = if is_selected { "> " } else { "  " };
            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(chain_short, style),
                Span::styled(format!(" [{}]", t.profile), style),
                Span::styled(
                    format!("  {}", &t.uuid[..8.min(t.uuid.len())]),
                    style,
                ),
            ]))
        })
        .collect();

    let thread_list = List::new(thread_items).block(thread_block);
    f.render_widget(thread_list, chunks[0]);

    // ── Pane 2: Context tree (tui-tree-widget) ──
    let ctx_border_color = if app.threads_focus == ThreadsFocus::ContextTree {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    let selected_uuid = app
        .threads
        .get(app.selected_thread)
        .map(|t| &t.uuid[..8.min(t.uuid.len())])
        .unwrap_or("?");
    let ctx_title = format!(" Context (thread {selected_uuid}) ");

    let ctx_block = Block::default()
        .title(ctx_title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ctx_border_color));

    if let Some(ctx) = &app.context {
        let items = context_tree::build_context_tree(ctx);
        if let Ok(tree) = tui_tree_widget::Tree::new(&items) {
            let tree = tree
                .block(ctx_block)
                .highlight_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
                .highlight_symbol(">> ");
            f.render_stateful_widget(tree, chunks[1], &mut app.context_tree_state);
        } else {
            let para = Paragraph::new("Error building context tree").block(ctx_block);
            f.render_widget(para, chunks[1]);
        }
    } else {
        let para = Paragraph::new(Span::styled(
            "No context for selected thread.",
            Style::default().fg(Color::DarkGray),
        ))
        .block(ctx_block);
        f.render_widget(para, chunks[1]);
    }

}

fn draw_debug(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let block = Block::default()
        .title(" Debug — Activity Trace ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let lines: Vec<Line> = if app.activity_log.is_empty() {
        vec![Line::from(Span::styled(
            "No activity yet. Submit a task to see the live trace.",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.activity_log
            .iter()
            .map(|entry| {
                let time_str = format_timestamp(entry.timestamp);
                let status_span = match entry.status {
                    super::app::ActivityStatus::InProgress => {
                        Span::styled("...", Style::default().fg(Color::Yellow))
                    }
                    super::app::ActivityStatus::Done => {
                        Span::styled(" OK", Style::default().fg(Color::Green))
                    }
                    super::app::ActivityStatus::Error => {
                        Span::styled("ERR", Style::default().fg(Color::Red))
                    }
                };
                let detail_text = if entry.detail.is_empty() {
                    String::new()
                } else {
                    format!("  {}", entry.detail)
                };
                Line::from(vec![
                    Span::styled(
                        format!("{time_str}  "),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("[{}]", entry.label),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(detail_text),
                    Span::raw("  "),
                    status_span,
                ])
            })
            .collect()
    };

    // Scroll clamping (same pattern as draw_messages)
    let inner_height = area.height.saturating_sub(2) as u32;
    let total_lines = lines.len() as u32;
    let max_scroll = total_lines.saturating_sub(inner_height);
    let max_scroll_u16 = max_scroll.min(u16::MAX as u32) as u16;
    let scroll = if app.activity_auto_scroll {
        max_scroll_u16
    } else {
        app.activity_scroll.min(max_scroll_u16)
    };
    app.activity_scroll = scroll;
    app.activity_viewport_height = inner_height.min(u16::MAX as u32) as u16;

    let para = Paragraph::new(lines)
        .block(block)
        .scroll((scroll, 0));
    f.render_widget(para, area);

    // Scrollbar
    if total_lines > inner_height {
        let mut scrollbar_state =
            ScrollbarState::new(max_scroll_u16 as usize).position(scroll as usize);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area,
            &mut scrollbar_state,
        );
    }
}

/// Format a unix timestamp as HH:MM:SS local time.
fn format_timestamp(secs: u64) -> String {
    // Simple: seconds since midnight (avoids chrono dependency)
    let total_secs = secs % 86400;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn draw_messages(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let block = Block::default()
        .title(" Messages ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let mut lines: Vec<Line> = Vec::new();

    for entry in &app.chat_log {
        match entry.role.as_str() {
            "user" => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        "[You] ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(&entry.text),
                ]));
            }
            "agent" => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![Span::styled(
                    "[Agent]",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )]));
                for text_line in entry.text.lines() {
                    lines.push(Line::from(text_line.to_string()));
                }
            }
            "system" => {
                lines.push(Line::from(""));
                for text_line in entry.text.lines() {
                    lines.push(Line::from(vec![Span::styled(
                        text_line.to_string(),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
            }
            _ => {}
        }
    }

    // Show thinking indicator when waiting
    if app.agent_status == AgentStatus::Thinking {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "thinking...",
            Style::default().fg(Color::Yellow),
        )]));
    } else if let AgentStatus::ToolCall(ref name) = app.agent_status {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            format!("using {name}..."),
            Style::default().fg(Color::Cyan),
        )]));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "No messages yet. Type a task and press Enter.",
            Style::default().fg(Color::DarkGray),
        )));
    }

    // Clamp scroll so we never scroll past content.
    // Account for line wrapping: each line may occupy multiple visual rows.
    // Use u32 to avoid overflow for very long responses (u16 max = 65535 lines).
    let inner_height = area.height.saturating_sub(2) as u32;
    let inner_width = area.width.saturating_sub(2).max(1) as usize;
    let total_lines: u32 = lines
        .iter()
        .map(|line| {
            let width: usize = line.spans.iter().map(|s| s.content.len()).sum();
            if width == 0 {
                1u32
            } else {
                width.div_ceil(inner_width) as u32
            }
        })
        .sum();
    let max_scroll = total_lines.saturating_sub(inner_height);
    // Clamp to u16 range for ratatui's scroll API (65535 lines is still enormous)
    let max_scroll_u16 = max_scroll.min(u16::MAX as u32) as u16;
    let scroll = if app.message_auto_scroll {
        max_scroll_u16
    } else {
        app.message_scroll.min(max_scroll_u16)
    };
    // Write clamped value back so up/down keys work immediately
    app.message_scroll = scroll;
    // Tell the app how tall the viewport is (for PageUp/PageDown)
    app.viewport_height = inner_height.min(u16::MAX as u32) as u16;

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(para, area);

    // Scrollbar
    if total_lines > inner_height {
        let mut scrollbar_state =
            ScrollbarState::new(max_scroll_u16 as usize).position(scroll as usize);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area,
            &mut scrollbar_state,
        );
    }
}

fn draw_yaml_editor(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    if let Some(ref editor) = app.yaml_editor {
        // Cache area for input routing (editor.input() needs the render Rect)
        app.yaml_area = area;
        f.render_widget(editor, area);
        // Position cursor within the editor
        let cursor_pos = editor.get_visible_cursor(&area);
        if let Some((x, y)) = cursor_pos {
            f.set_cursor_position(Position::new(x, y));
        }

        // Show diagnostic summary in the bottom-left
        if !app.diag_summary.is_empty() {
            let summary_area = Rect::new(
                area.x + 1,
                area.y + area.height.saturating_sub(1),
                app.diag_summary.len() as u16 + 2,
                1,
            );
            let style = if app.diag_summary.contains("error") {
                Style::default().fg(Color::Red).bg(Color::Black)
            } else {
                Style::default().fg(Color::Yellow).bg(Color::Black)
            };
            f.render_widget(Paragraph::new(Span::styled(&app.diag_summary, style)), summary_area);
        }

        // Show validation status (from Ctrl+S) in the bottom-right of the area
        if let Some(ref status) = app.yaml_status {
            let msg = if status.len() > (area.width as usize).saturating_sub(4) {
                &status[..area.width as usize - 4]
            } else {
                status.as_str()
            };
            let status_area = Rect::new(
                area.x + 1,
                area.y + area.height.saturating_sub(1),
                area.width.saturating_sub(2),
                1,
            );
            f.render_widget(
                Paragraph::new(Span::styled(msg, Style::default().fg(Color::Red))),
                status_area,
            );
        }

        // Completion popup
        if app.completion_visible && !app.completion_items.is_empty() {
            draw_yaml_completion_popup(f, app, cursor_pos, area);
        }

        // Hover overlay
        if let Some(ref hover) = app.hover_info {
            draw_yaml_hover_overlay(f, hover, cursor_pos, area);
        }
    } else {
        let block = Block::default()
            .title(" YAML ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let para = Paragraph::new(Span::styled(
            "No organism YAML loaded. Use --organism or load from File menu.",
            Style::default().fg(Color::DarkGray),
        ))
        .block(block);
        f.render_widget(para, area);
    }
}

/// Render completion popup near the cursor in the YAML editor.
fn draw_yaml_completion_popup(
    f: &mut Frame,
    app: &TuiApp,
    cursor_pos: Option<(u16, u16)>,
    area: Rect,
) {
    let (cx, cy) = cursor_pos.unwrap_or((area.x + 2, area.y + 2));
    let items = &app.completion_items;
    let popup_width = items
        .iter()
        .map(|i| i.label.len() + 2)
        .max()
        .unwrap_or(20)
        .min(40) as u16 + 2; // +2 for borders
    let popup_height = (items.len() as u16 + 2).min(10);

    // Position: below cursor if space, else above
    let popup_y = if cy + 1 + popup_height <= area.bottom() {
        cy + 1
    } else {
        cy.saturating_sub(popup_height)
    };
    let popup_x = cx.min(area.right().saturating_sub(popup_width));

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear background
    f.render_widget(
        Paragraph::new("").style(Style::default().bg(Color::Black)),
        popup_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .style(Style::default().bg(Color::Black));

    let lines: Vec<Line> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let is_selected = i == app.completion_index;
            let style = if is_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(Span::styled(&item.label, style))
        })
        .collect();

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, popup_area);
}

/// Render hover info overlay near the cursor in the YAML editor.
fn draw_yaml_hover_overlay(
    f: &mut Frame,
    hover: &crate::lsp::HoverInfo,
    cursor_pos: Option<(u16, u16)>,
    area: Rect,
) {
    let (cx, cy) = cursor_pos.unwrap_or((area.x + 2, area.y + 2));
    let text = &hover.content;
    let lines: Vec<&str> = text.lines().collect();
    let max_width = lines.iter().map(|l| l.len()).max().unwrap_or(20).min(60) as u16 + 4;
    let popup_height = (lines.len() as u16 + 2).min(12);

    // Position above cursor
    let popup_y = cy.saturating_sub(popup_height);
    let popup_x = cx.min(area.right().saturating_sub(max_width));

    let popup_area = Rect::new(popup_x, popup_y, max_width, popup_height);

    f.render_widget(
        Paragraph::new("").style(Style::default().bg(Color::Black)),
        popup_area,
    );

    let block = Block::default()
        .title(" Hover ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    let styled_lines: Vec<Line> = lines
        .iter()
        .map(|l| {
            if l.starts_with("**") {
                Line::from(Span::styled(
                    l.trim_matches('*'),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(Span::styled(*l, Style::default().fg(Color::White)))
            }
        })
        .collect();

    let para = Paragraph::new(styled_lines).block(block);
    f.render_widget(para, popup_area);
}

fn draw_wasm_placeholder(f: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" WASM ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let para = Paragraph::new(Span::styled(
        "Coming soon — WASM tool inventory",
        Style::default().fg(Color::DarkGray),
    ))
    .block(block);
    f.render_widget(para, area);
}

fn draw_status(f: &mut Frame, app: &TuiApp, area: Rect) {
    let status_text = match &app.agent_status {
        AgentStatus::Idle => Span::styled("idle", Style::default().fg(Color::Green)),
        AgentStatus::Thinking => Span::styled("thinking...", Style::default().fg(Color::Yellow)),
        AgentStatus::ToolCall(name) => Span::styled(
            format!("tool: {name}"),
            Style::default().fg(Color::Cyan),
        ),
        AgentStatus::Error(msg) => Span::styled(
            format!("error: {msg}"),
            Style::default().fg(Color::Red),
        ),
    };

    let tab_name = match app.active_tab {
        ActiveTab::Messages => "Messages",
        ActiveTab::Threads => "Threads",
        ActiveTab::Yaml => "YAML",
        ActiveTab::Wasm => "WASM",
        ActiveTab::Debug => "Debug",
    };

    let tab_hint = if app.debug_mode {
        "^1/2/3/4/5:Tabs"
    } else {
        "^1/2/3/4:Tabs"
    };

    let mut spans = vec![
        Span::styled(" [", Style::default().fg(Color::DarkGray)),
        status_text,
        Span::styled("]", Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(
            format!(
                "[Tokens: {}/{}]",
                dashboard::format_tokens(app.total_input_tokens),
                dashboard::format_tokens(app.total_output_tokens),
            ),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw("  "),
        Span::styled(
            format!("[Threads: {}]", app.threads.len()),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw("  "),
        Span::styled(
            format!("[{tab_name}]"),
            Style::default().fg(Color::Green),
        ),
    ];

    // YAML tab: show diagnostics + extra shortcuts
    if app.active_tab == ActiveTab::Yaml && !app.diag_summary.is_empty() {
        let diag_color = if app.diag_summary.contains("error") { Color::Red } else { Color::Yellow };
        spans.push(Span::raw("  "));
        spans.push(Span::styled(format!("[{}]", app.diag_summary), Style::default().fg(diag_color)));
    }

    let shortcuts = if app.active_tab == ActiveTab::Yaml {
        format!("^S:Validate  ^Space:Complete  ^H:Hover  {tab_hint}  ^C:Quit")
    } else {
        format!("Enter:Send  {tab_hint}  Tab:Focus  \u{2191}\u{2193}:Scroll  Esc:Clear  ^C:Quit")
    };
    spans.push(Span::raw("  "));
    spans.push(Span::styled(shortcuts, Style::default().fg(Color::DarkGray)));

    let status = Line::from(spans);

    let para = Paragraph::new(status);
    f.render_widget(para, area);
}
