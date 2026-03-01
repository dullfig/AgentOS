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
};
use ratatui::Frame;
use tui_menu::Menu;

use super::app::{ActiveTab, AgentStatus, ThreadsFocus, TuiApp};
use super::context_tree;
use super::dashboard;

/// Subtle dark background for fixed-width blocks (tables, code, diagrams).
/// Works on 256-color and truecolor terminals; falls back gracefully on 16-color.
const BLOCK_BG: Color = Color::Rgb(30, 30, 36);

/// Draw the full TUI layout.
pub fn draw(f: &mut Frame, app: &mut TuiApp) {
    // Messages tab: input is embedded inside draw_messages (single outline).
    // YAML tab: input is hidden (editor takes full area).
    // Other tabs: external input bar below content.
    let input_height = match app.active_tab {
        ActiveTab::Messages | ActiveTab::Yaml => 0,
        _ => 3,
    };
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                // menu bar
            Constraint::Length(1),                // tab bar
            Constraint::Min(5),                  // content area
            Constraint::Length(input_height),     // input (textarea) — hidden on Messages/YAML
            Constraint::Length(1),                // status bar
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
        ActiveTab::Messages => draw_messages(f, app, outer[2]),
        ActiveTab::Threads => draw_threads(f, app, outer[2]),
        ActiveTab::Yaml => draw_yaml_editor(f, app, outer[2]),
        ActiveTab::Graph => draw_graph(f, app, outer[2]),
        ActiveTab::Activity => draw_activity(f, app, outer[2]),
    }

    if input_height > 0 {
        // Cache input area for key routing
        app.input_area = outer[3];

        if app.in_wizard() {
            draw_wizard_input(f, app, outer[3]);
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
            let (cx, cy) = plain_cursor_xy(&content, app.input_line.cursor());
            f.set_cursor_position(Position::new(
                input_inner.x + cx,
                input_inner.y + cy,
            ));
            draw_ghost_text(f, app, outer[3]);
            draw_command_popup(f, app, outer[3]);
        }
    }
    draw_status(f, app, outer[4]);

    // Tool approval popup — centered overlay when awaiting user consent
    if app.pending_approval.is_some() {
        draw_approval_popup(f, app, outer[2]);
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
        .dropdown_width(16)
        .dropdown_style(Style::default().fg(Color::Black).bg(Color::White));
    f.render_stateful_widget(menu_widget, menu_area, &mut app.menu_state);

    // Underline accelerator letters by overwriting specific cells.
    // tui-menu renders items as: " " + " File " + " View " + " Modify " + ...
    // The accelerator letter offset within each name: F(0), V(0), O(1 in mOdify), A(0), M(0), H(0).
    let accel_style = Style::default()
        .fg(Color::Black)
        .bg(Color::White)
        .add_modifier(Modifier::UNDERLINED);
    let names = ["File", "View", "Modify", "Agents", "Model", "Help"];
    let accels = ['F', 'V', 'O', 'A', 'M', 'H'];
    let accel_offsets: [u16; 6] = [0, 0, 1, 0, 0, 0]; // char offset within each name
    let mut x = outer[0].x + 1; // skip initial " "
    for (i, name) in names.iter().enumerate() {
        x += 1; // leading space of " name "
        let cell = Rect::new(x + accel_offsets[i], outer[0].y, 1, 1);
        f.render_widget(
            Paragraph::new(Span::styled(accels[i].to_string(), accel_style)),
            cell,
        );
        x += name.len() as u16 + 1; // rest of name + trailing space
    }
}

fn draw_tab_bar(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let tabs: Vec<(&str, ActiveTab, &str)> = vec![
        ("Messages", ActiveTab::Messages, "1"),
        ("Threads", ActiveTab::Threads, "2"),
        ("Graph", ActiveTab::Graph, "3"),
        ("YAML", ActiveTab::Yaml, "4"),
        ("Activity", ActiveTab::Activity, "5"),
    ];

    // Build tab_regions for mouse hit-testing
    let mut tab_regions = Vec::new();
    let mut x = area.x;

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
            let label = format!("[^{num} {name}]");
            let label_len = label.len() as u16;
            // " " prefix (1 col) + label
            let x_start = x + 1; // after the space
            let x_end = x_start + label_len;
            tab_regions.push((x_start, x_end, *tab));
            x = x_end;
            vec![
                Span::raw(" "),
                Span::styled(label, style),
            ]
        })
        .collect();

    app.layout_areas.tab_regions = tab_regions;

    let line = Line::from(spans);
    let para = Paragraph::new(line);
    f.render_widget(para, area);
}

/// Render command popup above the input bar when typing `/`.
fn draw_command_popup(f: &mut Frame, app: &TuiApp, input_area: Rect) {
    use crate::lsp::LanguageService;

    let input = app.input_text();
    if !input.starts_with('/') {
        return;
    }

    let items = app
        .cmd_service
        .completions(&input, lsp_types::Position::new(0, input.len() as u32));
    if items.is_empty() {
        return;
    }

    // Popup dimensions
    let popup_width = 50u16.min(input_area.width);
    let popup_height = (items.len() as u16 + 2).min(10); // +2 for borders
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
    let lines: Vec<Line> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
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
            let detail = item.detail.as_deref().unwrap_or("");
            Line::from(vec![
                Span::styled(&item.label, style),
                Span::styled(format!("  {detail}"), desc_style),
            ])
        })
        .collect();

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, popup_area);
}

/// Render tool approval as a centered blue popup over the content area.
fn draw_approval_popup(f: &mut Frame, app: &TuiApp, content_area: Rect) {
    let request = match &app.pending_approval {
        Some(r) => r,
        None => return,
    };

    // Popup size: 5 lines tall, up to 50 cols wide (or content width - 4)
    let popup_w = 50u16.min(content_area.width.saturating_sub(4));
    let popup_h = 5u16;
    if content_area.height < popup_h + 2 || popup_w < 20 {
        return; // terminal too small
    }
    let x = content_area.x + (content_area.width.saturating_sub(popup_w)) / 2;
    let y = content_area.y + (content_area.height.saturating_sub(popup_h)) / 2;
    let popup = Rect::new(x, y, popup_w, popup_h);

    // Blue background, no borders
    let bg = Style::default().bg(Color::Blue).fg(Color::White);

    // Build lines
    let inner_w = popup_w as usize;
    let tool_line = format!(" {}", request.tool_name);
    let args_line = format!(" {}", request.args_summary);
    // Truncate args if too wide
    let args_display = if args_line.len() > inner_w {
        format!("{}...", &args_line[..inner_w.saturating_sub(3)])
    } else {
        args_line
    };
    let keys_line = Line::from(vec![
        Span::styled(" [1] ", Style::default().bg(Color::Blue).fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled("approve  ", bg),
        Span::styled("[2] ", Style::default().bg(Color::Blue).fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled("deny", bg),
    ]);

    let text = vec![
        Line::styled(tool_line, bg.add_modifier(Modifier::BOLD)),
        Line::styled(args_display, bg),
        Line::styled("", bg), // spacer
        keys_line,
        Line::styled("", bg), // bottom padding
    ];

    // Clear the popup area with blue background
    f.render_widget(
        Paragraph::new("").style(bg),
        popup,
    );
    f.render_widget(Paragraph::new(text).style(bg), popup);
}

/// Render the wizard input bar (single-step: API key).
fn draw_wizard_input(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    use super::app::InputMode;

    let title = match &app.input_mode {
        InputMode::ProviderWizard { provider } => {
            format!(" /provider {provider} ")
        }
        InputMode::Normal => return,
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Draw the prompt prefix
    let prompt = "> API key: ";
    let prompt_width = prompt.len() as u16;
    f.render_widget(
        Paragraph::new(Span::styled(prompt, Style::default().fg(Color::Yellow))),
        Rect::new(inner.x, inner.y, prompt_width.min(inner.width), 1),
    );
    // Input gets remaining width
    let edit_x = inner.x + prompt_width;
    let edit_width = inner.width.saturating_sub(prompt_width);
    if edit_width > 0 {
        let edit_area = Rect::new(edit_x, inner.y, edit_width, 1);
        let content = app.input_line.content().to_string();
        f.render_widget(Paragraph::new(content.clone()), edit_area);
        let (cx, _) = plain_cursor_xy(&content, app.input_line.cursor());
        f.set_cursor_position(Position::new(edit_area.x + cx, edit_area.y));
    }
}

/// Render ghost-text autocomplete overlay after the cursor in the input bar.
fn draw_ghost_text(f: &mut Frame, app: &TuiApp, area: Rect) {
    let input = app.input_text();
    if let Some(suffix) = crate::lsp::command_line::ghost_suffix(&input) {
        let inner = Block::default().borders(Borders::ALL).inner(area);
        let (cx, cy) = plain_cursor_xy(&input, app.input_line.cursor());
        let (x, y) = (inner.x + cx, inner.y + cy);
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
    // Three-pane vertical split: thread list, conversation, context tree
    let thread_rows = (app.threads.len() as u16 + 2).clamp(3, 7);
    let ctx_rows = 7u16; // collapsed context tree
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(thread_rows), // threads (compact)
            Constraint::Min(10),            // conversation (gets bulk)
            Constraint::Length(ctx_rows),    // context tree (collapsed)
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
            // Active indicator: ● when agent is working on this thread
            let indicator = if is_selected {
                match &app.agent_status {
                    AgentStatus::Thinking => " \u{25cf}",
                    AgentStatus::ToolCall(_) => " \u{25cf}",
                    _ => "",
                }
            } else {
                ""
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
                Span::styled(indicator, Style::default().fg(Color::Yellow)),
            ]))
        })
        .collect();

    let thread_list = List::new(thread_items).block(thread_block);
    f.render_widget(thread_list, chunks[0]);

    // ── Pane 2: Conversation ──
    draw_conversation(f, app, chunks[1]);

    // ── Pane 3: Context tree (tui-tree-widget) ──
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
            f.render_stateful_widget(tree, chunks[2], &mut app.context_tree_state);
        } else {
            let para = Paragraph::new("Error building context tree").block(ctx_block);
            f.render_widget(para, chunks[2]);
        }
    } else {
        let para = Paragraph::new(Span::styled(
            "No context for selected thread.",
            Style::default().fg(Color::DarkGray),
        ))
        .block(ctx_block);
        f.render_widget(para, chunks[2]);
    }
}

/// Render the conversation pane for the selected thread.
fn draw_conversation(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let border_color = if app.threads_focus == ThreadsFocus::Conversation {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let block = Block::default()
        .title(" Conversation ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    // Find conversation entries for the selected thread
    let selected_thread_id = app
        .threads
        .get(app.selected_thread)
        .map(|t| t.uuid.clone());

    let entries = selected_thread_id
        .as_ref()
        .and_then(|id| app.thread_conversations.get(id));

    let lines: Vec<Line> = if let Some(entries) = entries {
        let mut lines = Vec::new();
        for entry in entries {
            match entry.role.as_str() {
                "user" => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "[You] ",
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(&entry.summary),
                    ]));
                }
                "assistant" if entry.is_tool_use => {
                    let check = if entry.is_error { "\u{2717}" } else { "" };
                    lines.push(Line::from(vec![
                        Span::styled("[Tool] ", Style::default().fg(Color::Yellow)),
                        Span::raw(&entry.summary),
                        Span::styled(
                            format!(" {check}"),
                            Style::default().fg(if entry.is_error {
                                Color::Red
                            } else {
                                Color::White
                            }),
                        ),
                    ]));
                }
                "assistant" => {
                    // Truncate to ~80 chars for compact view
                    let text = if entry.summary.len() > 80 {
                        format!("{}...", &entry.summary[..77])
                    } else {
                        entry.summary.clone()
                    };
                    lines.push(Line::from(vec![
                        Span::styled(
                            "[Agent] ",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(text),
                    ]));
                }
                "tool_result" => {
                    let style = if entry.is_error {
                        Style::default().fg(Color::Red)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    let prefix = if entry.is_error {
                        "  \u{2514}\u{2500} error: "
                    } else {
                        "  \u{2514}\u{2500} "
                    };
                    let text = if entry.summary.len() > 60 {
                        format!("{}...", &entry.summary[..57])
                    } else {
                        entry.summary.clone()
                    };
                    lines.push(Line::from(Span::styled(
                        format!("{prefix}{text}"),
                        style,
                    )));
                }
                _ => {}
            }
        }

        // Thinking indicator when agent is active
        match &app.agent_status {
            AgentStatus::Thinking => {
                lines.push(Line::from(Span::styled(
                    "\u{2847} thinking...",
                    Style::default().fg(Color::Yellow),
                )));
            }
            AgentStatus::ToolCall(name) => {
                lines.push(Line::from(Span::styled(
                    format!("\u{2847} using {name}..."),
                    Style::default().fg(Color::Cyan),
                )));
            }
            _ => {}
        }

        lines
    } else {
        vec![Line::from(Span::styled(
            "No conversation yet.",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    // Scroll clamping
    let inner_height = area.height.saturating_sub(2) as u32;
    let total_lines = lines.len() as u32;
    let max_scroll = total_lines.saturating_sub(inner_height);
    let max_scroll_u16 = max_scroll.min(u16::MAX as u32) as u16;
    let scroll = if app.conversation_auto_scroll {
        max_scroll_u16
    } else {
        app.conversation_scroll.min(max_scroll_u16)
    };
    app.conversation_scroll = scroll;
    app.conversation_viewport_height = inner_height.min(u16::MAX as u32) as u16;

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

fn draw_activity(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let block = Block::default()
        .title(" Activity Trace ")
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
    // ── Single-outline layout: messages + embedded input ──
    //
    // ┌─ Messages ────────────────────────┐
    // │  chat content                     │
    // │                                   │
    // │                                   │  ← 1-line gap
    // │░░> input text_                   ░│  ← shaded, grows up
    // └───────────────────────────────────┘
    let block = Block::default()
        .title(" Messages ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let wrap_width = inner.width.max(1) as usize;

    // Calculate how many visual lines the input content wraps to.
    // Uses the same wrap_line logic for consistent rendering.
    // Cap at 1/3 of inner height so messages always have room.
    let input_content = app.input_line.content().to_string();
    let input_wrap_width = inner.width.saturating_sub(2).max(1) as usize; // minus "> " prompt
    let wrapped_input: Vec<Line<'static>> = if input_content.is_empty() {
        vec![Line::from("")]
    } else {
        // Split on newlines (from multiline paste), then wrap each line
        input_content
            .split('\n')
            .flat_map(|l| wrap_line(Line::from(l.to_string()), input_wrap_width))
            .collect()
    };
    let input_line_count = wrapped_input.len().max(1);
    let max_input_h = (inner.height / 3).max(1);
    let input_h = (input_line_count as u16).min(max_input_h);
    let gap_h = 1_u16; // separator line between messages and input
    let input_total = input_h + gap_h;

    // Split inner area: messages on top, gap + input on bottom
    let msg_height = inner.height.saturating_sub(input_total);
    let msg_area = Rect::new(inner.x, inner.y, inner.width, msg_height);
    let gap_area = Rect::new(inner.x, inner.y + msg_height, inner.width, gap_h);
    let input_area = Rect::new(
        inner.x,
        inner.y + msg_height + gap_h,
        inner.width,
        input_h,
    );

    // ── Render messages ──
    let mut lines: Vec<Line> = Vec::new();
    let mut nowrap: Vec<bool> = Vec::new();
    let mut entry_map: Vec<Option<usize>> = Vec::new(); // visual line → chat_log index
    let mut code_block_copies: Vec<(usize, String)> = Vec::new(); // (visual_line, raw fenced text)
    let mut last_entry_start: u32 = 0;

    for (entry_idx, entry) in app.chat_log.iter().enumerate() {
        last_entry_start = lines.len() as u32;
        match entry.role.as_str() {
            "user" => {
                lines.push(Line::from(""));
                nowrap.push(false);
                entry_map.push(None); // separator
                let user_line = Line::from(vec![
                    Span::styled(
                        "[You] ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(entry.text.clone()),
                ]);
                let wrapped = wrap_line(user_line, wrap_width);
                let n = wrapped.len();
                nowrap.extend(std::iter::repeat(false).take(n));
                entry_map.extend(std::iter::repeat(Some(entry_idx)).take(n));
                lines.extend(wrapped);
            }
            "agent" => {
                lines.push(Line::from(""));
                nowrap.push(false);
                entry_map.push(None); // separator
                lines.push(Line::from(vec![Span::styled(
                    "[Agent]",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )]));
                nowrap.push(false);
                entry_map.push(Some(entry_idx)); // header belongs to this entry
                for tagged in super::markdown::render_markdown(&entry.text) {
                    // Track code block header lines for click-to-copy
                    if let Some(raw_fenced) = tagged.copy_block {
                        code_block_copies.push((lines.len(), raw_fenced));
                    }
                    if tagged.nowrap {
                        let mut line = tagged.line;
                        let content_w: usize = line.spans.iter()
                            .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
                            .sum();
                        // Pad well past the viewport so background extends through any h-scroll.
                        // The Paragraph's scroll clips the visible portion.
                        let fill_width = wrap_width + 200;
                        let pad = fill_width.saturating_sub(content_w);
                        if pad > 0 {
                            line.spans.push(Span::styled(
                                " ".repeat(pad),
                                Style::default().bg(BLOCK_BG),
                            ));
                        }
                        for span in &mut line.spans {
                            span.style = span.style.bg(BLOCK_BG);
                        }
                        lines.push(line);
                        nowrap.push(true);
                        entry_map.push(Some(entry_idx));
                    } else {
                        let wrapped = wrap_line(tagged.line, wrap_width);
                        let n = wrapped.len();
                        nowrap.extend(std::iter::repeat(false).take(n));
                        entry_map.extend(std::iter::repeat(Some(entry_idx)).take(n));
                        lines.extend(wrapped);
                    }
                }
            }
            "system" => {
                lines.push(Line::from(""));
                nowrap.push(false);
                entry_map.push(None); // separator
                for text_line in entry.text.lines() {
                    let sys_line = Line::from(vec![Span::styled(
                        text_line.to_string(),
                        Style::default().fg(Color::DarkGray),
                    )]);
                    let wrapped = wrap_line(sys_line, wrap_width);
                    let n = wrapped.len();
                    nowrap.extend(std::iter::repeat(false).take(n));
                    entry_map.extend(std::iter::repeat(Some(entry_idx)).take(n));
                    lines.extend(wrapped);
                }
            }
            _ => {}
        }
    }

    if app.agent_status == AgentStatus::Thinking {
        lines.push(Line::from(""));
        nowrap.push(false);
        lines.push(Line::from(vec![Span::styled(
            "thinking...",
            Style::default().fg(Color::Yellow),
        )]));
        nowrap.push(false);
    } else if let AgentStatus::ToolCall(ref name) = app.agent_status {
        lines.push(Line::from(""));
        nowrap.push(false);
        lines.push(Line::from(vec![Span::styled(
            format!("using {name}..."),
            Style::default().fg(Color::Cyan),
        )]));
        nowrap.push(false);
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "No messages yet. Type a task and press Enter.",
            Style::default().fg(Color::DarkGray),
        )));
        nowrap.push(false);
    }

    // Anchored horizontal scroll
    let h = app.message_h_scroll as usize;
    if h > 0 {
        let pad = " ".repeat(h);
        for (i, line) in lines.iter_mut().enumerate() {
            if !nowrap.get(i).copied().unwrap_or(false) {
                line.spans.insert(0, Span::raw(pad.clone()));
            }
        }
    }

    let inner_height = msg_area.height as u32;
    let total_lines = lines.len() as u32;
    let max_scroll = total_lines.saturating_sub(inner_height);
    let max_scroll_u16 = max_scroll.min(u16::MAX as u32) as u16;
    let scroll = if app.scroll_to_last_entry {
        app.scroll_to_last_entry = false;
        let target = last_entry_start.min(max_scroll);
        target.min(u16::MAX as u32) as u16
    } else if app.message_auto_scroll {
        max_scroll_u16
    } else {
        app.message_scroll.min(max_scroll_u16)
    };
    app.message_scroll = scroll;
    app.viewport_height = inner_height.min(u16::MAX as u32) as u16;

    // Cache rendered text and entry map for mouse selection
    app.rendered_messages_text = lines.iter()
        .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
        .collect();
    app.rendered_messages_entry_map = entry_map;
    app.code_block_copies = code_block_copies;
    app.rendered_messages_scroll = scroll;
    app.layout_areas.messages_content = msg_area;

    // Apply selection highlight to selected lines (bg color on each span)
    if app.text_selection.active {
        let sel = &app.text_selection;
        for abs_line in sel.start_line..=sel.end_line {
            if abs_line < lines.len() {
                for span in &mut lines[abs_line].spans {
                    span.style = span.style.bg(super::mouse::SELECTION_BG);
                }
            }
        }
    }

    let para = Paragraph::new(lines)
        .scroll((scroll, app.message_h_scroll));
    f.render_widget(para, msg_area);

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

    // ── Gap line (thin horizontal rule) ──
    let rule = "─".repeat(gap_area.width as usize);
    f.render_widget(
        Paragraph::new(Span::styled(rule, Style::default().fg(Color::Rgb(50, 50, 56)))),
        gap_area,
    );

    // ── Shaded input area ──
    let input_bg = Color::Rgb(35, 35, 42);

    // Fill the input zone with the shaded background
    for row in 0..input_area.height {
        let fill_area = Rect::new(input_area.x, input_area.y + row, input_area.width, 1);
        f.render_widget(
            Paragraph::new(Span::styled(
                " ".repeat(input_area.width as usize),
                Style::default().bg(input_bg),
            )),
            fill_area,
        );
    }

    // Render "> " prompt on each visible line (only first logical line gets "> ")
    for row in 0..input_area.height {
        let prompt_area = Rect::new(input_area.x, input_area.y + row, 2, 1);
        let prompt_text = if row == 0 && app.input_scroll == 0 { "> " } else { "  " };
        f.render_widget(
            Paragraph::new(Span::styled(
                prompt_text,
                Style::default().fg(Color::Cyan).bg(input_bg).add_modifier(Modifier::BOLD),
            )),
            prompt_area,
        );
    }

    // Render wrapped input text (after "> ")
    let editor_area = Rect::new(
        input_area.x + 2,
        input_area.y,
        input_area.width.saturating_sub(2),
        input_area.height,
    );
    app.input_area = editor_area;

    // Compute cursor position in wrapped lines for auto-scroll
    let (cx, cy) = multiline_cursor_xy(&input_content, app.input_line.cursor(), input_wrap_width);
    let cursor_row = cy as usize;
    let input_h = input_area.height as usize;
    let total_wrapped = wrapped_input.len();
    let current_cursor = app.input_line.cursor();

    // Auto-scroll only when cursor moved (typing/pasting) — not on every frame,
    // so mouse-wheel scroll doesn't get fought by auto-scroll.
    if current_cursor != app.input_cursor_last {
        app.input_cursor_last = current_cursor;
        if cursor_row < app.input_scroll {
            app.input_scroll = cursor_row;
        } else if cursor_row >= app.input_scroll + input_h {
            app.input_scroll = cursor_row + 1 - input_h;
        }
    }
    // Clamp scroll to valid range (always, in case content/window changed)
    let max_input_scroll = total_wrapped.saturating_sub(input_h);
    app.input_scroll = app.input_scroll.min(max_input_scroll);

    let v_scroll = app.input_scroll;

    // Render the visible slice of wrapped lines
    let display_lines: Vec<Line> = wrapped_input
        .iter()
        .skip(v_scroll)
        .take(input_h)
        .map(|l| {
            let mut styled = l.clone();
            for span in &mut styled.spans {
                span.style = span.style.bg(input_bg);
            }
            styled
        })
        .collect();
    f.render_widget(Paragraph::new(display_lines), editor_area);

    // Position cursor (adjusted for scroll offset)
    let cursor_x = editor_area.x + cx;
    let cursor_y = editor_area.y + (cursor_row - v_scroll) as u16;
    if cursor_y >= editor_area.y && cursor_y < editor_area.y + editor_area.height {
        f.set_cursor_position(Position::new(cursor_x, cursor_y));
    }

    // Ghost text (inline completion hint)
    {
        let input = app.input_text();
        if let Some(suffix) = crate::lsp::command_line::ghost_suffix(&input) {
            let max_w = area.right().saturating_sub(cursor_x + 1); // stay inside border
            if max_w > 0 && cursor_y >= editor_area.y && cursor_y < editor_area.y + editor_area.height {
                let ghost_rect = Rect::new(cursor_x, cursor_y, max_w.min(suffix.len() as u16), 1);
                f.render_widget(
                    Paragraph::new(Span::styled(
                        &suffix[..suffix.len().min(max_w as usize)],
                        Style::default().fg(Color::DarkGray).bg(input_bg),
                    )),
                    ghost_rect,
                );
            }
        }
    }

    // Command popup (anchored above the input area)
    draw_command_popup(f, app, input_area);
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

fn draw_graph(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let block = Block::default()
        .title(" Graph — Organism Topology ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as u32;

    // Lazy render: regenerate only when width changes or cache is empty
    if app.graph_rendered_width != inner_width || app.graph_rendered_lines.is_empty() {
        if app.graph_d2_source.is_empty() {
            app.graph_rendered_lines = vec![Line::from(Span::styled(
                "No organism loaded.",
                Style::default().fg(Color::DarkGray),
            ))];
        } else {
            app.graph_rendered_lines = super::diagram::render_d2(&app.graph_d2_source, inner_width);
        }
        app.graph_rendered_width = inner_width;
    }

    // Scroll clamping
    let total_lines = app.graph_rendered_lines.len() as u32;
    let max_scroll = total_lines.saturating_sub(inner_height);
    let max_scroll_u16 = max_scroll.min(u16::MAX as u32) as u16;
    app.graph_scroll = app.graph_scroll.min(max_scroll_u16);
    app.graph_viewport_height = inner_height.min(u16::MAX as u32) as u16;

    let para = Paragraph::new(app.graph_rendered_lines.clone())
        .block(block)
        .scroll((app.graph_scroll, app.graph_h_scroll));
    f.render_widget(para, area);

    // Scrollbar
    if total_lines > inner_height {
        let mut scrollbar_state =
            ScrollbarState::new(max_scroll_u16 as usize).position(app.graph_scroll as usize);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area,
            &mut scrollbar_state,
        );
    }
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
        ActiveTab::Graph => "Graph",
        ActiveTab::Yaml => "YAML",
        ActiveTab::Activity => "Activity",
    };

    let tab_hint = "^1/2/3/4/5:Tabs";

    let agent_label = match &app.selected_agent {
        Some(name) => format!("[Agent: {name}]"),
        None => String::new(),
    };

    let mut spans = vec![
        Span::styled(" [", Style::default().fg(Color::DarkGray)),
        status_text,
        Span::styled("]", Style::default().fg(Color::DarkGray)),
    ];

    if !agent_label.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            &agent_label,
            Style::default().fg(Color::Magenta),
        ));
    }

    spans.extend([
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
    ]);

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

// ─── Selective word-wrapping helpers ────────────────────────────────────────

/// Word-wrap a single `Line` at `max_width` display columns, preserving span styles.
/// Returns the line unchanged if it already fits.
/// Compute cursor (col, row) for plain unwrapped text.
/// Used for the external input bar (single-line, no wrapping).
fn plain_cursor_xy(content: &str, cursor_char: usize) -> (u16, u16) {
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
fn wrapped_cursor_xy(wrapped: &[Line], cursor_char: usize) -> (u16, u16) {
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
fn multiline_cursor_xy(content: &str, cursor_char: usize, wrap_width: usize) -> (u16, u16) {
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

fn wrap_line(line: Line<'static>, max_width: usize) -> Vec<Line<'static>> {
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
