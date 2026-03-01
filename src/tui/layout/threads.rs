//! Threads tab: thread list + conversation + context tree.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::Frame;

use super::super::app::{AgentStatus, ThreadsFocus, TuiApp};
use super::super::context_tree;

pub(super) fn draw_threads(f: &mut Frame, app: &mut TuiApp, area: Rect) {
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
