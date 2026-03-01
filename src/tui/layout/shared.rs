//! Shared layout helpers: status bar, popups, wizard input, ghost text.

use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::super::app::{TabId, AgentStatus, TuiApp};
use super::super::dashboard;
use super::wrap::plain_cursor_xy;

/// Render command popup above the input bar when typing `/`.
pub(super) fn draw_command_popup(f: &mut Frame, app: &TuiApp, input_area: Rect) {
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
pub(super) fn draw_approval_popup(f: &mut Frame, app: &TuiApp, content_area: Rect) {
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
pub(super) fn draw_wizard_input(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    use super::super::app::InputMode;

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
pub(super) fn draw_ghost_text(f: &mut Frame, app: &TuiApp, area: Rect) {
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

pub(super) fn draw_status(f: &mut Frame, app: &TuiApp, area: Rect) {
    let agent_status = if let Some(tab) = app.active_agent_tab() {
        &tab.agent_status
    } else {
        &app.agent_status
    };
    let status_text = match agent_status {
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

    let tab_name = app.active_tab.label();

    let tab_hint = "^1..9:Tabs";

    let agent_label = if let TabId::Agent(ref name) = app.active_tab {
        format!("[Agent: {name}]")
    } else {
        String::new()
    };

    let mut spans = Vec::new();
    if app.debug_mode {
        spans.push(Span::styled(
            " [DEBUG]",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ));
    }
    spans.extend([
        Span::styled(" [", Style::default().fg(Color::DarkGray)),
        status_text,
        Span::styled("]", Style::default().fg(Color::DarkGray)),
    ]);

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
    if app.active_tab == TabId::Yaml && !app.diag_summary.is_empty() {
        let diag_color = if app.diag_summary.contains("error") { Color::Red } else { Color::Yellow };
        spans.push(Span::raw("  "));
        spans.push(Span::styled(format!("[{}]", app.diag_summary), Style::default().fg(diag_color)));
    }

    let shortcuts = if app.active_tab == TabId::Yaml {
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
