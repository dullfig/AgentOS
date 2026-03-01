//! Messages tab: chat log with embedded input.

use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::Frame;

use super::super::app::{AgentStatus, TuiApp};
use super::shared::draw_command_popup;
use super::wrap::{multiline_cursor_xy, wrap_line};
use super::BLOCK_BG;

pub(super) fn draw_messages(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    // ── Single-outline layout: messages + embedded input ──
    //
    // ┌─ [agent name] ─────────────────────┐
    // │  chat content                       │
    // │                                     │
    // │                                     │  ← 1-line gap
    // │░░> input text_                     ░│  ← shaded, grows up
    // └─────────────────────────────────────┘
    let title = if let super::super::app::TabId::Agent(ref name) = app.active_tab {
        format!(" {} ", name)
    } else {
        " Messages ".to_string()
    };
    let block = Block::default()
        .title(title)
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

    // Read chat log from active agent tab (fallback to global)
    let chat_log = if let Some(tab) = app.active_agent_tab() {
        tab.chat_log.clone()
    } else {
        app.chat_log.clone()
    };
    let agent_status = if let Some(tab) = app.active_agent_tab() {
        tab.agent_status.clone()
    } else {
        app.agent_status.clone()
    };
    for (entry_idx, entry) in chat_log.iter().enumerate() {
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
                for tagged in super::super::markdown::render_markdown(&entry.text) {
                    // Track code block header lines for click-to-copy
                    if let Some(raw_fenced) = tagged.copy_block {
                        code_block_copies.push((lines.len(), raw_fenced));
                    }
                    if tagged.nowrap {
                        let mut line = tagged.line;
                        let content_w: usize = line.spans.iter()
                            .map(|s| crate::tui::box_drawing::display_width(s.content.as_ref()))
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

    if agent_status == AgentStatus::Thinking {
        lines.push(Line::from(""));
        nowrap.push(false);
        lines.push(Line::from(vec![Span::styled(
            "thinking...",
            Style::default().fg(Color::Yellow),
        )]));
        nowrap.push(false);
    } else if let AgentStatus::ToolCall(ref name) = agent_status {
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
                    span.style = span.style.bg(super::super::mouse::SELECTION_BG);
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
