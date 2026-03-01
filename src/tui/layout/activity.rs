//! Activity trace tab rendering.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::layout::Rect;
use ratatui::Frame;

use super::super::app::TuiApp;

pub(super) fn draw_activity(f: &mut Frame, app: &mut TuiApp, area: Rect) {
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
                    super::super::app::ActivityStatus::InProgress => {
                        Span::styled("...", Style::default().fg(Color::Yellow))
                    }
                    super::super::app::ActivityStatus::Done => {
                        Span::styled(" OK", Style::default().fg(Color::Green))
                    }
                    super::super::app::ActivityStatus::Error => {
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
