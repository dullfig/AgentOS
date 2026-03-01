//! YAML editor tab: editor + completion popup + hover overlay.

use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::super::app::TuiApp;

pub(super) fn draw_yaml_editor(f: &mut Frame, app: &mut TuiApp, area: Rect) {
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
