//! Tool editor tab: Python code editor with tree-sitter highlighting.

use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::super::app::{TabId, TuiApp};

pub(super) fn draw_tool_editor(f: &mut Frame, app: &mut TuiApp, area: Rect) {
    let tool_name = match &app.active_tab {
        TabId::Tool(name) => name.clone(),
        _ => return,
    };

    if let Some(state) = app.tool_editors.get(&tool_name) {
        // Draw border with filename as title
        let title = format!(" {} ", tool_name);
        let block = Block::default()
            .title(title)
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(Style::default().fg(Color::Cyan));
        let inner = block.inner(area);
        f.render_widget(block, area);

        app.tool_editor_area = inner;
        f.render_widget(&state.editor, inner);
        // Position cursor
        if let Some((x, y)) = state.editor.get_visible_cursor(&inner) {
            f.set_cursor_position(Position::new(x, y));
        }

        // Modified indicator in bottom-left
        if state.modified {
            let indicator_area = Rect::new(
                inner.x,
                area.y + area.height.saturating_sub(1),
                12,
                1,
            );
            f.render_widget(
                Paragraph::new(Span::styled(
                    " [modified] ",
                    Style::default().fg(Color::Yellow).bg(Color::Black),
                )),
                indicator_area,
            );
        }
    } else {
        // Top border drawn by folder-tab bar.
        let block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray));
        let para = Paragraph::new(Span::styled(
            "Editor not loaded.",
            Style::default().fg(Color::DarkGray),
        ))
        .block(block);
        f.render_widget(para, area);
    }
}
