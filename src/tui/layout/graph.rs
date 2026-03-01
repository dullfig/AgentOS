//! Graph tab rendering (organism topology as D2 diagrams).

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::layout::Rect;
use ratatui::Frame;

use super::super::app::TuiApp;

pub(super) fn draw_graph(f: &mut Frame, app: &mut TuiApp, area: Rect) {
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
            app.graph_rendered_lines = super::super::diagram::render_d2(&app.graph_d2_source, inner_width);
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
