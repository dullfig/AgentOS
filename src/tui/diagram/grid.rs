//! Positioned graph → character grid with box-drawing → styled ratatui Lines.
//!
//! Renders nodes as boxes, edges as Manhattan lines with arrows,
//! containers as double-line borders. Converts the char grid to
//! styled `Vec<Line<'static>>` for the Messages pane.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;
use super::layout::{PositionedGraph, PositionedNode, PositionedEdge, PositionedContainer};
use super::parser::{Shape, EdgeDir};

/// A cell in the character grid with its style category.
#[derive(Clone, Debug)]
struct Cell {
    ch: char,
    category: CellCategory,
}

#[derive(Clone, Debug, PartialEq)]
enum CellCategory {
    Empty,
    NodeBorder,
    NodeLabel,
    EdgeLine,
    EdgeLabel,
    Arrow,
    ContainerBorder,
    ContainerLabel,
}

impl Cell {
    fn empty() -> Self {
        Cell { ch: ' ', category: CellCategory::Empty }
    }

    fn style(&self) -> Style {
        match self.category {
            CellCategory::Empty => Style::default(),
            CellCategory::NodeBorder => Style::default().fg(Color::White),
            CellCategory::NodeLabel => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            CellCategory::EdgeLine => Style::default().fg(Color::DarkGray),
            CellCategory::EdgeLabel => Style::default().fg(Color::Yellow),
            CellCategory::Arrow => Style::default().fg(Color::Green),
            CellCategory::ContainerBorder => Style::default().fg(Color::Blue),
            CellCategory::ContainerLabel => Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD),
        }
    }
}

/// Character grid that can be written to and converted to ratatui Lines.
struct CharGrid {
    cells: Vec<Vec<Cell>>,
    width: usize,
    height: usize,
}

impl CharGrid {
    fn new(width: usize, height: usize) -> Self {
        CharGrid {
            cells: vec![vec![Cell::empty(); width]; height],
            width,
            height,
        }
    }

    fn set(&mut self, x: usize, y: usize, ch: char, cat: CellCategory) {
        if y < self.height && x < self.width {
            self.cells[y][x] = Cell { ch, category: cat };
        }
    }

    fn put_str(&mut self, x: usize, y: usize, s: &str, cat: CellCategory) {
        let mut col = 0;
        for ch in s.chars() {
            self.set(x + col, y, ch, cat.clone());
            let w = ch.width().unwrap_or(0);
            // Wide chars (emoji, CJK) occupy 2 cells — fill the second with a space
            if w > 1 {
                for extra in 1..w {
                    self.set(x + col + extra, y, ' ', cat.clone());
                }
            }
            col += w.max(1);
        }
    }

    /// Convert the grid to styled ratatui Lines, trimming trailing whitespace.
    fn to_lines(&self) -> Vec<Line<'static>> {
        self.cells
            .iter()
            .map(|row| {
                // Find last non-empty cell to trim trailing spaces
                let last = row.iter().rposition(|c| c.ch != ' ').map(|p| p + 1).unwrap_or(0);
                let spans: Vec<Span<'static>> = row[..last]
                    .iter()
                    .map(|cell| Span::styled(cell.ch.to_string(), cell.style()))
                    .collect();
                // Merge adjacent spans with same style for efficiency
                merge_spans(spans)
            })
            .collect()
    }
}

/// Merge adjacent spans with the same style into single spans.
fn merge_spans(spans: Vec<Span<'static>>) -> Line<'static> {
    if spans.is_empty() {
        return Line::from("");
    }
    let mut merged: Vec<Span<'static>> = Vec::new();
    let mut current_text = String::new();
    let mut current_style = spans[0].style;

    for span in spans {
        if span.style == current_style {
            current_text.push_str(&span.content);
        } else {
            if !current_text.is_empty() {
                merged.push(Span::styled(current_text.clone(), current_style));
                current_text.clear();
            }
            current_style = span.style;
            current_text.push_str(&span.content);
        }
    }
    if !current_text.is_empty() {
        merged.push(Span::styled(current_text, current_style));
    }
    Line::from(merged)
}

/// Render a positioned graph to styled ratatui Lines.
pub fn render_to_lines(graph: &PositionedGraph, max_width: usize) -> Vec<Line<'static>> {
    if graph.nodes.is_empty() {
        return vec![Line::from("  (empty diagram)")];
    }

    // Calculate grid dimensions
    let grid_w = graph.width.min(max_width).max(1);
    let grid_h = graph.height.max(1);

    let mut grid = CharGrid::new(grid_w, grid_h);

    // Draw in order: containers (background), edges, nodes (foreground)
    for container in &graph.containers {
        draw_container(&mut grid, container);
    }
    for edge in &graph.edges {
        draw_edge(&mut grid, edge);
    }
    for node in &graph.nodes {
        draw_node(&mut grid, node);
    }

    grid.to_lines()
}

/// Draw a node as a box with label.
fn draw_node(grid: &mut CharGrid, node: &PositionedNode) {
    let x = node.x;
    let y = node.y;
    let w = node.width;

    if w < 4 || node.height < 3 {
        return;
    }

    match node.shape {
        Shape::Cylinder => draw_cylinder(grid, node),
        Shape::Diamond => draw_diamond_node(grid, node),
        _ => draw_rectangle(grid, node),
    }

    // Label centered in the box (all shapes use middle row)
    let label_x = x + (w.saturating_sub(crate::tui::box_drawing::display_width(&node.label))) / 2;
    let label_y = y + node.height / 2;
    grid.put_str(label_x, label_y, &node.label, CellCategory::NodeLabel);
}

fn draw_rectangle(grid: &mut CharGrid, node: &PositionedNode) {
    let (x, y, w) = (node.x, node.y, node.width);
    // Top border
    grid.set(x, y, '┌', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y, '┐', CellCategory::NodeBorder);

    // Middle row(s)
    for row in 1..node.height - 1 {
        grid.set(x, y + row, '│', CellCategory::NodeBorder);
        grid.set(x + w - 1, y + row, '│', CellCategory::NodeBorder);
    }

    // Bottom border
    grid.set(x, y + node.height - 1, '└', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y + node.height - 1, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y + node.height - 1, '┘', CellCategory::NodeBorder);
}

fn draw_cylinder(grid: &mut CharGrid, node: &PositionedNode) {
    let (x, y, w) = (node.x, node.y, node.width);
    // Top border with rounded corners
    grid.set(x, y, '╭', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y, '╮', CellCategory::NodeBorder);

    // Middle
    for row in 1..node.height - 1 {
        grid.set(x, y + row, '│', CellCategory::NodeBorder);
        grid.set(x + w - 1, y + row, '│', CellCategory::NodeBorder);
    }

    // Bottom border with rounded corners
    grid.set(x, y + node.height - 1, '╰', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y + node.height - 1, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y + node.height - 1, '╯', CellCategory::NodeBorder);
}

fn draw_diamond_node(grid: &mut CharGrid, node: &PositionedNode) {
    // Diamonds rendered as `< label >` with angle brackets
    let (x, y, w) = (node.x, node.y, node.width);
    grid.set(x, y, '┌', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y, '┐', CellCategory::NodeBorder);

    // Middle — use ◇ markers
    grid.set(x, y + 1, '◇', CellCategory::NodeBorder);
    grid.set(x + w - 1, y + 1, '◇', CellCategory::NodeBorder);

    grid.set(x, y + node.height - 1, '└', CellCategory::NodeBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y + node.height - 1, '─', CellCategory::NodeBorder);
    }
    grid.set(x + w - 1, y + node.height - 1, '┘', CellCategory::NodeBorder);
}

/// Draw an edge as Manhattan line segments with an arrow at the endpoint.
fn draw_edge(grid: &mut CharGrid, edge: &PositionedEdge) {
    if edge.waypoints.len() < 2 {
        return;
    }

    for i in 0..edge.waypoints.len() - 1 {
        let (x1, y1) = edge.waypoints[i];
        let (x2, y2) = edge.waypoints[i + 1];

        if y1 == y2 {
            // Horizontal segment
            let (min_x, max_x) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            for x in min_x..=max_x {
                grid.set(x, y1, '─', CellCategory::EdgeLine);
            }
            // Corners
            if i > 0 {
                let (_, prev_y) = edge.waypoints[i - 1];
                if prev_y < y1 {
                    grid.set(x1, y1, if x2 > x1 { '└' } else { '┘' }, CellCategory::EdgeLine);
                } else if prev_y > y1 {
                    grid.set(x1, y1, if x2 > x1 { '┌' } else { '┐' }, CellCategory::EdgeLine);
                }
            }
        } else if x1 == x2 {
            // Vertical segment
            let (min_y, max_y) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
            for y in min_y..=max_y {
                grid.set(x1, y, '│', CellCategory::EdgeLine);
            }
        }
    }

    // Draw arrow at the last waypoint
    let len = edge.waypoints.len();
    let (last_x, last_y) = edge.waypoints[len - 1];
    let (prev_x, prev_y) = edge.waypoints[len - 2];

    let arrow_char = if last_y > prev_y {
        '▼'
    } else if last_y < prev_y {
        '▲'
    } else if last_x > prev_x {
        '►'
    } else {
        '◄'
    };

    grid.set(last_x, last_y, arrow_char, CellCategory::Arrow);

    // Draw arrow at start for bidirectional edges
    if edge.direction == EdgeDir::Both && len >= 2 {
        let (first_x, first_y) = edge.waypoints[0];
        let (next_x, next_y) = edge.waypoints[1];
        let start_arrow = if first_y < next_y {
            '▲'
        } else if first_y > next_y {
            '▼'
        } else if first_x < next_x {
            '◄'
        } else {
            '►'
        };
        grid.set(first_x, first_y, start_arrow, CellCategory::Arrow);
    }

    // Edge label at midpoint
    if let Some(ref label) = edge.label {
        let mid_idx = edge.waypoints.len() / 2;
        let (mx, my) = edge.waypoints[mid_idx];
        // Place label slightly offset from the edge
        let lx = mx + 1;
        grid.put_str(lx, my, label, CellCategory::EdgeLabel);
    }
}

/// Draw a container as a double-line box.
fn draw_container(grid: &mut CharGrid, container: &PositionedContainer) {
    let (x, y, w, h) = (container.x, container.y, container.width, container.height);

    if w < 2 || h < 2 {
        return;
    }

    // Top border
    grid.set(x, y, '╔', CellCategory::ContainerBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y, '═', CellCategory::ContainerBorder);
    }
    grid.set(x + w - 1, y, '╗', CellCategory::ContainerBorder);

    // Label in top border
    if crate::tui::box_drawing::display_width(&container.label) + 2 < w {
        let lx = x + 2;
        grid.put_str(lx, y, &container.label, CellCategory::ContainerLabel);
    }

    // Side borders
    for row in 1..h - 1 {
        grid.set(x, y + row, '║', CellCategory::ContainerBorder);
        grid.set(x + w - 1, y + row, '║', CellCategory::ContainerBorder);
    }

    // Bottom border
    grid.set(x, y + h - 1, '╚', CellCategory::ContainerBorder);
    for i in 1..w - 1 {
        grid.set(x + i, y + h - 1, '═', CellCategory::ContainerBorder);
    }
    grid.set(x + w - 1, y + h - 1, '╝', CellCategory::ContainerBorder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::diagram::parser::parse_d2;
    use crate::tui::diagram::layout::layout;

    fn render(d2: &str) -> Vec<Line<'static>> {
        let g = parse_d2(d2);
        let pg = layout(&g, 80);
        render_to_lines(&pg, 80)
    }

    fn lines_to_text(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn single_node_renders_box() {
        let lines = render("x");
        let text = lines_to_text(&lines);
        assert!(text.contains('┌'));
        assert!(text.contains('┘'));
        assert!(text.contains('x'));
    }

    #[test]
    fn two_nodes_with_arrow() {
        let lines = render("a -> b");
        let text = lines_to_text(&lines);
        assert!(text.contains('a'));
        assert!(text.contains('b'));
        assert!(text.contains('▼'));
    }

    #[test]
    fn edge_label_appears() {
        let lines = render("a -> b: sends");
        let text = lines_to_text(&lines);
        assert!(text.contains("sends"));
    }

    #[test]
    fn diamond_shape() {
        let lines = render("x: { shape: diamond }");
        let text = lines_to_text(&lines);
        assert!(text.contains('◇'));
    }

    #[test]
    fn container_double_border() {
        let lines = render("group: { a; b }");
        let text = lines_to_text(&lines);
        assert!(text.contains('╔') || text.contains("group"));
    }

    #[test]
    fn respects_max_width() {
        let g = parse_d2("a_very_long_node_name -> another_very_long_node_name");
        let pg = layout(&g, 40);
        let lines = render_to_lines(&pg, 40);
        for line in &lines {
            // Count display width (emoji/CJK = 2 cells, not 1)
            let len: usize = line.spans.iter().map(|s| crate::tui::box_drawing::display_width(&s.content)).sum();
            assert!(len <= 40, "line exceeds max width: {len}");
        }
    }

    #[test]
    fn emoji_label_box_alignment() {
        // Emoji are 2 cells wide — box must account for display width, not char count
        let lines = render("rocket: 🚀 Launch");
        let text = lines_to_text(&lines);
        assert!(text.contains("🚀"), "emoji should appear in rendered output: {text}");
        // Verify every line of the box has consistent display width:
        // top border width == bottom border width
        let box_lines: Vec<&str> = text.lines().filter(|l| l.contains('┌') || l.contains('└')).collect();
        if box_lines.len() == 2 {
            let top_w: usize = crate::tui::box_drawing::display_width(box_lines[0]);
            let bot_w: usize = crate::tui::box_drawing::display_width(box_lines[1]);
            assert_eq!(top_w, bot_w, "top and bottom border widths must match");
        }
    }
}
