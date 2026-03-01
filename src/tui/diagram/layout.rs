//! Sugiyama layered layout — assigns (x, y, width, height) to every node and edge.
//!
//! Simple implementation: Kahn's topological sort for layer assignment,
//! barycenter heuristic for crossing minimization, centered coordinate assignment.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::tui::box_drawing::display_width;
use super::parser::{Graph, Shape, EdgeDir};

/// A node with computed position and size.
#[derive(Debug, Clone)]
pub struct PositionedNode {
    pub id: String,
    pub label: String,
    pub shape: Shape,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub container: Option<String>,
}

/// An edge with waypoints for Manhattan routing.
#[derive(Debug, Clone)]
pub struct PositionedEdge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
    pub direction: EdgeDir,
    pub waypoints: Vec<(usize, usize)>,
}

/// A container with computed bounding box.
#[derive(Debug, Clone)]
pub struct PositionedContainer {
    pub id: String,
    pub label: String,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// The fully laid-out graph ready for grid rendering.
#[derive(Debug, Clone)]
pub struct PositionedGraph {
    pub nodes: Vec<PositionedNode>,
    pub edges: Vec<PositionedEdge>,
    pub containers: Vec<PositionedContainer>,
    pub width: usize,
    pub height: usize,
}

const NODE_PAD_X: usize = 4;  // horizontal gap between nodes
const NODE_PAD_Y: usize = 2;  // vertical gap between layers (for edge routing)
const NODE_HEIGHT: usize = 3; // box height (top border + label + bottom border)
const CONTAINER_PAD: usize = 2; // padding inside container borders

/// Lay out a parsed Graph using a simplified Sugiyama algorithm.
pub fn layout(graph: &Graph, max_width: usize) -> PositionedGraph {
    if graph.nodes.is_empty() {
        return PositionedGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            containers: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    // Build adjacency for forward edges (treat Back as reversed Forward)
    let mut forward_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for node in &graph.nodes {
        forward_adj.entry(node.id.as_str()).or_default();
        in_degree.entry(node.id.as_str()).or_insert(0);
    }
    for edge in &graph.edges {
        let (from, to) = match edge.direction {
            EdgeDir::Back => (edge.to.as_str(), edge.from.as_str()),
            _ => (edge.from.as_str(), edge.to.as_str()),
        };
        forward_adj.entry(from).or_default().push(to);
        *in_degree.entry(to).or_insert(0) += 1;
        in_degree.entry(from).or_insert(0);
    }

    // Step 1: Layer assignment via Kahn's algorithm (longest path)
    let layers = assign_layers(&forward_adj, &in_degree, &graph.nodes);

    // Step 2: Order nodes within each layer (barycenter)
    let ordered_layers = minimize_crossings(&layers, &forward_adj);

    // Step 3: Compute node sizes
    let node_widths: HashMap<String, usize> = graph
        .nodes
        .iter()
        .map(|n| {
            let label_w = display_width(&n.label);
            // Box width = label + 2 border chars + 2 padding spaces
            (n.id.clone(), label_w + 4)
        })
        .collect();

    // Step 4: Assign coordinates
    let mut positioned_nodes: Vec<PositionedNode> = Vec::new();
    let mut node_positions: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut total_height: usize = 0;
    let mut total_width: usize = 0;

    for (layer_idx, layer) in ordered_layers.iter().enumerate() {
        let y = layer_idx * (NODE_HEIGHT + NODE_PAD_Y);

        // Calculate total width of this layer
        let layer_total_w: usize = layer.iter().map(|id| node_widths[*id]).sum::<usize>()
            + if layer.len() > 1 { (layer.len() - 1) * NODE_PAD_X } else { 0 };

        // Center the layer
        let start_x = if layer_total_w < max_width {
            (max_width.saturating_sub(layer_total_w)) / 2
        } else {
            0
        };

        let mut x = start_x;
        for id in layer {
            let w = node_widths[*id].min(max_width.saturating_sub(x));
            let clamped_x = x.min(max_width.saturating_sub(w));
            let node = graph.nodes.iter().find(|n| n.id == *id).unwrap();
            positioned_nodes.push(PositionedNode {
                id: id.to_string(),
                label: node.label.clone(),
                shape: node.shape.clone(),
                x: clamped_x,
                y,
                width: w,
                height: NODE_HEIGHT,
                container: node.container.clone(),
            });
            node_positions.insert(id.to_string(), (clamped_x, y, w, NODE_HEIGHT));
            x = clamped_x + w + NODE_PAD_X;
        }
        total_width = total_width.max(x.min(max_width));
        total_height = y + NODE_HEIGHT;
    }

    // Step 5: Route edges (Manhattan routing)
    let positioned_edges = route_edges(graph, &node_positions);

    // Step 6: Container bounds
    let positioned_containers = compute_container_bounds(graph, &node_positions);

    // Extend height to include any peer-edge U-shapes that route below the last layer.
    let edge_max_y = positioned_edges
        .iter()
        .flat_map(|e| e.waypoints.iter().map(|&(_, y)| y))
        .max()
        .unwrap_or(0);
    let total_height = total_height.max(edge_max_y + 1);

    PositionedGraph {
        nodes: positioned_nodes,
        edges: positioned_edges,
        containers: positioned_containers,
        width: total_width.min(max_width),
        height: total_height,
    }
}

/// Assign layers using Kahn's topological sort with longest-path assignment.
fn assign_layers<'a>(
    adj: &HashMap<&'a str, Vec<&'a str>>,
    in_degree: &HashMap<&'a str, usize>,
    nodes: &'a [super::parser::Node],
) -> Vec<Vec<&'a str>> {
    let mut in_deg: HashMap<&str, usize> = in_degree.clone();
    let mut queue: VecDeque<&str> = VecDeque::new();
    let mut node_layer: HashMap<&str, usize> = HashMap::new();

    // Start with nodes that have no incoming edges
    for node in nodes {
        let id = node.id.as_str();
        if *in_deg.get(id).unwrap_or(&0) == 0 {
            queue.push_back(id);
            node_layer.insert(id, 0);
        }
    }

    // Handle cycles: if no sources, pick first node
    if queue.is_empty() && !nodes.is_empty() {
        let id = nodes[0].id.as_str();
        queue.push_back(id);
        node_layer.insert(id, 0);
    }

    let mut max_layer = 0;
    let mut visited: HashSet<&str> = HashSet::new();

    while let Some(node) = queue.pop_front() {
        if visited.contains(node) {
            continue;
        }
        visited.insert(node);
        let current_layer = node_layer[node];

        if let Some(neighbors) = adj.get(node) {
            for &next in neighbors {
                let new_layer = current_layer + 1;
                let existing = node_layer.get(next).copied().unwrap_or(0);
                if new_layer > existing {
                    node_layer.insert(next, new_layer);
                }
                max_layer = max_layer.max(new_layer);

                // Decrease in-degree
                if let Some(deg) = in_deg.get_mut(next) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    // Any unvisited nodes (isolated or in cycles) go to layer 0
    for node in nodes {
        let id = node.id.as_str();
        if !visited.contains(id) {
            node_layer.insert(id, 0);
            visited.insert(id);
        }
    }

    // Group by layer
    let mut layers: Vec<Vec<&str>> = vec![Vec::new(); max_layer + 1];
    for node in nodes {
        let id = node.id.as_str();
        let layer = node_layer.get(id).copied().unwrap_or(0);
        if layer <= max_layer {
            layers[layer].push(id);
        }
    }

    // Remove empty layers
    layers.retain(|l| !l.is_empty());
    layers
}

/// Barycenter heuristic: order nodes by average neighbor position.
fn minimize_crossings<'a>(
    layers: &[Vec<&'a str>],
    adj: &HashMap<&str, Vec<&str>>,
) -> Vec<Vec<&'a str>> {
    let mut result = layers.to_vec();

    if result.len() <= 1 {
        return result;
    }

    // Build position lookup
    let build_positions = |layers: &[Vec<&str>]| -> HashMap<String, usize> {
        let mut pos = HashMap::new();
        for layer in layers {
            for (i, id) in layer.iter().enumerate() {
                pos.insert(id.to_string(), i);
            }
        }
        pos
    };

    // Downward pass
    let positions = build_positions(&result);
    for li in 1..result.len() {
        let prev_layer = &result[li - 1];
        let mut scored: Vec<(&str, f64)> = result[li]
            .iter()
            .map(|&id| {
                // Find neighbors in previous layer
                let mut sum = 0.0;
                let mut count = 0;
                for &prev_id in prev_layer {
                    if let Some(neighbors) = adj.get(prev_id) {
                        if neighbors.contains(&id) {
                            sum += *positions.get(prev_id).unwrap_or(&0) as f64;
                            count += 1;
                        }
                    }
                }
                let bary = if count > 0 { sum / count as f64 } else { *positions.get(id).unwrap_or(&0) as f64 };
                (id, bary)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result[li] = scored.into_iter().map(|(id, _)| id).collect();
    }

    // Upward pass
    let positions = build_positions(&result);
    for li in (0..result.len().saturating_sub(1)).rev() {
        let next_layer = &result[li + 1];
        let mut scored: Vec<(&str, f64)> = result[li]
            .iter()
            .map(|&id| {
                let mut sum = 0.0;
                let mut count = 0;
                if let Some(neighbors) = adj.get(id) {
                    for &next_id in next_layer {
                        if neighbors.contains(&next_id) {
                            sum += *positions.get(next_id).unwrap_or(&0) as f64;
                            count += 1;
                        }
                    }
                }
                let bary = if count > 0 { sum / count as f64 } else { *positions.get(id).unwrap_or(&0) as f64 };
                (id, bary)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result[li] = scored.into_iter().map(|(id, _)| id).collect();
    }

    result
}

/// Manhattan routing for edges.
fn route_edges(
    graph: &Graph,
    positions: &HashMap<String, (usize, usize, usize, usize)>,
) -> Vec<PositionedEdge> {
    graph
        .edges
        .iter()
        .filter_map(|edge| {
            let (from, to) = match edge.direction {
                EdgeDir::Back => (&edge.to, &edge.from),
                _ => (&edge.from, &edge.to),
            };
            let &(fx, fy, fw, fh) = positions.get(from.as_str())?;
            let &(tx, ty, tw, _th) = positions.get(to.as_str())?;

            // Exit below source, arrive above target (so arrow isn't overwritten by node box)
            let start_x = fx + fw / 2;
            let start_y = fy + fh; // one row below source bottom border
            let end_x = tx + tw / 2;
            let end_y = ty.saturating_sub(1); // one row above target top border

            let waypoints = if end_y <= start_y {
                // Same layer (or overlapping): route as a U-shape below the row.
                // Go down one row, across, then back up to the target's bottom border.
                let bypass_y = start_y.max(ty + fh) + 1;
                if start_x == end_x {
                    // Directly above/below — just a stub down to the bypass row
                    vec![
                        (start_x, start_y),
                        (start_x, bypass_y),
                        (end_x, bypass_y),
                        (end_x, ty + fh - 1),
                    ]
                } else {
                    // U-shape: down, across, up
                    vec![
                        (start_x, start_y),
                        (start_x, bypass_y),
                        (end_x, bypass_y),
                        (end_x, ty + fh - 1),
                    ]
                }
            } else {
                let mid_y = (start_y + end_y) / 2;

                if start_x == end_x {
                    // Straight vertical
                    vec![(start_x, start_y), (start_x, end_y)]
                } else {
                    // Manhattan: down, across, down
                    vec![
                        (start_x, start_y),
                        (start_x, mid_y),
                        (end_x, mid_y),
                        (end_x, end_y),
                    ]
                }
            };

            Some(PositionedEdge {
                from: edge.from.clone(),
                to: edge.to.clone(),
                label: edge.label.clone(),
                direction: edge.direction.clone(),
                waypoints,
            })
        })
        .collect()
}

/// Compute container bounding boxes around their children.
fn compute_container_bounds(
    graph: &Graph,
    positions: &HashMap<String, (usize, usize, usize, usize)>,
) -> Vec<PositionedContainer> {
    graph
        .containers
        .iter()
        .filter_map(|c| {
            let child_positions: Vec<&(usize, usize, usize, usize)> = c
                .children
                .iter()
                .filter_map(|id| positions.get(id.as_str()))
                .collect();

            if child_positions.is_empty() {
                return None;
            }

            let min_x = child_positions.iter().map(|p| p.0).min().unwrap();
            let min_y = child_positions.iter().map(|p| p.1).min().unwrap();
            let max_x = child_positions.iter().map(|p| p.0 + p.2).max().unwrap();
            let max_y = child_positions.iter().map(|p| p.1 + p.3).max().unwrap();

            Some(PositionedContainer {
                id: c.id.clone(),
                label: c.label.clone(),
                x: min_x.saturating_sub(CONTAINER_PAD),
                y: min_y.saturating_sub(CONTAINER_PAD),
                width: (max_x - min_x) + 2 * CONTAINER_PAD,
                height: (max_y - min_y) + 2 * CONTAINER_PAD,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::diagram::parser::parse_d2;

    #[test]
    fn single_node_centered() {
        let g = parse_d2("x");
        let pg = layout(&g, 80);
        assert_eq!(pg.nodes.len(), 1);
        assert!(pg.nodes[0].x > 0); // centered, not at 0
        assert_eq!(pg.nodes[0].y, 0);
    }

    #[test]
    fn linear_chain_stacked() {
        let g = parse_d2("a -> b -> c");
        let pg = layout(&g, 80);
        assert_eq!(pg.nodes.len(), 3);
        // Each node should be in a different layer (increasing y)
        let a = pg.nodes.iter().find(|n| n.id == "a").unwrap();
        let b = pg.nodes.iter().find(|n| n.id == "b").unwrap();
        let c = pg.nodes.iter().find(|n| n.id == "c").unwrap();
        assert!(a.y < b.y);
        assert!(b.y < c.y);
    }

    #[test]
    fn diamond_layout() {
        let g = parse_d2("a -> b\na -> c\nb -> d\nc -> d");
        let pg = layout(&g, 80);
        let a = pg.nodes.iter().find(|n| n.id == "a").unwrap();
        let b = pg.nodes.iter().find(|n| n.id == "b").unwrap();
        let d = pg.nodes.iter().find(|n| n.id == "d").unwrap();
        assert!(a.y < b.y);
        assert!(b.y < d.y);
    }

    #[test]
    fn container_bounds() {
        let g = parse_d2("group: { a; b }");
        let pg = layout(&g, 80);
        assert_eq!(pg.containers.len(), 1);
        assert!(pg.containers[0].width > 0);
        assert!(pg.containers[0].height > 0);
    }

    #[test]
    fn cycle_still_lays_out() {
        let g = parse_d2("a -> b\nb -> a");
        let pg = layout(&g, 80);
        assert_eq!(pg.nodes.len(), 2);
        // Should not panic or infinite loop
    }

    #[test]
    fn empty_graph() {
        let g = parse_d2("");
        let pg = layout(&g, 80);
        assert_eq!(pg.width, 0);
        assert_eq!(pg.height, 0);
    }
}
