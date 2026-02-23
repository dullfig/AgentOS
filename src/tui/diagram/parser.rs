//! D2 text → Graph IR parser.
//!
//! Handles node declarations, connections (all 4 directions), chains,
//! containers, comments, and semicolons. Implicit node creation from edges.

/// Supported node shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Rectangle,
    Diamond,
    Cylinder,
    Circle,
    Hexagon,
    Cloud,
}

/// A node in the graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub shape: Shape,
    pub container: Option<String>,
}

/// Edge direction.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeDir {
    Forward,  // ->
    Back,     // <-
    Both,     // <->
    None,     // --
}

/// An edge between two nodes.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
    pub direction: EdgeDir,
}

/// A container grouping nodes.
#[derive(Debug, Clone)]
pub struct Container {
    pub id: String,
    pub label: String,
    pub children: Vec<String>,
}

/// The parsed graph intermediate representation.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub containers: Vec<Container>,
}

impl Graph {
    fn ensure_node(&mut self, id: &str) {
        if !self.nodes.iter().any(|n| n.id == id) {
            self.nodes.push(Node {
                id: id.to_string(),
                label: id.to_string(),
                shape: Shape::Rectangle,
                container: None,
            });
        }
    }

    fn set_node_label(&mut self, id: &str, label: &str) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.label = label.to_string();
        }
    }

    fn set_node_shape(&mut self, id: &str, shape: Shape) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.shape = shape;
        }
    }

    fn set_node_container(&mut self, id: &str, container: &str) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.container = Some(container.to_string());
        }
    }
}

/// Parse a shape name string into a Shape enum.
fn parse_shape(s: &str) -> Shape {
    match s.trim().to_lowercase().as_str() {
        "diamond" => Shape::Diamond,
        "cylinder" => Shape::Cylinder,
        "circle" => Shape::Circle,
        "hexagon" => Shape::Hexagon,
        "cloud" => Shape::Cloud,
        _ => Shape::Rectangle,
    }
}

/// Strip surrounding quotes from a string if present.
fn unquote(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Parse a container block: `id: { child1; child2 }` or multi-line.
/// Returns (container, lines consumed).
fn parse_container_block(id: &str, label: &str, lines: &[&str], start: usize) -> (Container, Vec<String>, usize) {
    let mut children = Vec::new();
    let mut i = start;
    let mut brace_depth = 1;

    while i < lines.len() && brace_depth > 0 {
        let line = lines[i].trim();
        for ch in line.chars() {
            match ch {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }
        }
        if brace_depth > 0 {
            // Parse children inside the block
            let inner = line.trim_end_matches('}').trim();
            for part in inner.split(';') {
                let part = part.trim();
                if !part.is_empty() && !part.starts_with('#') {
                    // Could be a node id or "node: label"
                    let child_id = if let Some(colon_pos) = part.find(':') {
                        part[..colon_pos].trim().to_string()
                    } else {
                        part.to_string()
                    };
                    if !child_id.is_empty() {
                        children.push(child_id);
                    }
                }
            }
        }
        i += 1;
    }

    let container = Container {
        id: id.to_string(),
        label: if label.is_empty() { id.to_string() } else { label.to_string() },
        children: children.clone(),
    };
    (container, children, i)
}

/// Check if a `{ ... }` block is a property block (contains `shape:`, etc.)
/// rather than a container.
fn is_property_block(inner: &str) -> bool {
    let lower = inner.to_lowercase();
    lower.contains("shape:") || lower.contains("style:") || lower.contains("icon:")
        || lower.contains("label:")
}

/// Split a line on semicolons, but not those inside `{ }` braces.
fn split_semicolons(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: usize = 0;
    let mut start = 0;
    for (i, ch) in line.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ';' if depth == 0 => {
                parts.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&line[start..]);
    parts
}

/// Parse D2 source text into a Graph IR.
pub fn parse_d2(input: &str) -> Graph {
    let mut graph = Graph {
        nodes: Vec::new(),
        edges: Vec::new(),
        containers: Vec::new(),
    };

    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let raw_line = lines[i];
        let line = raw_line.trim();

        // Skip blanks and comments
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        // Split on semicolons (respecting braces)
        let statements = split_semicolons(line);
        for stmt in &statements {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }

            // Check for `id: { ... }` patterns (container or property block)
            if let Some(colon_pos) = stmt.find(':') {
                let after_colon = stmt[colon_pos + 1..].trim();
                if after_colon.starts_with('{') {
                    let id = stmt[..colon_pos].trim();

                    if after_colon.contains('}') {
                        let inner = after_colon
                            .trim_start_matches('{')
                            .trim_end_matches('}')
                            .trim();

                        if is_property_block(inner) {
                            // Property block: `x: { shape: diamond }`
                            // Delegate to parse_statement which handles parse_node_decl
                            parse_statement(&mut graph, stmt);
                            continue;
                        }

                        // Container: `group: { a; b }`
                        let children: Vec<String> = inner
                            .split(';')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        for child_id in &children {
                            graph.ensure_node(child_id);
                            graph.set_node_container(child_id, id);
                        }
                        graph.containers.push(Container {
                            id: id.to_string(),
                            label: id.to_string(),
                            children,
                        });
                        continue;
                    }

                    // Multi-line container
                    let (container, children, next_i) =
                        parse_container_block(id, "", &lines, i + 1);
                    for child_id in &children {
                        graph.ensure_node(child_id);
                        graph.set_node_container(child_id, id);
                    }
                    graph.containers.push(container);
                    i = next_i;
                    break;
                }
            }

            parse_statement(&mut graph, stmt);
        }

        i += 1;
    }

    graph
}

/// Parse a single statement (no semicolons, no container blocks).
fn parse_statement(graph: &mut Graph, stmt: &str) {
    // Try to find edge operators to split into a chain
    let mut tokens: Vec<String> = Vec::new();
    let mut dirs: Vec<EdgeDir> = Vec::new();
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut remaining = stmt.trim();

    // First token (node id, possibly with colon label)
    loop {
        // Find the next edge operator
        let mut found_op = false;
        let mut best_pos = remaining.len();
        let mut best_dir = EdgeDir::Forward;
        let mut best_op_len = 0;

        for (op_str, dir, op_len) in &[
            ("<->", EdgeDir::Both, 3),
            ("->", EdgeDir::Forward, 2),
            ("<-", EdgeDir::Back, 2),
            ("--", EdgeDir::None, 2),
        ] {
            if let Some(pos) = remaining.find(op_str) {
                if pos < best_pos {
                    best_pos = pos;
                    best_dir = dir.clone();
                    best_op_len = *op_len;
                    found_op = true;
                }
            }
        }

        if found_op {
            let before = remaining[..best_pos].trim();
            if !before.is_empty() {
                tokens.push(before.to_string());
            }
            let after_op = remaining[best_pos + best_op_len..].trim();

            // Check for edge label: `-> "label": target` or `-> label: target`
            // Actually D2 uses: `a -> b: label` where label is after the LAST colon
            dirs.push(best_dir);
            labels.push(Option::None); // label parsed later
            remaining = after_op;
        } else {
            // No more operators
            if !remaining.is_empty() {
                tokens.push(remaining.to_string());
            }
            break;
        }
    }

    if tokens.is_empty() {
        return;
    }

    // If no edges, it's a node declaration
    if dirs.is_empty() {
        let (id, label, shape) = parse_node_decl(&tokens[0]);
        graph.ensure_node(&id);
        if let Some(l) = label {
            graph.set_node_label(&id, &l);
        }
        if let Some(s) = shape {
            graph.set_node_shape(&id, s);
        }
        return;
    }

    // Parse the chain: for each pair of adjacent tokens with an edge between
    // The LAST segment may have a label after colon: `a -> b: label`
    // Actually, for chains like `a -> b -> c: label`, the label is on the last edge
    for idx in 0..dirs.len() {
        let from_raw = &tokens[idx];
        let to_raw = if idx + 1 < tokens.len() {
            &tokens[idx + 1]
        } else {
            continue;
        };

        let (from_id, from_label, from_shape) = parse_node_decl(from_raw);
        graph.ensure_node(&from_id);
        if let Some(l) = from_label {
            graph.set_node_label(&from_id, &l);
        }
        if let Some(s) = from_shape {
            graph.set_node_shape(&from_id, s);
        }

        // The to-node may have a colon-separated label for the EDGE
        let (to_id, edge_label) = parse_edge_target(to_raw);
        let (to_id_clean, to_label, to_shape) = parse_node_decl(&to_id);
        graph.ensure_node(&to_id_clean);
        if let Some(l) = to_label {
            graph.set_node_label(&to_id_clean, &l);
        }
        if let Some(s) = to_shape {
            graph.set_node_shape(&to_id_clean, s);
        }

        graph.edges.push(Edge {
            from: from_id.clone(),
            to: to_id_clean,
            label: edge_label,
            direction: dirs[idx].clone(),
        });
    }
}

/// Parse a node declaration like `x`, `x: "Label"`, `x: { shape: diamond }`.
/// Returns (id, optional_label, optional_shape).
fn parse_node_decl(raw: &str) -> (String, Option<String>, Option<Shape>) {
    let raw = raw.trim();

    // Check for property block: `x: { shape: diamond }`
    if let Some(colon_pos) = raw.find(':') {
        let id = raw[..colon_pos].trim().to_string();
        let value = raw[colon_pos + 1..].trim();

        if value.starts_with('{') && value.ends_with('}') {
            let inner = value[1..value.len() - 1].trim();
            // Parse shape property
            if let Some(shape_pos) = inner.find("shape:") {
                let shape_val = inner[shape_pos + 6..].trim().trim_end_matches(';');
                return (id, None, Some(parse_shape(shape_val)));
            }
            return (id, None, None);
        }

        // Simple label: `x: "Label"` or `x: Label`
        let label = unquote(value).to_string();
        return (id, Some(label), None);
    }

    (raw.to_string(), None, None)
}

/// Parse an edge target that may contain an edge label after colon.
/// `b: "label"` → (node_id="b", edge_label=Some("label"))
/// `b` → (node_id="b", edge_label=None)
fn parse_edge_target(raw: &str) -> (String, Option<String>) {
    let raw = raw.trim();
    if let Some(colon_pos) = raw.find(':') {
        let node_part = raw[..colon_pos].trim().to_string();
        let label_part = unquote(raw[colon_pos + 1..].trim()).to_string();
        if label_part.is_empty() {
            (node_part, None)
        } else {
            (node_part, Some(label_part))
        }
    } else {
        (raw.to_string(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_node() {
        let g = parse_d2("x");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].id, "x");
        assert_eq!(g.nodes[0].label, "x");
    }

    #[test]
    fn parse_node_with_label() {
        let g = parse_d2("x: \"Hello World\"");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].id, "x");
        assert_eq!(g.nodes[0].label, "Hello World");
    }

    #[test]
    fn parse_node_with_shape() {
        let g = parse_d2("x: { shape: diamond }");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].shape, Shape::Diamond);
    }

    #[test]
    fn parse_forward_edge() {
        let g = parse_d2("a -> b");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].from, "a");
        assert_eq!(g.edges[0].to, "b");
        assert_eq!(g.edges[0].direction, EdgeDir::Forward);
    }

    #[test]
    fn parse_back_edge() {
        let g = parse_d2("a <- b");
        assert_eq!(g.edges[0].direction, EdgeDir::Back);
    }

    #[test]
    fn parse_both_edge() {
        let g = parse_d2("a <-> b");
        assert_eq!(g.edges[0].direction, EdgeDir::Both);
    }

    #[test]
    fn parse_none_edge() {
        let g = parse_d2("a -- b");
        assert_eq!(g.edges[0].direction, EdgeDir::None);
    }

    #[test]
    fn parse_edge_with_label() {
        let g = parse_d2("a -> b: sends data");
        assert_eq!(g.edges[0].label, Some("sends data".to_string()));
    }

    #[test]
    fn parse_chain() {
        let g = parse_d2("a -> b -> c");
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].from, "a");
        assert_eq!(g.edges[0].to, "b");
        assert_eq!(g.edges[1].from, "b");
        assert_eq!(g.edges[1].to, "c");
    }

    #[test]
    fn parse_container() {
        let g = parse_d2("group: { a; b }");
        assert_eq!(g.containers.len(), 1);
        assert_eq!(g.containers[0].id, "group");
        assert_eq!(g.containers[0].children, vec!["a", "b"]);
        assert!(g.nodes.iter().any(|n| n.id == "a" && n.container == Some("group".to_string())));
    }

    #[test]
    fn parse_comments_and_blanks() {
        let g = parse_d2("# comment\n\na -> b\n# another");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn parse_implicit_nodes() {
        let g = parse_d2("a -> b");
        assert!(g.nodes.iter().any(|n| n.id == "a"));
        assert!(g.nodes.iter().any(|n| n.id == "b"));
    }

    #[test]
    fn parse_empty_input() {
        let g = parse_d2("");
        assert!(g.nodes.is_empty());
        assert!(g.edges.is_empty());
    }

    #[test]
    fn parse_semicolons() {
        let g = parse_d2("a; b; c");
        assert_eq!(g.nodes.len(), 3);
    }
}
