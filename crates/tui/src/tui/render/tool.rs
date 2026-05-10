//! Tool request/response renderers.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use super::XmlRenderer;

/// Renders `<ToolResponse>` with green check / red X.
pub struct ToolResponseRenderer;

impl XmlRenderer for ToolResponseRenderer {
    fn tags(&self) -> &[&str] {
        &["ToolResponse"]
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        let is_error = xml.contains("<error>") || xml.contains("<success>false</success>");
        let (icon, color) = if is_error {
            ("X", Color::Red)
        } else {
            ("OK", Color::Green)
        };

        // Extract result or error content
        let content = extract_between(xml, "<result>", "</result>")
            .or_else(|| extract_between(xml, "<error>", "</error>"))
            .unwrap_or_else(|| xml.to_string());

        Text::from(vec![Line::from(vec![
            Span::styled(format!("[{icon}] "), Style::default().fg(color)),
            Span::raw(truncate(&content, 120)),
        ])])
    }
}

/// Renders tool request tags: `<FileOpsRequest>`, `<ShellRequest>`, `<CodeIndexRequest>`.
pub struct ToolRequestRenderer;

impl XmlRenderer for ToolRequestRenderer {
    fn tags(&self) -> &[&str] {
        &["FileOpsRequest", "ShellRequest", "CodeIndexRequest"]
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        let mut lines = Vec::new();

        if let Some(action) = extract_between(xml, "<action>", "</action>") {
            if let Some(path) = extract_between(xml, "<path>", "</path>") {
                lines.push(Line::from(vec![
                    Span::styled("action: ", Style::default().fg(Color::Cyan)),
                    Span::raw(action),
                    Span::raw(" "),
                    Span::styled("path: ", Style::default().fg(Color::Cyan)),
                    Span::raw(path),
                ]));
                return Text::from(lines);
            }
        }

        if let Some(cmd) = extract_between(xml, "<command>", "</command>") {
            lines.push(Line::from(vec![
                Span::styled("$ ", Style::default().fg(Color::Yellow)),
                Span::raw(cmd),
            ]));
            return Text::from(lines);
        }

        Text::from(vec![Line::from(truncate(xml, 120))])
    }
}

fn extract_between(xml: &str, open: &str, close: &str) -> Option<String> {
    let start = xml.find(open)? + open.len();
    let end = xml[start..].find(close)? + start;
    Some(xml[start..end].to_string())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_tool_response_success() {
        let renderer = ToolResponseRenderer;
        let xml = "<ToolResponse><success>true</success><result>file contents here</result></ToolResponse>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        // Should start with green OK
        assert!(line.to_string().contains("OK"));
        assert!(line.to_string().contains("file contents"));
    }

    #[test]
    fn render_tool_response_error() {
        let renderer = ToolResponseRenderer;
        let xml = "<ToolResponse><error>file not found</error></ToolResponse>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        assert!(line.to_string().contains("X"));
        assert!(line.to_string().contains("file not found"));
    }

    #[test]
    fn render_file_ops_request() {
        let renderer = ToolRequestRenderer;
        let xml = "<FileOpsRequest><action>read</action><path>/src/main.rs</path></FileOpsRequest>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        assert!(line.to_string().contains("read"));
        assert!(line.to_string().contains("/src/main.rs"));
    }

    #[test]
    fn render_shell_request() {
        let renderer = ToolRequestRenderer;
        let xml = "<ShellRequest><command>cargo test</command></ShellRequest>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        assert!(line.to_string().contains("cargo test"));
    }
}
