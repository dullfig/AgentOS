//! Agent task/response renderers.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};

use super::XmlRenderer;

/// Renders `<AgentTask>` with emphasized task text.
pub struct AgentTaskRenderer;

impl XmlRenderer for AgentTaskRenderer {
    fn tags(&self) -> &[&str] {
        &["AgentTask"]
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        let task = extract_between(xml, "<task>", "</task>")
            .unwrap_or_else(|| xml.to_string());

        Text::from(vec![Line::from(vec![
            Span::styled(
                "TASK: ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(task),
        ])])
    }
}

/// Renders `<AgentResponse>` with result text.
pub struct AgentResponseRenderer;

impl XmlRenderer for AgentResponseRenderer {
    fn tags(&self) -> &[&str] {
        &["AgentResponse"]
    }

    fn render<'a>(&self, xml: &str) -> Text<'a> {
        let result = extract_between(xml, "<result>", "</result>")
            .or_else(|| extract_between(xml, "<error>", "</error>"))
            .unwrap_or_else(|| xml.to_string());

        let is_error = xml.contains("<error>");
        let color = if is_error { Color::Red } else { Color::Green };

        Text::from(vec![Line::from(vec![
            Span::styled(
                if is_error { "ERROR: " } else { "RESULT: " },
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(result),
        ])])
    }
}

fn extract_between(xml: &str, open: &str, close: &str) -> Option<String> {
    let start = xml.find(open)? + open.len();
    let end = xml[start..].find(close)? + start;
    Some(xml[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_agent_task() {
        let renderer = AgentTaskRenderer;
        let xml = "<AgentTask><task>Fix the login bug</task></AgentTask>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        assert!(line.to_string().contains("TASK"));
        assert!(line.to_string().contains("Fix the login bug"));
    }

    #[test]
    fn render_agent_response() {
        let renderer = AgentResponseRenderer;
        let xml = "<AgentResponse><result>Done. Fixed 3 files.</result></AgentResponse>";
        let text = renderer.render(xml);
        let line = &text.lines[0];
        assert!(line.to_string().contains("RESULT"));
        assert!(line.to_string().contains("Done. Fixed 3 files."));
    }
}
