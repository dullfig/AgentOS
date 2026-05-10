//! Tag-based XML renderer framework.
//!
//! Renderers register for XML tag names (strings). WASM tools' custom
//! tags get custom renderers without recompilation. `GenericXmlRenderer`
//! handles unknown tags.

pub mod agent;
pub mod context;
pub mod generic;
pub mod tool;

use ratatui::text::Text;

/// A renderer that can convert XML content into ratatui Text.
pub trait XmlRenderer: Send + Sync {
    /// Which XML tags this renderer handles.
    fn tags(&self) -> &[&str];
    /// Render the XML content into styled text.
    fn render<'a>(&self, xml: &str) -> Text<'a>;
}

/// Registry of XML renderers with fallback to generic.
pub struct RendererRegistry {
    renderers: Vec<Box<dyn XmlRenderer>>,
    fallback: generic::GenericXmlRenderer,
}

impl RendererRegistry {
    /// Create a new registry with built-in renderers.
    pub fn new() -> Self {
        let renderers: Vec<Box<dyn XmlRenderer>> = vec![
            Box::new(tool::ToolResponseRenderer),
            Box::new(tool::ToolRequestRenderer),
            Box::new(agent::AgentTaskRenderer),
            Box::new(agent::AgentResponseRenderer),
            Box::new(context::ContextSegmentRenderer),
        ];
        Self {
            renderers,
            fallback: generic::GenericXmlRenderer,
        }
    }

    /// Find the appropriate renderer and render the XML.
    pub fn render<'a>(&self, tag: &str, xml: &str) -> Text<'a> {
        for renderer in &self.renderers {
            if renderer.tags().contains(&tag) {
                return renderer.render(xml);
            }
        }
        self.fallback.render(xml)
    }
}

impl Default for RendererRegistry {
    fn default() -> Self {
        Self::new()
    }
}
