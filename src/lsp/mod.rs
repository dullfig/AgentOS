//! In-process language intelligence — no JSON-RPC, no transport.
//!
//! Pure functions that operate directly on editor buffer content.
//! Uses `lsp-types` for standard data structures (Diagnostic, CompletionItem, etc.)
//! but calls them synchronously — no tower-lsp, no async.
//!
//! Each language gets a struct implementing `LanguageService`.
//! The TUI wires buffer changes to the service and renders results
//! (marks for errors, popup for completions, overlay for hover).

pub mod organism;

use lsp_types::{CompletionItem, Diagnostic, Position};

/// Hover information for a position in the document.
pub struct HoverInfo {
    /// Content to display (plain text or markdown).
    pub content: String,
    /// Optional range the hover applies to.
    pub range: Option<lsp_types::Range>,
}

/// In-process language service — pure functions, no transport.
pub trait LanguageService {
    /// Compute diagnostics for the given content.
    fn diagnostics(&self, content: &str) -> Vec<Diagnostic>;

    /// Compute completions at the given position.
    fn completions(&self, content: &str, pos: Position) -> Vec<CompletionItem>;

    /// Compute hover information at the given position.
    fn hover(&self, content: &str, pos: Position) -> Option<HoverInfo>;
}
