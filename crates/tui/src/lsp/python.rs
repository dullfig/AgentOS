//! Language service for Python tool files.
//!
//! Uses tree-sitter to extract symbols from the current buffer
//! and provide completions and basic diagnostics.

use lsp_types::{
    CompletionItem, CompletionItemKind, Diagnostic, DiagnosticSeverity, Position, Range,
};

use super::{HoverInfo, LanguageService};

/// Language service for Python files in the tool editor.
pub struct PythonLanguageService {
    /// Reusable tree-sitter parser (avoids re-init on every call).
    parser: tree_sitter::Parser,
}

impl PythonLanguageService {
    pub fn new() -> Self {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("failed to load Python grammar");
        Self { parser }
    }

    /// Extract all symbol names (functions, classes, variables) from the buffer.
    fn extract_symbols(&mut self, content: &str) -> Vec<Symbol> {
        let Some(tree) = self.parser.parse(content, None) else {
            return vec![];
        };

        let mut symbols = Vec::new();
        Self::walk_node(tree.root_node(), content.as_bytes(), &mut symbols);

        // Add Python builtins for convenience
        for &builtin in PYTHON_BUILTINS {
            symbols.push(Symbol {
                name: builtin.to_string(),
                kind: SymbolKind::Builtin,
                line: 0,
                doc: None,
            });
        }

        symbols
    }

    /// Recursively walk the AST collecting symbol definitions.
    fn walk_node(node: tree_sitter::Node, source: &[u8], symbols: &mut Vec<Symbol>) {
        match node.kind() {
            "function_definition" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = name_node.utf8_text(source).unwrap_or("").to_string();
                    let line = name_node.start_position().row as usize;

                    // Try to extract docstring
                    let doc = Self::extract_docstring(node, source);

                    // Extract parameter names
                    let params = Self::extract_params(node, source);
                    let doc_with_params = if params.is_empty() {
                        doc
                    } else {
                        let sig = format!("def {}({})", name, params.join(", "));
                        Some(match doc {
                            Some(d) => format!("{}\n{}", sig, d),
                            None => sig,
                        })
                    };

                    symbols.push(Symbol {
                        name,
                        kind: SymbolKind::Function,
                        line,
                        doc: doc_with_params,
                    });
                }
            }
            "class_definition" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let name = name_node.utf8_text(source).unwrap_or("").to_string();
                    let line = name_node.start_position().row as usize;
                    let doc = Self::extract_docstring(node, source);
                    symbols.push(Symbol {
                        name,
                        kind: SymbolKind::Class,
                        line,
                        doc,
                    });
                }
            }
            "assignment" => {
                // Top-level assignments: `x = ...`
                // AST: module → expression_statement → assignment
                let is_top_level = node.parent().map_or(false, |p| {
                    p.kind() == "module"
                        || (p.kind() == "expression_statement"
                            && p.parent().map_or(false, |gp| gp.kind() == "module"))
                });
                if is_top_level {
                    if let Some(left) = node.child_by_field_name("left") {
                        if left.kind() == "identifier" {
                            let name = left.utf8_text(source).unwrap_or("").to_string();
                            let line = left.start_position().row;
                            symbols.push(Symbol {
                                name,
                                kind: SymbolKind::Variable,
                                line,
                                doc: None,
                            });
                        }
                    }
                }
            }
            "import_from_statement" | "import_statement" => {
                // Extract imported names
                for i in 0..node.child_count() {
                    let child = node.child(i as u32).unwrap();
                    if child.kind() == "dotted_name" || child.kind() == "aliased_import" {
                        let name = if child.kind() == "aliased_import" {
                            // `import foo as bar` → use "bar"
                            child
                                .child_by_field_name("alias")
                                .or_else(|| child.child_by_field_name("name"))
                                .map(|n| n.utf8_text(source).unwrap_or("").to_string())
                        } else {
                            Some(child.utf8_text(source).unwrap_or("").to_string())
                        };
                        if let Some(name) = name {
                            symbols.push(Symbol {
                                name,
                                kind: SymbolKind::Import,
                                line: child.start_position().row,
                                doc: None,
                            });
                        }
                    }
                }
            }
            _ => {}
        }

        // Recurse into children (but not into function/class bodies for top-level symbols)
        let recurse = !matches!(node.kind(), "function_definition" | "class_definition");
        if recurse {
            for i in 0..node.child_count() {
                Self::walk_node(node.child(i as u32).unwrap(), source, symbols);
            }
        }
    }

    /// Extract docstring from a function or class definition.
    fn extract_docstring(node: tree_sitter::Node, source: &[u8]) -> Option<String> {
        let body = node.child_by_field_name("body")?;
        let first_stmt = body.child(0)?;
        if first_stmt.kind() == "expression_statement" {
            let expr = first_stmt.child(0)?;
            if expr.kind() == "string" {
                let text = expr.utf8_text(source).ok()?;
                // Strip triple quotes
                let trimmed = text
                    .trim_start_matches("\"\"\"")
                    .trim_start_matches("'''")
                    .trim_end_matches("\"\"\"")
                    .trim_end_matches("'''")
                    .trim();
                return Some(trimmed.to_string());
            }
        }
        None
    }

    /// Extract parameter names from a function definition.
    fn extract_params(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
        let Some(params) = node.child_by_field_name("parameters") else {
            return vec![];
        };
        let mut names = Vec::new();
        for i in 0..params.child_count() {
            let child = params.child(i as u32).unwrap();
            match child.kind() {
                "identifier" => {
                    let name = child.utf8_text(source).unwrap_or("");
                    if name != "self" {
                        names.push(name.to_string());
                    }
                }
                "default_parameter" | "typed_parameter" | "typed_default_parameter" => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = name_node.utf8_text(source).unwrap_or("");
                        if name != "self" {
                            names.push(name.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        names
    }

    /// Get the word being typed at the cursor position (public accessor).
    pub fn word_at_cursor_pub(content: &str, pos: Position) -> Option<String> {
        Self::word_at_cursor(content, pos)
    }

    /// Get the word being typed at the cursor position.
    fn word_at_cursor(content: &str, pos: Position) -> Option<String> {
        let line = content.lines().nth(pos.line as usize)?;
        let col = pos.character as usize;
        if col == 0 || col > line.len() {
            return None;
        }
        // Walk backward from cursor to find word start
        let before = &line[..col];
        let word_start = before
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &before[word_start..];
        if word.is_empty() {
            None
        } else {
            Some(word.to_string())
        }
    }
}

impl LanguageService for PythonLanguageService {
    fn diagnostics(&self, content: &str) -> Vec<Diagnostic> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("Python grammar");

        let Some(tree) = parser.parse(content, None) else {
            return vec![Diagnostic {
                range: Range::new(Position::new(0, 0), Position::new(0, 1)),
                severity: Some(DiagnosticSeverity::ERROR),
                message: "Failed to parse Python".to_string(),
                ..Default::default()
            }];
        };

        let mut diags = Vec::new();

        // Report ERROR nodes from tree-sitter
        Self::collect_errors(tree.root_node(), &mut diags);

        diags
    }

    fn completions(&self, content: &str, pos: Position) -> Vec<CompletionItem> {
        let Some(prefix) = Self::word_at_cursor(content, pos) else {
            return vec![];
        };
        let prefix_lower = prefix.to_lowercase();

        // We need a mutable parser but trait takes &self.
        // Create a temporary parser for symbol extraction.
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("Python grammar");

        let Some(tree) = parser.parse(content, None) else {
            return vec![];
        };

        let mut symbols = Vec::new();
        Self::walk_node(tree.root_node(), content.as_bytes(), &mut symbols);

        // Add builtins
        for &builtin in PYTHON_BUILTINS {
            symbols.push(Symbol {
                name: builtin.to_string(),
                kind: SymbolKind::Builtin,
                line: 0,
                doc: None,
            });
        }

        // Filter by prefix and deduplicate
        let mut seen = std::collections::HashSet::new();
        let mut items = Vec::new();

        for sym in &symbols {
            let name_lower = sym.name.to_lowercase();
            if name_lower.starts_with(&prefix_lower) && name_lower != prefix_lower && seen.insert(&sym.name) {
                items.push(CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(match sym.kind {
                        SymbolKind::Function => CompletionItemKind::FUNCTION,
                        SymbolKind::Class => CompletionItemKind::CLASS,
                        SymbolKind::Variable => CompletionItemKind::VARIABLE,
                        SymbolKind::Import => CompletionItemKind::MODULE,
                        SymbolKind::Builtin => CompletionItemKind::KEYWORD,
                    }),
                    detail: sym.doc.clone(),
                    ..Default::default()
                });
            }
        }

        // Sort: user-defined first, then builtins
        items.sort_by(|a, b| {
            let a_builtin = a.kind == Some(CompletionItemKind::KEYWORD);
            let b_builtin = b.kind == Some(CompletionItemKind::KEYWORD);
            a_builtin.cmp(&b_builtin).then_with(|| a.label.cmp(&b.label))
        });

        items
    }

    fn hover(&self, content: &str, pos: Position) -> Option<HoverInfo> {
        let word = Self::word_at_cursor(content, pos)?;

        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("Python grammar");
        let tree = parser.parse(content, None)?;

        let mut symbols = Vec::new();
        Self::walk_node(tree.root_node(), content.as_bytes(), &mut symbols);

        // Find matching symbol
        let sym = symbols.iter().find(|s| s.name == word)?;
        let kind_str = match sym.kind {
            SymbolKind::Function => "function",
            SymbolKind::Class => "class",
            SymbolKind::Variable => "variable",
            SymbolKind::Import => "import",
            SymbolKind::Builtin => "builtin",
        };

        let content = match &sym.doc {
            Some(doc) => format!("({kind_str}) **{}**\n\n{doc}", sym.name),
            None => format!("({kind_str}) **{}** — line {}", sym.name, sym.line + 1),
        };

        Some(HoverInfo {
            content,
            range: None,
        })
    }
}

impl PythonLanguageService {
    /// Collect ERROR nodes from the tree-sitter parse tree.
    fn collect_errors(node: tree_sitter::Node, diags: &mut Vec<Diagnostic>) {
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            let end = node.end_position();
            diags.push(Diagnostic {
                range: Range::new(
                    Position::new(start.row as u32, start.column.min(u32::MAX as usize) as u32),
                    Position::new(end.row as u32, end.column.min(u32::MAX as usize) as u32),
                ),
                severity: Some(DiagnosticSeverity::ERROR),
                message: format!("syntax error: unexpected '{}'", node.kind()),
                ..Default::default()
            });
        }
        for i in 0..node.child_count() {
            Self::collect_errors(node.child(i as u32).unwrap(), diags);
        }
    }
}

/// Internal symbol representation.
struct Symbol {
    name: String,
    kind: SymbolKind,
    line: usize,
    doc: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SymbolKind {
    Function,
    Class,
    Variable,
    Import,
    Builtin,
}

/// Common Python builtins for completion.
const PYTHON_BUILTINS: &[&str] = &[
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "isinstance", "issubclass", "type", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "None", "True", "False",
    "open", "input", "sorted", "reversed", "min", "max", "sum", "abs",
    "any", "all", "hasattr", "getattr", "setattr", "delattr",
    "super", "property", "staticmethod", "classmethod",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "Exception", "StopIteration", "FileNotFoundError",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_symbols() {
        let mut svc = PythonLanguageService::new();
        let code = r#"
def hello(name):
    """Greet someone."""
    print(f"Hello, {name}!")

def calculate(x, y):
    return x + y
"#;
        let symbols: Vec<_> = svc
            .extract_symbols(code)
            .into_iter()
            .filter(|s| s.kind == SymbolKind::Function)
            .collect();

        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].name, "hello");
        assert!(symbols[0].doc.as_ref().unwrap().contains("Greet someone"));
        assert_eq!(symbols[1].name, "calculate");
    }

    #[test]
    fn extract_class_symbols() {
        let mut svc = PythonLanguageService::new();
        let code = r#"
class MyTool:
    """A custom tool."""
    def run(self):
        pass
"#;
        let symbols: Vec<_> = svc
            .extract_symbols(code)
            .into_iter()
            .filter(|s| s.kind == SymbolKind::Class)
            .collect();

        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "MyTool");
        assert!(symbols[0].doc.as_ref().unwrap().contains("custom tool"));
    }

    #[test]
    fn extract_variable_symbols() {
        let mut svc = PythonLanguageService::new();
        let code = r#"
MAX_RETRIES = 3
api_url = "https://example.com"
def foo():
    local_var = 1
"#;
        let symbols: Vec<_> = svc
            .extract_symbols(code)
            .into_iter()
            .filter(|s| s.kind == SymbolKind::Variable)
            .collect();

        // Only top-level variables, not local_var inside foo
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].name, "MAX_RETRIES");
        assert_eq!(symbols[1].name, "api_url");
    }

    #[test]
    fn completions_filter_by_prefix() {
        let svc = PythonLanguageService::new();
        let code = r#"
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b

result = calculate_sum(1, 2)
"#;
        let items = svc.completions(code, Position::new(7, 19)); // after "calc" on last line
        let names: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // Should NOT include exact match "calculate_sum" at current word
        // Should find both calculate_ functions if prefix matches
        assert!(names.contains(&"calculate_product") || names.contains(&"calculate_sum"));
    }

    #[test]
    fn diagnostics_on_valid_python() {
        let svc = PythonLanguageService::new();
        let code = "def hello():\n    pass\n";
        let diags = svc.diagnostics(code);
        assert!(diags.is_empty());
    }

    #[test]
    fn diagnostics_on_syntax_error() {
        let svc = PythonLanguageService::new();
        let code = "def hello(\n";
        let diags = svc.diagnostics(code);
        assert!(!diags.is_empty());
    }

    #[test]
    fn hover_on_function() {
        let svc = PythonLanguageService::new();
        let code = r#"
def greet(name):
    """Say hello."""
    print(name)

greet("world")
"#;
        let hover = svc.hover(code, Position::new(5, 5)); // "greet" on last line (cursor at end of word)
        assert!(hover.is_some());
        let info = hover.unwrap();
        assert!(info.content.contains("greet"));
        assert!(info.content.contains("Say hello"));
    }

    #[test]
    fn word_at_cursor_basic() {
        assert_eq!(
            PythonLanguageService::word_at_cursor("hello world", Position::new(0, 5)),
            Some("hello".to_string())
        );
        assert_eq!(
            PythonLanguageService::word_at_cursor("calculate_sum", Position::new(0, 4)),
            Some("calc".to_string())
        );
        assert_eq!(
            PythonLanguageService::word_at_cursor("x = foo(", Position::new(0, 0)),
            None
        );
    }
}
