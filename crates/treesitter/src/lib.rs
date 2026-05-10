//! Tree-sitter code indexing — symbol extraction and codebase mapping.
//!
//! Ported from ClaudeRLM. In-memory HashMap-backed (no SQLite).
//! Indexed files can become context segments for the librarian.

pub mod handler;
pub mod languages;
pub mod symbols;

use std::collections::HashMap;
use std::path::Path;

use languages::Lang;
use symbols::ExtractedSymbol;

/// Stats from indexing a directory.
#[derive(Debug, Default)]
pub struct IndexStats {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub total_symbols: usize,
}

/// An entry in the codebase map (file → symbol summary).
#[derive(Debug, Clone)]
pub struct FileMapEntry {
    pub path: String,
    pub language: String,
    pub symbols: Vec<SymbolSummary>,
}

/// Summary of a symbol for codebase maps (no line numbers, just name + kind).
#[derive(Debug, Clone)]
pub struct SymbolSummary {
    pub name: String,
    pub kind: String,
}

/// In-memory code index: file_path → extracted symbols.
pub struct CodeIndex {
    symbols: HashMap<String, Vec<ExtractedSymbol>>,
    languages: HashMap<String, String>, // path → language name
}

impl CodeIndex {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            languages: HashMap::new(),
        }
    }

    /// Index a single file. Returns the number of symbols found.
    pub fn index_file(&mut self, path: &Path) -> Result<usize, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| format!("no extension: {}", path.display()))?;

        let lang =
            Lang::from_extension(ext).ok_or_else(|| format!("unsupported language: .{ext}"))?;

        let source =
            std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;

        let extracted = symbols::extract_symbols(lang, &source)
            .map_err(|e| format!("parse error for {}: {e}", path.display()))?;

        let count = extracted.len();
        let path_str = path.to_string_lossy().to_string();
        self.languages
            .insert(path_str.clone(), lang.name().to_string());
        self.symbols.insert(path_str, extracted);
        Ok(count)
    }

    /// Index a source string directly (for testing / in-memory use).
    pub fn index_source(&mut self, path: &str, lang: Lang, source: &[u8]) -> Result<usize, String> {
        let extracted =
            symbols::extract_symbols(lang, source).map_err(|e| format!("parse error: {e}"))?;
        let count = extracted.len();
        self.languages
            .insert(path.to_string(), lang.name().to_string());
        self.symbols.insert(path.to_string(), extracted);
        Ok(count)
    }

    /// Index all supported files in a directory (non-recursive for now).
    pub fn index_directory(&mut self, dir: &Path) -> Result<IndexStats, String> {
        let mut stats = IndexStats::default();

        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("failed to read dir {}: {e}", dir.display()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            match self.index_file(&path) {
                Ok(count) => {
                    stats.files_indexed += 1;
                    stats.total_symbols += count;
                }
                Err(_) => {
                    stats.files_skipped += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Search for symbols by name (substring match) and optional kind filter.
    pub fn search(&self, query: &str, kind: Option<&str>) -> Vec<(&str, &ExtractedSymbol)> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for (path, syms) in &self.symbols {
            for sym in syms {
                let name_match = sym.name.to_lowercase().contains(&query_lower);
                let kind_match = kind.is_none_or(|k| sym.kind == k);
                if name_match && kind_match {
                    results.push((path.as_str(), sym));
                }
            }
        }

        results
    }

    /// Build a codebase map: per-file symbol summaries.
    pub fn codebase_map(&self) -> Vec<FileMapEntry> {
        let mut entries: Vec<FileMapEntry> = self
            .symbols
            .iter()
            .map(|(path, syms)| {
                let language = self
                    .languages
                    .get(path)
                    .cloned()
                    .unwrap_or_else(|| "unknown".into());
                let symbols = syms
                    .iter()
                    .map(|s| SymbolSummary {
                        name: s.name.clone(),
                        kind: s.kind.clone(),
                    })
                    .collect();
                FileMapEntry {
                    path: path.clone(),
                    language,
                    symbols,
                }
            })
            .collect();
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        entries
    }

    /// Get symbols for a specific file.
    pub fn get_file_symbols(&self, path: &str) -> Option<&[ExtractedSymbol]> {
        self.symbols.get(path).map(|v| v.as_slice())
    }

    /// Number of indexed files.
    pub fn file_count(&self) -> usize {
        self.symbols.len()
    }

    /// Total number of symbols across all files.
    pub fn symbol_count(&self) -> usize {
        self.symbols.values().map(|v| v.len()).sum()
    }
}

impl Default for CodeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_SOURCE: &[u8] = br#"
/// A sample struct.
pub struct Foo {
    pub bar: i32,
}

/// An enum for testing.
pub enum Color {
    Red,
    Green,
    Blue,
}

/// A function.
pub fn do_stuff(x: i32) -> i32 {
    x + 1
}

impl Foo {
    pub fn new(bar: i32) -> Self {
        Self { bar }
    }
}

trait Drawable {
    fn draw(&self);
}
"#;

    const PYTHON_SOURCE: &[u8] = br#"
class MyClass:
    def __init__(self, value):
        self.value = value

    def method(self):
        return self.value

def helper_function(x):
    return x * 2

CONSTANT = 42
"#;

    #[test]
    fn index_rust_source() {
        let mut idx = CodeIndex::new();
        let count = idx
            .index_source("test.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();
        assert!(count >= 4); // Foo, Color, do_stuff, impl Foo, Drawable

        let syms = idx.get_file_symbols("test.rs").unwrap();
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"Color"));
        assert!(names.contains(&"do_stuff"));
        assert!(names.contains(&"Drawable"));
    }

    #[test]
    fn index_python_source() {
        let mut idx = CodeIndex::new();
        let count = idx
            .index_source("test.py", Lang::Python, PYTHON_SOURCE)
            .unwrap();
        assert!(count >= 2); // MyClass, helper_function

        let syms = idx.get_file_symbols("test.py").unwrap();
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MyClass"));
        assert!(names.contains(&"helper_function"));
    }

    #[test]
    fn search_by_name() {
        let mut idx = CodeIndex::new();
        idx.index_source("test.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();

        let results = idx.search("foo", None);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(_, s)| s.name == "Foo"));
    }

    #[test]
    fn search_by_kind() {
        let mut idx = CodeIndex::new();
        idx.index_source("test.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();

        let functions = idx.search("", Some("function"));
        assert!(functions.iter().any(|(_, s)| s.name == "do_stuff"));
        // Foo is a struct, should not appear
        assert!(!functions.iter().any(|(_, s)| s.name == "Foo"));
    }

    #[test]
    fn codebase_map_entries() {
        let mut idx = CodeIndex::new();
        idx.index_source("src/main.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();
        idx.index_source("lib/util.py", Lang::Python, PYTHON_SOURCE)
            .unwrap();

        let map = idx.codebase_map();
        assert_eq!(map.len(), 2);
        // Sorted by path
        assert_eq!(map[0].path, "lib/util.py");
        assert_eq!(map[0].language, "python");
        assert_eq!(map[1].path, "src/main.rs");
        assert_eq!(map[1].language, "rust");
    }

    #[test]
    fn symbol_extraction_has_line_numbers() {
        let mut idx = CodeIndex::new();
        idx.index_source("test.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();

        let syms = idx.get_file_symbols("test.rs").unwrap();
        let foo = syms.iter().find(|s| s.name == "Foo").unwrap();
        assert!(foo.start_line > 0);
        assert!(foo.end_line >= foo.start_line);
    }

    #[test]
    fn symbol_has_signature() {
        let mut idx = CodeIndex::new();
        idx.index_source("test.rs", Lang::Rust, RUST_SOURCE)
            .unwrap();

        let syms = idx.get_file_symbols("test.rs").unwrap();
        let func = syms.iter().find(|s| s.name == "do_stuff").unwrap();
        assert!(func.signature.is_some());
        let sig = func.signature.as_ref().unwrap();
        assert!(sig.contains("do_stuff"));
    }

    #[test]
    fn empty_index() {
        let idx = CodeIndex::new();
        assert_eq!(idx.file_count(), 0);
        assert_eq!(idx.symbol_count(), 0);
        assert!(idx.search("foo", None).is_empty());
        assert!(idx.codebase_map().is_empty());
    }
}
