//! Language grammars and query patterns for tree-sitter.
//!
//! Ported from ClaudeRLM. Supports Rust and Python initially.

use tree_sitter::Language;

/// Supported languages with their tree-sitter grammars and symbol queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Python,
}

impl Lang {
    /// Detect language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            _ => None,
        }
    }

    /// Get the tree-sitter Language grammar.
    pub fn grammar(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
        }
    }

    /// Get the symbol extraction query for this language.
    pub fn symbol_query(&self) -> &'static str {
        match self {
            Self::Rust => RUST_QUERY,
            Self::Python => PYTHON_QUERY,
        }
    }

    /// Language name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
        }
    }
}

const RUST_QUERY: &str = r#"
; Functions
(function_item
  name: (identifier) @name
) @function

; Structs
(struct_item
  name: (type_identifier) @name
) @struct

; Enums
(enum_item
  name: (type_identifier) @name
) @enum

; Traits
(trait_item
  name: (type_identifier) @name
) @trait

; Impl blocks
(impl_item
  type: (_) @name
) @impl

; Type aliases
(type_item
  name: (type_identifier) @name
) @type_alias

; Constants
(const_item
  name: (identifier) @name
) @const

; Static items
(static_item
  name: (identifier) @name
) @static

; Macros
(macro_definition
  name: (identifier) @name
) @macro
"#;

const PYTHON_QUERY: &str = r#"
; Functions
(function_definition
  name: (identifier) @name
) @function

; Classes
(class_definition
  name: (identifier) @name
) @class

; Decorated functions
(decorated_definition
  definition: (function_definition
    name: (identifier) @name
  )
) @function

; Decorated classes
(decorated_definition
  definition: (class_definition
    name: (identifier) @name
  )
) @class
"#;
