//! Hierarchical agent addressing.
//!
//! Addresses identify agent instances, their buffers, and their cache compositions.
//! The grammar supports parameterized instances (`bob[alice]`), nested buffers
//! (`bob[alice].calendar`), and cache composition (`bob[main+alice]`).
//!
//! # Grammar
//!
//! ```text
//! address     = segment ("." segment)*
//! segment     = name key?
//! name        = [a-zA-Z0-9_-]+
//! key         = "[" key_inner "]"
//! key_inner   = cache_list | simple_key
//! cache_list  = simple_key ("+" simple_key)+
//! simple_key  = [^[\]]+
//! ```
//!
//! # Reserved Characters
//!
//! - `.` separates segments (dot-separated path)
//! - `[` `]` delimit instance keys
//! - `+` separates cache composition entries within a key
//!
//! # Examples
//!
//! ```
//! use agentos_platform::address::Address;
//!
//! // Simple organism (singleton, backward-compatible)
//! let addr = Address::parse("bob").unwrap();
//! assert_eq!(addr.segments().len(), 1);
//! assert_eq!(addr.segments()[0].name(), "bob");
//! assert_eq!(addr.segments()[0].key(), None);
//!
//! // Parameterized instance
//! let addr = Address::parse("bob[alice]").unwrap();
//! assert_eq!(addr.segments()[0].key(), Some("alice"));
//!
//! // Full path: namespace.organism[key].buffer
//! let addr = Address::parse("ringhub.bob[alice].calendar").unwrap();
//! assert_eq!(addr.segments().len(), 3);
//!
//! // Cache composition
//! let addr = Address::parse("bob[main+alice]").unwrap();
//! assert_eq!(addr.segments()[0].cache_keys(), vec!["main", "alice"]);
//! ```

use std::fmt;

/// A parsed hierarchical agent address.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Address {
    segments: Vec<Segment>,
    raw: String,
}

/// One segment of an address path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Segment {
    name: String,
    key: Option<String>,
}

/// Errors from address parsing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AddressError {
    #[error("empty address")]
    Empty,

    #[error("empty segment name")]
    EmptyName,

    #[error("unclosed bracket in '{0}'")]
    UnclosedBracket(String),

    #[error("unexpected closing bracket in '{0}'")]
    UnexpectedClose(String),

    #[error("invalid character in segment name: '{0}'")]
    InvalidChar(char),
}

impl Address {
    /// Parse an address string into structured segments.
    pub fn parse(input: &str) -> Result<Self, AddressError> {
        let input = input.trim();
        if input.is_empty() {
            return Err(AddressError::Empty);
        }

        let raw = input.to_string();
        let segments = split_segments(input)?;

        if segments.is_empty() {
            return Err(AddressError::Empty);
        }

        Ok(Self { segments, raw })
    }

    /// The parsed segments of this address.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// The raw address string.
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Index of the organism segment.
    ///
    /// The organism is the first segment that has a key (bracket parameter).
    /// If no segment has a key, the first segment is the organism.
    /// Everything before the organism is namespace; everything after is buffer.
    fn organism_index(&self) -> usize {
        self.segments
            .iter()
            .position(|s| s.key.is_some())
            .unwrap_or(0)
    }

    /// The organism name.
    /// For `bob[alice]` → "bob". For `ringhub.bob[alice]` → "bob".
    pub fn organism(&self) -> &str {
        &self.segments[self.organism_index()].name
    }

    /// The namespace prefix, if present (segments before the organism).
    /// For `ringhub.bob[alice]` → Some("ringhub"). For `bob[alice]` → None.
    pub fn namespace(&self) -> Option<&str> {
        let idx = self.organism_index();
        if idx > 0 {
            Some(&self.segments[0].name)
        } else {
            None
        }
    }

    /// The instance key from the organism segment, if present.
    pub fn instance_key(&self) -> Option<&str> {
        self.segments.get(self.organism_index()).and_then(|s| s.key())
    }

    /// The cache composition keys, if the instance key contains `+`.
    pub fn cache_keys(&self) -> Vec<&str> {
        self.segments
            .get(self.organism_index())
            .map(|s| s.cache_keys())
            .unwrap_or_default()
    }

    /// The buffer segment (first segment after the organism), if present.
    /// For `bob[alice].dm` → Some(Segment("dm")). For `bob[alice]` → None.
    pub fn buffer(&self) -> Option<&Segment> {
        let org_idx = self.organism_index();
        if self.segments.len() > org_idx + 1 {
            Some(&self.segments[org_idx + 1])
        } else {
            None
        }
    }

    /// Returns the instance-level address (namespace + organism[key], without buffer).
    ///
    /// For `concierge[alice].dm` → `concierge[alice]`
    /// For `ringhub.concierge[alice].help[email]` → `ringhub.concierge[alice]`
    /// For `concierge[alice]` → `concierge[alice]` (unchanged)
    pub fn instance_address(&self) -> Address {
        let org_idx = self.organism_index();
        let instance_segments: Vec<Segment> = self.segments[..=org_idx].to_vec();
        let raw = instance_segments
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(".");
        Address {
            segments: instance_segments,
            raw,
        }
    }

    /// Whether this address refers to the ephemeral scratch namespace.
    pub fn is_ephemeral(&self) -> bool {
        self.segments.iter().any(|s| s.name == "scratch")
    }
}

impl Segment {
    /// The segment name (before the brackets).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The raw key string inside brackets, if present.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Cache composition keys (split on `+`). Returns a single-element vec
    /// if no `+` is present, or empty if no key at all.
    pub fn cache_keys(&self) -> Vec<&str> {
        match &self.key {
            Some(k) => k.split('+').collect(),
            None => vec![],
        }
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(ref key) = self.key {
            write!(f, "[{key}]")?;
        }
        Ok(())
    }
}

/// Split an address string into segments, respecting brackets.
fn split_segments(input: &str) -> Result<Vec<Segment>, AddressError> {
    let mut segments = Vec::new();
    let mut current_name = String::new();
    let mut current_key: Option<String> = None;
    let mut in_bracket = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '[' => {
                if in_bracket {
                    return Err(AddressError::UnclosedBracket(input.to_string()));
                }
                in_bracket = true;
                current_key = Some(String::new());
            }
            ']' => {
                if !in_bracket {
                    return Err(AddressError::UnexpectedClose(input.to_string()));
                }
                in_bracket = false;
            }
            '.' if !in_bracket => {
                // Segment boundary
                if current_name.is_empty() {
                    return Err(AddressError::EmptyName);
                }
                segments.push(Segment {
                    name: std::mem::take(&mut current_name),
                    key: current_key.take(),
                });
            }
            _ => {
                if in_bracket {
                    current_key.as_mut().unwrap().push(ch);
                } else {
                    current_name.push(ch);
                }
            }
        }
    }

    if in_bracket {
        return Err(AddressError::UnclosedBracket(input.to_string()));
    }

    // Push final segment
    if !current_name.is_empty() {
        segments.push(Segment {
            name: current_name,
            key: current_key,
        });
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_organism() {
        let addr = Address::parse("bob").unwrap();
        assert_eq!(addr.segments().len(), 1);
        assert_eq!(addr.segments()[0].name(), "bob");
        assert_eq!(addr.segments()[0].key(), None);
        assert_eq!(addr.organism(), "bob");
        assert_eq!(addr.namespace(), None);
        assert_eq!(addr.instance_key(), None);
        assert_eq!(addr.buffer(), None);
    }

    #[test]
    fn parameterized_instance() {
        let addr = Address::parse("bob[alice]").unwrap();
        assert_eq!(addr.segments().len(), 1);
        assert_eq!(addr.organism(), "bob");
        assert_eq!(addr.instance_key(), Some("alice"));
    }

    #[test]
    fn namespaced_instance() {
        let addr = Address::parse("ringhub.bob[alice]").unwrap();
        assert_eq!(addr.segments().len(), 2);
        assert_eq!(addr.namespace(), Some("ringhub"));
        assert_eq!(addr.organism(), "bob");
        assert_eq!(addr.instance_key(), Some("alice"));
    }

    #[test]
    fn full_path_with_buffer() {
        let addr = Address::parse("ringhub.bob[alice].calendar").unwrap();
        assert_eq!(addr.segments().len(), 3);
        assert_eq!(addr.namespace(), Some("ringhub"));
        assert_eq!(addr.organism(), "bob");
        assert_eq!(addr.instance_key(), Some("alice"));
        let buf = addr.buffer().unwrap();
        assert_eq!(buf.name(), "calendar");
    }

    #[test]
    fn cache_composition() {
        let addr = Address::parse("bob[main+alice]").unwrap();
        assert_eq!(addr.cache_keys(), vec!["main", "alice"]);
    }

    #[test]
    fn cache_composition_triple() {
        let addr = Address::parse("bob[main+events+alice]").unwrap();
        assert_eq!(addr.cache_keys(), vec!["main", "events", "alice"]);
    }

    #[test]
    fn buffer_with_subkey() {
        let addr = Address::parse("ringhub.bob[alice].help[email-issue]").unwrap();
        assert_eq!(addr.segments().len(), 3);
        let buf = addr.buffer().unwrap();
        assert_eq!(buf.name(), "help");
        assert_eq!(buf.key(), Some("email-issue"));
    }

    #[test]
    fn ephemeral_scratch() {
        let addr = Address::parse("ringhub.scratch.bob[query-123]").unwrap();
        assert!(addr.is_ephemeral());
    }

    #[test]
    fn not_ephemeral() {
        let addr = Address::parse("ringhub.bob[alice]").unwrap();
        assert!(!addr.is_ephemeral());
    }

    #[test]
    fn display_roundtrip() {
        let cases = [
            "bob",
            "bob[alice]",
            "ringhub.bob[alice]",
            "ringhub.bob[alice].calendar",
            "bob[main+alice]",
        ];
        for case in cases {
            let addr = Address::parse(case).unwrap();
            assert_eq!(addr.to_string(), case);
        }
    }

    #[test]
    fn error_empty() {
        assert_eq!(Address::parse(""), Err(AddressError::Empty));
        assert_eq!(Address::parse("  "), Err(AddressError::Empty));
    }

    #[test]
    fn error_unclosed_bracket() {
        assert!(matches!(
            Address::parse("bob[alice"),
            Err(AddressError::UnclosedBracket(_))
        ));
    }

    #[test]
    fn error_unexpected_close() {
        assert!(matches!(
            Address::parse("bob]"),
            Err(AddressError::UnexpectedClose(_))
        ));
    }

    #[test]
    fn error_empty_name() {
        assert_eq!(Address::parse(".bob"), Err(AddressError::EmptyName));
    }

    #[test]
    fn key_with_dots() {
        // Dots inside brackets are part of the key, not segment separators
        let addr = Address::parse("bob[user.alice.session-1]").unwrap();
        assert_eq!(addr.instance_key(), Some("user.alice.session-1"));
    }

    #[test]
    fn key_with_dashes_and_numbers() {
        let addr = Address::parse("event-coordinator[winter-2027:user-alice]").unwrap();
        assert_eq!(addr.organism(), "event-coordinator");
        assert_eq!(addr.instance_key(), Some("winter-2027:user-alice"));
    }
}
