//! WIT subset parser — focused parser for tool interface definitions.
//!
//! Parses a minimal WIT subset sufficient for tool-peer schemas:
//! - `interface` with doc comments
//! - `record request { ... }` with typed fields
//! - Primitive types: string, bool, u32, u64, s32, s64, f32, f64
//! - Wrappers: `option<T>`, `list<T>`
//! - `func` declaration (parsed but not used beyond validation)

use super::{ToolField, ToolFieldType, ToolInterface, ToolRecord};

/// Parse a WIT interface definition from text.
///
/// Returns a `ToolInterface` with the interface name, description (from doc
/// comments), and request record (fields with types and descriptions).
pub fn parse_wit(input: &str) -> Result<ToolInterface, String> {
    let mut lines = input.lines().peekable();
    let mut interface_doc = Vec::new();

    // Collect leading doc comments (/// lines before `interface`)
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("///") {
            interface_doc.push(trimmed.strip_prefix("///").unwrap_or("").trim().to_string());
            lines.next();
        } else if trimmed.is_empty() {
            lines.next();
        } else {
            break;
        }
    }

    // Parse `interface <name> {`
    let interface_line = skip_blank(&mut lines)
        .ok_or_else(|| "expected 'interface <name> {', found EOF".to_string())?;
    let interface_trimmed = interface_line.trim();
    if !interface_trimmed.starts_with("interface ") {
        return Err(format!(
            "expected 'interface <name> {{', found: {interface_trimmed}"
        ));
    }
    let rest = interface_trimmed
        .strip_prefix("interface ")
        .unwrap()
        .trim();
    let name = rest
        .strip_suffix('{')
        .ok_or_else(|| format!("expected '{{' after interface name, found: {rest}"))?
        .trim()
        .to_string();

    if name.is_empty() {
        return Err("interface name cannot be empty".to_string());
    }

    let description = if interface_doc.is_empty() {
        name.clone()
    } else {
        interface_doc.join(" ")
    };

    // Parse body: expect `record request { ... }` and optionally a func line
    let mut record: Option<ToolRecord> = None;

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "}" {
            // End of interface or blank line
            if trimmed == "}" {
                break;
            }
            continue;
        }

        if trimmed.starts_with("record ") {
            record = Some(parse_record(trimmed, &mut lines)?);
        }
        // Skip func declarations and other lines inside the interface
    }

    let request = record.unwrap_or_else(|| ToolRecord {
        fields: Vec::new(),
    });

    Ok(ToolInterface {
        name,
        description,
        request,
    })
}

/// Parse a `record <name> { ... }` block.
fn parse_record(
    first_line: &str,
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Result<ToolRecord, String> {
    // first_line is like `record request {`
    let rest = first_line
        .strip_prefix("record ")
        .unwrap()
        .trim();
    if !rest.ends_with('{') {
        return Err(format!("expected '{{' after record name, found: {rest}"));
    }

    let mut fields = Vec::new();
    let mut field_doc = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim();

        if trimmed == "}" {
            break;
        }

        if trimmed.is_empty() {
            continue;
        }

        // Collect doc comments for the next field
        if trimmed.starts_with("///") {
            field_doc.push(trimmed.strip_prefix("///").unwrap_or("").trim().to_string());
            continue;
        }

        // Parse field: `name: type,` or `name: type`
        let field_str = trimmed.trim_end_matches(',');
        let (field_name, type_str) = field_str
            .split_once(':')
            .ok_or_else(|| format!("expected 'name: type' in record, found: {trimmed}"))?;

        let field_name = field_name.trim().to_string();
        let type_str = type_str.trim();
        let field_type = parse_type(type_str)?;

        let description = if field_doc.is_empty() {
            None
        } else {
            Some(field_doc.join(" "))
        };

        fields.push(ToolField {
            name: field_name,
            field_type,
            description,
        });

        field_doc.clear();
    }

    Ok(ToolRecord { fields })
}

/// Parse a WIT type string into a `ToolFieldType`.
fn parse_type(s: &str) -> Result<ToolFieldType, String> {
    let s = s.trim();
    match s {
        "string" => Ok(ToolFieldType::String),
        "bool" => Ok(ToolFieldType::Bool),
        "u32" => Ok(ToolFieldType::U32),
        "u64" => Ok(ToolFieldType::U64),
        "s32" => Ok(ToolFieldType::S32),
        "s64" => Ok(ToolFieldType::S64),
        "f32" => Ok(ToolFieldType::F32),
        "f64" => Ok(ToolFieldType::F64),
        _ if s.starts_with("option<") && s.ends_with('>') => {
            let inner = &s[7..s.len() - 1];
            let inner_type = parse_type(inner)?;
            Ok(ToolFieldType::Option(Box::new(inner_type)))
        }
        _ if s.starts_with("list<") && s.ends_with('>') => {
            let inner = &s[5..s.len() - 1];
            let inner_type = parse_type(inner)?;
            Ok(ToolFieldType::List(Box::new(inner_type)))
        }
        _ => Err(format!("unknown WIT type: {s}")),
    }
}

/// Skip blank lines and return the next non-blank line.
fn skip_blank(lines: &mut std::iter::Peekable<std::str::Lines<'_>>) -> Option<String> {
    for line in lines.by_ref() {
        if !line.trim().is_empty() {
            return Some(line.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_interface() {
        let wit = r#"
/// Read file contents.
interface file-read {
    record request {
        /// The file path
        path: string,
    }
    read: func(req: request) -> result<string, string>;
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(iface.name, "file-read");
        assert_eq!(iface.description, "Read file contents.");
        assert_eq!(iface.request.fields.len(), 1);
        assert_eq!(iface.request.fields[0].name, "path");
        assert_eq!(iface.request.fields[0].field_type, ToolFieldType::String);
        assert_eq!(
            iface.request.fields[0].description.as_deref(),
            Some("The file path")
        );
    }

    #[test]
    fn parse_multiple_fields() {
        let wit = r#"
/// Read file contents with optional offset/limit.
interface file-read {
    record request {
        /// The file path to read
        path: string,
        /// Starting line number (1-based, default: 1)
        offset: option<u32>,
        /// Maximum lines to read (default: 2000)
        limit: option<u32>,
    }
    read: func(req: request) -> result<string, string>;
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(iface.request.fields.len(), 3);

        assert_eq!(iface.request.fields[0].name, "path");
        assert_eq!(iface.request.fields[0].field_type, ToolFieldType::String);

        assert_eq!(iface.request.fields[1].name, "offset");
        assert_eq!(
            iface.request.fields[1].field_type,
            ToolFieldType::Option(Box::new(ToolFieldType::U32))
        );

        assert_eq!(iface.request.fields[2].name, "limit");
        assert_eq!(
            iface.request.fields[2].field_type,
            ToolFieldType::Option(Box::new(ToolFieldType::U32))
        );
    }

    #[test]
    fn parse_all_primitive_types() {
        let wit = r#"
interface all-types {
    record request {
        a: string,
        b: bool,
        c: u32,
        d: u64,
        e: s32,
        f: s64,
        g: f32,
        h: f64,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(iface.request.fields.len(), 8);
        assert_eq!(iface.request.fields[0].field_type, ToolFieldType::String);
        assert_eq!(iface.request.fields[1].field_type, ToolFieldType::Bool);
        assert_eq!(iface.request.fields[2].field_type, ToolFieldType::U32);
        assert_eq!(iface.request.fields[3].field_type, ToolFieldType::U64);
        assert_eq!(iface.request.fields[4].field_type, ToolFieldType::S32);
        assert_eq!(iface.request.fields[5].field_type, ToolFieldType::S64);
        assert_eq!(iface.request.fields[6].field_type, ToolFieldType::F32);
        assert_eq!(iface.request.fields[7].field_type, ToolFieldType::F64);
    }

    #[test]
    fn parse_option_type() {
        let wit = r#"
interface opt {
    record request {
        val: option<string>,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(
            iface.request.fields[0].field_type,
            ToolFieldType::Option(Box::new(ToolFieldType::String))
        );
    }

    #[test]
    fn parse_list_type() {
        let wit = r#"
interface listy {
    record request {
        items: list<string>,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(
            iface.request.fields[0].field_type,
            ToolFieldType::List(Box::new(ToolFieldType::String))
        );
    }

    #[test]
    fn parse_nested_option_list() {
        let wit = r#"
interface nested {
    record request {
        data: option<list<u32>>,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(
            iface.request.fields[0].field_type,
            ToolFieldType::Option(Box::new(ToolFieldType::List(Box::new(ToolFieldType::U32))))
        );
    }

    #[test]
    fn parse_no_doc_comments() {
        let wit = r#"
interface bare {
    record request {
        name: string,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(iface.name, "bare");
        assert_eq!(iface.description, "bare"); // defaults to name
        assert!(iface.request.fields[0].description.is_none());
    }

    #[test]
    fn parse_multiline_doc_comment() {
        let wit = r#"
/// First line of description.
/// Second line of description.
interface multi-doc {
    record request {
        x: string,
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(
            iface.description,
            "First line of description. Second line of description."
        );
    }

    #[test]
    fn parse_no_trailing_comma() {
        let wit = r#"
interface no-comma {
    record request {
        name: string
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert_eq!(iface.request.fields.len(), 1);
        assert_eq!(iface.request.fields[0].name, "name");
    }

    #[test]
    fn parse_empty_record() {
        let wit = r#"
interface empty {
    record request {
    }
}
"#;
        let iface = parse_wit(wit).unwrap();
        assert!(iface.request.fields.is_empty());
    }

    #[test]
    fn error_missing_interface() {
        let wit = "record request { name: string }";
        assert!(parse_wit(wit).is_err());
    }

    #[test]
    fn error_unknown_type() {
        let wit = r#"
interface bad {
    record request {
        x: unknown_type,
    }
}
"#;
        assert!(parse_wit(wit).is_err());
    }
}
