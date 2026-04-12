//! Template variable expansion for trigger addresses and messages.
//!
//! Supports `{variable.name}` syntax in `send_to` addresses and `message` bodies.
//! Variables are expanded from a context map at fire time.
//!
//! # Example
//!
//! ```
//! use agentos_platform::template::expand;
//! use std::collections::HashMap;
//!
//! let mut vars = HashMap::new();
//! vars.insert("event.user_id".to_string(), "alice".to_string());
//! vars.insert("event.thread_id".to_string(), "thread-42".to_string());
//!
//! assert_eq!(
//!     expand("ringhub.concierge[{event.user_id}].public[{event.thread_id}]", &vars),
//!     "ringhub.concierge[alice].public[thread-42]"
//! );
//! ```

use std::collections::HashMap;

/// Expand `{variable.name}` placeholders in a template string.
///
/// Unknown variables are left as-is (not expanded, not removed).
/// This is intentional — it makes debugging easier when a variable name
/// is misspelled in the YAML.
pub fn expand(template: &str, vars: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            // Collect the variable name until '}'
            let mut var_name = String::new();
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == '}' {
                    found_close = true;
                    break;
                }
                var_name.push(inner);
            }

            if found_close {
                if let Some(value) = vars.get(&var_name) {
                    result.push_str(value);
                } else {
                    // Unknown variable — leave as-is for debugging
                    result.push('{');
                    result.push_str(&var_name);
                    result.push('}');
                }
            } else {
                // Unclosed brace — emit as literal
                result.push('{');
                result.push_str(&var_name);
            }
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn simple_expansion() {
        let v = vars(&[("event.user_id", "alice")]);
        assert_eq!(expand("user.{event.user_id}", &v), "user.alice");
    }

    #[test]
    fn multiple_variables() {
        let v = vars(&[
            ("event.user_id", "alice"),
            ("event.thread_id", "thread-42"),
        ]);
        assert_eq!(
            expand("ringhub.concierge[{event.user_id}].public[{event.thread_id}]", &v),
            "ringhub.concierge[alice].public[thread-42]"
        );
    }

    #[test]
    fn unknown_variable_preserved() {
        let v = vars(&[]);
        assert_eq!(
            expand("user.{event.unknown}", &v),
            "user.{event.unknown}"
        );
    }

    #[test]
    fn no_variables() {
        let v = vars(&[]);
        assert_eq!(expand("ringhub.concierge", &v), "ringhub.concierge");
    }

    #[test]
    fn empty_template() {
        let v = vars(&[("x", "y")]);
        assert_eq!(expand("", &v), "");
    }

    #[test]
    fn adjacent_variables() {
        let v = vars(&[("a", "hello"), ("b", "world")]);
        assert_eq!(expand("{a}{b}", &v), "helloworld");
    }

    #[test]
    fn unclosed_brace_literal() {
        let v = vars(&[]);
        assert_eq!(expand("hello {unclosed", &v), "hello {unclosed");
    }

    #[test]
    fn trigger_context_variables() {
        let v = vars(&[
            ("trigger.name", "morning_digest"),
            ("trigger.target", "bob"),
        ]);
        assert_eq!(
            expand("Trigger {trigger.name} fired for {trigger.target}", &v),
            "Trigger morning_digest fired for bob"
        );
    }
}
