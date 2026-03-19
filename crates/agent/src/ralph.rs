//! Ralph Method — task decomposition into atomic stories.
//!
//! Before executing, the coding agent asks Opus to decompose the task
//! into independently testable stories that each fit in one context window.

/// A single story in a decomposed task plan.
#[derive(Debug, Clone)]
pub struct Story {
    /// Story number (1-based).
    pub number: usize,
    /// One-line title.
    pub title: String,
    /// What this story accomplishes.
    pub goal: String,
    /// Files to read/modify.
    pub files: Vec<String>,
    /// How to verify this story is done.
    pub test: String,
}

/// A complete task plan — a sequence of stories.
#[derive(Debug, Clone)]
pub struct TaskPlan {
    /// Original task description.
    pub task: String,
    /// Ordered list of stories.
    pub stories: Vec<Story>,
}

/// Parse Opus's plan output into a structured TaskPlan.
///
/// Expects a numbered list with markdown-ish formatting:
/// ```text
/// 1. **Title**: ...
///    **Goal**: ...
///    **Files**: file1.rs, file2.rs
///    **Test**: ...
/// ```
pub fn parse_plan(task: &str, plan_text: &str) -> TaskPlan {
    let mut stories = Vec::new();
    let mut current: Option<StoryBuilder> = None;

    for line in plan_text.lines() {
        let trimmed = line.trim();

        // Check for numbered story start: "1. " or "1. **Title**:"
        if let Some(rest) = try_parse_story_start(trimmed) {
            // Save previous story if any
            if let Some(builder) = current.take() {
                stories.push(builder.build());
            }
            current = Some(StoryBuilder {
                number: stories.len() + 1,
                title: rest,
                goal: String::new(),
                files: Vec::new(),
                test: String::new(),
            });
            continue;
        }

        // Parse fields within a story
        if let Some(ref mut builder) = current {
            if let Some(value) = try_extract_field(trimmed, "Goal") {
                builder.goal = value;
            } else if let Some(value) = try_extract_field(trimmed, "Files") {
                builder.files = value
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            } else if let Some(value) = try_extract_field(trimmed, "Test") {
                builder.test = value;
            }
        }
    }

    // Don't forget the last story
    if let Some(builder) = current {
        stories.push(builder.build());
    }

    TaskPlan {
        task: task.to_string(),
        stories,
    }
}

struct StoryBuilder {
    number: usize,
    title: String,
    goal: String,
    files: Vec<String>,
    test: String,
}

impl StoryBuilder {
    fn build(self) -> Story {
        Story {
            number: self.number,
            title: self.title,
            goal: self.goal,
            files: self.files,
            test: self.test,
        }
    }
}

/// Try to parse a numbered story start like "1. Title here" or "1. **Title**: ..."
fn try_parse_story_start(line: &str) -> Option<String> {
    // Match "N. " or "N. **"
    let mut chars = line.chars();
    let first = chars.next()?;
    if !first.is_ascii_digit() {
        return None;
    }
    // Consume remaining digits
    let rest: String = chars.collect();
    let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());

    if !rest.starts_with(". ") {
        return None;
    }

    let after_dot = &rest[2..];
    // Strip optional **Title**: prefix
    let title = after_dot
        .trim_start_matches("**")
        .trim_start_matches("Title")
        .trim_start_matches("**")
        .trim_start_matches(':')
        .trim()
        .to_string();

    if title.is_empty() {
        Some(after_dot.trim().to_string())
    } else {
        Some(title)
    }
}

/// Try to extract a field like "**Goal**: value" or "- Goal: value"
fn try_extract_field(line: &str, field: &str) -> Option<String> {
    // Try "**Field**: value"
    let pattern1 = format!("**{field}**:");
    if let Some(idx) = line.find(&pattern1) {
        return Some(line[idx + pattern1.len()..].trim().to_string());
    }
    // Try "**Field:**: value" (common LLM output)
    let pattern2 = format!("**{field}:");
    if let Some(idx) = line.find(&pattern2) {
        let rest = &line[idx + pattern2.len()..];
        return Some(rest.trim_start_matches("**").trim().to_string());
    }
    // Try "- Field: value"
    let pattern3 = format!("- {field}:");
    if let Some(idx) = line.find(&pattern3) {
        return Some(line[idx + pattern3.len()..].trim().to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_plan() {
        let plan_text = r#"
1. **Title**: Set up project structure
   **Goal**: Create the module files
   **Files**: src/agent/mod.rs, src/agent/handler.rs
   **Test**: cargo check passes

2. **Title**: Implement handler
   **Goal**: Write the core handler logic
   **Files**: src/agent/handler.rs
   **Test**: cargo test agent passes
"#;

        let plan = parse_plan("Build the agent", plan_text);
        assert_eq!(plan.task, "Build the agent");
        assert_eq!(plan.stories.len(), 2);

        assert_eq!(plan.stories[0].number, 1);
        assert_eq!(plan.stories[0].title, "Set up project structure");
        assert_eq!(plan.stories[0].goal, "Create the module files");
        assert_eq!(plan.stories[0].files.len(), 2);
        assert!(plan.stories[0].test.contains("cargo check"));

        assert_eq!(plan.stories[1].number, 2);
        assert!(plan.stories[1].title.contains("Implement handler"));
    }

    #[test]
    fn parse_plan_empty() {
        let plan = parse_plan("task", "no numbered items here");
        assert!(plan.stories.is_empty());
    }

    #[test]
    fn parse_plan_minimal() {
        let plan_text = "1. Do the thing\n";
        let plan = parse_plan("task", plan_text);
        assert_eq!(plan.stories.len(), 1);
        assert_eq!(plan.stories[0].title, "Do the thing");
    }

    #[test]
    fn story_fields() {
        let story = Story {
            number: 1,
            title: "Test story".into(),
            goal: "Test goal".into(),
            files: vec!["a.rs".into()],
            test: "cargo test".into(),
        };
        assert_eq!(story.number, 1);
        assert_eq!(story.files.len(), 1);
    }

    #[test]
    fn task_plan_fields() {
        let plan = TaskPlan {
            task: "Build something".into(),
            stories: vec![],
        };
        assert_eq!(plan.task, "Build something");
        assert!(plan.stories.is_empty());
    }
}
