//! File operations on a VDrive — all sandboxed to the drive root.

use std::fs;

use crate::{VDrive, VDriveError, VDriveResult};

/// Metadata about a file or directory.
#[derive(Debug, Clone)]
pub struct EntryInfo {
    /// Path relative to the drive root.
    pub path: String,
    /// True if this is a directory.
    pub is_dir: bool,
    /// Size in bytes (0 for directories).
    pub size: u64,
}

/// A grep match result.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    /// Path relative to the drive root.
    pub path: String,
    /// 1-based line number.
    pub line_num: usize,
    /// The matching line content.
    pub line: String,
}

/// Result of a read operation.
#[derive(Debug)]
pub struct ReadResult {
    /// Line-numbered content.
    pub content: String,
    /// Total lines in the file.
    pub total_lines: usize,
    /// Lines returned in this read.
    pub lines_returned: usize,
}

impl VDrive {
    // ── Read ──

    /// Read a file with line numbers, optional offset and limit.
    pub fn read_file(&self, path: &str, offset: usize, limit: usize) -> VDriveResult<ReadResult> {
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            return Err(VDriveError::IsDirectory(path.to_string()));
        }

        let raw = fs::read(&resolved)?;
        if is_binary(&raw) {
            return Err(VDriveError::BinaryFile(path.to_string()));
        }

        let content = String::from_utf8_lossy(&raw);
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = if offset > 0 { offset - 1 } else { 0 };
        let end = (start + limit).min(total_lines);

        let mut output = String::new();
        for (i, line) in lines.iter().enumerate().skip(start).take(end - start) {
            let line_num = i + 1;
            if line.len() > 2000 {
                output.push_str(&format!("{line_num}| {}...\n", &line[..2000]));
            } else {
                output.push_str(&format!("{line_num}| {line}\n"));
            }
        }

        if end < total_lines {
            output.push_str(&format!(
                "\n... ({} more lines, {} total)",
                total_lines - end,
                total_lines
            ));
        }

        Ok(ReadResult {
            content: output,
            total_lines,
            lines_returned: end - start,
        })
    }

    // ── Write ──

    /// Write content to a file (creates or overwrites).
    /// Creates parent directories as needed.
    pub fn write_file(&self, path: &str, content: &str) -> VDriveResult<()> {
        let resolved = self.resolve_new(path)?;
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&resolved, content)?;
        Ok(())
    }

    // ── Edit ──

    /// Replace exact text in a file. The old_string must appear exactly once
    /// unless replace_all is true.
    pub fn edit_file(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> VDriveResult<()> {
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            return Err(VDriveError::IsDirectory(path.to_string()));
        }

        let content = fs::read_to_string(&resolved)?;
        let count = content.matches(old_string).count();

        if count == 0 {
            return Err(VDriveError::EditNotFound(path.to_string()));
        }
        if count > 1 && !replace_all {
            return Err(VDriveError::EditAmbiguous {
                path: path.to_string(),
                count,
            });
        }

        let new_content = content.replace(old_string, new_string);
        fs::write(&resolved, new_content)?;
        Ok(())
    }

    // ── Glob ──

    /// Find files matching a glob pattern within the drive.
    pub fn glob(&self, pattern: &str) -> VDriveResult<Vec<String>> {
        let full_pattern = self.root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy().to_string();

        let entries = glob::glob(&pattern_str).map_err(|e| {
            VDriveError::InvalidPattern(format!("{e}"))
        })?;

        let mut results = Vec::new();
        for entry in entries {
            let path = entry.map_err(|e| VDriveError::Io(e.into_error()))?;
            // Verify the match is within root (paranoid check)
            if let Ok(canonical) = path.canonicalize() {
                if canonical.starts_with(&self.root) {
                    results.push(self.relative(&canonical));
                }
            }
        }

        results.sort();
        Ok(results)
    }

    // ── Grep ──

    /// Search file contents with a regex pattern.
    /// Searches all files matching the optional file_glob, or all files if None.
    pub fn grep(
        &self,
        pattern: &str,
        file_glob: Option<&str>,
        max_results: usize,
    ) -> VDriveResult<Vec<GrepMatch>> {
        let re = regex::Regex::new(pattern).map_err(|e| {
            VDriveError::InvalidRegex(format!("{e}"))
        })?;

        let files = self.glob(file_glob.unwrap_or("**/*"))?;
        let mut matches = Vec::new();

        for file_path in &files {
            let resolved = match self.resolve(file_path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if resolved.is_dir() {
                continue;
            }

            let content = match fs::read(&resolved) {
                Ok(raw) if !is_binary(&raw) => String::from_utf8_lossy(&raw).into_owned(),
                _ => continue,
            };

            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    matches.push(GrepMatch {
                        path: file_path.clone(),
                        line_num: i + 1,
                        line: if line.len() > 500 {
                            format!("{}...", &line[..500])
                        } else {
                            line.to_string()
                        },
                    });
                    if matches.len() >= max_results {
                        return Ok(matches);
                    }
                }
            }
        }

        Ok(matches)
    }

    // ── Stat ──

    /// Get metadata for a file or directory.
    pub fn stat(&self, path: &str) -> VDriveResult<EntryInfo> {
        let resolved = self.resolve(path)?;
        let meta = fs::metadata(&resolved)?;
        Ok(EntryInfo {
            path: self.relative(&resolved),
            is_dir: meta.is_dir(),
            size: meta.len(),
        })
    }

    // ── List directory ──

    /// List entries in a directory.
    pub fn list_dir(&self, path: &str) -> VDriveResult<Vec<EntryInfo>> {
        let resolved = self.resolve(path)?;
        if !resolved.is_dir() {
            return Err(VDriveError::NotDirectory(path.to_string()));
        }

        let mut entries = Vec::new();
        for entry in fs::read_dir(&resolved)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let abs_path = entry.path();
            // Verify within root (symlink protection)
            if let Ok(canonical) = abs_path.canonicalize() {
                if canonical.starts_with(&self.root) {
                    entries.push(EntryInfo {
                        path: self.relative(&canonical),
                        is_dir: meta.is_dir(),
                        size: meta.len(),
                    });
                }
            }
        }

        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }

    // ── Mkdir ──

    /// Create a directory (and parents) within the drive.
    pub fn mkdir(&self, path: &str) -> VDriveResult<()> {
        let resolved = self.resolve_new(path)?;
        fs::create_dir_all(&resolved)?;
        Ok(())
    }

    // ── Delete ──

    /// Delete a file within the drive.
    pub fn delete_file(&self, path: &str) -> VDriveResult<()> {
        let resolved = self.resolve(path)?;
        if resolved.is_dir() {
            return Err(VDriveError::IsDirectory(path.to_string()));
        }
        fs::remove_file(&resolved)?;
        Ok(())
    }

    /// Delete a directory (must be empty) within the drive.
    pub fn delete_dir(&self, path: &str) -> VDriveResult<()> {
        let resolved = self.resolve(path)?;
        if !resolved.is_dir() {
            return Err(VDriveError::NotDirectory(path.to_string()));
        }
        // Never allow deleting the root itself
        if resolved == self.root {
            return Err(VDriveError::Escape("cannot delete drive root".to_string()));
        }
        fs::remove_dir(&resolved)?;
        Ok(())
    }

    // ── Diff ──

    /// Compute a unified diff between two files within the drive.
    pub fn diff(&self, path_a: &str, path_b: &str) -> VDriveResult<String> {
        let a_resolved = self.resolve(path_a)?;
        let b_resolved = self.resolve(path_b)?;

        let a_content = fs::read_to_string(&a_resolved)?;
        let b_content = fs::read_to_string(&b_resolved)?;

        let diff = similar::TextDiff::from_lines(&a_content, &b_content);
        let mut output = String::new();
        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                similar::ChangeTag::Delete => "-",
                similar::ChangeTag::Insert => "+",
                similar::ChangeTag::Equal => " ",
            };
            output.push_str(sign);
            output.push_str(change.value());
            if !change.value().ends_with('\n') {
                output.push('\n');
            }
        }
        Ok(output)
    }

    // ── Exists ──

    /// Check if a path exists within the drive (without error on not-found).
    pub fn exists(&self, path: &str) -> bool {
        self.resolve(path).is_ok()
    }
}

/// Check if a byte slice looks like binary (null bytes in first 8KB).
fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    data[..check_len].contains(&0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup() -> (TempDir, VDrive) {
        let dir = TempDir::new().unwrap();
        let vd = VDrive::open(dir.path()).unwrap();
        (dir, vd)
    }

    // ── Read tests ──

    #[test]
    fn read_basic_file() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("hello.txt"), "line one\nline two\nline three\n").unwrap();
        let result = vd.read_file("hello.txt", 1, 2000).unwrap();
        assert!(result.content.contains("1| line one"));
        assert!(result.content.contains("2| line two"));
        assert_eq!(result.total_lines, 3);
    }

    #[test]
    fn read_with_offset_and_limit() {
        let (dir, vd) = setup();
        let content: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        fs::write(dir.path().join("big.txt"), content).unwrap();

        let result = vd.read_file("big.txt", 5, 3).unwrap();
        assert!(result.content.contains("5| line 5"));
        assert!(result.content.contains("7| line 7"));
        assert!(!result.content.contains("8| line 8"));
        assert_eq!(result.lines_returned, 3);
    }

    #[test]
    fn read_nonexistent_file() {
        let (_dir, vd) = setup();
        let result = vd.read_file("nope.txt", 1, 100);
        assert!(matches!(result, Err(VDriveError::NotFound(_))));
    }

    #[test]
    fn read_directory_rejected() {
        let (dir, vd) = setup();
        fs::create_dir(dir.path().join("subdir")).unwrap();
        let result = vd.read_file("subdir", 1, 100);
        assert!(matches!(result, Err(VDriveError::IsDirectory(_))));
    }

    #[test]
    fn read_binary_rejected() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("bin"), &[0x00, 0x01, 0xFF]).unwrap();
        let result = vd.read_file("bin", 1, 100);
        assert!(matches!(result, Err(VDriveError::BinaryFile(_))));
    }

    #[test]
    fn read_escape_blocked() {
        let (_dir, vd) = setup();
        let result = vd.read_file("../../etc/passwd", 1, 100);
        assert!(result.is_err());
    }

    // ── Write tests ──

    #[test]
    fn write_new_file() {
        let (_dir, vd) = setup();
        vd.write_file("output.txt", "hello world").unwrap();
        let result = vd.read_file("output.txt", 1, 100).unwrap();
        assert!(result.content.contains("hello world"));
    }

    #[test]
    fn write_creates_parents() {
        let (_dir, vd) = setup();
        vd.write_file("deep/nested/dir/file.rs", "fn main() {}").unwrap();
        let result = vd.read_file("deep/nested/dir/file.rs", 1, 100).unwrap();
        assert!(result.content.contains("fn main()"));
    }

    #[test]
    fn write_overwrites_existing() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "old").unwrap();
        vd.write_file("f.txt", "new").unwrap();
        let result = vd.read_file("f.txt", 1, 100).unwrap();
        assert!(result.content.contains("new"));
    }

    #[test]
    fn write_escape_blocked() {
        let (_dir, vd) = setup();
        let result = vd.write_file("../../escape.txt", "bad");
        assert!(result.is_err());
    }

    // ── Edit tests ──

    #[test]
    fn edit_single_replacement() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "fn foo() {}\nfn bar() {}\n").unwrap();
        vd.edit_file("code.rs", "fn foo()", "fn baz()", false).unwrap();
        let result = vd.read_file("code.rs", 1, 100).unwrap();
        assert!(result.content.contains("fn baz()"));
        assert!(!result.content.contains("fn foo()"));
    }

    #[test]
    fn edit_not_found_error() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        let result = vd.edit_file("code.rs", "nonexistent_text", "replacement", false);
        assert!(matches!(result, Err(VDriveError::EditNotFound(_))));
    }

    #[test]
    fn edit_ambiguous_error() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "let x = 1;\nlet x = 2;\n").unwrap();
        let result = vd.edit_file("code.rs", "let x", "let y", false);
        assert!(matches!(result, Err(VDriveError::EditAmbiguous { count: 2, .. })));
    }

    #[test]
    fn edit_replace_all() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "let x = 1;\nlet x = 2;\n").unwrap();
        vd.edit_file("code.rs", "let x", "let y", true).unwrap();
        let result = vd.read_file("code.rs", 1, 100).unwrap();
        assert!(!result.content.contains("let x"));
        assert!(result.content.contains("let y = 1"));
        assert!(result.content.contains("let y = 2"));
    }

    // ── Glob tests ──

    #[test]
    fn glob_finds_files() {
        let (dir, vd) = setup();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        fs::write(dir.path().join("README.md"), "").unwrap();

        let results = vd.glob("src/*.rs").unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.ends_with("main.rs")));
        assert!(results.iter().any(|r| r.ends_with("lib.rs")));
    }

    #[test]
    fn glob_recursive() {
        let (dir, vd) = setup();
        fs::create_dir_all(dir.path().join("a/b/c")).unwrap();
        fs::write(dir.path().join("a/one.txt"), "").unwrap();
        fs::write(dir.path().join("a/b/two.txt"), "").unwrap();
        fs::write(dir.path().join("a/b/c/three.txt"), "").unwrap();

        let results = vd.glob("**/*.txt").unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn glob_no_matches() {
        let (_dir, vd) = setup();
        let results = vd.glob("*.xyz").unwrap();
        assert!(results.is_empty());
    }

    // ── Grep tests ──

    #[test]
    fn grep_finds_matches() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("a.txt"), "hello world\ngoodbye world\nhello again\n").unwrap();
        fs::write(dir.path().join("b.txt"), "no match here\n").unwrap();

        let matches = vd.grep("hello", None, 100).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.path == "a.txt"));
        assert_eq!(matches[0].line_num, 1);
        assert_eq!(matches[1].line_num, 3);
    }

    #[test]
    fn grep_with_file_glob() {
        let (dir, vd) = setup();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("src/lib.rs"), "fn lib() {}\n").unwrap();
        fs::write(dir.path().join("README.md"), "fn not_code() {}\n").unwrap();

        let matches = vd.grep("fn ", Some("src/*.rs"), 100).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.path.starts_with("src/")));
    }

    #[test]
    fn grep_regex() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("code.rs"), "let x = 42;\nlet y = \"hello\";\nlet z = 99;\n").unwrap();

        let matches = vd.grep(r"let \w+ = \d+", None, 100).unwrap();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn grep_max_results() {
        let (dir, vd) = setup();
        let content: String = (1..=100).map(|i| format!("match line {i}\n")).collect();
        fs::write(dir.path().join("big.txt"), content).unwrap();

        let matches = vd.grep("match", None, 5).unwrap();
        assert_eq!(matches.len(), 5);
    }

    #[test]
    fn grep_invalid_regex() {
        let (_dir, vd) = setup();
        let result = vd.grep("[invalid", None, 100);
        assert!(matches!(result, Err(VDriveError::InvalidRegex(_))));
    }

    // ── Stat tests ──

    #[test]
    fn stat_file() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "12345").unwrap();
        let info = vd.stat("f.txt").unwrap();
        assert!(!info.is_dir);
        assert_eq!(info.size, 5);
    }

    #[test]
    fn stat_dir() {
        let (dir, vd) = setup();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let info = vd.stat("sub").unwrap();
        assert!(info.is_dir);
    }

    // ── List dir tests ──

    #[test]
    fn list_dir_contents() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();

        let entries = vd.list_dir(".").unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn list_dir_on_file_errors() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "").unwrap();
        let result = vd.list_dir("f.txt");
        assert!(matches!(result, Err(VDriveError::NotDirectory(_))));
    }

    // ── Mkdir tests ──

    #[test]
    fn mkdir_creates_dirs() {
        let (_dir, vd) = setup();
        vd.mkdir("a/b/c").unwrap();
        let info = vd.stat("a/b/c").unwrap();
        assert!(info.is_dir);
    }

    // ── Delete tests ──

    #[test]
    fn delete_file_works() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "bye").unwrap();
        vd.delete_file("f.txt").unwrap();
        assert!(!vd.exists("f.txt"));
    }

    #[test]
    fn delete_nonexistent_errors() {
        let (_dir, vd) = setup();
        let result = vd.delete_file("nope.txt");
        assert!(result.is_err());
    }

    #[test]
    fn delete_dir_on_file_errors() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "").unwrap();
        let result = vd.delete_dir("f.txt");
        assert!(matches!(result, Err(VDriveError::NotDirectory(_))));
    }

    #[test]
    fn delete_root_blocked() {
        let (_dir, vd) = setup();
        let result = vd.delete_dir(".");
        assert!(result.is_err());
    }

    // ── Diff tests ──

    #[test]
    fn diff_shows_changes() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("a.txt"), "line 1\nline 2\nline 3\n").unwrap();
        fs::write(dir.path().join("b.txt"), "line 1\nchanged\nline 3\n").unwrap();

        let diff = vd.diff("a.txt", "b.txt").unwrap();
        assert!(diff.contains("-line 2"));
        assert!(diff.contains("+changed"));
    }

    // ── Exists tests ──

    #[test]
    fn exists_true_for_real_file() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("f.txt"), "").unwrap();
        assert!(vd.exists("f.txt"));
    }

    #[test]
    fn exists_false_for_missing() {
        let (_dir, vd) = setup();
        assert!(!vd.exists("nope.txt"));
    }

    #[test]
    fn exists_false_for_escape() {
        let (_dir, vd) = setup();
        assert!(!vd.exists("../../etc/passwd"));
    }
}
