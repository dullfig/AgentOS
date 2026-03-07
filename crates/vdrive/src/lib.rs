//! VDrive — sandboxed folder-backed virtual drive.
//!
//! Exposes a real directory as a contained workspace. Every path operation
//! is resolved and validated against the root — no escape via `..`, symlinks,
//! or any other trick. Safe to expose directly to WASM tool guests.
//!
//! The agent clones a repo, points a VDrive at it, and all file tools
//! operate through the drive. Real files on disk, real git, real compilers —
//! but structurally contained.

mod ops;

use std::path::{Path, PathBuf};

pub use ops::*;

/// Errors from VDrive operations.
#[derive(Debug, thiserror::Error)]
pub enum VDriveError {
    #[error("path escapes workspace: {0}")]
    Escape(String),

    #[error("path not found: {0}")]
    NotFound(String),

    #[error("path is a directory: {0}")]
    IsDirectory(String),

    #[error("path is not a directory: {0}")]
    NotDirectory(String),

    #[error("file already exists: {0}")]
    AlreadyExists(String),

    #[error("binary file: {0}")]
    BinaryFile(String),

    #[error("invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("invalid regex: {0}")]
    InvalidRegex(String),

    #[error("edit failed: old_string not found in {0}")]
    EditNotFound(String),

    #[error("edit failed: old_string matches {count} times in {path} (must be unique or use replace_all)")]
    EditAmbiguous { path: String, count: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type VDriveResult<T> = Result<T, VDriveError>;

/// A sandboxed view of a real directory on disk.
///
/// All path operations are resolved against `root`. Any attempt to
/// escape (via `..`, symlinks, etc.) is rejected before I/O occurs.
#[derive(Debug, Clone)]
pub struct VDrive {
    /// Canonical root path. All resolved paths must start with this.
    root: PathBuf,
    /// Human-friendly name (basename of the root directory).
    name: String,
}

impl VDrive {
    /// Open an existing directory as a VDrive.
    ///
    /// The directory must exist. The path is canonicalized at construction
    /// time so symlinked roots are handled correctly. The name defaults to
    /// the folder's basename (e.g., `my-project` for `/home/user/my-project`).
    pub fn open(root: &Path) -> VDriveResult<Self> {
        let root = root.canonicalize().map_err(|_| {
            VDriveError::NotFound(root.display().to_string())
        })?;
        if !root.is_dir() {
            return Err(VDriveError::NotDirectory(root.display().to_string()));
        }
        let name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| root.display().to_string());
        Ok(Self { root, name })
    }

    /// Create a new directory and open it as a VDrive.
    pub fn create(root: &Path) -> VDriveResult<Self> {
        std::fs::create_dir_all(root)?;
        Self::open(root)
    }

    /// The canonical root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Human-friendly workspace name (folder basename).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Resolve a user-provided path to an absolute path within the drive.
    ///
    /// - Relative paths are joined to root
    /// - Absolute paths must be under root
    /// - `..` segments are resolved via canonicalization
    /// - Symlinks that escape are rejected
    ///
    /// For paths to files that don't exist yet (e.g., file-write target),
    /// use `resolve_parent()` which validates the parent directory exists
    /// and the final component doesn't escape.
    pub fn resolve(&self, user_path: &str) -> VDriveResult<PathBuf> {
        let candidate = if Path::new(user_path).is_absolute() {
            PathBuf::from(user_path)
        } else {
            self.root.join(user_path)
        };

        let resolved = candidate.canonicalize().map_err(|_| {
            VDriveError::NotFound(user_path.to_string())
        })?;

        if !resolved.starts_with(&self.root) {
            return Err(VDriveError::Escape(user_path.to_string()));
        }

        Ok(resolved)
    }

    /// Resolve a path where the target (and parents) may not exist yet.
    ///
    /// Walks up to the highest existing ancestor, canonicalizes it,
    /// verifies it's within root, then appends the remaining segments.
    /// Used for write/mkdir operations that create intermediate dirs.
    pub fn resolve_new(&self, user_path: &str) -> VDriveResult<PathBuf> {
        let candidate = if Path::new(user_path).is_absolute() {
            PathBuf::from(user_path)
        } else {
            self.root.join(user_path)
        };

        // Reject any path component that is ".."
        for component in candidate.components() {
            if let std::path::Component::ParentDir = component {
                return Err(VDriveError::Escape(user_path.to_string()));
            }
        }

        // Walk up to find the highest existing ancestor
        let mut existing = candidate.as_path();
        let mut tail_parts = Vec::new();
        loop {
            if existing.exists() {
                break;
            }
            match (existing.parent(), existing.file_name()) {
                (Some(parent), Some(name)) => {
                    tail_parts.push(name.to_os_string());
                    existing = parent;
                }
                _ => return Err(VDriveError::Escape(user_path.to_string())),
            }
        }

        // Canonicalize the existing portion and verify it's in root
        let resolved_base = existing.canonicalize().map_err(|_| {
            VDriveError::NotFound(user_path.to_string())
        })?;
        if !resolved_base.starts_with(&self.root) {
            return Err(VDriveError::Escape(user_path.to_string()));
        }

        // Rebuild the full path with the non-existent tail
        let mut result = resolved_base;
        for part in tail_parts.into_iter().rev() {
            let s = part.to_string_lossy();
            if s == "." || s == ".." {
                return Err(VDriveError::Escape(user_path.to_string()));
            }
            result = result.join(part);
        }

        Ok(result)
    }

    /// Convert an absolute path back to a drive-relative path for display.
    pub fn relative(&self, abs_path: &Path) -> String {
        abs_path
            .strip_prefix(&self.root)
            .unwrap_or(abs_path)
            .to_string_lossy()
            .replace('\\', "/")
    }
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

    #[test]
    fn open_existing_dir() {
        let dir = TempDir::new().unwrap();
        let vd = VDrive::open(dir.path()).unwrap();
        assert!(vd.root().is_absolute());
    }

    #[test]
    fn open_nonexistent_fails() {
        let result = VDrive::open(Path::new("/nonexistent/dir/12345"));
        assert!(result.is_err());
    }

    #[test]
    fn open_file_fails() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("afile");
        fs::write(&file, "hi").unwrap();
        let result = VDrive::open(&file);
        assert!(matches!(result, Err(VDriveError::NotDirectory(_))));
    }

    #[test]
    fn create_new_dir() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub").join("deep");
        let vd = VDrive::create(&sub).unwrap();
        assert!(vd.root().exists());
    }

    #[test]
    fn resolve_relative_path() {
        let (dir, vd) = setup();
        fs::write(dir.path().join("hello.txt"), "world").unwrap();
        let resolved = vd.resolve("hello.txt").unwrap();
        assert!(resolved.starts_with(vd.root()));
        assert!(resolved.ends_with("hello.txt"));
    }

    #[test]
    fn resolve_subdirectory() {
        let (dir, vd) = setup();
        fs::create_dir_all(dir.path().join("src").join("lib")).unwrap();
        fs::write(dir.path().join("src/lib/mod.rs"), "").unwrap();
        let resolved = vd.resolve("src/lib/mod.rs").unwrap();
        assert!(resolved.starts_with(vd.root()));
    }

    #[test]
    fn escape_dotdot_blocked() {
        let (_dir, vd) = setup();
        let result = vd.resolve("../../../etc/passwd");
        assert!(matches!(result, Err(VDriveError::NotFound(_)) | Err(VDriveError::Escape(_))));
    }

    #[test]
    fn escape_absolute_blocked() {
        let (_dir, vd) = setup();
        // Try an absolute path outside the root
        let result = vd.resolve("/etc/passwd");
        // On Windows this may be NotFound, on Linux it may be Escape
        assert!(result.is_err());
    }

    #[test]
    fn resolve_new_for_nonexistent_file() {
        let (dir, vd) = setup();
        // Parent exists, file doesn't
        let _ = fs::create_dir(dir.path().join("src"));
        let resolved = vd.resolve_new("src/new_file.rs").unwrap();
        assert!(resolved.starts_with(vd.root()));
        assert!(resolved.ends_with("new_file.rs"));
    }

    #[test]
    fn resolve_new_escape_blocked() {
        let (_dir, vd) = setup();
        let result = vd.resolve_new("../../escape.txt");
        assert!(result.is_err());
    }

    #[test]
    fn resolve_new_dotdot_filename_blocked() {
        let (_dir, vd) = setup();
        let result = vd.resolve_new("..");
        assert!(matches!(result, Err(VDriveError::Escape(_))));
    }

    #[test]
    fn relative_path_display() {
        let (dir, vd) = setup();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "").unwrap();
        let abs = vd.resolve("src/main.rs").unwrap();
        let rel = vd.relative(&abs);
        assert_eq!(rel, "src/main.rs");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_blocked() {
        let (dir, vd) = setup();
        // Create a symlink that points outside the root
        let link = dir.path().join("escape_link");
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        let result = vd.resolve("escape_link/passwd");
        assert!(matches!(result, Err(VDriveError::Escape(_))));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_within_root_ok() {
        let (dir, vd) = setup();
        fs::create_dir(dir.path().join("real")).unwrap();
        fs::write(dir.path().join("real/file.txt"), "ok").unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("real"),
            dir.path().join("link"),
        ).unwrap();
        // Symlink within root should work
        let resolved = vd.resolve("link/file.txt").unwrap();
        assert!(resolved.starts_with(vd.root()));
    }
}
