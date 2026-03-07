//! Virtual drive management — mount folders as sandboxed agent workspaces.
//!
//! A VDrive is a real directory on disk, exposed to the agent through
//! sandboxed file tools. No QCOW2, no virtual filesystem — real files,
//! real git, real compilers. Containment comes from the WASM sandbox:
//! tools can only access the mounted directory.
//!
//! This module handles the lifecycle (mount/unmount/create). The actual
//! sandboxed file operations live in `agentos_vdrive::VDrive`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agentos_vdrive::VDrive;

/// Default directory for created workspaces: `~/.agentos/workspaces/`.
pub fn default_workspace_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".agentos").join("workspaces")
}

/// Mount an existing directory as a VDrive.
pub fn mount(path: &Path) -> Result<Arc<VDrive>, String> {
    if !path.exists() {
        return Err(format!("directory not found: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("not a directory: {}", path.display()));
    }
    let drive = VDrive::open(path).map_err(|e| format!("mount failed: {e}"))?;
    Ok(Arc::new(drive))
}

/// Create a new empty workspace directory and mount it.
pub fn create_and_mount(name: &str) -> Result<(PathBuf, Arc<VDrive>), String> {
    if name.is_empty() {
        return Err("workspace name cannot be empty".into());
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err("workspace name must be alphanumeric, hyphens, or underscores".into());
    }

    let dir = default_workspace_dir().join(name);
    if dir.exists() {
        return Err(format!("workspace already exists: {}", dir.display()));
    }

    let drive = VDrive::create(&dir).map_err(|e| format!("create failed: {e}"))?;
    Ok((dir, Arc::new(drive)))
}

/// List existing workspace directories.
pub fn list_workspaces() -> Result<Vec<WorkspaceInfo>, String> {
    let dir = default_workspace_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut workspaces = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| format!("failed to read {}: {e}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read error: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            workspaces.push(WorkspaceInfo { name, path });
        }
    }

    workspaces.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(workspaces)
}

/// Info about an existing workspace.
pub struct WorkspaceInfo {
    pub name: String,
    pub path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_workspace_dir_is_under_agentos() {
        let dir = default_workspace_dir();
        assert!(dir.to_string_lossy().contains(".agentos"));
        assert!(dir.to_string_lossy().contains("workspaces"));
    }

    #[test]
    fn mount_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let drive = mount(dir.path()).unwrap();
        assert!(drive.root().exists());
    }

    #[test]
    fn mount_nonexistent_errors() {
        let result = mount(Path::new("/nonexistent/dir/12345"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn mount_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("afile");
        std::fs::write(&file, "hi").unwrap();
        let result = mount(&file);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a directory"));
    }

    #[test]
    fn create_rejects_empty_name() {
        let result = create_and_mount("");
        assert!(result.is_err());
    }

    #[test]
    fn create_rejects_bad_chars() {
        let result = create_and_mount("foo/bar");
        assert!(result.is_err());
    }

    #[test]
    fn list_empty() {
        // Just verify it doesn't crash on nonexistent dir
        let workspaces = list_workspaces().unwrap();
        // May or may not be empty depending on whether ~/.agentos/workspaces/ exists
        let _ = workspaces;
    }
}
