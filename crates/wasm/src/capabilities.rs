//! WASM Capability Grants — structural security for WASM tools.
//!
//! Default = nothing. No filesystem, no env, no stdio.
//! Capabilities are granted via organism.yaml and enforced structurally
//! by building the WasiCtx with only the granted imports.
//!
//! Data types live in `agentos-events` for cross-crate access.
//! The WasiCtx builder stays here (depends on wasmtime).

use wasmtime_wasi::filesystem::{DirPerms, FilePerms};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

use super::error::WasmError;

// Re-export shared types from events crate
pub use agentos_events::{EnvGrant, FsGrant, KvGrant, WasmCapabilities};

/// Build a WasiCtx from capability grants.
///
/// Only the explicitly granted capabilities are wired in.
/// Missing capability = WASI import doesn't exist = structural impossibility.
pub fn build_wasi_ctx(caps: &WasmCapabilities) -> Result<WasiCtx, WasmError> {
    let mut builder = WasiCtxBuilder::new();

    if caps.stdio {
        builder.inherit_stdio();
    }

    for env in &caps.env_vars {
        builder.env(&env.key, &env.value);
    }

    for fs in &caps.filesystem {
        let host = std::path::Path::new(&fs.host_path);
        if !host.exists() {
            return Err(WasmError::Capability(format!(
                "host path does not exist: {}",
                fs.host_path
            )));
        }

        let (dir_perms, file_perms) = if fs.read_only {
            (DirPerms::READ, FilePerms::READ)
        } else {
            (DirPerms::all(), FilePerms::all())
        };

        builder
            .preopened_dir(&fs.host_path, &fs.guest_path, dir_perms, file_perms)
            .map_err(|e| {
                WasmError::Capability(format!(
                    "failed to preopen '{}' → '{}': {e}",
                    fs.host_path, fs.guest_path
                ))
            })?;
    }

    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capabilities_empty() {
        let caps = WasmCapabilities::default();
        assert!(caps.filesystem.is_empty());
        assert!(caps.env_vars.is_empty());
        assert!(!caps.stdio);
    }

    #[test]
    fn build_wasi_ctx_empty() {
        let caps = WasmCapabilities::default();
        let result = build_wasi_ctx(&caps);
        assert!(result.is_ok(), "empty caps should build: {:?}", result.err());
    }

    #[test]
    fn build_wasi_ctx_with_stdio() {
        let caps = WasmCapabilities {
            stdio: true,
            ..Default::default()
        };
        let result = build_wasi_ctx(&caps);
        assert!(result.is_ok());
    }

    #[test]
    fn build_wasi_ctx_with_env() {
        let caps = WasmCapabilities {
            env_vars: vec![
                EnvGrant {
                    key: "RUST_LOG".into(),
                    value: "info".into(),
                },
                EnvGrant {
                    key: "HOME".into(),
                    value: "/home/tool".into(),
                },
            ],
            ..Default::default()
        };
        let result = build_wasi_ctx(&caps);
        assert!(result.is_ok());
    }

    #[test]
    fn build_wasi_ctx_with_fs_read_only() {
        let dir = tempfile::TempDir::new().unwrap();
        let caps = WasmCapabilities {
            filesystem: vec![FsGrant {
                host_path: dir.path().to_string_lossy().into_owned(),
                guest_path: "/data".into(),
                read_only: true,
            }],
            ..Default::default()
        };
        let result = build_wasi_ctx(&caps);
        assert!(result.is_ok(), "read-only fs grant failed: {:?}", result.err());
    }

    #[test]
    fn build_wasi_ctx_with_fs_read_write() {
        let dir = tempfile::TempDir::new().unwrap();
        let caps = WasmCapabilities {
            filesystem: vec![FsGrant {
                host_path: dir.path().to_string_lossy().into_owned(),
                guest_path: "/workspace".into(),
                read_only: false,
            }],
            ..Default::default()
        };
        let result = build_wasi_ctx(&caps);
        assert!(result.is_ok(), "read-write fs grant failed: {:?}", result.err());
    }

    #[test]
    fn validate_missing_path() {
        let caps = WasmCapabilities {
            filesystem: vec![FsGrant {
                host_path: "/nonexistent/path/that/does/not/exist".into(),
                guest_path: "/data".into(),
                read_only: true,
            }],
            ..Default::default()
        };
        let result = build_wasi_ctx(&caps);
        assert!(result.is_err());
        let err = result.err().unwrap();
        match err {
            WasmError::Capability(msg) => {
                assert!(msg.contains("does not exist"), "unexpected error: {msg}");
            }
            other => panic!("expected Capability error, got: {other}"),
        }
    }
}
