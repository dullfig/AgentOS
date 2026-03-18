//! WASM Capability Grants — structural security for WASM tools.
//!
//! Default = nothing. No filesystem, no env, no stdio.
//! Capabilities are granted via organism.yaml and enforced structurally
//! by building the WasiCtx with only the granted imports.

use wasmtime_wasi::filesystem::{DirPerms, FilePerms};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

use super::error::WasmError;

/// Capability grants for a WASM tool component.
///
/// Default is empty — no access to anything.
#[derive(Debug, Clone, Default)]
pub struct WasmCapabilities {
    pub filesystem: Vec<FsGrant>,
    pub env_vars: Vec<EnvGrant>,
    pub stdio: bool,
    /// KV store grants. None = no KV access. Some = private namespace
    /// plus any additional read/write grants to shared namespaces.
    pub kv: Option<KvGrant>,
}

/// KV store access grant for a WASM tool.
#[derive(Debug, Clone)]
pub struct KvGrant {
    /// Shared namespaces this tool can read (in addition to its own private namespace).
    pub read: Vec<String>,
    /// Shared namespaces this tool can write (in addition to its own private namespace).
    pub write: Vec<String>,
}

/// A filesystem access grant.
#[derive(Debug, Clone)]
pub struct FsGrant {
    pub host_path: String,
    pub guest_path: String,
    pub read_only: bool,
}

/// An environment variable grant.
#[derive(Debug, Clone)]
pub struct EnvGrant {
    pub key: String,
    pub value: String,
}

impl WasmCapabilities {
    /// Build a WasiCtx from these capability grants.
    ///
    /// Only the explicitly granted capabilities are wired in.
    /// Missing capability = WASI import doesn't exist = structural impossibility.
    pub fn build_wasi_ctx(&self) -> Result<WasiCtx, WasmError> {
        let mut builder = WasiCtxBuilder::new();

        if self.stdio {
            builder.inherit_stdio();
        }

        for env in &self.env_vars {
            builder.env(&env.key, &env.value);
        }

        for fs in &self.filesystem {
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
        let result = caps.build_wasi_ctx();
        assert!(result.is_ok(), "empty caps should build: {:?}", result.err());
    }

    #[test]
    fn build_wasi_ctx_with_stdio() {
        let caps = WasmCapabilities {
            stdio: true,
            ..Default::default()
        };
        let result = caps.build_wasi_ctx();
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
        let result = caps.build_wasi_ctx();
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
        let result = caps.build_wasi_ctx();
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
        let result = caps.build_wasi_ctx();
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
        let result = caps.build_wasi_ctx();
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
