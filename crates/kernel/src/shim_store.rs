//! Shim store — fourth pillar of nuclear-proof kernel state.
//!
//! Per `project_shim_store_design.md`: shim_store is to cognition what
//! context_store is to working memory. The kernel manages a directory
//! tree of named shim_stores; each store holds an immutable transformer
//! base's cognitive substrate (ONNX shim weights + composition rules +
//! per-shim metadata). Two agents pointing at the same store have the
//! same cognition; forking the store name forks the cognition.
//!
//! ## Storage layout
//!
//! ```text
//! <base_dir>/<store_name>/
//! ├── manifest.json         (ShimStoreManifest)
//! ├── composition.json      (raw bytes — kernel doesn't interpret schema)
//! ├── activation_state.json (per-shim runtime stats; v1 stub for v3+ aging)
//! ├── shims/
//! │   ├── <shim_id>.onnx
//! │   └── <shim_id>.manifest.json
//! └── retired/              (soft-retired shims live here for archival)
//!     ├── <shim_id>.onnx
//!     └── <shim_id>.manifest.json
//! ```
//!
//! ## Atomicity
//!
//! All mutating ops follow the same pattern as the platform registry
//! snapshot (`crates/platform/src/snapshot.rs`): write file via temp +
//! fsync + atomic rename, then commit a WAL entry, then update in-memory
//! state. Crash before WAL = orphan file (visible via `list-stores`);
//! crash after WAL but before in-memory apply = replay reconstructs from
//! WAL + disk, verifying `content_hash` (mismatch = log + drop).
//!
//! ## Schema agnosticism
//!
//! The kernel stores shim manifests and composition bytes as opaque
//! `Vec<u8>` — it doesn't depend on `agentos-llm` (for `ShimAttachment`)
//! or `agentos-cortex-shim` (for `ShimManifest`). Validation happens at
//! the caller: the `shim-store` tool / pipeline / shim-expert agent
//! parses + validates before invoking kernel APIs. This keeps the kernel
//! crate's dependency surface minimal.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{KernelError, KernelResult};
use crate::wal::{EntryType, WalEntry};

/// Schema version for the on-disk shim_store format.
pub const SCHEMA_VERSION: u32 = 1;

/// Top-level manifest written to `<store>/manifest.json`. Lists which
/// transformer bases this cognition was trained against (compatibility
/// is the caller's responsibility — kernel doesn't enforce it).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShimStoreManifest {
    pub name: String,
    pub schema_version: u32,
    /// Base model names this shim_store's cognition was trained against.
    pub base_compat: Vec<String>,
    pub version: u32,
}

impl ShimStoreManifest {
    pub fn new(name: impl Into<String>, base_compat: Vec<String>) -> Self {
        Self {
            name: name.into(),
            schema_version: SCHEMA_VERSION,
            base_compat,
            version: 1,
        }
    }
}

/// One shim's metadata + reference to its on-disk ONNX file.
#[derive(Debug, Clone)]
pub struct ShimRecord {
    pub shim_id: String,
    /// Raw bytes of the per-shim manifest sidecar. Kernel doesn't
    /// interpret; callers parse as `agentos_cortex_shim::ShimManifest`.
    pub manifest_json: Vec<u8>,
    /// Absolute path to the ONNX file under `<store>/shims/`.
    pub onnx_path: PathBuf,
    /// SHA-256 hex of the ONNX bytes (set when added; verified on replay).
    pub content_hash: String,
}

/// In-memory state for one shim_store.
#[derive(Debug, Clone)]
pub struct ShimStoreState {
    pub manifest: ShimStoreManifest,
    /// Raw bytes of `composition.json`. Empty `{}` until first update.
    pub composition_bytes: Vec<u8>,
    /// Active (non-retired) shims, keyed by shim id.
    pub shims: HashMap<String, ShimRecord>,
}

/// Fourth pillar: cognitive substrate management.
#[derive(Debug)]
pub struct ShimStore {
    base_dir: PathBuf,
    stores: HashMap<String, ShimStoreState>,
}

impl ShimStore {
    /// Open or initialize the shim_store under `base_dir/`. Reads any
    /// existing stores from disk so manually-created or imported state
    /// is visible immediately; WAL replay then layers on top.
    pub fn open(base_dir: PathBuf) -> KernelResult<Self> {
        fs::create_dir_all(&base_dir)?;
        let mut stores = HashMap::new();
        if let Ok(entries) = fs::read_dir(&base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = match path.file_name().and_then(|n| n.to_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                if let Some(state) = load_store_state(&path) {
                    stores.insert(name, state);
                }
            }
        }
        Ok(Self { base_dir, stores })
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn path_for(&self, name: &str) -> PathBuf {
        self.base_dir.join(name)
    }

    pub fn list_stores(&self) -> Vec<String> {
        self.stores.keys().cloned().collect()
    }

    pub fn state_for(&self, name: &str) -> Option<&ShimStoreState> {
        self.stores.get(name)
    }

    pub fn composition_bytes_for(&self, name: &str) -> Option<&[u8]> {
        self.stores.get(name).map(|s| s.composition_bytes.as_slice())
    }

    pub fn manifest_for(&self, name: &str) -> Option<&ShimStoreManifest> {
        self.stores.get(name).map(|s| &s.manifest)
    }

    pub fn shims_in(&self, name: &str) -> Option<&HashMap<String, ShimRecord>> {
        self.stores.get(name).map(|s| &s.shims)
    }

    pub fn exists(&self, name: &str) -> bool {
        self.stores.contains_key(name)
    }

    // ── WAL replay ──

    pub fn apply_wal_entry(&mut self, entry: &WalEntry) {
        match entry.entry_type {
            EntryType::ShimStoreCreate => self.replay_create(&entry.payload),
            EntryType::ShimAdd => self.replay_add(&entry.payload),
            EntryType::ShimRetire => self.replay_retire(&entry.payload),
            EntryType::ShimStoreDelete => self.replay_delete(&entry.payload),
            EntryType::CompositionUpdate => self.replay_composition(&entry.payload),
            _ => {}
        }
    }

    fn replay_create(&mut self, payload: &[u8]) {
        let mut parts = payload.splitn(2, |b| *b == 0);
        let name = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let manifest_json = match parts.next() {
            Some(b) => b,
            None => return,
        };
        let manifest: ShimStoreManifest = match serde_json::from_slice(manifest_json) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(name = %name, error = %e, "ShimStoreCreate replay: bad manifest");
                return;
            }
        };
        // Re-load from disk if files exist (the boot scan in `open` may
        // have already loaded this; idempotent).
        let dir = self.base_dir.join(&name);
        let state = load_store_state(&dir).unwrap_or(ShimStoreState {
            manifest: manifest.clone(),
            composition_bytes: b"{}".to_vec(),
            shims: HashMap::new(),
        });
        self.stores.insert(name, state);
    }

    fn replay_add(&mut self, payload: &[u8]) {
        let mut parts = payload.splitn(4, |b| *b == 0);
        let store_name = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let shim_id = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let manifest_json = match parts.next() {
            Some(b) => b.to_vec(),
            None => return,
        };
        let content_hash = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };

        let store_dir = self.base_dir.join(&store_name);
        let onnx_path = store_dir.join("shims").join(format!("{shim_id}.onnx"));
        if !onnx_path.exists() {
            tracing::warn!(
                store = %store_name,
                shim = %shim_id,
                path = %onnx_path.display(),
                "ShimAdd replay: ONNX missing on disk; dropping entry"
            );
            return;
        }
        let actual_hash = match hash_file(&onnx_path) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(
                    store = %store_name,
                    shim = %shim_id,
                    error = %e,
                    "ShimAdd replay: hash compute failed; dropping"
                );
                return;
            }
        };
        if actual_hash != content_hash {
            tracing::warn!(
                store = %store_name,
                shim = %shim_id,
                expected = %content_hash,
                actual = %actual_hash,
                "ShimAdd replay: content_hash mismatch; dropping entry"
            );
            return;
        }

        let record = ShimRecord {
            shim_id: shim_id.clone(),
            manifest_json,
            onnx_path,
            content_hash,
        };
        if let Some(state) = self.stores.get_mut(&store_name) {
            state.shims.insert(shim_id, record);
        } else {
            tracing::warn!(
                store = %store_name,
                shim = %shim_id,
                "ShimAdd replay: target store not in memory; dropping"
            );
        }
    }

    fn replay_retire(&mut self, payload: &[u8]) {
        let mut parts = payload.splitn(2, |b| *b == 0);
        let store_name = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let shim_id = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        if let Some(state) = self.stores.get_mut(&store_name) {
            state.shims.remove(&shim_id);
        }
    }

    fn replay_delete(&mut self, payload: &[u8]) {
        let name = match std::str::from_utf8(payload) {
            Ok(s) => s.to_string(),
            Err(_) => return,
        };
        self.stores.remove(&name);
    }

    fn replay_composition(&mut self, payload: &[u8]) {
        let mut parts = payload.splitn(2, |b| *b == 0);
        let store_name = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let expected_hash = match parts.next().and_then(|b| std::str::from_utf8(b).ok()) {
            Some(s) => s.to_string(),
            None => return,
        };
        let composition_path = self.base_dir.join(&store_name).join("composition.json");
        let bytes = match fs::read(&composition_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    store = %store_name,
                    error = %e,
                    "CompositionUpdate replay: composition.json unreadable"
                );
                return;
            }
        };
        let actual_hash = hash_bytes(&bytes);
        if actual_hash != expected_hash {
            tracing::warn!(
                store = %store_name,
                expected = %expected_hash,
                actual = %actual_hash,
                "CompositionUpdate replay: hash mismatch; dropping"
            );
            return;
        }
        if let Some(state) = self.stores.get_mut(&store_name) {
            state.composition_bytes = bytes;
        }
    }

    // ── Mutating ops (called by Kernel) ──
    //
    // Each method writes files to disk first (atomically), updates
    // in-memory state, and returns a WalEntry the caller must commit.
    // The caller (Kernel) drives the WAL append so cross-pillar batches
    // can use the same WAL transaction.

    /// Idempotent: re-creating an existing store is a no-op (returns a
    /// fresh ShimStoreCreate WAL entry over the existing manifest).
    pub fn create_store(
        &mut self,
        name: &str,
        base_compat: Vec<String>,
    ) -> KernelResult<WalEntry> {
        validate_name("shim_store name", name)?;
        let dir = self.base_dir.join(name);
        fs::create_dir_all(dir.join("shims"))?;
        fs::create_dir_all(dir.join("retired"))?;

        let manifest = if let Some(existing) = self.stores.get(name) {
            existing.manifest.clone()
        } else {
            ShimStoreManifest::new(name, base_compat)
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
            KernelError::InvalidData(format!("serialize manifest: {e}"))
        })?;
        write_atomic(&dir.join("manifest.json"), &manifest_bytes)?;

        let composition_path = dir.join("composition.json");
        if !composition_path.exists() {
            write_atomic(&composition_path, b"{}")?;
        }
        let activation_path = dir.join("activation_state.json");
        if !activation_path.exists() {
            write_atomic(&activation_path, b"{}")?;
        }

        let composition_bytes = fs::read(&composition_path).unwrap_or_else(|_| b"{}".to_vec());
        self.stores
            .entry(name.to_string())
            .or_insert_with(|| ShimStoreState {
                manifest: manifest.clone(),
                composition_bytes,
                shims: HashMap::new(),
            });

        let mut payload = Vec::with_capacity(name.len() + 1 + manifest_bytes.len());
        payload.extend_from_slice(name.as_bytes());
        payload.push(0);
        payload.extend_from_slice(&manifest_bytes);
        Ok(WalEntry::new(EntryType::ShimStoreCreate, payload))
    }

    pub fn add_shim(
        &mut self,
        store_name: &str,
        shim_id: &str,
        manifest_json: Vec<u8>,
        onnx_bytes: Vec<u8>,
    ) -> KernelResult<WalEntry> {
        validate_name("shim_store name", store_name)?;
        validate_name("shim_id", shim_id)?;
        if !self.stores.contains_key(store_name) {
            return Err(KernelError::InvalidData(format!(
                "shim_store `{store_name}` does not exist; create it first"
            )));
        }
        let store_dir = self.base_dir.join(store_name);
        let onnx_path = store_dir.join("shims").join(format!("{shim_id}.onnx"));
        let manifest_path = store_dir.join("shims").join(format!("{shim_id}.manifest.json"));

        let content_hash = hash_bytes(&onnx_bytes);
        write_atomic(&onnx_path, &onnx_bytes)?;
        write_atomic(&manifest_path, &manifest_json)?;

        let record = ShimRecord {
            shim_id: shim_id.to_string(),
            manifest_json: manifest_json.clone(),
            onnx_path: onnx_path.clone(),
            content_hash: content_hash.clone(),
        };
        if let Some(state) = self.stores.get_mut(store_name) {
            state.shims.insert(shim_id.to_string(), record);
        }

        let mut payload = Vec::with_capacity(
            store_name.len() + shim_id.len() + manifest_json.len() + content_hash.len() + 3,
        );
        payload.extend_from_slice(store_name.as_bytes());
        payload.push(0);
        payload.extend_from_slice(shim_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(&manifest_json);
        payload.push(0);
        payload.extend_from_slice(content_hash.as_bytes());
        Ok(WalEntry::new(EntryType::ShimAdd, payload))
    }

    pub fn retire_shim(
        &mut self,
        store_name: &str,
        shim_id: &str,
    ) -> KernelResult<WalEntry> {
        validate_name("shim_store name", store_name)?;
        validate_name("shim_id", shim_id)?;
        let state = self.stores.get_mut(store_name).ok_or_else(|| {
            KernelError::InvalidData(format!("shim_store `{store_name}` does not exist"))
        })?;
        let record = state.shims.remove(shim_id).ok_or_else(|| {
            KernelError::InvalidData(format!(
                "shim `{shim_id}` not active in store `{store_name}`"
            ))
        })?;

        let store_dir = self.base_dir.join(store_name);
        let from_onnx = record.onnx_path;
        let from_manifest = store_dir
            .join("shims")
            .join(format!("{shim_id}.manifest.json"));
        let to_onnx = store_dir.join("retired").join(format!("{shim_id}.onnx"));
        let to_manifest = store_dir
            .join("retired")
            .join(format!("{shim_id}.manifest.json"));

        fs::create_dir_all(store_dir.join("retired"))?;
        if from_onnx.exists() {
            let _ = fs::rename(&from_onnx, &to_onnx);
        }
        if from_manifest.exists() {
            let _ = fs::rename(&from_manifest, &to_manifest);
        }

        let mut payload = Vec::with_capacity(store_name.len() + shim_id.len() + 1);
        payload.extend_from_slice(store_name.as_bytes());
        payload.push(0);
        payload.extend_from_slice(shim_id.as_bytes());
        Ok(WalEntry::new(EntryType::ShimRetire, payload))
    }

    pub fn update_composition(
        &mut self,
        store_name: &str,
        composition_bytes: Vec<u8>,
    ) -> KernelResult<WalEntry> {
        validate_name("shim_store name", store_name)?;
        if !self.stores.contains_key(store_name) {
            return Err(KernelError::InvalidData(format!(
                "shim_store `{store_name}` does not exist"
            )));
        }
        let composition_path = self.base_dir.join(store_name).join("composition.json");
        let content_hash = hash_bytes(&composition_bytes);
        write_atomic(&composition_path, &composition_bytes)?;
        if let Some(state) = self.stores.get_mut(store_name) {
            state.composition_bytes = composition_bytes;
        }

        let mut payload = Vec::with_capacity(store_name.len() + content_hash.len() + 1);
        payload.extend_from_slice(store_name.as_bytes());
        payload.push(0);
        payload.extend_from_slice(content_hash.as_bytes());
        Ok(WalEntry::new(EntryType::CompositionUpdate, payload))
    }

    pub fn delete_store(&mut self, name: &str) -> KernelResult<WalEntry> {
        validate_name("shim_store name", name)?;
        let dir = self.base_dir.join(name);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        self.stores.remove(name);
        Ok(WalEntry::new(
            EntryType::ShimStoreDelete,
            name.as_bytes().to_vec(),
        ))
    }
}

// ── Helpers ──

/// Validate a shim_store name or shim_id. Strict ASCII allowlist so
/// these strings are safe to splice into filesystem paths without
/// `..` traversal, drive-letter override (Windows `C:\Users\foo`), or
/// path-separator injection. Used at every public boundary that turns
/// a name into a file path.
///
/// Charset: `[A-Za-z0-9_-]`, 1-64 chars. Real-world names like
/// `bob-coastliners` or `should_respond_v3` fit; nothing else does.
fn validate_name(kind: &str, s: &str) -> KernelResult<()> {
    let len = s.len();
    if !(1..=64).contains(&len) {
        return Err(KernelError::InvalidData(format!(
            "{kind} must be 1-64 chars (got {len})"
        )));
    }
    if !s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Err(KernelError::InvalidData(format!(
            "{kind} contains invalid characters; allowed: [A-Za-z0-9_-]"
        )));
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> KernelResult<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };
    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn hash_file(path: &Path) -> KernelResult<String> {
    let bytes = fs::read(path)?;
    Ok(hash_bytes(&bytes))
}

/// Boot-time loader: read a store's on-disk state into memory. Returns
/// None when the dir doesn't have a parseable manifest (the store is
/// being created or was deleted partially).
fn load_store_state(dir: &Path) -> Option<ShimStoreState> {
    let manifest_bytes = fs::read(dir.join("manifest.json")).ok()?;
    let manifest: ShimStoreManifest = serde_json::from_slice(&manifest_bytes).ok()?;
    let composition_bytes =
        fs::read(dir.join("composition.json")).unwrap_or_else(|_| b"{}".to_vec());

    let mut shims = HashMap::new();
    let shims_dir = dir.join("shims");
    if let Ok(entries) = fs::read_dir(&shims_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
                continue;
            }
            let shim_id = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let manifest_path = shims_dir.join(format!("{shim_id}.manifest.json"));
            let manifest_json = match fs::read(&manifest_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let content_hash = match hash_file(&path) {
                Ok(h) => h,
                Err(_) => continue,
            };
            shims.insert(
                shim_id.clone(),
                ShimRecord {
                    shim_id,
                    manifest_json,
                    onnx_path: path,
                    content_hash,
                },
            );
        }
    }

    Some(ShimStoreState {
        manifest,
        composition_bytes,
        shims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh() -> (TempDir, ShimStore) {
        let dir = TempDir::new().unwrap();
        let store = ShimStore::open(dir.path().to_path_buf()).unwrap();
        (dir, store)
    }

    #[test]
    fn open_empty_dir_has_no_stores() {
        let (_dir, s) = fresh();
        assert!(s.list_stores().is_empty());
    }

    #[test]
    fn create_store_writes_manifest_and_records_state() {
        let (_dir, mut s) = fresh();
        let entry = s
            .create_store("bob-coastliners", vec!["qwen-2.5-3b".into()])
            .unwrap();
        assert_eq!(entry.entry_type, EntryType::ShimStoreCreate);
        assert!(s.exists("bob-coastliners"));

        let manifest_path = s.path_for("bob-coastliners").join("manifest.json");
        assert!(manifest_path.exists());
        let manifest: ShimStoreManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.name, "bob-coastliners");
        assert_eq!(manifest.base_compat, vec!["qwen-2.5-3b"]);
        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn create_store_is_idempotent() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec!["q".into()]).unwrap();
        s.create_store("bob", vec!["q".into()]).unwrap();
        assert_eq!(s.list_stores().len(), 1);
    }

    #[test]
    fn add_shim_writes_files_and_records_hash() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        let manifest = br#"{"id":"should_respond","phase":"gate"}"#.to_vec();
        let onnx = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let entry = s
            .add_shim("bob", "should_respond", manifest.clone(), onnx.clone())
            .unwrap();
        assert_eq!(entry.entry_type, EntryType::ShimAdd);

        let onnx_path = s
            .path_for("bob")
            .join("shims")
            .join("should_respond.onnx");
        assert!(onnx_path.exists());
        assert_eq!(fs::read(&onnx_path).unwrap(), onnx);

        let record = s.shims_in("bob").unwrap().get("should_respond").unwrap();
        assert_eq!(record.manifest_json, manifest);
        assert_eq!(record.content_hash, hash_bytes(&onnx));
    }

    #[test]
    fn add_shim_to_missing_store_errors() {
        let (_dir, mut s) = fresh();
        let err = s
            .add_shim("noop", "x", b"{}".to_vec(), vec![1, 2, 3])
            .unwrap_err();
        match err {
            KernelError::InvalidData(msg) => assert!(msg.contains("does not exist")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn retire_shim_moves_files_and_drops_record() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        s.add_shim("bob", "x", b"{}".to_vec(), vec![9, 9]).unwrap();
        s.retire_shim("bob", "x").unwrap();

        assert!(!s.path_for("bob").join("shims").join("x.onnx").exists());
        assert!(s.path_for("bob").join("retired").join("x.onnx").exists());
        assert!(s.shims_in("bob").unwrap().is_empty());
    }

    #[test]
    fn update_composition_writes_file_and_records_bytes() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        let body = br#"{"gate_shims":["should_respond"]}"#.to_vec();
        let entry = s.update_composition("bob", body.clone()).unwrap();
        assert_eq!(entry.entry_type, EntryType::CompositionUpdate);
        assert_eq!(s.composition_bytes_for("bob").unwrap(), body.as_slice());
        let on_disk = fs::read(s.path_for("bob").join("composition.json")).unwrap();
        assert_eq!(on_disk, body);
    }

    #[test]
    fn delete_store_removes_dir_and_state() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        s.add_shim("bob", "x", b"{}".to_vec(), vec![1, 2]).unwrap();
        let entry = s.delete_store("bob").unwrap();
        assert_eq!(entry.entry_type, EntryType::ShimStoreDelete);
        assert!(!s.path_for("bob").exists());
        assert!(!s.exists("bob"));
    }

    #[test]
    fn open_picks_up_existing_store_from_disk() {
        let dir = TempDir::new().unwrap();
        {
            let mut s = ShimStore::open(dir.path().to_path_buf()).unwrap();
            s.create_store("bob", vec!["q".into()]).unwrap();
            s.add_shim("bob", "x", b"{}".to_vec(), vec![5, 5]).unwrap();
            s.update_composition("bob", b"{\"steer_shims\":[]}".to_vec())
                .unwrap();
        }
        let s = ShimStore::open(dir.path().to_path_buf()).unwrap();
        assert!(s.exists("bob"));
        assert_eq!(s.shims_in("bob").unwrap().len(), 1);
        assert_eq!(
            s.composition_bytes_for("bob").unwrap(),
            b"{\"steer_shims\":[]}"
        );
    }

    #[test]
    fn replay_create_inserts_state() {
        let (_dir, mut s) = fresh();
        // Manually craft a payload as if WAL replayed.
        let manifest = ShimStoreManifest::new("alice", vec!["base".into()]);
        let manifest_json = serde_json::to_vec(&manifest).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(b"alice");
        payload.push(0);
        payload.extend_from_slice(&manifest_json);
        // Create the dir on disk so load_store_state succeeds; if it
        // doesn't, replay_create still inserts an empty fallback state.
        fs::create_dir_all(s.path_for("alice").join("shims")).unwrap();
        write_atomic(
            &s.path_for("alice").join("manifest.json"),
            &manifest_json,
        )
        .unwrap();
        write_atomic(&s.path_for("alice").join("composition.json"), b"{}").unwrap();

        s.apply_wal_entry(&WalEntry::new(EntryType::ShimStoreCreate, payload));
        assert!(s.exists("alice"));
        assert_eq!(s.manifest_for("alice").unwrap().name, "alice");
    }

    #[test]
    fn replay_add_drops_entry_on_hash_mismatch() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        // Write a real shim file but craft a WAL payload with a wrong
        // content_hash; replay should drop the entry.
        let onnx = vec![1u8, 2, 3];
        let onnx_path = s.path_for("bob").join("shims").join("hijacked.onnx");
        fs::create_dir_all(onnx_path.parent().unwrap()).unwrap();
        write_atomic(&onnx_path, &onnx).unwrap();
        write_atomic(
            &s.path_for("bob").join("shims").join("hijacked.manifest.json"),
            b"{}",
        )
        .unwrap();

        let mut payload = Vec::new();
        payload.extend_from_slice(b"bob");
        payload.push(0);
        payload.extend_from_slice(b"hijacked");
        payload.push(0);
        payload.extend_from_slice(b"{}");
        payload.push(0);
        payload.extend_from_slice(b"deadbeef-not-the-real-hash");

        s.apply_wal_entry(&WalEntry::new(EntryType::ShimAdd, payload));
        // Shim should NOT have been recorded — hash mismatch dropped it.
        assert!(s.shims_in("bob").unwrap().get("hijacked").is_none());
    }

    #[test]
    fn replay_retire_drops_active_record() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        s.add_shim("bob", "x", b"{}".to_vec(), vec![7]).unwrap();

        let mut payload = Vec::new();
        payload.extend_from_slice(b"bob");
        payload.push(0);
        payload.extend_from_slice(b"x");
        s.apply_wal_entry(&WalEntry::new(EntryType::ShimRetire, payload));
        assert!(s.shims_in("bob").unwrap().get("x").is_none());
    }

    #[test]
    fn replay_delete_removes_store_from_memory() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        s.apply_wal_entry(&WalEntry::new(
            EntryType::ShimStoreDelete,
            b"bob".to_vec(),
        ));
        assert!(!s.exists("bob"));
    }

    #[test]
    fn create_store_rejects_path_traversal_in_name() {
        let (dir, mut s) = fresh();
        let bad_names = [
            "../escape",
            "../../etc/agentos-evil",
            "good/../bad",
            "a/b",
            "a\\b",
            "..",
            ".",
        ];
        for name in bad_names {
            let err = s.create_store(name, vec![]).unwrap_err();
            assert!(
                matches!(err, KernelError::InvalidData(_)),
                "expected InvalidData for {name:?}, got {err:?}"
            );
        }
        // No directories created outside the base.
        let parent = dir.path().parent().unwrap();
        let escape_dir = parent.join("escape");
        assert!(
            !escape_dir.exists(),
            "path traversal created directory at {}",
            escape_dir.display()
        );
    }

    #[test]
    #[cfg(windows)]
    fn create_store_rejects_drive_letter_override_on_windows() {
        // On Windows, Path::join with an absolute path overrides the
        // base entirely. validate_name must reject `:` and `\` etc.
        let (_dir, mut s) = fresh();
        let err = s.create_store("C:\\Users\\evil", vec![]).unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));
    }

    #[test]
    fn add_shim_rejects_traversal_in_shim_id_and_store() {
        let (_dir, mut s) = fresh();
        s.create_store("bob", vec![]).unwrap();
        // Bad shim_id.
        let err = s
            .add_shim("bob", "../../etc/cron.d/x", b"{}".to_vec(), vec![1])
            .unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));
        // Bad store_name.
        let err = s
            .add_shim("../escape", "ok", b"{}".to_vec(), vec![1])
            .unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));
    }

    #[test]
    fn delete_store_rejects_traversal_and_preserves_unrelated_dirs() {
        let (dir, mut s) = fresh();
        // Plant a directory we don't want destroyed.
        let sibling = dir.path().parent().unwrap().join("agentos-keep-me");
        fs::create_dir_all(&sibling).unwrap();

        let err = s.delete_store("../agentos-keep-me").unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));
        assert!(
            sibling.exists(),
            "delete_store with traversal destroyed sibling at {}",
            sibling.display()
        );
        fs::remove_dir_all(&sibling).ok();
    }

    #[test]
    fn empty_and_oversize_names_rejected() {
        let (_dir, mut s) = fresh();
        assert!(matches!(
            s.create_store("", vec![]).unwrap_err(),
            KernelError::InvalidData(_)
        ));
        let huge = "a".repeat(65);
        assert!(matches!(
            s.create_store(&huge, vec![]).unwrap_err(),
            KernelError::InvalidData(_)
        ));
        let just_right = "a".repeat(64);
        s.create_store(&just_right, vec![]).unwrap();
    }

    #[test]
    fn typical_names_still_accepted() {
        let (_dir, mut s) = fresh();
        for name in [
            "bob",
            "bob-coastliners",
            "qa_expert_dev",
            "tenant-2026-05",
        ] {
            s.create_store(name, vec![]).unwrap();
            assert!(s.exists(name));
        }
        // Typical shim_id shapes from shim-expert workflows.
        for id in ["should_respond_v3", "gate-eager", "steer_polite_v1"] {
            s.add_shim("bob", id, b"{}".to_vec(), vec![1]).unwrap();
        }
    }
}
