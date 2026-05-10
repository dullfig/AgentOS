//! JSON snapshot persistence for [`crate::registry::InstanceRegistry`].
//!
//! The platform registry is metadata about live agent instances —
//! address↔thread_id bindings, organism templates, lifecycle policies,
//! cache shard names. To make instances survive process restart, the
//! registry writes a snapshot to disk on every materialize / kill /
//! idle-eviction; on boot, it replays the snapshot back.
//!
//! # Format
//!
//! JSON. Plain `serde_json::to_vec_pretty` round-trips. Versioned at
//! the top level so a future schema change can be detected.
//!
//! # Atomicity
//!
//! Writes go through [`write_atomic`]: serialize to a `<path>.tmp`
//! sibling, fsync, then rename over the target. POSIX rename is atomic
//! within a directory; NTFS rename is atomic too as long as the source
//! and target sit on the same volume (always true here — same dir).
//! A crash mid-write leaves either the old snapshot intact (rename
//! didn't happen) or the new one in place (rename did happen). The
//! `.tmp` may be left behind on crash; that's a benign leftover.
//!
//! # On corruption
//!
//! [`read`] treats a malformed snapshot file the same as a missing
//! one: returns `Ok(None)`. The caller treats this as "first boot,"
//! the registry comes up empty, and the next materialize writes a
//! fresh snapshot. Corruption shouldn't happen given the atomic write
//! discipline, but degrading gracefully is cheap.

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::registry::Lifetime;

/// Top-level snapshot. `version` lets a future schema migration detect
/// stale on-disk data and either upgrade or treat as first-boot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub version: u32,
    pub instances: Vec<InstanceRecord>,
}

impl RegistrySnapshot {
    /// Schema version of snapshots produced by this build.
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(instances: Vec<InstanceRecord>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            instances,
        }
    }
}

/// One instance entry. Mirrors the durable subset of
/// [`crate::registry::InstanceInfo`]:
///
/// - `created_at` / `last_accessed`: omitted. They're `Instant` values
///   that don't survive process boundaries; the next idle-eviction
///   tick will refresh `last_accessed` naturally.
/// - `tier`: omitted. Restart always brings instances back as `Active`;
///   re-tiering happens via the normal eviction path.
/// - `buffers`: omitted. Buffer thread_ids are derived deterministically
///   from `(instance_thread_id, buffer_id)`, so they reconstruct on
///   first message after restart without separate persistence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstanceRecord {
    pub address_raw: String,
    pub organism: String,
    pub thread_id: String,
    pub lifetime: LifetimeRecord,
    pub parent_raw: Option<String>,
    pub cache_shards: Vec<String>,
}

/// Wire form of [`Lifetime`] — `Duration` round-trips through serde
/// natively so this is a thin tag-on-discriminant mirror. Kept separate
/// from `Lifetime` itself to avoid dragging serde into the in-memory
/// type's surface (and to make schema versioning explicit).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LifetimeRecord {
    UntilIdle { idle: Duration },
    UntilTaskComplete,
    Pinned,
    Ephemeral,
}

impl From<&Lifetime> for LifetimeRecord {
    fn from(value: &Lifetime) -> Self {
        match value {
            Lifetime::UntilIdle(d) => LifetimeRecord::UntilIdle { idle: *d },
            Lifetime::UntilTaskComplete => LifetimeRecord::UntilTaskComplete,
            Lifetime::Pinned => LifetimeRecord::Pinned,
            Lifetime::Ephemeral => LifetimeRecord::Ephemeral,
        }
    }
}

impl From<LifetimeRecord> for Lifetime {
    fn from(value: LifetimeRecord) -> Self {
        match value {
            LifetimeRecord::UntilIdle { idle } => Lifetime::UntilIdle(idle),
            LifetimeRecord::UntilTaskComplete => Lifetime::UntilTaskComplete,
            LifetimeRecord::Pinned => Lifetime::Pinned,
            LifetimeRecord::Ephemeral => Lifetime::Ephemeral,
        }
    }
}

/// Serialize and atomically replace the file at `path`.
///
/// Algorithm: write to `<path>.tmp`, fsync the temp, rename over.
/// The parent directory must already exist; this function does not
/// create it. Caller (the registry) is responsible for ensuring the
/// kernel's data dir exists before the first snapshot write.
pub fn write_atomic(path: &Path, snapshot: &RegistrySnapshot) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        std::path::PathBuf::from(p)
    };

    let bytes = serde_json::to_vec_pretty(snapshot)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }

    fs::rename(&tmp_path, path)?;
    // Best-effort: fsync the parent directory so the rename itself is
    // durable. Unix-only — opening a directory for read on Windows is
    // disallowed, and NTFS commits directory entries on rename anyway.
    #[cfg(unix)]
    {
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
    }
    Ok(())
}

/// Read the snapshot at `path`. Returns `Ok(None)` when the file is
/// missing or unparseable — both are treated as "first boot." Only
/// IO errors other than NotFound bubble up.
pub fn read(path: &Path) -> io::Result<Option<RegistrySnapshot>> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };

    match serde_json::from_slice::<RegistrySnapshot>(&bytes) {
        Ok(snap) if snap.version == RegistrySnapshot::CURRENT_VERSION => Ok(Some(snap)),
        Ok(snap) => {
            tracing::warn!(
                file = %path.display(),
                version = snap.version,
                expected = RegistrySnapshot::CURRENT_VERSION,
                "platform registry snapshot has unknown version; ignoring (first-boot)"
            );
            Ok(None)
        }
        Err(e) => {
            tracing::warn!(
                file = %path.display(),
                error = %e,
                "platform registry snapshot is unparseable; ignoring (first-boot)"
            );
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_snapshot() -> RegistrySnapshot {
        RegistrySnapshot::new(vec![
            InstanceRecord {
                address_raw: "bob[alice]".to_string(),
                organism: "bob".to_string(),
                thread_id: "inst-000001".to_string(),
                lifetime: LifetimeRecord::UntilIdle {
                    idle: Duration::from_secs(300),
                },
                parent_raw: None,
                cache_shards: vec!["shared.public".to_string(), "user.alice".to_string()],
            },
            InstanceRecord {
                address_raw: "scratch-bot[query-1]".to_string(),
                organism: "scratch-bot".to_string(),
                thread_id: "inst-000002".to_string(),
                lifetime: LifetimeRecord::Ephemeral,
                parent_raw: Some("bob[alice]".to_string()),
                cache_shards: vec![],
            },
        ])
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("registry.json");

        let snap = sample_snapshot();
        write_atomic(&path, &snap).unwrap();
        let loaded = read(&path).unwrap().expect("snapshot should load");

        assert_eq!(loaded, snap);
    }

    #[test]
    fn read_missing_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("does-not-exist.json");
        assert!(read(&path).unwrap().is_none());
    }

    #[test]
    fn read_corrupt_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("registry.json");
        fs::write(&path, b"{ not really json").unwrap();
        assert!(read(&path).unwrap().is_none());
    }

    #[test]
    fn read_wrong_version_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("registry.json");
        fs::write(&path, b"{\"version\": 99, \"instances\": []}").unwrap();
        assert!(read(&path).unwrap().is_none());
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("registry.json");

        write_atomic(&path, &RegistrySnapshot::new(vec![])).unwrap();
        write_atomic(&path, &sample_snapshot()).unwrap();

        let loaded = read(&path).unwrap().unwrap();
        assert_eq!(loaded.instances.len(), 2);
    }

    #[test]
    fn write_atomic_does_not_leave_tmp_when_successful() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("registry.json");
        write_atomic(&path, &sample_snapshot()).unwrap();

        let tmp = dir.path().join("registry.json.tmp");
        assert!(!tmp.exists(), "tmp file should be renamed away");
    }

    #[test]
    fn lifetime_record_round_trips_all_variants() {
        for lt in [
            Lifetime::UntilIdle(Duration::from_secs(42)),
            Lifetime::UntilTaskComplete,
            Lifetime::Pinned,
            Lifetime::Ephemeral,
        ] {
            let rec: LifetimeRecord = (&lt).into();
            let back: Lifetime = rec.into();
            assert_eq!(back, lt);
        }
    }
}
