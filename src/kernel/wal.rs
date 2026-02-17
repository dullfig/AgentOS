//! Write-ahead log — append-only, CRC-checked, crash-recoverable.
//!
//! On-disk entry format:
//! ```text
//! [length: u32][crc32: u32][entry_type: u8][payload: &[u8]]
//! ```
//!
//! All mutations to kernel state flow through here first.
//! On crash recovery, replay all entries and apply any that
//! weren't reflected in the mmap'd state files.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher;

use super::error::{KernelError, KernelResult};

/// Discriminant byte for WAL entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryType {
    // Thread ops
    ThreadCreate = 1,
    ThreadExtend = 2,
    ThreadPrune = 3,
    ThreadCleanup = 4,

    // Context ops
    ContextAllocate = 10,
    ContextAppend = 11, // Legacy — replaced by segment ops in Phase 3
    ContextRelease = 12,

    // Context segment ops (Phase 3)
    ContextSegmentAdd = 13,
    ContextSegmentRemove = 14,
    ContextSegmentPageIn = 15,
    ContextSegmentPageOut = 16,
    ContextSegmentRelevance = 17,

    // Journal ops
    JournalDispatched = 20,
    JournalDelivered = 21,
    JournalFailed = 22,

    // Compound
    AtomicBatch = 50,
}

impl EntryType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::ThreadCreate),
            2 => Some(Self::ThreadExtend),
            3 => Some(Self::ThreadPrune),
            4 => Some(Self::ThreadCleanup),
            10 => Some(Self::ContextAllocate),
            11 => Some(Self::ContextAppend),
            12 => Some(Self::ContextRelease),
            13 => Some(Self::ContextSegmentAdd),
            14 => Some(Self::ContextSegmentRemove),
            15 => Some(Self::ContextSegmentPageIn),
            16 => Some(Self::ContextSegmentPageOut),
            17 => Some(Self::ContextSegmentRelevance),
            20 => Some(Self::JournalDispatched),
            21 => Some(Self::JournalDelivered),
            22 => Some(Self::JournalFailed),
            50 => Some(Self::AtomicBatch),
            _ => None,
        }
    }
}

/// A single WAL entry (in-memory representation).
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub entry_type: EntryType,
    pub payload: Vec<u8>,
}

impl WalEntry {
    pub fn new(entry_type: EntryType, payload: Vec<u8>) -> Self {
        Self {
            entry_type,
            payload,
        }
    }

    /// Serialize to on-disk format: [length: u32][crc32: u32][entry_type: u8][payload]
    fn to_bytes(&self) -> Vec<u8> {
        // length = 1 (entry_type) + payload.len()
        let content_len = 1 + self.payload.len();
        let mut buf = Vec::with_capacity(4 + 4 + content_len);

        // length
        buf.extend_from_slice(&(content_len as u32).to_le_bytes());

        // CRC over entry_type + payload
        let mut hasher = Hasher::new();
        hasher.update(&[self.entry_type as u8]);
        hasher.update(&self.payload);
        let crc = hasher.finalize();
        buf.extend_from_slice(&crc.to_le_bytes());

        // entry_type
        buf.push(self.entry_type as u8);

        // payload
        buf.extend_from_slice(&self.payload);

        buf
    }
}

/// Append-only write-ahead log with CRC integrity checks.
pub struct Wal {
    file: File,
    path: PathBuf,
}

impl Wal {
    /// Open or create a WAL file at the given path.
    pub fn open(path: &Path) -> KernelResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                KernelError::Wal(format!("failed to open WAL at {}: {e}", path.display()))
            })?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Append a single entry. Writes + fsync for durability.
    pub fn append(&mut self, entry: &WalEntry) -> KernelResult<()> {
        let bytes = entry.to_bytes();
        self.file.write_all(&bytes)?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Append multiple entries atomically as a batch.
    /// Wraps them in an AtomicBatch entry so replay treats them as all-or-nothing.
    pub fn append_batch(&mut self, entries: &[WalEntry]) -> KernelResult<()> {
        // Serialize each sub-entry into the batch payload
        let mut batch_payload = Vec::new();
        let count = entries.len() as u32;
        batch_payload.extend_from_slice(&count.to_le_bytes());
        for entry in entries {
            let bytes = entry.to_bytes();
            batch_payload.extend_from_slice(&bytes);
        }

        let batch = WalEntry::new(EntryType::AtomicBatch, batch_payload);
        let bytes = batch.to_bytes();
        self.file.write_all(&bytes)?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Replay all entries from the beginning of the WAL.
    /// Corrupted entries are skipped with a warning; subsequent entries are still read.
    pub fn replay(&self) -> KernelResult<Vec<WalEntry>> {
        let mut file = File::open(&self.path)
            .map_err(|e| KernelError::Wal(format!("failed to open WAL for replay: {e}")))?;

        let mut entries = Vec::new();
        let mut offset: u64 = 0;

        loop {
            // Read length (4 bytes)
            let mut len_buf = [0u8; 4];
            match file.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let content_len = u32::from_le_bytes(len_buf) as usize;

            // Read CRC (4 bytes)
            let mut crc_buf = [0u8; 4];
            match file.read_exact(&mut crc_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    tracing::warn!("WAL truncated at offset {offset} (incomplete header)");
                    break;
                }
                Err(e) => return Err(e.into()),
            }
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Read content (entry_type + payload)
            if content_len == 0 {
                tracing::warn!("WAL entry at offset {offset} has zero length, skipping");
                offset += 8;
                continue;
            }
            let mut content = vec![0u8; content_len];
            match file.read_exact(&mut content) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    tracing::warn!("WAL truncated at offset {offset} (incomplete payload)");
                    break;
                }
                Err(e) => return Err(e.into()),
            }

            // Verify CRC
            let mut hasher = Hasher::new();
            hasher.update(&content);
            let computed_crc = hasher.finalize();

            if computed_crc != stored_crc {
                tracing::warn!(
                    "WAL entry corrupted at offset {offset}: CRC mismatch (stored={stored_crc:#x}, computed={computed_crc:#x}), skipping"
                );
                offset += 8 + content_len as u64;
                continue;
            }

            // Parse entry type
            let type_byte = content[0];
            let payload = content[1..].to_vec();

            match EntryType::from_u8(type_byte) {
                Some(EntryType::AtomicBatch) => {
                    // Unpack batch into individual entries
                    match Self::unpack_batch(&payload) {
                        Ok(batch_entries) => entries.extend(batch_entries),
                        Err(e) => {
                            tracing::warn!("WAL batch at offset {offset} malformed: {e}, skipping");
                        }
                    }
                }
                Some(entry_type) => {
                    entries.push(WalEntry {
                        entry_type,
                        payload,
                    });
                }
                None => {
                    tracing::warn!(
                        "WAL entry at offset {offset}: unknown type {type_byte}, skipping"
                    );
                }
            }

            offset += 8 + content_len as u64;
        }

        Ok(entries)
    }

    /// Checkpoint: truncate the WAL after state files have been synced.
    /// The caller is responsible for ensuring all state is persisted before calling this.
    pub fn checkpoint(&mut self) -> KernelResult<()> {
        // Reopen the file truncated
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| KernelError::Wal(format!("failed to checkpoint WAL: {e}")))?;

        // Reopen in append mode for subsequent writes
        self.file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| KernelError::Wal(format!("failed to reopen WAL after checkpoint: {e}")))?;

        self.file.sync_data()?;
        Ok(())
    }

    /// Current WAL file size in bytes.
    pub fn size(&self) -> KernelResult<u64> {
        let mut file = self.file.try_clone()?;
        let pos = file.seek(SeekFrom::End(0))?;
        Ok(pos)
    }

    /// Unpack a batch payload into individual WalEntry values.
    fn unpack_batch(payload: &[u8]) -> KernelResult<Vec<WalEntry>> {
        if payload.len() < 4 {
            return Err(KernelError::InvalidData("batch too short for count".into()));
        }

        let count = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
        let mut entries = Vec::with_capacity(count);
        let mut pos = 4;

        for _ in 0..count {
            if pos + 8 > payload.len() {
                return Err(KernelError::InvalidData("batch entry truncated".into()));
            }

            let content_len =
                u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let stored_crc = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap());
            pos += 4;

            if pos + content_len > payload.len() {
                return Err(KernelError::InvalidData(
                    "batch entry payload truncated".into(),
                ));
            }

            let content = &payload[pos..pos + content_len];

            // Verify CRC of sub-entry
            let mut hasher = Hasher::new();
            hasher.update(content);
            let computed_crc = hasher.finalize();
            if computed_crc != stored_crc {
                return Err(KernelError::WalCorrupted {
                    offset: pos as u64,
                    reason: "CRC mismatch in batch sub-entry".into(),
                });
            }

            let type_byte = content[0];
            let entry_payload = content[1..].to_vec();

            let entry_type = EntryType::from_u8(type_byte).ok_or_else(|| {
                KernelError::InvalidData(format!("unknown entry type in batch: {type_byte}"))
            })?;

            entries.push(WalEntry {
                entry_type,
                payload: entry_payload,
            });

            pos += content_len;
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn wal_in_tmp(dir: &TempDir) -> Wal {
        Wal::open(&dir.path().join("test.wal")).unwrap()
    }

    #[test]
    fn write_and_replay() {
        let dir = TempDir::new().unwrap();
        {
            let mut wal = wal_in_tmp(&dir);
            wal.append(&WalEntry::new(
                EntryType::ThreadCreate,
                b"thread-1".to_vec(),
            ))
            .unwrap();
            wal.append(&WalEntry::new(
                EntryType::ContextAllocate,
                b"ctx-1".to_vec(),
            ))
            .unwrap();
            wal.append(&WalEntry::new(
                EntryType::JournalDispatched,
                b"msg-1".to_vec(),
            ))
            .unwrap();
        }

        // Reopen and replay
        let wal = Wal::open(&dir.path().join("test.wal")).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].entry_type, EntryType::ThreadCreate);
        assert_eq!(entries[0].payload, b"thread-1");
        assert_eq!(entries[1].entry_type, EntryType::ContextAllocate);
        assert_eq!(entries[2].entry_type, EntryType::JournalDispatched);
    }

    #[test]
    fn corrupt_entry_skipped() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write two valid entries
        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&WalEntry::new(EntryType::ThreadCreate, b"first".to_vec()))
                .unwrap();
            wal.append(&WalEntry::new(EntryType::ThreadCreate, b"second".to_vec()))
                .unwrap();
        }

        // Corrupt the CRC of the first entry (bytes 4..8)
        {
            let mut file = OpenOptions::new().write(true).open(&wal_path).unwrap();
            file.seek(SeekFrom::Start(4)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
        }

        // Replay — first entry skipped, second still readable
        let wal = Wal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, b"second");
    }

    #[test]
    fn checkpoint_truncates() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut wal = Wal::open(&wal_path).unwrap();
        wal.append(&WalEntry::new(EntryType::ThreadCreate, b"data".to_vec()))
            .unwrap();
        assert!(wal.size().unwrap() > 0);

        wal.checkpoint().unwrap();
        assert_eq!(wal.size().unwrap(), 0);

        // Can still write after checkpoint
        wal.append(&WalEntry::new(EntryType::ThreadExtend, b"more".to_vec()))
            .unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].entry_type, EntryType::ThreadExtend);
    }

    #[test]
    fn empty_wal_replays_empty() {
        let dir = TempDir::new().unwrap();
        let wal = wal_in_tmp(&dir);
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn batch_append_and_replay() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append_batch(&[
                WalEntry::new(EntryType::ThreadPrune, b"prune-1".to_vec()),
                WalEntry::new(EntryType::ContextRelease, b"release-1".to_vec()),
                WalEntry::new(EntryType::JournalDelivered, b"delivered-1".to_vec()),
            ])
            .unwrap();
        }

        let wal = Wal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].entry_type, EntryType::ThreadPrune);
        assert_eq!(entries[1].entry_type, EntryType::ContextRelease);
        assert_eq!(entries[2].entry_type, EntryType::JournalDelivered);
    }

    #[test]
    fn large_payload() {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");
        let big = vec![0xAB; 100_000];

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&WalEntry::new(EntryType::ContextAppend, big.clone()))
                .unwrap();
        }

        let wal = Wal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, big);
    }
}
