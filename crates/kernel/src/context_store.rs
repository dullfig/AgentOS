//! Context store — per-thread context with named segments.
//!
//! Each thread gets a `ThreadContext` containing named segments (the "pages"
//! in our VMM metaphor). Segments can be Active (in working set) or Shelved
//! (in backing store). The librarian scores relevance and pages in/out.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{KernelError, KernelResult};
use crate::wal::{EntryType, WalEntry};

/// Status of a context segment — Active (working set) or Shelved (backing store).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentStatus {
    Active,
    Shelved,
    Folded,
}

/// A single context segment — the "page" in our VMM metaphor.
#[derive(Debug, Clone)]
pub struct ContextSegment {
    /// Unique ID within thread: "msg-001", "code:foo.rs", "map:crate"
    pub id: String,
    /// Type tag: "message", "code", "search-result", "codebase-map"
    pub tag: String,
    /// The actual data. For Folded segments: this is the SUMMARY.
    pub content: Vec<u8>,
    /// Active (in working set), Shelved (backing store), or Folded (compressed).
    pub status: SegmentStatus,
    /// Relevance score 0.0–1.0, scored by librarian, decays over time.
    pub relevance: f32,
    /// Creation timestamp (epoch millis).
    pub created_at: u64,
    /// Key into fold_store when status == Folded.
    pub fold_ref: Option<String>,
}

/// Metadata about a segment (for librarian inventory — no content).
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    pub id: String,
    pub tag: String,
    pub size: usize,
    pub status: SegmentStatus,
    pub relevance: f32,
    pub created_at: u64,
}

/// Inventory of all segments in a thread's context (metadata only).
#[derive(Debug, Clone)]
pub struct ContextInventory {
    pub thread_id: String,
    pub segments: Vec<SegmentMeta>,
    pub active_count: usize,
    pub shelved_count: usize,
    pub folded_count: usize,
    pub total_bytes: usize,
    pub active_bytes: usize,
}

/// Per-thread context container.
#[derive(Debug, Clone, Default)]
pub struct ThreadContext {
    pub segments: HashMap<String, ContextSegment>,
}

/// The context store — manages all thread contexts.
pub struct ContextStore {
    contexts: HashMap<String, ThreadContext>,
    /// Fold store: fold_ref → stashed full content for folded segments.
    pub(crate) fold_store: HashMap<String, Vec<u8>>,
    #[allow(dead_code)]
    base_dir: PathBuf,
}

impl ContextStore {
    /// Open or create the context store.
    pub fn open(base_dir: &Path) -> KernelResult<Self> {
        std::fs::create_dir_all(base_dir)?;
        Ok(Self {
            contexts: HashMap::new(),
            fold_store: HashMap::new(),
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Apply a WAL entry during replay.
    pub fn apply_wal_entry(&mut self, entry: &WalEntry) {
        match entry.entry_type {
            EntryType::ContextAllocate => {
                let thread_id = String::from_utf8_lossy(&entry.payload).to_string();
                self.contexts.entry(thread_id).or_default();
            }
            EntryType::ContextAppend => {
                // Legacy: map to add_segment with auto-generated ID.
                if let Some((thread_id, data)) = parse_append_payload(&entry.payload) {
                    let ctx = self.contexts.entry(thread_id).or_default();
                    let seg_id = format!("legacy-{}", ctx.segments.len());
                    ctx.segments.insert(
                        seg_id.clone(),
                        ContextSegment {
                            id: seg_id,
                            tag: "legacy".into(),
                            content: data,
                            status: SegmentStatus::Active,
                            relevance: 0.5,
                            created_at: now_millis(),
                            fold_ref: None,
                        },
                    );
                }
            }
            EntryType::ContextRelease => {
                let thread_id = String::from_utf8_lossy(&entry.payload).to_string();
                self.contexts.remove(&thread_id);
            }
            EntryType::ContextSegmentAdd => {
                if let Some((thread_id, seg)) = parse_segment_add_payload(&entry.payload) {
                    let ctx = self.contexts.entry(thread_id).or_default();
                    ctx.segments.insert(seg.id.clone(), seg);
                }
            }
            EntryType::ContextSegmentRemove => {
                if let Some((thread_id, seg_id)) = parse_two_part_payload(&entry.payload) {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        ctx.segments.remove(&seg_id);
                    }
                }
            }
            EntryType::ContextSegmentPageIn => {
                if let Some((thread_id, seg_id)) = parse_two_part_payload(&entry.payload) {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        if let Some(seg) = ctx.segments.get_mut(&seg_id) {
                            seg.status = SegmentStatus::Active;
                        }
                    }
                }
            }
            EntryType::ContextSegmentPageOut => {
                if let Some((thread_id, seg_id)) = parse_two_part_payload(&entry.payload) {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        if let Some(seg) = ctx.segments.get_mut(&seg_id) {
                            seg.status = SegmentStatus::Shelved;
                        }
                    }
                }
            }
            EntryType::ContextSegmentRelevance => {
                if let Some((thread_id, seg_id, rel)) = parse_relevance_payload(&entry.payload) {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        if let Some(seg) = ctx.segments.get_mut(&seg_id) {
                            seg.relevance = rel;
                        }
                    }
                }
            }
            EntryType::ContextFold => {
                // Payload: thread_id\0segment_id\0fold_ref\0summary_bytes
                if let Some((thread_id, seg_id, fold_ref, summary)) =
                    parse_fold_payload(&entry.payload)
                {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        if let Some(seg) = ctx.segments.get_mut(&seg_id) {
                            self.fold_store.insert(fold_ref.clone(), seg.content.clone());
                            seg.content = summary;
                            seg.status = SegmentStatus::Folded;
                            seg.fold_ref = Some(fold_ref);
                        }
                    }
                }
            }
            EntryType::ContextUnfold => {
                // Payload: thread_id\0segment_id
                if let Some((thread_id, seg_id)) = parse_two_part_payload(&entry.payload) {
                    if let Some(ctx) = self.contexts.get_mut(&thread_id) {
                        if let Some(seg) = ctx.segments.get_mut(&seg_id) {
                            if let Some(ref fr) = seg.fold_ref {
                                if let Some(original) = self.fold_store.remove(fr) {
                                    seg.content = original;
                                    seg.status = SegmentStatus::Active;
                                    seg.fold_ref = None;
                                }
                            }
                        }
                    }
                }
            }
            _ => {} // not a context op
        }
    }

    // ── Lifecycle ops (unchanged semantics) ──

    /// Create a context for a thread (allocate empty ThreadContext).
    pub fn create(&mut self, thread_id: &str) -> KernelResult<()> {
        self.contexts.entry(thread_id.to_string()).or_default();
        Ok(())
    }

    /// Build a WAL entry for context creation.
    pub fn wal_entry_create(thread_id: &str) -> WalEntry {
        WalEntry::new(EntryType::ContextAllocate, thread_id.as_bytes().to_vec())
    }

    /// Release (free) a thread's context — prune = free().
    pub fn release(&mut self, thread_id: &str) -> KernelResult<()> {
        self.contexts.remove(thread_id);
        Ok(())
    }

    /// Build a WAL entry for context release.
    pub fn wal_entry_release(thread_id: &str) -> WalEntry {
        WalEntry::new(EntryType::ContextRelease, thread_id.as_bytes().to_vec())
    }

    /// Check if a context exists for a thread.
    pub fn exists(&self, thread_id: &str) -> bool {
        self.contexts.contains_key(thread_id)
    }

    /// List all thread IDs that have contexts.
    pub fn all_thread_ids(&self) -> Vec<&str> {
        self.contexts.keys().map(|s| s.as_str()).collect()
    }

    /// Number of active contexts.
    pub fn count(&self) -> usize {
        self.contexts.len()
    }

    // ── Legacy compatibility ──

    /// Get flat context data (legacy — concatenates all active segment content).
    pub fn get(&self, thread_id: &str) -> Option<&ThreadContext> {
        self.contexts.get(thread_id)
    }

    /// Append data (legacy — creates a segment).
    pub fn append(&mut self, thread_id: &str, data: &[u8]) -> KernelResult<()> {
        let ctx = self
            .contexts
            .get_mut(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        let seg_id = format!("legacy-{}", ctx.segments.len());
        ctx.segments.insert(
            seg_id.clone(),
            ContextSegment {
                id: seg_id,
                tag: "legacy".into(),
                content: data.to_vec(),
                status: SegmentStatus::Active,
                relevance: 0.5,
                created_at: now_millis(),
                fold_ref: None,
            },
        );
        Ok(())
    }

    /// Build a WAL entry for context append (legacy).
    pub fn wal_entry_append(thread_id: &str, data: &[u8]) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(data);
        WalEntry::new(EntryType::ContextAppend, payload)
    }

    // ── Segment ops (Phase 3) ──

    /// Add a named segment to a thread's context.
    pub fn add_segment(&mut self, thread_id: &str, segment: ContextSegment) -> KernelResult<()> {
        let ctx = self
            .contexts
            .get_mut(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        ctx.segments.insert(segment.id.clone(), segment);
        Ok(())
    }

    /// Build a WAL entry for add_segment.
    pub fn wal_entry_segment_add(thread_id: &str, segment: &ContextSegment) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment.id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment.tag.as_bytes());
        payload.push(0);
        payload.push(segment.status as u8);
        payload.extend_from_slice(&segment.relevance.to_le_bytes());
        payload.extend_from_slice(&segment.created_at.to_le_bytes());
        payload.push(0);
        // For Folded segments, serialize fold_ref before content
        if segment.status == SegmentStatus::Folded {
            if let Some(ref fr) = segment.fold_ref {
                payload.extend_from_slice(fr.as_bytes());
            }
            payload.push(0);
        }
        payload.extend_from_slice(&segment.content);
        WalEntry::new(EntryType::ContextSegmentAdd, payload)
    }

    /// Remove a segment from a thread's context.
    pub fn remove_segment(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()> {
        let ctx = self
            .contexts
            .get_mut(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        ctx.segments.remove(segment_id);
        Ok(())
    }

    /// Build a WAL entry for remove_segment.
    pub fn wal_entry_segment_remove(thread_id: &str, segment_id: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        WalEntry::new(EntryType::ContextSegmentRemove, payload)
    }

    /// Page in: shelved → active.
    pub fn page_in(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()> {
        let seg = self.get_segment_mut(thread_id, segment_id)?;
        seg.status = SegmentStatus::Active;
        Ok(())
    }

    /// Build a WAL entry for page_in.
    pub fn wal_entry_page_in(thread_id: &str, segment_id: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        WalEntry::new(EntryType::ContextSegmentPageIn, payload)
    }

    /// Page out: active → shelved.
    pub fn page_out(&mut self, thread_id: &str, segment_id: &str) -> KernelResult<()> {
        let seg = self.get_segment_mut(thread_id, segment_id)?;
        seg.status = SegmentStatus::Shelved;
        Ok(())
    }

    /// Build a WAL entry for page_out.
    pub fn wal_entry_page_out(thread_id: &str, segment_id: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        WalEntry::new(EntryType::ContextSegmentPageOut, payload)
    }

    /// Update relevance score for a segment.
    pub fn update_relevance(
        &mut self,
        thread_id: &str,
        segment_id: &str,
        relevance: f32,
    ) -> KernelResult<()> {
        let seg = self.get_segment_mut(thread_id, segment_id)?;
        seg.relevance = relevance;
        Ok(())
    }

    /// Build a WAL entry for update_relevance.
    pub fn wal_entry_relevance(thread_id: &str, segment_id: &str, relevance: f32) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(&relevance.to_le_bytes());
        WalEntry::new(EntryType::ContextSegmentRelevance, payload)
    }

    // ── Fold ops (Folding Contexts) ──

    /// Fold a segment: stash content in fold_store, replace with summary.
    /// Active/Shelved → Folded.
    pub fn fold(
        &mut self,
        thread_id: &str,
        segment_id: &str,
        summary: Vec<u8>,
    ) -> KernelResult<()> {
        let ctx = self.contexts.get_mut(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        let seg = ctx.segments.get_mut(segment_id)
            .ok_or_else(|| KernelError::ContextNotFound(format!("{thread_id}/{segment_id}")))?;
        if seg.status == SegmentStatus::Folded {
            return Err(KernelError::InvalidData(format!(
                "segment {segment_id} is already folded"
            )));
        }
        let fold_ref = format!("fold-{}-{}", thread_id, segment_id);
        self.fold_store.insert(fold_ref.clone(), seg.content.clone());
        seg.content = summary;
        seg.status = SegmentStatus::Folded;
        seg.fold_ref = Some(fold_ref);
        Ok(())
    }

    /// Build a WAL entry for fold.
    /// Payload: thread_id\0segment_id\0fold_ref\0summary_bytes
    pub fn wal_entry_fold(thread_id: &str, segment_id: &str, fold_ref: &str, summary: &[u8]) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(fold_ref.as_bytes());
        payload.push(0);
        payload.extend_from_slice(summary);
        WalEntry::new(EntryType::ContextFold, payload)
    }

    /// Unfold a segment: restore content from fold_store.
    /// Folded → Active.
    pub fn unfold(
        &mut self,
        thread_id: &str,
        segment_id: &str,
    ) -> KernelResult<()> {
        let ctx = self.contexts.get(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        let seg = ctx.segments.get(segment_id)
            .ok_or_else(|| KernelError::ContextNotFound(format!("{thread_id}/{segment_id}")))?;
        if seg.status != SegmentStatus::Folded {
            return Err(KernelError::InvalidData(format!(
                "segment {segment_id} is not folded (status: {:?})",
                seg.status
            )));
        }
        let fold_ref = seg.fold_ref.as_ref().ok_or_else(|| {
            KernelError::InvalidData(format!("folded segment {segment_id} has no fold_ref"))
        })?.clone();

        let original = self.fold_store.remove(&fold_ref).ok_or_else(|| {
            KernelError::InvalidData(format!("fold_ref {fold_ref} not found in fold_store"))
        })?;

        let ctx = self.contexts.get_mut(thread_id).unwrap();
        let seg = ctx.segments.get_mut(segment_id).unwrap();
        seg.content = original;
        seg.status = SegmentStatus::Active;
        seg.fold_ref = None;
        Ok(())
    }

    /// Build a WAL entry for unfold.
    /// Payload: thread_id\0segment_id
    pub fn wal_entry_unfold(thread_id: &str, segment_id: &str) -> WalEntry {
        let mut payload = Vec::new();
        payload.extend_from_slice(thread_id.as_bytes());
        payload.push(0);
        payload.extend_from_slice(segment_id.as_bytes());
        WalEntry::new(EntryType::ContextUnfold, payload)
    }

    /// Evict a fold: remove from fold_store, keep summary as Shelved.
    /// Folded → Shelved (summary only, full content lost).
    pub fn evict_fold(
        &mut self,
        thread_id: &str,
        segment_id: &str,
    ) -> KernelResult<()> {
        let ctx = self.contexts.get(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        let seg = ctx.segments.get(segment_id)
            .ok_or_else(|| KernelError::ContextNotFound(format!("{thread_id}/{segment_id}")))?;
        if seg.status != SegmentStatus::Folded {
            return Err(KernelError::InvalidData(format!(
                "segment {segment_id} is not folded"
            )));
        }
        let fold_ref_key = seg.fold_ref.clone();

        if let Some(ref fr) = fold_ref_key {
            self.fold_store.remove(fr);
        }
        let ctx = self.contexts.get_mut(thread_id).unwrap();
        let seg = ctx.segments.get_mut(segment_id).unwrap();
        seg.status = SegmentStatus::Shelved;
        seg.fold_ref = None;
        Ok(())
    }

    /// Number of entries in the fold store.
    pub fn fold_store_len(&self) -> usize {
        self.fold_store.len()
    }

    /// Get all Active segments sorted by relevance (highest first).
    pub fn get_working_set(&self, thread_id: &str) -> KernelResult<Vec<&ContextSegment>> {
        let ctx = self
            .contexts
            .get(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        let mut active: Vec<&ContextSegment> = ctx
            .segments
            .values()
            .filter(|s| s.status == SegmentStatus::Active)
            .collect();
        active.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(active)
    }

    /// Get metadata inventory of all segments (for the librarian).
    pub fn get_inventory(&self, thread_id: &str) -> KernelResult<ContextInventory> {
        let ctx = self
            .contexts
            .get(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;

        let mut segments = Vec::new();
        let mut active_count = 0;
        let mut shelved_count = 0;
        let mut folded_count = 0;
        let mut total_bytes = 0;
        let mut active_bytes = 0;

        for seg in ctx.segments.values() {
            let size = seg.content.len();
            total_bytes += size;
            match seg.status {
                SegmentStatus::Active => {
                    active_count += 1;
                    active_bytes += size;
                }
                SegmentStatus::Shelved => {
                    shelved_count += 1;
                }
                SegmentStatus::Folded => {
                    folded_count += 1;
                }
            }
            segments.push(SegmentMeta {
                id: seg.id.clone(),
                tag: seg.tag.clone(),
                size,
                status: seg.status,
                relevance: seg.relevance,
                created_at: seg.created_at,
            });
        }

        Ok(ContextInventory {
            thread_id: thread_id.to_string(),
            segments,
            active_count,
            shelved_count,
            folded_count,
            total_bytes,
            active_bytes,
        })
    }

    /// Get a segment by ID (read-only).
    pub fn get_segment(&self, thread_id: &str, segment_id: &str) -> KernelResult<&ContextSegment> {
        let ctx = self
            .contexts
            .get(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        ctx.segments
            .get(segment_id)
            .ok_or_else(|| KernelError::ContextNotFound(format!("{thread_id}/{segment_id}")))
    }

    // ── Internal ──

    fn get_segment_mut(
        &mut self,
        thread_id: &str,
        segment_id: &str,
    ) -> KernelResult<&mut ContextSegment> {
        let ctx = self
            .contexts
            .get_mut(thread_id)
            .ok_or_else(|| KernelError::ContextNotFound(thread_id.to_string()))?;
        ctx.segments
            .get_mut(segment_id)
            .ok_or_else(|| KernelError::ContextNotFound(format!("{thread_id}/{segment_id}")))
    }
}

// ── Payload parsing helpers ──

fn parse_append_payload(payload: &[u8]) -> Option<(String, Vec<u8>)> {
    let pos = payload.iter().position(|&b| b == 0)?;
    let thread_id = String::from_utf8_lossy(&payload[..pos]).to_string();
    let data = payload[pos + 1..].to_vec();
    Some((thread_id, data))
}

fn parse_two_part_payload(payload: &[u8]) -> Option<(String, String)> {
    let pos = payload.iter().position(|&b| b == 0)?;
    let part1 = String::from_utf8_lossy(&payload[..pos]).to_string();
    let part2 = String::from_utf8_lossy(&payload[pos + 1..]).to_string();
    Some((part1, part2))
}

fn parse_segment_add_payload(payload: &[u8]) -> Option<(String, ContextSegment)> {
    // Format: thread_id\0seg_id\0tag\0status(1byte)+relevance(4bytes)+created_at(8bytes)\0content
    let mut parts = Vec::new();
    let mut start = 0;

    // Parse first three null-delimited strings: thread_id, seg_id, tag
    for _ in 0..3 {
        let pos = payload[start..].iter().position(|&b| b == 0)? + start;
        parts.push(String::from_utf8_lossy(&payload[start..pos]).to_string());
        start = pos + 1;
    }

    if start + 13 > payload.len() {
        return None; // need 1 (status) + 4 (f32) + 8 (u64)
    }

    let status = match payload[start] {
        0 => SegmentStatus::Active,
        2 => SegmentStatus::Folded,
        _ => SegmentStatus::Shelved,
    };
    start += 1;

    let relevance = f32::from_le_bytes(payload[start..start + 4].try_into().ok()?);
    start += 4;

    let created_at = u64::from_le_bytes(payload[start..start + 8].try_into().ok()?);
    start += 8;

    // Skip null separator before content
    if start < payload.len() && payload[start] == 0 {
        start += 1;
    }

    // Parse fold_ref: if status is Folded, the next null-delimited string is the fold_ref
    let fold_ref = if status == SegmentStatus::Folded && start < payload.len() {
        if let Some(null_pos) = payload[start..].iter().position(|&b| b == 0) {
            let fr = String::from_utf8_lossy(&payload[start..start + null_pos]).to_string();
            start = start + null_pos + 1;
            Some(fr)
        } else {
            None
        }
    } else {
        None
    };

    let content = payload[start..].to_vec();

    Some((
        parts[0].clone(),
        ContextSegment {
            id: parts[1].clone(),
            tag: parts[2].clone(),
            content,
            status,
            relevance,
            created_at,
            fold_ref,
        },
    ))
}

fn parse_fold_payload(payload: &[u8]) -> Option<(String, String, String, Vec<u8>)> {
    // Format: thread_id\0segment_id\0fold_ref\0summary_bytes
    let mut parts = Vec::new();
    let mut start = 0;
    // Parse three null-delimited strings: thread_id, segment_id, fold_ref
    for _ in 0..3 {
        let pos = payload[start..].iter().position(|&b| b == 0)? + start;
        parts.push(String::from_utf8_lossy(&payload[start..pos]).to_string());
        start = pos + 1;
    }
    let summary = payload[start..].to_vec();
    Some((parts[0].clone(), parts[1].clone(), parts[2].clone(), summary))
}

fn parse_relevance_payload(payload: &[u8]) -> Option<(String, String, f32)> {
    // Format: thread_id\0seg_id\0relevance(4 bytes le f32)
    let first_null = payload.iter().position(|&b| b == 0)?;
    let thread_id = String::from_utf8_lossy(&payload[..first_null]).to_string();
    let rest = &payload[first_null + 1..];

    let second_null = rest.iter().position(|&b| b == 0)?;
    let seg_id = String::from_utf8_lossy(&rest[..second_null]).to_string();
    let rel_bytes = &rest[second_null + 1..];

    if rel_bytes.len() < 4 {
        return None;
    }
    let relevance = f32::from_le_bytes(rel_bytes[..4].try_into().ok()?);
    Some((thread_id, seg_id, relevance))
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_segment(id: &str, tag: &str, content: &[u8]) -> ContextSegment {
        ContextSegment {
            id: id.to_string(),
            tag: tag.to_string(),
            content: content.to_vec(),
            status: SegmentStatus::Active,
            relevance: 0.5,
            created_at: now_millis(),
            fold_ref: None,
        }
    }

    #[test]
    fn create_and_exists() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("thread-1").unwrap();
        assert!(store.exists("thread-1"));
    }

    #[test]
    fn release_frees() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("thread-1").unwrap();
        store.release("thread-1").unwrap();
        assert!(!store.exists("thread-1"));
    }

    #[test]
    fn add_and_get_segment() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();

        let seg = make_segment("code:foo.rs", "code", b"fn main() {}");
        store.add_segment("t1", seg).unwrap();

        let retrieved = store.get_segment("t1", "code:foo.rs").unwrap();
        assert_eq!(retrieved.content, b"fn main() {}");
        assert_eq!(retrieved.tag, "code");
    }

    #[test]
    fn add_segment_nonexistent_thread_fails() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        let seg = make_segment("s1", "msg", b"hello");
        let err = store.add_segment("nonexistent", seg).unwrap_err();
        assert!(matches!(err, KernelError::ContextNotFound(_)));
    }

    #[test]
    fn remove_segment() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store
            .add_segment("t1", make_segment("s1", "msg", b"data"))
            .unwrap();
        store.remove_segment("t1", "s1").unwrap();
        assert!(store.get_segment("t1", "s1").is_err());
    }

    #[test]
    fn page_in_and_page_out() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();

        let mut seg = make_segment("s1", "code", b"data");
        seg.status = SegmentStatus::Shelved;
        store.add_segment("t1", seg).unwrap();

        // Initially shelved
        assert_eq!(
            store.get_segment("t1", "s1").unwrap().status,
            SegmentStatus::Shelved
        );

        // Page in
        store.page_in("t1", "s1").unwrap();
        assert_eq!(
            store.get_segment("t1", "s1").unwrap().status,
            SegmentStatus::Active
        );

        // Page out
        store.page_out("t1", "s1").unwrap();
        assert_eq!(
            store.get_segment("t1", "s1").unwrap().status,
            SegmentStatus::Shelved
        );
    }

    #[test]
    fn update_relevance() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store
            .add_segment("t1", make_segment("s1", "msg", b"hello"))
            .unwrap();

        store.update_relevance("t1", "s1", 0.9).unwrap();
        let seg = store.get_segment("t1", "s1").unwrap();
        assert!((seg.relevance - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn working_set_sorted_by_relevance() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();

        let mut s1 = make_segment("s1", "msg", b"low");
        s1.relevance = 0.2;
        let mut s2 = make_segment("s2", "msg", b"high");
        s2.relevance = 0.9;
        let mut s3 = make_segment("s3", "msg", b"mid");
        s3.relevance = 0.5;
        let mut s4 = make_segment("s4", "code", b"shelved");
        s4.status = SegmentStatus::Shelved;

        store.add_segment("t1", s1).unwrap();
        store.add_segment("t1", s2).unwrap();
        store.add_segment("t1", s3).unwrap();
        store.add_segment("t1", s4).unwrap();

        let ws = store.get_working_set("t1").unwrap();
        assert_eq!(ws.len(), 3); // s4 is shelved
        assert_eq!(ws[0].id, "s2"); // highest relevance first
        assert_eq!(ws[1].id, "s3");
        assert_eq!(ws[2].id, "s1");
    }

    #[test]
    fn inventory_metadata() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();

        let mut shelved = make_segment("s2", "code", b"code data");
        shelved.status = SegmentStatus::Shelved;

        store
            .add_segment("t1", make_segment("s1", "msg", b"hello"))
            .unwrap();
        store.add_segment("t1", shelved).unwrap();

        let inv = store.get_inventory("t1").unwrap();
        assert_eq!(inv.thread_id, "t1");
        assert_eq!(inv.segments.len(), 2);
        assert_eq!(inv.active_count, 1);
        assert_eq!(inv.shelved_count, 1);
        assert_eq!(inv.total_bytes, 5 + 9);
        assert_eq!(inv.active_bytes, 5);
    }

    #[test]
    fn wal_replay_segment_add() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();

        // Allocate via WAL
        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));

        // Add segment via WAL
        let seg = make_segment("code:main.rs", "code", b"fn main() {}");
        let wal = ContextStore::wal_entry_segment_add("t1", &seg);
        store.apply_wal_entry(&wal);

        assert!(store.exists("t1"));
        let retrieved = store.get_segment("t1", "code:main.rs").unwrap();
        assert_eq!(retrieved.content, b"fn main() {}");
        assert_eq!(retrieved.tag, "code");
    }

    #[test]
    fn wal_replay_page_in_out() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();

        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));
        let seg = make_segment("s1", "msg", b"data");
        store.apply_wal_entry(&ContextStore::wal_entry_segment_add("t1", &seg));

        // Page out via WAL
        store.apply_wal_entry(&ContextStore::wal_entry_page_out("t1", "s1"));
        assert_eq!(
            store.get_segment("t1", "s1").unwrap().status,
            SegmentStatus::Shelved
        );

        // Page in via WAL
        store.apply_wal_entry(&ContextStore::wal_entry_page_in("t1", "s1"));
        assert_eq!(
            store.get_segment("t1", "s1").unwrap().status,
            SegmentStatus::Active
        );
    }

    #[test]
    fn wal_replay_segment_remove() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();

        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));
        let seg = make_segment("s1", "msg", b"data");
        store.apply_wal_entry(&ContextStore::wal_entry_segment_add("t1", &seg));
        store.apply_wal_entry(&ContextStore::wal_entry_segment_remove("t1", "s1"));

        assert!(store.get_segment("t1", "s1").is_err());
    }

    #[test]
    fn wal_replay_relevance() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();

        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));
        let seg = make_segment("s1", "msg", b"data");
        store.apply_wal_entry(&ContextStore::wal_entry_segment_add("t1", &seg));
        store.apply_wal_entry(&ContextStore::wal_entry_relevance("t1", "s1", 0.85));

        let retrieved = store.get_segment("t1", "s1").unwrap();
        assert!((retrieved.relevance - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn context_store_all_thread_ids() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("thread-a").unwrap();
        store.create("thread-b").unwrap();

        let ids = store.all_thread_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"thread-a"));
        assert!(ids.contains(&"thread-b"));
    }

    #[test]
    fn legacy_append_creates_segment() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.append("t1", b"hello ").unwrap();
        store.append("t1", b"world").unwrap();

        let ctx = store.get("t1").unwrap();
        assert_eq!(ctx.segments.len(), 2);
    }

    #[test]
    fn legacy_wal_replay() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();

        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));

        let mut payload = b"t1".to_vec();
        payload.push(0);
        payload.extend_from_slice(b"context data");
        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAppend, payload));

        let ctx = store.get("t1").unwrap();
        assert_eq!(ctx.segments.len(), 1);
        let seg = ctx.segments.values().next().unwrap();
        assert_eq!(seg.content, b"context data");

        // Release via WAL replay
        store.apply_wal_entry(&WalEntry::new(EntryType::ContextRelease, b"t1".to_vec()));
        assert!(!store.exists("t1"));
    }

    // ── Folding Context tests (Milestone 1) ──

    #[test]
    fn fold_active_segment() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"fn main() {}")).unwrap();

        store.fold("t1", "s1", b"[summary: main function]".to_vec()).unwrap();

        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Folded);
        assert_eq!(seg.content, b"[summary: main function]");
        assert!(seg.fold_ref.is_some());
    }

    #[test]
    fn fold_shelved_segment() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        let mut seg = make_segment("s1", "code", b"shelved data");
        seg.status = SegmentStatus::Shelved;
        store.add_segment("t1", seg).unwrap();

        store.fold("t1", "s1", b"[folded]".to_vec()).unwrap();

        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Folded);
        assert_eq!(seg.content, b"[folded]");
    }

    #[test]
    fn unfold_restores_content() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"original content")).unwrap();

        store.fold("t1", "s1", b"[summary]".to_vec()).unwrap();
        assert_eq!(store.get_segment("t1", "s1").unwrap().content, b"[summary]");

        store.unfold("t1", "s1").unwrap();

        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Active);
        assert_eq!(seg.content, b"original content");
        assert!(seg.fold_ref.is_none());
    }

    #[test]
    fn unfold_nonfolded_fails() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "msg", b"active")).unwrap();

        let err = store.unfold("t1", "s1").unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));

        // Also test shelved
        store.page_out("t1", "s1").unwrap();
        let err = store.unfold("t1", "s1").unwrap_err();
        assert!(matches!(err, KernelError::InvalidData(_)));
    }

    #[test]
    fn fold_ref_populated() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"data")).unwrap();

        store.fold("t1", "s1", b"[folded]".to_vec()).unwrap();

        let seg = store.get_segment("t1", "s1").unwrap();
        let fold_ref = seg.fold_ref.as_ref().unwrap();
        assert!(fold_ref.contains("t1"));
        assert!(fold_ref.contains("s1"));
    }

    #[test]
    fn fold_store_grows_on_fold() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"data1")).unwrap();
        store.add_segment("t1", make_segment("s2", "code", b"data2")).unwrap();

        assert_eq!(store.fold_store_len(), 0);
        store.fold("t1", "s1", b"[f1]".to_vec()).unwrap();
        assert_eq!(store.fold_store_len(), 1);
        store.fold("t1", "s2", b"[f2]".to_vec()).unwrap();
        assert_eq!(store.fold_store_len(), 2);
    }

    #[test]
    fn fold_store_shrinks_on_unfold() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"data")).unwrap();

        store.fold("t1", "s1", b"[folded]".to_vec()).unwrap();
        assert_eq!(store.fold_store_len(), 1);

        store.unfold("t1", "s1").unwrap();
        assert_eq!(store.fold_store_len(), 0);
    }

    #[test]
    fn get_inventory_folded_count() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"active")).unwrap();
        store.add_segment("t1", make_segment("s2", "msg", b"to fold")).unwrap();
        let mut shelved = make_segment("s3", "doc", b"shelved");
        shelved.status = SegmentStatus::Shelved;
        store.add_segment("t1", shelved).unwrap();

        store.fold("t1", "s2", b"[folded]".to_vec()).unwrap();

        let inv = store.get_inventory("t1").unwrap();
        assert_eq!(inv.active_count, 1);
        assert_eq!(inv.shelved_count, 1);
        assert_eq!(inv.folded_count, 1);
    }

    #[test]
    fn working_set_excludes_folded() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();
        store.add_segment("t1", make_segment("s1", "code", b"active")).unwrap();
        store.add_segment("t1", make_segment("s2", "msg", b"will fold")).unwrap();

        store.fold("t1", "s2", b"[folded]".to_vec()).unwrap();

        let ws = store.get_working_set("t1").unwrap();
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0].id, "s1");
    }

    #[test]
    fn wal_replay_fold() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));
        let seg = make_segment("s1", "code", b"original content");
        store.apply_wal_entry(&ContextStore::wal_entry_segment_add("t1", &seg));

        // Replay fold WAL entry
        let fold_wal = ContextStore::wal_entry_fold("t1", "s1", "fold-t1-s1", b"[summary]");
        store.apply_wal_entry(&fold_wal);

        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Folded);
        assert_eq!(seg.content, b"[summary]");
        assert_eq!(seg.fold_ref.as_deref(), Some("fold-t1-s1"));
        assert_eq!(store.fold_store_len(), 1);
    }

    #[test]
    fn wal_replay_unfold() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.apply_wal_entry(&WalEntry::new(EntryType::ContextAllocate, b"t1".to_vec()));
        let seg = make_segment("s1", "code", b"original");
        store.apply_wal_entry(&ContextStore::wal_entry_segment_add("t1", &seg));

        // Fold via WAL
        let fold_wal = ContextStore::wal_entry_fold("t1", "s1", "fold-t1-s1", b"[summary]");
        store.apply_wal_entry(&fold_wal);

        // Unfold via WAL
        let unfold_wal = ContextStore::wal_entry_unfold("t1", "s1");
        store.apply_wal_entry(&unfold_wal);

        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Active);
        assert_eq!(seg.content, b"original");
        assert!(seg.fold_ref.is_none());
        assert_eq!(store.fold_store_len(), 0);
    }

    #[test]
    fn fold_then_unfold_roundtrip() {
        let dir = TempDir::new().unwrap();
        let mut store = ContextStore::open(&dir.path().join("contexts")).unwrap();
        store.create("t1").unwrap();

        let original_content = b"fn complex_function() { /* lots of code */ }";
        store.add_segment("t1", make_segment("s1", "code", original_content)).unwrap();

        // Fold
        store.fold("t1", "s1", b"[summary: complex function]".to_vec()).unwrap();
        assert_eq!(store.get_segment("t1", "s1").unwrap().status, SegmentStatus::Folded);
        assert_ne!(store.get_segment("t1", "s1").unwrap().content, original_content);

        // Unfold
        store.unfold("t1", "s1").unwrap();
        let seg = store.get_segment("t1", "s1").unwrap();
        assert_eq!(seg.status, SegmentStatus::Active);
        assert_eq!(seg.content, original_content);
        assert!(seg.fold_ref.is_none());
        assert_eq!(store.fold_store_len(), 0);
    }
}
