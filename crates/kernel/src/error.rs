//! Kernel-specific error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("WAL error: {0}")]
    Wal(String),

    #[error("WAL entry corrupted at offset {offset}: {reason}")]
    WalCorrupted { offset: u64, reason: String },

    #[error("thread not found: {0}")]
    ThreadNotFound(String),

    #[error("thread table full (capacity: {0})")]
    ThreadTableFull(u32),

    #[error("context not found: {0}")]
    ContextNotFound(String),

    #[error("journal entry not found: {0}")]
    JournalEntryNotFound(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid data: {0}")]
    InvalidData(String),
}

pub type KernelResult<T> = Result<T, KernelError>;
