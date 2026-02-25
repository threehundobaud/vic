//! Error types for vib3.

use crate::core::types::{PageId, Tier};

/// Result type alias for vib3 operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error type for the vib3 engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // ── Storage errors ───────────────────────────────────────────────
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid .vib3 file: {reason}")]
    InvalidFormat { reason: String },

    #[error("Page not found: {page:?}")]
    PageNotFound { page: PageId },

    #[error("Page transfer failed: {page:?} from {from_tier:?} to {to_tier:?}: {reason}")]
    TransferFailed {
        page: PageId,
        from_tier: Tier,
        to_tier: Tier,
        reason: String,
    },

    // ── Buffer pool errors ───────────────────────────────────────────
    #[error("Out of memory in {tier:?}: need {needed} bytes, have {available} bytes")]
    OutOfMemory {
        tier: Tier,
        needed: usize,
        available: usize,
    },

    #[error("Page {page:?} is pinned and cannot be evicted")]
    PagePinned { page: PageId },

    #[error("Tier {tier:?} is full ({used}/{capacity} slots)")]
    TierFull {
        tier: Tier,
        used: usize,
        capacity: usize,
    },

    // ── Decompression errors ────────────────────────────────────────
    #[error("Decompression failed: {msg}")]
    DecompressFailed { msg: String },

    // ── CUDA errors ──────────────────────────────────────────────────
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("No CUDA device available")]
    NoCudaDevice,

    #[error(
        "CUDA device {device} has insufficient VRAM: need {needed_mb} MB, have {available_mb} MB"
    )]
    InsufficientVram {
        device: i32,
        needed_mb: usize,
        available_mb: usize,
    },

    // ── Model errors ─────────────────────────────────────────────────
    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Model config error: {0}")]
    ConfigError(String),

    #[error("Weight conversion error: {0}")]
    ConversionError(String),

    // ── Index errors ─────────────────────────────────────────────────
    #[error("Vector index not available (model was converted without profiling)")]
    NoVectorIndex,

    #[error("Index build failed: {0}")]
    IndexBuildError(String),

    // ── Runtime errors ───────────────────────────────────────────────
    #[error("Engine not initialized")]
    NotInitialized,

    #[error("Generation cancelled")]
    Cancelled,

    #[error("Timeout waiting for page {page:?}: waited {waited_ms} ms")]
    Timeout { page: PageId, waited_ms: u64 },

    #[error("Context length exceeded: {requested} > {maximum}")]
    ContextLengthExceeded { requested: usize, maximum: usize },

    // ── Catch-all ────────────────────────────────────────────────────
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Whether this error is likely transient (worth retrying).
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Error::Io(_)
                | Error::TransferFailed { .. }
                | Error::Timeout { .. }
                | Error::OutOfMemory { .. }
        )
    }

    /// Whether this error indicates a misconfiguration (not worth retrying).
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            Error::InvalidFormat { .. }
                | Error::UnsupportedArchitecture(_)
                | Error::ConfigError(_)
                | Error::NoCudaDevice
                | Error::InsufficientVram { .. }
        )
    }
}
