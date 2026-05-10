//! Errors from cortex shim operations.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShimClientError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("not found: shim id `{0}` is not registered")]
    NotFound(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
}
