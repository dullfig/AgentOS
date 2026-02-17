//! Librarian — intelligent context curation service.
//!
//! Uses Haiku to decide what to page in/out of the context store.
//! The "prefrontal cortex" — curates what Opus sees before it sees it.

pub mod handler;
pub mod prompt;

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::kernel::Kernel;
use crate::llm::types::Message;
use crate::llm::LlmPool;

/// Errors from librarian operations.
#[derive(Debug, thiserror::Error)]
pub enum LibrarianError {
    #[error("kernel error: {0}")]
    Kernel(#[from] crate::kernel::error::KernelError),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("parse error: {0}")]
    Parse(String),
}

/// Result of a curation pass.
#[derive(Debug)]
pub struct CurationResult {
    /// System prompt additions from curated context.
    pub system_context: Option<String>,
    /// Segment IDs that were paged in.
    pub paged_in: Vec<String>,
    /// Segment IDs that were paged out.
    pub paged_out: Vec<String>,
    /// Estimated tokens in the working set.
    pub working_set_tokens: usize,
}

/// The Librarian service — curates context before LLM calls.
pub struct Librarian {
    pool: Arc<Mutex<LlmPool>>,
    pub(crate) kernel: Arc<Mutex<Kernel>>,
    model: String,
}

impl Librarian {
    pub fn new(pool: Arc<Mutex<LlmPool>>, kernel: Arc<Mutex<Kernel>>) -> Self {
        Self {
            pool,
            kernel,
            model: "haiku".into(),
        }
    }

    /// Curate context for a thread before an LLM call.
    pub async fn curate(
        &self,
        thread_id: &str,
        incoming_messages: &[Message],
        token_budget: usize,
    ) -> Result<CurationResult, LibrarianError> {
        // Get the inventory
        let inventory = {
            let kernel = self.kernel.lock().await;
            kernel.contexts().get_inventory(thread_id)?
        };

        // If no segments, nothing to curate
        if inventory.segments.is_empty() {
            return Ok(CurationResult {
                system_context: None,
                paged_in: vec![],
                paged_out: vec![],
                working_set_tokens: 0,
            });
        }

        // Build the curation prompt for Haiku
        let haiku_prompt =
            prompt::build_curation_prompt(&inventory, incoming_messages, token_budget);

        // Call Haiku for curation decision
        let response_text = {
            let pool = self.pool.lock().await;
            let messages = vec![Message::text("user", &haiku_prompt)];
            let resp = pool
                .complete(
                    Some(&self.model),
                    messages,
                    1024,
                    Some(prompt::CURATION_SYSTEM),
                )
                .await
                .map_err(|e| LibrarianError::Llm(e.to_string()))?;
            resp.text().unwrap_or("").to_string()
        };

        // Parse the decision
        let decision =
            prompt::parse_curation_response(&response_text).map_err(LibrarianError::Parse)?;

        // Apply paging decisions to the context store
        let mut paged_in = Vec::new();
        let mut paged_out = Vec::new();
        {
            let mut kernel = self.kernel.lock().await;
            for seg_id in &decision.page_in {
                if kernel.contexts_mut().page_in(thread_id, seg_id).is_ok() {
                    paged_in.push(seg_id.clone());
                }
            }
            for seg_id in &decision.page_out {
                if kernel.contexts_mut().page_out(thread_id, seg_id).is_ok() {
                    paged_out.push(seg_id.clone());
                }
            }
        }

        // Build system context from working set
        let system_context = {
            let kernel = self.kernel.lock().await;
            let working_set = kernel.contexts().get_working_set(thread_id)?;
            if working_set.is_empty() {
                None
            } else {
                let mut ctx = String::new();
                for seg in &working_set {
                    if let Ok(text) = std::str::from_utf8(&seg.content) {
                        ctx.push_str(&format!("[{}: {}]\n{}\n\n", seg.tag, seg.id, text));
                    }
                }
                if ctx.is_empty() {
                    None
                } else {
                    Some(ctx)
                }
            }
        };

        // Estimate tokens (rough: 4 chars per token)
        let working_set_tokens = system_context.as_ref().map_or(0, |s| s.len() / 4);

        Ok(CurationResult {
            system_context,
            paged_in,
            paged_out,
            working_set_tokens,
        })
    }

    /// Score all segments for relevance given a query.
    pub async fn score_relevance(
        &self,
        thread_id: &str,
        query: &str,
    ) -> Result<Vec<(String, f32)>, LibrarianError> {
        let inventory = {
            let kernel = self.kernel.lock().await;
            kernel.contexts().get_inventory(thread_id)?
        };

        if inventory.segments.is_empty() {
            return Ok(vec![]);
        }

        // Build a scoring prompt
        let prompt_text = prompt::build_scoring_prompt(&inventory, query);

        let response_text = {
            let pool = self.pool.lock().await;
            let messages = vec![Message::text("user", &prompt_text)];
            let resp = pool
                .complete(
                    Some(&self.model),
                    messages,
                    512,
                    Some(prompt::SCORING_SYSTEM),
                )
                .await
                .map_err(|e| LibrarianError::Llm(e.to_string()))?;
            resp.text().unwrap_or("").to_string()
        };

        let scores =
            prompt::parse_scoring_response(&response_text).map_err(LibrarianError::Parse)?;

        // Apply scores to the context store
        {
            let mut kernel = self.kernel.lock().await;
            for (seg_id, score) in &scores {
                let _ = kernel
                    .contexts_mut()
                    .update_relevance(thread_id, seg_id, *score);
            }
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn now_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    #[tokio::test]
    async fn curate_empty_context() {
        let dir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let pool = LlmPool::with_base_url("test".into(), "haiku", "http://localhost:19999".into());

        let kernel = Arc::new(Mutex::new(kernel));
        let pool = Arc::new(Mutex::new(pool));

        // Create thread context
        {
            let mut k = kernel.lock().await;
            k.contexts_mut().create("t1").unwrap();
        }

        let lib = Librarian::new(pool, kernel);
        let result = lib.curate("t1", &[], 8000).await.unwrap();

        assert!(result.system_context.is_none());
        assert!(result.paged_in.is_empty());
        assert!(result.paged_out.is_empty());
        assert_eq!(result.working_set_tokens, 0);
    }

    #[tokio::test]
    async fn curate_nonexistent_thread_fails() {
        let dir = tempfile::TempDir::new().unwrap();
        let kernel = Kernel::open(&dir.path().join("data")).unwrap();
        let pool = LlmPool::with_base_url("test".into(), "haiku", "http://localhost:19999".into());

        let kernel = Arc::new(Mutex::new(kernel));
        let pool = Arc::new(Mutex::new(pool));

        let lib = Librarian::new(pool, kernel);
        let result = lib.curate("nonexistent", &[], 8000).await;
        assert!(result.is_err());
    }

    #[test]
    fn curation_result_fields() {
        let result = CurationResult {
            system_context: Some("test context".into()),
            paged_in: vec!["s1".into()],
            paged_out: vec!["s2".into()],
            working_set_tokens: 100,
        };
        assert_eq!(result.paged_in.len(), 1);
        assert_eq!(result.paged_out.len(), 1);
        assert_eq!(result.working_set_tokens, 100);
    }
}
