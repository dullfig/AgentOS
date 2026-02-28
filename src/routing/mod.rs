//! Semantic routing — embedding-based tool discovery.
//!
//! The router intercepts Opus's natural language output, matches it against
//! tool descriptions via embedding similarity, and dispatches invisibly.
//! No tool call ceremony. Just thought, and result.

pub mod form_filler;
pub mod local_engine;

use std::collections::HashMap;

use crate::embedding::{EmbeddingIndex, EmbeddingProvider};
use crate::organism::Organism;

use form_filler::{FormFillResult, FormFillStrategy};

/// Register all tools with semantic descriptions into the embedding index.
///
/// Iterates over the organism's listeners, embeds each `semantic_description`,
/// and registers the result in the index. Listeners without descriptions are skipped.
pub fn register_tools(
    index: &mut EmbeddingIndex,
    provider: &dyn EmbeddingProvider,
    organism: &Organism,
) {
    for listener in organism.listeners().values() {
        if let Some(ref desc) = listener.semantic_description {
            let embedding = provider.embed(desc);
            index.register(&listener.name, embedding);
        }
    }
}

/// Metadata for a registered tool (used by the router for form-filling).
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    pub description: String,
    pub xml_template: String,
    pub payload_tag: String,
}

/// What the router decided.
#[derive(Debug)]
pub enum RouteDecision {
    /// Text matched a tool — result available.
    ToolResult {
        tool_name: String,
        result_xml: String,
    },
    /// Text matched a tool but form-filling failed.
    ToolFailed {
        note: String,
    },
    /// No tool match — text is a response.
    Response,
}

/// The semantic router: binary fork between tool dispatch and response.
pub struct SemanticRouter {
    provider: Box<dyn EmbeddingProvider>,
    index: EmbeddingIndex,
    form_filler: Box<dyn FormFillStrategy>,
    /// Tool metadata: name → (description, XML template, payload tag)
    tool_metadata: HashMap<String, ToolMetadata>,
}

impl SemanticRouter {
    /// Create a new semantic router.
    pub fn new(
        provider: Box<dyn EmbeddingProvider>,
        index: EmbeddingIndex,
        form_filler: Box<dyn FormFillStrategy>,
        tool_metadata: HashMap<String, ToolMetadata>,
    ) -> Self {
        Self {
            provider,
            index,
            form_filler,
            tool_metadata,
        }
    }

    /// Route LLM output: tool call or response?
    ///
    /// `allowed_tools` pre-filters candidates by security profile.
    /// If `allowed_tools` is empty, no tool can match (structural impossibility).
    ///
    /// Binary fork:
    /// - Match above threshold → form-fill → ToolResult or ToolFailed
    /// - No match → Response
    pub async fn route(&self, text: &str, allowed_tools: &[String]) -> RouteDecision {
        if allowed_tools.is_empty() {
            return RouteDecision::Response;
        }

        // Embed the text
        let query = self.provider.embed(text);

        // Search filtered by security profile
        let match_result = self.index.search_filtered(&query, allowed_tools);

        match match_result {
            Some(m) => {
                // Match found — try to fill the form
                if let Some(meta) = self.tool_metadata.get(&m.name) {
                    let fill_result = self
                        .form_filler
                        .fill(
                            text,
                            &m.name,
                            &meta.description,
                            &meta.xml_template,
                            &meta.payload_tag,
                        )
                        .await;

                    match fill_result {
                        FormFillResult::Success {
                            tool_name,
                            filled_xml,
                        } => RouteDecision::ToolResult {
                            tool_name,
                            result_xml: filled_xml,
                        },
                        FormFillResult::Failed {
                            tool_name,
                            last_error,
                        } => RouteDecision::ToolFailed {
                            note: format!(
                                "Could not extract parameters for {tool_name}: {last_error}"
                            ),
                        },
                    }
                } else {
                    // Tool matched but no metadata — shouldn't happen, treat as response
                    RouteDecision::Response
                }
            }
            None => RouteDecision::Response,
        }
    }

    /// Register tool metadata.
    pub fn register_metadata(&mut self, name: &str, metadata: ToolMetadata) {
        self.tool_metadata.insert(name.to_string(), metadata);
    }

    /// Get a reference to the embedding index.
    pub fn index(&self) -> &EmbeddingIndex {
        &self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use crate::embedding::tfidf::TfIdfProvider;
    use crate::embedding::EmbeddingIndex;
    use crate::llm::LlmPool;
    use crate::organism::parser::parse_organism;
    use form_filler::CloudFormFiller;

    fn routing_organism() -> Organism {
        let yaml = r#"
organism:
  name: routing-test

listeners:
  - name: file-ops
    payload_class: tools.FileOpsRequest
    handler: tools.file_ops.handle
    description: "File operations"
    semantic_description: |
      This tool reads, writes, and manages files on the local filesystem.
      Use it when you need to examine source code, read configuration files,
      write new files, create directories, or check if files exist.

  - name: shell
    payload_class: tools.ShellRequest
    handler: tools.shell.handle
    description: "Shell execution"
    semantic_description: |
      This tool executes shell commands and returns their output.
      Use it when you need to run programs, compile code, run tests,
      check system state, or execute any command-line operation.

  - name: coding-agent
    payload_class: agent.AgentTask
    handler: agent.handle
    description: "Opus coding agent"
    agent: true

profiles:
  admin:
    linux_user: agentos-admin
    listeners: [file-ops, shell, coding-agent]
    journal: retain_forever
  restricted:
    linux_user: agentos-restricted
    listeners: [file-ops]
    journal: prune_on_delivery
"#;
        parse_organism(yaml).unwrap()
    }

    fn corpus_from_organism(org: &Organism) -> Vec<String> {
        org.listeners()
            .values()
            .filter_map(|l| l.semantic_description.clone())
            .collect()
    }

    fn mock_pool() -> Arc<Mutex<LlmPool>> {
        Arc::new(Mutex::new(LlmPool::with_base_url(
            "test-key".into(),
            "haiku",
            "http://localhost:19999".into(),
        )))
    }

    fn build_test_router(threshold: f32) -> (SemanticRouter, TfIdfProvider) {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);

        let mut index = EmbeddingIndex::new(threshold);
        register_tools(&mut index, &provider, &org);

        let pool = mock_pool();
        let filler = CloudFormFiller::new(pool, 3);

        let mut metadata = HashMap::new();
        metadata.insert(
            "file-ops".to_string(),
            ToolMetadata {
                description: "File operations tool".into(),
                xml_template: "<FileOpsRequest><action/><path/></FileOpsRequest>".into(),
                payload_tag: "FileOpsRequest".into(),
            },
        );
        metadata.insert(
            "shell".to_string(),
            ToolMetadata {
                description: "Shell execution tool".into(),
                xml_template: "<ShellRequest><command/></ShellRequest>".into(),
                payload_tag: "ShellRequest".into(),
            },
        );

        let router = SemanticRouter::new(Box::new(provider.clone()), index, Box::new(filler), metadata);
        (router, provider)
    }

    // ── M2 Tests (register_tools) ──

    #[test]
    fn register_tools_into_index() {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);
        let mut index = EmbeddingIndex::new(0.1);

        register_tools(&mut index, &provider, &org);

        // file-ops and shell have semantic descriptions → registered
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn register_tools_skips_no_description() {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);
        let mut index = EmbeddingIndex::new(0.1);

        register_tools(&mut index, &provider, &org);

        // coding-agent has no semantic_description → not registered
        assert_eq!(index.len(), 2);

        let query = provider.embed("I need to think about this problem");
        let result = index.search(&query);
        if let Some(r) = result {
            assert_ne!(r.name, "coding-agent");
        }
    }

    #[test]
    fn hot_reload_updates_index() {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);
        let mut index = EmbeddingIndex::new(0.1);

        register_tools(&mut index, &provider, &org);
        assert_eq!(index.len(), 2);

        index.remove("shell");
        assert_eq!(index.len(), 1);

        register_tools(&mut index, &provider, &org);
        assert_eq!(index.len(), 2);
    }

    // ── M4 Tests (SemanticRouter) ──

    #[tokio::test]
    async fn route_matches_tool() {
        let (router, _) = build_test_router(0.05);
        let allowed = vec!["file-ops".to_string(), "shell".to_string()];

        // This text should match file-ops (contains "read", "file", "source code")
        let decision = router
            .route("I need to read the source code file at src/parser.rs", &allowed)
            .await;

        // The form-filler will fail (mock URL), so we should get ToolFailed
        // The key assertion is that it didn't return Response — it matched a tool
        assert!(
            !matches!(decision, RouteDecision::Response),
            "expected tool match, got Response"
        );
    }

    #[tokio::test]
    async fn route_no_match() {
        let (router, _) = build_test_router(0.9); // Very high threshold
        let allowed = vec!["file-ops".to_string(), "shell".to_string()];

        // Generic philosophical text — should not match any tool at high threshold
        let decision = router
            .route("The meaning of life is to create meaning", &allowed)
            .await;

        assert!(matches!(decision, RouteDecision::Response));
    }

    #[tokio::test]
    async fn route_below_threshold() {
        let (router, _) = build_test_router(0.99); // Impossibly high threshold
        let allowed = vec!["file-ops".to_string(), "shell".to_string()];

        let decision = router.route("read a file", &allowed).await;
        // Even with matching terms, 0.99 threshold should not be reached
        assert!(matches!(decision, RouteDecision::Response));
    }

    #[tokio::test]
    async fn route_security_filtered() {
        let (router, _) = build_test_router(0.05);
        // Restricted profile: only file-ops allowed, not shell
        let allowed = vec!["file-ops".to_string()];

        // This text is about shell commands — but shell isn't allowed
        let decision = router
            .route("execute the compile command run tests", &allowed)
            .await;

        // If it matches anything, it should match file-ops (the only allowed tool),
        // not shell. Or it might match nothing if the threshold is too high for file-ops.
        match &decision {
            RouteDecision::ToolResult { tool_name, .. } => {
                // Whatever matched, it shouldn't be shell (it's filtered out)
                assert_ne!(tool_name, "shell");
            }
            RouteDecision::ToolFailed { .. } => {
                // Tool matched (not shell, since it's filtered), but form-fill failed
            }
            RouteDecision::Response => {
                // Also acceptable — no match above threshold for allowed tools
            }
        }
    }

    #[tokio::test]
    async fn route_tool_failed() {
        let (router, _) = build_test_router(0.05);
        let allowed = vec!["file-ops".to_string(), "shell".to_string()];

        // Should match a tool, but form-filler will fail (mock URL)
        let decision = router
            .route("read the filesystem source code files configuration", &allowed)
            .await;

        // With mock URL, form-filling fails → ToolFailed
        assert!(
            matches!(decision, RouteDecision::ToolFailed { .. }),
            "expected ToolFailed with mock pool, got {:?}",
            decision
        );
    }

    #[test]
    fn route_decision_variants() {
        let result = RouteDecision::ToolResult {
            tool_name: "file-ops".into(),
            result_xml: "<result>data</result>".into(),
        };
        assert!(matches!(result, RouteDecision::ToolResult { .. }));

        let failed = RouteDecision::ToolFailed {
            note: "Could not fill params".into(),
        };
        assert!(matches!(failed, RouteDecision::ToolFailed { .. }));

        let response = RouteDecision::Response;
        assert!(matches!(response, RouteDecision::Response));
    }

    #[test]
    fn search_filtered_respects_allowlist() {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);
        let mut index = EmbeddingIndex::new(0.05);
        register_tools(&mut index, &provider, &org);

        // Query matching shell-like terms
        let query = provider.embed("execute shell commands run programs compile");
        let allowed = vec!["file-ops".to_string()]; // Only file-ops allowed

        let result = index.search_filtered(&query, &allowed);
        // Should not match "shell" since it's not in the allowed list
        if let Some(r) = &result {
            assert_ne!(r.name, "shell");
        }
    }

    #[test]
    fn search_filtered_empty_allowlist() {
        let org = routing_organism();
        let corpus: Vec<String> = corpus_from_organism(&org);
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let provider = TfIdfProvider::from_corpus(&corpus_refs);
        let mut index = EmbeddingIndex::new(0.05);
        register_tools(&mut index, &provider, &org);

        let query = provider.embed("read files");
        let empty: Vec<String> = vec![];
        assert!(index.search_filtered(&query, &empty).is_none());
    }

    #[test]
    fn register_tool_metadata() {
        let (mut router, _) = build_test_router(0.3);
        router.register_metadata(
            "new-tool",
            ToolMetadata {
                description: "A new tool".into(),
                xml_template: "<NewToolRequest/>".into(),
                payload_tag: "NewToolRequest".into(),
            },
        );
        assert!(router.tool_metadata.contains_key("new-tool"));
    }

    #[tokio::test]
    async fn route_invisible_error_message() {
        let (router, _) = build_test_router(0.05);
        let allowed = vec!["file-ops".to_string(), "shell".to_string()];

        let decision = router
            .route("read the filesystem source code files configuration", &allowed)
            .await;

        if let RouteDecision::ToolFailed { note } = decision {
            // Error note should be natural language, not leak internals
            assert!(note.contains("Could not"));
            // Should not contain raw HTTP errors or stack traces
            assert!(!note.contains("panic"));
        }
    }
}
