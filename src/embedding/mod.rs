//! Embedding infrastructure for semantic routing.
//!
//! Pluggable embedding providers (TF-IDF today, ONNX tomorrow) produce
//! vectors from text. The `EmbeddingIndex` stores pre-embedded tool
//! descriptions and provides cosine similarity search.

pub mod tfidf;

/// A single embedding vector.
pub type Embedding = Vec<f32>;

/// Trait for embedding text into vectors. Pluggable — TF-IDF today, ONNX tomorrow.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a text string into a vector.
    fn embed(&self, text: &str) -> Embedding;
    /// Dimensionality of the embedding space.
    fn dimensions(&self) -> usize;
}

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Tool/listener name.
    pub name: String,
    /// Cosine similarity score [0.0, 1.0].
    pub score: f32,
}

/// Cosine similarity between two vectors.
///
/// For unit vectors, this is just the dot product. We compute the full
/// formula for safety (handles non-normalized vectors gracefully).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Index of pre-embedded tool descriptions for similarity search.
pub struct EmbeddingIndex {
    entries: Vec<(String, Embedding)>,
    threshold: f32,
}

impl EmbeddingIndex {
    /// Create a new index with a minimum similarity threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            threshold,
        }
    }

    /// Register a tool with its pre-computed embedding.
    pub fn register(&mut self, name: &str, embedding: Embedding) {
        // Replace existing entry with same name
        self.entries.retain(|(n, _)| n != name);
        self.entries.push((name.to_string(), embedding));
    }

    /// Remove a tool by name.
    pub fn remove(&mut self, name: &str) {
        self.entries.retain(|(n, _)| n != name);
    }

    /// Find the best match above threshold.
    pub fn search(&self, query: &Embedding) -> Option<MatchResult> {
        self.best_match(query, &[])
    }

    /// Find the best match above threshold, restricted to allowed tool names.
    ///
    /// If `allowed` is empty, no matches are returned (security: empty allow-list = no access).
    pub fn search_filtered(&self, query: &Embedding, allowed: &[String]) -> Option<MatchResult> {
        if allowed.is_empty() {
            return None;
        }
        self.best_match(query, allowed)
    }

    /// Return top K matches sorted by descending score (for debugging/observability).
    pub fn search_top_k(&self, query: &Embedding, k: usize) -> Vec<MatchResult> {
        let mut results: Vec<MatchResult> = self
            .entries
            .iter()
            .map(|(name, emb)| MatchResult {
                name: name.clone(),
                score: cosine_similarity(query, emb),
            })
            .filter(|r| r.score >= self.threshold)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Internal: find best match, optionally filtered by allowed list.
    /// If `allowed` is empty and this is called from `search()`, no filtering.
    /// If `allowed` is non-empty, only those names are candidates.
    fn best_match(&self, query: &Embedding, allowed: &[String]) -> Option<MatchResult> {
        let filter_active = !allowed.is_empty();

        self.entries
            .iter()
            .filter(|(name, _)| !filter_active || allowed.iter().any(|a| a == name))
            .map(|(name, emb)| MatchResult {
                name: name.clone(),
                score: cosine_similarity(query, emb),
            })
            .filter(|r| r.score >= self.threshold)
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::tfidf::TfIdfProvider;
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similar_texts() {
        let docs = vec![
            "search for files in the directory listing",
            "find files in directory and list contents",
            "compile the database schema migration tool",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        let a = provider.embed("search for files");
        let b = provider.embed("find files in directory");
        let score = cosine_similarity(&a, &b);
        // These should be somewhat similar (shared vocabulary)
        assert!(score > 0.1, "expected similar texts to have score > 0.1, got {score}");
    }

    #[test]
    fn cosine_dissimilar_texts() {
        let docs = vec![
            "search for files in the directory listing",
            "find files in directory and list contents",
            "compile the database schema migration tool",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        let a = provider.embed("search for files");
        let b = provider.embed("compile the database");
        let similar = cosine_similarity(
            &provider.embed("search for files"),
            &provider.embed("find files in directory"),
        );
        let dissimilar = cosine_similarity(&a, &b);
        // Dissimilar should be lower than similar
        assert!(
            dissimilar < similar,
            "expected dissimilar ({dissimilar}) < similar ({similar})"
        );
    }

    #[test]
    fn index_register_and_search() {
        let docs = vec![
            "read write manage files on the local filesystem source code configuration",
            "execute shell commands run programs compile code run tests",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        let mut index = EmbeddingIndex::new(0.1);
        index.register("file-ops", provider.embed(docs[0]));
        index.register("shell", provider.embed(docs[1]));

        let query = provider.embed("read the source code file");
        let result = index.search(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "file-ops");
    }

    #[test]
    fn index_search_no_match() {
        let docs = vec!["read files from the filesystem"];
        let provider = TfIdfProvider::from_corpus(&docs);
        let mut index = EmbeddingIndex::new(0.5);
        index.register("file-ops", provider.embed(docs[0]));

        // Query with completely unrelated terms — should be below threshold
        let query = provider.embed("xyzzy quantum blockchain");
        let result = index.search(&query);
        assert!(result.is_none());
    }

    #[test]
    fn index_search_top_k() {
        let docs = vec![
            "read write manage files on the local filesystem",
            "execute shell commands run programs",
            "search for code symbols tree-sitter indexing",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        let mut index = EmbeddingIndex::new(0.0); // low threshold to get all
        for (i, doc) in docs.iter().enumerate() {
            index.register(&format!("tool-{i}"), provider.embed(doc));
        }

        let query = provider.embed("search code files");
        let results = index.search_top_k(&query, 3);
        assert!(!results.is_empty());
        // Results should be sorted by descending score
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn index_remove() {
        let docs = vec!["read files from the filesystem"];
        let provider = TfIdfProvider::from_corpus(&docs);
        let mut index = EmbeddingIndex::new(0.1);
        index.register("file-ops", provider.embed(docs[0]));
        assert_eq!(index.len(), 1);

        index.remove("file-ops");
        assert_eq!(index.len(), 0);

        let query = provider.embed("read files");
        assert!(index.search(&query).is_none());
    }
}
