//! TF-IDF embedding provider — pure Rust, zero new dependencies.
//!
//! Tokenizes text, builds IDF from a corpus of semantic descriptions,
//! produces sparse TF-IDF vectors normalized to unit length for
//! cosine similarity via dot product.

use std::collections::HashMap;

use super::{Embedding, EmbeddingProvider};

/// Simple stop words to filter out common English words.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or", "for", "with", "this",
    "that", "be", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "shall", "not", "no",
    "but", "if", "at", "by", "from", "as", "into", "about", "up", "out", "so", "its", "you",
    "your", "i", "my", "we", "our", "they", "them", "their", "he", "she", "his", "her",
];

/// TF-IDF embedding provider.
///
/// Builds a vocabulary + IDF weights from a corpus of documents (semantic descriptions).
/// Embeds text into sparse TF-IDF vectors normalized to unit length.
#[derive(Clone)]
pub struct TfIdfProvider {
    /// term → dimension index
    vocabulary: HashMap<String, usize>,
    /// IDF weight per dimension
    idf: Vec<f32>,
    /// Total dimensions (vocabulary size)
    dims: usize,
}

impl TfIdfProvider {
    /// Build from a corpus of documents (semantic descriptions).
    ///
    /// Tokenizes all documents, builds a vocabulary, computes IDF weights.
    pub fn from_corpus(documents: &[&str]) -> Self {
        let n = documents.len() as f32;
        if documents.is_empty() {
            return Self {
                vocabulary: HashMap::new(),
                idf: Vec::new(),
                dims: 0,
            };
        }

        // Tokenize all documents and build vocabulary
        let tokenized: Vec<Vec<String>> = documents.iter().map(|d| tokenize(d)).collect();

        let mut vocabulary: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for tokens in &tokenized {
            // Unique terms in this document
            let unique: std::collections::HashSet<&str> =
                tokens.iter().map(|t| t.as_str()).collect();
            for term in unique {
                *doc_freq.entry(term.to_string()).or_insert(0) += 1;
                if !vocabulary.contains_key(term) {
                    let idx = vocabulary.len();
                    vocabulary.insert(term.to_string(), idx);
                }
            }
        }

        let dims = vocabulary.len();
        let mut idf = vec![0.0f32; dims];
        for (term, &idx) in &vocabulary {
            let df = *doc_freq.get(term).unwrap_or(&0) as f32;
            // Standard IDF: log(N / df) — smooth to avoid division by zero
            idf[idx] = (n / df.max(1.0)).ln() + 1.0;
        }

        Self {
            vocabulary,
            idf,
            dims,
        }
    }

    /// Rebuild from an updated corpus (hot-reload).
    pub fn rebuild(&mut self, documents: &[&str]) {
        let new = Self::from_corpus(documents);
        self.vocabulary = new.vocabulary;
        self.idf = new.idf;
        self.dims = new.dims;
    }
}

impl EmbeddingProvider for TfIdfProvider {
    fn embed(&self, text: &str) -> Embedding {
        if self.dims == 0 {
            return vec![];
        }

        let tokens = tokenize(text);
        let mut tf: HashMap<&str, f32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
        }

        let mut vector = vec![0.0f32; self.dims];
        for (term, &count) in &tf {
            if let Some(&idx) = self.vocabulary.get(*term) {
                // TF-IDF = term frequency × inverse document frequency
                vector[idx] = count * self.idf[idx];
            }
        }

        // Normalize to unit vector
        normalize(&mut vector);
        vector
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

/// Tokenize text: lowercase, split on non-alphanumeric, filter stop words.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .filter(|w| !STOP_WORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Normalize a vector to unit length (in-place).
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tfidf_from_corpus() {
        let docs = vec![
            "read files from the filesystem",
            "execute shell commands and return output",
            "search for code symbols using tree-sitter",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        assert!(provider.dims > 0);
        assert!(!provider.vocabulary.is_empty());
        assert_eq!(provider.idf.len(), provider.dims);
    }

    #[test]
    fn tfidf_embed_known_term() {
        let docs = vec!["read files from the filesystem"];
        let provider = TfIdfProvider::from_corpus(&docs);
        let embedding = provider.embed("read files");
        // Should have non-zero values for known terms
        assert!(embedding.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn tfidf_embed_unknown_term() {
        let docs = vec!["read files from the filesystem"];
        let provider = TfIdfProvider::from_corpus(&docs);
        let embedding = provider.embed("xyzzy quantum blockchain");
        // All zeros (or very close) for completely unknown terms
        let sum: f32 = embedding.iter().map(|v| v.abs()).sum();
        assert!(sum < f32::EPSILON);
    }

    #[test]
    fn tfidf_embed_dimensions() {
        let docs = vec![
            "read files from the filesystem",
            "execute shell commands",
        ];
        let provider = TfIdfProvider::from_corpus(&docs);
        let embedding = provider.embed("read something");
        assert_eq!(embedding.len(), provider.dimensions());
    }
}
