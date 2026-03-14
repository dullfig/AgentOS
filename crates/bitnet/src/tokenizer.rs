//! BPE tokenizer — loads vocabulary from GGUF metadata.
//!
//! Supports two BPE variants:
//! - **SentencePiece** (LLaMA): score-based merges, `▁` for spaces, `<0xHH>` byte fallback
//! - **GPT-2** (BitNet b1.58): merge-list based, byte-level Unicode mapping, regex pre-tokenization
//!
//! The variant is auto-detected from `tokenizer.ggml.model` metadata.

use std::collections::HashMap;

use crate::gguf::{GgufError, GgufFile};

/// Token type flags from GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    fn from_i32(v: i32) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal,
        }
    }
}

/// Which BPE variant this tokenizer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpeMode {
    /// SentencePiece BPE: score-based merges, `▁` for spaces.
    SentencePiece,
    /// GPT-2 BPE: merge-list based, byte-level Unicode mapping.
    Gpt2,
}

/// Pre-tokenizer type — controls how text is split before BPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizerType {
    /// Standard GPT-2 regex: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ...`
    Gpt2,
    /// LLaMA3/Falcon3 regex: case-insensitive contractions, 3-digit numbers.
    Llama3,
}

/// A BPE tokenizer loaded from GGUF metadata.
pub struct Tokenizer {
    /// Token ID → token string.
    vocab: Vec<String>,
    /// Token ID → merge score (SentencePiece) or 0.0 (GPT-2).
    scores: Vec<f32>,
    /// Token ID → token type.
    token_types: Vec<TokenType>,
    /// Token string → token ID (for encoding).
    token_to_id: HashMap<String, u32>,
    /// Byte fallback tokens: byte value → token ID.
    byte_to_token: [u32; 256],
    /// Beginning of sequence token ID.
    bos_token_id: u32,
    /// End of sequence token ID.
    eos_token_id: u32,
    /// BPE variant.
    mode: BpeMode,
    /// Pre-tokenizer type (how text is split before BPE).
    pre_type: PreTokenizerType,
    /// GPT-2 merge list: (left, right) → rank (lower = merge first).
    merge_ranks: HashMap<(String, String), u32>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF file metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        // Detect tokenizer model type
        let model_type = gguf
            .get_metadata("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();

        let mode = if model_type == "gpt2" {
            BpeMode::Gpt2
        } else {
            BpeMode::SentencePiece
        };

        // Detect pre-tokenizer type from tokenizer.ggml.pre metadata
        let pre_str = gguf
            .get_metadata("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let pre_type = match pre_str {
            "llama3" | "falcon3" | "llama-v3" | "llama-bpe" => PreTokenizerType::Llama3,
            _ => PreTokenizerType::Gpt2,
        };

        // Extract token strings
        let tokens_meta = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".into()))?;
        let tokens_arr = tokens_meta
            .as_array()
            .ok_or_else(|| GgufError::MetadataTypeMismatch {
                key: "tokenizer.ggml.tokens".into(),
                expected: "array",
            })?;

        let vocab: Vec<String> = tokens_arr
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let vocab_size = vocab.len();

        // Extract scores (may be absent for GPT-2)
        let scores: Vec<f32> =
            if let Some(scores_meta) = gguf.get_metadata("tokenizer.ggml.scores") {
                if let Some(arr) = scores_meta.as_array() {
                    arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect()
                } else {
                    vec![0.0; vocab_size]
                }
            } else {
                vec![0.0; vocab_size]
            };

        // Extract token types (optional)
        let token_types: Vec<TokenType> =
            if let Some(types_meta) = gguf.get_metadata("tokenizer.ggml.token_type") {
                if let Some(arr) = types_meta.as_array() {
                    arr.iter()
                        .map(|v| TokenType::from_i32(v.as_i32().unwrap_or(1)))
                        .collect()
                } else {
                    vec![TokenType::Normal; vocab_size]
                }
            } else {
                vec![TokenType::Normal; vocab_size]
            };

        // Extract merge list for GPT-2
        let merges: Vec<String> = if mode == BpeMode::Gpt2 {
            if let Some(merges_meta) = gguf.get_metadata("tokenizer.ggml.merges") {
                if let Some(arr) = merges_meta.as_array() {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Special tokens
        let bos_token_id = gguf
            .get_metadata("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);

        let eos_token_id = gguf
            .get_metadata("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        Self::from_parts_with_mode(
            vocab, scores, token_types, bos_token_id, eos_token_id, mode, pre_type, &merges,
        )
    }

    /// Build tokenizer from raw parts (SentencePiece mode, for testing).
    pub fn from_parts(
        vocab: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Result<Self, GgufError> {
        Self::from_parts_with_mode(
            vocab,
            scores,
            token_types,
            bos_token_id,
            eos_token_id,
            BpeMode::SentencePiece,
            PreTokenizerType::Gpt2,
            &[],
        )
    }

    /// Build tokenizer from raw parts (GPT-2 mode, for testing).
    pub fn from_parts_gpt2(
        vocab: Vec<String>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
        merges: &[String],
    ) -> Result<Self, GgufError> {
        let scores = vec![0.0; vocab.len()];
        Self::from_parts_with_mode(
            vocab,
            scores,
            token_types,
            bos_token_id,
            eos_token_id,
            BpeMode::Gpt2,
            PreTokenizerType::Gpt2,
            merges,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts_with_mode(
        vocab: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<TokenType>,
        bos_token_id: u32,
        eos_token_id: u32,
        mode: BpeMode,
        pre_type: PreTokenizerType,
        merges: &[String],
    ) -> Result<Self, GgufError> {
        // Build reverse lookup
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Build byte fallback table
        let mut byte_to_token = [u32::MAX; 256];
        match mode {
            BpeMode::SentencePiece => {
                // SentencePiece uses "<0xHH>" format
                for byte_val in 0..=255u8 {
                    let hex_token = format!("<0x{:02X}>", byte_val);
                    if let Some(&id) = token_to_id.get(&hex_token) {
                        byte_to_token[byte_val as usize] = id;
                    }
                }
            }
            BpeMode::Gpt2 => {
                // GPT-2 uses Unicode-mapped single characters for each byte
                for byte_val in 0..=255u8 {
                    let ch = gpt2_byte_to_char(byte_val);
                    let s: String = std::iter::once(ch).collect();
                    if let Some(&id) = token_to_id.get(&s) {
                        byte_to_token[byte_val as usize] = id;
                    }
                }
            }
        }

        // Build merge rank table for GPT-2
        let mut merge_ranks = HashMap::new();
        for (rank, merge_str) in merges.iter().enumerate() {
            if let Some((left, right)) = merge_str.split_once(' ') {
                merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
            }
        }

        Ok(Self {
            vocab,
            scores,
            token_types,
            token_to_id,
            byte_to_token,
            bos_token_id,
            eos_token_id,
            mode,
            pre_type,
            merge_ranks,
        })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// Beginning-of-sequence token ID.
    pub fn bos_token_id(&self) -> u32 { self.bos_token_id }

    /// End-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 { self.eos_token_id }

    /// Get token string by ID.
    pub fn token(&self, id: u32) -> &str { &self.vocab[id as usize] }

    /// Get token ID by string.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token type.
    pub fn token_type(&self, id: u32) -> TokenType {
        self.token_types[id as usize]
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos {
            tokens.push(self.bos_token_id);
        }
        if text.is_empty() {
            return tokens;
        }

        match self.mode {
            BpeMode::SentencePiece => self.encode_sentencepiece(text, &mut tokens),
            BpeMode::Gpt2 => self.encode_gpt2(text, &mut tokens),
        }

        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        match self.mode {
            BpeMode::SentencePiece => self.decode_sentencepiece(tokens),
            BpeMode::Gpt2 => self.decode_gpt2(tokens),
        }
    }

    // -----------------------------------------------------------------------
    // SentencePiece BPE
    // -----------------------------------------------------------------------

    fn encode_sentencepiece(&self, text: &str, tokens: &mut Vec<u32>) {
        // Prepend space, replace all spaces with ▁
        let text = format!(" {}", text).replace(' ', "\u{2581}");

        let mut symbols = self.sp_initial_tokenize(&text);
        self.sp_bpe_merge(&mut symbols);
        tokens.extend(symbols.iter().map(|s| s.token_id));
    }

    fn decode_sentencepiece(&self, tokens: &[u32]) -> String {
        let mut text = String::new();

        for &id in tokens {
            if id == self.bos_token_id || id == self.eos_token_id {
                continue;
            }
            let token_str = &self.vocab[id as usize];
            let token_type = self.token_types[id as usize];

            match token_type {
                TokenType::Byte => {
                    if let Some(byte_val) = parse_byte_token(token_str) {
                        text.push(byte_val as char);
                    }
                }
                TokenType::Control => {}
                _ => {
                    text.push_str(token_str);
                }
            }
        }

        text = text.replace('\u{2581}', " ");
        if text.starts_with(' ') {
            text = text[1..].to_string();
        }
        text
    }

    fn sp_initial_tokenize(&self, text: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&ch_str) {
                symbols.push(Symbol { text: ch_str, token_id: id });
            } else {
                let mut buf = [0u8; 4];
                let bytes = ch.encode_utf8(&mut buf);
                for b in bytes.bytes() {
                    let byte_id = self.byte_to_token[b as usize];
                    if byte_id != u32::MAX {
                        symbols.push(Symbol {
                            text: format!("<0x{:02X}>", b),
                            token_id: byte_id,
                        });
                    }
                }
            }
        }
        symbols
    }

    fn sp_bpe_merge(&self, symbols: &mut Vec<Symbol>) {
        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_score = f32::INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i].text, symbols[i + 1].text);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score < best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            let merged_text = format!("{}{}", symbols[best_idx].text, symbols[best_idx + 1].text);
            symbols[best_idx] = Symbol { text: merged_text, token_id: best_id };
            symbols.remove(best_idx + 1);
        }
    }

    // -----------------------------------------------------------------------
    // GPT-2 BPE
    // -----------------------------------------------------------------------

    fn encode_gpt2(&self, text: &str, tokens: &mut Vec<u32>) {
        // Pre-tokenize: split into words using the appropriate regex pattern
        let words = match self.pre_type {
            PreTokenizerType::Gpt2 => gpt2_pre_tokenize(text),
            PreTokenizerType::Llama3 => llama3_pre_tokenize(text),
        };

        for word in &words {
            // Convert each byte to its GPT-2 Unicode character
            let chars: Vec<String> = word
                .bytes()
                .map(|b| gpt2_byte_to_char(b).to_string())
                .collect();

            // Apply BPE merges
            let merged = self.gpt2_bpe_merge(chars);

            // Look up each merged piece in vocabulary
            for piece in &merged {
                if let Some(&id) = self.token_to_id.get(piece) {
                    tokens.push(id);
                } else {
                    // Byte-level fallback: shouldn't happen in GPT-2 (every byte is in vocab)
                    for b in piece.bytes() {
                        let byte_id = self.byte_to_token[b as usize];
                        if byte_id != u32::MAX {
                            tokens.push(byte_id);
                        }
                    }
                }
            }
        }
    }

    fn decode_gpt2(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in tokens {
            if id == self.bos_token_id || id == self.eos_token_id {
                continue;
            }
            let token_type = self.token_types[id as usize];
            if token_type == TokenType::Control {
                continue;
            }

            let token_str = &self.vocab[id as usize];
            // Map GPT-2 Unicode characters back to bytes
            for ch in token_str.chars() {
                bytes.push(gpt2_char_to_byte(ch));
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    fn gpt2_bpe_merge(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the pair with the lowest merge rank
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        symbols
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tokenizer(vocab={}, bos={}, eos={}, mode={:?})",
            self.vocab.len(),
            self.bos_token_id,
            self.eos_token_id,
            self.mode,
        )
    }
}

/// A symbol during BPE processing.
#[derive(Debug, Clone)]
struct Symbol {
    text: String,
    token_id: u32,
}

/// Parse a byte fallback token like "<0x41>" → Some(0x41).
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// GPT-2 byte ↔ Unicode mapping
// ---------------------------------------------------------------------------

/// GPT-2's byte→Unicode mapping. Printable ASCII and Latin-1 supplement map
/// to themselves; control chars and gaps map to U+0100 onwards.
fn gpt2_byte_to_char(b: u8) -> char {
    // Build the mapping: printable ranges stay put, gaps fill from U+0100
    let b = b as u32;
    let ch = match b {
        // ASCII printable: ! (33) through ~ (126)
        33..=126 => b,
        // Latin-1 supplement: ¡ (161) through ¬ (172)
        161..=172 => b,
        // Latin-1 supplement: ® (174) through ÿ (255)
        174..=255 => b,
        // Everything else: map to U+0100+
        _ => {
            // Count how many "direct" bytes come before this one
            let mut offset = 0u32;
            for i in 0..=255u32 {
                let is_direct = matches!(i, 33..=126 | 161..=172 | 174..=255);
                if i == b {
                    return char::from_u32(256 + offset).unwrap();
                }
                if !is_direct {
                    offset += 1;
                }
            }
            unreachable!()
        }
    };
    char::from_u32(ch).unwrap()
}

/// Reverse GPT-2 Unicode→byte mapping.
fn gpt2_char_to_byte(ch: char) -> u8 {
    let cp = ch as u32;
    // Direct mappings: if the codepoint falls in a "direct" range, it IS the byte
    if matches!(cp, 33..=126 | 161..=172 | 174..=255) {
        return cp as u8;
    }
    // Indirect: codepoints U+0100..U+013F map to the "gap" bytes
    if cp >= 256 {
        let offset = (cp - 256) as usize;
        let gap_bytes: Vec<u8> = (0..=255u8)
            .filter(|&b| !matches!(b as u32, 33..=126 | 161..=172 | 174..=255))
            .collect();
        if offset < gap_bytes.len() {
            return gap_bytes[offset];
        }
    }
    // Fallback: shouldn't happen with valid GPT-2 tokens
    b'?'
}

// ---------------------------------------------------------------------------
// GPT-2 pre-tokenization
// ---------------------------------------------------------------------------

/// Split text into words using a simplified GPT-2 regex pattern.
///
/// GPT-2 pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// We implement this without a regex crate via manual character-class matching.
fn gpt2_pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut i = 0;

    while i < n {
        // Try contractions first: 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < n {
            if let Some(contraction) = try_contraction(&chars, i) {
                result.push(contraction.0.to_string());
                i += contraction.1;
                continue;
            }
        }

        // Optional leading space + letters
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_alphabetic())
            || chars[i].is_alphabetic()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Optional leading space + digits
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_ascii_digit())
            || chars[i].is_ascii_digit()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Optional leading space + other (non-whitespace, non-letter, non-digit)
        if (chars[i] == ' ' && i + 1 < n && is_other(chars[i + 1]))
            || (chars[i] != ' ' && is_other(chars[i]))
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && is_other(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Whitespace (including lone spaces not followed by letter/digit/other)
        if chars[i].is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // If followed by non-whitespace, emit all but last whitespace separately
            // (GPT-2: `\s+(?!\S)|\s+` — trailing whitespace as one group, else split)
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Catch-all: single character
        result.push(chars[i].to_string());
        i += 1;
    }

    result
}

fn try_contraction(chars: &[char], i: usize) -> Option<(&'static str, usize)> {
    let remaining = chars.len() - i;
    if remaining >= 3 {
        let two: String = chars[i..i + 3].iter().collect();
        match two.to_lowercase().as_str() {
            "'re" => return Some(("'re", 3)),
            "'ve" => return Some(("'ve", 3)),
            "'ll" => return Some(("'ll", 3)),
            _ => {}
        }
    }
    if remaining >= 2 {
        let one: String = chars[i..i + 2].iter().collect();
        match one.to_lowercase().as_str() {
            "'s" => return Some(("'s", 2)),
            "'t" => return Some(("'t", 2)),
            "'m" => return Some(("'m", 2)),
            "'d" => return Some(("'d", 2)),
            _ => {}
        }
    }
    None
}

fn is_other(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_ascii_digit()
}

// ---------------------------------------------------------------------------
// LLaMA3 / Falcon3 pre-tokenization
// ---------------------------------------------------------------------------

/// Split text using the LLaMA3 regex pattern (used by Falcon3, LLaMA3, etc.).
///
/// Pattern: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// Key differences from GPT-2:
/// - Case-insensitive contractions (`'S`, `'T`, `'Re`, etc.)
/// - Numbers split into groups of 1-3 digits
/// - Unicode letter class (`\p{L}`) instead of just ASCII alphabetic
/// - Explicit `\r\n` handling in whitespace rules
fn llama3_pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut result = Vec::new();
    let mut i = 0;

    while i < n {
        // 1. Case-insensitive contractions: 's|'t|'re|'ve|'m|'ll|'d
        if (chars[i] == '\'' || chars[i] == '\u{2019}') && i + 1 < n {
            if let Some(contraction) = try_contraction_ci(&chars, i) {
                result.push(contraction.0);
                i += contraction.1;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+ — optional non-letter-digit-newline char + letters
        if chars[i].is_alphabetic() {
            let start = i;
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }
        if !chars[i].is_alphanumeric() && chars[i] != '\r' && chars[i] != '\n'
            && !chars[i].is_whitespace()
            && i + 1 < n && chars[i + 1].is_alphabetic()
        {
            let start = i;
            i += 1; // skip the non-letter-digit prefix char
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 3. \p{N}{1,3} — digits in groups of 1-3
        if chars[i].is_numeric() {
            let start = i;
            let mut count = 0;
            while i < n && chars[i].is_numeric() && count < 3 {
                i += 1;
                count += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 4. \s*[\r\n]+ — whitespace followed by newlines
        if chars[i] == '\r' || chars[i] == '\n' {
            let start = i;
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }
        if chars[i].is_whitespace() {
            // Look ahead: is there a \r\n coming?
            let start = i;
            let mut j = i;
            while j < n && chars[j].is_whitespace() && chars[j] != '\r' && chars[j] != '\n' {
                j += 1;
            }
            if j < n && (chars[j] == '\r' || chars[j] == '\n') {
                // \s*[\r\n]+
                i = j;
                while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                result.push(word);
                continue;
            }

            // 6. \s+(?!\S)|\s+ — trailing whitespace or whitespace before more content
            while i < n && chars[i].is_whitespace() && chars[i] != '\r' && chars[i] != '\n' {
                i += 1;
            }
            // If followed by non-whitespace, this is the \s+ branch
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // 5. " ?[^\s\p{L}\p{N}]+[\r\n]*" — optional space + non-alphanum-non-whitespace + optional newlines
        if is_other(chars[i]) {
            let start = i;
            while i < n && is_other(chars[i]) {
                i += 1;
            }
            // Consume trailing \r\n
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            result.push(word);
            continue;
        }

        // Catch-all: single character
        result.push(chars[i].to_string());
        i += 1;
    }

    result
}

/// Case-insensitive contraction matching for LLaMA3.
fn try_contraction_ci(chars: &[char], i: usize) -> Option<(String, usize)> {
    let remaining = chars.len() - i;
    if remaining >= 3 {
        let c1 = chars[i + 1].to_ascii_lowercase();
        let c2 = chars[i + 2].to_ascii_lowercase();
        match (c1, c2) {
            ('r', 'e') | ('v', 'e') | ('l', 'l') => {
                let s: String = chars[i..i + 3].iter().collect();
                return Some((s, 3));
            }
            _ => {}
        }
    }
    if remaining >= 2 {
        let c1 = chars[i + 1].to_ascii_lowercase();
        match c1 {
            's' | 't' | 'm' | 'd' => {
                let s: String = chars[i..i + 2].iter().collect();
                return Some((s, 2));
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // SentencePiece tests (existing)
    // =======================================================================

    fn make_test_tokenizer() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        let mut token_types = Vec::new();

        // 0: unknown
        vocab.push("<unk>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Unknown);

        // 1: BOS
        vocab.push("<s>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Control);

        // 2: EOS
        vocab.push("</s>".to_string());
        scores.push(0.0);
        token_types.push(TokenType::Control);

        // 3..258: byte tokens
        for b in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", b));
            scores.push(0.0);
            token_types.push(TokenType::Byte);
        }

        let extra_tokens: Vec<(&str, f32)> = vec![
            ("\u{2581}", 0.0),  // 259
            ("h", 0.0),         // 260
            ("e", 0.0),         // 261
            ("l", 0.0),         // 262
            ("o", 0.0),         // 263
            ("w", 0.0),         // 264
            ("r", 0.0),         // 265
            ("d", 0.0),         // 266
            ("he", 1.0),        // 267
            ("ll", 2.0),        // 268
            ("lo", 3.0),        // 269
            ("hel", 4.0),       // 270
            ("hell", 5.0),      // 271
            ("hello", 6.0),     // 272
            ("\u{2581}he", 7.0), // 273
            ("wo", 8.0),        // 274
            ("wor", 9.0),       // 275
            ("worl", 10.0),     // 276
            ("world", 11.0),    // 277
            ("\u{2581}world", 12.0), // 278
        ];

        for (text, score) in extra_tokens {
            vocab.push(text.to_string());
            scores.push(score);
            token_types.push(TokenType::Normal);
        }

        Tokenizer::from_parts(vocab, scores, token_types, 1, 2).unwrap()
    }

    #[test]
    fn construction() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.vocab_size(), 279);
        assert_eq!(tok.bos_token_id(), 1);
        assert_eq!(tok.eos_token_id(), 2);
    }

    #[test]
    fn token_lookup() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token(0), "<unk>");
        assert_eq!(tok.token(1), "<s>");
        assert_eq!(tok.token(272), "hello");
    }

    #[test]
    fn token_id_lookup() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token_id("hello"), Some(272));
        assert_eq!(tok.token_id("<s>"), Some(1));
        assert_eq!(tok.token_id("nonexistent"), None);
    }

    #[test]
    fn token_types() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.token_type(0), TokenType::Unknown);
        assert_eq!(tok.token_type(1), TokenType::Control);
        assert_eq!(tok.token_type(3), TokenType::Byte);
        assert_eq!(tok.token_type(272), TokenType::Normal);
    }

    #[test]
    fn byte_fallback_table() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.byte_to_token[0x41], 68);
        assert_eq!(tok.byte_to_token[0x00], 3);
        assert_eq!(tok.byte_to_token[0xFF], 258);
    }

    #[test]
    fn encode_hello() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", false);
        assert_eq!(tokens, vec![259, 272]);
    }

    #[test]
    fn encode_with_bos() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", true);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn encode_empty() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.encode("", false), Vec::<u32>::new());
        assert_eq!(tok.encode("", true), vec![1]);
    }

    #[test]
    fn encode_hello_world() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello world", false);
        assert_eq!(tokens, vec![259, 272, 278]);
    }

    #[test]
    fn decode_hello() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[259, 272]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn decode_skips_bos_eos() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[1, 259, 272, 2]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn decode_byte_fallback() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[68]);
        assert_eq!(text, "A");
    }

    #[test]
    fn roundtrip_hello() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello", false);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn roundtrip_hello_world() {
        let tok = make_test_tokenizer();
        let tokens = tok.encode("hello world", false);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn parse_byte_token_valid() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
    }

    #[test]
    fn parse_byte_token_invalid() {
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
        assert_eq!(parse_byte_token("<0x0>"), None);
    }

    #[test]
    fn debug_format() {
        let tok = make_test_tokenizer();
        let debug = format!("{:?}", tok);
        assert!(debug.contains("Tokenizer"));
        assert!(debug.contains("vocab=279"));
    }

    // =======================================================================
    // GPT-2 byte mapping tests
    // =======================================================================

    #[test]
    fn gpt2_byte_char_roundtrip() {
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            let back = gpt2_char_to_byte(ch);
            assert_eq!(b, back, "roundtrip failed for byte {b}: char={ch}");
        }
    }

    #[test]
    fn gpt2_printable_ascii_identity() {
        // Printable ASCII (33-126) maps to itself
        for b in 33..=126u8 {
            let ch = gpt2_byte_to_char(b);
            assert_eq!(ch as u32, b as u32, "byte {b} should map to itself");
        }
    }

    #[test]
    fn gpt2_space_maps_to_unicode() {
        // Space (32) is NOT in the direct range, should map to U+0100+
        let ch = gpt2_byte_to_char(b' ');
        assert!(ch as u32 >= 256, "space should map above U+00FF, got U+{:04X}", ch as u32);
    }

    #[test]
    fn gpt2_all_chars_unique() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            assert!(seen.insert(ch), "duplicate char for byte {b}");
        }
    }

    // =======================================================================
    // GPT-2 pre-tokenization tests
    // =======================================================================

    #[test]
    fn gpt2_pretokenize_simple() {
        let words = gpt2_pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", " world"]);
    }

    #[test]
    fn gpt2_pretokenize_contractions() {
        let words = gpt2_pre_tokenize("I'm don't");
        // "I" "'m" " don" "'t"
        assert_eq!(words.len(), 4);
        assert_eq!(words[1], "'m");
        assert_eq!(words[3], "'t");
    }

    #[test]
    fn gpt2_pretokenize_numbers() {
        let words = gpt2_pre_tokenize("test 123 abc");
        assert!(words.contains(&" 123".to_string()));
    }

    #[test]
    fn gpt2_pretokenize_punctuation() {
        let words = gpt2_pre_tokenize("hello, world!");
        // "hello" "," " world" "!"
        assert!(words.contains(&",".to_string()));
        assert!(words.contains(&"!".to_string()));
    }

    // =======================================================================
    // GPT-2 BPE tests
    // =======================================================================

    /// Build a minimal GPT-2 tokenizer for testing.
    fn make_gpt2_tokenizer() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut token_types = Vec::new();

        // Token 0: BOS (control)
        vocab.push("<|endoftext|>".to_string());
        token_types.push(TokenType::Control);

        // Tokens 1-256: single-byte tokens (GPT-2 Unicode mapping)
        for b in 0..=255u8 {
            let ch = gpt2_byte_to_char(b);
            vocab.push(ch.to_string());
            token_types.push(TokenType::Normal);
        }

        // Token 257: "he" (merge of "h" + "e")
        vocab.push("he".to_string());
        token_types.push(TokenType::Normal);

        // Token 258: "ll" (merge of "l" + "l")
        vocab.push("ll".to_string());
        token_types.push(TokenType::Normal);

        // Token 259: "lo" (merge of "l" + "o")
        vocab.push("lo".to_string());
        token_types.push(TokenType::Normal);

        // Token 260: "hel" (merge of "he" + "l")
        vocab.push("hel".to_string());
        token_types.push(TokenType::Normal);

        // Token 261: "hell" (merge of "hel" + "l")
        vocab.push("hell".to_string());
        token_types.push(TokenType::Normal);

        // Token 262: "hello" (merge of "hell" + "o")
        vocab.push("hello".to_string());
        token_types.push(TokenType::Normal);

        // Merge list (order = priority)
        let merges = vec![
            "h e".to_string(),     // rank 0: h+e → he
            "l l".to_string(),     // rank 1: l+l → ll
            "l o".to_string(),     // rank 2: l+o → lo
            "he l".to_string(),    // rank 3: he+l → hel
            "hel l".to_string(),   // rank 4: hel+l → hell
            "hell o".to_string(),  // rank 5: hell+o → hello
        ];

        // BOS=0, EOS=0 (simplified)
        Tokenizer::from_parts_gpt2(vocab, token_types, 0, 0, &merges).unwrap()
    }

    #[test]
    fn gpt2_encode_hello() {
        let tok = make_gpt2_tokenizer();
        let tokens = tok.encode("hello", false);
        // "hello" → pre-tokenize → ["hello"]
        // bytes: h(104) e(101) l(108) l(108) o(111) → GPT-2 chars: same (all printable ASCII)
        // BPE merges: h+e→he, l+l→ll, he+l→hel, hel+l→hell... wait.
        // Initial: ["h", "e", "l", "l", "o"]
        // Rank 0 (h+e): ["he", "l", "l", "o"]
        // Rank 1 (l+l): ["he", "ll", "o"]
        // No l+o pair (it's "ll" now). Rank 3 (he+l): need "he"+"l", but we have "he"+"ll", not a match.
        // Actually rank 3 is "he"+"l". The symbols are ["he", "ll", "o"]. The pair (he, ll) != (he, l).
        // So no more merges! Result: ["he", "ll", "o"]
        // That's token IDs for "he"(257), "ll"(258), "o"(1 + 'o' offset)
        // 'o' is byte 111, and since 111 is in 33..=126, gpt2_byte_to_char(111) = 'o'
        // The single-byte tokens start at index 1. So 'o' is at index 1 + 111 = 112? No.
        // Token index = 1 + byte_value. Token for byte 0 is index 1, byte 1 is index 2, etc.
        // Wait, I iterate 0..=255 starting at token 1. So byte 0 → token 1, byte 111 → token 112.
        // Actually no. Let me re-check. We push byte 0's char at index 1, byte 1 at index 2, ..., byte 255 at index 256.
        // So byte 111 ('o') → token 112.
        let o_token = 1 + 111; // = 112
        assert_eq!(tokens, vec![257, 258, o_token]);
    }

    #[test]
    fn gpt2_encode_with_space() {
        let tok = make_gpt2_tokenizer();
        let tokens = tok.encode("hi there", false);
        // Pre-tokenize: ["hi", " there"]
        // Both words get byte-encoded, and since we have limited merges, most stay as bytes
        assert!(!tokens.is_empty());
        // Decode should roundtrip
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "hi there");
    }

    #[test]
    fn gpt2_roundtrip_ascii() {
        let tok = make_gpt2_tokenizer();
        for text in &["hello", "test", "a b c", "123"] {
            let tokens = tok.encode(text, false);
            let decoded = tok.decode(&tokens);
            assert_eq!(&decoded, text, "roundtrip failed for {:?}", text);
        }
    }

    #[test]
    fn gpt2_merge_priority() {
        let tok = make_gpt2_tokenizer();
        // "hell" should merge: h+e→he, l+l→ll. Then he+ll doesn't match any merge.
        // So result should be ["he", "ll"]
        let tokens = tok.encode("hell", false);
        assert_eq!(tokens, vec![257, 258]); // he=257, ll=258
    }

    #[test]
    fn gpt2_debug_format() {
        let tok = make_gpt2_tokenizer();
        let debug = format!("{:?}", tok);
        assert!(debug.contains("Gpt2"));
    }

    // =======================================================================
    // LLaMA3 pre-tokenization tests
    // =======================================================================

    #[test]
    fn llama3_pretokenize_simple() {
        let words = llama3_pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", " ", "world"]);
    }

    #[test]
    fn llama3_pretokenize_contractions_case_insensitive() {
        let words = llama3_pre_tokenize("I'M DON'T");
        // Should split: "I" "'M" " " "DON" "'T"
        assert!(words.contains(&"'M".to_string()), "got: {:?}", words);
        assert!(words.contains(&"'T".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_numbers_grouped() {
        let words = llama3_pre_tokenize("test 123456 abc");
        // Numbers split into groups of 1-3: "123", "456"
        assert!(words.contains(&"123".to_string()), "got: {:?}", words);
        assert!(words.contains(&"456".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_punctuation() {
        let words = llama3_pre_tokenize("hello, world!");
        assert!(words.contains(&",".to_string()), "got: {:?}", words);
        assert!(words.contains(&"!".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_newlines() {
        let words = llama3_pre_tokenize("hello\nworld");
        // Newline should be its own token
        assert!(words.contains(&"\n".to_string()), "got: {:?}", words);
    }

    #[test]
    fn llama3_pretokenize_whitespace_separate() {
        // Unlike GPT-2, LLaMA3 does NOT attach leading space to words
        let words = llama3_pre_tokenize("hello world");
        assert_eq!(words[0], "hello");
        // Space should be separate from "world"
        assert!(words.iter().any(|w| w == "world"), "got: {:?}", words);
    }
}
