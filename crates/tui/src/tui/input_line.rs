//! Lightweight single-line input with cursor.
//!
//! Replaces `ratatui-code-editor` for the input bar. Stores a `String` and a
//! character-offset cursor. Handles insert, delete, move, and clipboard paste
//! (via `arboard`). Rendering uses `wrap_line()` for soft word-wrapping —
//! the whole reason we built this instead of using the code editor.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// A single-line text buffer with cursor position (character offset).
#[derive(Debug)]
pub struct InputLine {
    content: String,
    /// Cursor position as a character offset (0 = before first char).
    cursor: usize,
}

impl InputLine {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor: 0,
        }
    }

    /// Current content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Cursor position (character offset).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Clear content and reset cursor.
    pub fn clear(&mut self) {
        self.content.clear();
        self.cursor = 0;
    }

    /// Set content and move cursor to end.
    pub fn set_content(&mut self, text: &str) {
        self.content = text.to_string();
        self.cursor = self.content.chars().count();
    }

    /// Extract content (trimmed) if non-empty, clearing the buffer.
    pub fn take(&mut self) -> Option<String> {
        let text = self.content.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.clear();
        Some(text)
    }

    /// Insert a character at the cursor position.
    /// Bare `\r` is silently dropped — only physical Enter submits.
    pub fn insert_char(&mut self, ch: char) {
        if ch == '\r' {
            return;
        }
        let byte_offset = self.byte_offset();
        self.content.insert(byte_offset, ch);
        self.cursor += 1;
    }

    /// Insert a string at the cursor position.
    /// Normalizes `\r\n` → `\n` and strips bare `\r`.
    pub fn insert_str(&mut self, s: &str) {
        let clean = s.replace("\r\n", "\n").replace('\r', "");
        let byte_offset = self.byte_offset();
        self.content.insert_str(byte_offset, &clean);
        self.cursor += clean.chars().count();
    }

    /// Delete the character before the cursor (Backspace).
    pub fn delete_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.cursor -= 1;
        let byte_offset = self.byte_offset();
        let ch = self.content[byte_offset..].chars().next().unwrap();
        self.content.replace_range(byte_offset..byte_offset + ch.len_utf8(), "");
    }

    /// Delete the character at the cursor (Delete key).
    pub fn delete_forward(&mut self) {
        let byte_offset = self.byte_offset();
        if byte_offset >= self.content.len() {
            return;
        }
        let ch = self.content[byte_offset..].chars().next().unwrap();
        self.content.replace_range(byte_offset..byte_offset + ch.len_utf8(), "");
    }

    /// Move cursor one character left.
    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Move cursor one character right.
    pub fn move_right(&mut self) {
        let max = self.content.chars().count();
        if self.cursor < max {
            self.cursor += 1;
        }
    }

    /// Move cursor to start.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end.
    pub fn move_end(&mut self) {
        self.cursor = self.content.chars().count();
    }

    /// Set cursor to an absolute character position (clamped to content length).
    pub fn set_cursor(&mut self, pos: usize) {
        let max = self.content.chars().count();
        self.cursor = pos.min(max);
    }

    /// Delete the word before the cursor (Ctrl+Backspace / Ctrl+W).
    pub fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let chars: Vec<char> = self.content.chars().collect();
        let mut pos = self.cursor;
        // Skip trailing whitespace
        while pos > 0 && chars[pos - 1].is_whitespace() {
            pos -= 1;
        }
        // Skip word characters
        while pos > 0 && !chars[pos - 1].is_whitespace() {
            pos -= 1;
        }
        let start_byte = self.char_to_byte(pos);
        let end_byte = self.byte_offset();
        self.content.replace_range(start_byte..end_byte, "");
        self.cursor = pos;
    }

    /// Paste from system clipboard (Ctrl+V).
    /// Preserves newlines (multiline input). Normalizes \r\n → \n.
    pub fn paste_clipboard(&mut self) {
        if let Ok(mut clip) = arboard::Clipboard::new() {
            if let Ok(text) = clip.get_text() {
                let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                self.insert_str(&normalized);
            }
        }
    }

    /// Handle a key event. Returns `true` if the key was consumed.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('v') if ctrl => {
                self.paste_clipboard();
                true
            }
            KeyCode::Char('w') if ctrl => {
                self.delete_word_back();
                true
            }
            KeyCode::Backspace if ctrl => {
                self.delete_word_back();
                true
            }
            KeyCode::Char(ch) => {
                self.insert_char(ch);
                true
            }
            KeyCode::Backspace => {
                self.delete_back();
                true
            }
            KeyCode::Delete => {
                self.delete_forward();
                true
            }
            KeyCode::Left => {
                self.move_left();
                true
            }
            KeyCode::Right => {
                self.move_right();
                true
            }
            KeyCode::Home => {
                self.move_home();
                true
            }
            KeyCode::End => {
                self.move_end();
                true
            }
            _ => false,
        }
    }

    // ── Internal helpers ──

    /// Convert cursor (char offset) to byte offset.
    fn byte_offset(&self) -> usize {
        self.char_to_byte(self.cursor)
    }

    /// Convert a character offset to a byte offset.
    fn char_to_byte(&self, char_pos: usize) -> usize {
        self.content
            .char_indices()
            .nth(char_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.content.len())
    }
}

impl Default for InputLine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let il = InputLine::new();
        assert_eq!(il.content(), "");
        assert_eq!(il.cursor(), 0);
    }

    #[test]
    fn insert_and_cursor() {
        let mut il = InputLine::new();
        il.insert_char('h');
        il.insert_char('i');
        assert_eq!(il.content(), "hi");
        assert_eq!(il.cursor(), 2);
    }

    #[test]
    fn insert_at_middle() {
        let mut il = InputLine::new();
        il.set_content("ac");
        il.cursor = 1; // after 'a'
        il.insert_char('b');
        assert_eq!(il.content(), "abc");
        assert_eq!(il.cursor(), 2);
    }

    #[test]
    fn delete_back() {
        let mut il = InputLine::new();
        il.set_content("abc");
        il.delete_back();
        assert_eq!(il.content(), "ab");
        assert_eq!(il.cursor(), 2);
    }

    #[test]
    fn delete_back_at_start() {
        let mut il = InputLine::new();
        il.set_content("abc");
        il.cursor = 0;
        il.delete_back();
        assert_eq!(il.content(), "abc"); // no-op
    }

    #[test]
    fn delete_forward() {
        let mut il = InputLine::new();
        il.set_content("abc");
        il.cursor = 1;
        il.delete_forward();
        assert_eq!(il.content(), "ac");
        assert_eq!(il.cursor(), 1);
    }

    #[test]
    fn delete_forward_at_end() {
        let mut il = InputLine::new();
        il.set_content("abc");
        il.delete_forward();
        assert_eq!(il.content(), "abc"); // no-op — cursor at end
    }

    #[test]
    fn move_left_right() {
        let mut il = InputLine::new();
        il.set_content("abc");
        assert_eq!(il.cursor(), 3);
        il.move_left();
        assert_eq!(il.cursor(), 2);
        il.move_right();
        assert_eq!(il.cursor(), 3);
        il.move_right(); // clamp
        assert_eq!(il.cursor(), 3);
    }

    #[test]
    fn move_home_end() {
        let mut il = InputLine::new();
        il.set_content("hello");
        il.move_home();
        assert_eq!(il.cursor(), 0);
        il.move_end();
        assert_eq!(il.cursor(), 5);
    }

    #[test]
    fn clear() {
        let mut il = InputLine::new();
        il.set_content("stuff");
        il.clear();
        assert_eq!(il.content(), "");
        assert_eq!(il.cursor(), 0);
    }

    #[test]
    fn take_returns_content() {
        let mut il = InputLine::new();
        il.set_content("task");
        let result = il.take();
        assert_eq!(result, Some("task".into()));
        assert_eq!(il.content(), "");
    }

    #[test]
    fn take_empty_returns_none() {
        let mut il = InputLine::new();
        assert_eq!(il.take(), None);
        il.set_content("  ");
        assert_eq!(il.take(), None); // whitespace-only
    }

    #[test]
    fn insert_str() {
        let mut il = InputLine::new();
        il.set_content("hello ");
        il.insert_str("world");
        assert_eq!(il.content(), "hello world");
        assert_eq!(il.cursor(), 11);
    }

    #[test]
    fn unicode_chars() {
        let mut il = InputLine::new();
        il.insert_char('é');
        il.insert_char('ñ');
        assert_eq!(il.content(), "éñ");
        assert_eq!(il.cursor(), 2);
        il.delete_back();
        assert_eq!(il.content(), "é");
        assert_eq!(il.cursor(), 1);
    }

    #[test]
    fn delete_word_back() {
        let mut il = InputLine::new();
        il.set_content("hello world");
        il.delete_word_back();
        assert_eq!(il.content(), "hello ");
        il.delete_word_back();
        assert_eq!(il.content(), "");
    }

    #[test]
    fn delete_word_back_at_start() {
        let mut il = InputLine::new();
        il.set_content("hello");
        il.cursor = 0;
        il.delete_word_back();
        assert_eq!(il.content(), "hello");
    }

    #[test]
    fn handle_key_char() {
        let mut il = InputLine::new();
        let consumed = il.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(consumed);
        assert_eq!(il.content(), "x");
    }

    #[test]
    fn handle_key_backspace() {
        let mut il = InputLine::new();
        il.set_content("ab");
        let consumed = il.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert!(consumed);
        assert_eq!(il.content(), "a");
    }

    #[test]
    fn set_cursor_clamps() {
        let mut il = InputLine::new();
        il.set_content("abc");
        il.set_cursor(1);
        assert_eq!(il.cursor(), 1);
        il.set_cursor(100);
        assert_eq!(il.cursor(), 3); // clamped to length
        il.set_cursor(0);
        assert_eq!(il.cursor(), 0);
    }

    #[test]
    fn handle_key_unknown_not_consumed() {
        let mut il = InputLine::new();
        let consumed = il.handle_key(KeyEvent::new(KeyCode::F(5), KeyModifiers::NONE));
        assert!(!consumed);
    }
}
