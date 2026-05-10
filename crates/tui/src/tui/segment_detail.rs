//! Segment detail viewer — popup for full segment content.

/// Full detail view for a single context segment.
#[derive(Debug, Clone)]
pub struct SegmentDetail {
    pub id: String,
    pub tag: String,
    pub content: String,
    pub scroll: u16,
    pub max_scroll: u16,
}

impl SegmentDetail {
    /// Create a new segment detail view.
    pub fn new(id: String, tag: String, content: String) -> Self {
        let line_count = content.lines().count() as u16;
        Self {
            id,
            tag,
            content,
            scroll: 0,
            max_scroll: line_count.saturating_sub(1),
        }
    }

    /// Scroll down.
    pub fn scroll_down(&mut self) {
        if self.scroll < self.max_scroll {
            self.scroll += 1;
        }
    }

    /// Scroll up.
    pub fn scroll_up(&mut self) {
        self.scroll = self.scroll.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_detail_creation() {
        let detail = SegmentDetail::new(
            "code:main.rs".into(),
            "code".into(),
            "fn main() {\n    println!(\"hello\");\n}".into(),
        );
        assert_eq!(detail.id, "code:main.rs");
        assert_eq!(detail.tag, "code");
        assert_eq!(detail.scroll, 0);
        assert_eq!(detail.max_scroll, 2); // 3 lines, max scroll = 2
    }

    #[test]
    fn segment_detail_scroll_clamped() {
        let mut detail = SegmentDetail::new("s1".into(), "msg".into(), "line1\nline2".into());
        detail.scroll_down();
        assert_eq!(detail.scroll, 1);
        detail.scroll_down();
        assert_eq!(detail.scroll, 1); // clamped

        detail.scroll_up();
        assert_eq!(detail.scroll, 0);
        detail.scroll_up();
        assert_eq!(detail.scroll, 0); // clamped at 0
    }
}
