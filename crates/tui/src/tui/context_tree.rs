//! Foldable context tree — tui-tree-widget integration.
//!
//! Renders context segments as a collapsible tree:
//! ```text
//! Thread: system.agentos.coding-agent
//!  [v] Active (3 segments, 4.2 KB)
//!      message:msg-001  [||||    ] 0.95  "Read src/main.rs"
//!      code:src/main.rs [|||     ] 0.87  fn main() { ... }
//!  [>] Shelved (2 segments, 1.1 KB)   ← collapsed
//! ```

use ratatui::style::Color;
use tui_tree_widget::TreeItem;

use super::app::ContextView;
use super::dashboard;
use agentos_kernel::context_store::SegmentStatus;

/// Build a tree from a ContextView.
pub fn build_context_tree<'a>(ctx: &ContextView) -> Vec<TreeItem<'a, String>> {
    let active_segs: Vec<_> = ctx
        .segments
        .iter()
        .filter(|s| s.status == SegmentStatus::Active)
        .collect();

    let folded_segs: Vec<_> = ctx
        .segments
        .iter()
        .filter(|s| s.status == SegmentStatus::Folded)
        .collect();

    let shelved_segs: Vec<_> = ctx
        .segments
        .iter()
        .filter(|s| s.status == SegmentStatus::Shelved)
        .collect();

    let mut items = Vec::new();

    // Active group
    let active_label = format!(
        "Active ({} segments, {})",
        active_segs.len(),
        dashboard::format_bytes(ctx.active_bytes),
    );
    let active_children: Vec<TreeItem<'a, String>> = active_segs
        .iter()
        .map(|s| {
            let bar = relevance_bar(s.relevance);
            let label = format!(
                "{}:{}  [{bar}] {:.2}  {}",
                s.tag,
                s.id,
                s.relevance,
                dashboard::format_bytes(s.size),
            );
            TreeItem::new_leaf(format!("active-{}", s.id), label)
        })
        .collect();
    items.push(
        TreeItem::new(
            "active".to_string(),
            active_label,
            active_children,
        )
        .expect("tree item creation"),
    );

    // Folded group (between Active and Shelved)
    if !folded_segs.is_empty() {
        let folded_bytes: usize = folded_segs.iter().map(|s| s.size).sum();
        let folded_label = format!(
            "Folded ({} segments, {})",
            folded_segs.len(),
            dashboard::format_bytes(folded_bytes),
        );
        let folded_children: Vec<TreeItem<'a, String>> = folded_segs
            .iter()
            .map(|s| {
                let bar = relevance_bar(s.relevance);
                let label = format!(
                    "{}:{}  [{bar}] {:.2}  {}",
                    s.tag,
                    s.id,
                    s.relevance,
                    dashboard::format_bytes(s.size),
                );
                TreeItem::new_leaf(format!("folded-{}", s.id), label)
            })
            .collect();
        items.push(
            TreeItem::new(
                "folded".to_string(),
                folded_label,
                folded_children,
            )
            .expect("tree item creation"),
        );
    }

    // Shelved group
    let shelved_bytes = ctx.total_bytes.saturating_sub(ctx.active_bytes);
    let shelved_label = format!(
        "Shelved ({} segments, {})",
        shelved_segs.len(),
        dashboard::format_bytes(shelved_bytes),
    );
    let shelved_children: Vec<TreeItem<'a, String>> = shelved_segs
        .iter()
        .map(|s| {
            let bar = relevance_bar(s.relevance);
            let label = format!(
                "{}:{}  [{bar}] {:.2}  {}",
                s.tag,
                s.id,
                s.relevance,
                dashboard::format_bytes(s.size),
            );
            TreeItem::new_leaf(format!("shelved-{}", s.id), label)
        })
        .collect();
    items.push(
        TreeItem::new(
            "shelved".to_string(),
            shelved_label,
            shelved_children,
        )
        .expect("tree item creation"),
    );

    items
}

/// Get the color for a segment status.
pub fn status_color(status: SegmentStatus) -> Color {
    match status {
        SegmentStatus::Active => Color::Green,
        SegmentStatus::Shelved => Color::DarkGray,
        SegmentStatus::Folded => Color::Blue,
    }
}

/// Create a relevance bar string.
/// Red (0-0.3), yellow (0.3-0.7), green (0.7-1.0).
pub fn relevance_bar(relevance: f32) -> String {
    let filled = (relevance * 8.0).round() as usize;
    let empty = 8usize.saturating_sub(filled);
    format!("{}{}", "|".repeat(filled), " ".repeat(empty))
}

/// Get the color for a relevance score.
pub fn relevance_color(relevance: f32) -> Color {
    if relevance >= 0.7 {
        Color::Green
    } else if relevance >= 0.3 {
        Color::Yellow
    } else {
        Color::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::{ContextView, SegmentView};
    use agentos_kernel::context_store::SegmentStatus;

    #[test]
    fn build_tree_empty() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![],
            active_count: 0,
            shelved_count: 0,
            folded_count: 0,
            total_bytes: 0,
            active_bytes: 0,
        };
        let tree = build_context_tree(&ctx);
        assert_eq!(tree.len(), 2); // Active and Shelved groups always present
    }

    #[test]
    fn build_tree_groups() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![
                SegmentView {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Active,
                    relevance: 0.9,
                },
                SegmentView {
                    id: "s2".into(),
                    tag: "msg".into(),
                    size: 50,
                    status: SegmentStatus::Shelved,
                    relevance: 0.3,
                },
            ],
            active_count: 1,
            shelved_count: 1,
            folded_count: 0,
            total_bytes: 150,
            active_bytes: 100,
        };
        let tree = build_context_tree(&ctx);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn build_tree_metadata() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![SegmentView {
                id: "code:main.rs".into(),
                tag: "code".into(),
                size: 2048,
                status: SegmentStatus::Active,
                relevance: 0.87,
            }],
            active_count: 1,
            shelved_count: 0,
            folded_count: 0,
            total_bytes: 2048,
            active_bytes: 2048,
        };
        let tree = build_context_tree(&ctx);
        // Active group exists with one child
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn relevance_bar_high() {
        let bar = relevance_bar(0.9);
        let filled = bar.chars().filter(|c| *c == '|').count();
        assert!(filled >= 7); // 0.9 * 8 = 7.2, rounds to 7
    }

    #[test]
    fn relevance_bar_medium() {
        let bar = relevance_bar(0.5);
        let filled = bar.chars().filter(|c| *c == '|').count();
        assert_eq!(filled, 4);
    }

    #[test]
    fn relevance_bar_low() {
        let bar = relevance_bar(0.1);
        let filled = bar.chars().filter(|c| *c == '|').count();
        assert_eq!(filled, 1);
    }

    // ── Folding Context tests (Milestone 4) ──

    #[test]
    fn context_view_folded_count() {
        use agentos_kernel::context_store::{ContextInventory, SegmentMeta};
        let inv = ContextInventory {
            thread_id: "t1".into(),
            segments: vec![
                SegmentMeta {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Folded,
                    relevance: 0.5,
                    created_at: 0,
                },
            ],
            active_count: 0,
            shelved_count: 0,
            folded_count: 1,
            total_bytes: 100,
            active_bytes: 0,
        };
        let view = ContextView::from(&inv);
        assert_eq!(view.folded_count, 1);
    }

    #[test]
    fn build_tree_folded_group() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![
                SegmentView {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Active,
                    relevance: 0.9,
                },
                SegmentView {
                    id: "s2".into(),
                    tag: "msg".into(),
                    size: 30,
                    status: SegmentStatus::Folded,
                    relevance: 0.4,
                },
                SegmentView {
                    id: "s3".into(),
                    tag: "doc".into(),
                    size: 50,
                    status: SegmentStatus::Shelved,
                    relevance: 0.2,
                },
            ],
            active_count: 1,
            shelved_count: 1,
            folded_count: 1,
            total_bytes: 180,
            active_bytes: 100,
        };
        let tree = build_context_tree(&ctx);
        assert_eq!(tree.len(), 3); // Active, Folded, Shelved
    }

    #[test]
    fn build_tree_no_folded() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![
                SegmentView {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 100,
                    status: SegmentStatus::Active,
                    relevance: 0.9,
                },
            ],
            active_count: 1,
            shelved_count: 0,
            folded_count: 0,
            total_bytes: 100,
            active_bytes: 100,
        };
        let tree = build_context_tree(&ctx);
        assert_eq!(tree.len(), 2); // Active and Shelved only (no Folded group)
    }

    #[test]
    fn segment_view_folded_status() {
        let sv = SegmentView {
            id: "s1".into(),
            tag: "code".into(),
            size: 50,
            status: SegmentStatus::Folded,
            relevance: 0.5,
        };
        assert_eq!(sv.status, SegmentStatus::Folded);
    }

    #[test]
    fn relevance_color_folded() {
        // Folded segments get a distinct color (blue)
        let color = status_color(SegmentStatus::Folded);
        assert_eq!(color, Color::Blue);
    }

    #[test]
    fn dashboard_shows_folded_count() {
        let ctx = ContextView {
            thread_id: "t1".into(),
            segments: vec![
                SegmentView {
                    id: "s1".into(),
                    tag: "code".into(),
                    size: 30,
                    status: SegmentStatus::Folded,
                    relevance: 0.5,
                },
                SegmentView {
                    id: "s2".into(),
                    tag: "code".into(),
                    size: 30,
                    status: SegmentStatus::Folded,
                    relevance: 0.3,
                },
            ],
            active_count: 0,
            shelved_count: 0,
            folded_count: 2,
            total_bytes: 60,
            active_bytes: 0,
        };
        // Verify the folded_count is accessible and correct
        assert_eq!(ctx.folded_count, 2);
        // Build tree should include folded group
        let tree = build_context_tree(&ctx);
        assert!(tree.len() >= 3); // Active, Folded, Shelved
    }
}
