//! Security profiles and dispatch tables.
//!
//! A profile = named dispatch table (subset of routing table) + Linux user + retention policy.

use std::collections::{HashMap, HashSet};

use super::ListenerDef;

/// Retention policy for journal entries under a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Keep forever (coding agent).
    Forever,
    /// Delete after delivery confirmation.
    PruneOnDelivery,
    /// Keep for N days.
    RetainDays(u16),
}

/// A security profile: defines what a thread can access.
#[derive(Debug, Clone)]
pub struct SecurityProfile {
    /// Profile name (e.g., "root", "admin", "public").
    pub name: String,
    /// Linux user for process isolation.
    pub linux_user: String,
    /// Set of allowed listener names. Ignored if `allow_all` is true.
    pub allowed_listeners: HashSet<String>,
    /// If true, all listeners are allowed.
    pub allow_all: bool,
    /// Journal retention policy for messages under this profile.
    pub journal_retention: RetentionPolicy,
    /// Which listeners' ports this profile can use (for network access).
    /// Empty means no network restrictions beyond listener access.
    pub network: Vec<String>,
}

/// A materialized dispatch table for a specific profile.
///
/// Contains only the listeners the profile is allowed to access.
/// Route resolution against this table structurally prevents access
/// to listeners not in the profile.
#[derive(Debug)]
pub struct DispatchTable {
    /// Which profile this table was built from.
    pub profile_name: String,
    /// Listeners accessible under this profile.
    pub listeners: HashMap<String, ListenerDef>,
}

impl DispatchTable {
    /// Check if a listener is reachable under this profile.
    pub fn has_listener(&self, name: &str) -> bool {
        self.listeners.contains_key(name)
    }

    /// Get all listener names in this dispatch table.
    pub fn listener_names(&self) -> Vec<&str> {
        self.listeners.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_table_has_listener() {
        let mut listeners = HashMap::new();
        listeners.insert(
            "echo".to_string(),
            ListenerDef {
                name: "echo".into(),
                payload_tag: "Greeting".into(),
                handler: "handlers.echo.handle".into(),
                description: "Echo".into(),
                is_agent: false,
                peers: vec![],
                model: None,
                ports: vec![],
                librarian: false,
            },
        );

        let table = DispatchTable {
            profile_name: "test".into(),
            listeners,
        };

        assert!(table.has_listener("echo"));
        assert!(!table.has_listener("secret"));
    }
}
