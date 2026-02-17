//! Security — profile resolution and dispatch table enforcement.
//!
//! The security layer resolves a thread's profile and provides
//! the corresponding dispatch table. If a route doesn't exist
//! in the profile's dispatch table → structural impossibility.

use std::collections::HashMap;

use crate::kernel::thread_table::ThreadTable;
use crate::organism::profile::DispatchTable;
use crate::organism::Organism;

/// Resolves thread → profile → dispatch table.
pub struct SecurityResolver {
    /// Cached dispatch tables by profile name.
    tables: HashMap<String, DispatchTable>,
}

impl SecurityResolver {
    /// Build a resolver from the organism configuration.
    pub fn from_organism(organism: &Organism) -> Result<Self, String> {
        let mut tables = HashMap::new();
        for name in organism.profile_names() {
            let table = organism.dispatch_table(name)?;
            tables.insert(name.to_string(), table);
        }
        Ok(Self { tables })
    }

    /// Resolve the dispatch table for a thread.
    /// Looks up the thread's profile in the thread table, then returns
    /// the corresponding dispatch table.
    pub fn resolve<'a>(
        &'a self,
        threads: &ThreadTable,
        thread_id: &str,
    ) -> Result<&'a DispatchTable, String> {
        let profile_name = threads
            .get_profile(thread_id)
            .ok_or_else(|| format!("no profile for thread '{thread_id}'"))?;

        self.tables
            .get(profile_name)
            .ok_or_else(|| format!("profile '{profile_name}' not found in resolver"))
    }

    /// Get a dispatch table by profile name directly.
    pub fn get_table(&self, profile_name: &str) -> Option<&DispatchTable> {
        self.tables.get(profile_name)
    }

    /// Check if a specific listener is reachable under a profile.
    pub fn can_reach(&self, profile_name: &str, listener_name: &str) -> bool {
        self.tables
            .get(profile_name)
            .map(|t| t.has_listener(listener_name))
            .unwrap_or(false)
    }

    /// Rebuild tables after a hot reload.
    pub fn rebuild(&mut self, organism: &Organism) -> Result<(), String> {
        let mut tables = HashMap::new();
        for name in organism.profile_names() {
            let table = organism.dispatch_table(name)?;
            tables.insert(name.to_string(), table);
        }
        self.tables = tables;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::profile::{RetentionPolicy, SecurityProfile};
    use crate::organism::ListenerDef;
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn setup_organism() -> Organism {
        let mut org = Organism::new("test");

        for name in &["file-ops", "shell", "faq", "scheduling"] {
            org.register_listener(ListenerDef {
                name: name.to_string(),
                payload_tag: format!("{name}Request"),
                handler: format!("handlers.{name}.handle"),
                description: name.to_string(),
                is_agent: false,
                peers: vec![],
                model: None,
                ports: vec![],
                librarian: false,
            })
            .unwrap();
        }

        // Admin profile: file-ops + shell
        org.add_profile(SecurityProfile {
            name: "admin".into(),
            linux_user: "agentos-admin".into(),
            allowed_listeners: ["file-ops", "shell"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            allow_all: false,
            journal_retention: RetentionPolicy::RetainDays(90),
            network: vec![],
        })
        .unwrap();

        // Public profile: faq + scheduling
        org.add_profile(SecurityProfile {
            name: "public".into(),
            linux_user: "agentos-public".into(),
            allowed_listeners: ["faq", "scheduling"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            allow_all: false,
            journal_retention: RetentionPolicy::PruneOnDelivery,
            network: vec![],
        })
        .unwrap();

        // Root profile: all
        org.add_profile(SecurityProfile {
            name: "root".into(),
            linux_user: "agentos-root".into(),
            allowed_listeners: HashSet::new(),
            allow_all: true,
            journal_retention: RetentionPolicy::Forever,
            network: vec![],
        })
        .unwrap();

        org
    }

    #[test]
    fn restricted_profile_blocks_listener() {
        let org = setup_organism();
        let resolver = SecurityResolver::from_organism(&org).unwrap();

        // Public can reach faq, not file-ops
        assert!(resolver.can_reach("public", "faq"));
        assert!(resolver.can_reach("public", "scheduling"));
        assert!(!resolver.can_reach("public", "file-ops"));
        assert!(!resolver.can_reach("public", "shell"));
    }

    #[test]
    fn admin_profile_allows_subset() {
        let org = setup_organism();
        let resolver = SecurityResolver::from_organism(&org).unwrap();

        assert!(resolver.can_reach("admin", "file-ops"));
        assert!(resolver.can_reach("admin", "shell"));
        assert!(!resolver.can_reach("admin", "faq"));
    }

    #[test]
    fn root_profile_allows_all() {
        let org = setup_organism();
        let resolver = SecurityResolver::from_organism(&org).unwrap();

        assert!(resolver.can_reach("root", "file-ops"));
        assert!(resolver.can_reach("root", "shell"));
        assert!(resolver.can_reach("root", "faq"));
        assert!(resolver.can_reach("root", "scheduling"));
    }

    #[test]
    fn resolve_thread_profile() {
        let org = setup_organism();
        let resolver = SecurityResolver::from_organism(&org).unwrap();

        let dir = TempDir::new().unwrap();
        let mut threads =
            crate::kernel::thread_table::ThreadTable::open(&dir.path().join("t.bin")).unwrap();
        let root = threads.initialize_root("org", "admin");
        let child = threads.extend_chain(&root, "handler");

        // Child inherits admin profile
        let table = resolver.resolve(&threads, &child).unwrap();
        assert_eq!(table.profile_name, "admin");
        assert!(table.has_listener("file-ops"));
        assert!(!table.has_listener("faq"));
    }

    #[test]
    fn resolve_missing_profile_errors() {
        let org = setup_organism();
        let resolver = SecurityResolver::from_organism(&org).unwrap();

        let dir = TempDir::new().unwrap();
        let mut threads =
            crate::kernel::thread_table::ThreadTable::open(&dir.path().join("t.bin")).unwrap();
        let root = threads.initialize_root("org", "nonexistent");

        let err = resolver.resolve(&threads, &root).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn rebuild_after_hot_reload() {
        let mut org = setup_organism();
        let mut resolver = SecurityResolver::from_organism(&org).unwrap();

        // Initially, public cannot reach file-ops
        assert!(!resolver.can_reach("public", "file-ops"));

        // Hot reload: add file-ops to public
        let new_allowed: HashSet<String> = ["faq", "scheduling", "file-ops"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        org.add_profile(SecurityProfile {
            name: "public".into(),
            linux_user: "agentos-public".into(),
            allowed_listeners: new_allowed,
            allow_all: false,
            journal_retention: RetentionPolicy::PruneOnDelivery,
            network: vec![],
        })
        .unwrap();

        resolver.rebuild(&org).unwrap();

        // Now public can reach file-ops
        assert!(resolver.can_reach("public", "file-ops"));
    }
}
