//! Firewall rule generation from port declarations + security profiles.
//!
//! Generates iptables-syntax rules as strings. Portable — works on any OS
//! (just strings, no system calls). Applied on Linux deployment only.

use super::{Direction, PortManager, Protocol};
use crate::organism::Organism;

/// Action for a firewall rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Allow,
    Deny,
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Allow => write!(f, "ACCEPT"),
            Action::Deny => write!(f, "DROP"),
        }
    }
}

/// A generated firewall rule.
#[derive(Debug, Clone)]
pub struct FirewallRule {
    pub linux_user: String,
    pub port: u16,
    pub direction: Direction,
    pub protocol: Protocol,
    pub allowed_hosts: Vec<String>,
    pub action: Action,
}

impl FirewallRule {
    /// Render to an iptables command string.
    pub fn to_iptables(&self) -> String {
        let chain = match self.direction {
            Direction::Inbound => "INPUT",
            Direction::Outbound => "OUTPUT",
        };

        let port_flag = match self.direction {
            Direction::Inbound => "--dport",
            Direction::Outbound => "--dport",
        };

        let base = format!(
            "iptables -A {chain} -p {} {port_flag} {} -m owner --uid-owner {} -j {}",
            self.protocol.ip_protocol(),
            self.port,
            self.linux_user,
            self.action,
        );

        if self.allowed_hosts.is_empty() {
            return base;
        }

        // Generate one rule per allowed host
        self.allowed_hosts
            .iter()
            .map(|host| {
                let dst_flag = match self.direction {
                    Direction::Outbound => format!(" -d {host}"),
                    Direction::Inbound => format!(" -s {host}"),
                };
                format!(
                    "iptables -A {chain} -p {} {port_flag} {}{dst_flag} -m owner --uid-owner {} -j {}",
                    self.protocol.ip_protocol(),
                    self.port,
                    self.linux_user,
                    self.action,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Generate firewall rules from port declarations and organism configuration.
///
/// For each listener with port declarations:
/// - Find the profiles that include that listener
/// - For each such profile, look up the linux_user
/// - Generate ALLOW rules for declared ports under that user
pub fn generate_rules(port_manager: &PortManager, organism: &Organism) -> Vec<FirewallRule> {
    let mut rules = Vec::new();

    for (listener_name, decl) in port_manager.all_ports() {
        // Find all profiles that grant access to this listener
        for profile_name in organism.profile_names() {
            if let Some(profile) = organism.get_profile(profile_name) {
                let listener_str = listener_name.to_string();
                let has_access =
                    profile.allow_all || profile.allowed_listeners.contains(listener_name);

                // Also check network field if present
                let has_network =
                    profile.network.is_empty() || profile.network.contains(&listener_str);

                if has_access && has_network {
                    rules.push(FirewallRule {
                        linux_user: profile.linux_user.clone(),
                        port: decl.port,
                        direction: decl.direction,
                        protocol: decl.protocol,
                        allowed_hosts: decl.allowed_hosts.clone(),
                        action: Action::Allow,
                    });
                }
            }
        }
    }

    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::profile::{RetentionPolicy, SecurityProfile};
    use crate::organism::ListenerDef;
    use crate::ports::PortDeclaration;

    fn setup() -> (PortManager, Organism) {
        let mut pm = PortManager::new();
        pm.declare(
            "llm-pool",
            PortDeclaration {
                port: 443,
                direction: Direction::Outbound,
                protocol: Protocol::Https,
                allowed_hosts: vec!["api.anthropic.com".into()],
            },
        )
        .unwrap();

        let mut org = Organism::new("test");
        org.register_listener(ListenerDef {
            name: "llm-pool".into(),
            payload_tag: "LlmRequest".into(),
            handler: "llm.handle".into(),
            description: "LLM pool".into(),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
        })
        .unwrap();
        org.register_listener(ListenerDef {
            name: "file-ops".into(),
            payload_tag: "FileOpsRequest".into(),
            handler: "tools.file_ops.handle".into(),
            description: "File ops".into(),
            is_agent: false,
            peers: vec![],
            model: None,
            ports: vec![],
        })
        .unwrap();

        org.add_profile(SecurityProfile {
            name: "admin".into(),
            linux_user: "agentos-admin".into(),
            allowed_listeners: ["llm-pool", "file-ops"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            allow_all: false,
            journal_retention: RetentionPolicy::Forever,
            network: vec!["llm-pool".into()],
        })
        .unwrap();

        org.add_profile(SecurityProfile {
            name: "public".into(),
            linux_user: "agentos-public".into(),
            allowed_listeners: ["file-ops"].iter().map(|s| s.to_string()).collect(),
            allow_all: false,
            journal_retention: RetentionPolicy::PruneOnDelivery,
            network: vec![],
        })
        .unwrap();

        (pm, org)
    }

    #[test]
    fn generate_rules_for_allowed_profile() {
        let (pm, org) = setup();
        let rules = generate_rules(&pm, &org);

        // Admin profile has llm-pool access and network access → rule generated
        let admin_rules: Vec<_> = rules
            .iter()
            .filter(|r| r.linux_user == "agentos-admin")
            .collect();
        assert!(!admin_rules.is_empty());
        assert_eq!(admin_rules[0].port, 443);
        assert_eq!(admin_rules[0].direction, Direction::Outbound);
    }

    #[test]
    fn no_rules_for_restricted_profile() {
        let (pm, org) = setup();
        let rules = generate_rules(&pm, &org);

        // Public profile doesn't have llm-pool access → no rule for port 443
        let public_rules: Vec<_> = rules
            .iter()
            .filter(|r| r.linux_user == "agentos-public")
            .collect();
        assert!(public_rules.is_empty());
    }

    #[test]
    fn iptables_syntax_no_hosts() {
        let rule = FirewallRule {
            linux_user: "agentos-admin".into(),
            port: 8080,
            direction: Direction::Inbound,
            protocol: Protocol::Http,
            allowed_hosts: vec![],
            action: Action::Allow,
        };

        let cmd = rule.to_iptables();
        assert!(cmd.contains("iptables -A INPUT"));
        assert!(cmd.contains("-p tcp"));
        assert!(cmd.contains("--dport 8080"));
        assert!(cmd.contains("--uid-owner agentos-admin"));
        assert!(cmd.contains("-j ACCEPT"));
    }

    #[test]
    fn iptables_syntax_with_hosts() {
        let rule = FirewallRule {
            linux_user: "agentos-admin".into(),
            port: 443,
            direction: Direction::Outbound,
            protocol: Protocol::Https,
            allowed_hosts: vec!["api.anthropic.com".into()],
            action: Action::Allow,
        };

        let cmd = rule.to_iptables();
        assert!(cmd.contains("iptables -A OUTPUT"));
        assert!(cmd.contains("-d api.anthropic.com"));
        assert!(cmd.contains("-j ACCEPT"));
    }

    #[test]
    fn iptables_deny_rule() {
        let rule = FirewallRule {
            linux_user: "nobody".into(),
            port: 22,
            direction: Direction::Outbound,
            protocol: Protocol::Tcp,
            allowed_hosts: vec![],
            action: Action::Deny,
        };

        let cmd = rule.to_iptables();
        assert!(cmd.contains("-j DROP"));
    }

    #[test]
    fn iptables_udp_rule() {
        let rule = FirewallRule {
            linux_user: "agentos-admin".into(),
            port: 53,
            direction: Direction::Outbound,
            protocol: Protocol::Udp,
            allowed_hosts: vec![],
            action: Action::Allow,
        };

        let cmd = rule.to_iptables();
        assert!(cmd.contains("-p udp"));
    }
}
