//! Port Manager — tracks port declarations per listener, validates no conflicts.
//!
//! Each listener declares its port requirements (inbound/outbound, protocol, hosts).
//! The PortManager validates that no two listeners conflict on the same port+direction.

pub mod firewall;

use std::collections::HashMap;

/// Direction of network traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Inbound,
    Outbound,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Inbound => write!(f, "inbound"),
            Direction::Outbound => write!(f, "outbound"),
        }
    }
}

/// Network protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http,
    Https,
    Tcp,
    Udp,
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Protocol::Http => write!(f, "http"),
            Protocol::Https => write!(f, "https"),
            Protocol::Tcp => write!(f, "tcp"),
            Protocol::Udp => write!(f, "udp"),
        }
    }
}

impl Protocol {
    /// Parse a protocol from a string.
    pub fn from_str_lc(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "http" => Ok(Protocol::Http),
            "https" => Ok(Protocol::Https),
            "tcp" => Ok(Protocol::Tcp),
            "udp" => Ok(Protocol::Udp),
            _ => Err(format!("unknown protocol: '{s}'")),
        }
    }

    /// IP protocol for iptables.
    pub fn ip_protocol(&self) -> &str {
        match self {
            Protocol::Http | Protocol::Https | Protocol::Tcp => "tcp",
            Protocol::Udp => "udp",
        }
    }
}

/// A port declaration by a listener.
#[derive(Debug, Clone)]
pub struct PortDeclaration {
    pub port: u16,
    pub direction: Direction,
    pub protocol: Protocol,
    pub allowed_hosts: Vec<String>,
}

/// Manages port allocations across all listeners. Detects conflicts.
pub struct PortManager {
    allocations: HashMap<String, Vec<PortDeclaration>>,
}

impl PortManager {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
        }
    }

    /// Declare a port for a listener. Returns error on conflict.
    pub fn declare(&mut self, listener: &str, decl: PortDeclaration) -> Result<(), String> {
        // Check for conflicts: same port + same direction on a different listener
        for (existing_listener, decls) in &self.allocations {
            if existing_listener == listener {
                continue;
            }
            for existing in decls {
                if existing.port == decl.port && existing.direction == decl.direction {
                    return Err(format!(
                        "port conflict: {} port {} already declared by '{}', cannot declare for '{}'",
                        decl.direction, decl.port, existing_listener, listener
                    ));
                }
            }
        }

        self.allocations
            .entry(listener.to_string())
            .or_default()
            .push(decl);

        Ok(())
    }

    /// Validate all declarations for conflicts. Returns all conflict errors.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let all: Vec<(&str, &PortDeclaration)> = self.all_ports();

        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let (l1, d1) = &all[i];
                let (l2, d2) = &all[j];
                if l1 != l2 && d1.port == d2.port && d1.direction == d2.direction {
                    errors.push(format!(
                        "port conflict: {} port {} declared by both '{}' and '{}'",
                        d1.direction, d1.port, l1, l2
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get port declarations for a specific listener.
    pub fn get_ports(&self, listener: &str) -> &[PortDeclaration] {
        self.allocations
            .get(listener)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all port declarations across all listeners.
    pub fn all_ports(&self) -> Vec<(&str, &PortDeclaration)> {
        let mut result = Vec::new();
        for (listener, decls) in &self.allocations {
            for decl in decls {
                result.push((listener.as_str(), decl));
            }
        }
        result
    }

    /// Get all listener names that have port declarations.
    pub fn listeners_with_ports(&self) -> Vec<&str> {
        self.allocations.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for PortManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_ports() {
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

        let ports = pm.get_ports("llm-pool");
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, 443);
        assert_eq!(ports[0].direction, Direction::Outbound);
    }

    #[test]
    fn detect_conflict_same_port_same_direction() {
        let mut pm = PortManager::new();
        pm.declare(
            "listener-a",
            PortDeclaration {
                port: 8080,
                direction: Direction::Inbound,
                protocol: Protocol::Http,
                allowed_hosts: vec![],
            },
        )
        .unwrap();

        let err = pm
            .declare(
                "listener-b",
                PortDeclaration {
                    port: 8080,
                    direction: Direction::Inbound,
                    protocol: Protocol::Http,
                    allowed_hosts: vec![],
                },
            )
            .unwrap_err();
        assert!(err.contains("port conflict"));
        assert!(err.contains("8080"));
    }

    #[test]
    fn no_conflict_same_port_different_direction() {
        let mut pm = PortManager::new();
        pm.declare(
            "listener-a",
            PortDeclaration {
                port: 443,
                direction: Direction::Outbound,
                protocol: Protocol::Https,
                allowed_hosts: vec![],
            },
        )
        .unwrap();

        // Same port, different direction — OK
        pm.declare(
            "listener-b",
            PortDeclaration {
                port: 443,
                direction: Direction::Inbound,
                protocol: Protocol::Https,
                allowed_hosts: vec![],
            },
        )
        .unwrap();
    }

    #[test]
    fn same_listener_same_port_ok() {
        let mut pm = PortManager::new();
        // A listener can declare the same port twice (e.g., different host lists)
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
        pm.declare(
            "llm-pool",
            PortDeclaration {
                port: 443,
                direction: Direction::Outbound,
                protocol: Protocol::Https,
                allowed_hosts: vec!["backup.anthropic.com".into()],
            },
        )
        .unwrap();
        assert_eq!(pm.get_ports("llm-pool").len(), 2);
    }

    #[test]
    fn validate_detects_all_conflicts() {
        let mut pm = PortManager::new();
        // Build conflicting state by hand (bypass declare's check)
        pm.allocations.insert(
            "a".into(),
            vec![PortDeclaration {
                port: 80,
                direction: Direction::Inbound,
                protocol: Protocol::Http,
                allowed_hosts: vec![],
            }],
        );
        pm.allocations.insert(
            "b".into(),
            vec![PortDeclaration {
                port: 80,
                direction: Direction::Inbound,
                protocol: Protocol::Http,
                allowed_hosts: vec![],
            }],
        );

        let errs = pm.validate().unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("port conflict"));
    }

    #[test]
    fn all_ports_lists_everything() {
        let mut pm = PortManager::new();
        pm.declare(
            "a",
            PortDeclaration {
                port: 80,
                direction: Direction::Inbound,
                protocol: Protocol::Http,
                allowed_hosts: vec![],
            },
        )
        .unwrap();
        pm.declare(
            "b",
            PortDeclaration {
                port: 443,
                direction: Direction::Outbound,
                protocol: Protocol::Https,
                allowed_hosts: vec![],
            },
        )
        .unwrap();

        let all = pm.all_ports();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn empty_manager_validates_ok() {
        let pm = PortManager::new();
        assert!(pm.validate().is_ok());
    }

    #[test]
    fn protocol_from_str() {
        assert_eq!(Protocol::from_str_lc("http").unwrap(), Protocol::Http);
        assert_eq!(Protocol::from_str_lc("HTTPS").unwrap(), Protocol::Https);
        assert_eq!(Protocol::from_str_lc("tcp").unwrap(), Protocol::Tcp);
        assert_eq!(Protocol::from_str_lc("UDP").unwrap(), Protocol::Udp);
        assert!(Protocol::from_str_lc("ftp").is_err());
    }
}
