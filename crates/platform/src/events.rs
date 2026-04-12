//! Platform events — observable lifecycle transitions.
//!
//! These events are emitted by the router during instance and buffer operations.
//! The pipeline broadcasts them via its event bus so that the TUI, monitoring,
//! triggers, and admin tools can observe what's happening in the orchestrator.

use crate::address::Address;
use crate::buffers::ChannelType;
use crate::registry::Tier;

/// Events emitted by the platform orchestrator.
#[derive(Debug, Clone)]
pub enum PlatformEvent {
    /// A new agent instance was materialized.
    InstanceSpawned {
        /// Full address of the instance (without buffer).
        address: Address,
        /// Organism template name.
        organism: String,
        /// Kernel thread_id assigned.
        thread_id: String,
        /// Cache shard names bound to this instance.
        cache_shards: Vec<String>,
    },

    /// An agent instance was evicted (idle timeout or explicit kill).
    InstanceEvicted {
        /// Address of the evicted instance.
        address: Address,
        /// Organism template name.
        organism: String,
        /// Reason for eviction.
        reason: EvictionReason,
    },

    /// An instance's tier changed (Active → Shelved → Folded).
    InstanceTierChanged {
        address: Address,
        from: Tier,
        to: Tier,
    },

    /// A new buffer was created within an instance.
    BufferOpened {
        /// Full address including the buffer segment.
        address: Address,
        /// Instance address (without buffer).
        instance_address: Address,
        /// Buffer name (e.g., "dm", "help[email-issue]").
        buffer_name: String,
        /// Inferred channel type.
        channel: ChannelType,
        /// Kernel thread_id for this buffer.
        thread_id: String,
    },

    /// A message was delivered to a buffer.
    MessageDelivered {
        /// Full target address.
        address: Address,
        /// Buffer's kernel thread_id.
        thread_id: String,
        /// Source address, if known.
        from: Option<Address>,
    },

    /// A namespace violation was blocked.
    NamespaceViolation {
        /// Who tried to send.
        source: Address,
        /// Where they tried to send.
        target: Address,
    },
}

/// Why an instance was evicted.
#[derive(Debug, Clone)]
pub enum EvictionReason {
    /// Idle timeout expired.
    IdleTimeout,
    /// Explicitly killed by admin or parent.
    Killed,
    /// Memory pressure (LRU eviction).
    MemoryPressure,
}
