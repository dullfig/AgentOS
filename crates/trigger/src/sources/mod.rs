//! Trigger source implementations — one module per source type.

pub mod file_watch;
pub mod timer;
pub mod cron;
pub mod event_bus;
pub mod rhai_trigger;
