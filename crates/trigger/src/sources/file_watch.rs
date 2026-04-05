//! File watch trigger — fires when files matching a glob pattern change.

use std::path::Path;
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::time::Duration;

use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::{debug, error};

use crate::runtime::{TriggerEvent, TriggerPayload};
use crate::TriggerError;

/// Run a file watch trigger. Watches `pattern` with debounce, fires on changes.
pub async fn run(
    name: String,
    target: String,
    pattern: String,
    debounce_ms: u64,
    tx: mpsc::Sender<TriggerEvent>,
) -> Result<(), TriggerError> {
    // Extract the directory to watch from the glob pattern.
    // e.g., "organisms/*.yaml" → watch "organisms/"
    let watch_dir = glob_base_dir(&pattern);

    let (notify_tx, notify_rx) = std_mpsc::channel();

    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                let _ = notify_tx.send(event);
            }
        },
        Config::default().with_poll_interval(Duration::from_millis(debounce_ms)),
    )
    .map_err(|e| TriggerError::FileWatch(format!("Failed to create watcher: {e}")))?;

    let watch_path = Path::new(&watch_dir);
    if watch_path.exists() {
        watcher
            .watch(watch_path, RecursiveMode::Recursive)
            .map_err(|e| TriggerError::FileWatch(format!("Failed to watch {watch_dir}: {e}")))?;
    } else {
        return Err(TriggerError::FileWatch(format!(
            "Watch directory does not exist: {watch_dir}"
        )));
    }

    debug!("FileWatch trigger '{name}' watching {pattern} (dir: {watch_dir})");

    // Compile the glob pattern for filtering
    let glob_pattern =
        glob::Pattern::new(&pattern).map_err(|e| TriggerError::FileWatch(format!("Invalid glob: {e}")))?;

    let notify_rx = Arc::new(Mutex::new(notify_rx));

    // Process events in a blocking-friendly way
    loop {
        // Check for notify events with a timeout so we can detect channel closure
        let event = tokio::task::spawn_blocking({
            let rx = Arc::clone(&notify_rx);
            move || rx.lock().unwrap().recv_timeout(Duration::from_secs(1))
        })
        .await;

        match event {
            Ok(Ok(notify_event)) => {
                // Filter paths against the glob pattern
                let matched_paths: Vec<String> = notify_event
                    .paths
                    .iter()
                    .filter(|p| {
                        p.to_str()
                            .is_some_and(|s| glob_pattern.matches(s) || glob_pattern.matches_path(p))
                    })
                    .map(|p| p.display().to_string())
                    .collect();

                if !matched_paths.is_empty() {
                    debug!("FileWatch trigger '{name}' matched: {matched_paths:?}");

                    let trigger_event = TriggerEvent {
                        trigger_name: name.clone(),
                        target: target.clone(),
                        payload: TriggerPayload::FileChanged {
                            paths: matched_paths,
                        },
                    };

                    if tx.send(trigger_event).await.is_err() {
                        break; // Pipeline shutting down
                    }
                }
            }
            Ok(Err(_)) => {
                // Timeout — just loop and check again
                continue;
            }
            Err(_) => {
                // spawn_blocking panicked — shouldn't happen
                error!("FileWatch trigger '{name}' internal error");
                break;
            }
        }
    }

    Ok(())
}

/// Extract the base directory from a glob pattern.
/// "organisms/*.yaml" → "organisms"
/// "**/*.rs" → "."
/// "src/pipeline/*.rs" → "src/pipeline"
fn glob_base_dir(pattern: &str) -> String {
    // Walk the path components until we hit a glob character
    let parts: Vec<&str> = pattern.split('/').collect();
    let mut dir_parts = Vec::new();
    for part in &parts {
        if part.contains('*') || part.contains('?') || part.contains('[') {
            break;
        }
        dir_parts.push(*part);
    }
    if dir_parts.is_empty() {
        ".".to_string()
    } else {
        dir_parts.join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_base_dir_simple() {
        assert_eq!(glob_base_dir("organisms/*.yaml"), "organisms");
    }

    #[test]
    fn glob_base_dir_recursive() {
        assert_eq!(glob_base_dir("**/*.rs"), ".");
    }

    #[test]
    fn glob_base_dir_nested() {
        assert_eq!(glob_base_dir("src/pipeline/*.rs"), "src/pipeline");
    }

    #[test]
    fn glob_base_dir_no_glob() {
        assert_eq!(glob_base_dir("src/main.rs"), "src/main.rs");
    }
}
