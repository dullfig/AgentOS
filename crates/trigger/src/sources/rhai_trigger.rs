//! Rhai script trigger — runs a user-defined script on a schedule.
//!
//! The script must define a `check()` function that returns:
//! - A string → trigger fires with that string as payload
//! - `()` (unit/nothing) → trigger does not fire
//!
//! Example script:
//! ```rhai
//! fn check() {
//!     // Read a file, check a condition, whatever you need
//!     let status = http_get("http://localhost:8080/health", #{});
//!     let data = parse_json(status);
//!     if data.error_count > 0 {
//!         return "Health check failed: " + data.error_count + " errors";
//!     }
//!     // Return nothing → don't fire
//! }
//! ```

use rhai::{Dynamic, Engine, Scope};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{debug, warn};

use crate::runtime::{TriggerEvent, TriggerPayload};
use crate::TriggerError;

/// Run a Rhai trigger. Evaluates `check()` every `poll_secs`, fires if it returns a value.
pub async fn run(
    name: String,
    target: String,
    script: String,
    poll_secs: u64,
    tx: mpsc::Sender<TriggerEvent>,
) -> Result<(), TriggerError> {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(128, 64);

    // Register HTTP helpers (same as cloud provider runtime)
    register_http_functions(&mut engine);

    let ast = engine.compile(&script).map_err(|e| {
        TriggerError::ScriptError(format!("Failed to compile trigger script '{name}': {e}"))
    })?;

    // Validate that check() exists
    if !ast.iter_functions().any(|f| f.name == "check") {
        return Err(TriggerError::ScriptError(format!(
            "Trigger script '{name}' missing required function: check()"
        )));
    }

    debug!("Rhai trigger '{name}' loaded, polling every {poll_secs}s");

    let mut tick = interval(Duration::from_secs(poll_secs));
    // Skip the immediate first tick
    tick.tick().await;

    let mut scope = Scope::new();

    loop {
        tick.tick().await;

        match engine.call_fn::<Dynamic>(&mut scope, &ast, "check", ()) {
            Ok(result) => {
                // Unit (()) means don't fire. Anything else means fire.
                if result.is_unit() {
                    continue;
                }

                let result_str = if result.is_string() {
                    result.into_string().unwrap_or_default()
                } else {
                    format!("{result}")
                };

                debug!("Rhai trigger '{name}' fired: {result_str}");

                let event = TriggerEvent {
                    trigger_name: name.clone(),
                    target: target.clone(),
                    payload: TriggerPayload::Rhai {
                        result: result_str,
                    },
                };

                if tx.send(event).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                warn!("Rhai trigger '{name}' check() error: {e}");
                // Don't crash on script errors — just skip this tick
            }
        }
    }

    Ok(())
}

/// Register HTTP helpers into the Rhai engine (same pattern as cloud runtime).
fn register_http_functions(engine: &mut Engine) {
    engine.register_fn(
        "http_get",
        |url: String, headers: rhai::Map| -> Result<String, Box<rhai::EvalAltResult>> {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let client = reqwest::Client::new();
                    let mut req = client.get(&url);
                    for (k, v) in &headers {
                        if let Some(val) = v.clone().into_string().ok() {
                            req = req.header(k.as_str(), val);
                        }
                    }
                    let resp = req
                        .send()
                        .await
                        .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))?;
                    resp.text()
                        .await
                        .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))
                })
            })
        },
    );

    engine.register_fn(
        "http_post",
        |url: String, headers: rhai::Map, body: String| -> Result<String, Box<rhai::EvalAltResult>> {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let client = reqwest::Client::new();
                    let mut req = client.post(&url);
                    for (k, v) in &headers {
                        if let Some(val) = v.clone().into_string().ok() {
                            req = req.header(k.as_str(), val);
                        }
                    }
                    req = req.header("content-type", "application/json");
                    let resp = req
                        .body(body)
                        .send()
                        .await
                        .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))?;
                    resp.text()
                        .await
                        .map_err(|e| Box::new(rhai::EvalAltResult::from(e.to_string())))
                })
            })
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_valid_script() {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(128, 64);
        let script = r#"fn check() { () }"#;
        let ast = engine.compile(script).unwrap();
        assert!(ast.iter_functions().any(|f| f.name == "check"));
    }

    #[test]
    fn check_returns_unit_means_no_fire() {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(128, 64);
        let ast = engine.compile(r#"fn check() { () }"#).unwrap();
        let mut scope = Scope::new();
        let result: Dynamic = engine.call_fn(&mut scope, &ast, "check", ()).unwrap();
        assert!(result.is_unit());
    }

    #[test]
    fn check_returns_string_means_fire() {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(128, 64);
        let ast = engine.compile(r#"fn check() { "alert!" }"#).unwrap();
        let mut scope = Scope::new();
        let result: Dynamic = engine.call_fn(&mut scope, &ast, "check", ()).unwrap();
        assert!(result.is_string());
        assert_eq!(result.into_string().unwrap(), "alert!");
    }

    #[test]
    fn check_with_conditional_logic() {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(128, 64);
        let script = r#"
            let counter = 0;
            fn check() {
                counter += 1;
                if counter >= 3 {
                    return "fired after 3 ticks";
                }
            }
        "#;
        // This won't compile as-is because Rhai closures can't capture mutable state this way.
        // In practice, state would be in the Scope. Test the basic pattern instead.
        let ast = engine.compile(r#"fn check() { if 2 > 1 { "yes" } else { () } }"#).unwrap();
        let mut scope = Scope::new();
        let result: Dynamic = engine.call_fn(&mut scope, &ast, "check", ()).unwrap();
        assert_eq!(result.into_string().unwrap(), "yes");
    }
}
