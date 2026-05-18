//! Prometheus `/metrics` exporter for agentos-server.
//!
//! Feeds the six-number capacity dashboard pinned in `roadmap.md`
//! under Track 3 (Observability & QA). Exposed at `GET /metrics` and
//! consumed by QA-expert (plus any standard Prometheus scraper).
//!
//! Initialization is idempotent — `init()` is safe to call from any
//! number of `build_router` invocations (one bin, many tests). The
//! recorder is installed globally on first call; subsequent calls
//! return the same handle.
//!
//! Metric naming follows Prometheus conventions: namespace prefix
//! (`agentos_`), snake_case, units in the suffix (`_seconds`,
//! `_total`). Labels are low-cardinality strings: HTTP status class,
//! idempotency result kind. No user IDs, no addresses — those would
//! blow up cardinality and leak privacy.

use std::sync::OnceLock;
use std::time::Duration;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Metric names. Constants rather than inline strings so the test
/// suite can assert on them without typo risk.
pub const REQUEST_DURATION_SECONDS: &str = "agentos_request_duration_seconds";
pub const REQUESTS_TOTAL: &str = "agentos_requests_total";
pub const IDEMPOTENCY_LOOKUPS_TOTAL: &str = "agentos_idempotency_lookups_total";
pub const IDEMPOTENCY_CACHE_ENTRIES: &str = "agentos_idempotency_cache_entries";
pub const IDEMPOTENCY_LRU_EVICTIONS_TOTAL: &str = "agentos_idempotency_lru_evictions_total";
pub const BROADCAST_LAG_TOTAL: &str = "agentos_broadcast_lag_total";
pub const ACTIVE_SSE_STREAMS: &str = "agentos_active_sse_streams";

/// Idempotency lookup outcome labels. Match `LookupResult` variants.
pub const RESULT_MISS: &str = "miss";
pub const RESULT_REPLAY: &str = "replay";
pub const RESULT_CONFLICT: &str = "conflict";
pub const RESULT_INFLIGHT: &str = "inflight";

/// Request status labels. Cardinality-bounded by deliberate choice:
/// 2xx → "ok", 4xx → "client_error", 5xx → "server_error". The exact
/// status code lives in tracing; histograms don't need it.
pub const STATUS_OK: &str = "ok";
pub const STATUS_CLIENT_ERROR: &str = "client_error";
pub const STATUS_SERVER_ERROR: &str = "server_error";

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the Prometheus recorder if it hasn't been already, and
/// return a static reference to the handle. Safe to call concurrently
/// and repeatedly — the first call wins, subsequent calls return the
/// same handle.
pub fn init() -> &'static PrometheusHandle {
    HANDLE.get_or_init(|| {
        let recorder = PrometheusBuilder::new()
            .build_recorder();
        let handle = recorder.handle();
        // `set_global_recorder` errors if a recorder was already installed
        // by some other code path (rare; effectively impossible in the
        // bin, possible in mixed test setups). If it fails we still return
        // a handle — it just won't see any new recordings. The /metrics
        // endpoint will render with whatever state the other recorder
        // accumulated, or empty if nothing was recorded. Either way we
        // don't crash the server over an observability concern.
        let _ = metrics::set_global_recorder(recorder);
        describe_all();
        handle
    })
}

/// Read-side handle to the recorder. Returns `None` if `init` was
/// never called (which can't actually happen since `build_router`
/// always calls it, but the type makes the dependency explicit).
pub fn handle() -> Option<&'static PrometheusHandle> {
    HANDLE.get()
}

/// Render the current metrics in Prometheus exposition format.
pub fn render() -> String {
    HANDLE.get().map(|h| h.render()).unwrap_or_default()
}

fn describe_all() {
    metrics::describe_histogram!(
        REQUEST_DURATION_SECONDS,
        metrics::Unit::Seconds,
        "Wall-clock time from request receipt to terminal SSE done event."
    );
    metrics::describe_counter!(
        REQUESTS_TOTAL,
        "POST /v1/messages requests by outcome class (ok / client_error / server_error)."
    );
    metrics::describe_counter!(
        IDEMPOTENCY_LOOKUPS_TOTAL,
        "Idempotency cache lookups by result (miss / replay / conflict / inflight)."
    );
    metrics::describe_gauge!(
        IDEMPOTENCY_CACHE_ENTRIES,
        "Current number of entries in the idempotency cache."
    );
    metrics::describe_counter!(
        IDEMPOTENCY_LRU_EVICTIONS_TOTAL,
        "Idempotency-cache entries dropped by LRU eviction when the \
         entry ceiling was breached. Non-zero indicates sustained insert \
         rate is outpacing 24h TTL — bump max_entries or shorten TTL."
    );
    metrics::describe_counter!(
        BROADCAST_LAG_TOTAL,
        "Times the SSE handler observed pipeline broadcast lag (a Lagged \
         recv error). Non-zero indicates the broadcast buffer was too small \
         to keep up with one subscriber."
    );
    metrics::describe_gauge!(
        ACTIVE_SSE_STREAMS,
        "Currently active SSE response streams. Tracks both live and replay paths."
    );
}

// ── recording helpers (called from handler.rs) ────────────────────────

pub fn record_request(status: &'static str, duration: Duration) {
    metrics::counter!(REQUESTS_TOTAL, "status" => status).increment(1);
    metrics::histogram!(REQUEST_DURATION_SECONDS, "status" => status)
        .record(duration.as_secs_f64());
}

pub fn record_idempotency_lookup(result: &'static str) {
    metrics::counter!(IDEMPOTENCY_LOOKUPS_TOTAL, "result" => result).increment(1);
}

pub fn record_idempotency_lru_evictions(n: usize) {
    metrics::counter!(IDEMPOTENCY_LRU_EVICTIONS_TOTAL).increment(n as u64);
}

pub fn record_broadcast_lag() {
    metrics::counter!(BROADCAST_LAG_TOTAL).increment(1);
}

pub fn set_idempotency_cache_entries(n: usize) {
    metrics::gauge!(IDEMPOTENCY_CACHE_ENTRIES).set(n as f64);
}

pub fn inc_active_sse_streams() {
    metrics::gauge!(ACTIVE_SSE_STREAMS).increment(1.0);
}

pub fn dec_active_sse_streams() {
    metrics::gauge!(ACTIVE_SSE_STREAMS).decrement(1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent() {
        let h1 = init();
        let h2 = init();
        // OnceLock guarantees same pointer.
        assert!(std::ptr::eq(h1, h2));
    }

    #[test]
    fn render_contains_described_metrics_after_recording() {
        init();
        record_request(STATUS_OK, Duration::from_millis(42));
        record_idempotency_lookup(RESULT_MISS);
        set_idempotency_cache_entries(7);

        let out = render();
        assert!(
            out.contains(REQUEST_DURATION_SECONDS),
            "render missing histogram: {out}"
        );
        assert!(out.contains(REQUESTS_TOTAL), "render missing counter: {out}");
        assert!(
            out.contains(IDEMPOTENCY_LOOKUPS_TOTAL),
            "render missing idem counter: {out}"
        );
        assert!(
            out.contains(IDEMPOTENCY_CACHE_ENTRIES),
            "render missing cache gauge: {out}"
        );
    }
}
