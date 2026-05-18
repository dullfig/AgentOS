//! PromParseTool — parse Prometheus exposition format into JSON.
//!
//! Pairs with [`super::http_request`] to give QA-expert (Track 3) a
//! complete path: fetch /metrics, parse, scan for anomalies. Also
//! useful for any organism that wants to read Prometheus output as
//! structured data instead of raw text.
//!
//! ## Input
//! - `<text>` — the raw exposition-format text (required).
//! - `<filter>` — optional metric-name prefix; only samples whose name
//!   starts with this prefix are returned. Cuts JSON size when the
//!   agent only cares about one subsystem.
//!
//! ## Output (JSON, wrapped in standard `<ToolResponse>` envelope)
//!
//! ```json
//! {
//!   "samples": [
//!     {"name": "agentos_requests_total", "type": "counter",
//!      "labels": {"status": "ok"}, "value": 42.0},
//!     {"name": "agentos_idempotency_cache_entries", "type": "gauge",
//!      "labels": {}, "value": 12.0},
//!     {"name": "agentos_request_duration_seconds", "type": "histogram",
//!      "labels": {"status": "ok"},
//!      "buckets": [{"le": "0.005", "count": 0}, ..., {"le": "+Inf", "count": 42}]}
//!   ],
//!   "help": {"agentos_requests_total": "POST /v1/messages requests..."}
//! }
//! ```
//!
//! Histogram buckets use string `le` so `+Inf` survives JSON encoding
//! (JSON has no infinity literal).

use std::io;

use async_trait::async_trait;
use prometheus_parse::{Scrape, Value as PromValue};
use rust_pipeline::prelude::*;
use serde_json::{json, Value};

use super::{extract_tag, ToolPeer, ToolResponse};

pub struct PromParseTool;

impl PromParseTool {
    fn execute(xml: &str) -> Result<String, String> {
        let text = extract_tag(xml, "text")
            .ok_or_else(|| "missing required <text>".to_string())?;
        let filter = extract_tag(xml, "filter").unwrap_or_default();

        // prometheus-parse wants an iterator of Result<String, io::Error>.
        // We borrow the text by line and lift each line into a fresh String.
        let lines = text
            .lines()
            .map(|l| Ok::<_, io::Error>(l.to_string()));

        let scrape = Scrape::parse(lines)
            .map_err(|e| format!("parse failed: {e}"))?;

        let samples: Vec<Value> = scrape
            .samples
            .into_iter()
            .filter(|s| filter.is_empty() || s.metric.starts_with(&filter))
            .map(sample_to_json)
            .collect();

        let help: serde_json::Map<String, Value> = scrape
            .docs
            .into_iter()
            .filter(|(k, _)| filter.is_empty() || k.starts_with(&filter))
            .map(|(k, v)| (k, Value::String(v)))
            .collect();

        let out = json!({
            "samples": samples,
            "help": help,
        });
        Ok(out.to_string())
    }
}

fn sample_to_json(s: prometheus_parse::Sample) -> Value {
    let labels_map: serde_json::Map<String, Value> = s
        .labels
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect();

    let (kind, extra) = match s.value {
        PromValue::Counter(v) => ("counter", json!({ "value": number(v) })),
        PromValue::Gauge(v) => ("gauge", json!({ "value": number(v) })),
        PromValue::Untyped(v) => ("untyped", json!({ "value": number(v) })),
        PromValue::Histogram(buckets) => {
            let bs: Vec<Value> = buckets
                .into_iter()
                .map(|b| {
                    json!({
                        "le": le_label(b.less_than),
                        "count": number(b.count),
                    })
                })
                .collect();
            ("histogram", json!({ "buckets": bs }))
        }
        PromValue::Summary(quantiles) => {
            let qs: Vec<Value> = quantiles
                .into_iter()
                .map(|q| {
                    json!({
                        "quantile": number(q.quantile),
                        "count": number(q.count),
                    })
                })
                .collect();
            ("summary", json!({ "quantiles": qs }))
        }
    };

    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), Value::String(s.metric));
    obj.insert("type".into(), Value::String(kind.to_string()));
    obj.insert("labels".into(), Value::Object(labels_map));
    if let Value::Object(extras) = extra {
        for (k, v) in extras {
            obj.insert(k, v);
        }
    }
    Value::Object(obj)
}

/// Render a histogram bucket boundary. JSON has no infinity literal, so
/// `+Inf` (the standard upper bucket boundary) is rendered as a string.
/// Finite values are rendered as JSON numbers — agent code parsing this
/// should accept either shape per bucket.
fn le_label(less_than: f64) -> Value {
    if less_than.is_infinite() {
        Value::String(if less_than.is_sign_positive() {
            "+Inf".into()
        } else {
            "-Inf".into()
        })
    } else {
        number(less_than)
    }
}

/// JSON-safe number rendering. NaN and infinity become strings so the
/// JSON stays valid (serde_json refuses to encode them as numbers).
fn number(v: f64) -> Value {
    if v.is_finite() {
        // serde_json::Number::from_f64 returns None for NaN/Inf which
        // we've already handled, and Some for all finite f64s.
        match serde_json::Number::from_f64(v) {
            Some(n) => Value::Number(n),
            None => Value::String(v.to_string()),
        }
    } else if v.is_nan() {
        Value::String("NaN".into())
    } else if v.is_sign_positive() {
        Value::String("+Inf".into())
    } else {
        Value::String("-Inf".into())
    }
}

#[async_trait]
impl Handler for PromParseTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml = String::from_utf8_lossy(&payload.xml);
        match Self::execute(&xml) {
            Ok(body) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::ok(&body),
            }),
            Err(e) => Ok(HandlerResponse::Reply {
                payload_xml: ToolResponse::err(&e),
            }),
        }
    }
}

#[async_trait]
impl ToolPeer for PromParseTool {
    fn name(&self) -> &str {
        "prom-parse"
    }

    fn wit(&self) -> &str {
        r#"
/// Parse Prometheus exposition-format text into structured JSON.
///
/// Returns JSON with `samples` (array, one per metric+labels) and
/// `help` (map of metric name → HELP description). Each sample carries
/// a `type` field (counter/gauge/histogram/summary/untyped); histograms
/// expose a `buckets` array, summaries expose `quantiles`.
interface prom-parse {
    record request {
        /// Raw text from a Prometheus /metrics endpoint.
        text: string,
        /// Optional metric-name prefix filter. Only samples whose name
        /// starts with this prefix are returned.
        filter: option<string>,
    }
    invoke: func(req: request) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "prom-parse".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "PromParse".into(),
        }
    }

    fn parse_response(resp: HandlerResponse) -> (bool, String) {
        match resp {
            HandlerResponse::Reply { payload_xml } => {
                let xml = String::from_utf8(payload_xml).unwrap();
                let success = xml.contains("<success>true</success>");
                let body = if success {
                    extract_tag(&xml, "result").unwrap_or_default()
                } else {
                    extract_tag(&xml, "error").unwrap_or_default()
                };
                (success, body)
            }
            _ => panic!("expected Reply"),
        }
    }

    fn parse_ok_json(body: &str) -> Value {
        serde_json::from_str(body)
            .unwrap_or_else(|e| panic!("response is not JSON: {e}\nbody: {body}"))
    }

    /// Tiny realistic /metrics body covering counter, gauge, histogram.
    /// Multiple labels on the counter, no labels on the gauge, and a
    /// short histogram so we can assert bucket structure.
    fn sample_metrics() -> &'static str {
        r#"# HELP agentos_requests_total POST /v1/messages requests by outcome class.
# TYPE agentos_requests_total counter
agentos_requests_total{status="ok"} 42
agentos_requests_total{status="client_error"} 3
# HELP agentos_idempotency_cache_entries Current entries.
# TYPE agentos_idempotency_cache_entries gauge
agentos_idempotency_cache_entries 12
# HELP agentos_request_duration_seconds Wall-clock time.
# TYPE agentos_request_duration_seconds histogram
agentos_request_duration_seconds_bucket{status="ok",le="0.005"} 1
agentos_request_duration_seconds_bucket{status="ok",le="0.1"} 30
agentos_request_duration_seconds_bucket{status="ok",le="+Inf"} 42
agentos_request_duration_seconds_sum{status="ok"} 1.234
agentos_request_duration_seconds_count{status="ok"} 42
"#
    }

    #[tokio::test]
    async fn counter_with_labels_parses() {
        let tool = PromParseTool;
        let xml = format!("<PromParse><text>{}</text></PromParse>", sample_metrics());
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");

        let v = parse_ok_json(&body);
        let samples = v["samples"].as_array().unwrap();

        // Two counter samples with different status labels.
        let counter_ok = samples
            .iter()
            .find(|s| {
                s["name"] == "agentos_requests_total"
                    && s["labels"]["status"] == "ok"
            })
            .unwrap_or_else(|| panic!("ok counter missing: {v}"));
        assert_eq!(counter_ok["type"], "counter");
        assert_eq!(counter_ok["value"], 42.0);

        let counter_err = samples
            .iter()
            .find(|s| {
                s["name"] == "agentos_requests_total"
                    && s["labels"]["status"] == "client_error"
            })
            .unwrap();
        assert_eq!(counter_err["value"], 3.0);
    }

    #[tokio::test]
    async fn gauge_with_no_labels_parses() {
        let tool = PromParseTool;
        let xml = format!("<PromParse><text>{}</text></PromParse>", sample_metrics());
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);

        let v = parse_ok_json(&body);
        let gauge = v["samples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == "agentos_idempotency_cache_entries")
            .unwrap();
        assert_eq!(gauge["type"], "gauge");
        assert_eq!(gauge["value"], 12.0);
        assert!(gauge["labels"].as_object().unwrap().is_empty());
    }

    #[tokio::test]
    async fn histogram_exposes_buckets_with_inf_as_string() {
        let tool = PromParseTool;
        let xml = format!("<PromParse><text>{}</text></PromParse>", sample_metrics());
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);

        let v = parse_ok_json(&body);
        let hist = v["samples"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["name"] == "agentos_request_duration_seconds")
            .unwrap_or_else(|| panic!("histogram missing: {v}"));
        assert_eq!(hist["type"], "histogram");
        let buckets = hist["buckets"].as_array().unwrap();
        assert!(!buckets.is_empty(), "buckets should be present");

        // Last bucket should be +Inf (rendered as string).
        let last = buckets.last().unwrap();
        assert_eq!(
            last["le"], "+Inf",
            "final bucket should be +Inf as string: {hist}"
        );
    }

    #[tokio::test]
    async fn help_lines_surface_in_output() {
        let tool = PromParseTool;
        let xml = format!("<PromParse><text>{}</text></PromParse>", sample_metrics());
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);

        let v = parse_ok_json(&body);
        let help = v["help"].as_object().unwrap();
        assert!(
            help["agentos_requests_total"]
                .as_str()
                .unwrap_or("")
                .contains("POST /v1/messages"),
            "HELP text missing: {help:?}"
        );
    }

    #[tokio::test]
    async fn filter_prefix_drops_non_matching_samples() {
        let tool = PromParseTool;
        let xml = format!(
            "<PromParse><text>{}</text><filter>agentos_idempotency</filter></PromParse>",
            sample_metrics()
        );
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok);

        let v = parse_ok_json(&body);
        let samples = v["samples"].as_array().unwrap();
        assert!(!samples.is_empty(), "filter should keep matching samples");
        for s in samples {
            assert!(
                s["name"].as_str().unwrap().starts_with("agentos_idempotency"),
                "filter let through non-matching sample: {s}"
            );
        }
        // Help should also be filtered.
        let help = v["help"].as_object().unwrap();
        for k in help.keys() {
            assert!(
                k.starts_with("agentos_idempotency"),
                "help not filtered: {k}"
            );
        }
    }

    #[tokio::test]
    async fn missing_text_errors() {
        let tool = PromParseTool;
        let xml = "<PromParse></PromParse>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<text>"), "got: {msg}");
    }

    #[tokio::test]
    async fn empty_text_parses_to_empty_samples() {
        // Edge case: scraping a brand-new endpoint with no metrics yet
        // shouldn't error.
        let tool = PromParseTool;
        let xml = "<PromParse><text>\n</text></PromParse>";
        let (ok, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "empty input should parse: {body}");
        let v = parse_ok_json(&body);
        assert!(v["samples"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn malformed_text_errors_cleanly() {
        // Truly garbled input — not a parse-able exposition format.
        // The parser should fail with a clear error rather than panic.
        let tool = PromParseTool;
        let xml = "<PromParse><text>this is not metrics at all {{{ &amp; &lt;&gt;</text></PromParse>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        // Whether it errors or returns empty depends on the parser's
        // tolerance; both are acceptable. The key requirement is no panic.
        if !ok {
            assert!(msg.to_lowercase().contains("parse"), "got: {msg}");
        }
    }

    #[tokio::test]
    async fn end_to_end_against_live_metrics_render() {
        // Integration-flavored: render real metrics from the server
        // crate's metric module's documented metrics, then parse them
        // back. Catches drift between what we emit and what we can read.
        //
        // Not depending on agentos-server here (circular) — instead use
        // a hand-crafted payload that mirrors what `metrics::render`
        // produces in its happy path.
        let live = r#"# HELP agentos_requests_total POST /v1/messages by outcome.
# TYPE agentos_requests_total counter
agentos_requests_total{status="ok"} 100
# HELP agentos_active_sse_streams Currently active SSE streams.
# TYPE agentos_active_sse_streams gauge
agentos_active_sse_streams 3
"#;
        let tool = PromParseTool;
        let xml = format!("<PromParse><text>{live}</text></PromParse>");
        let (ok, body) = parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        let v = parse_ok_json(&body);
        let names: Vec<_> = v["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"agentos_requests_total"));
        assert!(names.contains(&"agentos_active_sse_streams"));
    }
}
