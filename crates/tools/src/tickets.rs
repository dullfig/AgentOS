//! TicketsTool — async work-item store.
//!
//! Defer-and-return pattern: an agent encounters something that needs
//! follow-up but doesn't want to block its primary task. It files a
//! ticket and moves on; a later agent (or human) picks the ticket up
//! out of band.
//!
//! Origin: coding-expert's test loop wanted to log bugs and continue
//! rather than stop on every failure. The pattern generalizes —
//! QA-expert files cortex/ringhub anomalies the same way, columbo
//! (when it lands) will file asymmetric-gap reports the same way.
//!
//! ## Storage
//! Filesystem-backed: one JSON file per ticket under `{root}/tickets/`.
//! The directory listing IS the index. Each filename starts with a
//! sortable timestamp prefix, so naive `read_dir + sort` gives
//! chronological order.
//!
//! The original design seed (`[[roadmap]]` Track 4) puts tickets on
//! top of a thread-scoped KV store. That's still the target — when KV
//! lands, this tool can switch backends without changing its tool-side
//! contract. Filesystem is the unblock-now path.
//!
//! ## Tool surface (action-dispatched, like shim-store / cortex-shim)
//!
//! | action   | inputs                                            | output                |
//! |----------|---------------------------------------------------|-----------------------|
//! | `file`   | title, body, severity, tags, by                   | ticket_id             |
//! | `list`   | optional status / tag / severity / limit          | array of ticket summaries |
//! | `get`    | ticket_id                                         | full ticket JSON      |
//! | `update` | ticket_id, status, by, optional note              | updated ticket JSON   |
//!
//! Severities: `info`, `warn`, `error`, `critical`.
//! Statuses: `open`, `claimed`, `done`, `failed`.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_pipeline::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;

use super::{extract_tag, ToolPeer, ToolResponse};

const TICKETS_SUBDIR: &str = "tickets";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warn,
    Error,
    Critical,
}

impl Severity {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            "critical" | "crit" => Ok(Self::Critical),
            other => Err(format!(
                "unknown severity '{other}'; expected info|warn|error|critical"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Open,
    Claimed,
    Done,
    Failed,
}

impl Status {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "claimed" => Ok(Self::Claimed),
            "done" => Ok(Self::Done),
            "failed" => Ok(Self::Failed),
            other => Err(format!(
                "unknown status '{other}'; expected open|claimed|done|failed"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at: DateTime<Utc>,
    pub by: String,
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticket {
    pub id: String,
    pub title: String,
    pub body: String,
    pub severity: Severity,
    pub status: Status,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub updated_at: DateTime<Utc>,
    pub history: Vec<HistoryEntry>,
}

/// Subset of fields returned by `list` — keeps response sizes bounded
/// when there are many tickets.
#[derive(Debug, Clone, Serialize)]
pub struct TicketSummary {
    pub id: String,
    pub title: String,
    pub severity: Severity,
    pub status: Status,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<&Ticket> for TicketSummary {
    fn from(t: &Ticket) -> Self {
        Self {
            id: t.id.clone(),
            title: t.title.clone(),
            severity: t.severity,
            status: t.status,
            tags: t.tags.clone(),
            created_at: t.created_at,
            updated_at: t.updated_at,
        }
    }
}

/// Filesystem-backed ticket store. All operations are sync I/O behind
/// an async lock so concurrent agents writing tickets serialize cleanly
/// (filing is rare relative to other work; contention isn't a concern).
pub struct TicketStore {
    dir: PathBuf,
    lock: Mutex<()>,
}

impl TicketStore {
    /// Open (or create) a ticket store rooted at `{root}/tickets/`.
    pub fn open(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = root.into().join(TICKETS_SUBDIR);
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
        })
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    async fn file_ticket(
        &self,
        title: String,
        body: String,
        severity: Severity,
        tags: Vec<String>,
        by: String,
    ) -> Result<Ticket, String> {
        let _g = self.lock.lock().await;
        let now = Utc::now();
        // ID prefix: timestamp for sort, short UUID suffix for uniqueness.
        let suffix = uuid::Uuid::new_v4().simple().to_string()[..6].to_string();
        let id = format!("tk-{}-{suffix}", now.format("%Y%m%dT%H%M%S"));

        let ticket = Ticket {
            id: id.clone(),
            title,
            body,
            severity,
            status: Status::Open,
            tags,
            created_at: now,
            created_by: by.clone(),
            updated_at: now,
            history: vec![HistoryEntry {
                at: now,
                by,
                action: "filed".into(),
                note: None,
            }],
        };

        self.write_ticket(&ticket)?;
        Ok(ticket)
    }

    async fn get(&self, id: &str) -> Result<Ticket, String> {
        let _g = self.lock.lock().await;
        self.read_ticket(id)
    }

    async fn list(
        &self,
        status: Option<Status>,
        tag: Option<String>,
        severity: Option<Severity>,
        limit: usize,
    ) -> Result<Vec<TicketSummary>, String> {
        let _g = self.lock.lock().await;
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .map_err(|e| format!("list dir: {e}"))?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        // Newest first (timestamp prefix is sortable in lexical order).
        entries.sort();
        entries.reverse();

        let mut out = Vec::new();
        for path in entries {
            let id = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let t = match self.read_ticket(&id) {
                Ok(t) => t,
                // Corrupted or partially-written tickets shouldn't fail
                // the entire list. Tracing the skip is enough.
                Err(e) => {
                    tracing::warn!(ticket = %id, error = %e, "skipping unreadable ticket");
                    continue;
                }
            };
            if let Some(s) = status {
                if t.status != s {
                    continue;
                }
            }
            if let Some(s) = severity {
                if t.severity != s {
                    continue;
                }
            }
            if let Some(tg) = &tag {
                if !t.tags.iter().any(|x| x == tg) {
                    continue;
                }
            }
            out.push(TicketSummary::from(&t));
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    async fn update(
        &self,
        id: &str,
        status: Status,
        by: String,
        note: Option<String>,
    ) -> Result<Ticket, String> {
        let _g = self.lock.lock().await;
        let mut t = self.read_ticket(id)?;
        let now = Utc::now();
        let action = format!("{:?}", status).to_lowercase();
        t.status = status;
        t.updated_at = now;
        t.history.push(HistoryEntry {
            at: now,
            by,
            action,
            note,
        });
        self.write_ticket(&t)?;
        Ok(t)
    }

    fn write_ticket(&self, t: &Ticket) -> Result<(), String> {
        let path = self.dir.join(format!("{}.json", t.id));
        let bytes = serde_json::to_vec_pretty(t)
            .map_err(|e| format!("serialize ticket: {e}"))?;
        // Write to temp + rename for atomicity. Crash mid-write doesn't
        // leave a half-formed JSON file in the listing.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("rename {} → {}: {e}", tmp.display(), path.display()))?;
        Ok(())
    }

    fn read_ticket(&self, id: &str) -> Result<Ticket, String> {
        if !is_safe_id(id) {
            return Err(format!("invalid ticket id: {id}"));
        }
        let path = self.dir.join(format!("{id}.json"));
        let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!("ticket not found: {id}"),
            _ => format!("read {}: {e}", path.display()),
        })?;
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
    }
}

/// Reject ticket IDs containing path separators or `..`. Belt-and-
/// suspenders even though our own `file_ticket` only produces sanitized
/// IDs — agents can pass arbitrary strings to `get`/`update`.
fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() < 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn parse_tags(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[derive(Clone)]
pub struct TicketsTool {
    store: Arc<TicketStore>,
}

impl TicketsTool {
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        Ok(Self {
            store: Arc::new(TicketStore::open(root)?),
        })
    }

    pub fn with_store(store: Arc<TicketStore>) -> Self {
        Self { store }
    }

    async fn handle_file(&self, xml: &str) -> Result<String, String> {
        let title = extract_tag(xml, "title")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing required <title>".to_string())?;
        let body = extract_tag(xml, "body").unwrap_or_default();
        let severity = match extract_tag(xml, "severity") {
            Some(s) if !s.is_empty() => Severity::parse(&s)?,
            _ => Severity::Info,
        };
        let tags = parse_tags(&extract_tag(xml, "tags").unwrap_or_default());
        let by = extract_tag(xml, "by")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string());

        let t = self.store.file_ticket(title, body, severity, tags, by).await?;
        Ok(json!({
            "ticket_id": t.id,
            "status": t.status,
            "created_at": t.created_at,
        })
        .to_string())
    }

    async fn handle_list(&self, xml: &str) -> Result<String, String> {
        let status = match extract_tag(xml, "status") {
            Some(s) if !s.is_empty() => Some(Status::parse(&s)?),
            _ => None,
        };
        let severity = match extract_tag(xml, "severity") {
            Some(s) if !s.is_empty() => Some(Severity::parse(&s)?),
            _ => None,
        };
        let tag = extract_tag(xml, "tag").filter(|s| !s.is_empty());
        let limit = extract_tag(xml, "limit")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(50)
            .clamp(1, 500);

        let summaries = self.store.list(status, tag, severity, limit).await?;
        serde_json::to_string(&json!({
            "tickets": summaries,
            "count": summaries.len(),
        }))
        .map_err(|e| format!("serialize list: {e}"))
    }

    async fn handle_get(&self, xml: &str) -> Result<String, String> {
        let id = extract_tag(xml, "ticket_id")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing required <ticket_id>".to_string())?;
        let t = self.store.get(&id).await?;
        serde_json::to_string(&t).map_err(|e| format!("serialize ticket: {e}"))
    }

    async fn handle_update(&self, xml: &str) -> Result<String, String> {
        let id = extract_tag(xml, "ticket_id")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing required <ticket_id>".to_string())?;
        let status = extract_tag(xml, "status")
            .filter(|s| !s.is_empty())
            .map(|s| Status::parse(&s))
            .ok_or_else(|| "missing required <status>".to_string())??;
        let by = extract_tag(xml, "by")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let note = extract_tag(xml, "note").filter(|s| !s.is_empty());

        let t = self.store.update(&id, status, by, note).await?;
        serde_json::to_string(&t).map_err(|e| format!("serialize ticket: {e}"))
    }
}

#[async_trait]
impl Handler for TicketsTool {
    async fn handle(&self, payload: ValidatedPayload, _ctx: HandlerContext) -> HandlerResult {
        let xml = String::from_utf8_lossy(&payload.xml).to_string();
        let action = extract_tag(&xml, "action").unwrap_or_default();

        let result = match action.as_str() {
            "file" => self.handle_file(&xml).await,
            "list" => self.handle_list(&xml).await,
            "get" => self.handle_get(&xml).await,
            "update" => self.handle_update(&xml).await,
            "" => Err("missing required <action>".to_string()),
            other => Err(format!(
                "unknown action '{other}'; expected file|list|get|update"
            )),
        };

        Ok(HandlerResponse::Reply {
            payload_xml: match result {
                Ok(body) => ToolResponse::ok(&body),
                Err(e) => ToolResponse::err(&e),
            },
        })
    }
}

#[async_trait]
impl ToolPeer for TicketsTool {
    fn name(&self) -> &str {
        "tickets"
    }

    fn wit(&self) -> &str {
        r#"
/// Async work-item store: file/list/get/update tickets.
///
/// Tickets are filed by agents that hit something needing follow-up
/// but want to keep working. Severities: info|warn|error|critical.
/// Statuses: open|claimed|done|failed. Storage is filesystem-backed
/// (one JSON file per ticket); migrates to KV-backed when that lands.
interface tickets {
    record file-request {
        title: string,
        body: option<string>,
        /// info|warn|error|critical (default info)
        severity: option<string>,
        /// comma-separated tags
        tags: option<string>,
        /// filer's identity (agent name); defaults to "unknown"
        by: option<string>,
    }
    record list-request {
        /// open|claimed|done|failed — omit for all
        status: option<string>,
        /// match a single tag
        tag: option<string>,
        severity: option<string>,
        /// max results (1-500, default 50)
        limit: option<u64>,
    }
    record get-request { ticket-id: string }
    record update-request {
        ticket-id: string,
        status: string,
        by: option<string>,
        note: option<string>,
    }
    /// Action dispatched via <action> tag; payload tag is "Tickets".
    invoke: func(action: string, req: string) -> result<string, string>;
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::TempDir;

    fn make_ctx() -> HandlerContext {
        HandlerContext {
            thread_id: "t1".into(),
            from: "agent".into(),
            own_name: "tickets".into(),
        }
    }

    fn make_payload(xml: &str) -> ValidatedPayload {
        ValidatedPayload {
            xml: xml.as_bytes().to_vec(),
            tag: "Tickets".into(),
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

    async fn boot_tool() -> (TempDir, TicketsTool) {
        let dir = TempDir::new().unwrap();
        let tool = TicketsTool::new(dir.path()).unwrap();
        (dir, tool)
    }

    #[tokio::test]
    async fn file_creates_ticket_returns_id() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>file</action><title>test failure on auth_login</title><body>got timeout after 5s</body><severity>error</severity><tags>tests,auth</tags><by>coder</by></Tickets>";
        let (ok, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(ok, "{body}");
        let v = parse_ok_json(&body);
        assert!(v["ticket_id"].as_str().unwrap().starts_with("tk-"));
        assert_eq!(v["status"], "open");
    }

    #[tokio::test]
    async fn missing_title_errors() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>file</action></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<title>"), "got: {msg}");
    }

    #[tokio::test]
    async fn invalid_severity_errors() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>file</action><title>x</title><severity>banana</severity></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("severity"), "got: {msg}");
    }

    #[tokio::test]
    async fn get_returns_full_ticket_including_history() {
        let (_dir, tool) = boot_tool().await;
        let file_xml = "<Tickets><action>file</action><title>cortex latency</title><body>p99 spiked</body><severity>warn</severity><by>qa-expert</by></Tickets>";
        let (ok, body) = parse_response(tool.handle(make_payload(file_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let id = parse_ok_json(&body)["ticket_id"].as_str().unwrap().to_string();

        let get_xml = format!("<Tickets><action>get</action><ticket_id>{id}</ticket_id></Tickets>");
        let (ok, body) =
            parse_response(tool.handle(make_payload(&get_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let v = parse_ok_json(&body);
        assert_eq!(v["id"], id);
        assert_eq!(v["title"], "cortex latency");
        assert_eq!(v["body"], "p99 spiked");
        assert_eq!(v["severity"], "warn");
        assert_eq!(v["status"], "open");
        assert_eq!(v["created_by"], "qa-expert");
        let history = v["history"].as_array().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["action"], "filed");
    }

    #[tokio::test]
    async fn get_nonexistent_errors() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>get</action><ticket_id>tk-00000000T000000-zzzzzz</ticket_id></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[tokio::test]
    async fn get_rejects_path_traversal_id() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>get</action><ticket_id>../../etc/passwd</ticket_id></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("invalid"), "got: {msg}");
    }

    #[tokio::test]
    async fn list_returns_filed_tickets_newest_first() {
        let (_dir, tool) = boot_tool().await;
        for i in 0..3 {
            let xml = format!(
                "<Tickets><action>file</action><title>ticket {i}</title><severity>info</severity></Tickets>"
            );
            parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
            // Sleep enough that timestamp-prefix ordering is unambiguous
            // (resolution is seconds).
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        }

        let list_xml = "<Tickets><action>list</action></Tickets>";
        let (ok, body) =
            parse_response(tool.handle(make_payload(list_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let v = parse_ok_json(&body);
        let tickets = v["tickets"].as_array().unwrap();
        assert_eq!(tickets.len(), 3);
        // Newest first → last-filed title is first.
        assert_eq!(tickets[0]["title"], "ticket 2");
        assert_eq!(tickets[2]["title"], "ticket 0");
    }

    #[tokio::test]
    async fn list_filters_by_status() {
        let (_dir, tool) = boot_tool().await;
        let file_xml = "<Tickets><action>file</action><title>one</title></Tickets>";
        let (ok, body) =
            parse_response(tool.handle(make_payload(file_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let id = parse_ok_json(&body)["ticket_id"].as_str().unwrap().to_string();

        let update_xml = format!(
            "<Tickets><action>update</action><ticket_id>{id}</ticket_id><status>done</status><by>coder</by></Tickets>"
        );
        parse_response(tool.handle(make_payload(&update_xml), make_ctx()).await.unwrap());

        let open_only = "<Tickets><action>list</action><status>open</status></Tickets>";
        let (_, body) =
            parse_response(tool.handle(make_payload(open_only), make_ctx()).await.unwrap());
        assert_eq!(parse_ok_json(&body)["tickets"].as_array().unwrap().len(), 0);

        let done_only = "<Tickets><action>list</action><status>done</status></Tickets>";
        let (_, body) =
            parse_response(tool.handle(make_payload(done_only), make_ctx()).await.unwrap());
        assert_eq!(parse_ok_json(&body)["tickets"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn list_filters_by_tag() {
        let (_dir, tool) = boot_tool().await;
        let a = "<Tickets><action>file</action><title>cortex thing</title><tags>cortex,latency</tags></Tickets>";
        let b = "<Tickets><action>file</action><title>memex thing</title><tags>memex</tags></Tickets>";
        parse_response(tool.handle(make_payload(a), make_ctx()).await.unwrap());
        parse_response(tool.handle(make_payload(b), make_ctx()).await.unwrap());

        let xml = "<Tickets><action>list</action><tag>cortex</tag></Tickets>";
        let (_, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        let tickets = parse_ok_json(&body)["tickets"].as_array().unwrap().clone();
        assert_eq!(tickets.len(), 1);
        assert_eq!(tickets[0]["title"], "cortex thing");
    }

    #[tokio::test]
    async fn update_appends_history_and_changes_status() {
        let (_dir, tool) = boot_tool().await;
        let file_xml = "<Tickets><action>file</action><title>regression in shim test</title><by>coder</by></Tickets>";
        let (ok, body) =
            parse_response(tool.handle(make_payload(file_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let id = parse_ok_json(&body)["ticket_id"].as_str().unwrap().to_string();

        let claim_xml = format!(
            "<Tickets><action>update</action><ticket_id>{id}</ticket_id><status>claimed</status><by>coder-v2</by><note>looking now</note></Tickets>"
        );
        let (ok, body) =
            parse_response(tool.handle(make_payload(&claim_xml), make_ctx()).await.unwrap());
        assert!(ok);
        let v = parse_ok_json(&body);
        assert_eq!(v["status"], "claimed");
        let history = v["history"].as_array().unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1]["action"], "claimed");
        assert_eq!(history[1]["by"], "coder-v2");
        assert_eq!(history[1]["note"], "looking now");
    }

    #[tokio::test]
    async fn update_missing_status_errors() {
        let (_dir, tool) = boot_tool().await;
        let file_xml = "<Tickets><action>file</action><title>x</title></Tickets>";
        let (_, body) =
            parse_response(tool.handle(make_payload(file_xml), make_ctx()).await.unwrap());
        let id = parse_ok_json(&body)["ticket_id"].as_str().unwrap().to_string();

        let bad = format!("<Tickets><action>update</action><ticket_id>{id}</ticket_id></Tickets>");
        let (ok, msg) =
            parse_response(tool.handle(make_payload(&bad), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<status>"), "got: {msg}");
    }

    #[tokio::test]
    async fn unknown_action_errors() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets><action>delete</action></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("unknown action"), "got: {msg}");
    }

    #[tokio::test]
    async fn missing_action_errors() {
        let (_dir, tool) = boot_tool().await;
        let xml = "<Tickets></Tickets>";
        let (ok, msg) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert!(!ok);
        assert!(msg.contains("<action>"), "got: {msg}");
    }

    #[tokio::test]
    async fn list_respects_limit() {
        let (_dir, tool) = boot_tool().await;
        for i in 0..5 {
            let xml = format!(
                "<Tickets><action>file</action><title>t{i}</title></Tickets>"
            );
            parse_response(tool.handle(make_payload(&xml), make_ctx()).await.unwrap());
        }
        let xml = "<Tickets><action>list</action><limit>2</limit></Tickets>";
        let (_, body) = parse_response(tool.handle(make_payload(xml), make_ctx()).await.unwrap());
        assert_eq!(parse_ok_json(&body)["tickets"].as_array().unwrap().len(), 2);
    }
}
