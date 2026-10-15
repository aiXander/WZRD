//! The WZRD hub (studio plan L4, package A1): the long-lived MCP server
//! process (`python -m wzrd_mcp`, streamable HTTP on `127.0.0.1:8787`) that
//! owns `library.db`, render jobs, the Studio selection and the spend cap.
//! The app and every Claude Code session talk to it; the app talks JSON on
//! `/api/*`.
//!
//! Lifecycle (L4, S12): the app **attaches** when `GET /api/hello` answers
//! as a WZRD hub, and otherwise **spawns** one — detached into its own
//! process group, so it outlives the app (in-flight renders finish; the hub
//! exits itself after 1 h idle). `hub.lock` (`{pid, port}`) guards against
//! a second copy: a lock whose pid is alive means a hub is still starting,
//! so we wait instead of spawning. A foreign answer on the port is a red
//! lamp ("port 8787 taken"), never a spawn. The spawn uses an absolute `uv`
//! path — a Finder-launched app has no shell `PATH`.
//!
//! A monitor thread re-checks `hello` every few seconds and emits
//! `hub:status` on every change; the TopBar lamp renders it. "Restart hub"
//! SIGTERMs the lock's pid and spawns a fresh one.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter};

pub const HUB_PORT: u16 = 8787;
const HELLO_TIMEOUT: Duration = Duration::from_millis(800);
/// How long a fresh spawn may take to answer `hello` (uv sync + imports).
const SPAWN_WAIT: Duration = Duration::from_secs(45);
const MONITOR_EVERY: Duration = Duration::from_secs(3);

/// The lamp: what the TopBar shows, pushed as `hub:status`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HubStatus {
    /// `up` · `starting` · `down` · `foreign` (port taken by something else).
    pub state: &'static str,
    pub pid: Option<u32>,
    pub version: Option<String>,
    pub media_root: String,
    /// Why it is down / foreign / failed to spawn.
    pub error: Option<String>,
}

pub struct Hub {
    status: Mutex<HubStatus>,
    /// Serialises spawn / restart.
    spawning: Mutex<()>,
}

/// `~/WZRD/media` (`WZRD_MEDIA_ROOT` overrides) — L1. Must match the hub's
/// `library.media_root()`.
pub fn media_root() -> PathBuf {
    match std::env::var("WZRD_MEDIA_ROOT") {
        Ok(p) if !p.is_empty() => expand_home(&p),
        _ => home().join("WZRD").join("media"),
    }
}

pub fn wzrd_dir() -> PathBuf {
    media_root().join(".wzrd")
}

pub fn db_path() -> PathBuf {
    wzrd_dir().join("library.db")
}

fn home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_default()
}

fn expand_home(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None => PathBuf::from(p),
    }
}

/// The repo holding `wzrd_mcp` — `WZRD_REPO`, else this checkout (the app
/// is built from it; a bundled app would set the env var).
fn repo_dir() -> PathBuf {
    if let Ok(p) = std::env::var("WZRD_REPO") {
        return expand_home(&p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(manifest)
}

/// Absolute `uv`: `WZRD_UV`, else the usual install spots, else bare `uv`
/// (works only when launched from a shell).
fn uv_path() -> PathBuf {
    if let Ok(p) = std::env::var("WZRD_UV") {
        return expand_home(&p);
    }
    let h = home();
    [
        h.join(".local/bin/uv"),
        h.join(".cargo/bin/uv"),
        PathBuf::from("/opt/homebrew/bin/uv"),
        PathBuf::from("/usr/local/bin/uv"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .unwrap_or_else(|| PathBuf::from("uv"))
}

// ---------------------------------------------------------------------------
// Minimal HTTP/1.1 client — loopback only, one request per connection.
// Dependency-free on purpose: every call is a small JSON body to our own
// hub on 127.0.0.1.
// ---------------------------------------------------------------------------

/// A hub reply: HTTP status + parsed JSON body (`Null` if empty / not JSON).
pub struct Reply {
    pub status: u16,
    pub body: Value,
}

pub fn request(
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<Reply, String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], HUB_PORT));
    let mut stream = TcpStream::connect_timeout(&addr, timeout.min(Duration::from_secs(2)))
        .map_err(|e| format!("hub unreachable: {e}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let payload = body.map(Value::to_string).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{HUB_PORT}\r\nAccept: application/json\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(payload.as_bytes()))
        .map_err(|e| format!("hub write: {e}"))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("hub read: {e}"))?;
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> Result<Reply, String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("hub reply has no header end")?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or("hub reply has no status line")?;
    let chunked = lines.clone().any(|l| {
        l.split_once(':').is_some_and(|(k, v)| {
            k.eq_ignore_ascii_case("transfer-encoding") && v.trim().eq_ignore_ascii_case("chunked")
        })
    });
    let mut body = raw[split + 4..].to_vec();
    if chunked {
        body = dechunk(&body);
    }
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    Ok(Reply { status, body })
}

fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(eol) = data.windows(2).position(|w| w == b"\r\n") else { break };
        let size = std::str::from_utf8(&data[..eol])
            .ok()
            .and_then(|s| usize::from_str_radix(s.split(';').next().unwrap_or("").trim(), 16).ok())
            .unwrap_or(0);
        data = &data[eol + 2..];
        if size == 0 || data.len() < size {
            break;
        }
        out.extend_from_slice(&data[..size]);
        data = data.get(size + 2..).unwrap_or(&[]);
    }
    out
}

/// What `GET /api/hello` says about port 8787.
enum Hello {
    Wzrd { pid: Option<u32>, version: Option<String> },
    Foreign(String),
    Nothing(String),
}

fn hello() -> Hello {
    match request("GET", "/api/hello", None, HELLO_TIMEOUT) {
        Ok(r) if r.status == 200 && r.body.get("app").and_then(Value::as_str) == Some("wzrd-hub") => {
            Hello::Wzrd {
                pid: r.body.get("pid").and_then(Value::as_u64).map(|p| p as u32),
                version: r.body.get("version").map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                }),
            }
        }
        Ok(r) => Hello::Foreign(format!(
            "port {HUB_PORT} is taken by something that isn't a WZRD hub (HTTP {})",
            r.status
        )),
        Err(e) => Hello::Nothing(e),
    }
}

/// `hub.lock`'s pid, if the lock exists and that process is alive.
fn live_lock_pid() -> Option<u32> {
    let text = std::fs::read_to_string(wzrd_dir().join("hub.lock")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let pid = v.get("pid")?.as_u64()? as u32;
    pid_alive(pid).then_some(pid)
}

fn pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // Signal 0: existence + permission check, sends nothing.
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

impl Hub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(HubStatus {
                state: "starting",
                pid: None,
                version: None,
                media_root: media_root().to_string_lossy().into_owned(),
                error: None,
            }),
            spawning: Mutex::new(()),
        })
    }

    pub fn status(&self) -> HubStatus {
        self.status.lock().expect("hub status lock").clone()
    }

    fn set(&self, app: &AppHandle, next: HubStatus) {
        let mut cur = self.status.lock().expect("hub status lock");
        if *cur != next {
            log::info!("hub: {} {}", next.state, next.error.as_deref().unwrap_or(""));
            *cur = next.clone();
            drop(cur);
            let _ = app.emit("hub:status", next);
        }
    }

    fn status_from(&self, h: &Hello) -> HubStatus {
        let media_root = media_root().to_string_lossy().into_owned();
        match h {
            Hello::Wzrd { pid, version } => HubStatus {
                state: "up",
                pid: *pid,
                version: version.clone(),
                media_root,
                error: None,
            },
            Hello::Foreign(e) => HubStatus {
                state: "foreign",
                pid: None,
                version: None,
                media_root,
                error: Some(e.clone()),
            },
            Hello::Nothing(e) => HubStatus {
                state: "down",
                pid: None,
                version: None,
                media_root,
                error: Some(e.clone()),
            },
        }
    }

    /// Attach or spawn (L4). Blocking — call off the main thread.
    pub fn ensure(&self, app: &AppHandle) {
        let _guard = self.spawning.lock().expect("hub spawn lock");
        let h = hello();
        if !matches!(h, Hello::Nothing(_)) {
            self.set(app, self.status_from(&h));
            return;
        }
        if let Some(pid) = live_lock_pid() {
            log::info!("hub: lock held by live pid {pid} — waiting for it to answer");
            self.wait_up(app, Some(pid));
            return;
        }
        if let Err(e) = spawn_detached() {
            self.set(app, HubStatus { state: "down", error: Some(e), ..self.status() });
            return;
        }
        self.wait_up(app, None);
    }

    fn wait_up(&self, app: &AppHandle, pid: Option<u32>) {
        self.set(app, HubStatus { state: "starting", pid, error: None, ..self.status() });
        let deadline = Instant::now() + SPAWN_WAIT;
        loop {
            let h = hello();
            if !matches!(h, Hello::Nothing(_)) {
                self.set(app, self.status_from(&h));
                return;
            }
            if Instant::now() > deadline {
                let log = wzrd_dir().join("hub.log");
                self.set(
                    app,
                    HubStatus {
                        state: "down",
                        error: Some(format!(
                            "hub did not answer within {} s — see {}",
                            SPAWN_WAIT.as_secs(),
                            log.display()
                        )),
                        ..self.status()
                    },
                );
                return;
            }
            std::thread::sleep(Duration::from_millis(400));
        }
    }

    /// SIGTERM the running hub (by `hello`'s pid, else the lock's), wait for
    /// the port to go quiet, spawn a fresh one.
    pub fn restart(&self, app: &AppHandle) -> Result<(), String> {
        {
            let _guard = self.spawning.lock().expect("hub spawn lock");
            let pid = match hello() {
                Hello::Wzrd { pid, .. } => pid,
                Hello::Foreign(e) => return Err(e),
                Hello::Nothing(_) => None,
            }
            .or_else(live_lock_pid);
            if let Some(pid) = pid {
                log::info!("hub: restarting — SIGTERM {pid}");
                let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
                let deadline = Instant::now() + Duration::from_secs(10);
                while (pid_alive(pid) || !matches!(hello(), Hello::Nothing(_)))
                    && Instant::now() < deadline
                {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
            self.set(app, HubStatus { state: "starting", pid: None, error: None, ..self.status() });
        }
        self.ensure(app);
        Ok(())
    }

    /// Re-check `hello` every few seconds for the app's life. Never spawns
    /// by itself — a hub that died (idle exit, crash) shows as down until
    /// the next `ensure` (Restart hub, or a Studio action that needs it).
    pub fn spawn_monitor(self: &Arc<Self>, app: AppHandle) {
        let hub = Arc::clone(self);
        let _ = std::thread::Builder::new().name("hub-monitor".into()).spawn(move || {
            hub.ensure(&app);
            loop {
                std::thread::sleep(MONITOR_EVERY);
                if hub.spawning.try_lock().is_err() {
                    continue; // a spawn/restart owns the lamp right now
                }
                let h = hello();
                hub.set(&app, hub.status_from(&h));
            }
        });
    }
}

/// Launch `uv run --directory <repo> --extra mcp --extra engine --extra
/// islands python -m wzrd_mcp` in its own process
/// group (so app quit / Ctrl-C doesn't take it down), stdin closed, output
/// appended to `.wzrd/hub.log`. A reaper thread waits on it so an exit
/// while the app runs leaves no zombie; after the app quits, launchd
/// inherits it.
fn spawn_detached() -> Result<(), String> {
    let repo = repo_dir();
    let uv = uv_path();
    let dir = wzrd_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let log_path = dir.join("hub.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("opening {}: {e}", log_path.display()))?;
    let log2 = log.try_clone().map_err(|e| format!("{e}"))?;
    let mut cmd = std::process::Command::new(&uv);
    // The extras: a plain `uv run` on a fresh venv lacks fastmcp.
    cmd.args(["run", "--directory"])
        .arg(&repo)
        .args(["--extra", "mcp", "--extra", "engine", "--extra", "islands"])
        .args(["python", "-m", "wzrd_mcp"])
        .current_dir(&repo)
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log2);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawning hub ({} in {}): {e}", uv.display(), repo.display()))?;
    log::info!(
        "hub: spawned pid {} ({} run --directory {} --extra … python -m wzrd_mcp; log {})",
        child.id(),
        uv.display(),
        repo.display(),
        log_path.display()
    );
    let _ = std::thread::Builder::new()
        .name("hub-reaper".into())
        .spawn(move || {
            if let Ok(s) = child.wait() {
                log::info!("hub process exited ({s})");
            }
        });
    Ok(())
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn hub_status(hub: tauri::State<'_, Arc<Hub>>) -> HubStatus {
    hub.status()
}

/// Attach or spawn now (the lamp's click when down). Async: off the main
/// thread, it can take seconds.
#[tauri::command]
pub async fn hub_ensure(app: AppHandle, hub: tauri::State<'_, Arc<Hub>>) -> Result<HubStatus, String> {
    let hub = Arc::clone(&hub);
    tauri::async_runtime::spawn_blocking(move || {
        hub.ensure(&app);
        hub.status()
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn hub_restart(app: AppHandle, hub: tauri::State<'_, Arc<Hub>>) -> Result<HubStatus, String> {
    let hub = Arc::clone(&hub);
    tauri::async_runtime::spawn_blocking(move || {
        hub.restart(&app)?;
        Ok(hub.status())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The thin `/api` client: one JSON call to the hub. `path` must start with
/// `/api/`. Non-2xx → `Err(<the hub's {error}> or "HTTP n")`.
#[tauri::command]
pub async fn hub_call(method: String, path: String, body: Option<Value>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || hub_call_in(&method, &path, body.as_ref()))
        .await
        .map_err(|e| e.to_string())?
}

/// `hub_call` body — also the devport's entry.
pub fn hub_call_in(method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
    if !path.starts_with("/api/") {
        return Err(format!("hub_call: {path:?} is not an /api/ path"));
    }
    let method = method.to_ascii_uppercase();
    if !matches!(method.as_str(), "GET" | "POST" | "PUT" | "DELETE") {
        return Err(format!("hub_call: method {method:?}"));
    }
    // Long-poll job feeds may hold the request open; give them room.
    let r = request(&method, path, body, Duration::from_secs(60))?;
    if (200..300).contains(&r.status) {
        Ok(r.body)
    } else {
        Err(r
            .body
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("hub: HTTP {} on {method} {path}", r.status)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_plain_and_chunked_replies() {
        let r = parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"a\": true}").unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body, json!({"a": true}));
        let r = parse_response(
            b"HTTP/1.1 404 Not Found\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"err\r\n9\r\nor\": \"x\"}\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(r.status, 404);
        assert_eq!(r.body, json!({"error": "x"}));
    }
}
