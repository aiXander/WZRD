//! Render-core sidecar — Tauri spawns `render-core --ws-addr 127.0.0.1:PORT`
//! at startup and talks JSON-RPC over the resulting WebSocket. Same RPC
//! surface Phase 7 will expose to MCP (D13/§3.11), just routed locally
//! through Tauri commands here.
//!
//! Single I/O thread owns the socket; Tauri commands push outbound frames
//! through a channel and block on a reply oneshot. The I/O thread also fans
//! `telemetry.event` notifications onto the `engine:telemetry` Tauri event
//! channel so the webview gets them with no further plumbing.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use tungstenite::{Message, WebSocket};

/// Default WS port the engine binds to. Fixed for v1.
pub const DEFAULT_WS_PORT: u16 = 9123;

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    pub running: bool,
    pub ws_addr: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TelemetryFrameOut {
    pub channel: String,
    pub payload: Value,
}

pub struct EngineHandle {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    child: Mutex<Option<Child>>,
    /// Outbound frames go here; the io thread is the sole socket writer.
    out_tx: Sender<String>,
    waiters: Mutex<HashMap<u64, Sender<Result<Value, String>>>>,
    next_id: AtomicU64,
    ws_addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    /// Snapshots of the sticky telemetry channels — let routes that mount
    /// late render their pills immediately.
    last_payloads: Mutex<HashMap<String, Value>>,
    /// Set to false when the io thread observes a closed socket.
    alive: Arc<AtomicBool>,
}

impl EngineHandle {
    pub fn spawn(app: AppHandle, scene_path: PathBuf, exe: PathBuf) -> Result<Self> {
        let ws_addr: SocketAddr = format!("127.0.0.1:{DEFAULT_WS_PORT}")
            .parse()
            .expect("valid addr");

        let mut cmd = Command::new(&exe);
        cmd.arg("--scene")
            .arg(&scene_path)
            .arg("--windowed")
            .arg("--ws-addr")
            .arg(ws_addr.to_string());
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        let child = cmd
            .spawn()
            .with_context(|| format!("spawning {}", exe.display()))?;
        log::info!("spawned render-core pid {}", child.id());

        let socket = connect_with_retry(ws_addr, Duration::from_secs(5))?;
        let (out_tx, out_rx) = unbounded::<String>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));

        let inner = Arc::new(EngineInner {
            child: Mutex::new(Some(child)),
            out_tx,
            waiters: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            ws_addr,
            shutdown: Arc::clone(&shutdown),
            last_payloads: Mutex::new(HashMap::new()),
            alive: Arc::clone(&alive),
        });

        // Single I/O thread.
        {
            let inner = Arc::clone(&inner);
            let app = app.clone();
            thread::Builder::new()
                .name("engine-io".into())
                .spawn(move || io_loop(inner, app, socket, out_rx))
                .context("spawning engine io thread")?;
        }

        let handle = Self { inner };
        // Subscribe to every telemetry channel up front.
        let _ = handle.request(
            "telemetry.subscribe",
            json!({
                "channels": [
                    "preview", "hot_reload", "audio_freshness", "fps",
                    "log", "frame_stats", "drivers", "audio", "connectivity",
                    "masters"
                ]
            }),
            Duration::from_secs(2),
        );
        Ok(handle)
    }

    /// Issue a JSON-RPC request, block until reply or timeout.
    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = bounded::<Result<Value, String>>(1);
        self.inner
            .waiters
            .lock()
            .expect("waiters lock")
            .insert(id, tx);
        let env = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.inner
            .out_tx
            .send(env.to_string())
            .map_err(|_| anyhow!("engine io thread gone"))?;
        match rx.recv_timeout(timeout) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(msg)) => Err(anyhow!(msg)),
            Err(_) => {
                self.inner.waiters.lock().expect("waiters lock").remove(&id);
                bail!("engine RPC timeout after {:?} (method={method})", timeout)
            }
        }
    }

    pub fn last_payload(&self, channel: &str) -> Option<Value> {
        self.inner
            .last_payloads
            .lock()
            .expect("last payloads lock")
            .get(channel)
            .cloned()
    }

    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            running: self.inner.alive.load(Ordering::Relaxed)
                && !self.inner.shutdown.load(Ordering::Relaxed),
            ws_addr: Some(self.inner.ws_addr.to_string()),
            last_error: None,
        }
    }

    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        if let Ok(mut child) = self.inner.child.lock() {
            if let Some(mut c) = child.take() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }
}

fn io_loop(
    inner: Arc<EngineInner>,
    app: AppHandle,
    mut socket: WebSocket<TcpStream>,
    out_rx: Receiver<String>,
) {
    let _ = socket.get_mut().set_nonblocking(true);
    let mut last_status_emit = Instant::now() - Duration::from_secs(1);

    while !inner.shutdown.load(Ordering::Relaxed) {
        // Read pending inbound frames.
        match socket.read() {
            Ok(Message::Text(t)) => handle_inbound(&inner, &app, &t),
            Ok(Message::Ping(p)) => {
                let _ = socket.send(Message::Pong(p));
            }
            Ok(Message::Pong(_)) | Ok(Message::Binary(_)) | Ok(Message::Frame(_)) => {}
            Ok(Message::Close(_)) => {
                let _ = socket.close(None);
                break;
            }
            Err(tungstenite::Error::Io(e)) if e.kind() == ErrorKind::WouldBlock => {}
            Err(tungstenite::Error::ConnectionClosed)
            | Err(tungstenite::Error::AlreadyClosed) => {
                log::warn!("engine WS closed");
                break;
            }
            Err(e) => {
                log::trace!("engine WS read: {e}");
            }
        }

        // Send all pending outbound frames.
        while let Ok(payload) = out_rx.try_recv() {
            if let Err(e) = socket.send(Message::Text(payload)) {
                log::warn!("engine WS send: {e}");
                break;
            }
        }
        let _ = socket.flush();

        if last_status_emit.elapsed() >= Duration::from_secs(1) {
            let _ = app.emit(
                "engine:status",
                json!({ "running": true, "ws_addr": inner.ws_addr.to_string() }),
            );
            last_status_emit = Instant::now();
        }

        thread::sleep(Duration::from_millis(8));
    }
    inner.alive.store(false, Ordering::Relaxed);
    let _ = app.emit("engine:status", json!({ "running": false }));
    log::info!("engine io thread exiting");
}

fn handle_inbound(inner: &EngineInner, app: &AppHandle, text: &str) {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return,
    };
    if let Some(id) = v.get("id").and_then(Value::as_u64) {
        let mut waiters = inner.waiters.lock().expect("waiters lock");
        if let Some(tx) = waiters.remove(&id) {
            if let Some(err) = v.get("error") {
                let msg = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("rpc error")
                    .to_string();
                let _ = tx.send(Err(msg));
            } else if let Some(result) = v.get("result") {
                let _ = tx.send(Ok(result.clone()));
            } else {
                let _ = tx.send(Ok(Value::Null));
            }
        }
        return;
    }
    if v.get("method").and_then(Value::as_str) == Some("telemetry.event") {
        if let Some(params) = v.get("params") {
            let channel = params
                .get("channel")
                .and_then(Value::as_str)
                .unwrap_or("");
            let payload = params.get("payload").cloned().unwrap_or(Value::Null);
            if matches!(
                channel,
                "hot_reload" | "audio_freshness" | "connectivity" | "fps" | "masters"
            ) {
                let mut last = inner.last_payloads.lock().expect("last payloads lock");
                last.insert(channel.to_string(), payload.clone());
            }
            let _ = app.emit(
                "engine:telemetry",
                TelemetryFrameOut {
                    channel: channel.to_string(),
                    payload,
                },
            );
        }
    }
}

fn connect_with_retry(addr: SocketAddr, total: Duration) -> Result<WebSocket<TcpStream>> {
    let started = Instant::now();
    let url = format!("ws://{addr}/");
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => match tungstenite::client(&url, stream) {
                Ok((ws, _)) => return Ok(ws),
                Err(e) => log::trace!("WS handshake: {e}"),
            },
            Err(e) => log::trace!("WS connect: {e}"),
        }
        if started.elapsed() > total {
            bail!("engine WS at {addr} never came up within {total:?}");
        }
        thread::sleep(Duration::from_millis(150));
    }
}
