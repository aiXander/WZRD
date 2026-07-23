//! JSON-RPC over WebSocket — the IPC surface the Tauri shell talks to
//! (§3.11). Synchronous tungstenite + thread-per-connection because the
//! expected fan-out is ~1 (the Tauri sidecar process) plus occasional MCP
//! consumers in Phase 7. No async runtime required.
//!
//! Each accepted connection runs in its own thread that:
//!   - reads JSON-RPC requests off the wire,
//!   - synchronously dispatches via `rpc::dispatch` (which queues commands
//!     and blocks on a reply channel for state mutations),
//!   - and pulls telemetry frames off its bus subscription, forwarding them
//!     as JSON-RPC notifications.
//!
//! The render thread never touches a socket and never blocks on a client.

use std::collections::HashSet;
use std::io::ErrorKind;
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use serde_json::{json, Value};
use tungstenite::{accept, Message};

use crate::rpc::{self, EngineCommand, JsonRpcRequest, RpcContext, RpcError};
use crate::telemetry::TelemetryFrame;

/// Stops the listen + accept loop when dropped.
pub struct ServerHandle {
    shutdown: Arc<AtomicBool>,
    _join: thread::JoinHandle<()>,
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Bind on `addr`, spawn the accept loop on its own thread, return a handle
/// whose drop signals shutdown. Failure to bind is treated as fatal — the
/// caller can choose to fall back to headless if that's the right call.
pub fn serve(
    addr: SocketAddr,
    cmd_tx: Sender<EngineCommand>,
    ctx: RpcContext,
) -> Result<ServerHandle> {
    let listener =
        TcpListener::bind(addr).with_context(|| format!("binding WS socket on {addr}"))?;
    listener
        .set_nonblocking(true)
        .context("setting WS listener nonblocking")?;
    let bound = listener.local_addr().unwrap_or(addr);
    log::info!("WS JSON-RPC listening on ws://{bound}");

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_c = Arc::clone(&shutdown);
    let join = thread::Builder::new()
        .name("ws-accept".into())
        .spawn(move || accept_loop(listener, cmd_tx, ctx, shutdown_c))
        .context("spawning WS accept thread")?;

    Ok(ServerHandle {
        shutdown,
        _join: join,
    })
}

fn accept_loop(
    listener: TcpListener,
    cmd_tx: Sender<EngineCommand>,
    ctx: RpcContext,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, peer)) => {
                log::info!("WS connection from {peer}");
                if let Err(e) = stream.set_nonblocking(false) {
                    log::warn!("WS: could not set blocking on accepted stream: {e}");
                }
                let ws = match accept(stream) {
                    Ok(w) => w,
                    Err(e) => {
                        log::warn!("WS handshake failed: {e}");
                        continue;
                    }
                };
                let cmd_tx = cmd_tx.clone();
                let ctx = ctx.clone();
                thread::Builder::new()
                    .name(format!("ws-conn-{peer}"))
                    .spawn(move || conn_loop(ws, cmd_tx, ctx))
                    .ok();
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                log::warn!("WS accept: {e}");
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
    log::info!("WS accept loop exiting");
}

fn conn_loop(
    mut ws: tungstenite::WebSocket<std::net::TcpStream>,
    cmd_tx: Sender<EngineCommand>,
    ctx: RpcContext,
) {
    // Subscription identity for this connection — starts unsubscribed.
    let mut subscription: Option<(u64, crossbeam_channel::Receiver<TelemetryFrame>)> = None;
    // §5.10 per-connection actor identity: WS connections default to
    // `agent` (a remote operator UI re-declares via `hello {actor: "ui"}`).
    let mut actor = rpc::Actor::Agent;

    if let Err(e) = ws.get_mut().set_nonblocking(true) {
        log::warn!("WS: nonblocking toggle failed: {e}");
    }

    loop {
        // Drain inbound requests.
        match ws.read() {
            Ok(Message::Text(text)) => {
                let response = handle_text(&text, &ctx, &cmd_tx, &mut subscription, &mut actor);
                if let Some(resp) = response {
                    if let Err(e) = ws.send(Message::Text(resp)) {
                        log::trace!("WS send failed (closing): {e}");
                        break;
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                // Binary frames are not part of the protocol; drop silently.
            }
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p));
            }
            Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) | Ok(Message::Frame(_)) => {
                let _ = ws.close(None);
                break;
            }
            Err(tungstenite::Error::Io(e)) if e.kind() == ErrorKind::WouldBlock => {}
            Err(tungstenite::Error::ConnectionClosed)
            | Err(tungstenite::Error::AlreadyClosed) => {
                break;
            }
            Err(e) => {
                log::trace!("WS read error (closing): {e}");
                break;
            }
        }

        // Drain outbound telemetry.
        if let Some((_, rx)) = subscription.as_ref() {
            while let Ok(frame) = rx.try_recv() {
                let notif = json!({
                    "jsonrpc": "2.0",
                    "method": "telemetry.event",
                    "params": {
                        "channel": frame.channel,
                        "payload": frame.payload,
                    },
                });
                if let Err(e) = ws.send(Message::Text(notif.to_string())) {
                    log::trace!("WS notify send failed: {e}");
                    break;
                }
            }
        }

        // Flush + idle gently.
        let _ = ws.flush();
        thread::sleep(Duration::from_millis(8));
    }

    if let Some((id, _)) = subscription.take() {
        ctx.bus.unsubscribe(id);
    }
}

fn handle_text(
    raw: &str,
    ctx: &RpcContext,
    cmd_tx: &Sender<EngineCommand>,
    subscription: &mut Option<(u64, crossbeam_channel::Receiver<TelemetryFrame>)>,
    actor: &mut rpc::Actor,
) -> Option<String> {
    let req: JsonRpcRequest = match serde_json::from_str(raw) {
        Ok(r) => r,
        Err(e) => {
            return Some(
                json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") },
                })
                .to_string(),
            );
        }
    };
    let id = req.id.clone().unwrap_or(Value::Null);
    // Subscriptions are routed in-thread; they touch the per-connection
    // receiver, so they don't go through `rpc::dispatch`.
    if req.method == "telemetry.subscribe" {
        let channels = extract_channels(&req.params);
        if let Some((sub_id, _)) = subscription.take() {
            ctx.bus.unsubscribe(sub_id);
        }
        let (sub_id, rx) = ctx.bus.subscribe(channels.clone());
        *subscription = Some((sub_id, rx));
        return Some(
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "ok": true, "subscription_id": sub_id, "channels": channels },
            })
            .to_string(),
        );
    }
    if req.method == "telemetry.unsubscribe" {
        if let Some((sub_id, _)) = subscription.take() {
            ctx.bus.unsubscribe(sub_id);
        }
        return Some(
            json!({ "jsonrpc": "2.0", "id": id, "result": { "ok": true } }).to_string(),
        );
    }

    let result = rpc::dispatch(ctx, cmd_tx, &req, actor);
    let envelope = match result {
        Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
        Err(err) => RpcError::to_json(&err, &id),
    };
    Some(envelope.to_string())
}

fn extract_channels(params: &Value) -> HashSet<String> {
    let arr = params.get("channels").and_then(Value::as_array);
    match arr {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        None => HashSet::new(),
    }
}
