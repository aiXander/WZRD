//! Debug-build-only HTTP twin of the shell's non-engine commands, for the
//! browser bridge (`src/api/wsBridge.ts`). Engine methods go to the
//! engine's own WS; everything the shell does itself (scene/effect/look
//! files, projects, clips, engine restart) comes here through Vite's
//! `/__shell` proxy, so a browser seat follows the shell's *current*
//! project instead of a path pinned at `pnpm tauri dev` time.
//!
//! `POST /cmd/<tauri command>` with the same camelCase JSON args the
//! webview passes to `invoke`; replies `{result}` or HTTP 500 `{error}`.
//! Every command runs on the main thread, exactly like a sync Tauri
//! command (a project switch attaches GPU surfaces). Bound to loopback on
//! `WZRD_DEVPORT` (default 9124); never compiled into a release build.
//!
//! Plus one agent hook with no Tauri twin: `POST /cmd/webview_eval {js}`
//! runs a script in the app's own webview (switch rooms, click a toggle)
//! so an agent can drive the native-preview rooms without taking the
//! operator's keyboard focus. Fire-and-forget: no return value — to read
//! something back, have the script POST it to `/__shell/cmd/debug_log
//! {msg}`, which lands in the app log (`[webview] …`). Pair it with the
//! engine's `preview.snapshot` to see the result.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Manager};

use crate::{clips, projects, rpc, AppState};

pub fn spawn(app: AppHandle) {
    let port: u16 = std::env::var("WZRD_DEVPORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9124);
    let _ = std::thread::Builder::new()
        .name("devport".into())
        .spawn(move || {
            // `tauri dev` restarts the app on a Rust change before the old
            // process has let go of the port: retry for a few seconds.
            let mut tries = 0;
            let listener = loop {
                match TcpListener::bind(("127.0.0.1", port)) {
                    Ok(l) => break l,
                    Err(e) if tries < 40 => {
                        tries += 1;
                        log::debug!("dev shell port {port}: {e}, retrying");
                        std::thread::sleep(Duration::from_millis(250));
                    }
                    Err(e) => {
                        log::warn!("dev shell port {port} unavailable ({e}) — browser bridge shell commands off");
                        return;
                    }
                }
            };
            log::info!("dev shell port on http://127.0.0.1:{port} (browser bridge)");
            for stream in listener.incoming().flatten() {
                let app = app.clone();
                let _ = std::thread::Builder::new()
                    .name("devport-conn".into())
                    .spawn(move || {
                        if let Err(e) = serve(&app, stream) {
                            log::debug!("devport: {e}");
                        }
                    });
            }
        });
}

fn serve(app: &AppHandle, mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let path = request_line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    let args: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));

    let cmd = path.trim_start_matches("/cmd/").to_string();
    let (tx, rx) = crossbeam_channel::bounded(1);
    let app2 = app.clone();
    let result = match app.run_on_main_thread(move || {
        let _ = tx.send(dispatch(&app2, &cmd, &args));
    }) {
        Ok(()) => rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| Err("shell command timed out".into())),
        Err(e) => Err(format!("main thread unavailable: {e}")),
    };
    let (status, payload) = match result {
        Ok(v) => ("200 OK", json!({ "result": v })),
        Err(e) => ("500 Internal Server Error", json!({ "error": e })),
    };
    let text = payload.to_string();
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )?;
    stream.flush()
}

fn ok<T: Serialize>(r: Result<T, String>) -> Result<Value, String> {
    r.map(|v| serde_json::to_value(v).unwrap_or(Value::Null))
}

fn dispatch(app: &AppHandle, cmd: &str, a: &Value) -> Result<Value, String> {
    let st = || app.state::<AppState>();
    let s = |k: &str| {
        a.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{cmd}: missing string arg {k}"))
    };
    let opt = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
    match cmd {
        "engine_status" => Ok(rpc::engine_status(st())),
        "read_scene_file" => ok(rpc::read_scene_file(st())),
        "write_scene_file" => ok(rpc::write_scene_file(st(), s("contents")?)),
        "list_effects" => ok(rpc::list_effects(st())),
        "read_effect" => ok(rpc::read_effect(st(), s("name")?)),
        "read_mask_png" => ok(rpc::read_mask_png(st(), s("maskPath")?)),
        "looks_list" => ok(rpc::looks_list(st())),
        "look_read" => ok(rpc::look_read(st(), s("name")?)),
        "look_write" => ok(rpc::look_write(st(), s("name")?, s("scene")?, opt("thumbB64"))),
        "look_delete" => ok(rpc::look_delete(st(), s("name")?)),
        "projects_list" => Ok(projects::projects_list(app.clone(), st())),
        "project_current" => Ok(projects::project_current(st())),
        "project_open" => projects::project_open(app.clone(), st(), s("scenePath")?),
        "engine_restart" => projects::engine_restart(app.clone(), st()),
        "clips_list" => clips::clips_list(st()),
        "clip_thumb" => ok(clips::clip_thumb(st(), s("path")?)),
        "clip_file" => ok(clips::clip_file(st(), s("path")?, opt("region"))),
        "debug_log" => {
            log::info!("[webview] {}", s("msg")?);
            Ok(Value::Null)
        }
        "webview_eval" => app
            .get_webview_window("main")
            .ok_or_else(|| "main window missing".to_string())?
            .eval(s("js")?)
            .map(|()| Value::Null)
            .map_err(|e| format!("{e}")),
        other => Err(format!("no dev shell handler for {other}")),
    }
}
