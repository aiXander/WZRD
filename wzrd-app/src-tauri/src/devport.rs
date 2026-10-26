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
//! Studio (A1): `hub_*` / `library_*` twins run on the connection thread,
//! not the main thread (a hub call may block for seconds), and `GET
//! /file?path=<abs>` serves library media (thumbnails, mp4s — with `Range`,
//! so `<video>` can seek) from the same scope the asset protocol allows —
//! the browser seat's stand-in for `convertFileSrc`.
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

use crate::{hub, layout, library, output, projects, rpc, AppState};

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
    let mut range: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if k.eq_ignore_ascii_case("range") {
                range = Some(v.trim().to_string());
            }
        }
    }
    if let Some(query) = path.strip_prefix("/file?") {
        return serve_file(app, &mut stream, query, range.as_deref());
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    let args: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));

    let cmd = path.trim_start_matches("/cmd/").to_string();
    // Hub + library calls block on I/O (seconds, for a hub spawn): run them
    // here, never on the main thread.
    if let Some(r) = off_main(app, &cmd, &args) {
        return reply_json(&mut stream, r);
    }
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
    reply_json(&mut stream, result)
}

fn reply_json(stream: &mut TcpStream, result: Result<Value, String>) -> std::io::Result<()> {
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

fn off_main(app: &AppHandle, cmd: &str, a: &Value) -> Option<Result<Value, String>> {
    let s = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
    let hub_state = || app.state::<std::sync::Arc<hub::Hub>>().inner().clone();
    Some(match cmd {
        "hub_status" => ok(Ok(hub_state().status())),
        // Async (builds webview windows) — block this connection thread on it.
        "output_identify" => tauri::async_runtime::block_on(output::identify(app.clone())).map(|()| Value::Null),
        "relink_missing" => crate::relink::run(
            &app.state::<AppState>().session().scene_dir(),
            &crate::relink::search_roots(app),
            a.get("scene").and_then(Value::as_str),
        ),
        "hub_ensure" => {
            let h = hub_state();
            h.ensure(app);
            ok(Ok(h.status()))
        }
        "hub_restart" => {
            let h = hub_state();
            h.restart(app).and_then(|()| ok(Ok(h.status())))
        }
        "hub_call" => match (s("method"), s("path")) {
            (Some(m), Some(p)) => hub::hub_call_in(&m, &p, a.get("body").filter(|b| !b.is_null())),
            _ => Err("hub_call: needs method + path".into()),
        },
        _ => {
            if let Some(r) = crate::decks::devport_dispatch(cmd, a, &app.state::<AppState>()) {
                return Some(r);
            }
            return library::devport_dispatch(cmd, a, app);
        }
    })
}

/// `GET /file?path=<percent-encoded abs path>` — library media for the
/// browser seat. Same scope as the asset protocol; supports one `Range`.
fn serve_file(
    app: &AppHandle,
    stream: &mut TcpStream,
    query: &str,
    range: Option<&str>,
) -> std::io::Result<()> {
    let raw = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("path="))
        .map(percent_decode)
        .unwrap_or_default();
    let path = std::path::PathBuf::from(&raw);
    let deny = |stream: &mut TcpStream, code: &str| {
        write!(stream, "HTTP/1.1 {code}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    };
    if !library::file_allowed(app, &path) {
        return deny(stream, "403 Forbidden");
    }
    let Ok(mut file) = std::fs::File::open(&path) else {
        return deny(stream, "404 Not Found");
    };
    let len = file.metadata()?.len();
    let ctype = library::content_type(&path);
    let (start, end, status) = match range.and_then(|r| parse_range(r, len)) {
        Some((a, b)) => (a, b, "206 Partial Content"),
        None => (0, len.saturating_sub(1), "200 OK"),
    };
    let count = if len == 0 { 0 } else { end + 1 - start };
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {count}\r\n\
         Accept-Ranges: bytes\r\nCache-Control: max-age=60\r\nConnection: close\r\n"
    );
    if status.starts_with("206") {
        head.push_str(&format!("Content-Range: bytes {start}-{end}/{len}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(start))?;
    std::io::copy(&mut file.take(count), stream)?;
    stream.flush()
}

/// `bytes=a-b` / `bytes=a-` / `bytes=-n` → inclusive `(start, end)`.
fn parse_range(r: &str, len: u64) -> Option<(u64, u64)> {
    let spec = r.strip_prefix("bytes=")?.split(',').next()?.trim();
    let (a, b) = spec.split_once('-')?;
    if len == 0 {
        return None;
    }
    let (start, end) = match (a.parse::<u64>().ok(), b.parse::<u64>().ok()) {
        (Some(s), Some(e)) => (s, e.min(len - 1)),
        (Some(s), None) => (s, len - 1),
        (None, Some(n)) => (len.saturating_sub(n), len - 1),
        (None, None) => return None,
    };
    (start <= end && start < len).then_some((start, end))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = (b[i] == b'%')
            .then(|| b.get(i + 1..i + 3))
            .flatten()
            .and_then(|h| u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok());
        match (hex, b[i]) {
            (Some(v), _) => {
                out.push(v);
                i += 3;
                continue;
            }
            (None, b'+') => out.push(b' '),
            (None, c) => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
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
        "effect_thumbs" => rpc::effect_thumbs(st()),
        "effect_library_read" => rpc::effect_library_read(st()),
        "effect_library_write" => ok(rpc::effect_library_write(
            st(),
            s("scenePath")?,
            a.get("doc").cloned().unwrap_or(Value::Null),
        )),
        "read_mask_png" => ok(rpc::read_mask_png_in(&st(), s("maskPath")?)),
        "looks_list" => ok(rpc::looks_list_in(&st())),
        "look_read" => ok(rpc::look_read(st(), s("name")?)),
        "look_write" => ok(rpc::look_write(st(), s("name")?, s("scene")?, opt("thumbB64"))),
        "look_delete" => ok(rpc::look_delete(st(), s("name")?)),
        "projects_list" => Ok(projects::projects_list(app.clone(), st())),
        "project_current" => Ok(projects::project_current(st())),
        "project_open" => projects::project_open(app.clone(), st(), s("scenePath")?),
        "engine_restart" => projects::engine_restart(app.clone(), st()),
        "project_save_as" => crate::fork::project_save_as(app.clone(), st(), s("name")?),
        "output_displays" => ok(Ok(output::displays(app))),
        "output_state" => ok(Ok(app.state::<output::Output>().snapshot())),
        "output_set" => ok(output::set_target(app, &s("target")?, opt("display").as_deref())),
        "output_options" => ok(Ok(output::set_options(
            app,
            a.get("hideCursor").and_then(Value::as_bool),
            a.get("keepAwake").and_then(Value::as_bool),
        ))),
        "ui_layout_read" => layout::ui_layout_read(s("scenePath")?),
        "ui_layout_write" => ok(layout::ui_layout_write(
            s("scenePath")?,
            a.get("layout").cloned().unwrap_or(Value::Null),
        )),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_paths_and_ranges() {
        assert_eq!(percent_decode("%2FUsers%2Fx%20y%2Fa.mp4"), "/Users/x y/a.mp4");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(parse_range("bytes=0-99", 1000), Some((0, 99)));
        assert_eq!(parse_range("bytes=900-", 1000), Some((900, 999)));
        assert_eq!(parse_range("bytes=-100", 1000), Some((900, 999)));
        assert_eq!(parse_range("bytes=2000-", 1000), None);
    }
}
