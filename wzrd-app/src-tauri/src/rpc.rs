//! Tauri command surface — every command is a thin wrapper around the same
//! engine RPC the WS clients call. The webview can therefore exercise the
//! exact §3.11 method set, and Phase 7's MCP wrapper later proxies the
//! identical surface remotely with no shim code in between.

use std::time::Duration;

use serde_json::{json, Value};
use tauri::State;

use crate::AppState;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
const LONG_TIMEOUT: Duration = Duration::from_secs(15);

#[tauri::command]
pub fn engine_status(state: State<AppState>) -> Value {
    match state.engine() {
        Ok(engine) => serde_json::to_value(engine.status()).unwrap_or(Value::Null),
        Err(e) => json!({ "running": false, "ws_addr": null, "last_error": e }),
    }
}

#[tauri::command]
pub fn pack_info(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("pack.info", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// `leg`: "design" (engine default) | "live" — the deck compares the two to
/// light TAKE when design holds something the crowd hasn't seen.
#[tauri::command]
pub fn scene_get_state(state: State<AppState>, leg: Option<String>) -> Result<Value, String> {
    state
        .engine()?
        .request("scene.getState", json!({ "leg": leg }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// Live tempo / downbeat (`transport.set`). `leg: None` = both legs.
#[tauri::command]
pub fn transport_set(
    state: State<AppState>,
    bpm: Option<f64>,
    resync: Option<bool>,
    sync: Option<bool>,
    leg: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "transport.set",
            json!({
                "bpm": bpm,
                "resync": resync.unwrap_or(false),
                "sync": sync.unwrap_or(false),
                "leg": leg,
            }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// `param.list` — one leg's knob values + override table (incl. the
/// reserved `@opacity`/`@mute`/`@solo` layer-mixer entries).
#[tauri::command]
pub fn param_list(state: State<AppState>, leg: Option<String>) -> Result<Value, String> {
    state
        .engine()?
        .request("param.list", json!({ "leg": leg }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn scene_load(state: State<AppState>, json_text: String) -> Result<Value, String> {
    state
        .engine()?
        .request("scene.load", json!({ "json": json_text }), LONG_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn scene_reload(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("scene.reload", json!({}), LONG_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// Live knob path — sets a named `ui.slider` value inside the engine. No
/// scene rebuild, no disk write; the bound params pick it up next frame.
/// §5.6: `leg` targets the deck-toggle leg (engine default: design).
#[tauri::command]
pub fn param_set(
    state: State<AppState>,
    name: String,
    value: f64,
    leg: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "param.set",
            json!({ "name": name, "value": value, "leg": leg }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.5 live per-binding override — `param.set { binding, param, value }`.
/// `value: None` clears the override (the param falls back to its scene
/// value / driver next frame). Zero rebuild, persisted via the session
/// sidecar. §5.6: per-leg via `leg`.
#[tauri::command]
pub fn param_override(
    state: State<AppState>,
    binding: String,
    param: String,
    value: Option<f64>,
    leg: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "param.set",
            json!({ "binding": binding, "param": param, "value": value, "leg": leg }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.4 masters — operator-owned globals (brightness / speed / saturation /
/// audioListen), per leg since §5.6 (the deck toggle picks the target).
/// Both legs' values come back on the sticky `masters` telemetry channel.
#[tauri::command]
pub fn master_set(
    state: State<AppState>,
    name: String,
    value: f64,
    leg: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "master.set",
            json!({ "name": name, "value": value, "leg": leg }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.5 — effect input descriptors (ranges/steps/widgets) so controls render
/// without guessing. `name: None` returns the whole catalog.
#[tauri::command]
pub fn effect_describe(state: State<AppState>, name: Option<String>) -> Result<Value, String> {
    state
        .engine()?
        .request("effect.describe", json!({ "name": name }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.3 — explicit session sidecar save (masters + knobs + calibration).
#[tauri::command]
pub fn session_save(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("session.save", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — crossfade the projector to the design composite, then adopt
/// design's plan into the live slot. `quantize`: "bar" (default — fade
/// starts on the next bar boundary) | "now". Long timeout: the reply may
/// wait on a bar boundary decision but not on the fade itself.
#[tauri::command]
pub fn promote(
    state: State<AppState>,
    fade_ms: Option<f64>,
    quantize: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "promote",
            json!({ "fade_ms": fade_ms, "quantize": quantize }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — hard-copy live's scene back into design (the explicit reverse).
#[tauri::command]
pub fn pull(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("pull", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — LIVE ⇄ DESIGN toggle: which composite the native preview samples.
#[tauri::command]
pub fn preview_set_source(state: State<AppState>, source: String) -> Result<Value, String> {
    state
        .engine()?
        .request("preview.setSource", json!({ "source": source }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// Studio monitor (studio plan P1): open one HAP clip / still for the
/// `"studio"` preview slot, or close it (`path: None`). Never touches a leg.
#[tauri::command]
pub fn preview_file(state: State<AppState>, path: Option<String>) -> Result<Value, String> {
    state
        .engine()?
        .request("preview.file", json!({ "path": path }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — pre-flight probe thresholds A < B (ms of predicted full-res p95).
#[tauri::command]
pub fn probe_get_thresholds(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("probe.getThresholds", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn probe_set_thresholds(state: State<AppState>, a_ms: f64, b_ms: f64) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "probe.setThresholds",
            json!({ "a_ms": a_ms, "b_ms": b_ms }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.14 — the full alignment document plus `{output, points_max}`.
#[tauri::command]
pub fn alignment_get(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("alignment.get", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.14 — partial merge. Sending `corners` alone carries the extra handles
/// with the content (the engine recomputes their dest positions); sending
/// `points` replaces the handle list outright. A point with no `anchor` is
/// anchored at the current field, making an add a no-op on the rendered
/// image. Rejections are prescriptive and the previous alignment keeps
/// rendering.
#[tauri::command]
pub fn alignment_set(
    state: State<AppState>,
    enabled: Option<bool>,
    background: Option<String>,
    corners: Option<Value>,
    points: Option<Value>,
    flip_h: Option<bool>,
    flip_v: Option<bool>,
    fit: Option<String>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "alignment.set",
            json!({
                "enabled": enabled,
                "background": background,
                "corners": corners,
                "points": points,
                "flip_h": flip_h,
                "flip_v": flip_v,
                "fit": fit,
            }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

/// §5.14 — identity corners, no handles, black background. Leaves the
/// enabled flag alone.
#[tauri::command]
pub fn alignment_reset(state: State<AppState>) -> Result<Value, String> {
    state
        .engine()?
        .request("alignment.reset", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §3.6 — "none" | "grid" | "border" | "corners". Generated in source space,
/// so it warps with the content; runtime-only, never persisted.
#[tauri::command]
pub fn alignment_set_test_pattern(state: State<AppState>, pattern: String) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "alignment.setTestPattern",
            json!({ "pattern": pattern }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn wgsl_validate(state: State<AppState>, source: String) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "wgsl.validate",
            json!({ "source": source }),
            DEFAULT_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn effect_upsert(
    state: State<AppState>,
    name: String,
    wgsl: String,
    descriptor: Option<Value>,
) -> Result<Value, String> {
    state
        .engine()?
        .request(
            "effect.upsert",
            json!({
                "name": name,
                "wgsl": wgsl,
                "descriptor": descriptor,
            }),
            LONG_TIMEOUT,
        )
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn effect_remove(state: State<AppState>, name: String) -> Result<Value, String> {
    state
        .engine()?
        .request("effect.remove", json!({ "name": name }), LONG_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn last_payload(state: State<AppState>, channel: String) -> Option<Value> {
    state.engine().ok()?.last_payload(&channel)
}

/// Place the native preview over a `<NativeSlot>` rect (CSS px,
/// viewport-relative) with what it should show, or hide it.
#[tauri::command]
pub fn preview_set_slot(
    app: tauri::AppHandle,
    state: State<AppState>,
    slot: crate::engine::PreviewSlot,
) -> Result<(), String> {
    let engine = state.engine().ok();
    state
        .windows
        .set_preview_slot(&app, engine.as_deref(), &slot)
        .map_err(|e| format!("{e:#}"))
}

/// `"underlay"` (HTML draws over the native preview; every room uses it)
/// or `"window"` (`WZRD_PREVIEW=window`: Perform only, JPEG elsewhere).
#[tauri::command]
pub fn preview_mode(state: State<AppState>) -> &'static str {
    state.windows.preview_mode()
}

/// The streaming telemetry channels the webview displays right now (the
/// full set — `preview`, `audio`, `transport`, `drivers`, `frame_stats`,
/// `perf`). The webview refcounts its consumers (`api/telemetryDemand.ts`)
/// and sends the union on every change; sticky channels + `log` always
/// flow. See `engine::ChannelDemand`.
#[tauri::command]
pub fn telemetry_demand(state: State<AppState>, channels: Vec<String>) {
    state.windows.telemetry_demand(channels);
}

/// Convenience for the front-end: read the project-local effects directory
/// into a list of names so the Monaco tab list doesn't need its own watcher.
#[tauri::command]
pub fn list_effects(state: State<AppState>) -> Result<Vec<String>, String> {
    let dir = state.session().effects_dir.clone();
    let Some(dir) = dir else { return Ok(Vec::new()) };
    let mut names = Vec::new();
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{e}"))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() && p.join("shader.wgsl").exists() {
            if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
                names.push(n.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

#[tauri::command]
pub fn read_effect(state: State<AppState>, name: String) -> Result<Value, String> {
    let dir = state
        .session()
        .effects_dir
        .clone()
        .ok_or_else(|| "no effects directory bound".to_string())?;
    let path = dir.join(&name);
    let shader = std::fs::read_to_string(path.join("shader.wgsl"))
        .map_err(|e| format!("reading shader.wgsl: {e}"))?;
    let descriptor = std::fs::read_to_string(path.join("descriptor.json")).ok();
    Ok(json!({
        "name": name,
        "wgsl": shader,
        "descriptor": descriptor,
    }))
}

#[tauri::command]
pub fn read_scene_file(state: State<AppState>) -> Result<String, String> {
    std::fs::read_to_string(&state.session().scene_path).map_err(|e| format!("{e}"))
}

#[tauri::command]
pub fn write_scene_file(state: State<AppState>, contents: String) -> Result<(), String> {
    std::fs::write(&state.session().scene_path, contents).map_err(|e| format!("{e}"))
}

/// `async`: one call per mask at hydrate — file reads (and a `pack.info`
/// round trip on a cold cache) stay off the main thread.
#[tauri::command]
pub async fn read_mask_png(state: State<'_, AppState>, mask_path: String) -> Result<String, String> {
    read_mask_png_in(&state, mask_path)
}

/// `read_mask_png` body — also the devport's entry.
pub fn read_mask_png_in(state: &AppState, mask_path: String) -> Result<String, String> {
    use std::path::PathBuf;
    // mask_path is relative to the pack directory. The pack dir is static
    // for the engine's lifetime, so resolve it once and cache it instead of
    // paying one RPC round trip per mask load.
    let session = state.session();
    let pack_dir = match session.pack_dir.get() {
        Some(p) => p.clone(),
        None => {
            // Not cached on failure, so a mask load during boot can retry.
            let p = session
                .engine
                .as_ref()
                .and_then(|e| e.request("pack.info", json!({}), DEFAULT_TIMEOUT).ok())
                .and_then(|v| v.get("pack_dir").and_then(Value::as_str).map(PathBuf::from))
                .ok_or_else(|| "pack.info unavailable — engine not ready".to_string())?;
            session.pack_dir.get_or_init(|| p).clone()
        }
    };
    let full = pack_dir.join(&mask_path);
    let bytes = std::fs::read(&full).map_err(|e| format!("reading {}: {e}", full.display()))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

// ---------- looks: saved scenes, the clip bank ----------
//
// A look is a plain scene file in `<scene_dir>/looks/<name>.json` plus an
// optional `<name>.jpg` thumbnail captured from the preview at save time.
// Plain files on purpose (design spec §12): copy, version, share.

fn looks_dir(state: &AppState) -> std::path::PathBuf {
    state.session().scene_dir().join("looks")
}

fn look_stem(name: &str) -> Result<String, String> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.'))
        && !name.starts_with('.');
    if ok {
        Ok(name.to_string())
    } else {
        Err(format!(
            "look name {name:?} must be 1-64 chars of letters, digits, space, _ - ."
        ))
    }
}

/// `async`: reads every look's thumbnail — off the main thread.
#[tauri::command]
pub async fn looks_list(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    looks_list_in(&state)
}

/// `looks_list` body — also the devport's entry.
pub fn looks_list_in(state: &AppState) -> Result<Vec<Value>, String> {
    use base64::Engine;
    let dir = looks_dir(state);
    let Ok(entries) = std::fs::read_dir(&dir) else { return Ok(Vec::new()) };
    let mut out: Vec<(std::time::SystemTime, Value)> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
        let created = entry
            .metadata()
            .and_then(|m| m.created().or_else(|_| m.modified()))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let thumb = std::fs::read(p.with_extension("jpg"))
            .ok()
            .map(|b| base64::engine::general_purpose::STANDARD.encode(b));
        out.push((created, json!({ "name": stem, "thumb_b64": thumb })));
    }
    // Bank order = creation order, so slots don't reshuffle on resave.
    out.sort_by_key(|(t, _)| *t);
    Ok(out.into_iter().map(|(_, v)| v).collect())
}

#[tauri::command]
pub fn look_read(state: State<AppState>, name: String) -> Result<String, String> {
    let stem = look_stem(&name)?;
    std::fs::read_to_string(looks_dir(&state).join(format!("{stem}.json")))
        .map_err(|e| format!("reading look {stem}: {e}"))
}

#[tauri::command]
pub fn look_write(
    state: State<AppState>,
    name: String,
    scene: String,
    thumb_b64: Option<String>,
) -> Result<(), String> {
    use base64::Engine;
    let stem = look_stem(&name)?;
    serde_json::from_str::<Value>(&scene).map_err(|e| format!("look is not JSON: {e}"))?;
    let dir = looks_dir(&state);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{e}"))?;
    std::fs::write(dir.join(format!("{stem}.json")), scene).map_err(|e| format!("{e}"))?;
    if let Some(b64) = thumb_b64 {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
            let _ = std::fs::write(dir.join(format!("{stem}.jpg")), bytes);
        }
    }
    Ok(())
}

#[tauri::command]
pub fn look_delete(state: State<AppState>, name: String) -> Result<(), String> {
    let stem = look_stem(&name)?;
    let dir = looks_dir(&state);
    std::fs::remove_file(dir.join(format!("{stem}.json"))).map_err(|e| format!("{e}"))?;
    let _ = std::fs::remove_file(dir.join(format!("{stem}.jpg")));
    Ok(())
}
