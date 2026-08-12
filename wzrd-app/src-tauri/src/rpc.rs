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
    serde_json::to_value(state.engine.status()).unwrap_or(Value::Null)
}

#[tauri::command]
pub fn pack_info(state: State<AppState>) -> Result<Value, String> {
    state
        .engine
        .request("pack.info", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn scene_get_state(state: State<AppState>) -> Result<Value, String> {
    state
        .engine
        .request("scene.getState", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn scene_load(state: State<AppState>, json_text: String) -> Result<Value, String> {
    state
        .engine
        .request("scene.load", json!({ "json": json_text }), LONG_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn scene_reload(state: State<AppState>) -> Result<Value, String> {
    state
        .engine
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
        .engine
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
        .engine
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
        .engine
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
        .engine
        .request("effect.describe", json!({ "name": name }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.3 — explicit session sidecar save (masters + knobs + calibration).
#[tauri::command]
pub fn session_save(state: State<AppState>) -> Result<Value, String> {
    state
        .engine
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
        .engine
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
        .engine
        .request("pull", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — LIVE ⇄ DESIGN toggle: which composite the native preview samples.
#[tauri::command]
pub fn preview_set_source(state: State<AppState>, source: String) -> Result<Value, String> {
    state
        .engine
        .request("preview.setSource", json!({ "source": source }), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §5.6 — pre-flight probe thresholds A < B (ms of predicted full-res p95).
#[tauri::command]
pub fn probe_get_thresholds(state: State<AppState>) -> Result<Value, String> {
    state
        .engine
        .request("probe.getThresholds", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn probe_set_thresholds(state: State<AppState>, a_ms: f64, b_ms: f64) -> Result<Value, String> {
    state
        .engine
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
        .engine
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
) -> Result<Value, String> {
    state
        .engine
        .request(
            "alignment.set",
            json!({
                "enabled": enabled,
                "background": background,
                "corners": corners,
                "points": points,
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
        .engine
        .request("alignment.reset", json!({}), DEFAULT_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

/// §3.6 — "none" | "grid" | "border" | "corners". Generated in source space,
/// so it warps with the content; runtime-only, never persisted.
#[tauri::command]
pub fn alignment_set_test_pattern(state: State<AppState>, pattern: String) -> Result<Value, String> {
    state
        .engine
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
        .engine
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
        .engine
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
        .engine
        .request("effect.remove", json!({ "name": name }), LONG_TIMEOUT)
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub fn last_payload(state: State<AppState>, channel: String) -> Option<Value> {
    state.engine.last_payload(&channel)
}

/// Collapse Step 3 — position the native preview window over the React
/// layout's preview slot (CSS px, viewport-relative), or hide it.
#[tauri::command]
pub fn preview_set_bounds(
    app: tauri::AppHandle,
    state: State<AppState>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    visible: bool,
) -> Result<(), String> {
    state
        .engine
        .set_preview_bounds(&app, x, y, width, height, visible)
        .map_err(|e| format!("{e:#}"))
}

/// Convenience for the front-end: read the project-local effects directory
/// into a list of names so the Monaco tab list doesn't need its own watcher.
#[tauri::command]
pub fn list_effects(state: State<AppState>) -> Result<Vec<String>, String> {
    let dir = state.effects_dir.clone();
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
        .effects_dir
        .as_ref()
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
    std::fs::read_to_string(&state.scene_path).map_err(|e| format!("{e}"))
}

#[tauri::command]
pub fn write_scene_file(state: State<AppState>, contents: String) -> Result<(), String> {
    std::fs::write(&state.scene_path, contents).map_err(|e| format!("{e}"))
}

#[tauri::command]
pub fn read_mask_png(state: State<AppState>, mask_path: String) -> Result<String, String> {
    use std::path::PathBuf;
    // mask_path is relative to the pack directory. The pack dir is static
    // for the engine's lifetime, so resolve it once and cache it instead of
    // paying one RPC round trip per mask load.
    let pack_dir = state
        .pack_dir
        .get_or_init(|| {
            state
                .engine
                .request("pack.info", json!({}), DEFAULT_TIMEOUT)
                .ok()
                .and_then(|v| {
                    v.get("pack_dir")
                        .and_then(Value::as_str)
                        .map(PathBuf::from)
                })
                .unwrap_or_default()
        })
        .clone();
    if pack_dir.as_os_str().is_empty() {
        return Err("pack.info unavailable — engine not ready".to_string());
    }
    let full = pack_dir.join(&mask_path);
    let bytes = std::fs::read(&full).map_err(|e| format!("reading {}: {e}", full.display()))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}
