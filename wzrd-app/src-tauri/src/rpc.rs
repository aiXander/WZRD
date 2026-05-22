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
    // mask_path is relative to the pack directory.
    let pack_dir = match state.engine.request("pack.info", json!({}), DEFAULT_TIMEOUT) {
        Ok(v) => v
            .get("pack_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| "pack.info missing pack_dir".to_string())?,
        Err(e) => return Err(format!("{e:#}")),
    };
    let full = pack_dir.join(&mask_path);
    let bytes = std::fs::read(&full).map_err(|e| format!("reading {}: {e}", full.display()))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}
