//! Per-project operator-UI layout (panel sizes, arrangement, lock state):
//! `<scene_dir>/ui_layout.json`, written only by the webview
//! (`src/ui/layoutStore.ts`).
//!
//! A sibling file rather than a key in `project.json`: that file is the
//! MCP's (`wzrd_mcp/project.py` — locked canvas + spend cap). Its writers
//! do preserve unknown keys, but they read-modify-write from another
//! process, and the UI writes on every drag — two writers racing on one
//! file could silently drop the canvas lock. One writer per file, like
//! `alignment.json` / `session.json` (engine) and `decks.json` (app).
//!
//! Both commands name the project by its scene path instead of using the
//! current session: a debounced write that lands just after a project
//! switch still goes to the project it was made in.

use std::path::{Path, PathBuf};

use serde_json::Value;

const LAYOUT_FILE: &str = "ui_layout.json";

/// `<dir of scene_path>/ui_layout.json`; the scene file must exist, so a
/// webview can only ever write this one file name next to a real scene.
fn layout_path(scene_path: &str) -> Result<PathBuf, String> {
    let scene = Path::new(scene_path);
    if !scene.is_file() {
        return Err(format!("no scene at {scene_path}"));
    }
    let dir = scene.parent().ok_or("scene has no parent dir")?;
    Ok(dir.join(LAYOUT_FILE))
}

/// The project's layout document, or null when it has none (or it's
/// unreadable — the webview falls back to the default layout).
#[tauri::command]
pub fn ui_layout_read(scene_path: String) -> Result<Value, String> {
    let p = layout_path(&scene_path)?;
    Ok(std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .filter(Value::is_object)
        .unwrap_or(Value::Null))
}

/// Replace the project's layout document (atomic: tmp + rename).
#[tauri::command]
pub fn ui_layout_write(scene_path: String, layout: Value) -> Result<(), String> {
    if !layout.is_object() {
        return Err("layout must be a JSON object".into());
    }
    let target = layout_path(&scene_path)?;
    let tmp = target.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(&layout).map_err(|e| format!("{e}"))?;
    std::fs::write(&tmp, text).map_err(|e| format!("{e}"))?;
    std::fs::rename(&tmp, &target).map_err(|e| format!("{e}"))
}
