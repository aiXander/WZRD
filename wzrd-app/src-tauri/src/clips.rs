//! Clip grid file commands (gig plan A2, D12). Reads the gig contracts:
//! C3 clip files (`generated/<stem>.mov` HAP + `<stem>.thumb.jpg`) and the
//! C2 provenance sidecar (`<stem>.mov.meta.json`, whose `region` says which
//! row a clip starts in). Paths go over the wire relative to the scene dir —
//! the same form a C5 binding's `image.clip` takes.
//!
//! The operator's re-filing (dragging a clip onto another row) is app
//! state, not provenance: it lives in `<scene_dir>/clipgrid.json`
//! (`{"schema":1, "rows": {"generated/x.mov": "<layer id>" | null}}`) and
//! never touches the sidecar, which keeps saying what the clip was made for.

use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};
use tauri::State;

use crate::AppState;

const GRID_FILE: &str = "clipgrid.json";

/// `rel` resolved inside the scene dir; rejects absolute paths and `..`.
fn inside(scene_dir: &Path, rel: &str) -> Result<PathBuf, String> {
    let p = Path::new(rel);
    if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("clip path {rel:?} must be relative to the project"));
    }
    Ok(scene_dir.join(p))
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

fn read_grid(scene_dir: &Path) -> Value {
    read_json(&scene_dir.join(GRID_FILE))
        .filter(|v| v.get("rows").map_or(false, Value::is_object))
        .unwrap_or_else(|| json!({ "schema": 1, "rows": {} }))
}

/// Every HAP clip in the project's `generated/`, newest first, with its
/// sidecar (verbatim, or null), its source mp4's sidecar (`source`, from the
/// HAP sidecar's `parent`), thumbnail path (or null) and the operator's
/// row filing (absent = follow the sidecar's `region`). Cheap enough to
/// poll: one directory scan + small JSON reads.
#[tauri::command]
pub fn clips_list(state: State<AppState>) -> Result<Value, String> {
    let scene_dir = state.session().scene_dir();
    let gen = scene_dir.join("generated");
    let mut clips: Vec<(u64, Value)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&gen) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("mov") {
                continue;
            }
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
            let stem = name.trim_end_matches(".mov");
            let mtime = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let thumb = format!("{stem}.thumb.jpg");
            let sidecar = read_json(&gen.join(format!("{name}.meta.json")));
            // The HAP sidecar's `parent` is the source mp4, whose own sidecar
            // carries the prompt / tool / model that made the clip.
            let source = sidecar
                .as_ref()
                .and_then(|s| s.get("parent")?.as_str().map(PathBuf::from))
                .map(|p| if p.is_absolute() { p } else { scene_dir.join(p) })
                .and_then(|p| read_json(&PathBuf::from(format!("{}.meta.json", p.display()))));
            clips.push((
                mtime,
                json!({
                    "path": format!("generated/{name}"),
                    "name": stem,
                    "mtime_ms": mtime,
                    "sidecar": sidecar,
                    "source": source,
                    "thumb": gen.join(&thumb).is_file().then(|| format!("generated/{thumb}")),
                }),
            ));
        }
    }
    clips.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(json!({
        "clips": clips.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        "rows": read_grid(&scene_dir)["rows"],
    }))
}

/// A thumbnail as base64 JPEG (the webview has no asset protocol).
#[tauri::command]
pub fn clip_thumb(state: State<AppState>, path: String) -> Result<String, String> {
    use base64::Engine;
    let full = inside(&state.session().scene_dir(), &path)?;
    let bytes = std::fs::read(&full).map_err(|e| format!("reading {}: {e}", full.display()))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// File a clip onto a row: a layer id, or `null` for "unassigned". Stored
/// in `clipgrid.json` (atomic write); the sidecar is never touched.
#[tauri::command]
pub fn clip_file(state: State<AppState>, path: String, region: Option<String>) -> Result<(), String> {
    let scene_dir = state.session().scene_dir();
    inside(&scene_dir, &path)?;
    let mut grid = read_grid(&scene_dir);
    grid["schema"] = json!(1);
    grid["rows"][&path] = region.map_or(Value::Null, Value::String);
    let target = scene_dir.join(GRID_FILE);
    let tmp = target.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(&grid).map_err(|e| format!("{e}"))?;
    std::fs::write(&tmp, text).map_err(|e| format!("{e}"))?;
    std::fs::rename(&tmp, &target).map_err(|e| format!("{e}"))
}
