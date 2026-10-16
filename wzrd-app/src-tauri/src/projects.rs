//! Project switcher + "restart engine" (gig plan A1, D13).
//!
//! The engine's scene dir (watcher, `effects/`, pack, `session.json`,
//! `alignment.json`) is fixed at boot and `scene.load` only takes JSON, so
//! switching projects = tear the in-process engine down and boot a new one
//! on the other scene path. A restart is the same move on the current path
//! (the one-click recovery after an engine crash).
//!
//! A project is a folder under the projects root (`WZRD_PROJECTS_ROOT`,
//! else settings.json `projects_root`, else `<repo>/projects`) that holds a
//! `scene.json`. Recently opened scenes — which may live anywhere, e.g. the
//! render-core examples — are kept in a per-machine `settings.json` in the
//! Tauri app config dir (`~/Library/Application Support/build.gigabrain.wzrd/`
//! on macOS). Other keys in that file are preserved (MIDI's machine default
//! lands there later).

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::engine::EngineHandle;
use crate::{AppState, Session};

const RECENT_MAX: usize = 12;

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("settings.json"))
}

/// The settings document; `{}` when missing or unreadable.
pub fn read_settings(app: &AppHandle) -> Value {
    settings_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}))
}

/// Atomic write (tmp + rename) so a crash mid-write never corrupts it.
pub(crate) fn write_settings(app: &AppHandle, doc: &Value) -> Result<(), String> {
    let path = settings_path(app).ok_or("no app config dir")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{e}"))?;
    }
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string_pretty(doc).map_err(|e| format!("{e}"))?;
    std::fs::write(&tmp, text).map_err(|e| format!("{e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{e}"))
}

pub fn projects_root(app: &AppHandle) -> PathBuf {
    if let Ok(p) = std::env::var("WZRD_PROJECTS_ROOT") {
        return PathBuf::from(p);
    }
    if let Some(p) = read_settings(app).get("projects_root").and_then(Value::as_str) {
        return PathBuf::from(p);
    }
    // <repo>/projects — the MCP's default root (wzrd_mcp/project.py).
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = repo.join("projects");
    root.canonicalize().unwrap_or(root)
}

fn recent(app: &AppHandle) -> Vec<(PathBuf, u64)> {
    read_settings(app)
        .get("recent")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|e| {
                    Some((
                        PathBuf::from(e.get("scene")?.as_str()?),
                        e.get("opened_ms").and_then(Value::as_u64).unwrap_or(0),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The most recently opened scene that still exists — the boot default
/// when neither `WZRD_SCENE` nor `--scene` is given.
pub fn last_opened(app: &AppHandle) -> Option<PathBuf> {
    recent(app).into_iter().map(|(p, _)| p).find(|p| p.is_file())
}

fn record_recent(app: &AppHandle, scene: &Path) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut list: Vec<Value> = vec![json!({ "scene": scene.to_string_lossy(), "opened_ms": now })];
    for (p, t) in recent(app) {
        if p != scene && list.len() < RECENT_MAX {
            list.push(json!({ "scene": p.to_string_lossy(), "opened_ms": t }));
        }
    }
    let mut doc = read_settings(app);
    doc["schema"] = json!(1);
    doc["recent"] = Value::Array(list);
    if let Err(e) = write_settings(app, &doc) {
        log::warn!("could not record recent project: {e}");
    }
}

/// Folder name for a project's `scene.json`; `<stem> · <folder>` for any
/// other scene file (e.g. `phase3_smoke · examples`).
fn display_name(scene: &Path) -> String {
    let dir = scene
        .parent()
        .and_then(Path::file_name)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let file = scene
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if file == "scene.json" {
        dir
    } else {
        let stem = file.trim_end_matches(".json").trim_end_matches(".scene");
        format!("{stem} · {dir}")
    }
}

/// A folder (→ its `scene.json`) or a scene file, canonicalised.
fn resolve_scene(raw: &str) -> Result<PathBuf, String> {
    let p = PathBuf::from(raw);
    let p = if p.is_dir() { p.join("scene.json") } else { p };
    if !p.is_file() {
        return Err(format!("no scene at {}", p.display()));
    }
    Ok(p.canonicalize().unwrap_or(p))
}

fn session_info(s: &Session) -> Value {
    json!({
        "name": display_name(&s.scene_path),
        "scene_path": s.scene_path.to_string_lossy(),
        "dir": s.scene_dir().to_string_lossy(),
        "running": s.engine.as_ref().map_or(false, |e| e.status().running),
        "error": s.boot_error.clone()
            .or_else(|| s.engine.as_ref().and_then(|e| e.status().last_error)),
    })
}

fn boot(app: &AppHandle, state: &AppState, scene: &Path) -> Result<EngineHandle, String> {
    EngineHandle::start_in_process(app.clone(), &state.windows, scene.to_path_buf(), |bus| {
        crate::set_log_bus(bus)
    })
    .map_err(|e| format!("{e:#}"))
}

/// Tear the current engine down and boot one on `scene`. If that fails the
/// previous project is booted again so the wall isn't left dark; the error
/// still comes back. Main thread only (surface attach).
pub fn open(app: &AppHandle, state: &AppState, scene: PathBuf) -> Result<Value, String> {
    let _guard = state.switching.lock().map_err(|_| "switch lock poisoned")?;
    let t0 = Instant::now();
    let prev = state.session();
    if let Some(engine) = prev.engine.as_ref() {
        // Same teardown as app exit: session + alignment saved, GPU context
        // dropped, WS/OSC ports released before this returns.
        engine.shutdown();
    }
    let result = match boot(app, state, &scene) {
        Ok(engine) => {
            state.set_session(Session::new(scene.clone(), Ok(engine)));
            record_recent(app, &scene);
            Ok(())
        }
        Err(e) => {
            log::error!("could not open {}: {e}", scene.display());
            let had_engine = prev.engine.is_some();
            let back = (had_engine && prev.scene_path != scene)
                .then(|| boot(app, state, &prev.scene_path));
            match back {
                Some(Ok(engine)) => {
                    state.set_session(Session::new(prev.scene_path.clone(), Ok(engine)));
                    Err(format!(
                        "could not open {}: {e} — still on {}",
                        display_name(&scene),
                        display_name(&prev.scene_path)
                    ))
                }
                _ => {
                    state.set_session(Session::new(scene.clone(), Err(e.clone())));
                    Err(e)
                }
            }
        }
    };
    let s = state.session();
    let mut info = session_info(&s);
    info["elapsed_ms"] = json!(t0.elapsed().as_millis() as u64);
    log::info!(
        "project {} {} in {} ms",
        info["name"].as_str().unwrap_or("?"),
        if result.is_ok() { "opened" } else { "kept" },
        t0.elapsed().as_millis()
    );
    let _ = app.emit("engine:project", &info);
    result.map(|()| info)
}

#[tauri::command]
pub fn projects_list(app: AppHandle, state: State<AppState>) -> Value {
    let root = projects_root(&app);
    let current = state.session().scene_path.clone();
    let recent = recent(&app);
    let mut scenes: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            let scene = e.path().join("scene.json");
            if scene.is_file() {
                scenes.push(scene.canonicalize().unwrap_or(scene));
            }
        }
    }
    for (p, _) in &recent {
        if p.is_file() && !scenes.contains(p) {
            scenes.push(p.clone());
        }
    }
    if current.is_file() && !scenes.contains(&current) {
        scenes.push(current.clone());
    }
    let mut out: Vec<Value> = scenes
        .iter()
        .map(|p| {
            let opened = recent.iter().find(|(r, _)| r == p).map(|(_, t)| *t);
            json!({
                "name": display_name(p),
                "scene_path": p.to_string_lossy(),
                "dir": p.parent().map(|d| d.to_string_lossy().into_owned()),
                "opened_ms": opened,
                "current": *p == current,
                "in_root": p.starts_with(&root),
            })
        })
        .collect();
    // Recent first (newest on top), then the rest by name.
    out.sort_by(|a, b| {
        let ta = a["opened_ms"].as_u64();
        let tb = b["opened_ms"].as_u64();
        tb.cmp(&ta).then_with(|| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .cmp(&b["name"].as_str().unwrap_or("").to_lowercase())
        })
    });
    json!({ "root": root.to_string_lossy(), "projects": out })
}

#[tauri::command]
pub fn project_current(state: State<AppState>) -> Value {
    session_info(&state.session())
}

#[tauri::command]
pub fn project_open(
    app: AppHandle,
    state: State<AppState>,
    scene_path: String,
) -> Result<Value, String> {
    let scene = resolve_scene(&scene_path)?;
    open(&app, &state, scene)
}

/// Re-boot the engine on the current scene — the crash recovery button.
#[tauri::command]
pub fn engine_restart(app: AppHandle, state: State<AppState>) -> Result<Value, String> {
    let scene = state.session().scene_path.clone();
    if !scene.is_file() {
        return Err("no project open — pick one from the project menu".into());
    }
    open(&app, &state, scene)
}
