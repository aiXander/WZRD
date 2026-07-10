//! WZRD Tauri shell — Phase 4 wrapper around the `render-core` engine.
//!
//! The shell spawns `render-core` as a subprocess (`--ws-addr
//! 127.0.0.1:9123`) and proxies every Tauri command through its JSON-RPC
//! WebSocket. The same RPC surface (§3.11, D13) is what Phase 7 will expose
//! to MCP — no second contract, no shim code in between. The headless agent
//! path (`render-core --scene foo.json` with no `--ws-addr`) is unchanged.

mod engine;
mod rpc;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tauri::Manager;

use crate::engine::EngineHandle;

pub struct AppState {
    pub engine: Arc<EngineHandle>,
    pub scene_path: PathBuf,
    pub effects_dir: Option<PathBuf>,
    /// Lazily-resolved pack directory (from `pack.info`), cached so mask
    /// loads don't pay an RPC round trip each.
    pub pack_dir: std::sync::OnceLock<PathBuf>,
}

/// Resolve `--scene` from process args, env var `WZRD_SCENE`, or fall back
/// to the bundled smoke example. The dev workflow expects the env var or a
/// CLI arg so we don't have to teach the front-end to bootstrap before any
/// scene is bound.
///
/// Relative paths are tried first against the current CWD (matches a normal
/// shell invocation) and then against the `wzrd-app/` directory — that lets
/// `cargo tauri dev` work with paths like `../render-core/examples/foo.json`
/// even though cargo silently switches CWD to `wzrd-app/src-tauri/` before
/// launching the binary.
fn resolve_scene_path() -> Result<PathBuf, String> {
    let raw = if let Ok(p) = std::env::var("WZRD_SCENE") {
        Some(p)
    } else {
        let mut args = std::env::args().skip(1);
        let mut found = None;
        while let Some(a) = args.next() {
            if a == "--scene" {
                if let Some(v) = args.next() {
                    found = Some(v);
                }
            }
        }
        found
    };
    let raw = raw.ok_or_else(|| {
        "no scene path bound — set WZRD_SCENE or pass --scene <path>".to_string()
    })?;
    let p = PathBuf::from(&raw);
    if p.is_absolute() {
        return Ok(p);
    }
    if p.exists() {
        return Ok(p);
    }
    // Fallback: interpret relative to the wzrd-app/ root (the parent of the
    // src-tauri manifest dir). Lets users type paths the way they would from
    // the project root regardless of cargo's CWD games.
    let manifest_parent = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf);
    if let Some(parent) = manifest_parent {
        let candidate = parent.join(&p);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "scene path {raw:?} not found (tried CWD {} and the wzrd-app/ root)",
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".into())
    ))
}

fn resolve_engine_exe() -> PathBuf {
    // `cargo tauri dev` puts both binaries in `target/debug`. In bundled
    // release the engine should sit next to wzrd-app in the same dir; we
    // search both. WZRD_ENGINE_EXE overrides for non-standard layouts.
    if let Ok(p) = std::env::var("WZRD_ENGINE_EXE") {
        return PathBuf::from(p);
    }
    let candidates = [
        // Tauri convention: target/{debug,release}/render-core
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("render-core"))),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("render-core.exe"))),
        Some(PathBuf::from("../../render-core/target/debug/render-core")),
        Some(PathBuf::from("../render-core/target/debug/render-core")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("render-core")
}

pub fn run() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let scene_path = match resolve_scene_path() {
                Ok(p) => p,
                Err(e) => {
                    log::error!("{e}");
                    return Err(e.into());
                }
            };
            let scene_path = scene_path
                .canonicalize()
                .unwrap_or_else(|_| scene_path.clone());
            let effects_dir = scene_path
                .parent()
                .map(|p| p.join("effects"))
                .filter(|p| p.exists());
            let exe = resolve_engine_exe();
            log::info!(
                "scene={} effects_dir={:?} engine_exe={}",
                scene_path.display(),
                effects_dir,
                exe.display()
            );

            let handle = match EngineHandle::spawn(app.handle().clone(), scene_path.clone(), exe) {
                Ok(h) => Arc::new(h),
                Err(e) => {
                    log::error!("could not spawn render-core: {e:#}");
                    return Err(e.to_string().into());
                }
            };

            app.manage(AppState {
                engine: handle,
                scene_path,
                effects_dir,
                pack_dir: std::sync::OnceLock::new(),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            rpc::engine_status,
            rpc::pack_info,
            rpc::scene_get_state,
            rpc::scene_load,
            rpc::scene_reload,
            rpc::param_set,
            rpc::wgsl_validate,
            rpc::effect_upsert,
            rpc::effect_remove,
            rpc::last_payload,
            rpc::list_effects,
            rpc::read_effect,
            rpc::read_scene_file,
            rpc::write_scene_file,
            rpc::read_mask_png,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                if let Some(state) = window.try_state::<AppState>() {
                    state.engine.shutdown();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running WZRD Tauri shell");

    let _ = json!(0); // ensure serde_json import not pruned in release
    let _ = Duration::from_secs(0);
}
