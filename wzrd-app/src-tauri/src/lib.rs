//! WZRD Tauri shell — single-process host for the `render-core` engine
//! (app-collapse Step 2).
//!
//! The shell runs the engine as a library on an in-process render thread:
//! one process, two windows (webview + engine output). Tauri commands call
//! `rpc::dispatch` directly — the same §3.11 surface the engine's WS server
//! (still alive on 127.0.0.1:9123) serves to external MCP / remote clients.
//! The headless agent path (`render-core --scene foo.json`, winit-hosted)
//! is unchanged and lives in the separate `render-core` binary.

mod engine;
mod rpc;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tauri::Manager;

use crate::engine::EngineHandle;

/// Tee logger: stderr via env_logger + the engine's `log` telemetry channel
/// once the bus exists (Core::new sets the global). Same shape as the
/// standalone binary's logger — the webview's Debug log stream depends on
/// it. Only Info and louder are forwarded; the bus internals log at trace,
/// which also breaks any recursion.
struct TeeLogger {
    inner: env_logger::Logger,
}

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        self.inner.log(record);
        if record.level() <= log::Level::Info && self.inner.matches(record) {
            if let Some(bus) = render_core::telemetry::global_bus() {
                bus.emit_log(
                    record.level().as_str().to_ascii_lowercase().as_str(),
                    record.target(),
                    &record.args().to_string(),
                );
            }
        }
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

pub struct AppState {
    pub engine: Arc<EngineHandle>,
    pub scene_path: PathBuf,
    pub effects_dir: Option<PathBuf>,
    /// Lazily-resolved pack directory (from `pack.info`), cached so mask
    /// loads don't pay an RPC round trip each.
    pub pack_dir: std::sync::OnceLock<PathBuf>,
    /// Audio feature server child, when launched with `--audio` /
    /// `WZRD_AUDIO=1`. Killed alongside the engine on window close.
    pub audio_child: std::sync::Mutex<Option<std::process::Child>>,
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

/// True when the shell should auto-start the audio feature server:
/// `--audio` in the process args (pnpm: `pnpm tauri dev -- -- --audio`) or
/// `WZRD_AUDIO=1` in the environment.
fn audio_requested() -> bool {
    if std::env::args().skip(1).any(|a| a == "--audio") {
        return true;
    }
    matches!(
        std::env::var("WZRD_AUDIO").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Where the Realtime_PyAudio_FFT checkout lives. `WZRD_AUDIO_DIR`
/// overrides; default matches the documented dev setup.
fn audio_server_dir() -> PathBuf {
    if let Ok(p) = std::env::var("WZRD_AUDIO_DIR") {
        return PathBuf::from(p);
    }
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join("Documents/GitHub/Realtime_PyAudio_FFT")
}

/// Spawn `uv run audio-server --open` in the audio repo. The server sends
/// OSC to the engine's :9000 and auto-connects mid-flight, so ordering
/// relative to the engine spawn doesn't matter.
fn spawn_audio_server() -> Result<std::process::Child, String> {
    let dir = audio_server_dir();
    if !dir.exists() {
        return Err(format!(
            "audio server dir {} not found (set WZRD_AUDIO_DIR)",
            dir.display()
        ));
    }
    std::process::Command::new("uv")
        .args(["run", "audio-server", "--open"])
        .current_dir(&dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawning audio server in {}: {e}", dir.display()))
}

/// Full teardown, callable from any exit path: stop + join the render
/// thread (Core persists the session and drops the GPU context on that
/// thread), then kill the audio child. Idempotent.
fn shutdown_all(state: &AppState) {
    state.engine.shutdown();
    if let Ok(mut guard) = state.audio_child.lock() {
        if let Some(mut c) = guard.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

pub fn run() {
    let inner = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .build();
    log::set_max_level(inner.filter());
    log::set_boxed_logger(Box::new(TeeLogger { inner })).expect("logger already set");

    // The engine render thread now lives in this process — hold the same
    // App Nap / timer-coalescing opt-out the standalone binary takes, or
    // the frame rate collapses to ~9 fps when the app loses focus (§3.1b).
    #[cfg(target_os = "macos")]
    render_core::hold_latency_critical_assertion();

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
            log::info!(
                "scene={} effects_dir={:?} (engine in-process)",
                scene_path.display(),
                effects_dir,
            );

            let handle = match EngineHandle::start_in_process(app.handle().clone(), scene_path.clone())
            {
                Ok(h) => Arc::new(h),
                Err(e) => {
                    log::error!("could not start in-process engine: {e:#}");
                    return Err(e.to_string().into());
                }
            };

            let audio_child = if audio_requested() {
                match spawn_audio_server() {
                    Ok(c) => {
                        log::info!("spawned audio server pid {}", c.id());
                        Some(c)
                    }
                    Err(e) => {
                        // Non-fatal: the shell is fully usable without audio.
                        log::error!("could not start audio server: {e}");
                        None
                    }
                }
            } else {
                None
            };

            app.manage(AppState {
                engine: handle,
                scene_path,
                effects_dir,
                pack_dir: std::sync::OnceLock::new(),
                audio_child: std::sync::Mutex::new(audio_child),
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
            rpc::param_override,
            rpc::master_set,
            rpc::effect_describe,
            rpc::session_save,
            rpc::promote,
            rpc::pull,
            rpc::preview_set_source,
            rpc::probe_get_thresholds,
            rpc::probe_set_thresholds,
            rpc::wgsl_validate,
            rpc::effect_upsert,
            rpc::effect_remove,
            rpc::last_payload,
            rpc::preview_set_bounds,
            rpc::list_effects,
            rpc::read_effect,
            rpc::read_scene_file,
            rpc::write_scene_file,
            rpc::read_mask_png,
        ])
        .on_window_event(|window, event| {
            // The engine window has its own handler (engine.rs); closing the
            // operator (webview) window quits the app — teardown is
            // centralized in the ExitRequested handler below.
            if window.label() != "main" {
                return;
            }
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                window.app_handle().exit(0);
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building WZRD Tauri shell")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Joins the render thread *before* tauri destroys windows —
                // Core's surface (and its window handle) drop while the
                // native window is still alive (spike a: clean teardown).
                if let Some(state) = app.try_state::<AppState>() {
                    shutdown_all(&state);
                }
            }
        });
}
