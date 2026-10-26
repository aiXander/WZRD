//! WZRD Tauri shell — single-process host for the `render-core` engine
//! (app-collapse Step 2).
//!
//! The shell runs the engine as a library on an in-process render thread:
//! one process, two windows (webview + engine output). Tauri commands call
//! `rpc::dispatch` directly — the same §3.11 surface the engine's WS server
//! (still alive on 127.0.0.1:9123) serves to external MCP / remote clients.
//! The headless agent path (`render-core --scene foo.json`, winit-hosted)
//! is unchanged and lives in the separate `render-core` binary.
//!
//! A1 (gig plan D13): the engine re-roots in-process — `projects.rs` swaps
//! the current `Session` (engine + scene path) on a project switch or an
//! "engine restart"; the windows outlive every engine.

mod decks;
#[cfg(debug_assertions)]
mod devport;
mod engine;
mod fork;
mod hub;
mod layout;
mod library;
mod output;
mod projects;
mod relink;
mod rpc;
#[cfg(target_os = "macos")]
mod underlay;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use tauri::Manager;

use crate::engine::{EngineHandle, EngineWindows};

/// The bus the tee logger forwards to. `render_core::telemetry::global_bus`
/// is a set-once global — it would keep pointing at the first engine's bus
/// after a project switch — so the shell keeps its own, re-pointed by every
/// engine boot.
static LOG_BUS: Mutex<Option<render_core::telemetry::Bus>> = Mutex::new(None);

pub(crate) fn set_log_bus(bus: &render_core::telemetry::Bus) {
    *LOG_BUS.lock().expect("log bus lock") = Some(bus.clone());
}

/// Info and louder, except the GPU stack's chatter below Warn: wgpu-core logs
/// "Device::maintain: waiting for submission index N" at INFO on every
/// maintain (~180/s), so `RUST_LOG=info` would flood the Debug log. stderr
/// still gets whatever `RUST_LOG` asks for. (Twin of render-core's main.rs.)
fn forward_to_bus(record: &log::Record) -> bool {
    let gpu_stack = ["wgpu_core", "wgpu_hal", "naga"]
        .iter()
        .any(|p| record.target().starts_with(p));
    record.level() <= if gpu_stack { log::Level::Warn } else { log::Level::Info }
}

/// Tee logger: stderr via env_logger + the engine's `log` telemetry channel
/// once the bus exists (Core::new sets the global). Same shape as the
/// standalone binary's logger — the webview's Debug log stream depends on
/// it. Only Info and louder are forwarded (GPU stack: Warn and louder); the
/// bus internals log at trace, which also breaks any recursion.
struct TeeLogger {
    inner: env_logger::Logger,
}

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        self.inner.log(record);
        if forward_to_bus(record) && self.inner.matches(record) {
            // Clone out of the lock: emit_log logs at trace internally.
            let bus = LOG_BUS.lock().ok().and_then(|b| b.clone());
            if let Some(bus) = bus {
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
    /// Engine + preview windows — outlive every engine instance.
    pub windows: Arc<EngineWindows>,
    /// The open project. Swapped whole by `projects::open`; readers clone
    /// the `Arc` and never hold the lock across an engine call.
    session: RwLock<Arc<Session>>,
    /// Serialises project switches / restarts.
    pub switching: Mutex<()>,
    /// Audio feature server child, when launched with `--audio` /
    /// `WZRD_AUDIO=1`. Killed alongside the engine on window close.
    pub audio_child: Mutex<Option<std::process::Child>>,
}

/// One engine instance on one scene path. `engine` is `None` when the boot
/// failed (bad scene, missing pack) — the shell stays up so the operator
/// can pick another project or retry.
pub struct Session {
    pub engine: Option<Arc<EngineHandle>>,
    pub scene_path: PathBuf,
    pub effects_dir: Option<PathBuf>,
    /// Lazily-resolved pack directory (from `pack.info`), cached so mask
    /// loads don't pay an RPC round trip each.
    pub pack_dir: OnceLock<PathBuf>,
    /// Why `engine` is `None`.
    pub boot_error: Option<String>,
}

impl Session {
    pub fn new(scene_path: PathBuf, engine: Result<EngineHandle, String>) -> Self {
        let effects_dir = scene_path
            .parent()
            .map(|p| p.join("effects"))
            .filter(|p| p.exists());
        let (engine, boot_error) = match engine {
            Ok(e) => (Some(Arc::new(e)), None),
            Err(e) => (None, Some(e)),
        };
        Self {
            engine,
            scene_path,
            effects_dir,
            pack_dir: OnceLock::new(),
            boot_error,
        }
    }

    pub fn scene_dir(&self) -> PathBuf {
        self.scene_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

impl AppState {
    pub fn session(&self) -> Arc<Session> {
        Arc::clone(&self.session.read().expect("session lock"))
    }

    pub fn set_session(&self, s: Session) {
        *self.session.write().expect("session lock") = Arc::new(s);
    }

    /// The running engine, or the reason there is none.
    pub fn engine(&self) -> Result<Arc<EngineHandle>, String> {
        let s = self.session();
        s.engine.clone().ok_or_else(|| {
            s.boot_error
                .clone()
                .unwrap_or_else(|| "engine not running".to_string())
        })
    }
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
fn resolve_scene_path() -> Result<Option<PathBuf>, String> {
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
    let Some(raw) = raw else { return Ok(None) };
    let p = PathBuf::from(&raw);
    if p.is_absolute() {
        return Ok(Some(p));
    }
    if p.exists() {
        return Ok(Some(p));
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
            return Ok(Some(candidate));
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
    if let Some(engine) = state.session().engine.as_ref() {
        engine.shutdown();
    }
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
            // Scene: `WZRD_SCENE` / `--scene`, else the most recently opened
            // project from settings.json (a Finder launch has no env).
            let scene_path = match resolve_scene_path() {
                Ok(Some(p)) => Some(p),
                Ok(None) => projects::last_opened(app.handle()),
                Err(e) => {
                    log::error!("{e}");
                    return Err(e.into());
                }
            };
            let scene_path = scene_path.map(|p| p.canonicalize().unwrap_or(p));

            app.manage(output::Output::new(app.handle()));
            let windows = EngineWindows::create(app.handle()).map_err(|e| {
                log::error!("could not create engine windows: {e:#}");
                e.to_string()
            })?;
            app.manage(AppState {
                windows,
                session: RwLock::new(Arc::new(Session::new(
                    scene_path.clone().unwrap_or_default(),
                    Err("no project open — pick one from the project menu".into()),
                ))),
                switching: Mutex::new(()),
                audio_child: Mutex::new(None),
            });

            // A failed boot keeps the shell up (engine down + the reason),
            // so the operator can pick another project instead of facing a
            // dead app.
            if let Some(scene_path) = scene_path {
                log::info!("scene={} (engine in-process)", scene_path.display());
                if let Err(e) = projects::open(app.handle(), &app.state::<AppState>(), scene_path) {
                    log::error!("could not start in-process engine: {e}");
                }
            } else {
                log::warn!("no scene bound (WZRD_SCENE / --scene / recent project) — starting without an engine");
            }

            // Output starts off; scripted launches may turn it on.
            output::apply_env(app.handle());
            output::spawn_watcher(app.handle().clone());

            if audio_requested() {
                match spawn_audio_server() {
                    Ok(c) => {
                        log::info!("spawned audio server pid {}", c.id());
                        *app.state::<AppState>().audio_child.lock().expect("audio lock") = Some(c);
                    }
                    // Non-fatal: the shell is fully usable without audio.
                    Err(e) => log::error!("could not start audio server: {e}"),
                }
            }

            // Studio plan A1: attach to (or spawn) the hub off the main
            // thread, then keep its lamp current; scope media URLs.
            let hub = hub::Hub::new();
            hub.spawn_monitor(app.handle().clone());
            app.manage(hub);
            library::allow_scope(app.handle());

            #[cfg(debug_assertions)]
            devport::spawn(app.handle().clone());
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
            rpc::preview_file,
            rpc::probe_get_thresholds,
            rpc::probe_set_thresholds,
            rpc::alignment_get,
            rpc::alignment_set,
            rpc::alignment_reset,
            rpc::alignment_set_test_pattern,
            rpc::wgsl_validate,
            rpc::effect_upsert,
            rpc::effect_remove,
            rpc::effect_rename,
            rpc::effect_thumbnail,
            rpc::effect_thumbs,
            rpc::effect_library_read,
            rpc::effect_library_write,
            rpc::last_payload,
            rpc::preview_set_slot,
            rpc::preview_mode,
            rpc::telemetry_demand,
            rpc::list_effects,
            rpc::read_effect,
            rpc::read_scene_file,
            rpc::write_scene_file,
            rpc::read_mask_png,
            rpc::transport_set,
            rpc::param_list,
            rpc::looks_list,
            rpc::look_read,
            rpc::look_write,
            rpc::look_delete,
            projects::projects_list,
            projects::project_current,
            projects::project_open,
            projects::engine_restart,
            fork::project_save_as,
            relink::relink_missing,
            output::output_displays,
            output::output_state,
            output::output_set,
            output::output_options,
            output::output_identify,
            layout::ui_layout_read,
            layout::ui_layout_write,
            hub::hub_status,
            hub::hub_ensure,
            hub::hub_restart,
            hub::hub_call,
            library::library_dir,
            library::library_roots,
            library::library_record,
            library::library_tags,
            library::library_items,
            decks::decks_read,
            decks::decks_write,
            decks::deck_slots,
            decks::library_incoming,
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
