//! In-process engine host (app-collapse Step 2). `render-core` runs as a
//! library inside the Tauri process: the shell owns a second, webview-less
//! window for the engine output, hands its raw-window-handle to
//! `Core::init_gpu`, and drives `Core` from a dedicated render thread with
//! the same per-frame contract `WinitHost` uses (poll → pace → redraw |
//! offscreen). No subprocess, no WS hop for local calls.
//!
//! `EngineHandle::request` keeps the exact API the subprocess version had —
//! `rpc.rs` is unchanged — but routes through `rpc::dispatch` directly, the
//! same function the WS server calls, so the §3.11 method set stays one
//! code path. The WS server itself stays alive inside Core (`--ws-addr`
//! equivalent) for external MCP / remote-operator clients.
//!
//! macOS specifics (spike results in docs/TODO/single-process-collapse.md):
//! - tao has no `Occluded` event, so the render thread polls
//!   `NSWindow.occlusionState` once per frame (a cheap AppKit property
//!   read) to uphold the §3.1 invariant — never block on the swapchain of
//!   a possibly-occluded window.
//! - The render thread never calls back into tauri window methods that
//!   dispatch to the main thread (deadlock risk against sync Tauri
//!   commands blocking on the render thread); window sizes arrive via the
//!   `Resized` event instead.
//!
//! Re-rooting (gig plan A1, D13): the engine + preview windows live in
//! `EngineWindows`, created once per app; an `EngineHandle` is one engine
//! instance on one scene path. A project switch / "restart engine" is
//! `shutdown()` (same teardown as app exit: session + alignment saved, GPU
//! context dropped, OSC/WS ports released) followed by a fresh
//! `start_in_process` on the same windows.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use render_core::core::{Core, PreviewParams};
use render_core::rpc::{self, EngineCommand, JsonRpcRequest, RpcContext};
use render_core::telemetry::{is_sticky, ALL_CHANNELS};
use render_core::wgpu;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

/// WS port the in-process engine still binds for external clients (MCP,
/// remote operator). Fixed for v1, same as the subprocess era.
pub const DEFAULT_WS_PORT: u16 = 9123;

/// `WZRD_SPIKE` fires once per process, not once per engine — otherwise the
/// "engine crash → restart" check would crash the restarted engine too.
static SPIKE_FIRED: AtomicBool = AtomicBool::new(false);

/// Methods that mutate engine state and therefore round-trip through the
/// render thread (queued `EngineCommand` + blocking reply). Everything else
/// in §3.11 dispatches inline and returns immediately.
const QUEUED_METHODS: &[&str] = &[
    "scene.load",
    "scene.reload",
    "effect.upsert",
    "effect.remove",
    "effect.describe",
    "session.save",
    "promote",
    "pull",
    "preview.setSource",
    "preview.snapshot",
    "preview.file",
    "identity.setGroups",
];

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    pub running: bool,
    pub ws_addr: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TelemetryFrameOut {
    pub channel: String,
    pub payload: Value,
}

/// Host → render-thread messages for the native preview surface. Sent by
/// `preview_set_slot`; drained at frame boundary.
enum PreviewCmd {
    Config(PreviewCfg),
}

/// One preview-slot state as the render thread needs it: physical drawable
/// size, whether the React slot wants it shown, and what to show.
#[derive(Clone, Copy)]
struct PreviewCfg {
    width: u32,
    height: u32,
    visible: bool,
    params: PreviewParams,
}

/// A `<NativeSlot>` rect from the webview (docs/TODO/native-preview-
/// underlay.md §4.1). CSS px, viewport-relative (== AppKit points).
#[derive(Debug, Clone, Deserialize)]
pub struct PreviewSlot {
    /// Slot id. `"main"` (default) shows a leg / the program view per
    /// `params`; `"studio"` is the Studio monitor (studio plan P1) — it
    /// forces `params.file`, so it shows the `preview.file` source and no
    /// leg renders for it. Rooms are exclusive, so both share the one
    /// underlay view.
    #[serde(default)]
    pub id: Option<String>,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub visible: bool,
    /// Omitted → Perform defaults (the deck-toggle leg, no warp, no photo).
    #[serde(default)]
    pub params: Option<PreviewParams>,
}

/// Where the native preview lives.
enum PreviewHost {
    /// Default on macOS: an `NSView` under a transparent webview, so HTML
    /// draws over the image (`underlay.rs`).
    #[cfg(target_os = "macos")]
    Underlay(crate::underlay::PreviewView),
    /// `WZRD_PREVIEW=window` (and every non-macOS build): the collapse-
    /// Step-3 borderless child window ordered *above* the webview. Nothing
    /// can draw over it, so Prepare/Align fall back to the JPEG channel.
    /// Kept as the stage escape hatch until the underlay has soaked (D6).
    Window {
        window: tauri::Window,
        #[cfg(target_os = "macos")]
        ns: Option<PreviewNsRefs>,
    },
}

pub struct EngineHandle {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    /// Inline-dispatch context — same one the WS server holds.
    ctx: RpcContext,
    /// Queued-command channel into the render thread's `poll_inbound`.
    cmd_tx: crossbeam_channel::Sender<EngineCommand>,
    ws_addr: SocketAddr,
    /// Host → render thread: stop at the next frame boundary.
    stop: Arc<AtomicBool>,
    /// Render thread → host: false once the render loop has exited
    /// (cleanly or by panic).
    alive: Arc<AtomicBool>,
    last_error: Mutex<Option<String>>,
    render_join: Mutex<Option<thread::JoinHandle<()>>>,
    /// Snapshots of the sticky telemetry channels — let routes that mount
    /// late render their pills immediately.
    last_payloads: Mutex<HashMap<String, Value>>,
    /// Preview-surface config channel into the render thread.
    preview_tx: crossbeam_channel::Sender<PreviewCmd>,
}

/// The engine output window + native preview host, created once per app
/// and reused by every engine instance (A1: project switch / restart).
pub struct EngineWindows {
    engine: tauri::Window,
    preview: PreviewHost,
    /// Shared with whichever render thread is current; a fresh engine
    /// starts from `last`.
    sizes: Arc<SizeState>,
    /// Stop flag of the current engine — the engine window's `Destroyed`
    /// handler is registered once and stops whoever is running.
    current_stop: Mutex<Arc<AtomicBool>>,
    /// Last preview-slot config from the webview, replayed into a fresh
    /// engine so the native preview survives a restart without a re-layout.
    preview_cfg: Mutex<Option<PreviewCfg>>,
    /// The engine window is sized to the first pack that boots on it; later
    /// engines keep whatever size/placement the operator gave it.
    sized: AtomicBool,
    /// The streaming telemetry channels the webview currently displays
    /// (`telemetry_demand`). Lives here (not per engine) because the
    /// consumers stay mounted across an engine restart.
    demand: Arc<ChannelDemand>,
}

/// Which *streaming* telemetry channels the webview wants right now — the
/// full set, replaced on every `telemetry_demand` call (the webview
/// refcounts its consumers and sends the union, so a page reload that
/// skipped unmount cleanups can't leak a channel). Sticky channels and
/// `log` always flow (cheap, and sticky state must replay); everything
/// else — `preview`, `audio`, `transport`, `drivers`, `frame_stats`,
/// `perf` — reaches the webview only while demanded. The engine gates the
/// work behind them on `subscriber_count` (JPEG readback + the design leg
/// for `preview`; payload building for `audio`/`transport`/`drivers`), so
/// an undemanded channel costs nothing anywhere.
#[derive(Default)]
pub struct ChannelDemand {
    set: Mutex<HashSet<String>>,
    /// Bumped on every change — the fan-in polls this, not the mutex.
    gen: AtomicU64,
}

impl ChannelDemand {
    fn replace(&self, channels: HashSet<String>) {
        let mut set = self.set.lock().expect("demand lock");
        if *set != channels {
            *set = channels;
            self.gen.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn generation(&self) -> u64 {
        self.gen.load(Ordering::Relaxed)
    }

    /// The fan-in's subscription: always-on channels + the demanded ones.
    fn channels(&self) -> HashSet<String> {
        let set = self.set.lock().expect("demand lock");
        ALL_CHANNELS
            .iter()
            .filter(|c| always_on(c) || set.contains(**c))
            .map(|s| s.to_string())
            .collect()
    }
}

/// Channels the fan-in subscribes to regardless of demand: the sticky
/// ones (state the UI must always hold; `last_payload` snapshots + replay
/// depend on them) and `log` (the Debug room shows history).
fn always_on(channel: &str) -> bool {
    is_sticky(channel) || channel == "log"
}

/// Size updates flow window-event thread → render thread through shared
/// state, never through blocking tauri window queries (see module docs).
#[derive(Default)]
struct SizeState {
    /// Most recent known inner size (physical px).
    last: Mutex<(u32, u32)>,
    /// Set on every `Resized` event; taken by the render thread.
    pending: Mutex<Option<(u32, u32)>>,
}

/// AppKit-side handles for the preview child window. Positioning goes
/// through `contentView` + `convertRectToScreen` + `setFrame:display:`
/// directly — tao's top-left↔bottom-left conversions proved unreliable for
/// this (the preview landed one title-bar too high), and AppKit's own
/// converters are ground truth for any title-bar style, display layout,
/// and DPI. Only touched from the main thread (Tauri commands are sync).
#[cfg(target_os = "macos")]
struct PreviewNsRefs {
    /// `NSWindow*` of the main (webview) window.
    main: *mut objc2::runtime::AnyObject,
    /// `NSWindow*` of the borderless preview window.
    preview: *mut objc2::runtime::AnyObject,
}

#[cfg(target_os = "macos")]
unsafe impl Send for PreviewNsRefs {}
#[cfg(target_os = "macos")]
unsafe impl Sync for PreviewNsRefs {}

#[cfg(target_os = "macos")]
impl PreviewNsRefs {
    /// Move/size the preview window over a rect given in CSS px (== AppKit
    /// points), top-left-origin, relative to the webview's origin. The
    /// reference is `NSWindow.contentLayoutRect` — the frame minus the
    /// title bar — which matches where wry places the webview regardless
    /// of the full-size-content-view style (contentView/-
    /// contentRectForFrameRect both span the whole frame under that style
    /// and land one title-bar too high).
    fn place(&self, x: f64, y: f64, w: f64, h: f64) {
        use objc2::msg_send;
        use objc2_foundation::{NSPoint, NSRect, NSSize};
        unsafe {
            // Window coords, bottom-left origin.
            let layout: NSRect = msg_send![self.main, contentLayoutRect];
            let in_window = NSRect::new(
                NSPoint::new(
                    layout.origin.x + x,
                    layout.origin.y + layout.size.height - y - h,
                ),
                NSSize::new(w, h),
            );
            let on_screen: NSRect = msg_send![self.main, convertRectToScreen: in_window];
            // Snap to whole points — a fractional frame leaves a sub-pixel
            // strip of window background exposed past the Metal drawable.
            let snapped = NSRect::new(
                NSPoint::new(on_screen.origin.x.round(), on_screen.origin.y.round()),
                NSSize::new(on_screen.size.width.round(), on_screen.size.height.round()),
            );
            let _: () = msg_send![self.preview, setFrame: snapped, display: true];
        }
    }

    /// Order the preview in *without* making it key (tao's `show()` calls
    /// `makeKeyAndOrderFront:`, which steals focus from the webview — the
    /// "click three times to leave Perform" bug) and (re-)attach it as a
    /// child window of main. Re-attach is required after every hide:
    /// `orderOut:` silently detaches a child window from its parent, after
    /// which it stops tracking window drags and falls behind the main
    /// window.
    fn show(&self) {
        use objc2::msg_send;
        let nil: *mut objc2::runtime::AnyObject = std::ptr::null_mut();
        unsafe {
            let parent: *mut objc2::runtime::AnyObject = msg_send![self.preview, parentWindow];
            if parent.is_null() {
                // NSWindowAbove = 1
                let _: () = msg_send![self.main, addChildWindow: self.preview, ordered: 1isize];
            }
            let _: () = msg_send![self.preview, orderFront: nil];
        }
    }

    fn hide(&self) {
        use objc2::msg_send;
        let nil: *mut objc2::runtime::AnyObject = std::ptr::null_mut();
        unsafe {
            let _: () = msg_send![self.preview, orderOut: nil];
        }
    }
}

#[cfg(target_os = "macos")]
struct OcclusionProbe {
    /// `NSWindow*` of the engine window. The window outlives the render
    /// thread (shutdown joins the thread before tauri destroys windows).
    ns_window: *mut objc2::runtime::AnyObject,
}

#[cfg(target_os = "macos")]
unsafe impl Send for OcclusionProbe {}

#[cfg(target_os = "macos")]
impl OcclusionProbe {
    fn new(window: &tauri::Window) -> Option<Self> {
        match window.ns_window() {
            Ok(ptr) => Self::from_ptr(ptr),
            Err(e) => {
                log::warn!("occlusion probe unavailable (ns_window: {e}); assuming visible");
                None
            }
        }
    }

    fn from_ptr(ptr: *mut std::ffi::c_void) -> Option<Self> {
        (!ptr.is_null()).then(|| Self {
            ns_window: ptr.cast(),
        })
    }

    /// `NSWindowOcclusionStateVisible = 1 << 1`. A cheap property read —
    /// safe to poll once per frame from the render thread.
    fn visible(&self) -> bool {
        let state: usize = unsafe { objc2::msg_send![self.ns_window, occlusionState] };
        state & (1 << 1) != 0
    }
}

impl EngineWindows {
    /// Create the engine output window and the native preview window. Main
    /// thread only (tauri window creation); the setup hook qualifies. The
    /// engine window starts at a placeholder size and is sized to the first
    /// pack that boots on it.
    pub fn create(app: &AppHandle) -> Result<Arc<Self>> {
        let engine = build_engine_window(app).context("creating engine output window")?;
        let preview = build_preview_host(app)?;
        let size = engine.inner_size().context("querying engine window size")?;
        let sizes = Arc::new(SizeState::default());
        *sizes.last.lock().expect("size lock") = (size.width.max(1), size.height.max(1));

        let windows = Arc::new(Self {
            engine,
            preview,
            sizes,
            current_stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            preview_cfg: Mutex::new(None),
            sized: AtomicBool::new(false),
            demand: Arc::new(ChannelDemand::default()),
        });

        // Engine-window events: sizes flow to the render thread; closing the
        // projector window means "quit the app" (parity with WinitHost).
        // Registered once — handlers would pile up per restart otherwise.
        {
            let weak = Arc::downgrade(&windows);
            let app = app.clone();
            windows.engine.on_window_event(move |event| {
                let Some(w) = weak.upgrade() else { return };
                match event {
                    tauri::WindowEvent::Resized(s) => {
                        let dims = (s.width.max(1), s.height.max(1));
                        *w.sizes.last.lock().expect("size lock") = dims;
                        *w.sizes.pending.lock().expect("size lock") = Some(dims);
                    }
                    tauri::WindowEvent::CloseRequested { api, .. } => {
                        // Route through the app-exit path so teardown (render
                        // thread join, session snapshot) happens exactly once.
                        api.prevent_close();
                        app.exit(0);
                    }
                    tauri::WindowEvent::Destroyed => {
                        w.current_stop
                            .lock()
                            .expect("stop lock")
                            .store(true, Ordering::Relaxed);
                    }
                    _ => {}
                }
            });
        }
        Ok(windows)
    }

    /// Replace the webview's streaming-channel demand (see
    /// [`ChannelDemand`]). Unknown and always-on names are ignored. The
    /// fan-in picks the change up within one recv timeout.
    pub fn telemetry_demand(&self, channels: Vec<String>) {
        self.demand.replace(
            channels
                .into_iter()
                .filter(|c| ALL_CHANNELS.contains(&c.as_str()) && !always_on(c))
                .collect(),
        );
    }

    /// `"underlay"` (native view under a transparent webview — HTML draws
    /// over it, every room uses it) or `"window"` (the child window above
    /// the webview — Perform only; Prepare/Align use the JPEG channel).
    pub fn preview_mode(&self) -> &'static str {
        match self.preview {
            #[cfg(target_os = "macos")]
            PreviewHost::Underlay(_) => "underlay",
            PreviewHost::Window { .. } => "window",
        }
    }

    /// Place the native preview over the webview's `<NativeSlot>` rect (or
    /// hide it) and hand the engine the slot's physical size + params. The
    /// config is remembered (replayed into a restarted engine) and forwarded
    /// to the running engine, if any.
    ///
    /// Ordering upholds §3.1 for the preview swapchain: on show the view is
    /// placed and unhidden *before* the engine is told to present; on hide
    /// the engine is told first.
    pub fn set_preview_slot(
        &self,
        app: &AppHandle,
        engine: Option<&EngineHandle>,
        slot: &PreviewSlot,
    ) -> Result<()> {
        use tauri::Manager;
        let show = slot.visible && slot.width >= 4.0 && slot.height >= 4.0;
        let mut params = slot.params.unwrap_or_default();
        params.file = slot.id.as_deref() == Some("studio");
        let hidden = PreviewCfg {
            width: 1,
            height: 1,
            visible: false,
            params,
        };
        if !show {
            *self.preview_cfg.lock().expect("preview cfg lock") = Some(hidden);
            if let Some(engine) = engine {
                engine.send_preview_cfg(hidden);
            }
        }
        let scale = app
            .get_webview_window("main")
            .ok_or_else(|| anyhow!("main window missing"))?
            .scale_factor()
            .unwrap_or(1.0);
        let (x, y, width, height) = (slot.x, slot.y, slot.width, slot.height);
        match &self.preview {
            #[cfg(target_os = "macos")]
            PreviewHost::Underlay(view) => {
                if show {
                    view.place(x, y, width, height);
                }
                view.set_hidden(!show);
            }
            PreviewHost::Window {
                window: preview,
                #[cfg(target_os = "macos")]
                ns,
            } => {
                #[cfg(target_os = "macos")]
                {
                    // Place + order in via AppKit (see PreviewNsRefs) —
                    // never through tauri show(), which would make the
                    // preview the key window and steal focus.
                    let _ = preview;
                    if let Some(refs) = ns.as_ref() {
                        if show {
                            refs.place(x, y, width, height);
                            refs.show();
                        } else {
                            refs.hide();
                        }
                    }
                }
                #[cfg(not(target_os = "macos"))]
                if show {
                    let main = app
                        .get_webview_window("main")
                        .ok_or_else(|| anyhow!("main window missing"))?;
                    let origin = main.inner_position().context("main window position")?;
                    let px = origin.x + (x * scale).round() as i32;
                    let py = origin.y + (y * scale).round() as i32;
                    preview
                        .set_position(tauri::Position::Physical(tauri::PhysicalPosition::new(
                            px, py,
                        )))
                        .context("positioning preview window")?;
                    preview
                        .set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                            ((width * scale).round() as u32).max(1),
                            ((height * scale).round() as u32).max(1),
                        )))
                        .context("sizing preview window")?;
                    preview.show().context("showing preview window")?;
                } else {
                    preview.hide().context("hiding preview window")?;
                }
            }
        }
        if show {
            let cfg = PreviewCfg {
                width: ((width * scale).round() as u32).max(1),
                height: ((height * scale).round() as u32).max(1),
                visible: true,
                params,
            };
            *self.preview_cfg.lock().expect("preview cfg lock") = Some(cfg);
            if let Some(engine) = engine {
                engine.send_preview_cfg(cfg);
            }
        }
        Ok(())
    }
}

impl EngineHandle {
    /// Boot an engine on `scene_path` inside this process: build `Core`,
    /// bring up wgpu on the (reused) engine window, attach the preview
    /// surface, and start the render thread. Must be called on the main
    /// thread (surface attach + window sizing); the setup hook and sync
    /// Tauri commands qualify. The previous engine, if any, must already be
    /// `shutdown()` — the WS/OSC ports are single-owner.
    pub fn start_in_process(
        app: AppHandle,
        windows: &Arc<EngineWindows>,
        scene_path: PathBuf,
        log_bus: impl FnOnce(&render_core::telemetry::Bus),
    ) -> Result<Self> {
        let launch = Instant::now();
        let ws_addr: SocketAddr = format!("127.0.0.1:{DEFAULT_WS_PORT}")
            .parse()
            .expect("valid addr");

        let cli = render_core::Cli {
            pack: None,
            scene: scene_path,
            effects: None,
            display: None,
            windowed: true,
            no_osc: false,
            osc_addr: "127.0.0.1:9000".parse().expect("valid osc addr"),
            ws_addr: Some(ws_addr),
            frame_cap_hz: 240,
            perf: render_core::PerfMode::On,
            // `WZRD_GPU_TIMING=0` turns the perf channel's GPU columns off
            // (overhead A/B); N times one frame in N.
            gpu_timing: std::env::var("WZRD_GPU_TIMING")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(render_core::DEFAULT_GPU_TIMING),
        };
        let mut core = Core::new(&cli).context("initialising engine core")?;
        let (ctx, cmd_tx) = core.control_channel();
        log_bus(&ctx.bus);

        let window = windows.engine.clone();
        if !windows.sized.swap(true, Ordering::Relaxed) {
            size_engine_window(&window, core.pack().atlas_width, core.pack().atlas_height)
                .context("sizing engine output window")?;
        }
        let size = window.inner_size().context("querying engine window size")?;
        let sizes = Arc::clone(&windows.sizes);
        *sizes.last.lock().expect("size lock") = (size.width.max(1), size.height.max(1));
        sizes.pending.lock().expect("size lock").take();
        core.init_gpu(window.clone(), size.width.max(1), size.height.max(1))
            .context("initialising wgpu on the engine window")?;
        log::info!(
            "engine core + GPU up in {:.0} ms (window {}x{})",
            launch.elapsed().as_secs_f64() * 1000.0,
            size.width,
            size.height
        );

        let attached = match &windows.preview {
            #[cfg(target_os = "macos")]
            PreviewHost::Underlay(view) => {
                core.attach_preview_surface(view.surface_target(), 320, 180)
            }
            PreviewHost::Window { window, .. } => {
                core.attach_preview_surface(window.clone(), 320, 180)
            }
        };
        if let Err(e) = attached {
            // Non-fatal — the JPEG thumbnail path still works.
            log::error!("could not attach native preview surface: {e:#}");
        }
        // The underlay may render below the slot's size but never above the
        // composite's (D3 / open question 3): its layer scales the drawable.
        #[cfg(target_os = "macos")]
        let cap_preview = matches!(windows.preview, PreviewHost::Underlay(_));
        #[cfg(not(target_os = "macos"))]
        let cap_preview = false;

        let stop = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        *windows.current_stop.lock().expect("stop lock") = Arc::clone(&stop);

        #[cfg(target_os = "macos")]
        let occlusion = OcclusionProbe::new(&window);
        // The underlay is visible exactly when the main window is; the
        // child window has its own occlusion state.
        #[cfg(target_os = "macos")]
        let preview_occlusion = match &windows.preview {
            PreviewHost::Underlay(view) => OcclusionProbe::from_ptr(view.ns_window()),
            PreviewHost::Window { window, .. } => OcclusionProbe::new(window),
        };

        let (preview_tx, preview_rx) = crossbeam_channel::unbounded::<PreviewCmd>();
        // A restarted engine picks up the preview slot where the webview
        // left it — the React side only re-sends bounds on layout changes.
        if let Some(cfg) = *windows.preview_cfg.lock().expect("preview cfg lock") {
            let _ = preview_tx.send(PreviewCmd::Config(cfg));
        }

        let inner = Arc::new(EngineInner {
            ctx: ctx.clone(),
            cmd_tx,
            ws_addr,
            stop: Arc::clone(&stop),
            alive: Arc::clone(&alive),
            last_error: Mutex::new(None),
            render_join: Mutex::new(None),
            last_payloads: Mutex::new(HashMap::new()),
            preview_tx,
        });

        // Render thread — owns Core for its whole life.
        let render_join = {
            let inner = Arc::clone(&inner);
            let app = app.clone();
            thread::Builder::new()
                .name("engine-render".into())
                .spawn(move || {
                    render_thread(
                        core,
                        inner,
                        app,
                        sizes,
                        preview_rx,
                        cap_preview,
                        #[cfg(target_os = "macos")]
                        occlusion,
                        #[cfg(target_os = "macos")]
                        preview_occlusion,
                        launch,
                    )
                })
                .context("spawning engine render thread")?
        };
        *inner.render_join.lock().expect("join lock") = Some(render_join);

        // Telemetry fan-in: bus subscription → Tauri events, replacing the
        // WS notification path for the local webview. Also the 1 Hz
        // `engine:status` heartbeat the status strip expects.
        {
            let inner = Arc::clone(&inner);
            let bus = ctx.bus.clone();
            let demand = Arc::clone(&windows.demand);
            thread::Builder::new()
                .name("engine-telemetry".into())
                .spawn(move || telemetry_loop(inner, app, bus, demand))
                .context("spawning telemetry fan-in thread")?;
        }

        Ok(Self { inner })
    }

    /// Issue a §3.11 request. Inline methods dispatch on the calling
    /// thread; queued (state-mutating) methods run on a helper thread so
    /// the timeout holds even if the render thread is wedged.
    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let req = JsonRpcRequest {
            jsonrpc: Some("2.0".into()),
            id: None,
            method: method.to_string(),
            params,
        };
        // §5.10 actor identity: Tauri direct dispatch is always the local
        // operator UI (per-connection default, never per call).
        if !QUEUED_METHODS.contains(&method) {
            let mut actor = rpc::Actor::Ui;
            return rpc::dispatch(&self.inner.ctx, &self.inner.cmd_tx, &req, &mut actor)
                .map_err(|e| anyhow!(e.message));
        }
        let ctx = self.inner.ctx.clone();
        let cmd_tx = self.inner.cmd_tx.clone();
        let (tx, rx) = crossbeam_channel::bounded(1);
        thread::Builder::new()
            .name("engine-rpc".into())
            .spawn(move || {
                let mut actor = rpc::Actor::Ui;
                let _ = tx.send(rpc::dispatch(&ctx, &cmd_tx, &req, &mut actor));
            })
            .context("spawning rpc helper thread")?;
        match rx.recv_timeout(timeout) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(anyhow!(e.message)),
            Err(_) => bail!("engine RPC timeout after {timeout:?} (method={method})"),
        }
    }

    fn send_preview_cfg(&self, cfg: PreviewCfg) {
        let _ = self.inner.preview_tx.send(PreviewCmd::Config(cfg));
    }

    pub fn last_payload(&self, channel: &str) -> Option<Value> {
        self.inner
            .last_payloads
            .lock()
            .expect("last payloads lock")
            .get(channel)
            .cloned()
    }

    pub fn status(&self) -> EngineStatus {
        EngineStatus {
            running: self.inner.alive.load(Ordering::Relaxed)
                && !self.inner.stop.load(Ordering::Relaxed),
            ws_addr: Some(self.inner.ws_addr.to_string()),
            last_error: self
                .inner
                .last_error
                .lock()
                .expect("last error lock")
                .clone(),
        }
    }

    /// Stop the render loop at the next frame boundary and join it. Core
    /// persists the session sidecar (`on_exit`) and drops its GPU context
    /// on that thread before this returns — clean teardown ordering for
    /// app exit (spike a). Idempotent.
    pub fn shutdown(&self) {
        self.inner.stop.store(true, Ordering::Relaxed);
        let join = self.inner.render_join.lock().expect("join lock").take();
        if let Some(join) = join {
            if let Err(e) = join.join() {
                log::error!("render thread join after panic: {e:?}");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_thread(
    mut core: Core,
    inner: Arc<EngineInner>,
    app: AppHandle,
    sizes: Arc<SizeState>,
    preview_rx: crossbeam_channel::Receiver<PreviewCmd>,
    cap_preview: bool,
    #[cfg(target_os = "macos")] occlusion: Option<OcclusionProbe>,
    #[cfg(target_os = "macos")] preview_occlusion: Option<OcclusionProbe>,
    launch: Instant,
) {
    let spike = if SPIKE_FIRED.load(Ordering::Relaxed) {
        String::new()
    } else {
        std::env::var("WZRD_SPIKE").unwrap_or_default()
    };
    if !spike.is_empty() {
        log::warn!("WZRD_SPIKE={spike} armed — will trigger ~5 s after launch");
    }
    let stop = Arc::clone(&inner.stop);
    let mut spike_armed = !spike.is_empty();
    let mut presented_first = false;
    // React-slot visibility of the native preview (set via
    // `preview_set_slot`); combined with the hosting window's occlusion
    // state each frame.
    let mut preview_wanted = false;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loop {
            if stop.load(Ordering::Relaxed) {
                core.on_exit();
                break;
            }
            core.poll_inbound();
            if core.exit_requested() {
                // SIGTERM/SIGINT — Core already snapshotted the session.
                app.exit(0);
                break;
            }
            if let Some((w, h)) = sizes.pending.lock().expect("size lock").take() {
                core.resize(w, h);
            }
            while let Ok(PreviewCmd::Config(cfg)) = preview_rx.try_recv() {
                let (w, h) = if cap_preview {
                    let (cw, ch) = core.composite_size();
                    fit_within(cfg.width, cfg.height, cw, ch)
                } else {
                    (cfg.width, cfg.height)
                };
                core.resize_preview_surface(w, h);
                core.set_preview_params(cfg.params);
                preview_wanted = cfg.visible;
            }
            #[cfg(target_os = "macos")]
            if let Some(probe) = occlusion.as_ref() {
                core.set_occluded(!probe.visible());
            }
            {
                // §3.1 applies to the preview swapchain too: only present
                // while the React slot wants it AND AppKit reports the
                // hosting window visible (not minimized/covered).
                #[cfg(target_os = "macos")]
                let effective = preview_wanted
                    && preview_occlusion.as_ref().map_or(true, |p| p.visible());
                #[cfg(not(target_os = "macos"))]
                let effective = preview_wanted;
                core.set_preview_visible(effective);
            }
            core.pace_frame();

            if spike_armed && launch.elapsed() >= Duration::from_secs(5) {
                spike_armed = false;
                SPIKE_FIRED.store(true, Ordering::Relaxed);
                match spike.as_str() {
                    "panic" => panic!("deliberate Step-2 spike panic (WZRD_SPIKE=panic)"),
                    "device_loss" => core.spike_force_device_loss(),
                    other => log::warn!("unknown WZRD_SPIKE value {other:?} — ignoring"),
                }
            }

            if core.occluded() {
                core.render_offscreen_frame();
                continue;
            }
            match core.redraw() {
                Ok(()) => {
                    if !presented_first {
                        presented_first = true;
                        log::info!(
                            "first frame presented {:.0} ms after launch",
                            launch.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    let (w, h) = *sizes.last.lock().expect("size lock");
                    core.resize(w, h);
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    log::error!("GPU out of memory — stopping render thread");
                    core.on_exit();
                    break;
                }
                Err(wgpu::SurfaceError::Timeout) => {
                    log::warn!("frame timeout, skipping");
                }
            }
        }
    }));

    if let Err(payload) = result {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        log::error!(
            "engine render thread panicked: {msg} — webview stays up; \
             session/scene state on disk is unaffected (atomic writes)"
        );
        *inner.last_error.lock().expect("last error lock") = Some(msg);
    }
    inner.alive.store(false, Ordering::Relaxed);
    let _ = app.emit(
        "engine:status",
        json!({
            "running": false,
            "ws_addr": inner.ws_addr.to_string(),
            "last_error": inner.last_error.lock().expect("last error lock").clone(),
        }),
    );
    log::info!("engine render thread exiting");
}

/// Bus subscription → batched `engine:telemetry` Tauri events + sticky
/// snapshots + the 1 Hz `engine:status` heartbeat. Exits when the stop flag
/// is set.
///
/// Subscribes to the always-on channels plus whatever the webview demands
/// ([`ChannelDemand`]). One Tauri emit is one `webview.eval` of a fresh JS
/// string, so frames are **batched**: the event payload is an array of
/// `{channel, payload}`, flushed at most once per `BATCH` (leading edge —
/// a lone event goes out immediately). Within a batch a streaming channel
/// keeps only its newest frame; sticky channels and `log` keep every frame
/// (each `deck` transition / `changes` entry / log line matters).
fn telemetry_loop(
    inner: Arc<EngineInner>,
    app: AppHandle,
    bus: render_core::telemetry::Bus,
    demand: Arc<ChannelDemand>,
) {
    /// ≈ the fastest channel's own rate (audio/transport, 30 Hz).
    const BATCH: Duration = Duration::from_millis(33);
    let mut seen_gen = demand.generation();
    let (sub_id, rx) = bus.subscribe(demand.channels());
    let mut last_status_emit = Instant::now() - Duration::from_secs(1);
    let mut last_flush = Instant::now() - BATCH;
    let mut batch: Vec<TelemetryFrameOut> = Vec::with_capacity(16);
    let mut cpu = render_core::perf::ThreadCpu::register("engine-telemetry");

    let push = |batch: &mut Vec<TelemetryFrameOut>, frame: render_core::telemetry::TelemetryFrame| {
        // Sticky set comes from the engine, never a local copy — a
        // second list here is how `alignment` went missing from the
        // Align tab (it only emits at boot, so the snapshot *is* the
        // hydration path).
        if is_sticky(&frame.channel) {
            inner
                .last_payloads
                .lock()
                .expect("last payloads lock")
                .insert(frame.channel.clone(), frame.payload.clone());
        } else if !always_on(&frame.channel) {
            if let Some(slot) = batch.iter_mut().find(|f| f.channel == frame.channel) {
                slot.payload = frame.payload;
                return;
            }
        }
        batch.push(TelemetryFrameOut {
            channel: frame.channel,
            payload: frame.payload,
        });
    };

    loop {
        cpu.tick();
        let gen = demand.generation();
        if gen != seen_gen {
            seen_gen = gen;
            bus.update_channels(sub_id, demand.channels());
        }
        let timeout = if batch.is_empty() {
            Duration::from_millis(200)
        } else {
            BATCH
                .saturating_sub(last_flush.elapsed())
                .max(Duration::from_millis(1))
        };
        match rx.recv_timeout(timeout) {
            Ok(frame) => {
                push(&mut batch, frame);
                // Emitters fire several channels back to back (audio +
                // transport + drivers in one engine tick): take them all.
                while let Ok(frame) = rx.try_recv() {
                    push(&mut batch, frame);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
        if !batch.is_empty() && last_flush.elapsed() >= BATCH {
            // Perf PM1 `telemetry_bps.webview` — the webview hop's load.
            // `json_len` re-serializes, so only while the profiler is on.
            if render_core::perf::enabled() {
                for f in &batch {
                    render_core::perf::count_telemetry(
                        render_core::perf::Hop::Webview,
                        &f.channel,
                        render_core::perf::json_len(&f.payload),
                    );
                }
            }
            let _ = app.emit("engine:telemetry", &batch);
            batch.clear();
            last_flush = Instant::now();
        }
        if inner.stop.load(Ordering::Relaxed) {
            break;
        }
        if last_status_emit.elapsed() >= Duration::from_secs(1) {
            let running = inner.alive.load(Ordering::Relaxed);
            let _ = app.emit(
                "engine:status",
                json!({ "running": running, "ws_addr": inner.ws_addr.to_string() }),
            );
            last_status_emit = Instant::now();
        }
    }
    bus.unsubscribe(sub_id);
    log::info!("engine telemetry fan-in thread exiting");
}

/// Create the engine output window. Default: a decorated window, sized to
/// the first pack that boots on it (`size_engine_window`; dev parity with
/// the old `--windowed` spawn). With `WZRD_DISPLAY=<idx>`: borderless,
/// filling that monitor — the projector deployment shape.
fn build_engine_window(app: &AppHandle) -> Result<tauri::Window> {
    let mut builder = tauri::window::WindowBuilder::new(app, "engine")
        .title("WZRD Engine Output")
        .resizable(true);
    if wzrd_display().is_some() {
        builder = builder.decorations(false);
    }
    builder.build().context("building engine window")
}

fn wzrd_display() -> Option<usize> {
    std::env::var("WZRD_DISPLAY").ok().and_then(|s| s.parse().ok())
}

fn size_engine_window(window: &tauri::Window, pack_w: u32, pack_h: u32) -> Result<()> {
    match wzrd_display() {
        Some(idx) => {
            let monitors = window
                .available_monitors()
                .context("listing monitors")?;
            let monitor = monitors.get(idx).ok_or_else(|| {
                anyhow!(
                    "WZRD_DISPLAY={idx} but only {} monitor(s) detected",
                    monitors.len()
                )
            })?;
            window
                .set_position(tauri::Position::Physical(*monitor.position()))
                .context("positioning engine window")?;
            window
                .set_size(tauri::Size::Physical(*monitor.size()))
                .context("sizing engine window")?;
        }
        None => {
            window
                .set_size(tauri::Size::Physical(tauri::PhysicalSize::new(
                    pack_w.max(320),
                    pack_h.max(240),
                )))
                .context("sizing engine window")?;
        }
    }
    Ok(())
}

/// Scale `(w, h)` down (never up) to fit inside `(max_w, max_h)`, keeping
/// aspect — a monitor drawable above composite resolution buys nothing.
fn fit_within(w: u32, h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if w <= max_w && h <= max_h {
        return (w, h);
    }
    let k = (max_w as f64 / w as f64).min(max_h as f64 / h as f64);
    (
        ((w as f64 * k).round() as u32).max(1),
        ((h as f64 * k).round() as u32).max(1),
    )
}

/// `WZRD_PREVIEW=window` selects the old child-window monitor (D6).
fn window_preview_requested() -> bool {
    std::env::var("WZRD_PREVIEW").is_ok_and(|v| v == "window")
}

/// App `--bg` (`src/index.css`, `6 7 9`) — the main window's background
/// once the webview stops painting its own.
#[cfg(target_os = "macos")]
const APP_BG: (f64, f64, f64) = (6.0 / 255.0, 7.0 / 255.0, 9.0 / 255.0);

/// Build the preview host: the underlay view (macOS default), else — or if
/// installing it fails — the child window.
fn build_preview_host(app: &AppHandle) -> Result<PreviewHost> {
    #[cfg(target_os = "macos")]
    if !window_preview_requested() {
        use tauri::Manager;
        let installed = app
            .get_webview_window("main")
            .ok_or_else(|| anyhow!("main window missing"))
            .and_then(|main| main.ns_window().map_err(|e| anyhow!("{e}")))
            .and_then(|ns| crate::underlay::PreviewView::install(ns, APP_BG));
        match installed {
            Ok(view) => return Ok(PreviewHost::Underlay(view)),
            Err(e) => log::error!(
                "preview underlay unavailable ({e:#}) — falling back to the child window"
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = window_preview_requested();
    let window = build_preview_window(app).context("creating preview window")?;
    #[cfg(target_os = "macos")]
    let ns = preview_ns_refs(app, &window);
    log::info!("native preview: child window (WZRD_PREVIEW=window)");
    Ok(PreviewHost::Window {
        window,
        #[cfg(target_os = "macos")]
        ns,
    })
}

/// Create the native operator-preview window (collapse Step 3): borderless,
/// hidden until the React slot mounts. Child-window attachment + placement
/// happen through `PreviewNsRefs` (macOS).
fn build_preview_window(app: &AppHandle) -> Result<tauri::Window> {
    tauri::window::WindowBuilder::new(app, "preview")
        .title("WZRD Preview")
        .decorations(false)
        .resizable(false)
        .visible(false)
        .focused(false)
        .build()
        .context("building preview window")
}

/// Resolve the AppKit window pointers for preview placement. `None` (with a
/// warning) if either handle is unavailable — the preview then simply won't
/// track/position correctly rather than failing the boot.
#[cfg(target_os = "macos")]
fn preview_ns_refs(app: &AppHandle, preview: &tauri::Window) -> Option<PreviewNsRefs> {
    use tauri::Manager;
    let main = app.get_webview_window("main")?;
    match (main.ns_window(), preview.ns_window()) {
        (Ok(main_ns), Ok(prev_ns)) if !main_ns.is_null() && !prev_ns.is_null() => {
            let refs = PreviewNsRefs {
                main: main_ns.cast(),
                preview: prev_ns.cast(),
            };
            // Click-through: the preview is display-only. This also means
            // it can never become the key window, so it cannot steal focus
            // from the webview no matter how it gets ordered.
            unsafe {
                let _: () = objc2::msg_send![refs.preview, setIgnoresMouseEvents: true];
                // Opaque black window background: the default (near-white)
                // system background peeks out as a hairline wherever the
                // Metal drawable doesn't pixel-exactly cover the frame.
                let black: *mut objc2::runtime::AnyObject =
                    objc2::msg_send![objc2::class!(NSColor), blackColor];
                let _: () = objc2::msg_send![refs.preview, setBackgroundColor: black];
                let _: () = objc2::msg_send![refs.preview, setOpaque: true];
            }
            Some(refs)
        }
        _ => {
            log::warn!("preview NSWindow handles unavailable — native preview disabled");
            None
        }
    }
}
