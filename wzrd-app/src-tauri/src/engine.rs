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

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use render_core::core::Core;
use render_core::rpc::{self, EngineCommand, JsonRpcRequest, RpcContext};
use render_core::telemetry::ALL_CHANNELS;
use render_core::wgpu;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

/// WS port the in-process engine still binds for external clients (MCP,
/// remote operator). Fixed for v1, same as the subprocess era.
pub const DEFAULT_WS_PORT: u16 = 9123;

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

/// Host → render-thread messages for the native preview surface (collapse
/// Step 3). Sent by `preview_set_bounds`; drained at frame boundary.
enum PreviewCmd {
    /// New physical surface size + whether the React slot wants it shown.
    Config {
        width: u32,
        height: u32,
        visible: bool,
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
    /// AppKit handles for preview placement + child re-attach (main thread
    /// only — see `PreviewNsRefs`).
    #[cfg(target_os = "macos")]
    preview_ns: Option<PreviewNsRefs>,
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
            Ok(ptr) if !ptr.is_null() => Some(Self {
                ns_window: ptr.cast(),
            }),
            Ok(_) => None,
            Err(e) => {
                log::warn!("occlusion probe unavailable (ns_window: {e}); assuming visible");
                None
            }
        }
    }

    /// `NSWindowOcclusionStateVisible = 1 << 1`. A cheap property read —
    /// safe to poll once per frame from the render thread.
    fn visible(&self) -> bool {
        let state: usize = unsafe { objc2::msg_send![self.ns_window, occlusionState] };
        state & (1 << 1) != 0
    }
}

impl EngineHandle {
    /// Boot the engine inside this process: build `Core`, create the engine
    /// output window, bring up wgpu on it, and start the render thread.
    /// Must be called on the main thread (tauri window creation + first
    /// surface attach); the setup hook qualifies.
    pub fn start_in_process(app: AppHandle, scene_path: PathBuf) -> Result<Self> {
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
        };
        let mut core = Core::new(&cli).context("initialising engine core")?;
        let (ctx, cmd_tx) = core.control_channel();

        let window = build_engine_window(&app, core.pack().atlas_width, core.pack().atlas_height)
            .context("creating engine output window")?;
        let size = window.inner_size().context("querying engine window size")?;
        core.init_gpu(window.clone(), size.width.max(1), size.height.max(1))
            .context("initialising wgpu on the engine window")?;
        log::info!(
            "engine core + GPU up in {:.0} ms (window {}x{})",
            launch.elapsed().as_secs_f64() * 1000.0,
            size.width,
            size.height
        );

        // Native operator-preview window (collapse Step 3): borderless
        // child window of the main webview window, positioned by the React
        // layout over its preview slot. Hidden until the slot mounts.
        let preview_window = build_preview_window(&app).context("creating preview window")?;
        #[cfg(target_os = "macos")]
        let preview_ns = preview_ns_refs(&app, &preview_window);
        if let Err(e) = core.attach_preview_surface(preview_window.clone(), 320, 180) {
            // Non-fatal — the JPEG thumbnail path still works.
            log::error!("could not attach native preview surface: {e:#}");
        }

        let stop = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));
        let sizes = Arc::new(SizeState::default());
        *sizes.last.lock().expect("size lock") = (size.width.max(1), size.height.max(1));

        #[cfg(target_os = "macos")]
        let occlusion = OcclusionProbe::new(&window);
        #[cfg(target_os = "macos")]
        let preview_occlusion = OcclusionProbe::new(&preview_window);

        let (preview_tx, preview_rx) = crossbeam_channel::unbounded::<PreviewCmd>();

        // Engine-window events: sizes flow to the render thread; closing the
        // projector window means "quit the app" (parity with WinitHost).
        {
            let sizes = Arc::clone(&sizes);
            let stop = Arc::clone(&stop);
            let app = app.clone();
            window.on_window_event(move |event| match event {
                tauri::WindowEvent::Resized(s) => {
                    let dims = (s.width.max(1), s.height.max(1));
                    *sizes.last.lock().expect("size lock") = dims;
                    *sizes.pending.lock().expect("size lock") = Some(dims);
                }
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    // Route through the app-exit path so teardown (render
                    // thread join, session snapshot) happens exactly once.
                    api.prevent_close();
                    app.exit(0);
                }
                tauri::WindowEvent::Destroyed => {
                    stop.store(true, Ordering::Relaxed);
                }
                _ => {}
            });
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
            #[cfg(target_os = "macos")]
            preview_ns,
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
            thread::Builder::new()
                .name("engine-telemetry".into())
                .spawn(move || telemetry_loop(inner, app, bus))
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

    /// Position the native preview window over the React layout's preview
    /// slot (CSS px, webview-viewport-relative) — or hide it. Called from
    /// the `preview_set_bounds` Tauri command on layout/visibility changes.
    /// The preview window is a child of the main window, so window drags
    /// track automatically; only layout-relative changes come through here.
    pub fn set_preview_bounds(
        &self,
        app: &AppHandle,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        visible: bool,
    ) -> Result<()> {
        use tauri::Manager;
        let preview = app
            .get_window("preview")
            .ok_or_else(|| anyhow!("preview window missing"))?;
        let show = visible && width >= 4.0 && height >= 4.0;
        if show {
            let main = app
                .get_webview_window("main")
                .ok_or_else(|| anyhow!("main window missing"))?;
            let scale = main.scale_factor().unwrap_or(1.0);
            let pw = ((width * scale).round() as u32).max(1);
            let ph = ((height * scale).round() as u32).max(1);
            #[cfg(target_os = "macos")]
            {
                // Place + order in via AppKit (see PreviewNsRefs) — never
                // through tauri show(), which would make the preview the
                // key window and steal focus from the webview.
                let _ = &preview;
                if let Some(refs) = self.inner.preview_ns.as_ref() {
                    refs.place(x, y, width, height);
                    refs.show();
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let origin = main.inner_position().context("main window position")?;
                let px = origin.x + (x * scale).round() as i32;
                let py = origin.y + (y * scale).round() as i32;
                preview
                    .set_position(tauri::Position::Physical(tauri::PhysicalPosition::new(
                        px, py,
                    )))
                    .context("positioning preview window")?;
                preview
                    .set_size(tauri::Size::Physical(tauri::PhysicalSize::new(pw, ph)))
                    .context("sizing preview window")?;
                preview.show().context("showing preview window")?;
            }
            let _ = self.inner.preview_tx.send(PreviewCmd::Config {
                width: pw,
                height: ph,
                visible: true,
            });
        } else {
            #[cfg(target_os = "macos")]
            {
                let _ = &preview;
                if let Some(refs) = self.inner.preview_ns.as_ref() {
                    refs.hide();
                }
            }
            #[cfg(not(target_os = "macos"))]
            preview.hide().context("hiding preview window")?;
            let _ = self.inner.preview_tx.send(PreviewCmd::Config {
                width: 1,
                height: 1,
                visible: false,
            });
        }
        Ok(())
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
    #[cfg(target_os = "macos")] occlusion: Option<OcclusionProbe>,
    #[cfg(target_os = "macos")] preview_occlusion: Option<OcclusionProbe>,
    launch: Instant,
) {
    let spike = std::env::var("WZRD_SPIKE").unwrap_or_default();
    if !spike.is_empty() {
        log::warn!("WZRD_SPIKE={spike} armed — will trigger ~5 s after launch");
    }
    let stop = Arc::clone(&inner.stop);
    let mut spike_armed = !spike.is_empty();
    let mut presented_first = false;
    // React-slot visibility of the native preview (set via
    // `preview_set_bounds`); combined with the preview window's own
    // occlusion state each frame.
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
            while let Ok(PreviewCmd::Config {
                width,
                height,
                visible,
            }) = preview_rx.try_recv()
            {
                core.resize_preview_surface(width, height);
                preview_wanted = visible;
            }
            #[cfg(target_os = "macos")]
            if let Some(probe) = occlusion.as_ref() {
                core.set_occluded(!probe.visible());
            }
            {
                // §3.1 applies to the preview swapchain too: only present
                // while the React slot wants it AND AppKit reports the
                // preview window visible (not hidden/minimized/covered).
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

/// Bus subscription → `engine:telemetry` Tauri events + sticky snapshots +
/// the 1 Hz `engine:status` heartbeat. Exits when the stop flag is set.
fn telemetry_loop(inner: Arc<EngineInner>, app: AppHandle, bus: render_core::telemetry::Bus) {
    let channels: std::collections::HashSet<String> =
        ALL_CHANNELS.iter().map(|s| s.to_string()).collect();
    let (sub_id, rx) = bus.subscribe(channels);
    let mut last_status_emit = Instant::now() - Duration::from_secs(1);

    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(frame) => {
                if matches!(
                    frame.channel.as_str(),
                    "hot_reload" | "audio_freshness" | "connectivity" | "fps" | "masters"
                        | "deck" | "changes"
                ) {
                    inner
                        .last_payloads
                        .lock()
                        .expect("last payloads lock")
                        .insert(frame.channel.clone(), frame.payload.clone());
                }
                let _ = app.emit(
                    "engine:telemetry",
                    TelemetryFrameOut {
                        channel: frame.channel,
                        payload: frame.payload,
                    },
                );
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
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

/// Create the engine output window. Default: a decorated window at pack
/// resolution (dev parity with the old `--windowed` spawn). With
/// `WZRD_DISPLAY=<idx>`: borderless, filling that monitor — the projector
/// deployment shape.
fn build_engine_window(app: &AppHandle, pack_w: u32, pack_h: u32) -> Result<tauri::Window> {
    let display: Option<usize> = std::env::var("WZRD_DISPLAY")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut builder = tauri::window::WindowBuilder::new(app, "engine")
        .title("WZRD Engine Output")
        .resizable(true);
    if display.is_some() {
        builder = builder.decorations(false);
    }
    let window = builder.build().context("building engine window")?;

    match display {
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
    Ok(window)
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
