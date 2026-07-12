//! Host-agnostic engine core (app-collapse Step 1).
//!
//! `Core` owns everything the engine *is* — GPU context, pass plan, driver
//! bus, OSC sink, effect registry, file watcher, telemetry bus, WS server —
//! and knows nothing about which windowing crate created the surface it
//! draws to. It receives a `wgpu::SurfaceTarget` (any raw-window-handle
//! carrier) at `init_gpu` time, so the same struct can be driven by:
//!
//! - `WinitHost` (`app.rs`) — the standalone `render-core` binary.
//! - A future TauriHost — the single-process collapse plan in
//!   `docs/TODO/single-process-collapse.md` (Step 2).
//!
//! State mutation stays single-writer (the render thread / whichever thread
//! the host drives Core from); inbound IPC commands are drained at frame
//! boundary in `poll_inbound`, mirroring the file-watcher path.
//!
//! Division of labour with the host:
//! - Host owns the window: creation, `request_redraw`, reading the inner
//!   size after a `SurfaceError::Lost`, deciding when to exit.
//! - Core owns everything behind the surface, plus the render policies both
//!   hosts must share identically: the occlusion invariant (§3.1 — never
//!   touch the swapchain of a possibly-occluded window) and frame pacing.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::compositor::{resolve_effect_def, PassPlan};
use crate::drivers::{Masters, ParamOverrides, SliderBank, Transport};
use crate::effects::{EffectKind, EffectRegistry, InputSlot};
use crate::gpu::{GpuContext, Leg, LayerIdentity, LayerParamsGpu};
use crate::osc::{try_spawn, AudioFeatures, OscListener};
use crate::pack::LoadedPack;
use crate::probe::{
    self, Band, ProbeItemSpec, ProbeSession, ProbeThresholds, PROBE_NULL_KEY, PROBE_NULL_WGSL,
};
use crate::rpc::{self, parking_lot_lite::SwapValue, EngineCommand, PackInfo, RpcContext};
use crate::scene::{resolve_selector, SceneFile};
use crate::session::{self, SessionFile};
use crate::telemetry::{
    AudioSnapshot, Bus, Connectivity, ConnectivityCell, DeckSnapshot, DriverSnapshot,
    FpsAccumulator, FrameCounts, HotReloadEvent, MastersState, PreviewSampler, ProbeReport,
};
use crate::watch::{ChangeKind, SceneWatcher};
use crate::Cli;
use crate::ws;

/// §5.6 — when a bar-quantized promote starts its fade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantize {
    /// Fade starts on the next transport bar boundary (default) so the
    /// visual change lands on a downbeat.
    Bar,
    /// Fade starts immediately.
    Now,
}

/// §5.6 promote state machine. Re-entrancy contract (part of the RPC
/// surface, not UI polish — the auto-pilot playlist hits these paths
/// programmatically): a *pending* quantized promote is replaced by a newer
/// `promote`; while `Ramping`, both `promote` and `pull` are rejected.
enum PromotePhase {
    Idle,
    /// Waiting for the bar boundary after `armed_bar`.
    Pending {
        fade_ms: f32,
        armed_bar: u64,
    },
    /// Fade in flight — `mix` ramps 0→1 over `fade_ms` of wall time.
    Ramping {
        start: Instant,
        fade_ms: f32,
    },
}

/// §5.6 design leg — the AI/operator scratchpad. `None` on single-leg
/// (headless) runs, where the engine collapses to exactly its pre-two-deck
/// behaviour.
struct DesignLeg {
    scene: SceneFile,
    plan: Option<PassPlan>,
    /// The leg's scene JSON — also what the design autosave persists.
    raw: String,
    /// Set on every applied design edit; the autosave debounces on it.
    autosave_dirty_at: Option<Instant>,
}

/// A design-scene apply that is waiting on its probe verdict. Carries the
/// parsed scene + raw JSON forward so the swap happens only after the probe
/// clears, plus the (optional) RPC reply channel to unblock.
struct PendingDesignApply {
    scene: SceneFile,
    raw: String,
    target_label: String,
    reply: Option<crossbeam_channel::Sender<Result<serde_json::Value, String>>>,
    started: Instant,
    session: ProbeSession,
}

pub struct Core {
    pack: LoadedPack,
    /// The **live** leg's scene — what the projector plays (§5.6).
    scene: SceneFile,
    scene_path: PathBuf,

    /// §5.6 two-deck mode: on whenever a control surface exists
    /// (`--ws-addr` set — both the WS server and the in-process Tauri host
    /// use it). Headless-only runs collapse to a single live leg exactly as
    /// pre-two-deck: watcher binds live, no design composite is allocated.
    two_leg: bool,
    /// The design leg (scratchpad). `Some` after `init_gpu` in two-leg mode.
    design: Option<DesignLeg>,
    /// Live-leg scene JSON, served by `scene.getState { leg: "live" }`.
    live_scene_state: Arc<SwapValue>,

    /// §5.6 promote machinery — see [`PromotePhase`].
    promote: PromotePhase,
    /// Live→design crossfade position (0..1) fed into the final pass.
    mix: f32,
    /// Which composite the native preview samples (LIVE ⇄ DESIGN toggle).
    preview_source: Leg,
    /// Throttle for the ~10 Hz deck trickle while a fade ramps.
    last_deck_emit: Instant,

    /// §5.6 pre-flight probe: operator thresholds (shared inline with the
    /// WS thread), the calibrated fixed-floor measurement, and the pending
    /// apply whose probe is in flight (at most one — a newer apply
    /// supersedes it).
    probe_thresholds: Arc<ProbeThresholds>,
    probe_overhead_ms: Option<f32>,
    pending_apply: Option<PendingDesignApply>,
    effects_dir: Option<PathBuf>,
    registry: EffectRegistry,
    transport: Transport,
    audio_state: Arc<AudioFeatures>,

    /// Kept alive for the lifetime of the engine — the OSC recv thread is
    /// detached and pumps `audio_state` from the audio feature server.
    _osc_listener: Option<OscListener>,

    /// Telemetry bus shared with the WS server (if any). Always present;
    /// when no subscribers are attached, emission is a noop write to a
    /// shared registry.
    bus: Bus,

    /// Shared latest-good-scene JSON snapshot. Read by `scene.getState` over
    /// IPC; written by the engine on every successful load/reload. Held as
    /// `Arc` so the WS server reads from the same value without grabbing a
    /// mutex on the Core.
    scene_state: Arc<SwapValue>,

    /// Inbound command channel from control-surface consumers (WS server
    /// and/or an embedding host). Drained at frame boundary. Always present;
    /// headless without `--ws-addr` it simply never receives anything.
    cmd_rx: crossbeam_channel::Receiver<EngineCommand>,

    /// Sender half of `cmd_rx`, cloned out to consumers via
    /// [`Core::control_channel`]. Core keeps one so the channel can be
    /// handed out after construction; command-channel-closed detection
    /// relies on the *receiver* (dropped with Core) going away.
    cmd_tx: crossbeam_channel::Sender<EngineCommand>,

    /// The same read-only/inline-dispatch context the WS server uses,
    /// exposed to embedding hosts via [`Core::control_channel`].
    rpc_ctx: RpcContext,

    /// Kept to keep the WS accept thread alive. `Some` only with
    /// `--ws-addr` — also the connectivity pill's "ws" cell.
    ws_handle: Option<ws::ServerHandle>,

    // Lazily initialised once the host hands over a surface target
    // (`init_gpu`) — for winit that's when `resumed` fires (0.30 contract).
    gpu: Option<GpuContext>,
    plan: Option<PassPlan>,
    watcher: Option<SceneWatcher>,

    fps: FpsAccumulator,
    preview: PreviewSampler,
    audio_was_fresh: Option<bool>,
    last_audio_pill: Instant,

    /// **Live-leg** `ui.slider` values, shared with the WS server
    /// (`param.set {leg:"live"}`).
    sliders: Arc<SliderBank>,

    /// §5.4 **live-leg** masters — operator-owned globals shared with the WS
    /// server (`master.set`). Never reachable from scene.json.
    masters: Arc<Masters>,

    /// §5.5 **live-leg** per-binding scalar overrides, shared with the WS
    /// server (`param.set {binding, param, value}`).
    overrides: Arc<ParamOverrides>,

    /// §5.6 full-control-switch: the design leg's own control state — the
    /// deck toggle switches the *entire* control surface between legs, so
    /// tuning design (speed, brightness, knobs…) never touches the show.
    /// Promote copies design→live (what you previewed is what goes live);
    /// pull copies live→design. On single-leg runs these alias the live
    /// Arcs and `design_transport` is simply never stepped.
    design_sliders: Arc<SliderBank>,
    design_masters: Arc<Masters>,
    design_overrides: Arc<ParamOverrides>,
    design_transport: Transport,

    /// §5.3 session sidecar path (`session.json` next to the scene) + the
    /// calibration loaded from it. `session_calibration` takes precedence
    /// over the deprecated scene.json field and is the only calibration the
    /// engine ever writes back.
    session_path: PathBuf,
    session_calibration: Option<[[f32; 3]; 3]>,
    scene_calib_warned: bool,

    /// Epoch-ms stamp of the last operator-state change (0 = clean). Written
    /// by the WS thread on master/knob changes; `poll_inbound` debounces the
    /// sidecar write on it.
    session_dirty: Arc<AtomicU64>,

    /// Flipped by SIGTERM/SIGINT — `poll_inbound` snapshots the session and
    /// requests a host exit (§5.11 power-blink snapshot).
    term_flag: Arc<AtomicBool>,
    exit_requested: bool,

    /// True while the OS reports the projector window fully occluded. On
    /// macOS an occluded window's swapchain throttles to ~1 Hz and
    /// `get_current_texture()` blocks the render thread for up to a second
    /// per frame — so while occluded the host must render via
    /// `render_offscreen_frame` (preview + telemetry stay live) and never
    /// call `redraw`.
    occluded: bool,

    /// App-collapse Step 3: present the native preview surface this frame.
    /// Host-owned (React slot visibility ∧ preview-window occlusion) — the
    /// same §3.1 rule applies to the preview swapchain, so this must be
    /// false whenever the preview window might be occluded.
    preview_visible: bool,

    // Telemetry emit pacing.
    last_drivers_emit: Instant,
    last_audio_emit: Instant,

    /// Frame-pacing state. `frame_budget` is the minimum time we want between
    /// consecutive frames (derived from `--frame-cap-hz`);
    /// `last_redraw_request` is when we last released one. `None` budget
    /// disables the cap (`--frame-cap-hz 0`).
    frame_budget: Option<Duration>,
    last_redraw_request: Option<Instant>,
}

impl Core {
    /// Everything that can be built before a window exists: pack + scene
    /// load, effect registry, OSC listener, telemetry bus, WS server.
    /// GPU resources arrive later via [`Core::init_gpu`].
    pub fn new(cli: &Cli) -> Result<Self> {
        let scene_path = cli.scene.canonicalize().with_context(|| {
            format!("resolving --scene path {}", cli.scene.display())
        })?;
        let scene = SceneFile::load(&scene_path)?;
        let pack_dir_raw = match &cli.pack {
            Some(p) => p.clone(),
            None => scene.pack_dir(&scene_path),
        };
        let pack_dir = pack_dir_raw
            .canonicalize()
            .with_context(|| format!("resolving pack dir {}", pack_dir_raw.display()))?;
        let pack = LoadedPack::load(&pack_dir)?;
        log::info!(
            "loaded pack {} ({} layers @ {}x{})",
            pack_dir.display(),
            pack.layer_count,
            pack.atlas_width,
            pack.atlas_height,
        );

        let effects_dir = match &cli.effects {
            Some(p) => Some(p.clone()),
            None => {
                let default = scene_path
                    .parent()
                    .map(|p| p.join("effects"))
                    .unwrap_or_else(|| PathBuf::from("effects"));
                if default.exists() {
                    Some(default)
                } else {
                    None
                }
            }
        };
        if let Some(d) = &effects_dir {
            log::info!("watching effects dir {}", d.display());
        }
        let registry = EffectRegistry::new(effects_dir.clone());

        let transport = Transport::new(scene.transport.bpm);
        let audio_state = AudioFeatures::new();
        let osc_listener = if cli.no_osc {
            log::info!("OSC ingest disabled (--no-osc)");
            None
        } else {
            try_spawn(Arc::clone(&audio_state), cli.osc_addr)
        };

        let bus = Bus::new();
        crate::telemetry::set_global_bus(&bus);
        let scene_state = Arc::new(SwapValue::new());
        let live_scene_state = Arc::new(SwapValue::new());
        // Seed both legs' state with what's on disk so `scene.getState`
        // returns something even before the first hot-reload. At boot the
        // legs are identical by construction.
        if let Ok(raw) = std::fs::read_to_string(&scene_path) {
            scene_state.set(raw.clone());
            live_scene_state.set(raw);
        }

        // §5.6: a control surface (WS server — set by the Tauri host too)
        // means the two-deck topology; headless-only runs stay single-leg.
        let two_leg = cli.ws_addr.is_some();

        let sliders = SliderBank::new();
        let masters = Masters::new();
        let overrides = ParamOverrides::new();
        // §5.6 full-control-switch: the design leg owns its control state.
        // Single-leg aliases the live Arcs so the `leg` RPC param is a no-op.
        let (design_sliders, design_masters, design_overrides) = if two_leg {
            (SliderBank::new(), Masters::new(), ParamOverrides::new())
        } else {
            (
                Arc::clone(&sliders),
                Arc::clone(&masters),
                Arc::clone(&overrides),
            )
        };
        let design_transport = Transport::new(scene.transport.bpm);
        let probe_thresholds = ProbeThresholds::new();
        let session_dirty = Arc::new(AtomicU64::new(0));

        // §5.3 — restore operator state from the session sidecar before the
        // WS surface (or the first frame) can observe defaults.
        let session_path = session::session_path(&scene_path);
        let mut session_calibration = None;
        match session::load(&session_path) {
            Ok(Some(s)) => {
                session_calibration = s.projector_calibration;
                if let Some(m) = &s.masters {
                    masters.restore(m);
                }
                for (name, value) in &s.params {
                    sliders.set(name, *value);
                }
                for (binding, params) in &s.overrides {
                    for (param, value) in params {
                        overrides.set(binding, param, *value);
                    }
                }
                if let Some(pt) = &s.probe_thresholds {
                    probe_thresholds.restore(pt);
                }
                // §5.6 — both legs boot with the same operator state (the
                // sidecar persists the live/show truth); they diverge only
                // through leg-targeted writes.
                if two_leg {
                    design_masters.copy_from(&masters);
                    design_sliders.copy_from(&sliders);
                    design_overrides.copy_from(&overrides);
                }
                log::info!(
                    "restored session sidecar {} ({} knobs, {} overrides{})",
                    session_path.display(),
                    s.params.len(),
                    s.overrides.values().map(|m| m.len()).sum::<usize>(),
                    if session_calibration.is_some() {
                        ", calibration"
                    } else {
                        ""
                    }
                );
            }
            Ok(None) => {}
            Err(err) => log::warn!(
                "ignoring session sidecar {}: {err:#}",
                session_path.display()
            ),
        }
        // Seed the sticky masters channel so late subscribers see the
        // restored values, not defaults.
        bus.emit_masters(MastersState {
            live: masters.snapshot(),
            design: design_masters.snapshot(),
        });

        // §5.11 — a termination signal snapshots the session before exit, so
        // a power-blink/systemd-stop comes back close to where it was.
        let term_flag = Arc::new(AtomicBool::new(false));
        #[cfg(unix)]
        {
            for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
                if let Err(err) = signal_hook::flag::register(sig, Arc::clone(&term_flag)) {
                    log::warn!("could not register signal {sig}: {err}");
                }
            }
        }

        // Command channel + inline-dispatch context always exist so an
        // embedding host (app-collapse Step 2) can attach via
        // `control_channel` even without a WS server. Headless with no
        // consumers, the channel just stays empty.
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let rpc_ctx = RpcContext {
            pack: Arc::new(PackInfo::from_pack(&pack)),
            scene_state: Arc::clone(&scene_state),
            live_scene_state: Arc::clone(&live_scene_state),
            effects_dir: effects_dir.clone(),
            bus: bus.clone(),
            sliders: Arc::clone(&sliders),
            masters: Arc::clone(&masters),
            overrides: Arc::clone(&overrides),
            design_sliders: Arc::clone(&design_sliders),
            design_masters: Arc::clone(&design_masters),
            design_overrides: Arc::clone(&design_overrides),
            session_dirty: Arc::clone(&session_dirty),
            probe_thresholds: Arc::clone(&probe_thresholds),
        };
        let ws_handle = match cli.ws_addr {
            Some(addr) => Some(
                ws::serve(addr, cmd_tx.clone(), rpc_ctx.clone())
                    .context("starting WS server")?,
            ),
            None => None,
        };

        let frame_budget = if cli.frame_cap_hz == 0 {
            None
        } else {
            Some(Duration::from_secs_f64(1.0 / cli.frame_cap_hz as f64))
        };
        if let Some(b) = frame_budget {
            log::info!(
                "frame cap: {} Hz (~{:.2} ms/frame)",
                cli.frame_cap_hz,
                b.as_secs_f64() * 1000.0
            );
        } else {
            log::info!("frame cap disabled (--frame-cap-hz 0)");
        }

        if two_leg {
            log::info!("two-deck mode: authoring targets the design leg; `promote` goes live");
        }

        Ok(Self {
            pack,
            scene,
            scene_path,
            two_leg,
            design: None,
            live_scene_state,
            promote: PromotePhase::Idle,
            mix: 0.0,
            preview_source: Leg::Design,
            last_deck_emit: Instant::now(),
            probe_thresholds,
            probe_overhead_ms: None,
            pending_apply: None,
            effects_dir,
            registry,
            transport,
            audio_state,
            _osc_listener: osc_listener,
            bus,
            scene_state,
            cmd_rx,
            cmd_tx,
            rpc_ctx,
            ws_handle,
            gpu: None,
            plan: None,
            watcher: None,
            fps: FpsAccumulator::new(),
            preview: PreviewSampler::new(),
            audio_was_fresh: None,
            last_audio_pill: Instant::now(),
            sliders,
            masters,
            overrides,
            design_sliders,
            design_masters,
            design_overrides,
            design_transport,
            session_path,
            session_calibration,
            scene_calib_warned: false,
            session_dirty,
            term_flag,
            exit_requested: false,
            occluded: false,
            preview_visible: false,
            last_drivers_emit: Instant::now(),
            last_audio_emit: Instant::now(),
            frame_budget,
            last_redraw_request: None,
        })
    }

    /// Bring up the GPU against any surface target — a winit `Arc<Window>`,
    /// a tao window's raw handle, anything `Into<wgpu::SurfaceTarget>`.
    /// `width`/`height` are the target's current inner size in physical
    /// pixels (Core can't query a raw handle for its size).
    ///
    /// Idempotent: a second call is a no-op, mirroring the old
    /// `resumed`-fires-twice guard.
    pub fn init_gpu(
        &mut self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let gpu = pollster::block_on(GpuContext::new(
            target,
            width,
            height,
            &self.pack,
            self.two_leg,
        ))
        .context("initialising wgpu")?;
        self.gpu = Some(gpu);
        // The atlas now lives on the GPU; drop the CPU-side copy.
        self.pack.mask_atlas = Vec::new();
        self.pack.mask_atlas.shrink_to_fit();
        let _ = self.rebuild_plan();

        // §5.6 — the design leg boots as a copy of live (same scene, its own
        // plan + composite), then a crash-saved draft is offered back into
        // it (probe-gated like any other design apply).
        if self.two_leg {
            let raw = self
                .scene_state
                .get()
                .unwrap_or_else(|| String::from("{}"));
            let plan = {
                let Self {
                    gpu,
                    pack,
                    registry,
                    scene,
                    ..
                } = self;
                gpu.as_mut()
                    .and_then(|g| match PassPlan::build(g, pack, scene, registry) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            log::error!("design leg initial build failed: {e:#}");
                            None
                        }
                    })
            };
            self.design = Some(DesignLeg {
                scene: self.scene.clone(),
                plan,
                raw: raw.clone(),
                autosave_dirty_at: None,
            });
            // Design clock starts in lockstep with live (§5.6 — they
            // diverge only when the operator bends design speed/tempo).
            self.design_transport.sync_from(&self.transport);
            self.emit_deck();

            // §5.6 design-leg autosave restore: a crash mid-design must not
            // eat the draft.
            let autosave = design_autosave_path(&self.scene_path);
            match std::fs::read_to_string(&autosave) {
                Ok(draft) if draft != raw => {
                    log::info!(
                        "restoring design draft from {} (differs from scene.json)",
                        autosave.display()
                    );
                    self.apply_design_scene(draft, "design-autosave", None);
                }
                _ => {}
            }
        }

        match SceneWatcher::new(&self.scene_path, self.effects_dir.as_deref()) {
            Ok(w) => self.watcher = Some(w),
            Err(err) => log::warn!("hot-reload disabled: {err:#}"),
        }
        Ok(())
    }

    /// Build the **live** plan from the live scene. Used at boot and on
    /// single-leg (headless) reloads; two-leg mutations go through
    /// [`Core::apply_design_scene`] instead.
    fn rebuild_plan(&mut self) -> Result<()> {
        let Some(gpu) = self.gpu.as_mut() else {
            return Ok(());
        };
        let started = Instant::now();
        match PassPlan::build(gpu, &self.pack, &self.scene, &self.registry) {
            Ok(plan) => {
                log::info!("scene plan built ({} layer passes)", plan.layer_passes.len());
                self.plan = Some(plan);
                self.gc_pipelines();
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: "scene".into(),
                    ok: true,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    message: None,
                    probe: None,
                });
                self.transport.set_bpm(self.scene.transport.bpm);
                // §5.3 — calibration now lives in session.json; the scene
                // field is a deprecated read-only fallback, never written.
                if self.scene.projector_calibration.is_some()
                    && self.session_calibration.is_none()
                    && !self.scene_calib_warned
                {
                    log::warn!(
                        "scene.json projectorCalibration is deprecated — calibration \
                         belongs in session.json (engine-written); the scene value is \
                         honoured as a fallback but will never be written back"
                    );
                    self.scene_calib_warned = true;
                }
                Ok(())
            }
            Err(err) => {
                log::error!("rejecting scene update: {err:#}");
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: "scene".into(),
                    ok: false,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    message: Some(format!("{err:#}")),
                    probe: None,
                });
                Err(err)
            }
        }
    }

    fn reload_scene(&mut self) {
        match std::fs::read_to_string(&self.scene_path) {
            // The UI saves via `scene.load` IPC *and* writes to disk; the
            // file watcher then fires on our own write. Skip the redundant
            // rebuild when the on-disk content matches the authoring leg
            // (design in two-leg mode — §5.6 blanket leg rule).
            Ok(raw) if self.scene_state.get().as_deref() == Some(raw.as_str()) => {
                log::debug!("scene reload skipped (content unchanged)");
            }
            Ok(raw) if self.two_leg => {
                // §5.6 blanket leg rule: the watcher authors the design leg.
                log::info!("hot-reloading {} into the design leg", self.scene_path.display());
                self.apply_design_scene(raw, "scene", None);
            }
            Ok(raw) => match SceneFile::parse(&raw) {
                Ok(scene) => {
                    log::info!("hot-reloaded {}", self.scene_path.display());
                    self.scene = scene;
                    if self.rebuild_plan().is_ok() {
                        self.scene_state.set(raw.clone());
                        self.live_scene_state.set(raw);
                    }
                }
                Err(err) => {
                    log::error!(
                        "ignoring scene reload (parse failed): {err:#}; previous plan remains active"
                    );
                    self.bus.emit_hot_reload(HotReloadEvent {
                        target: "scene".into(),
                        ok: false,
                        elapsed_ms: 0.0,
                        message: Some(format!("{err:#}")),
                        probe: None,
                    });
                }
            },
            Err(err) => {
                log::error!("ignoring scene reload (read failed): {err:#}");
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: "scene".into(),
                    ok: false,
                    elapsed_ms: 0.0,
                    message: Some(format!("{err:#}")),
                    probe: None,
                });
            }
        }
    }

    fn reload_effects(&mut self) {
        let changed = self.registry.rescan_disk();
        if changed.is_empty() {
            return;
        }
        // Content-derived pipeline keys (§5.6): an edited effect resolves to
        // a *new* key, so nothing is evicted here — the live leg keeps
        // drawing its old pipeline and the GC after the next successful swap
        // collects orphans. (The pre-two-deck remove-by-key here was the
        // silent-skip-on-cache-miss failure mode the roadmap closes.)
        log::info!("effects changed on disk: {:?}", changed);
        if self.two_leg {
            let raw = self
                .design
                .as_ref()
                .map(|d| d.raw.clone())
                .or_else(|| self.scene_state.get());
            if let Some(raw) = raw {
                self.apply_design_scene(raw, "effects", None);
            }
        } else {
            let _ = self.rebuild_plan();
        }
    }

    /// Apply a scene from raw JSON straight onto the **live** leg. This is
    /// the single-leg (headless) path; two-leg authoring goes through
    /// [`Core::apply_design_scene`] + `promote`.
    pub fn apply_scene_json(&mut self, raw: &str) -> Result<()> {
        let scene = SceneFile::parse(raw)?;
        self.scene = scene;
        self.rebuild_plan()?;
        self.scene_state.set(raw.to_string());
        self.live_scene_state.set(raw.to_string());
        Ok(())
    }

    /// `scene.load` / `scene.reload` entry point (render thread). Two-leg:
    /// route at the design leg, deferring the reply through the probe when
    /// new pipelines are involved. Single-leg: the pre-two-deck live apply.
    pub fn scene_load_rpc(
        &mut self,
        raw: String,
        reply: crossbeam_channel::Sender<Result<serde_json::Value, String>>,
    ) {
        if self.two_leg {
            self.apply_design_scene(raw, "scene", Some(reply));
        } else {
            let res = self
                .apply_scene_json(&raw)
                .map(|_| serde_json::json!({ "ok": true }))
                .map_err(|e| format!("{e:#}"));
            let _ = reply.send(res);
        }
    }

    /// §5.6 — apply a scene to the **design** leg, gating any pipelines new
    /// to the cache behind the pre-flight probe (whatever the entry path:
    /// `scene.load`, watcher reloads, the autosave restore). On red the
    /// previous design plan stays; on green/yellow the plan swaps in
    /// ([`Core::finish_design_apply`]).
    fn apply_design_scene(
        &mut self,
        raw: String,
        target_label: &str,
        reply: Option<crossbeam_channel::Sender<Result<serde_json::Value, String>>>,
    ) {
        let started = Instant::now();
        let fail = |bus: &Bus, msg: String, reply: Option<crossbeam_channel::Sender<Result<serde_json::Value, String>>>| {
            log::error!("rejecting design update ({target_label}): {msg}");
            bus.emit_hot_reload(HotReloadEvent {
                target: target_label.into(),
                ok: false,
                elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                message: Some(msg.clone()),
                probe: None,
            });
            if let Some(r) = reply {
                let _ = r.send(Err(msg));
            }
        };

        if self.gpu.is_none() || self.design.is_none() {
            fail(&self.bus, "engine not ready (no GPU/design leg yet)".into(), reply);
            return;
        }
        let scene = match SceneFile::parse(&raw) {
            Ok(s) => s,
            Err(e) => {
                fail(&self.bus, format!("{e:#}"), reply);
                return;
            }
        };

        // Pre-analysis: which pipelines would this scene pull in that the
        // cache hasn't seen? Those are the ones the probe must clear first.
        let specs = match self.analyze_new_pipelines(&scene) {
            Ok(s) => s,
            Err(e) => {
                fail(&self.bus, format!("{e:#}"), reply);
                return;
            }
        };
        if specs.is_empty() {
            self.finish_design_apply(scene, raw, target_label, reply, None, started);
            return;
        }

        // Compile the new pipelines (naga-validated), then probe them.
        {
            let gpu = self.gpu.as_mut().expect("checked above");
            for (key, wgsl, _) in &specs {
                if let Err(e) = gpu.upsert_user_pipeline(key, wgsl) {
                    let msg = format!("{e:#}");
                    // Drop any partially-compiled newcomers.
                    let bus = self.bus.clone();
                    self.gc_pipelines();
                    fail(&bus, msg, reply);
                    return;
                }
            }
            let calibrate = self.probe_overhead_ms.is_none();
            if calibrate && !gpu.pipeline_cache.contains_key(PROBE_NULL_KEY) {
                if let Err(e) = gpu.upsert_user_pipeline(PROBE_NULL_KEY, PROBE_NULL_WGSL) {
                    log::warn!("probe calibration shader failed to compile: {e:#}");
                }
            }
        }

        // A newer apply supersedes a probe already in flight (latest wins —
        // matches the pending-promote replacement rule).
        if let Some(old) = self.pending_apply.take() {
            log::info!("design apply superseded mid-probe ({})", old.target_label);
            if let Some(r) = old.reply {
                let _ = r.send(Err("superseded by a newer design apply".into()));
            }
        }

        let gpu = self.gpu.as_ref().expect("checked above");
        let calibrate =
            self.probe_overhead_ms.is_none() && gpu.pipeline_cache.contains_key(PROBE_NULL_KEY);
        let items: Vec<ProbeItemSpec> = specs
            .into_iter()
            .map(|(key, _, item)| ProbeItemSpec {
                key: key.clone(),
                label: item.0,
                layer_params: item.1,
            })
            .collect();
        log::info!(
            "probing {} new pipeline(s) before they enter the design leg{}",
            items.len(),
            if calibrate { " (with overhead calibration)" } else { "" }
        );
        let session = ProbeSession::new(
            gpu,
            gpu.composite_width,
            gpu.composite_height,
            items,
            calibrate,
        );
        self.pending_apply = Some(PendingDesignApply {
            scene,
            raw,
            target_label: target_label.to_string(),
            reply,
            started,
            session,
        });
    }

    /// Which user pipelines would `scene` need that the cache doesn't hold,
    /// plus a pessimistic probe payload for each (§5.6 probe amendments:
    /// audio pinned to 1.0, scalars at descriptor max where declared).
    #[allow(clippy::type_complexity)]
    fn analyze_new_pipelines(
        &self,
        scene: &SceneFile,
    ) -> Result<Vec<(String, String, (String, LayerParamsGpu))>> {
        let Some(gpu) = self.gpu.as_ref() else {
            return Ok(Vec::new());
        };
        let mut out: Vec<(String, String, (String, LayerParamsGpu))> = Vec::new();
        for binding in &scene.bindings {
            let def = resolve_effect_def(binding, &self.registry)?;
            let EffectKind::User { pipeline_key, wgsl, .. } = &def.kind else {
                continue;
            };
            if gpu.pipeline_cache.contains_key(pipeline_key)
                || out.iter().any(|(k, _, _)| k == pipeline_key)
            {
                continue;
            }
            let key = pipeline_key.clone();
            let source = wgsl.clone();
            let label = def.name.clone();
            let resolved =
                crate::effects::EffectBinding::from_params(def.clone(), &binding.params)
                    .map_err(|e| anyhow::anyhow!("binding {:?}: {e:#}", binding.id))?;

            // Pessimistic scalars, in declaration order (parallel to the
            // shader's params_f slots).
            let scalar_metas: Vec<Option<f32>> = def
                .inputs
                .iter()
                .filter_map(|i| match i {
                    InputSlot::Scalar { meta, .. } => Some(meta.max),
                    InputSlot::Color { .. } => None,
                })
                .collect();
            let scalar_names: Vec<&str> = def
                .inputs
                .iter()
                .filter_map(|i| match i {
                    InputSlot::Scalar { name, .. } => Some(name.as_str()),
                    InputSlot::Color { .. } => None,
                })
                .collect();
            let scalars: Vec<f32> = resolved
                .scalars
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    probe::pessimistic_scalar(
                        s,
                        scalar_metas.get(i).copied().flatten(),
                        scalar_names
                            .get(i)
                            .and_then(|n| self.design_overrides.get(&binding.id, n)),
                        // Probing gates the *design* leg — its knobs apply.
                        &self.design_sliders,
                    )
                })
                .collect();

            // Probe with the binding's first resolved layer — cost scales
            // with pixel count, not with which mask gates the fragments.
            let slices = resolve_selector(&binding.select, &self.pack)?;
            let slice = slices.first().copied().unwrap_or(0);
            let geom = self
                .pack
                .geoms
                .get(slice as usize)
                .copied()
                .unwrap_or(crate::pack::LayerGeom {
                    centroid_uv: [0.5, 0.5],
                    bbox_uv: [0.0, 0.0, 1.0, 1.0],
                });
            let identity = LayerIdentity {
                layer_seed: 0.5,
                layer_index: 0,
                layer_count: slices.len().max(1) as u32,
                centroid_uv: geom.centroid_uv,
                bbox_uv: geom.bbox_uv,
            };
            let params = LayerParamsGpu::build(slice, 0, &identity, &scalars, &resolved.colors);
            out.push((key, source, (label, params)));
        }
        Ok(out)
    }

    /// Drive the in-flight probe a few ms forward (called once per frame
    /// from [`Core::poll_inbound`], sequencing probe frames between live
    /// frames). On completion: red refuses the apply, green/yellow swaps
    /// the plan into design with the verdict attached to `hot_reload`.
    fn step_probe(&mut self) {
        let Some(mut pending) = self.pending_apply.take() else {
            return;
        };
        let Some(gpu) = self.gpu.as_ref() else {
            self.pending_apply = Some(pending);
            return;
        };
        if !pending.session.step(gpu) {
            self.pending_apply = Some(pending);
            return;
        }

        let result = pending.session.finalize(
            self.probe_overhead_ms,
            self.probe_thresholds.a_ms(),
            self.probe_thresholds.b_ms(),
        );
        if let Some(overhead) = result.measured_overhead_ms {
            log::info!("probe overhead calibrated: {:.3} ms fixed floor per frame", overhead);
            self.probe_overhead_ms = Some(overhead);
        }
        let worst = result
            .verdicts
            .iter()
            .max_by(|a, b| {
                a.predicted_p95_ms
                    .partial_cmp(&b.predicted_p95_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let report = ProbeReport {
            compiled: true,
            predicted_p95_ms: result.worst_predicted_ms,
            band: result.worst_band.as_str().to_string(),
            thumbnail_b64: worst.and_then(|v| v.thumbnail_b64.clone()),
            verdicts: result.verdicts.clone(),
        };

        if result.worst_band == Band::Red {
            let msg = format!(
                "probe refused entry: predicted full-res p95 {:.1} ms exceeds threshold B ({:.1} ms)",
                result.worst_predicted_ms,
                self.probe_thresholds.b_ms()
            );
            log::error!("{msg}");
            self.bus.emit_hot_reload(HotReloadEvent {
                target: pending.target_label.clone(),
                ok: false,
                elapsed_ms: pending.started.elapsed().as_secs_f32() * 1000.0,
                message: Some(msg.clone()),
                probe: Some(report),
            });
            if let Some(r) = pending.reply {
                let _ = r.send(Err(msg));
            }
            // Drop the refused pipelines (nothing references them).
            self.gc_pipelines();
            return;
        }

        if result.worst_band == Band::Yellow {
            log::warn!(
                "probe: heavy but doable ({:.1} ms predicted) — swapping into design, flagged",
                result.worst_predicted_ms
            );
        }
        self.finish_design_apply(
            pending.scene,
            pending.raw,
            &pending.target_label,
            pending.reply,
            Some(report),
            pending.started,
        );
    }

    /// Build + swap a probe-cleared (or pipeline-neutral) scene into the
    /// design leg. Live is untouched by construction.
    fn finish_design_apply(
        &mut self,
        scene: SceneFile,
        raw: String,
        target_label: &str,
        reply: Option<crossbeam_channel::Sender<Result<serde_json::Value, String>>>,
        probe_report: Option<ProbeReport>,
        started: Instant,
    ) {
        let build = {
            let Self {
                gpu,
                pack,
                registry,
                ..
            } = self;
            let gpu = gpu.as_mut().expect("apply_design_scene checked gpu");
            PassPlan::build(gpu, pack, &scene, registry)
        };
        match build {
            Ok(plan) => {
                log::info!(
                    "design plan built ({} layer passes)",
                    plan.layer_passes.len()
                );
                let design = self.design.as_mut().expect("two-leg checked");
                design.scene = scene;
                design.plan = Some(plan);
                design.raw = raw.clone();
                design.autosave_dirty_at = Some(Instant::now());
                // The design leg's clock follows the design scene's tempo
                // (§5.6 full-control-switch: live tempo is untouched until
                // promote).
                self.design_transport
                    .set_bpm(design.scene.transport.bpm);
                self.scene_state.set(raw);
                self.gc_pipelines();
                let probe_json = probe_report
                    .as_ref()
                    .map(|p| serde_json::to_value(p).expect("probe report"));
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: target_label.into(),
                    ok: true,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    message: None,
                    probe: probe_report,
                });
                if let Some(r) = reply {
                    let _ = r.send(Ok(
                        serde_json::json!({ "ok": true, "leg": "design", "probe": probe_json }),
                    ));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                log::error!("rejecting design update ({target_label}): {msg}");
                self.gc_pipelines();
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: target_label.into(),
                    ok: false,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    message: Some(msg.clone()),
                    probe: probe_report,
                });
                if let Some(r) = reply {
                    let _ = r.send(Err(msg));
                }
            }
        }
    }

    /// §5.6 cross-leg pipeline GC: retain only pipelines referenced by
    /// *either* leg (plus built-ins, the calibration shader, and anything an
    /// in-flight probe is testing). Replaces the old evict-by-key — which,
    /// with two plans, could silently stop live layers drawing mid-show.
    fn gc_pipelines(&mut self) {
        let Self {
            gpu,
            plan,
            design,
            pending_apply,
            ..
        } = self;
        let Some(gpu) = gpu.as_mut() else { return };
        let mut keep: std::collections::HashSet<String> = std::collections::HashSet::new();
        keep.insert(crate::gpu::BUILTIN_PIPELINE_KEY.to_string());
        keep.insert(PROBE_NULL_KEY.to_string());
        if let Some(p) = plan.as_ref() {
            keep.extend(p.pipeline_keys().map(String::from));
        }
        if let Some(d) = design.as_ref() {
            if let Some(p) = d.plan.as_ref() {
                keep.extend(p.pipeline_keys().map(String::from));
            }
        }
        if let Some(pa) = pending_apply.as_ref() {
            keep.extend(pa.session.keys().map(String::from));
        }
        let before = gpu.pipeline_cache.len();
        gpu.pipeline_cache.retain(|k, _| keep.contains(k));
        let dropped = before - gpu.pipeline_cache.len();
        if dropped > 0 {
            log::debug!("pipeline GC: dropped {dropped} unreferenced pipeline(s)");
        }
    }

    // ---------- §5.6 promote / pull / preview source ----------

    /// Current **live** transport bar time (4 beats/bar, matching
    /// `FrameContext`) — promote quantizes to the crowd's musical time.
    fn bar_time(&self) -> f32 {
        self.transport.elapsed_sec() * self.transport.bpm().max(1.0) / 60.0 / 4.0
    }

    /// `promote {fade_ms, quantize}` — crossfade the projector to the design
    /// composite, then adopt design's plan into the live slot (pointer swap
    /// on ramp completion; design rebuilds in the background).
    pub fn cmd_promote(&mut self, fade_ms: f32, quantize: Quantize) -> Result<serde_json::Value, String> {
        if !self.two_leg {
            return Err("promote requires two-deck mode (engine started with a control surface)".into());
        }
        if self.design.as_ref().and_then(|d| d.plan.as_ref()).is_none() {
            return Err("design leg has no built plan to promote".into());
        }
        if matches!(self.promote, PromotePhase::Ramping { .. }) {
            return Err("a promote fade is already ramping — wait for it to complete".into());
        }
        // A pending (bar-quantized) promote is *replaced* by a newer one.
        let state = match quantize {
            Quantize::Now => {
                self.promote = PromotePhase::Ramping {
                    start: Instant::now(),
                    fade_ms,
                };
                "ramping"
            }
            Quantize::Bar => {
                self.promote = PromotePhase::Pending {
                    fade_ms,
                    armed_bar: self.bar_time().floor().max(0.0) as u64,
                };
                "pending"
            }
        };
        log::info!(
            "promote accepted: fade {} ms, quantize {:?} → {state}",
            fade_ms,
            quantize
        );
        self.emit_deck();
        Ok(serde_json::json!({ "ok": true, "state": state, "fade_ms": fade_ms }))
    }

    /// `pull` — hard-copy live's scene back into design (the explicit
    /// reverse). Cancels a pending promote and any in-flight probe apply
    /// (pull is the newest operator intent).
    pub fn cmd_pull(&mut self) -> Result<serde_json::Value, String> {
        if !self.two_leg {
            return Err("pull requires two-deck mode".into());
        }
        if matches!(self.promote, PromotePhase::Ramping { .. }) {
            return Err("a promote fade is ramping — wait for it to complete".into());
        }
        self.promote = PromotePhase::Idle;
        if let Some(old) = self.pending_apply.take() {
            if let Some(r) = old.reply {
                let _ = r.send(Err("superseded by pull".into()));
            }
        }
        let raw = self
            .live_scene_state
            .get()
            .ok_or_else(|| "no live scene state".to_string())?;
        let live_scene = self.scene.clone();
        let plan = {
            let Self {
                gpu,
                pack,
                registry,
                ..
            } = self;
            let gpu = gpu.as_mut().ok_or_else(|| "engine not ready".to_string())?;
            PassPlan::build(gpu, pack, &live_scene, registry).map_err(|e| format!("{e:#}"))?
        };
        let design = self
            .design
            .as_mut()
            .ok_or_else(|| "design leg missing".to_string())?;
        design.scene = live_scene;
        design.plan = Some(plan);
        design.raw = raw.clone();
        design.autosave_dirty_at = Some(Instant::now());
        self.scene_state.set(raw);
        // §5.6 full-control-switch: pull copies the whole live control
        // state back too — design becomes an exact replica of the show.
        self.design_masters.copy_from(&self.masters);
        self.design_sliders.copy_from(&self.sliders);
        self.design_overrides.copy_from(&self.overrides);
        self.design_transport.sync_from(&self.transport);
        self.bus.emit_masters(MastersState {
            live: self.masters.snapshot(),
            design: self.design_masters.snapshot(),
        });
        self.gc_pipelines();
        log::info!("pull: design leg reset to the live scene (content + controls)");
        self.emit_deck();
        Ok(serde_json::json!({ "ok": true }))
    }

    /// `preview.setSource` — flip which composite the native preview blits.
    pub fn cmd_preview_source(&mut self, source: Leg) -> Result<serde_json::Value, String> {
        if !self.two_leg && source == Leg::Design {
            return Err("no design leg on a single-leg (headless) run".into());
        }
        self.preview_source = source;
        self.emit_deck();
        Ok(serde_json::json!({
            "ok": true,
            "source": match source { Leg::Live => "live", Leg::Design => "design" },
        }))
    }

    /// Advance the promote state machine (once per frame, after the
    /// transport steps). Bar-quantized fades start when the bar index
    /// crosses the armed boundary; ramp completion runs the pointer swap.
    fn update_promote(&mut self) {
        match self.promote {
            PromotePhase::Idle => {}
            PromotePhase::Pending { fade_ms, armed_bar } => {
                let bar = self.bar_time().floor().max(0.0) as u64;
                if bar != armed_bar {
                    log::info!("promote: bar boundary reached — fade starting ({fade_ms} ms)");
                    self.promote = PromotePhase::Ramping {
                        start: Instant::now(),
                        fade_ms,
                    };
                    self.mix = 0.0;
                    self.emit_deck();
                }
            }
            PromotePhase::Ramping { start, fade_ms } => {
                self.mix = if fade_ms <= 0.0 {
                    1.0
                } else {
                    (start.elapsed().as_secs_f32() * 1000.0 / fade_ms).min(1.0)
                };
                if self.mix >= 1.0 {
                    self.complete_promote();
                }
            }
        }
    }

    /// Ramp completion: live adopts design's already-built plan (zero
    /// rebuild, zero hitch on the projector leg — safe because picks are
    /// stateless and pipelines cache-hit), then design rebuilds from the
    /// same scene JSON on the leg nobody is projecting. Promote is
    /// semantically a **copy**: design keeps its content, so "promote, push
    /// further, promote again" needs no manual pull between rounds.
    fn complete_promote(&mut self) {
        let Some(design) = self.design.as_mut() else {
            self.promote = PromotePhase::Idle;
            self.mix = 0.0;
            return;
        };
        let Some(design_plan) = design.plan.take() else {
            log::error!("promote completion with no design plan — aborting");
            self.promote = PromotePhase::Idle;
            self.mix = 0.0;
            self.emit_deck();
            return;
        };
        // Pointer swap: the projector leg adopts the built plan.
        self.plan = Some(design_plan);
        self.scene = design.scene.clone();
        let raw = design.raw.clone();
        self.live_scene_state.set(raw);
        // §5.6 full-control-switch: live adopts design's *entire* control
        // state — masters (incl. speed), knobs, overrides, and the design
        // clock itself — so the promoted content continues exactly as it
        // looked in the design preview. Design keeps its copies (promote is
        // a copy, not an exchange).
        self.masters.copy_from(&self.design_masters);
        self.sliders.copy_from(&self.design_sliders);
        self.overrides.copy_from(&self.design_overrides);
        self.transport.sync_from(&self.design_transport);
        session::touch(&self.session_dirty);
        self.bus.emit_masters(MastersState {
            live: self.masters.snapshot(),
            design: self.design_masters.snapshot(),
        });
        // Both composites hold identical content at this instant (the fade
        // just finished at mix=1), so snapping mix back to 0 is invisible.
        self.mix = 0.0;
        self.promote = PromotePhase::Idle;

        // Rebuild design's plan from the same scene in the "background" —
        // pipelines cache-hit, so this is buffer/bind-group creation only.
        let rebuilt = {
            let Self {
                gpu,
                pack,
                registry,
                design,
                ..
            } = self;
            let design = design.as_mut().expect("checked above");
            gpu.as_mut().and_then(
                |g| match PassPlan::build(g, pack, &design.scene, registry) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        log::error!("design rebuild after promote failed: {e:#}");
                        None
                    }
                },
            )
        };
        if let Some(d) = self.design.as_mut() {
            d.plan = rebuilt;
        }
        self.gc_pipelines();
        log::info!("promote complete — design content is now live");
        self.emit_deck();
    }

    /// §5.6 — is anything consuming the design composite this frame? Both
    /// legs tick every frame (cheap; keeps transport/pick state coherent),
    /// but design's render passes only run on demand: the preview toggle
    /// sits on DESIGN, a WS `preview` subscriber exists, or a promote is
    /// pending/ramping (both composites must be current while the final
    /// pass lerps).
    fn design_demand(&self) -> bool {
        if !self.two_leg {
            return false;
        }
        if self.design.as_ref().and_then(|d| d.plan.as_ref()).is_none() {
            return false;
        }
        matches!(
            self.promote,
            PromotePhase::Pending { .. } | PromotePhase::Ramping { .. }
        ) || (self.preview_visible && self.preview_source == Leg::Design)
            || self.bus.subscriber_count("preview") > 0
    }

    fn emit_deck(&self) {
        let (state, fade_ms) = match self.promote {
            PromotePhase::Idle => ("idle", None),
            PromotePhase::Pending { fade_ms, .. } => ("pending", Some(fade_ms)),
            PromotePhase::Ramping { fade_ms, .. } => ("ramping", Some(fade_ms)),
        };
        self.bus.emit_deck(DeckSnapshot {
            promote: state.into(),
            mix: self.mix,
            fade_ms,
            quantize: match self.promote {
                PromotePhase::Pending { .. } => Some("bar".into()),
                _ => None,
            },
            preview_source: match self.preview_source {
                Leg::Live => "live".into(),
                Leg::Design => "design".into(),
            },
            two_leg: self.two_leg,
        });
    }

    pub fn scene(&self) -> &SceneFile {
        &self.scene
    }

    pub fn scene_path(&self) -> &std::path::Path {
        &self.scene_path
    }

    pub fn pack(&self) -> &LoadedPack {
        &self.pack
    }

    pub fn effects_dir(&self) -> Option<&std::path::Path> {
        self.effects_dir.as_deref()
    }

    /// True while the host has reported the output window fully occluded.
    /// While this holds, the host must drive `render_offscreen_frame`
    /// instead of `redraw` — see the field doc for the macOS swapchain trap.
    pub fn occluded(&self) -> bool {
        self.occluded
    }

    /// Host-reported occlusion transition (winit's `WindowEvent::Occluded`
    /// or a tao/NSWindow equivalent).
    pub fn set_occluded(&mut self, occluded: bool) {
        if self.occluded != occluded {
            log::info!(
                "projector window {} — {}",
                if occluded { "occluded" } else { "visible" },
                if occluded {
                    "rendering offscreen (composite + preview stay live)"
                } else {
                    "resuming swapchain presentation"
                }
            );
            // Snap the fps readout to the new mode instead of
            // blending 60 Hz and 30 Hz history across the switch.
            self.fps.reset();
        }
        self.occluded = occluded;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.resize(width, height);
        }
    }

    /// App-collapse Step 3 — attach the native operator-preview surface.
    /// Call after `init_gpu`; `target` is any rwh-0.6 window handle.
    pub fn attach_preview_surface(
        &mut self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("attach_preview_surface before init_gpu"))?;
        gpu.attach_preview(target, width, height)
    }

    pub fn resize_preview_surface(&mut self, width: u32, height: u32) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.resize_preview(width, height);
        }
    }

    /// Host-reported: present the preview surface each frame while true.
    /// The host owns the §3.1 responsibility — set false whenever the
    /// preview window is hidden or occluded.
    pub fn set_preview_visible(&mut self, visible: bool) {
        if self.preview_visible != visible {
            log::info!(
                "native preview {}",
                if visible { "presenting" } else { "paused (slot hidden or window occluded)" }
            );
        }
        self.preview_visible = visible;
    }

    /// Blit + present the preview if attached and visible. Errors degrade
    /// to a log line — the preview must never take the projector down.
    /// §5.6: the source toggle picks the leg; LIVE renders with the real
    /// masters, DESIGN un-mastered.
    fn present_preview(&self) {
        if !self.preview_visible {
            return;
        }
        if let Some(gpu) = self.gpu.as_ref() {
            // §5.6 full-control-switch: each position shows its own leg's
            // masters — the preview is WYSIWYG for the leg you're driving.
            let source = if self.two_leg { self.preview_source } else { Leg::Live };
            let (b, s) = match source {
                Leg::Live => (self.masters.brightness(), self.masters.saturation()),
                Leg::Design => (
                    self.design_masters.brightness(),
                    self.design_masters.saturation(),
                ),
            };
            if let Err(e) = gpu.render_preview(source, b, s) {
                log::warn!("preview present failed: {e}");
            }
        }
    }

    /// §5.6 blanket leg rule: the JPEG `preview` channel is an authoring
    /// surface, so it samples the design composite in two-leg mode.
    fn capture_preview_jpeg(&mut self) {
        if self.bus.subscriber_count("preview") == 0 {
            return;
        }
        let Self { gpu, preview, bus, two_leg, .. } = self;
        let Some(gpu) = gpu.as_ref() else { return };
        let texture = if *two_leg {
            gpu.design_texture.as_ref().unwrap_or(&gpu.composite_texture)
        } else {
            &gpu.composite_texture
        };
        preview.maybe_capture(gpu, bus, texture);
    }

    /// The plan whose structure telemetry reports — §5.6 full-control-switch:
    /// `frame_stats` counts + the `drivers` channel follow the deck toggle
    /// (the leg the UI is currently driving), so the rack shows the values
    /// your knobs actually move.
    fn stats_plan(&self) -> Option<&PassPlan> {
        if self.two_leg && self.preview_source == Leg::Design {
            self.design.as_ref().and_then(|d| d.plan.as_ref())
        } else {
            self.plan.as_ref()
        }
    }

    fn frame_counts(&self) -> FrameCounts {
        let (slices, passes) = self
            .stats_plan()
            .map(|p| {
                let n = p.layer_passes.len() as u32;
                (n, n + 1) // +1 for the homography pass
            })
            .unwrap_or((0, 0));
        FrameCounts {
            mask_slices: slices,
            pipelines: self
                .gpu
                .as_ref()
                .map(|g| g.pipeline_cache.len() as u32)
                .unwrap_or(0),
            passes,
        }
    }

    /// Effective projector calibration: session sidecar first (§5.3), the
    /// deprecated scene.json field as a read-only fallback.
    fn effective_calibration(&self) -> Option<[[f32; 3]; 3]> {
        self.session_calibration
            .or(self.scene.projector_calibration)
    }

    /// Tick both legs, each against **its own** control state (§5.6
    /// full-control-switch: own transport/speed, own knobs, own overrides,
    /// own audioListen — shared raw audio signal). Both tick every frame;
    /// rendering the design composite is demand-gated separately.
    fn tick_legs(&mut self) {
        let Self {
            gpu,
            plan,
            design,
            transport,
            design_transport,
            audio_state,
            sliders,
            masters,
            overrides,
            design_sliders,
            design_masters,
            design_overrides,
            ..
        } = self;
        let Some(gpu) = gpu.as_ref() else { return };
        if let Some(plan) = plan.as_mut() {
            let ctx = transport.frame_context(audio_state, sliders, masters.audio_listen());
            plan.tick(gpu, &ctx, overrides);
        }
        if let Some(d) = design.as_mut() {
            if let Some(p) = d.plan.as_mut() {
                let ctx = design_transport.frame_context(
                    audio_state,
                    design_sliders,
                    design_masters.audio_listen(),
                );
                p.tick(gpu, &ctx, design_overrides);
            }
        }
    }

    /// One presented frame (tick → record → submit → present). Must only be
    /// called while not occluded. On `SurfaceError::Lost`/`Outdated` the host
    /// should query the window's current size and call [`Core::resize`];
    /// on `OutOfMemory` it should shut down.
    pub fn redraw(&mut self) -> Result<(), wgpu::SurfaceError> {
        // §5.4 speed master bends the musical clock (integrated, never a
        // scaled absolute time). §5.6: each leg's clock bends under its own
        // speed master — design at 4× never touches the show.
        self.transport.step(self.masters.speed());
        if self.two_leg {
            self.design_transport.step(self.design_masters.speed());
        }
        self.update_promote();
        let calibration = self.effective_calibration();
        if self.gpu.is_none() || self.plan.is_none() {
            return Ok(());
        }
        self.tick_legs();
        let design_demand = self.design_demand();

        let gpu = self.gpu.as_ref().expect("checked above");
        let plan = self.plan.as_ref().expect("checked above");
        let frame = gpu.surface.get_current_texture()?;
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });
        // 1) Live leg → live composite.
        plan.encode_composite(gpu, &mut encoder, &gpu.composite_view);
        // 2) Design leg → design composite, only when something consumes it.
        if design_demand {
            if let (Some(d), Some(view)) = (self.design.as_ref(), gpu.design_view.as_ref()) {
                if let Some(p) = d.plan.as_ref() {
                    p.encode_composite(gpu, &mut encoder, view);
                }
            }
        }
        // 3) Final pass: live × design lerped by the promote mix → masters
        //    → homography → swapchain. Brightness/saturation refresh per
        //    presented frame so a master move lands next frame.
        gpu.write_homography(
            calibration,
            self.masters.brightness(),
            self.masters.saturation(),
            self.mix,
        );
        gpu.encode_final(&mut encoder, &swap_view);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        self.present_preview();
        let counts = self.frame_counts();
        self.fps.mark(&self.bus, counts, true);
        // §6.4 demand-gated capture: the JPEG readback path only runs
        // when someone actually subscribes to the `preview` channel
        // (remote WS clients; the webview's Prepare canvas underlay).
        self.capture_preview_jpeg();
        Ok(())
    }

    /// One frame with no swapchain interaction — used while occluded so the
    /// composites (and therefore the operator preview) keep updating without
    /// the render thread blocking on the throttled window. No homography
    /// pass runs here, so the brightness/saturation masters don't apply —
    /// by design, the preview shows the un-mastered composite.
    pub fn render_offscreen_frame(&mut self) {
        self.transport.step(self.masters.speed());
        if self.two_leg {
            self.design_transport.step(self.design_masters.speed());
        }
        self.update_promote();
        if self.gpu.is_none() || self.plan.is_none() {
            return;
        }
        self.tick_legs();
        let design_demand = self.design_demand();
        {
            let gpu = self.gpu.as_ref().expect("checked above");
            let plan = self.plan.as_ref().expect("checked above");
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("offscreen frame encoder"),
                });
            plan.encode_composite(gpu, &mut encoder, &gpu.composite_view);
            if design_demand {
                if let (Some(d), Some(view)) = (self.design.as_ref(), gpu.design_view.as_ref()) {
                    if let Some(p) = d.plan.as_ref() {
                        p.encode_composite(gpu, &mut encoder, view);
                    }
                }
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
        // The native preview keeps presenting while the projector window is
        // buried — that's the whole point of the occluded path.
        self.present_preview();
        let counts = self.frame_counts();
        self.fps.mark(&self.bus, counts, false);
        self.capture_preview_jpeg();
    }

    /// Periodic telemetry the UI depends on: `drivers` (~10 Hz), `audio`
    /// (~30 Hz), `connectivity` (piggybacks on the 1 Hz audio pill).
    fn emit_periodic_telemetry(&mut self, connectivity_heartbeat: bool) {
        if self.last_audio_emit.elapsed() >= Duration::from_millis(33) {
            self.bus.emit_audio(AudioSnapshot {
                band_low: self.audio_state.band(crate::osc::AudioBand::Low),
                band_mid: self.audio_state.band(crate::osc::AudioBand::Mid),
                band_high: self.audio_state.band(crate::osc::AudioBand::High),
                onset_low: self
                    .audio_state
                    .onset_envelope(crate::osc::AudioBand::Low, 0.18),
                onset_mid: self
                    .audio_state
                    .onset_envelope(crate::osc::AudioBand::Mid, 0.15),
                onset_high: self
                    .audio_state
                    .onset_envelope(crate::osc::AudioBand::High, 0.10),
            });
            self.last_audio_emit = Instant::now();
        }

        if self.last_drivers_emit.elapsed() >= Duration::from_millis(100) {
            // §5.6 full-control-switch: the driver rack reads whichever leg
            // the deck toggle selects, evaluated with that leg's own
            // transport/knobs/overrides.
            let control_design = self.two_leg && self.preview_source == Leg::Design;
            if let Some(plan) = self.stats_plan() {
                let rows = if control_design {
                    let ctx = self.design_transport.frame_context(
                        &self.audio_state,
                        &self.design_sliders,
                        self.design_masters.audio_listen(),
                    );
                    plan.driver_rows(&ctx, &self.design_overrides)
                } else {
                    let ctx = self.transport.frame_context(
                        &self.audio_state,
                        &self.sliders,
                        self.masters.audio_listen(),
                    );
                    plan.driver_rows(&ctx, &self.overrides)
                };
                self.bus.emit_drivers(DriverSnapshot { drivers: rows });
            }
            self.last_drivers_emit = Instant::now();
        }

        // Deck trickle while a fade ramps (~10 Hz) so the UI's mix readout
        // moves; transitions emit immediately from the state machine.
        if matches!(self.promote, PromotePhase::Ramping { .. })
            && self.last_deck_emit.elapsed() >= Duration::from_millis(100)
        {
            self.emit_deck();
            self.last_deck_emit = Instant::now();
        }

        if connectivity_heartbeat {
            let osc_state = if self.audio_state.is_fresh(2_000) {
                ("ok", None)
            } else if self.audio_state.last_packet_ms() == 0 {
                ("down", Some("no packets yet".to_string()))
            } else {
                ("warn", Some("stale (>2s)".to_string()))
            };
            self.bus.emit_connectivity(Connectivity {
                osc: ConnectivityCell {
                    status: osc_state.0.into(),
                    detail: osc_state.1,
                },
                file_watcher: ConnectivityCell {
                    status: if self.watcher.is_some() { "ok" } else { "down" }.into(),
                    detail: self
                        .watcher
                        .is_none()
                        .then(|| "watcher failed to start".to_string()),
                },
                ws: ConnectivityCell {
                    status: if self.ws_handle.is_some() { "ok" } else { "down" }.into(),
                    detail: self
                        .ws_handle
                        .is_none()
                        .then(|| "engine running headless (no --ws-addr)".to_string()),
                },
            });
        }
    }

    /// Frame-boundary housekeeping, host-agnostic: drain IPC commands, poll
    /// the file watcher, emit the audio-freshness pill + periodic telemetry.
    /// The host calls this once per loop iteration (winit: `about_to_wait`)
    /// before pacing and requesting the next frame.
    pub fn poll_inbound(&mut self) {
        // Drain IPC commands first — they can race the file watcher on the
        // same files but are individually cheaper.
        let queue: Vec<_> = self.cmd_rx.try_iter().collect();
        for cmd in queue {
            rpc::handle(self, cmd);
        }

        let changes = self
            .watcher
            .as_mut()
            .map(|w| w.poll())
            .unwrap_or_default();
        for change in changes {
            match change {
                ChangeKind::Effects => self.reload_effects(),
                ChangeKind::Scene => self.reload_scene(),
            }
        }

        // §5.6 pre-flight probe: run a few ms of probe frames between live
        // frames while an apply is waiting on its verdict.
        self.step_probe();

        // §5.6 design-leg autosave: debounce-write the draft ~2 s after the
        // last applied edit so a crash mid-design can't eat it.
        self.flush_design_autosave();

        // Audio freshness pill — emit on transition and on a 1 Hz heartbeat
        // so the UI has *something* to display before the first transition.
        let fresh = self.audio_state.is_fresh(2_000);
        let stamp = self.audio_state.last_packet_ms();
        let heartbeat = self.last_audio_pill.elapsed() >= Duration::from_secs(1);
        if heartbeat || self.audio_was_fresh != Some(fresh) {
            self.bus.emit_audio_freshness(fresh, stamp);
            self.audio_was_fresh = Some(fresh);
            self.last_audio_pill = Instant::now();
        }

        // Live-value channels the operator UI renders (driver rack, audio
        // strip, connectivity panel).
        self.emit_periodic_telemetry(heartbeat);

        // §5.3 session sidecar — debounced persist of operator state
        // (masters, knobs, overrides). One write ~1.5 s after the last
        // touch, not one per slider tick.
        let dirty_ms = self.session_dirty.load(Ordering::Relaxed);
        if dirty_ms != 0 && session::now_ms().saturating_sub(dirty_ms) >= 1_500 {
            if let Err(err) = self.save_session() {
                log::warn!("session sidecar write failed: {err:#}");
            }
        }

        // §5.11 — SIGTERM/SIGINT: snapshot the session, then ask the host to
        // exit. The host polls `exit_requested()` after this call.
        if self.term_flag.swap(false, Ordering::Relaxed) {
            log::info!("termination signal — snapshotting session and exiting");
            if let Err(err) = self.save_session() {
                log::warn!("session snapshot on shutdown failed: {err:#}");
            }
            self.exit_requested = true;
        }
    }

    /// True once a termination signal asked for a graceful exit. The host
    /// checks this each loop iteration and tears the event loop down.
    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }

    /// Host-initiated shutdown hook (window close). Persists the session
    /// sidecar so knobs/masters survive to the next run.
    pub fn on_exit(&mut self) {
        if let Err(err) = self.save_session() {
            log::warn!("session snapshot on exit failed: {err:#}");
        }
    }

    /// §5.6 design-leg autosave — debounced write to
    /// `<scene_dir>/.wzrd/design.scene.json` (atomic temp + rename), offered
    /// back into the design leg at next boot.
    fn flush_design_autosave(&mut self) {
        let Some(design) = self.design.as_mut() else { return };
        let Some(dirty_at) = design.autosave_dirty_at else { return };
        if dirty_at.elapsed() < Duration::from_secs(2) {
            return;
        }
        design.autosave_dirty_at = None;
        let path = design_autosave_path(&self.scene_path);
        let write = (|| -> Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, design.raw.as_bytes())?;
            std::fs::rename(&tmp, &path)?;
            Ok(())
        })();
        match write {
            Ok(()) => log::debug!("design draft autosaved to {}", path.display()),
            Err(e) => log::warn!("design autosave failed: {e:#}"),
        }
    }

    /// §5.3 — write the session sidecar now. Clears the dirty stamp first so
    /// a change racing the write just re-dirties and gets the next debounce.
    pub fn save_session(&mut self) -> Result<PathBuf> {
        self.session_dirty.store(0, Ordering::Relaxed);
        let mut file = SessionFile {
            version: crate::session::SESSION_VERSION,
            projector_calibration: self.session_calibration,
            masters: Some(self.masters.snapshot()),
            probe_thresholds: Some(self.probe_thresholds.snapshot()),
            ..Default::default()
        };
        for (name, value) in self.sliders.snapshot() {
            file.params.insert(name, value);
        }
        for (binding, param, value) in self.overrides.snapshot() {
            file.overrides.entry(binding).or_default().insert(param, value);
        }
        session::save(&self.session_path, &file)?;
        log::info!("session sidecar written: {}", self.session_path.display());
        Ok(self.session_path.clone())
    }

    /// §5.5 — serve `effect.describe` from the render thread's registry.
    pub fn describe_effects(&self, name: Option<&str>) -> Result<serde_json::Value> {
        self.registry.describe(name)
    }

    /// Control-surface handle for an embedding host (app-collapse Step 2):
    /// the same inline-dispatch context + command channel the WS server
    /// uses, so an in-process consumer (`rpc::dispatch`) exercises the
    /// identical §3.11 method set with no second code path. Command-channel
    /// senders fail once Core (the receiver) is gone — that's how a host
    /// detects a dead render thread.
    pub fn control_channel(
        &self,
    ) -> (RpcContext, crossbeam_channel::Sender<EngineCommand>) {
        (self.rpc_ctx.clone(), self.cmd_tx.clone())
    }

    /// Step-2 runtime-spike hook: forcibly destroy the wgpu device to
    /// simulate a mid-show device loss. Subsequent GPU work errors out (the
    /// default uncaptured-error handler panics on the render thread) —
    /// used to prove crash containment + state integrity, never in a show.
    pub fn spike_force_device_loss(&self) {
        if let Some(gpu) = self.gpu.as_ref() {
            log::warn!("SPIKE: forcing wgpu device destroy (simulated device loss)");
            gpu.device.destroy();
        }
    }

    /// Frame-pacing cap. macOS Metal disables vsync throttling for
    /// occluded surfaces, so without this the render thread freewheels at
    /// 2000+ fps and the preview readback can't keep up. Sleeping here
    /// keeps CPU + GPU in lockstep and makes "fps drops under heavy
    /// shaders" actually observable.
    ///
    /// While occluded we self-pace at ~30 Hz — enough to feed the 15 fps
    /// preview with headroom, without burning GPU on frames nobody sees.
    ///
    /// Blocks (sleeps) until the frame budget has elapsed since the previous
    /// call; the host then requests/renders the next frame.
    /// Test/host hook: is a probe currently holding a design apply?
    pub fn probe_in_flight(&self) -> bool {
        self.pending_apply.is_some()
    }

    pub fn pace_frame(&mut self) {
        let budget = if self.occluded {
            Some(
                self.frame_budget
                    .map_or(Duration::from_millis(33), |b| b.max(Duration::from_millis(33))),
            )
        } else {
            self.frame_budget
        };
        if let Some(budget) = budget {
            if let Some(last) = self.last_redraw_request {
                let elapsed = last.elapsed();
                if elapsed < budget {
                    std::thread::sleep(budget - elapsed);
                }
            }
            self.last_redraw_request = Some(Instant::now());
        }
    }
}

/// §5.6 design-leg autosave location: `<scene_dir>/.wzrd/design.scene.json`.
fn design_autosave_path(scene_path: &std::path::Path) -> PathBuf {
    scene_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".wzrd")
        .join("design.scene.json")
}
