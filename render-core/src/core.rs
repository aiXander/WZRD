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
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::compositor::PassPlan;
use crate::drivers::{SliderBank, Transport};
use crate::effects::EffectRegistry;
use crate::gpu::GpuContext;
use crate::osc::{try_spawn, AudioFeatures, OscListener};
use crate::pack::LoadedPack;
use crate::rpc::{self, parking_lot_lite::SwapValue, EngineCommand, PackInfo, RpcContext};
use crate::scene::SceneFile;
use crate::telemetry::{
    AudioSnapshot, Bus, Connectivity, ConnectivityCell, DriverSnapshot, FpsAccumulator,
    FrameCounts, HotReloadEvent, PreviewSampler,
};
use crate::watch::{ChangeKind, SceneWatcher};
use crate::Cli;
use crate::ws;

pub struct Core {
    pack: LoadedPack,
    scene: SceneFile,
    scene_path: PathBuf,
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

    /// Inbound command channel from the WS server. Drained at frame
    /// boundary. `None` when the engine is running headless without
    /// `--ws-addr`.
    cmd_rx: Option<crossbeam_channel::Receiver<EngineCommand>>,

    /// Kept to keep the WS accept thread alive.
    _ws_handle: Option<ws::ServerHandle>,

    // Lazily initialised once the host hands over a surface target
    // (`init_gpu`) — for winit that's when `resumed` fires (0.30 contract).
    gpu: Option<GpuContext>,
    plan: Option<PassPlan>,
    watcher: Option<SceneWatcher>,

    fps: FpsAccumulator,
    preview: PreviewSampler,
    audio_was_fresh: Option<bool>,
    last_audio_pill: Instant,

    /// Live `ui.slider` values, shared with the WS server (`param.set`).
    sliders: Arc<SliderBank>,

    /// True while the OS reports the projector window fully occluded. On
    /// macOS an occluded window's swapchain throttles to ~1 Hz and
    /// `get_current_texture()` blocks the render thread for up to a second
    /// per frame — so while occluded the host must render via
    /// `render_offscreen_frame` (preview + telemetry stay live) and never
    /// call `redraw`.
    occluded: bool,

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
        // Seed scene_state with what's on disk so `scene.getState` returns
        // something even before the first hot-reload.
        if let Ok(raw) = std::fs::read_to_string(&scene_path) {
            scene_state.set(raw);
        }

        let sliders = SliderBank::new();
        let (cmd_rx, ws_handle) = match cli.ws_addr {
            Some(addr) => {
                let (tx, rx) = crossbeam_channel::unbounded();
                let ctx = RpcContext {
                    pack: Arc::new(PackInfo::from_pack(&pack)),
                    scene_state: Arc::clone(&scene_state),
                    effects_dir: effects_dir.clone(),
                    bus: bus.clone(),
                    sliders: Arc::clone(&sliders),
                };
                let handle = ws::serve(addr, tx, ctx).context("starting WS server")?;
                (Some(rx), Some(handle))
            }
            None => (None, None),
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

        Ok(Self {
            pack,
            scene,
            scene_path,
            effects_dir,
            registry,
            transport,
            audio_state,
            _osc_listener: osc_listener,
            bus,
            scene_state,
            cmd_rx,
            _ws_handle: ws_handle,
            gpu: None,
            plan: None,
            watcher: None,
            fps: FpsAccumulator::new(),
            preview: PreviewSampler::new(),
            audio_was_fresh: None,
            last_audio_pill: Instant::now(),
            sliders,
            occluded: false,
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
        let gpu = pollster::block_on(GpuContext::new(target, width, height, &self.pack))
            .context("initialising wgpu")?;
        self.gpu = Some(gpu);
        // The atlas now lives on the GPU; drop the CPU-side copy.
        self.pack.mask_atlas = Vec::new();
        self.pack.mask_atlas.shrink_to_fit();
        let _ = self.rebuild_plan();

        match SceneWatcher::new(&self.scene_path, self.effects_dir.as_deref()) {
            Ok(w) => self.watcher = Some(w),
            Err(err) => log::warn!("hot-reload disabled: {err:#}"),
        }
        Ok(())
    }

    fn rebuild_plan(&mut self) -> Result<()> {
        let Some(gpu) = self.gpu.as_mut() else {
            return Ok(());
        };
        let started = Instant::now();
        match PassPlan::build(gpu, &self.pack, &self.scene, &self.registry) {
            Ok(plan) => {
                log::info!("scene plan built ({} layer passes)", plan.layer_passes.len());
                self.plan = Some(plan);
                self.bus.emit_hot_reload(HotReloadEvent {
                    target: "scene".into(),
                    ok: true,
                    elapsed_ms: started.elapsed().as_secs_f32() * 1000.0,
                    message: None,
                });
                self.transport.set_bpm(self.scene.transport.bpm);
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.set_homography(self.scene.projector_calibration);
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
                });
                Err(err)
            }
        }
    }

    fn reload_scene(&mut self) {
        match std::fs::read_to_string(&self.scene_path) {
            // The UI saves via `scene.load` IPC *and* writes to disk; the
            // file watcher then fires on our own write. Skip the redundant
            // rebuild when the on-disk content matches what's already live.
            Ok(raw) if self.scene_state.get().as_deref() == Some(raw.as_str()) => {
                log::debug!("scene reload skipped (content unchanged)");
            }
            Ok(raw) => match SceneFile::parse(&raw) {
                Ok(scene) => {
                    log::info!("hot-reloaded {}", self.scene_path.display());
                    self.scene = scene;
                    if self.rebuild_plan().is_ok() {
                        self.scene_state.set(raw);
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
                });
            }
        }
    }

    fn reload_effects(&mut self) {
        let changed = self.registry.rescan_disk();
        if changed.is_empty() {
            return;
        }
        log::info!("effect pipelines invalidated: {:?}", changed);
        if let Some(gpu) = self.gpu.as_mut() {
            for key in &changed {
                gpu.pipeline_cache.remove(key);
            }
        }
        for key in &changed {
            self.bus.emit_hot_reload(HotReloadEvent {
                target: format!("effect {key}"),
                ok: true,
                elapsed_ms: 0.0,
                message: None,
            });
        }
        let _ = self.rebuild_plan();
    }

    /// Apply a scene loaded from raw JSON (e.g. from `scene.load` IPC) instead
    /// of from disk. Used by the Tauri shell to push edits without writing to
    /// disk first.
    pub fn apply_scene_json(&mut self, raw: &str) -> Result<()> {
        let scene = SceneFile::parse(raw)?;
        self.scene = scene;
        self.rebuild_plan()?;
        self.scene_state.set(raw.to_string());
        Ok(())
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

    fn frame_counts(&self) -> FrameCounts {
        let (slices, passes) = self
            .plan
            .as_ref()
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

    /// One presented frame (tick → record → submit → present). Must only be
    /// called while not occluded. On `SurfaceError::Lost`/`Outdated` the host
    /// should query the window's current size and call [`Core::resize`];
    /// on `OutOfMemory` it should shut down.
    pub fn redraw(&mut self) -> Result<(), wgpu::SurfaceError> {
        if let (Some(gpu), Some(plan)) = (self.gpu.as_ref(), self.plan.as_ref()) {
            plan.tick(gpu, &self.transport, &self.audio_state, &self.sliders);
            plan.record_and_submit(gpu)?;
            let counts = self.frame_counts();
            self.fps.mark(&self.bus, counts, true);
            if let Some(gpu) = self.gpu.as_ref() {
                self.preview.maybe_capture(gpu, &self.bus);
            }
        }
        Ok(())
    }

    /// One frame with no swapchain interaction — used while occluded so the
    /// composite (and therefore the operator preview) keeps updating without
    /// the render thread blocking on the throttled window.
    pub fn render_offscreen_frame(&mut self) {
        if let (Some(gpu), Some(plan)) = (self.gpu.as_ref(), self.plan.as_ref()) {
            plan.tick(gpu, &self.transport, &self.audio_state, &self.sliders);
            plan.render_offscreen(gpu);
        }
        let counts = self.frame_counts();
        self.fps.mark(&self.bus, counts, false);
        if let Some(gpu) = self.gpu.as_ref() {
            self.preview.maybe_capture(gpu, &self.bus);
        }
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
            if let Some(plan) = self.plan.as_ref() {
                let rows =
                    plan.driver_rows(&self.transport, &self.audio_state, &self.sliders);
                self.bus.emit_drivers(DriverSnapshot { drivers: rows });
            }
            self.last_drivers_emit = Instant::now();
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
                    status: if self.cmd_rx.is_some() { "ok" } else { "down" }.into(),
                    detail: self
                        .cmd_rx
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
        if let Some(rx) = self.cmd_rx.as_ref() {
            let queue: Vec<_> = rx.try_iter().collect();
            for cmd in queue {
                rpc::handle(self, cmd);
            }
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
