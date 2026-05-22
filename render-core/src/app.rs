//! Engine event-loop integration.
//!
//! `App` owns the winit `ApplicationHandler` contract, the GPU context, the
//! current pass plan, the OSC listener, the file watcher, and (since
//! Phase 4) the IPC bus that lets the Tauri shell drive scene reloads and
//! receive telemetry over a localhost JSON-RPC WebSocket.
//!
//! State mutation stays single-writer (the render thread); inbound IPC
//! commands are drained at frame boundary in `about_to_wait`, mirroring the
//! file-watcher path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Fullscreen, Window, WindowId};

use crate::compositor::PassPlan;
use crate::drivers::Transport;
use crate::effects::EffectRegistry;
use crate::gpu::GpuContext;
use crate::osc::{try_spawn, AudioFeatures, OscListener};
use crate::pack::LoadedPack;
use crate::rpc::{self, parking_lot_lite::SwapValue, EngineCommand, PackInfo, RpcContext};
use crate::scene::SceneFile;
use crate::telemetry::{Bus, FpsAccumulator, HotReloadEvent, PreviewSampler};
use crate::watch::{ChangeKind, SceneWatcher};
use crate::ws;
use crate::Cli;

pub struct App {
    cli: Cli,
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
    /// mutex on the App.
    scene_state: Arc<SwapValue>,

    /// Inbound command channel from the WS server. Drained at frame
    /// boundary. `None` when the engine is running headless without
    /// `--ws-addr`.
    cmd_rx: Option<crossbeam_channel::Receiver<EngineCommand>>,

    /// Kept to keep the WS accept thread alive.
    _ws_handle: Option<ws::ServerHandle>,

    // Lazily initialised once `resumed` fires (winit 0.30 contract).
    gpu: Option<GpuContext>,
    plan: Option<PassPlan>,
    watcher: Option<SceneWatcher>,

    fps: FpsAccumulator,
    preview: PreviewSampler,
    audio_was_fresh: Option<bool>,
    last_audio_pill: Instant,

    /// Frame-pacing state. `frame_budget` is the minimum time we want between
    /// consecutive `request_redraw` calls (derived from `--frame-cap-hz`);
    /// `last_redraw_request` is when we last released one. `None` budget
    /// disables the cap (`--frame-cap-hz 0`).
    frame_budget: Option<Duration>,
    last_redraw_request: Option<Instant>,
}

impl App {
    pub fn new(cli: Cli) -> Result<Self> {
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
        let scene_state = Arc::new(SwapValue::new());
        // Seed scene_state with what's on disk so `scene.getState` returns
        // something even before the first hot-reload.
        if let Ok(raw) = std::fs::read_to_string(&scene_path) {
            scene_state.set(raw);
        }

        let (cmd_rx, ws_handle) = match cli.ws_addr {
            Some(addr) => {
                let (tx, rx) = crossbeam_channel::unbounded();
                let ctx = RpcContext {
                    pack: Arc::new(PackInfo::from_pack(&pack)),
                    scene_state: Arc::clone(&scene_state),
                    effects_dir: effects_dir.clone(),
                    bus: bus.clone(),
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
            cli,
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
            frame_budget,
            last_redraw_request: None,
        })
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

    fn handle_command(&mut self, cmd: EngineCommand) {
        rpc::handle(self, cmd);
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

    fn build_window(&self, event_loop: &ActiveEventLoop) -> Result<Window> {
        let monitors: Vec<_> = event_loop.available_monitors().collect();
        let target_monitor = match self.cli.display {
            Some(idx) => monitors.get(idx).cloned().ok_or_else(|| {
                anyhow!(
                    "--display {idx} but only {} monitor(s) detected",
                    monitors.len()
                )
            })?,
            None => event_loop
                .primary_monitor()
                .or_else(|| monitors.first().cloned())
                .ok_or_else(|| anyhow!("no monitors detected"))?,
        };

        let mut attrs = Window::default_attributes()
            .with_title("render-core")
            .with_resizable(true);

        if self.cli.windowed {
            attrs = attrs.with_inner_size(winit::dpi::PhysicalSize::new(
                self.pack.atlas_width,
                self.pack.atlas_height,
            ));
        } else {
            attrs = attrs
                .with_decorations(false)
                .with_fullscreen(Some(Fullscreen::Borderless(Some(target_monitor.clone()))));
        }

        event_loop
            .create_window(attrs)
            .context("creating native window")
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window = match self.build_window(event_loop) {
            Ok(w) => Arc::new(w),
            Err(err) => {
                log::error!("could not create window: {err:#}");
                event_loop.exit();
                return;
            }
        };
        let gpu = match pollster::block_on(GpuContext::new(window, &self.pack)) {
            Ok(g) => g,
            Err(err) => {
                log::error!("could not initialise wgpu: {err:#}");
                event_loop.exit();
                return;
            }
        };
        self.gpu = Some(gpu);
        self.pack.mask_atlas = Vec::new();
        self.pack.mask_atlas.shrink_to_fit();
        let _ = self.rebuild_plan();

        match SceneWatcher::new(&self.scene_path, self.effects_dir.as_deref()) {
            Ok(w) => self.watcher = Some(w),
            Err(err) => log::warn!("hot-reload disabled: {err:#}"),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(gpu), Some(plan)) = (self.gpu.as_ref(), self.plan.as_ref()) {
                    plan.tick(gpu, &self.transport, &self.audio_state);
                    match plan.record_and_submit(gpu) {
                        Ok(()) => {
                            self.fps.mark(&self.bus);
                            self.preview.maybe_capture(gpu, &self.bus);
                        }
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = gpu.window.inner_size();
                            if let Some(gpu_mut) = self.gpu.as_mut() {
                                gpu_mut.resize(size.width, size.height);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("GPU out of memory — exiting");
                            event_loop.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            log::warn!("frame timeout, skipping");
                        }
                    }
                }
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Drain IPC commands first — they can race the file watcher on the
        // same files but are individually cheaper.
        if let Some(rx) = self.cmd_rx.as_ref() {
            let queue: Vec<_> = rx.try_iter().collect();
            for cmd in queue {
                self.handle_command(cmd);
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

        // Frame-pacing cap. macOS Metal disables vsync throttling for
        // occluded surfaces, so without this the render thread freewheels at
        // 2000+ fps and the preview readback can't keep up. Sleeping here
        // keeps CPU + GPU in lockstep and makes "fps drops under heavy
        // shaders" actually observable.
        if let Some(budget) = self.frame_budget {
            if let Some(last) = self.last_redraw_request {
                let elapsed = last.elapsed();
                if elapsed < budget {
                    std::thread::sleep(budget - elapsed);
                }
            }
            self.last_redraw_request = Some(Instant::now());
        }

        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }
}
