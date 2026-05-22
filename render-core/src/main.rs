//! `render-core` — WZRD's realtime engine (Phase 2 skeleton).
//!
//! Usage:
//!     render-core --scene path/to/scene.json
//!     render-core --scene scene.json --pack path/to/layerpack/
//!     render-core --scene scene.json --display 1
//!     render-core --scene scene.json --windowed
//!
//! Headless agent loop: edit `scene.json`, save, projector window updates
//! within one frame budget — no UI process required (D13).

mod compositor;
mod effects;
mod gpu;
mod pack;
mod scene;
mod watch;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowId};

use crate::compositor::PassPlan;
use crate::gpu::GpuContext;
use crate::pack::LoadedPack;
use crate::scene::SceneFile;
use crate::watch::SceneWatcher;

#[derive(Parser, Debug)]
#[command(name = "render-core", version, about = "WZRD realtime engine")]
struct Cli {
    /// Path to a layer pack directory (containing scene.json + masks/).
    /// Defaults to the `pack` field inside `--scene`.
    #[arg(long)]
    pack: Option<PathBuf>,

    /// Path to a scene.json (the *bindings* file, distinct from the pack
    /// manifest).
    #[arg(long)]
    scene: PathBuf,

    /// Monitor index to fullscreen onto. Defaults to the primary monitor.
    #[arg(long)]
    display: Option<usize>,

    /// Run as a regular window instead of borderless fullscreen — useful
    /// on a single-display laptop while iterating on a scene.
    #[arg(long)]
    windowed: bool,
}

fn main() -> Result<()> {
    // Default: render-core at info, wgpu internals at warn. Override with
    // RUST_LOG, e.g. `RUST_LOG=render_core=debug,wgpu_core=info`.
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();

    let cli = Cli::parse();
    let event_loop = EventLoop::new().context("creating event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(cli)?;
    event_loop.run_app(&mut app).context("event loop terminated")?;
    Ok(())
}

/// Top-level app state. `winit` calls into here for window + event handling.
struct App {
    cli: Cli,
    pack: LoadedPack,
    scene: SceneFile,
    scene_path: PathBuf,

    // Lazily initialised once `resumed` fires (winit 0.30 contract).
    gpu: Option<GpuContext>,
    plan: Option<PassPlan>,
    watcher: Option<SceneWatcher>,
}

impl App {
    fn new(cli: Cli) -> Result<Self> {
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

        Ok(Self {
            cli,
            pack,
            scene,
            scene_path,
            gpu: None,
            plan: None,
            watcher: None,
        })
    }

    fn rebuild_plan(&mut self) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        match PassPlan::build(gpu, &self.pack, &self.scene) {
            Ok(plan) => {
                log::info!("scene plan built ({} layer passes)", plan.layer_passes.len());
                self.plan = Some(plan);
            }
            Err(err) => {
                log::error!("rejecting scene update: {err:#}");
                // Keep the previous good plan rendering (swap-on-success, §3.6).
            }
        }
        gpu.set_homography(self.scene.projector_calibration);
    }

    fn reload_scene(&mut self) {
        match SceneFile::load(&self.scene_path) {
            Ok(scene) => {
                log::info!("hot-reloaded {}", self.scene_path.display());
                self.scene = scene;
                self.rebuild_plan();
            }
            Err(err) => {
                log::error!(
                    "ignoring scene reload (file invalid): {err:#}; previous plan remains active"
                );
            }
        }
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
            // Per §6.1: borderless fullscreen sized to the target monitor is
            // the stable fallback on macOS where true exclusive fullscreen
            // misbehaves under operator interaction.
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
        self.rebuild_plan();

        match SceneWatcher::new(&self.scene_path) {
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
                    match plan.record_and_submit(gpu) {
                        Ok(()) => {}
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
        // Hot-reload poll. The watcher debounces editor save bursts; we only
        // hit `reload_scene` on the trailing event.
        let needs_reload = self
            .watcher
            .as_mut()
            .map(|w| w.poll())
            .unwrap_or(false);
        if needs_reload {
            self.reload_scene();
        }
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }
}
