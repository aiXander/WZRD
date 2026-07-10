//! Winit host — the standalone binary's event loop.
//!
//! Since the app-collapse Step-1 split, all engine logic lives in
//! [`crate::core::Core`]; this file is only the glue between winit's
//! `ApplicationHandler` contract and Core's host-agnostic API. It owns the
//! native window (creation, `request_redraw`, size queries after surface
//! loss) and the exit decision; Core owns everything behind the surface.
//!
//! A future TauriHost (app-collapse Step 2) drives the same Core from a tao
//! window inside the Tauri shell process instead.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Fullscreen, Window, WindowId};

use crate::core::Core;
use crate::Cli;

pub struct WinitHost {
    cli: Cli,
    core: Core,
    /// Created when `resumed` fires (winit 0.30 contract); also the
    /// "GPU is up" guard against duplicate `resumed` calls.
    window: Option<Arc<Window>>,
}

impl WinitHost {
    pub fn new(cli: Cli) -> Result<Self> {
        let core = Core::new(&cli)?;
        Ok(Self {
            cli,
            core,
            window: None,
        })
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
                self.core.pack().atlas_width,
                self.core.pack().atlas_height,
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

impl ApplicationHandler for WinitHost {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
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
        let size = window.inner_size();
        if let Err(err) = self
            .core
            .init_gpu(Arc::clone(&window), size.width, size.height)
        {
            log::error!("could not initialise wgpu: {err:#}");
            event_loop.exit();
            return;
        }
        self.window = Some(window);
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
                self.core.resize(size.width, size.height);
            }
            WindowEvent::Occluded(occluded) => {
                self.core.set_occluded(occluded);
            }
            WindowEvent::RedrawRequested => {
                if self.core.occluded() {
                    // Never touch the throttled swapchain — see §3.1.
                    self.core.render_offscreen_frame();
                    return;
                }
                match self.core.redraw() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        if let Some(window) = self.window.as_ref() {
                            let size = window.inner_size();
                            self.core.resize(size.width, size.height);
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
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.core.poll_inbound();
        self.core.pace_frame();

        if self.core.occluded() {
            // macOS stops delivering RedrawRequested to occluded windows —
            // drive the offscreen frame from here instead.
            self.core.render_offscreen_frame();
        } else if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
