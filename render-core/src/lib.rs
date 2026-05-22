//! WZRD `render-core` — realtime additive projection-mapping engine.
//!
//! Phases 0–3 shipped the standalone playable engine. Phase 4 wraps the same
//! engine in a Tauri shell that drives it as a subprocess over a localhost
//! JSON-RPC WebSocket — the same RPC surface Phase 7 needs for MCP. The
//! crate exposes a small public API (`Cli` + `run`) so both the bundled
//! `render-core` binary *and* the Tauri sidecar use the same entry point.
//!
//! Headless agent loop is unchanged: invoke `run(Cli { scene, … })` with no
//! `ws_addr` and the engine ignores its IPC surface entirely.

pub mod app;
pub mod compositor;
pub mod drivers;
pub mod effects;
pub mod gpu;
pub mod osc;
pub mod pack;
pub mod rpc;
pub mod scene;
pub mod telemetry;
pub mod watch;
pub mod ws;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

/// CLI surface exposed by the standalone binary. Re-used verbatim by Tauri's
/// sidecar invocation (see `wzrd-app/src-tauri/src/engine.rs`) — there is no
/// second CLI grammar.
#[derive(Parser, Debug, Clone)]
#[command(name = "render-core", version, about = "WZRD realtime engine")]
pub struct Cli {
    /// Path to a layer pack directory (containing pack.json + masks/).
    /// Defaults to the `pack` field inside `--scene`.
    #[arg(long)]
    pub pack: Option<PathBuf>,

    /// Path to a scene.json (the bindings file, distinct from the pack
    /// manifest).
    #[arg(long)]
    pub scene: PathBuf,

    /// Path to a project-local effects directory holding user-authored WGSL
    /// effects (D15). Defaults to `<scene_dir>/effects/` if it exists.
    #[arg(long)]
    pub effects: Option<PathBuf>,

    /// Monitor index to fullscreen onto. Defaults to the primary monitor.
    #[arg(long)]
    pub display: Option<usize>,

    /// Run as a regular window instead of borderless fullscreen.
    #[arg(long)]
    pub windowed: bool,

    /// Disable OSC ingest entirely. Clocks still tick; `audio.*` returns 0.
    #[arg(long)]
    pub no_osc: bool,

    /// UDP address the OSC receiver binds to. Default 127.0.0.1:9000.
    #[arg(long, default_value = "127.0.0.1:9000")]
    pub osc_addr: SocketAddr,

    /// Bind a JSON-RPC WebSocket server on this address (Phase 4 control
    /// surface). Tauri spawns the engine with `--ws-addr 127.0.0.1:<port>`;
    /// the headless agent path leaves this off and the engine has no
    /// control surface, only file-watcher hot-reload.
    #[arg(long)]
    pub ws_addr: Option<SocketAddr>,

    /// Cap the render loop at this many frames per second. The CPU thread
    /// sleeps between redraws so we don't outrun the GPU when the engine
    /// window is occluded (macOS Metal disables vsync for hidden surfaces,
    /// otherwise we'd freewheel at 2000+ fps and starve the readback path).
    /// `0` disables the cap entirely — useful when benchmarking pure GPU
    /// throughput.
    #[arg(long, default_value = "240")]
    pub frame_cap_hz: u32,
}

/// Run the engine with the provided CLI args. Blocks until the event loop
/// exits. Use this from both the standalone binary and a Tauri sidecar
/// spawn site.
pub fn run(cli: Cli) -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new().context("creating event loop")?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = app::App::new(cli)?;
    event_loop
        .run_app(&mut app)
        .context("event loop terminated")?;
    Ok(())
}
