//! Thin entry point — sets up logging, parses the CLI, hands off to the lib
//! (`render_core::run`). All real logic lives in the library so the Tauri
//! shell (Phase 4) can spawn this same binary as a sidecar and drive it over
//! its JSON-RPC WebSocket without re-implementing the entry path.
//!
//! Usage examples:
//!     render-core --scene path/to/scene.json
//!     render-core --scene scene.json --pack path/to/layerpack/
//!     render-core --scene scene.json --effects path/to/effects/
//!     render-core --scene scene.json --display 1
//!     render-core --scene scene.json --windowed
//!     render-core --scene scene.json --ws-addr 127.0.0.1:9123    # Phase 4 IPC

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .init();

    let cli = render_core::Cli::parse();
    render_core::run(cli)
}
