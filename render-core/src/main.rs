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

/// Tee logger: stderr via env_logger + the `log` telemetry channel once the
/// engine bus exists. Only Info and louder are forwarded to the bus — the
/// bus's own internals log at trace, so this also breaks any recursion.
struct TeeLogger {
    inner: env_logger::Logger,
}

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        self.inner.log(record);
        if record.level() <= log::Level::Info && self.inner.matches(record) {
            if let Some(bus) = render_core::telemetry::global_bus() {
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

fn main() -> Result<()> {
    let inner = env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .build();
    log::set_max_level(inner.filter());
    log::set_boxed_logger(Box::new(TeeLogger { inner })).expect("logger already set");

    let cli = render_core::Cli::parse();
    render_core::run(cli)
}
