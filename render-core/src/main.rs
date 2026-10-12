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
//!     render-core --bench perf/scenarios/smoke.json --bench-out r.json  # PM3 perf bench

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
        if forward_to_bus(record) && self.inner.matches(record) {
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

/// Info and louder, except the GPU stack's chatter below Warn: wgpu-core logs
/// "Device::maintain: waiting for submission index N" at INFO on every
/// maintain (~180/s), so a plain `RUST_LOG=info` would flood the `log`
/// channel (and the Debug room) and cost frame time. stderr still gets
/// whatever `RUST_LOG` asks for.
fn forward_to_bus(record: &log::Record) -> bool {
    let gpu_stack = ["wgpu_core", "wgpu_hal", "naga"]
        .iter()
        .any(|p| record.target().starts_with(p));
    record.level() <= if gpu_stack { log::Level::Warn } else { log::Level::Info }
}

fn main() -> Result<()> {
    let inner = env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"),
    )
    .build();
    log::set_max_level(inner.filter());
    log::set_boxed_logger(Box::new(TeeLogger { inner })).expect("logger already set");

    // PM3: `--bench` is its own grammar (no `--scene`, no window) — the
    // engine `Cli` the Tauri host shares stays untouched.
    if std::env::args().any(|a| a == "--bench" || a.starts_with("--bench=")) {
        return render_core::bench::run(render_core::bench::BenchArgs::parse());
    }
    let cli = render_core::Cli::parse();
    render_core::run(cli)
}
