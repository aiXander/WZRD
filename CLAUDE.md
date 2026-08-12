# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WZRD is an additive projection-mapping system: only changing regions get projected — unchanged areas stay dark and merge with physical reality. Three cooperating parts:

- **Python offline toolkit** (`wzrd/` + `wzrd_mcp/`) — extracts/segments surfaces from photos and authors layer packs.
- **Rust realtime engine** (`render-core/`) — wgpu compositor that plays layer packs on a projector; headless-agent-drivable via `scene.json` + WGSL hot-reload.
- **Tauri control shell** (`wzrd-app/`) — React UI wrapping the engine for humans.

**Keep the Python toolkit and `render-core/` Linux/NVIDIA-portable.** A future Linux-native UI is planned on top of those two, so new work in them must not acquire a macOS dependency: wgpu stays on `Backends::PRIMARY` with `Features::empty()`/`Limits::default()` (no Metal-only features or formats), platform calls stay `#[cfg]`-gated behind a working non-macOS branch, ffmpeg codecs stay parameters (never hardcoded `*_videotoolbox`), and the headless path must not assume AppKit. `wzrd-app/` is exempt — it may be fully Mac-native (AppKit child windows, `NSWindow.occlusionState`, App Nap opt-out); its non-macOS fallbacks are untested sketches, so don't treat them as a portability guarantee. Detail on where the current lines fall: `docs/reference/render-engine.md` §1b.

## Docs (`docs/`)

- `docs/reference/render-engine.md` — **primary engine context doc**: current system state, contracts (scene.json, layer pack, RPC/telemetry, WGSL), invariants from the 2026-07 performance pass, working agreements, rejected approaches. Read before touching `render-core/` or `wzrd-app/`.
- `docs/reference/user_design_spec.md` — the product north star; every engine trade-off answers to it.
- `docs/TODO/render-engine-roadmap.md` — structural roadmap; the stable index of what's next (§5.11 post-swap probation → §5.5 WGSL `#import` → auto-pilot playlist / §5.7 layer object). Landed sections are drained to pointers into the reference doc. Other in-flight plans live alongside it in `docs/TODO/`; you'll be pointed at one by name.
- Single-process collapse — **LANDED 2026-07-12** (Steps 2–3: in-process TauriHost + native lossless preview). Current-state residue: `docs/reference/render-engine.md` §1/§1b; the plan husk is retired in `docs/finished/`.
- `docs/finished/` — retired plans (v1 build plan lives here). Write-only; never read from it.

## Setup & Commands

```bash
# Install (editable, with all optional deps)
pip install -e ".[islands,mcp]"

# Run any module standalone
python -m wzrd.detect photo.jpg -o cropped.png
python -m wzrd.darken surface.jpg -o dark.png
python -m wzrd.align source.png target.png -o aligned.png
python -m wzrd.subtract_frame generated.png background.png -o creature.png
python -m wzrd.subtract_video animation.mp4 background.png -o output.mp4
python -m wzrd.islands input.png -o regions/
python -m wzrd.reproject regions_dir/ -o output.mp4

# Run smoke tests (all or specific modules)
python test.py --all
python test.py subtract_frame darken

# Run MCP server locally
python -m wzrd_mcp            # default: streamable-HTTP on 0.0.0.0:8787
python -m wzrd_mcp --no-debug # disable verbose tool logging
python -m wzrd_mcp --stdio    # stdio transport (Claude Desktop spawns this itself)

# Realtime engine (headless agent path — no UI process)
# NOTE: there is NO `render-core` on PATH. `cargo build` only writes the binary
# into render-core/target/. Launch via `cargo run --` or ./target/release/render-core.
cd render-core
python ../test.py layerpack   # one-off: build the example pack the sample scene points at
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc
# --no-osc runs without the audio server (clocks tick, audio.* drivers → 0).
# Drop it and start the audio server to make audio-reactive layers (flash/drift) respond.
# --windowed → drop for borderless fullscreen; add `--display 1` to pick a monitor.

# Realtime Audio Feature Server (separate repo — feeds audio.* drivers over OSC :9000)
cd ~/Documents/GitHub/Realtime_PyAudio_FFT
uv run audio-server --open    # auto-connects to a running engine mid-flight

# Tauri control shell (single-process since 2026-07-12 — engine runs in-process)
# No separate render-core build needed: the shell links render-core as a library.
cd wzrd-app
pnpm install                  # one-off
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
# One process, three windows: webview UI + engine output + native preview overlay.
# WZRD_DISPLAY=<idx> puts the engine window borderless-fullscreen on that monitor.
# Runs the engine WITH OSC enabled; audio.* layers stay at defaults until the audio server is up.
# Add WZRD_AUDIO=1 (or `pnpm tauri dev -- -- --audio`) to auto-start the audio server too —
# spawns `uv run audio-server --open` in ~/Documents/GitHub/Realtime_PyAudio_FFT
# (override with WZRD_AUDIO_DIR), killed on shell close. Failed spawn = logged, non-fatal.
```

No linting, formatting, or CI pipelines are configured. No pytest — tests use a custom `test.py` with assertions on file existence and return types. Test assets live in `test_imgs/`, outputs go to `test_results/`.

## Architecture

**Two Python packages** defined in `pyproject.toml`:
- `wzrd/` — Core image/video processing library
- `wzrd_mcp/` — FastMCP server layer wrapping wzrd functions as tools

**One Rust crate** at the repo root:
- `render-core/` — Realtime additive projection-mapping engine (wgpu + winit). Now a `[lib] + [[bin]]` crate. The standalone binary (`render-core --scene scene.json`) consumes a layer pack produced by `wzrd.layerpack` and is the headless agent deployment target. **Primary engine context doc: `docs/reference/render-engine.md`** (contracts, current state; roadmap in `docs/TODO/render-engine-roadmap.md`). Phases 0–4.2 are landed, plus the 2026-07 performance/telemetry pass (occlusion-aware rendering, live `param.set` knob path, all telemetry channels emitting), roadmap §5.2–§5.5 (per-layer identity + `pick` selectors; engine-written `session.json` sidecar for operator state; operator masters brightness/speed/saturation/audioListen via `master.set`; descriptor-driven param metadata + live per-binding overrides via `param.set {binding, param}`), roadmap **§5.6 two-deck** (landed 2026-07-12 — design/live legs, `promote`/`pull` with bar-quantized crossfade, LIVE⇄DESIGN toggle as a **full control switch** (per-leg transport/masters/knobs/overrides; promote copies design's tuning live), shader pre-flight probe gating the design leg, design-draft autosave; full contract in `docs/reference/render-engine.md` §2.6. With a control surface the engine runs two legs and all authoring targets **design**; headless stays single-leg), and roadmap **§5.10 authoring MCP + §5.13 identity-sidecar engine slice** (landed 2026-07-22 — `wzrd_mcp/engine_tools.py` exposes the authoring slice (scoped `get_scene_context`, facet-bound writes, probe verdict + thumbnail in every author reply) to a local Claude Code session; engine side: design rev/epoch + sticky `changes` ring, per-connection actor (`hello`), `base_rev` CAS on `scene.load`, probe-deferred `effect.upsert`, `identity.json` groups/labels via `identity.setGroups`, webview reverse-sync + ADOPT button, agent mirror file; contract in `docs/reference/render-engine.md` §2.7/§2.2. The agent never touches live — `promote` stays a human UI act), and roadmap **§5.14 alignment layer** (landed 2026-08-12 — the final pass is an n-point output warp whose base is the 4-corner homography, baked into an offset LUT so per-frame cost is one texel read; engine-written `alignment.json` per project directory, `alignment.get/set/reset/setTestPattern` + a sticky `alignment` channel, and the Align route. Engine-wide, never per leg, never scene content — no MCP verb touches it; contract in `docs/reference/render-engine.md` §2.8):
  - **Phases 0–3:** pack loader, scene-aware compositor, homography pass, driver bus (clock + audio via OSC + ui-slider stub), built-in effect catalog (`tint`, `hueCycle`, `flash`, `wobble`), **inline + project-local user-WGSL effects with `naga`-validated swap-on-success hot-reload (D15)**. Project-local effects live in `<scene_dir>/effects/<name>/{shader.wgsl, descriptor.json}`. Audio features (`audio.band`, `audio.onset`) arrive over OSC from the standalone Realtime Audio Feature Server (separate Python repo at `~/Documents/GitHub/Realtime_PyAudio_FFT`) — bind defaults to `127.0.0.1:9000`, override with `--osc-addr`, skip with `--no-osc`. No `audio.rms`/`audio.bpm`/`audio.fft` in v1.
  - **Phase 4 (Tauri shell) + single-process collapse (landed 2026-07-12):** `wzrd-app/` wraps the engine for humans, running `render-core` **as an in-process library** on a dedicated render thread (one process; the old subprocess split is retired). Tauri commands call `rpc::dispatch` directly; the engine's WS server still binds `127.0.0.1:9123` serving the identical §3.11 method set to external MCP / remote clients. Engine modules: `lib.rs` (`pub fn run(cli)` + `hold_latency_critical_assertion`), `core.rs` (host-agnostic `Core` — GPU/plan/drivers/telemetry/WS, takes any `wgpu::SurfaceTarget`; `control_channel()` for embedding hosts; native-preview attach), `app.rs` (thin `WinitHost` for the standalone binary), `gpu.rs` (device/pipelines + `PreviewTarget` second swapchain), `rpc.rs` (JSON-RPC dispatch + EngineCommand queue + `wgsl.validate`), `ws.rs` (tungstenite accept-loop + per-conn IO thread), `telemetry.rs` (sticky bus + FPS percentiles + demand-gated JPEG preview readback).
  - **Headless agent path is unchanged** — the standalone `render-core` binary (winit-hosted); omit `--ws-addr` and the engine has no control surface, only file-watcher hot-reload.
  See `docs/reference/render-engine.md` (primary), `render-core/README.md`, and `wzrd-app/README.md`. Audio-server internals are documented in the `Realtime_PyAudio_FFT` repo itself.

**One Tauri app** at the repo root:
- `wzrd-app/` — single-process control shell (engine in-process since 2026-07-12; see `docs/reference/render-engine.md` §1b for the TauriHost residue). React + TypeScript + Vite + Tailwind webview running the §3.11 RPC surface through Tauri commands. Four routes in load-in order (⌘1–⌘4 keyboard switching):
  - **Prepare** (`src/routes/Prepare.tsx`) — surface canvas with mask overlays + Monaco editor (scene.json + per-effect WGSL tabs, inline naga squiggles via `wgsl.validate` IPC) + binding inspector with selector / effect / driver dropdowns.
  - **Align** (`src/routes/Align.tsx` + `components/WarpCanvas.tsx`) — the §2.8 alignment layer: drag corner/extra handles until the light lands on the surface, test patterns, background picker. Commits via `state/alignment.ts`, **not** `sceneCommit.ts` (alignment is not scene content; the engine owns `alignment.json`).
  - **Perform** — native preview hero (`NativePreview` — a borderless child window the engine blits the composite onto directly; lossless, full-rate) + audio band/onset visualizer + driver rack listing every driver-bound param with live values.
  - **Debug** — collapsible panels (connectivity, render stats, driver bus snapshot, hot-reload events history, log stream with level filter, raw pack/scene dump).
  Top status strip (OSC / Engine / FPS / Reload pills) is constant across all routes. All scene mutations (Monaco *and* structured editors) funnel through `src/state/sceneCommit.ts` (optimistic local state, engine push debounced 150 ms, disk write debounced 800 ms) — don't add a second commit path. Backend in `src-tauri/`: `engine.rs` (in-process TauriHost — engine window, render thread with `NSWindow.occlusionState` polling, telemetry fan-in, native preview child window + `preview_set_bounds`), `rpc.rs` (Tauri commands proxying every method — unchanged by the collapse), `lib.rs` (tee logger, App Nap opt-out, scene-path resolution; handles `cargo tauri dev`'s CWD switching; teardown centralized in `RunEvent::ExitRequested`). Scene path bound via `WZRD_SCENE` env or `--scene` arg.

### wzrd/ package

**Import strategy** (`__init__.py`): Eagerly imports `subtract_frame` and `utils`. All other modules (`darken`, `detect`, `align`, `islands`, `subtract_video`, `reproject`, `prepare_surface`) are lazy-imported via `__getattr__` to avoid loading heavy deps (scikit-learn, etc.) when not needed.

**No centralized config.** Each module defines its own `DEFAULT_*` constants. Do not introduce a config system.

**Key modules and their relationships:**
- `utils.py` — Shared primitives: aspect ratio handling, `extract_creature()`, image I/O helpers
- `subtract_frame.py` — Single-frame background subtraction; calls `utils.extract_creature()`
- `subtract_video.py` — Video pipeline using the same algorithm; expensive mask ops run in parallel via `ThreadPoolExecutor`, creature extraction stays sequential
- `darken.py` — Gradient-weighted histogram equalization in CIELAB space
- `detect.py` — Projection surface detection (color proximity + Otsu + convex hull)
- `align.py` — Feature matching (SIFT/AKAZE + RANSAC) with template matching fallback, ECC sub-pixel refinement
- `islands.py` — KMeans color clustering + connected components (requires scikit-learn)
- `reproject.py` — Layer compositing for island videos
- `prepare_surface.py` — Full surface prep pipeline (detect → align → darken). Returns dict with `'image'` (PIL Image) and optional `'video'` (alignment aid)
- `layerpack.py` — Authors the offline ↔ runtime layer pack (`pack.json` + `masks/` + `references/`) consumed by the realtime `render-core` engine. CLI: `python -m wzrd.layerpack <masks_dir> --surface ... --tags tags.json -o pack/`

### wzrd_mcp/ package

MCP server that wraps wzrd functions as tools. **Local-first:** tool outputs
land in a project folder on this machine and every tool returns absolute local
paths (and accepts them as inputs), so steps chain path-to-path and a project
folder is directly loadable by the engine.

**Key files:**
- `server.py` — FastMCP instance, tool registry, config loading
- `project.py` — Project folders under `projects/` (gitignored; `WZRD_PROJECTS_ROOT` moves the root, `WZRD_PROJECT` picks one). Reserve destinations with `output_path`/`output_dir` *before* running work so libraries write straight into the project (sidecar files like alignment-aid videos follow for free); `store`/`download_async` bring in files produced elsewhere. **The layout is a render-core scene directory on purpose** — `scene.json` + `effects/` + `packs/<pack>/` — which is why `build_layerpack` output loads as-is
- `project_tools.py` — the `set_project` tool (session-level active project; step one of any content-pipeline session)
- `tools.py` — Tool wrappers for core wzrd functions (subtract, detect, align, darken, prepare_surface, islands, reproject, layerpack, simulate_view, texture_flow)
- `fal_tools.py` — FAL API integrations (kling, nano_banana_pro). Results download into the project; **local inputs are uploaded to FAL's own CDN just in time** (`_to_fal_url`) since FAL's workers can't read this disk — the only outbound file copy left
- `engine_tools.py` — §5.10 authoring tools against the realtime engine's WS (`ws://127.0.0.1:9123`); local-only (needs `.[engine]` → `websockets`, absent from the Modal image), default-off in `server.py`, on in local `tools_config.json`. Contract: `docs/reference/render-engine.md` §2.7
- `file_io.py` — Input resolution (URL → download, base64 decode, local path). Its S3 upload half is **dormant** — no tool calls it, `boto3` is lazily imported behind the optional `s3` extra; kept for a future self-owned bucket
- `_log.py` — Terminal logging with colors/timestamps; `@logged_tool` decorator auto-logs tool invocations. **All output goes to stderr via `_emit`** — under the `--stdio` transport stdout carries JSON-RPC, so a `print()` to stdout anywhere in a tool breaks the protocol
- `auth.py` / `modal_app.py` — remote-deployment surface, **parked**: a container's project folder is ephemeral scratch, so the cloud path needs a durable sink before it means anything again
- `tools_config.json` — Per-tool activation toggles (tools can be disabled here)

**Tool registration pattern:** Tools use `@mcp.tool()` for MCP registration stacked with `@logged_tool` for automatic parameter logging.

**Flow:** local paths (or URLs) in → call wzrd function → write into the active project → return local paths.

**`texture_flow` is disabled** — it's the one remaining Eden dependency (a ComfyUI deployment in the `edenartlab` Modal workspace). Don't re-enable it without reading `docs/TODO/eden-decoupling.md`.

**Environment variables (`.env` loaded automatically, see `.env.example`):**
- `FAL_KEY` — Required for FAL tools (kling, nano_banana_pro). The only key a local session needs
- `WZRD_PROJECT`, `WZRD_PROJECTS_ROOT` — default project name / root folder for outputs
- `WZRD_API_KEY`, `WZRD_MODAL_SECRET` — remote deployment only
- `AWS_*`, `CLOUDFRONT_URL` — read by nothing on the local path; they only wake the dormant S3 code

## Extras

- **opencv-python pinned** to <4.11.0 (missing macOS wheels above that).
- `wzrd/deprecated/` contains old implementations — not part of the package.
- `archive/` folder contains pre-engine planning docs — historical only.
