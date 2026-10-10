# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WZRD is an additive projection-mapping system: only changing regions get projected — unchanged areas stay dark and merge with physical reality. Three cooperating parts:

- **Python offline toolkit** (`wzrd/` + `wzrd_mcp/`) — extracts/segments surfaces from photos and authors layer packs.
- **Rust realtime engine** (`render-core/`) — wgpu compositor that plays layer packs on a projector; headless-agent-drivable via `scene.json` + WGSL hot-reload.
- **Tauri control shell** (`wzrd-app/`) — React UI wrapping the engine for humans.

**Keep the Python toolkit and `render-core/` Linux/NVIDIA-portable.** A future Linux-native UI is planned on top of those two, so new work in them must not acquire a macOS dependency: wgpu stays on `Backends::PRIMARY` with `Features::empty()`/`Limits::default()` (no Metal-only features or formats) — the **one exception** is `TEXTURE_COMPRESSION_BC`, requested only when the adapter offers it, for HAP clip playback (Apple Silicon and desktop NVIDIA/AMD/Intel all have it; without it HAP clips refuse per binding and everything else runs), platform calls stay `#[cfg]`-gated behind a working non-macOS branch, ffmpeg codecs stay parameters (never hardcoded `*_videotoolbox`), and the headless path must not assume AppKit. `wzrd-app/` is exempt — it may be fully Mac-native (AppKit child windows, `NSWindow.occlusionState`, App Nap opt-out); its non-macOS fallbacks are untested sketches, so don't treat them as a portability guarantee. Detail on where the current lines fall: `docs/reference/render-engine.md` §1b.

## Docs (`docs/`)

- `docs/reference/render-engine.md` — **primary engine context doc**: current system state, contracts (scene.json incl. clip bindings, layer pack incl. `source_box`, RPC/telemetry, WGSL incl. the `image` input / `sample_image` / HAP decode in §2.4), invariants from the 2026-07 performance pass, working agreements, rejected approaches. Read before touching `render-core/` or `wzrd-app/`.
- `docs/reference/user_design_spec.md` — the product north star; every engine trade-off answers to it.
- `docs/reference/operator-ui.md` — the `wzrd-app` webview: rooms, the deck/masters/structure rules, layer mixer (`@opacity`/`@mute`/`@solo` overrides), modulation ranges, looks + auto-pilot, project switcher + engine restart, clip grid, browser bridge (debug shell port), traps. Open when touching anything in `wzrd-app/src/` or the shell's `projects.rs` / `clips.rs` / `devport.rs`.
- `docs/reference/textureflow.md` — the TextureFlow GPU endpoint on Modal (`deploy/textureflow/`, app `wzrd-textureflow`, env `wzrd`): render contract, pinned stack, weights volume, input traps, GPU-snapshot fast boot, the MCP client, timings. Open when touching `deploy/textureflow/` or the `texture_flow` / `textureflow_warm` MCP tools.
- `docs/reference/mcp-server.md` — the MCP server: progressive disclosure (instructions → skills → references), the `server.tool()` registration stack, skills format and how they're served (tool / resources / prompts), `tools_config.json` gate, file map, how to add a tool or skill. Open when touching anything in `wzrd_mcp/` or its skills.
- `docs/reference/hosted-models.md` — which hosted models the MCP's `nano_banana` (Google direct) / `kling_*` (FAL) tools call and why (the 2026-10 blind shootout that put Nano Banana 2.1 in), Google backend traps (Vertex draft images, client config), prices, and how to re-run the comparisons (`scripts/fal_shootout.py`, `scripts/google_nb_probe.py`). Open when touching `wzrd_mcp/google_tools.py` / `fal_tools.py` or swapping a model.
- `docs/reference/resources.md` — running index of external links (repos, papers, tools) worth knowing about, each with a short "why we'd care" note. Nothing in it is a dependency; drop new links here.
- `docs/TODO/render-engine-roadmap.md` — engine **designs** keyed §5.x (§5.11 reliability/probation, §5.5 WGSL `#import`, §5.15 effect chains, §5.16 feedback + `post`, §5.17 TAKE transitions, §5.7 layer object…). It does not schedule them — the gig plan and the toolbox plan do. Landed sections are drained to pointers into the reference doc.
- `docs/TODO/gig-2026-11-live-video.md` — **the current priority**: six-week build plan for a Nov 2026 live gig (engine video, aspect fix + locked project canvas, provenance sidecars, `check_frames`, project switcher, region × clip-slot grid, seed clip effects, probation window, style recipes): settled decisions D1–D16, cross-lane contracts C1–C6 (project file, sidecars, clip files, `source_box`, clip binding, `library/` layout) and hand-off work packages (P*/E*/A*/F1/S1) with lanes and gates. Open it before starting any of those.
- `docs/TODO/vj-toolbox.md` — the **post-gig** build order toward a full VJ instrument: global effects library, effect chains + feedback + post, TAKE transitions, in-app generation (the app as a client of the local MCP server), the Site setup room, MIDI — as packages T0–T6 with dependencies. Don't start it before the gig plan is done.
- `docs/TODO/backlog.md` — small low-priority leftovers with no plan of their own (own `FAL_KEY`, per-call `project=`, image draft tier, reviving the parked cloud MCP deploy, modernizing the TextureFlow stack). Check it before adding a one-off TODO elsewhere.
- `docs/TODO/midi-controller.md` — hardware MIDI controller spec: `midir` bridge, control registry with stable ids + aliases, reusable profiles + per-project overlay, setup walkthrough, warn-don't-crash resolution. App-only; optional gig package after the clip grid, else post-gig.
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

# TextureFlow GPU endpoint on Modal (every call needs --env wzrd; see docs/reference/textureflow.md)
uv run modal deploy --env wzrd deploy/textureflow/app.py
uv run modal run --env wzrd deploy/textureflow/app.py::smoke    # one cheap render -> deploy/textureflow/test_results/smoke/
# NEVER run the full ::matrix without Xander's explicit permission (57 GPU renders, real money); use ::smoke or a narrow --only subset.
uv run modal run --env wzrd deploy/textureflow/app.py::warm --minutes 120   # show-day: keep one GPU warm (0 = cool down)

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
# WZRD_SCENE is optional: without it the app opens the most recent project; switch
# projects / restart the engine from the TopBar project menu.
# Test projects with HAP clips: uv run python wzrd-app/scripts/fixture_projects.py
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
- `render-core/` — Realtime additive projection-mapping engine (wgpu + winit). Now a `[lib] + [[bin]]` crate. The standalone binary (`render-core --scene scene.json`) consumes a layer pack produced by `wzrd.layerpack` and is the headless agent deployment target. **Primary engine context doc: `docs/reference/render-engine.md`** (contracts, current state; roadmap in `docs/TODO/render-engine-roadmap.md`). Phases 0–4.2 are landed, plus the 2026-07 performance/telemetry pass (occlusion-aware rendering, live `param.set` knob path, all telemetry channels emitting), roadmap §5.2–§5.5 (per-layer identity + `pick` selectors; engine-written `session.json` sidecar for operator state; operator masters brightness/speed/saturation/audioListen via `master.set`; descriptor-driven param metadata + live per-binding overrides via `param.set {binding, param}`), roadmap **§5.6 two-deck** (landed 2026-07-12 — design/live legs, `promote`/`pull` with bar-quantized crossfade, LIVE⇄DESIGN toggle as a **full control switch** (per-leg transport/masters/knobs/overrides; promote copies design's tuning live), shader pre-flight probe gating the design leg, design-draft autosave; full contract in `docs/reference/render-engine.md` §2.6. With a control surface the engine runs two legs and all authoring targets **design**; headless stays single-leg), and roadmap **§5.10 authoring MCP + §5.13 identity-sidecar engine slice** (landed 2026-07-22 — `wzrd_mcp/engine_tools.py` exposes the authoring slice (scoped `get_scene_context`, facet-bound writes, probe verdict + thumbnail in every author reply) to a local Claude Code session; engine side: design rev/epoch + sticky `changes` ring, per-connection actor (`hello`), `base_rev` CAS on `scene.load`, probe-deferred `effect.upsert`, `identity.json` groups/labels via `identity.setGroups`, webview reverse-sync + ADOPT button, agent mirror file; contract in `docs/reference/render-engine.md` §2.7/§2.2. The agent never touches live — `promote` stays a human UI act), and roadmap **§5.14 alignment layer** (landed 2026-08-12 — the final pass is an n-point output warp whose base is the 4-corner homography, baked into an offset LUT so per-frame cost is one texel read; engine-written `alignment.json` per project directory, `alignment.get/set/reset/setTestPattern` + a sticky `alignment` channel, and the Align route. Engine-wide, never per leg, never scene content — no MCP verb touches it; contract in `docs/reference/render-engine.md` §2.8), and roadmap **§5.9 video** (landed 2026-10-10 as gig package E1 — HAP clips as an effect `image` input read via `sample_image(uv)`, decoded in `video.rs` on a thread per stream, placed by the pack's `source_box`; contract `docs/reference/render-engine.md` §2.1/§2.4):
  - **Phases 0–3:** pack loader, scene-aware compositor, homography pass, driver bus (clock + audio via OSC + ui-slider stub), built-in effect catalog (`tint`, `hueCycle`, `flash`, `wobble`), **inline + project-local user-WGSL effects with `naga`-validated swap-on-success hot-reload (D15)**. Project-local effects live in `<scene_dir>/effects/<name>/{shader.wgsl, descriptor.json}`. Audio features (`audio.band`, `audio.onset`) arrive over OSC from the standalone Realtime Audio Feature Server (separate Python repo at `~/Documents/GitHub/Realtime_PyAudio_FFT`) — bind defaults to `127.0.0.1:9000`, override with `--osc-addr`, skip with `--no-osc`. No `audio.rms`/`audio.bpm`/`audio.fft` in v1.
  - **Phase 4 (Tauri shell) + single-process collapse (landed 2026-07-12):** `wzrd-app/` wraps the engine for humans, running `render-core` **as an in-process library** on a dedicated render thread (one process; the old subprocess split is retired). Tauri commands call `rpc::dispatch` directly; the engine's WS server still binds `127.0.0.1:9123` serving the identical §3.11 method set to external MCP / remote clients. Engine modules: `lib.rs` (`pub fn run(cli)` + `hold_latency_critical_assertion`), `core.rs` (host-agnostic `Core` — GPU/plan/drivers/telemetry/WS, takes any `wgpu::SurfaceTarget`; `control_channel()` for embedding hosts; native-preview attach), `app.rs` (thin `WinitHost` for the standalone binary), `gpu.rs` (device/pipelines + `PreviewTarget` second swapchain), `rpc.rs` (JSON-RPC dispatch + EngineCommand queue + `wgsl.validate`), `ws.rs` (tungstenite accept-loop + per-conn IO thread), `telemetry.rs` (sticky bus + FPS percentiles + demand-gated JPEG preview readback).
  - **Headless agent path is unchanged** — the standalone `render-core` binary (winit-hosted); omit `--ws-addr` and the engine has no control surface, only file-watcher hot-reload.
  See `docs/reference/render-engine.md` (primary), `render-core/README.md`, and `wzrd-app/README.md`. Audio-server internals are documented in the `Realtime_PyAudio_FFT` repo itself.

**One Tauri app** at the repo root:
- `wzrd-app/` — single-process control shell (engine in-process; TauriHost residue in `docs/reference/render-engine.md` §1b). React + TypeScript + Vite + Tailwind, a Resolume-style VJ desk: rooms **Prepare** (surface map, layers, effect cards, dials, bottom drawer = region × clip-slot grid or Monaco code), **Align** (§2.8 warp), **Perform** (monitor, DESIGN|LIVE deck + TAKE, looks bank + auto-pilot, layer mixer + compact clip grid, dials, masters) and **Debug** (⌘1–⌘4); a TopBar project menu switches projects / restarts the engine. Map, rules and traps: **`docs/reference/operator-ui.md`**. Two rules bite most: every operator action lives in `src/state/actions.ts` (control-leg writes are *throttled*, masters write both legs), and every scene mutation funnels through `src/state/sceneCommit.ts` (optimistic, engine push debounced, disk only after engine accept) — don't add a second commit path. Alignment commits via `state/alignment.ts`, never `sceneCommit` (not scene content). Backend in `src-tauri/`: `engine.rs` (TauriHost: `EngineWindows` created once + one `EngineHandle` per engine instance, render thread with `NSWindow.occlusionState` polling, telemetry fan-in, native preview child window), `projects.rs` (project list, per-machine `settings.json` recents, re-root = shutdown + boot on the same windows), `clips.rs` (clip grid file commands), `rpc.rs` (Tauri commands proxying engine methods + looks file I/O), `devport.rs` (debug-only HTTP twin of the shell commands for the browser bridge), `lib.rs` (tee logger, App Nap opt-out, scene-path resolution, swappable `Session`, teardown in `RunEvent::ExitRequested`). Scene path via `WZRD_SCENE` env or `--scene`, else the most recent project. The same UI also runs in a plain browser at `localhost:1420` during `pnpm tauri dev` (engine WS + the shell's debug port behind Vite's `/__shell` proxy) — use it for scripted UI testing.

**One Modal deploy** at `deploy/textureflow/` — the TextureFlow ComfyUI GPU endpoint, self-contained (imports nothing from `wzrd/`/`wzrd_mcp/`). Every Modal CLI call for it needs `--env wzrd`; details in `docs/reference/textureflow.md`.

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

FastMCP 4 / MCP SDK 2.x. **Server map, registration stack, skills and the
operator gate: `docs/reference/mcp-server.md`.** What bites in most sessions:

- **Register every tool with `server.tool(title=, tags=, …hints)`**, never a bare `@mcp.tool` — it applies the timeout from `tools_config.json`, text-only results, MCP annotations and `logged_tool` (logging + error wrapping), so tool bodies have no try/except/log boilerplate. Every tool needs a `tools_config.json` entry.
- **Craft goes in skills, contracts in docstrings.** Workflows, prompt formulas and recipes live in `wzrd_mcp/skills/<name>/SKILL.md` (loaded on demand via `load_skill` / `skill://` resources / prompts); the server `instructions` carry only physics, the file contract and the generated skills index. Skills name tools, so renaming a tool means grepping `skills/` too.
- `project.py` — Project folders under `projects/` (gitignored; `WZRD_PROJECTS_ROOT` moves the root, `WZRD_PROJECT` picks one). Reserve destinations with `output_path`/`output_dir` *before* running work so libraries write straight into the project (sidecar files like alignment-aid videos follow for free); `store`/`download_async` bring in files produced elsewhere. **The layout is a render-core scene directory on purpose** — `scene.json` + `effects/` + `packs/<pack>/` — which is why `build_layerpack` output loads as-is.
- **All logging goes to stderr via `_log._emit`** — under the `--stdio` transport stdout carries JSON-RPC, so a `print()` to stdout anywhere in a tool breaks the protocol.
- `fal_tools.py` uploads local inputs to FAL's own CDN just in time (FAL's workers can't read this disk); `google_tools.py` sends refs inline. `file_io.py`'s S3 half, `auth.py` and `modal_app.py` are **parked** (a container's project folder is ephemeral scratch).

**Flow:** local paths (or URLs) in → call wzrd function → write into the active project → return local paths.

**`texture_flow` renders on our own Modal app** `wzrd-textureflow` (env `wzrd`): local paths are sent as bytes, URLs pass through, only caller-set args go over the wire (server owns the api.yaml defaults), mp4s land in the project's `generated/`. No Eden, no S3. A redeploy of `deploy/textureflow/` discards its GPU snapshot, so the next 1–2 cold starts take 2–3 min — prime it before a show (`textureflow_warm`). Contract: `docs/reference/textureflow.md`.

**Environment variables (`.env` loaded automatically, see `.env.example`):**
- `GEMINI_API_KEY` — Required for `nano_banana` (Google Gemini API). `VERTEX_API_KEY` (+ optional `GOOGLE_CLOUD_PROJECT`) is the fallback when it's unset
- `FAL_KEY` — Required for the kling tools
- `WZRD_PROJECT`, `WZRD_PROJECTS_ROOT` — default project name / root folder for outputs
- `WZRD_API_KEY`, `WZRD_MODAL_SECRET` — remote deployment only
- `AWS_*`, `CLOUDFRONT_URL` — read by nothing on the local path; they only wake the dormant S3 code

## Extras

- **opencv-python pinned** to <4.11.0 (missing macOS wheels above that).
- `wzrd/deprecated/` contains old implementations — not part of the package.
- `archive/` folder contains pre-engine planning docs — historical only.
