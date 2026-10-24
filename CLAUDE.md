# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WZRD is an additive projection-mapping system: only changing regions get projected — unchanged areas stay dark and merge with physical reality. Three cooperating parts:

- **Python offline toolkit** (`wzrd/` + `wzrd_mcp/`) — extracts/segments surfaces from photos and authors layer packs.
- **Rust realtime engine** (`render-core/`) — wgpu compositor that plays layer packs on a projector; headless-agent-drivable via `scene.json` + WGSL hot-reload.
- **Tauri control shell** (`wzrd-app/`) — React UI wrapping the engine for humans.

**Keep the Python toolkit and `render-core/` Linux/NVIDIA-portable.** A future Linux-native UI is planned on top of those two, so new work in them must not acquire a macOS dependency: wgpu stays on `Backends::PRIMARY` with `Features::empty()`/`Limits::default()` (no Metal-only features or formats) — **two capability-gated exceptions**, each requested only when the adapter offers it: `TEXTURE_COMPRESSION_BC` for HAP clip playback (Apple Silicon and desktop NVIDIA/AMD/Intel all have it; without it HAP clips refuse per binding and everything else runs), and `TIMESTAMP_QUERY` for the perf monitor's per-pass GPU timing (also only when `--perf on` and `--gpu-timing` ≠ 0; without it `perf.gpu` is null) — platform calls stay `#[cfg]`-gated behind a working non-macOS branch, ffmpeg codecs stay parameters (never hardcoded `*_videotoolbox`), and the headless path must not assume AppKit. `wzrd-app/` is exempt — it may be fully Mac-native (AppKit child windows, `NSWindow.occlusionState`, App Nap opt-out); its non-macOS fallbacks are untested sketches, so don't treat them as a portability guarantee. Detail on where the current lines fall: `docs/reference/render-engine.md` §1b.

## Docs (`docs/`)

- `docs/reference/render-engine.md` — **primary engine context doc**: current system state, contracts (scene.json incl. clip bindings, layer pack incl. `source_box`, RPC/telemetry, WGSL incl. the `image` input / `sample_image` / HAP decode in §2.4; the native preview underlay + the Studio monitor's `preview.file` in §1b), invariants from the 2026-07 performance pass, working agreements, rejected approaches. Read before touching `render-core/` or `wzrd-app/`.
- `docs/reference/user_design_spec.md` — the product north star; every engine trade-off answers to it.
- `docs/reference/operator-ui.md` — the `wzrd-app` webview: rooms (incl. **Studio** + the hub client: spawn/attach, lamp, jobs pill, read-only `library.db`, media URLs), the deck/masters/structure rules, layer mixer (`@opacity`/`@mute`/`@solo` overrides), modulation ranges, looks + auto-pilot, project switcher + engine restart + Save as / Relink missing, the projector OUTPUT switch (output window hidden at launch, display targets, flips/fit), clip decks (`decks.json`), the Studio Render panel, the show-setup wizard (`components/setup/`, `state/setup.ts`), browser bridge (debug shell port), traps. Open when touching anything in `wzrd-app/src/` or the shell's `projects.rs` / `output.rs` / `decks.rs` / `fork.rs` / `relink.rs` / `devport.rs` / `hub.rs` / `library.rs`.
- `docs/reference/textureflow.md` — the WZRD side of the TextureFlow GPU endpoint: which Modal app the MCP calls (live `textureflow`, fallback `wzrd-textureflow`, env `wzrd` — both deployed from the separate `textureflow-lab` repo, which owns all endpoint code, contract and traps), the MCP client incl. the per-call `backend="modal"|"local"` switch (Xander's desktop GPU, $0, no fallback), the spec-driven tool (`texture_flow`'s args + docs generated from the endpoint's published `spec()`: cache → committed snapshot `wzrd_mcp/textureflow_spec.json`, `tf_spec.py`), show-day warmth, draft → pick → finish (the server's `request_id` in each render record + filename, finishes routed to the draft's backend, Studio's Upscale). Open when touching the `texture_flow` / `textureflow_warm` MCP tools, a TextureFlow arg, or switching apps.
- `docs/reference/mcp-server.md` — the MCP server: progressive disclosure (instructions → skills → references), the `server.tool()` registration stack, skills format and how they're served (tool / resources / prompts), `tools_config.json` gate, the project folder + locked canvas (`project.json`) and why `project=` is passed per call, the Studio hub (`hub.lock`, idle exit, the app's `/api/*`), the media library (`~/WZRD/media`, `library.db`, the embedded `wzrd` tag, render records, indexing), jobs + the daily spend cap, playable HAP clips beside their source, the style recipe library (`library/recipes/`, `WZRD_RECIPES_ROOT`), the show folder map + named assets (`assets.json`, `asset:<name>` args), prompt templates (`library/prompts/`), soft delete (trash, 30 min), `rectify_surface`, file map, how to add a tool or skill. Open when touching anything in `wzrd_mcp/` or its skills.
- `docs/reference/hosted-models.md` — which hosted models the MCP's `nano_banana` (Google direct) / `kling_*` (FAL) tools call and why (the 2026-10 blind shootout that put Nano Banana 2.1 in), Google backend traps (Vertex draft images, client config), prices, and how to re-run the comparisons (`scripts/fal_shootout.py`, `scripts/google_nb_probe.py`). Open when touching `wzrd_mcp/google_tools.py` / `fal_tools.py` or swapping a model.
- `docs/reference/effects-library.md` — the global clip-effects library (`library/effects/<name>/{shader.wgsl, descriptor.json, preview.jpg}`, gig C6/F1): install by `cp -R` into a project's `effects/`, the 15-effect catalog with each one's audio dial, authoring conventions (one `fx_in` sampling site, additive-safe premultiplied output, clip-box space, the strobe safety cap), and how to re-run the gates (`scripts/fx_gate.py`; why not the 15 fps `preview` channel). Open before writing or changing a clip effect.
- `docs/reference/resources.md` — running index of external links (repos, papers, tools) worth knowing about, each with a short "why we'd care" note. Nothing in it is a dependency; drop new links here.
- `docs/TODO/render-engine-roadmap.md` — engine **designs** keyed §5.x (§5.11 reliability/probation, §5.5 WGSL `#import`, §5.15 effect chains, §5.16 feedback + `post`, §5.17 TAKE transitions, §5.7 layer object…). It does not schedule them — the gig plan and the toolbox plan do. Landed sections are drained to pointers into the reference doc.
- `docs/TODO/gig-2026-11-live-video.md` — **the current priority**: six-week build plan for a Nov 2026 live gig (engine video, aspect fix + locked project canvas, provenance sidecars, `check_frames`, project switcher, region × clip-slot grid, seed clip effects, style recipes; the probation window E2 and MIDI are **on hold** since 2026-10-12): settled decisions D1–D16, cross-lane contracts C1–C6 (project file, sidecars, clip files, `source_box`, clip binding, `library/` layout) and hand-off work packages (P*/E*/A*/F1/S1) with lanes and gates. Its sidecars (C2), clip files (C3), per-project cap and grid file are being replaced by the Studio plan below. Open it before starting any of those.
- `docs/TODO/studio-content-library.md` — **agreed 2026-10-15, building now, all of it by 7 Nov (H1 by ≈ 24 Oct)**: the content side of the app. The MCP server becomes a long-lived **hub** (app-spawned, `127.0.0.1:8787`, owns `~/WZRD/media/.wzrd/library.db` + render jobs + spend), renders land in a media library (`~/WZRD/media` + extra roots indexed in place), each file carries its identity in an embedded `wzrd` metadata tag (sidecars retired), opt-in 0–9 scores + tags, a **Studio** room (browser, preview monitor, Render panel, jobs), and Resolume-style clip **decks** per show (`decks.json` replacing `clipgrid.json`, library files referenced in place, Relink by fingerprint). Every package has landed (pointers into the reference docs); open: A2's on-screen check (AFK block B9). Open it before touching render outputs, sidecars, `provenance.py`, the active project, the spend cap or the clip decks.
- `docs/TODO/show-setup.md` — **landed 2026-10-16**: the show-setup wizard (photos → surface → segmentation → regions → pack), named project assets, prompt templates, soft delete. Holds its decisions G1–G11 and the open on-site checks (Project white on a real projector, Finder drops, detection on a real night photo). Open it before changing the wizard, assets, the trash or the folder map.
- `docs/TODO/vj-tool-feature-survey.md` — **research survey, awaiting Xander's review (2026-10-12)**: UI/UX/control features of Resolume, TouchDesigner, MadMapper, HeavyM, Millumin, VDMX, Synesthesia, each tagged with WZRD status (have / planned / gap / rejected) and relevance, plus 14 top picks. Input for future build specs — check it before designing an operator-UI or control feature, so we borrow instead of reinventing.
- `docs/TODO/vj-toolbox.md` — the **post-gig** build order toward a full VJ instrument: global effects library, effect chains + feedback + post, TAKE transitions, the Site setup room, MIDI (in-app generation moved to the Studio plan) — as packages T0–T6 with dependencies, plus a phase-7 sketch of the in-app agent chat (host the Claude Code session in Foreman, `~/Documents/GitHub/agent_frameworks/foreman`; build only the WZRD panel). Don't start it before the gig plan is done.
- `docs/TODO/perf-monitor.md` — systematic perf measurement. **PM1–PM3 shipped** (the always-on CPU frame profiler `render-core/src/perf.rs`, sampled per-pass GPU timestamps `render-core/src/gpu_timing.rs`, `perf` / `perf_spike` channels — `docs/reference/render-engine.md` §2.9; the windowless `render-core --bench` + `scripts/perf_gate.py` regression gate, scenarios in `render-core/perf/scenarios/`, baselines in `render-core/perf/baselines/` — §2.10); still design: per-binding `perf.profile` + the graphed panel (PM4); open: the gate's budgets to confirm. Open it before touching frame timing, telemetry `fps`/`frame_stats`, or the E2 probation window; a new cost path adds a `perf::Stage` (CPU) or `perf::GpuPass` (a new render pass) **and** adds or extends a bench scenario.
- `docs/TODO/native-preview-underlay.md` — the native preview **underlay landed 2026-10-12** (a Metal view beneath a transparent webview in Perform/Prepare/Align; current state in `render-engine.md` §1b and `operator-ui.md`); what's left is U7: soak it through a gig, then delete the `WZRD_PREVIEW=window` child-window fallback. Open it before removing that fallback.
- `docs/TODO/afk-test-session.md` — the **one-sitting screen-test runbook**: every test that needs the engine window in front with nobody at the Mac (fps gates, 60 fps screen-recorded flash checks, window screenshots), as self-contained blocks run in order when Xander announces an AFK window. Append a block there (template at the bottom) instead of asking for your own session; open it when you're the runner.
- `.claude/skills/afk-session/` — **how to run an unattended session** (Xander AFK / "the screen is yours" / a perf sweep): hold the screen, auditor + worktree-implementer fan-out with a build gate and GPU lock, interleaved A/B instead of a gate run on a loaded machine, on-screen measurement recipes + traps, helper scripts (`ab.py`, `appcpu.sh`, `spikes.py`). Claude loads the `afk-session` skill; any other agent reads its `SKILL.md`.
- `docs/TODO/backlog.md` — small low-priority leftovers with no plan of their own (own `FAL_KEY`, image draft tier, reviving the parked cloud MCP deploy, native DXV playback). Check it before adding a one-off TODO elsewhere.
- `docs/TODO/midi-controller.md` — hardware MIDI controller spec: `midir` bridge, control registry with stable ids + aliases, reusable profiles + per-project overlay, setup walkthrough, warn-don't-crash resolution. App-only. **On hold (2026-10-12)** — don't start it until Xander lifts the hold.
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
python -m wzrd_mcp            # the Studio hub: HTTP on 127.0.0.1:8787 (/mcp + /api), one per machine (hub.lock), exits after 1 h idle
                              # (normally the app spawns it; Claude Code: claude mcp add --transport http wzrd http://127.0.0.1:8787/mcp)
python -m wzrd_mcp --no-debug # disable verbose tool logging
python -m wzrd_mcp --stdio    # stdio transport (Claude Desktop spawns this itself)

# TextureFlow GPU endpoint: code, deploys and tests live in the textureflow-lab repo (pushing its main redeploys it).
# From WZRD, show-day warmth is the MCP tool textureflow_warm(minutes) (0 = cool down).

# Realtime engine (headless agent path — no UI process)
# Toolchain: rust-toolchain.toml pins rustup 1.99.1 for both crates (Homebrew cargo ignores it and would
# rebuild the whole shared target/). In a shell that predates rustup: export PATH=$HOME/.cargo/bin:$PATH
# NOTE: there is NO `render-core` on PATH. `cargo build` only writes the binary
# into render-core/target/. Launch via `cargo run --` or ./target/release/render-core.
cd render-core
python ../test.py layerpack   # one-off: build the example pack the sample scene points at
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc
# --no-osc runs without the audio server (clocks tick, audio.* drivers → 0).
# Drop it and start the audio server to make audio-reactive layers (flash/drift) respond.
# --windowed → drop for borderless fullscreen; add `--display 1` to pick a monitor.

# Perf gate (repo root; windowless — runs from any shell, ~2 min; see the rule below)
cd ..
uv run scripts/perf_gate.py                         # build release, run the bench suite 3x, table vs baseline, exit 1 on FAIL
uv run scripts/perf_gate.py --update-baseline --reason "..."   # a cost you added on purpose (+ a scenario that covers it)
render-core/target/release/render-core --bench render-core/perf/scenarios/smoke.json --bench-out /tmp/r.json   # one scenario

# Realtime Audio Feature Server (separate repo — feeds audio.* drivers over OSC :9000)
cd ~/Documents/GitHub/Realtime_PyAudio_FFT
uv run audio-server --open    # auto-connects to a running engine mid-flight

# Tauri control shell (single-process since 2026-07-12 — engine runs in-process)
# No separate render-core build needed: the shell links render-core as a library.
cd wzrd-app
pnpm install                  # one-off
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
# One process, two windows: webview UI (native preview view underneath) + engine output.
# WZRD_PREVIEW=window restores the old preview child window (stage fallback until soaked).
# WZRD_SCENE is optional: without it the app opens the most recent project; switch
# projects / restart the engine from the TopBar project menu.
# Test projects with HAP clips: uv run python wzrd-app/scripts/fixture_projects.py
# Output window starts OFF (TopBar OUTPUT pill turns it on). WZRD_OUTPUT=window | display:<idx>
# (or WZRD_DISPLAY=<idx>) turns it on at boot — AFK/fps runs need this.
# Runs the engine WITH OSC enabled; audio.* layers stay at defaults until the audio server is up.
# Add WZRD_AUDIO=1 (or `pnpm tauri dev -- -- --audio`) to auto-start the audio server too —
# spawns `uv run audio-server --open` in ~/Documents/GitHub/Realtime_PyAudio_FFT
# (override with WZRD_AUDIO_DIR), killed on shell close. Failed spawn = logged, non-fatal.
```

**Perf gate rule:** after any change under `render-core/` or `wzrd-app/src-tauri/` — or a UI change that subscribes to a high-rate telemetry channel — run `uv run scripts/perf_gate.py` and paste its table into your summary. A FAIL is not done. A cost you added on purpose: add/extend a scenario in `render-core/perf/scenarios/` that covers it, then `--update-baseline --reason "…"`. Details: `docs/reference/render-engine.md` §2.10.

No linting, formatting, or CI pipelines are configured. No pytest — tests use a custom `test.py` with assertions on file existence and return types. Test assets live in `test_imgs/`, outputs go to `test_results/`.

## Architecture

**Two Python packages** defined in `pyproject.toml`:
- `wzrd/` — Core image/video processing library
- `wzrd_mcp/` — FastMCP server layer wrapping wzrd functions as tools

**One Rust crate** at the repo root:
- `render-core/` — Realtime additive projection-mapping engine (wgpu + winit). Now a `[lib] + [[bin]]` crate. The standalone binary (`render-core --scene scene.json`) consumes a layer pack produced by `wzrd.layerpack` and is the headless agent deployment target. **Primary engine context doc: `docs/reference/render-engine.md`** (contracts, current state; roadmap in `docs/TODO/render-engine-roadmap.md`). Phases 0–4.2 are landed, plus the 2026-07 performance/telemetry pass (occlusion-aware rendering, live `param.set` knob path, all telemetry channels emitting), roadmap §5.2–§5.5 (per-layer identity + `pick` selectors; engine-written `session.json` sidecar for operator state; operator masters brightness/speed/saturation/audioListen via `master.set`; descriptor-driven param metadata + live per-binding overrides via `param.set {binding, param}`), roadmap **§5.6 two-deck** (landed 2026-07-12 — design/live legs, `promote`/`pull` with bar-quantized crossfade, LIVE⇄DESIGN toggle as a **full control switch** (per-leg transport/masters/knobs/overrides; promote copies design's tuning live), shader pre-flight probe gating the design leg, design-draft autosave; full contract in `docs/reference/render-engine.md` §2.6. With a control surface the engine runs two legs and all authoring targets **design**; headless stays single-leg), and roadmap **§5.10 authoring MCP + §5.13 identity-sidecar engine slice** (landed 2026-07-22 — `wzrd_mcp/engine_tools.py` exposes the authoring slice (scoped `get_scene_context`, facet-bound writes, probe verdict + thumbnail in every author reply) to a local Claude Code session; engine side: design rev/epoch + sticky `changes` ring, per-connection actor (`hello`), `base_rev` CAS on `scene.load`, probe-deferred `effect.upsert`, `identity.json` groups/labels via `identity.setGroups`, webview reverse-sync + ADOPT button, agent mirror file; contract in `docs/reference/render-engine.md` §2.7/§2.2. The agent never touches live — `promote` stays a human UI act), and roadmap **§5.14 alignment layer** (landed 2026-08-12 — the final pass is an n-point output warp whose base is the 4-corner homography, baked into an offset LUT so per-frame cost is one texel read; engine-written `alignment.json` per project directory, `alignment.get/set/reset/setTestPattern` + a sticky `alignment` channel, and the Align route. Engine-wide, never per leg, never scene content — no MCP verb touches it; contract in `docs/reference/render-engine.md` §2.8), and roadmap **§5.9 video** (landed 2026-10-10 as gig package E1 — HAP clips as an effect `image` input read via `sample_image(uv)`, decoded in `video.rs` on a thread per stream, placed by the pack's `source_box`; contract `docs/reference/render-engine.md` §2.1/§2.4):
  - **Phases 0–3:** pack loader, scene-aware compositor, homography pass, driver bus (clock + audio via OSC + ui-slider stub), built-in effect catalog (`tint`, `hueCycle`, `flash`, `wobble`), **inline + project-local user-WGSL effects with `naga`-validated swap-on-success hot-reload (D15)**. Project-local effects live in `<scene_dir>/effects/<name>/{shader.wgsl, descriptor.json}`. Audio features (`audio.band`, `audio.onset`) arrive over OSC from the standalone Realtime Audio Feature Server (separate Python repo at `~/Documents/GitHub/Realtime_PyAudio_FFT`) — bind defaults to `127.0.0.1:9000`, override with `--osc-addr`, skip with `--no-osc`. No `audio.rms`/`audio.bpm`/`audio.fft` in v1.
  - **Phase 4 (Tauri shell) + single-process collapse (landed 2026-07-12):** `wzrd-app/` wraps the engine for humans, running `render-core` **as an in-process library** on a dedicated render thread (one process; the old subprocess split is retired). Tauri commands call `rpc::dispatch` directly; the engine's WS server still binds `127.0.0.1:9123` serving the identical §3.11 method set to external MCP / remote clients. Engine modules: `lib.rs` (`pub fn run(cli)` + `hold_latency_critical_assertion`), `core.rs` (host-agnostic `Core` — GPU/plan/drivers/telemetry/WS, takes any `wgpu::SurfaceTarget`; `control_channel()` for embedding hosts; native-preview attach), `app.rs` (thin `WinitHost` for the standalone binary), `gpu.rs` (device/pipelines + `PreviewTarget` second swapchain), `rpc.rs` (JSON-RPC dispatch + EngineCommand queue + `wgsl.validate`), `ws.rs` (tungstenite accept-loop + per-conn IO thread), `telemetry.rs` (sticky bus + FPS percentiles + demand-gated, GPU-downscaled JPEG preview + `preview.snapshot` readbacks).
  - **Headless agent path is unchanged** — the standalone `render-core` binary (winit-hosted); omit `--ws-addr` and the engine has no control surface, only file-watcher hot-reload.
  See `docs/reference/render-engine.md` (primary), `render-core/README.md`, and `wzrd-app/README.md`. Audio-server internals are documented in the `Realtime_PyAudio_FFT` repo itself.

**One Tauri app** at the repo root:
- `wzrd-app/` — single-process control shell (engine in-process; TauriHost residue in `docs/reference/render-engine.md` §1b). React + TypeScript + Vite + Tailwind, a Resolume-style VJ desk: rooms **Studio** (⌘0: browse/label/render content against the hub, Render panel — `hub.rs` spawns/attaches it, `library.rs` reads `library.db` read-only), **Prepare** (⌘1: surface map, layers, effect cards, dials, bottom drawer = clip decks or Monaco code) and **Perform** (⌘2: monitor, DESIGN|LIVE deck + TAKE, looks strip + auto-pilot, Layers|Clips mixer, dials, masters) as tabs, with **Align** (⌘3, §2.8 warp) and **Debug** (⌘4) one click down in the ⚙ Setup menu, which also holds the per-project panel-layout lock (`ui/layoutSplit.tsx`, saved in `<project>/ui_layout.json`); a TopBar project menu switches projects / restarts the engine. Map, rules and traps: **`docs/reference/operator-ui.md`**. Two rules bite most: every operator action lives in `src/state/actions.ts` (control-leg writes are *throttled*, masters write both legs), and every scene mutation funnels through `src/state/sceneCommit.ts` (optimistic, engine push debounced, disk only after engine accept) — don't add a second commit path. Alignment commits via `state/alignment.ts`, never `sceneCommit` (not scene content). Backend in `src-tauri/`: `engine.rs` (TauriHost: `EngineWindows` created once + one `EngineHandle` per engine instance, render thread with `NSWindow.occlusionState` polling, telemetry fan-in (streaming channels only on webview demand, batched emits), preview host), `underlay.rs` (the native preview `NSView` under the transparent webview), `projects.rs` (project list, per-machine `settings.json` recents, re-root = shutdown + boot on the same windows), `decks.rs` (per-show clip decks `decks.json`: read + one-time migration, CAS write, fingerprints), `fork.rs` (project "Save as…"), `relink.rs` ("Relink missing": moved clips by render id → fingerprint → size walk), `hub.rs` (hub spawn/attach + lamp + `/api` proxy), `library.rs` (read-only `library.db` + media URL scope), `layout.rs` (`ui_layout.json` read/write), `rpc.rs` (Tauri commands proxying engine methods + looks file I/O), `devport.rs` (debug-only HTTP twin of the shell commands for the browser bridge, plus `webview_eval`/`debug_log` so agents can drive the app's own webview), `lib.rs` (tee logger, App Nap opt-out, scene-path resolution, swappable `Session`, teardown in `RunEvent::ExitRequested`). Scene path via `WZRD_SCENE` env or `--scene`, else the most recent project. The same UI also runs in a plain browser at `localhost:1420` during `pnpm tauri dev` (engine WS + the shell's debug port behind Vite's `/__shell` proxy) — use it for scripted UI testing.

**No Modal deploy lives here.** The TextureFlow GPU endpoint (live app `textureflow` + the ComfyUI-graph fallback `wzrd-textureflow`, env `wzrd`) is built and deployed from the `textureflow-lab` repo (`~/Documents/GitHub/Gigabrain/VJ/textureflow-lab`); WZRD only calls it by name. Details: `docs/reference/textureflow.md`.

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
- **Generators write to the media library, venue tools to the project.** `nano_banana`, `texture_flow`, `kling_*`, `to_hap`, `extract_frames`, `concat_videos` land in `~/WZRD/media` (`WZRD_MEDIA_ROOT`; the Studio target folder, else `inbox/`; `dest=` overrides) via `library.output_path`. **No per-chat project:** MCP 2026-07-28 is sessionless, so every project-resolving tool takes `project=` (omitted = the show open in the app); `set_project` only creates/opens a show and locks its canvas.
- `project.py` — Project folders under `projects/` (gitignored; `WZRD_PROJECTS_ROOT` moves the root, `WZRD_PROJECT` is the last-resort default). Reserve destinations with `output_path`/`output_dir` *before* running work so libraries write straight into the project (sidecar files like alignment-aid videos follow for free); `store`/`download_async` bring in files produced elsewhere. **The layout is a render-core scene directory on purpose** — `scene.json` + `effects/` + `packs/<pack>/` — which is why `build_layerpack` output loads as-is. A new project locks a **canvas** once (`project.json`, a Nano Banana native size): surface/region/pack tools refuse their own sizing args then, and every model-output → surface mapping undoes the model's crop (`wzrd.utils.uncrop_*`) — never a stretch.
- **Render records are automatic:** `logged_tool` gives every media file a call creates a render id + creation record, embedded in the file as the `wzrd` tag and stored in `library.db` (`provenance.record_outputs`); `.meta.json` sidecars are retired (still read as a fallback). Tool bodies never write records; they return a private `_provenance` dict (cost per output, region, `variant_of` for a HAP). Paid tools check the **daily, global** spend cap first (`provenance.check_spend`, summed from the `jobs` table). Tests: point `WZRD_MEDIA_ROOT` at a temp dir.
- **All logging goes to stderr via `_log._emit`** — under the `--stdio` transport stdout carries JSON-RPC, so a `print()` to stdout anywhere in a tool breaks the protocol.
- `fal_tools.py` uploads local inputs to FAL's own CDN just in time (FAL's workers can't read this disk); `google_tools.py` sends refs inline. `file_io.py`'s S3 half, `auth.py` and `modal_app.py` are **parked** (a container's project folder is ephemeral scratch).

**Flow:** local paths (or URLs) in → call wzrd function → write into the active project → return local paths.

**`texture_flow` renders on our own Modal app** `textureflow` (env `wzrd`), deployed from the `textureflow-lab` repo (`deploy.py`; its CI redeploys on every push to its `main`, so WZRD always runs the lab's latest pipeline; the old ComfyUI-graph app `wzrd-textureflow`, now in the lab's `comfyui_app/`, stays deployed as the fallback via `TEXTUREFLOW_APP=wzrd-textureflow`): local paths are sent as bytes, URLs pass through, only caller-set args go over the wire (the server owns defaults + validation; the tool's args and their docs are generated from the endpoint's published spec — never hand-write a TextureFlow arg), mp4s land in the media library (previews in `.wzrd/intermediates/`). No Eden, no S3. A redeploy discards the app's GPU snapshot, so the next 1–2 cold starts take 2–3 min — prime it before a show (`textureflow_warm`). Contract: `docs/reference/textureflow.md`.

**Environment variables (`.env` loaded automatically, see `.env.example`):**
- `GEMINI_API_KEY` — Required for `nano_banana` (Google Gemini API). `VERTEX_API_KEY` (+ optional `GOOGLE_CLOUD_PROJECT`) is the fallback when it's unset
- `FAL_KEY` — Required for the kling tools
- `WZRD_PROJECT`, `WZRD_PROJECTS_ROOT` — fallback project name / root folder for projects
- `WZRD_MEDIA_ROOT` — the media library home (default `~/WZRD/media`)
- `TEXTUREFLOW_BACKEND` (`modal` default | `local`), `TEXTUREFLOW_LOCAL_URL`, `TEXTUREFLOW_LOCAL_TOKEN` — `texture_flow`'s default backend and the desktop GPU job server (`docs/reference/textureflow.md`); `TEXTUREFLOW_APP` picks the Modal app
- `WZRD_API_KEY`, `WZRD_MODAL_SECRET` — remote deployment only
- `AWS_*`, `CLOUDFRONT_URL` — read by nothing on the local path; they only wake the dormant S3 code

## Extras

- **opencv-python pinned** to <4.11.0 (missing macOS wheels above that).
- `wzrd/deprecated/` contains old implementations — not part of the package.
- `archive/` folder contains pre-engine planning docs — historical only.
