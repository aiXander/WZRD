# Render engine — architecture reference

> **Primary context doc for coding agents working on the realtime engine**
> (`render-core/` + `wzrd-app/`). It describes the system as it exists after
> the 2026-07 performance/UX pass and the contracts you must not break. The
> structural roadmap lives in
> [../TODO/render-engine-roadmap.md](../TODO/render-engine-roadmap.md) —
> references of the form **§5.x** in this file point there. The north star
> for every trade-off is [user_design_spec.md](user_design_spec.md) — read it
> first. The one-sentence thesis: *the segmentation of a physical surface is
> the central scene primitive; light is added to named regions; the dark
> stays dark.*
>
> Parenthetical references of the form *(v1 §x)* point at the retired
> original build plan (phases 0–4.2, all landed; now in `docs/finished/`).
> They are historical rationale only — nothing in v1 is needed to work on
> the engine; everything load-bearing is inlined here.

### Target scenes (the effect model's north star)

Three scenes define what the engine must express **without writing per-scene
shader code** — the litmus test for the effect/driver/selector model:

- **Tree at night.** ~20 leaf clusters, 1 trunk, 1 ground, 1 sky. Leaves:
  slow per-cluster hue cycle through a palette; all-leaves white flash on
  bass kick; one random leaf "blooms" (radial gradient from centroid) every
  4 bars. Trunk: vertical scroll like sap flow. Ground: green ripple
  flood-fill from trunk base on kick.
- **Rock formation.** ~8 segmented rocks + background. Each rock independent
  low-frequency Perlin color drift. A specific rock "wakes up" on a MIDI note
  (bright pulse propagating across that rock's mask). Background plays a
  pre-rendered cloud HAP clip masked to its region.
- **Building facade.** Pillars, windows, doors, roof. Building motion
  (ripples, sway), characters appearing and walking across mask regions,
  story beats triggered by MIDI/OSC cues.

Same engine, same primitives — only the layer pack and bindings change. When
a proposed feature can't be justified against one of these, it's probably out
of scope.

---

## 1. System shape (as built)

**Single-process collapse landed 2026-07-12** (Steps 2–3; the plan doc is
retired to `docs/finished/` — everything durable is in §1b below). The operator app is **one process, three native windows**;
the engine runs as a library on an in-process render thread. The standalone
`render-core` binary remains a separate deployment target for headless
agents.

```
┌────────────────────────────┐   layerpack/ (pack.json + masks/)
│ OFFLINE — Python           │ ──────────────────────────────────────────┐
│ wzrd/ + wzrd_mcp/          │                                           ▼
│ photo→detect→align→darken  │      ┌──────────────────────────────────────────────────┐
│ →islands→layerpack         │      │ wzrd-app (Tauri, ONE process)                    │
└────────────────────────────┘      │  main thread: webview window (React UI) +        │
                                    │    native preview NSView *under* the webview     │
┌────────────────────────────┐      │  engine-render thread: render-core::Core         │
│ Audio Feature Server       │      │    - engine output window (wgpu, tao handle)     │
│ (~/GitHub/Realtime_PyAudio │ ────▶│    - mask atlas + pass plan + driver bus         │
│ _FFT, separate repo)       │ OSC  │    - file watcher hot-reload                     │
└────────────────────────────┘ :9000│    - native preview blit (same device/composite) │
                                    │  ws threads: JSON-RPC server ws://127.0.0.1:9123 │
┌────────────────────────────┐      │    for external MCP / remote operators           │
│ render-core binary         │      └──────────────────────────────────────────────────┘
│ (headless agent path,      │      Tauri commands → rpc::dispatch directly (no WS hop);
│ winit-hosted, unchanged)   │      the WS server runs the same dispatch for externals.
└────────────────────────────┘
```

### 1b. Collapse residue (Step 2–3, landed + spike-verified 2026-07-12)

- **`wzrd-app/src-tauri/src/engine.rs` is the TauriHost.**
  `EngineHandle::start_in_process` builds `Core::new`, creates the engine
  output window (tauri `WindowBuilder`, needs the `unstable` tauri
  feature), hands the window into `Core::init_gpu` (tauri windows are
  rwh-0.6, exactly as the 2026-07-10 spike predicted), and drives Core
  from an `engine-render` thread with the same per-frame contract
  `WinitHost` uses. `Core::control_channel()` exposes the WS server's own
  `RpcContext` + `EngineCommand` sender, so Tauri commands run
  `rpc::dispatch` directly — one dispatch path for local UI, remote WS,
  and the future MCP wrapper.
- **Occlusion is polled, not evented** — tao has no `Occluded` event. The
  render thread reads `NSWindow.occlusionState` (one objc msg_send) per
  frame for *both* the engine window and the window hosting the preview
  (the main window for the underlay), upholding §3.1 for both swapchains.
- **Render-thread rule (deadlock):** the render thread must never call a
  tauri window method that dispatches to the main thread (`inner_size`,
  `set_*`…) — sync Tauri commands run *on* the main thread and block on
  render-thread replies, so that's a deadlock. Sizes flow in via `Resized`
  events through shared state (`SizeState`).
- **Native preview — the underlay (2026-10-12, replacing the Step-3 child
  window).** `wzrd-app/src-tauri/src/underlay.rs` inserts a plain `NSView`
  with our own `CAMetalLayer` into the main window *directly below the
  `WKWebView`* (both are subviews of wry's `WryWebViewParent`) and turns
  the webview's background off (`drawsBackground` KVC — private, so no Mac
  App Store; the window background is set to the app `--bg`). The render
  thread blits `gpu::PreviewTarget` into it; every bit of HTML composites
  *over* the image, and clicks always reach the webview (in-window
  `NSView` hit-testing ignores alpha). React's `<NativeSlot>` measures its
  rect and pushes `preview_set_slot {x,y,width,height,visible,params}`;
  the backend places the view via `NSWindow.contentLayoutRect` (the
  webview's own frame spans the title bar under the full-size-content
  style — measuring from it lands the image one title bar too high) and
  forwards physical size + `PreviewParams` to the render thread, which
  caps the drawable at composite resolution (the layer's resize gravity
  scales it). Rooms differ only in `PreviewParams` (`core.rs`): Perform =
  the deck-toggle leg, each with its own leg's brightness/saturation masters (DESIGN was un-mastered until 2026-10-25, so Master/Sat looked dead while authoring);
  Prepare = design + the pack's surface photo; any view may set
  `surface` (0..1), which replaces the legacy additive `photo_weight` /
  `light_weight` with `projected_on_surface` in `final_pass.wgsl` — a first
  approximation meant to be refined: albedo ≈ photo ÷ its mean linear luma
  (`Gpu::photo_mean`, computed at upload), light × mix(1, albedo, amount) +
  photo × 0.25·amount (the wall under room light); Align = `program: true`, the
  exact projector final pass incl. the real §2.8 warp (LUT indexed by
  **uv**, `fs_preview`) and test pattern, `dim` 0.55. Same device,
  full-rate, zero readback. Ordering upholds §3.1: on show the view is
  unhidden before the engine presents; on hide the engine stops first.
  **Fallback:** `WZRD_PREVIEW=window` restores the old borderless child
  window ordered *above* the webview (Perform only; Prepare/Align then
  draw the JPEG channel) — the stage escape hatch until the underlay has
  soaked through a gig, then it is deleted
  ([native-preview-underlay.md](../TODO/native-preview-underlay.md) U7).
  HTML rules for the hole: [operator-ui.md](operator-ui.md) trap list.
- **Studio file preview (studio plan P1, 2026-10-15).** A third kind of
  view on the same underlay: `PreviewParams.file` (set by the shell for a
  slot with **id `"studio"`** — rooms are exclusive, so it shares the one
  `NSView`) shows the file opened by `preview.file {path|null}` instead of
  any leg: one HAP clip or PNG/JPEG still, opened with the same
  `video::ImageSource` the compositor uses (its own decode thread +
  staging ring, playhead = wall clock since open, looping), blitted
  contain-fit over black by `shaders/file_preview.wgsl` (`fs_file`, HAP Q
  YCoCg handled like `sample_image`). **It never touches a leg:**
  `design_demand` / `live_demand_occluded` ignore a file-mode slot, so the
  Studio monitor alone renders no composite. The source is dropped (decode
  thread exits on channel disconnect — `video::tests::
  dropping_a_clip_ends_its_decode_thread`) on `preview.file null` **and**
  whenever a *visible* non-file slot takes over (`Core::set_preview_visible`),
  so a webview that forgets to close it can't leak a thread. Not on a
  hidden non-file config: those arrive transiently (React StrictMode's dev
  double-mount, a page reload) on the preview-cfg channel, which the render
  thread drains *after* the command queue — dropping there closed a file
  `preview.file` had just opened (found 2026-10-15). Open cost lands on the render
  thread (first-frame decode + staging alloc, ≈ 3–8 ms once per pick —
  one frame, inside budget). Cost is charged to `Stage::PreviewPresent` /
  `GpuPass::Preview`; bench: `studio-preview` (§2.10), which attaches an
  **offscreen preview target** (`GpuContext::attach_preview_offscreen`) so
  a windowless run pays the same upload + blit. Code: `core.rs`
  (`FilePreview`, `cmd_preview_file`, `present_file_preview`), `gpu.rs`
  (`PreviewOut`, `PreviewDraw`, `render_preview_file`,
  `FilePreviewUniforms::contain`).
- **Crash containment (spike b):** the render loop runs under
  `catch_unwind`; a render-thread panic (incl. wgpu's fatal-by-default
  validation panic after device loss) kills the engine but not the
  webview; `engine:status {running:false, last_error}` is emitted. Session
  sidecar + scene file stay byte-identical (atomic writes). Measured
  relaunch-to-first-presented-frame: **~160–290 ms** (dev build) — the
  ≤20 s contract has ~two orders of magnitude of margin.
- **Clean teardown (spike a):** every exit path funnels through
  `RunEvent::ExitRequested` → `EngineHandle::shutdown()` → stop flag →
  join render thread (Core saves the session and drops the GPU context on
  that thread, *before* tauri destroys windows) → audio child killed.
- **Re-rooting (gig A1, 2026-10-10):** the engine window + preview host live
  in `EngineWindows` (created once, event handlers registered once); an
  `EngineHandle` is one engine on one scene path. A project switch or
  "Restart engine" is `shutdown()` (the teardown above) + a fresh
  `start_in_process` on the same windows: ~150–250 ms, measured. This
  needs every Core-owned listener to release its port on drop —
  `osc.rs::OscListener` and `ws.rs::ServerHandle` stop + join (WS
  connections are closed, so remote clients reconnect to the new
  engine). The shell's tee logger follows the current engine's bus itself
  (`telemetry::global_bus` is set-once). Project list, settings and the UI
  side: [operator-ui.md](operator-ui.md) "Projects + engine restart".
- **Output window is a manual switch** (2026-10-15): created hidden,
  pre-sized to the first pack, shown/placed by `output.rs` from the TopBar
  OUTPUT pill — see [operator-ui.md](operator-ui.md) "Projector output".
  Hidden = occluded = offscreen frames, so nothing here changed.
- **Env knobs:** `WZRD_OUTPUT=window|display:<idx>` (or the older
  `WZRD_DISPLAY=<idx>`) → output on at boot; `WZRD_SPIKE=panic|device_loss` → deliberate
  render-thread crash ~5 s after launch, once per process (test hook,
  never in a show).
- **Known warts:** (1) after an engine crash, SIGTERM is a no-op (the
  signal-hook flag's poller is dead) — quit via Cmd+Q/window close, which
  still works; (2) the `NativePreview` placeholder div assumes a
  height-constrained slot when honouring pack aspect; (3) `engine.rs`
  spawns one short-lived helper thread per *queued* RPC so the request
  timeout holds even if the render thread wedges — inline methods dispatch
  on the caller.

- **The two GPU-feature exceptions, both capability-gated.**
  (1) Gig plan D2/E1b: `TEXTURE_COMPRESSION_BC` is requested **iff the
  adapter offers it** (`gpu::GpuContext::bc_supported`) — HAP clips upload
  as BC1/BC3. Apple Silicon and desktop NVIDIA/AMD/Intel all offer it;
  without it the engine runs unchanged and every HAP clip fails to open
  with a named per-binding warning (stills still work). (2) Perf-monitor
  PM2: `TIMESTAMP_QUERY` (`gpu_timing::GpuTimer::FEATURES`) iff the perf
  monitor wants GPU timing (`--perf on` and `--gpu-timing` ≠ 0) **and** the
  adapter offers it (`GpuContext::timestamps_supported`); without it
  `perf.gpu` is `null` and nothing else changes. `_INSIDE_ENCODERS` /
  `_INSIDE_PASSES` are deliberately not requested (§2.9 says why). Everything
  else stays `Features::empty()` / `Limits::default()`. Video decode is pure Rust (hand-written MOV reader +
  the `snap` crate) — no ffmpeg link, no VideoToolbox.

**The headless agent path is sacred.** `render-core --scene foo.json` with no
`--ws-addr` runs with no control surface at all: an agent writes `scene.json`
+ `effects/*.wgsl`, the file watcher hot-reloads, the projector updates.
Every feature must keep working on this path; the GUI is ergonomic sugar,
never load-bearing (design spec: "if an authoring path requires the GUI to
exist — find the headless path first").

### Load-bearing decisions carried forward from v1

Still committed, condensed (historical rationale in the retired v1 plan):

| # | Decision |
|---|---|
| D1 | Native Rust + wgpu render core; the browser never renders the projector output. |
| D3/D4 | Layer pack is the offline↔runtime contract; masks live in one `Texture2DArray<R8>` (256-slice cap). |
| D7 | Selectors (`id`/`tag`/`group`/`all`) over hard-coded indices; layer ids are semantic and survive re-segmentation. |
| D9 | Calibration = an in-engine **n-point warp** as the final pass, whose base is the 4-corner homography (§2.8, 2026-08-12). Lives in `alignment.json`; both legacy matrix fields (`session.projectorCalibration`, `scene.projectorCalibration`) are boot-time migration sources only — read once, never written. |
| D13 | `scene.json` is canonical on disk and on the wire. No TS DSL on any critical path. |
| D14 | Python stays Python (offline segmentation, content generation via `wzrd_mcp`). |
| D15 | Effects are user-authored WGSL (inline or `effects/<name>/`), naga-validated, swap-on-success hot-reload. Built-ins are reference implementations, not a ceiling. |
| P4 | **Superseded 2026-07-12 by the single-process collapse.** The Tauri shell now hosts the engine **in-process** (Core on a render thread; §1b): Tauri commands call `rpc::dispatch` directly, the WS server on `127.0.0.1:9123` serves the identical §3.11 surface to external MCP / remote clients, and the headless `render-core` binary (winit-hosted) is unchanged. The subprocess split served Phases 4.1–4.2 and was retired after both runtime spikes passed (clean teardown; crash containment + relaunch-with-restore ≪ 20 s). Decision history: the retired plan in `docs/finished/`. |

Additive blending (`One + One` into an `Rgba16Float` composite, premultiplied
RGBA out of every effect) is the pixel-level embodiment of the thesis — do
not "fix" it to alpha blending.

### Module map

**`render-core/src/`**

| File | Owns |
|---|---|
| `lib.rs` | `Cli` + `run()` — shared entry for the binary and any embedder. |
| `main.rs` | CLI parse + **tee logger** (stderr + `log` telemetry channel via `telemetry::global_bus`). |
| `core.rs` | **Host-agnostic `Core`** (app-collapse Step 1, 2026-07-10): owns GPU context, pass plan, driver bus, OSC, effect registry, watcher, telemetry, WS server, session persistence (§5.3 restore-at-boot, debounced write, SIGTERM/SIGINT snapshot → `exit_requested()`), plus the policies every host must share — the §3.1 occlusion invariant and frame pacing. Two-stage init: `Core::new(&cli)` pre-window, `Core::init_gpu(impl Into<wgpu::SurfaceTarget>, w, h)` when the host has a window. Per-frame host contract: `poll_inbound()` → `pace_frame()` → `redraw()` \| `render_offscreen_frame()`. Step-2/3 additions: `control_channel()` (hands an embedding host the WS server's `RpcContext` + command sender — one dispatch path for every consumer), `attach_preview_surface`/`resize_preview_surface`/`set_preview_visible`/`set_preview_params` (native preview; host owns §3.1 for it; `PreviewParams` = what the slot shows, see §1b; the surface photo loads on attach), `cmd_preview_snapshot` (`preview.snapshot`), `spike_force_device_loss()` (crash-spike test hook). The command channel + `RpcContext` now always exist (headless just never sends). |
| `app.rs` | Thin `WinitHost` — window creation/fullscreen/display selection, winit event delegation into `Core`, exit decisions. The only winit-aware file besides `lib.rs`. |
| `gpu.rs` | wgpu device + output target — a window surface (takes any `SurfaceTarget` + explicit size — no windowing-crate dependency) or the §2.10 bench's offscreen texture (`Output`, `acquire_output`/`present`), mask atlas upload, per-leg composite targets (§2.6: live always, design in two-leg mode), pipeline cache, WGSL composer (`prelude + body + main`), final-pass pipeline (`encode_final`: live×design lerp by the promote mix → masters → §2.8 warp) and `WarpTarget` (the alignment LUT + its bake pipeline). Keeps `instance`/`adapter` so `PreviewTarget` (second swapchain) can attach at runtime: it runs `final_pass.wgsl`'s `fs_preview` entry — the projector math plus a preview-only group 1 (surface photo + dim/photo/light weights) — with a bind group per leg (dummy 1×1 LUT, warp off) and a **program** group on the real LUT (rebuilt in `resize_warp`); `render_preview()` self-heals `Lost`/`Outdated`; `encode_preview_snapshot` renders the same view offscreen. `Downscaler` (`downscale.wgsl`, 3×3 bilinear-tap box filter → `Rgba8UnormSrgb`) makes the JPEG thumbnail on the GPU. Its test module holds the GPU-backed guards worth knowing about: `baked_lut_matches_the_cpu_model`, `final_pass_samples_through_the_warp` and `program_preview_samples_through_the_warp_at_another_size` (all skip cleanly with no adapter), plus a naga parse+validate of the output shaders so a WGSL typo fails on the desk rather than at device creation. |
| `alignment.rs` | §2.8 alignment layer: `AlignmentDoc` (the `alignment.json` schema), `homography_from_corners` (Heckbert) + analytic 3×3 inverse, the Wendland C² kernel and its LU solve, `AlignmentState` (solved doc + dirty stamps, shared with the render thread and the RPC surface), `TestPattern`, bake-uniform packing, and the boot migration from the legacy calibration matrix. Talks to no GPU. |
| `perf.rs` | §2.9 always-on frame profiler (perf-monitor PM1): the fixed `Stage` enum, `Perf` recorder (span/lap/enter tokens into a `[u32; Stage::COUNT]`, 300-frame ring, spike capture, 2 Hz aggregate), the cross-thread `Shared` half in `RpcContext` (knob stamps, trigger, latest snapshot), and the process-global lock-free side probes: `ThreadCpu`, `ClipStats`, `count_telemetry`, plus `wgpu_resources` / `texture_bytes`. Also the PM2 `GpuPass` taxonomy, the `GpuFrame` ring + `gpu` aggregate, and `gpu_pass_recent` (E2's per-leg accessor). |
| `bench.rs` | §2.10 PM3 windowless bench host (`render-core --bench`): `BenchArgs`, the scenario format, scene staging, the timeline RPC worker + rate threads, OSC audio pin, telemetry subscribers, the report + machine info. |
| `gpu_timing.rs` | §2.9 PM2 `GpuTimer`: per-slot timestamp query sets, the 4-slot resolve/readback ring (resolve in the *next* frame's main encoder, `map_async`, collect when the callback fired), sampling stride, per-submit span + pass-charging math (`collect_frame`). Its tests include a real-device round trip and the `Rig` the `perf_overhead` measurement drives. |
| `probe.rs` | §5.6 shader pre-flight probe: `ProbeThresholds` (A/B atomics, sidecar-persisted), `ProbeSession` (half-res interleaved probe frames, overhead-calibrated full-res p95 prediction, pessimistic driver values), three-band verdict + JPEG thumbnail. |
| `compositor.rs` | `PassPlan` — scene → ordered layer passes; per-frame `tick(&mut)` (driver eval → uniforms, §5.2 pick re-rolls via `active` flags); `record_and_submit()` (present path), `render_offscreen()` (occluded path), `driver_rows()` (telemetry snapshot). Owns the stable hashes (`fnv1a`/`seed01`/`pick_choice`) behind layer_seed + pick determinism. |
| `drivers.rs` | Driver bus: `const`, `clock.*`, `audio.band/onset`, `ui.slider`; `SliderBank` (live knob values, written by `param.set`); `Masters` (§5.4 operator globals: brightness/speed/saturation/audioListen, atomics written inline by `master.set`); `Crossfade` (§5.4 engine-wide crossfade-time master — default promote fade in seconds, 0–30 s); `ParamOverrides` (§5.5 per-binding scalar override table); `Transport` (BPM clock — integrates `time += dt·speed` per frame so the speed master bends time instead of jumping it); `PickRate` (§5.2 transport-locked pick cadence). |
| `session.rs` | §5.3 session sidecar: `SessionFile` load/save (atomic temp+rename), `session_path` (`session.json` next to the scene), the shared dirty stamp the debounced write keys off. |
| `effects.rs` | Effect registry: built-ins (`tint`, `hueCycle`, `flash`, `wobble`, `videoClip`), project-local + inline WGSL, mtime-based rescan; the `image` input slot + the C5 `ImageParam` parse (§2.4). |
| `video.rs` | Image inputs (§2.4): `ImageSource` = a still (PNG/JPEG → `Rgba8UnormSrgb`) or a HAP clip stream — MOV sample-table reader, HAP frame decode (snappy / raw / chunked), per-stream decode thread + 4-slot `MAP_WRITE` staging ring, stateless playhead. Fixtures + decode tests: `render-core/tests/fixtures/*.mov`. |
| `scene.rs` | `scene.json` parser + selector resolution. |
| `pack.rs` | Layer-pack loader; per-slice uv geometry (`LayerGeom`) from the manifest or the mask bytes (§5.2); §2.2 identity sidecar (`IdentityFile` lenient load/merge, strict `apply_identity_delta`, atomic `save_identity`). |
| `osc.rs` | UDP OSC sink for `/audio/lmh` + `/audio/onset/*` → lock-free `AudioFeatures` atomics. `OscListener` drop stops + joins the thread, so the port is free on return (an in-process re-root rebinds it, §1b). |
| `rpc.rs` | JSON-RPC dispatch: inline read-only + live-tuning methods (`param.set` both forms, `master.set`, `hello`, `changes.list`) + queued `EngineCommand`s for mutations — an envelope `{cmd: Command, queued_at}` so the render thread can time `rpc.queue` (`scene.load` incl. `base_rev` CAS, `effect.*`, `identity.setGroups`, `session.save`); `wgsl.validate` with user-source line remapping; §2.7 `Actor` (per-connection identity) + `PackInfoCell` (swappable `pack.info` snapshot). |
| `ws.rs` | tungstenite server, thread-per-connection, telemetry fan-out. `ServerHandle` drop joins the accept loop and closes every connection — port free on return, clients reconnect to the next engine (§1b re-rooting). |
| `telemetry.rs` | `Bus` (bounded per-subscriber channels + sticky replay + `subscriber_count`), `FpsAccumulator` (honest fps + percentiles), `PreviewSampler` (the `preview` channel + `preview.snapshot` readbacks: GPU downscale to a 320-px sRGB thumbnail, ~230 KB non-blocking map, then a `preview-encode` worker thread JPEG-encodes — busy worker drops the frame; demand-gated on ≥1 `preview` subscriber; render-thread cost ≈ a memcpy, 0.2 ms p95 in dev), §2.7 `ChangeLog` (design rev + boot epoch + change ring), payload types. |
| `watch.rs` | notify-based watcher over the scene file + effects dir. |

**`wzrd-app/`** — `src-tauri/src/{lib,engine,rpc}.rs` (in-process TauriHost —
see §1b: engine window + render thread + telemetry fan-in + preview child
window; `rpc.rs` command wrappers are unchanged from the subprocess era) and
`src/` (React + Zustand; `state/sceneCommit.ts` debounced commit path,
`state/actions.ts` every operator action, `WarpCanvas` + `state/alignment.ts`
(§2.8 — its own commit path, deliberately not `sceneCommit`) +
`state/warpMath.ts`). **The UI's structure, rules and traps are mapped in
[operator-ui.md](operator-ui.md)** (rooms, deck/masters/structure rules,
layer mixer, modulation, looks, auto-pilot, browser bridge). §2.7 reverse-sync:
`App.tsx` handles the `changes` channel (non-`ui` actor → re-pull the
affected facet into the store only), and the TopBar's **ADOPT AGENT SCENE**
button (`sceneCommit.adoptAgentScene`) is the one path that persists
agent-authored scenes to `scene.json`.

---

## 2. Contracts (do not break)

### 2.1 `scene.json` (control contract)

```jsonc
{
  "version": 1,
  "pack": "../../test_results/layerpack/pack",   // resolved relative to the scene file
  "transport": { "bpm": 120 },                   // static tempo for clock.* phase — no live BPM tracking (dropped, §5.1)
  "bindings": [
    {
      "id": "primary_flash",                     // stable — the hot-reload diff key
      "select": { "tag": "leaves" },             // or { "id": ... } | { "group": ... } | { "all": true }; optional "pick" narrows to one member (below)
      "effect": "flash",                         // built-in | project-local name | { "inline": true, "wgsl": "...", "inputs": [...] }
      "params": {
        "envelope": { "driver": "audio.onset", "band": "low", "decay": 0.15 },
        "base":     { "driver": "ui.slider", "name": "flash_base", "default": 0.1 },
        "color":    "#ffffff"                    // hex string or [r,g,b(,a)] floats
      }
    }
  ],
  "post": []                                     // parsed, not yet consumed
}
```

`projectorCalibration` in scene.json is **dead** (§2.8, 2026-08-12):
alignment lives in `alignment.json`. A scene-level value is read once at
boot as a migration source, warned about, and ignored thereafter.

Drivers: `const(value)`, `clock.bars(n)`, `clock.beats(n)`, `clock.phase(rate)`,
`clock.time`, `audio.band(low|mid|high)`, `audio.onset(band, decay)`,
`ui.slider(name, default)`. Any driver object may add **`"range": [from,
to]`** (2026-10-09): the param then sweeps `from → to` as the driver moves
0 → 1 (`ScalarValue::Mapped` in `drivers.rs`) — the modulation depth the
operator UI's MOD routing writes ("bright 0.3 → 1.5 on the kick"); without
it the raw driver value goes straight in. The probe treats a mapped param
pessimistically as `max(from, to)`.

**Clock shapes — `"shape"` (2026-10-14).** The three periodic clocks
(`clock.bars`/`clock.beats`/`clock.phase`) take an optional `"shape"`
applied to their 0..1 cycle before `range` mapping (`drivers::Shape`):
`saw` (0→1 then jump; **the default** — no `shape` = pre-2026-10-14
behaviour, telemetry label unchanged), `ramp_down`, `pingpong` (triangle
0→1→0), `sine` (`0.5−0.5·cos 2πx`, 0→1→0 eased — the smoothest),
`square` (1 for the first half-cycle), `random` (a new random target per
cycle, smoothstep-eased between them; optional integer `"seed"`, default
0). One cycle = the whole shape (a 4-bar `pingpong` goes up for 2 bars and
down for 2), and every shape starts at 0 = `range[0]` on the cycle line.
Shapes are a pure function of transport position (random = hash of cycle
index + seed), so both legs, a TAKE and a re-run trace the same curve —
the `PickRate::cycle` rule. All stay in [0,1], so the probe's
`max(from, to)` holds. `shape` on any other driver is a parse error
(invert audio by swapping the `range` ends); an unknown name is a parse
error listing the valid ones. Labels append the shape (`clock.bars(4)
pingpong`). The operator UI writes `pingpong` for new clock modulations
(`wzrd-app/src/components/ModEditor.tsx`). No `audio.rms`/`audio.fft` scene drivers in v1
(the server emits `/audio/fft` but the engine doesn't consume it; rms doesn't
exist). The server also streams `/audio/bpm` — **deliberately ignored**
(live BPM tracking dropped 2026-07-11, roadmap §5.1): `transport.bpm` is a
static scene value, and live musical energy reaches scenes through
`audio.onset`/`audio.band`, not tempo — but the tempo itself is live-tunable
via `transport.set` (tap tempo, below). Duplicate/empty binding ids are load
errors.

**Clip bindings — `image` params (gig plan C5, landed 2026-10-10).** An
effect input of type `image` takes an object, never a driver:

```jsonc
{ "id": "clip_fabric_left", "select": { "id": "fabric_left" },
  "effect": "videoClip",                       // or any effect declaring an `image` input
  "params": { "image": { "clip": "generated/tf_0412.mov",  // relative to the scene dir
                         "placement": "region",            // "region" (default) | "canvas"
                         "rate": 1.0 },                    // playhead rate; negative = reverse
              "gain": 1.0 } }
```

- `clip` is a HAP `.mov` (Hap1 / Hap5 / HapY; dimensions multiples of 4)
  or a `.png`/`.jpg` still. Unknown keys / bad placement are load errors.
- **Placement:** `"region"` targets each resolved layer's `source_box`
  (§2.2; `bbox` fallback), `"canvas"` the whole frame. Either way the image
  is fit **uniformly, centred ("contain")**, never stretched — transparent
  outside (gig plan D8: a model's native-aspect render is close to, not
  exactly, its box; contain is the inverse of the model's cover-crop). One
  binding over N layers opens the clip once and places it in each layer's
  own box.
- **Playhead:** `(leg transport seconds × rate + offset) rem_euclid
  duration` — the speed master bends it, promote's clock copy keeps it
  continuous across TAKE, and the stream itself moves into live with the
  plan (it never reopens at TAKE). `offset` is 0 for a newly bound clip
  (stateless) and only moves when a rebuild changes `rate`, so the
  position holds and a rate slider bends playback instead of jumping it.
- **Clip continuity (2026-10-12) — no edit restarts a playing clip.**
  Every plan build says what it inherits (`compositor::Carry`):
  `Replace(old)` for a same-leg rebuild (any design `scene.load`, effect
  save/upsert, `identity.setGroups`, headless hot-reload) hands the very
  stream on for each (binding id, resolved file) the old plan played — no
  reopen, no texture swap, no black frame; `Mirror(other leg)` (pull,
  design's rebuild after promote) opens own streams (never shared across
  legs) at the other leg's playhead; `Fresh` only at boot. Streams are
  `Arc<Mutex<ImageSource>>`, shared only between a plan and the one
  replacing it (dropped right after the swap). A new `Carry` call site
  must pick the right variant — `Fresh` on a rebuild reintroduces the
  flicker.
- **A clip that fails to open is a warning, not a load error:** missing
  file, not HAP, no BC on this GPU → that binding draws nothing, the plan
  still swaps in, and the reply carries `warnings: [...]` (also the
  `hot_reload` message, which stays `ok: true`).

**`pick` selectors (§5.2, landed 2026-07-11).** Any selector may add
`"pick": { "mode": "random_each" | "random_static", "rate": { "driver":
"clock.bars", "n": 4 } }`. The binding still resolves (and builds passes
for) the full member set, but only one member draws: `random_each` re-picks
each time the rate clock wraps; `random_static` picks once at scene load. A
re-pick is an `active`-flag flip on the pass plan — zero rebuild, zero GPU
work. Rate drivers are restricted to `clock.bars`/`clock.beats`/
`clock.phase`, and the choice is a pure hash of (binding id, cycle count) —
no RNG state — so runs are deterministic and the §2.6 design leg
picks the same layer its promote will. Strictness: `random_each` without
`rate`, `random_static` with `rate`, and selectors setting more or fewer
than one of `all`/`id`/`tag`/`group` are all load errors.

**Timed/authored cues are external in v1 (v1 §3.7).** The driver bus is
signal-driven (reactive). Repeatable narrative timing — "at bar 128 start the
reveal" — leans on an external sequencer (Ableton/Bitwig/Reaper) firing
MIDI/OSC that the engine reacts to deterministically. An in-engine
cue/timeline editor is a Phase 6+ feature; because it's just another
`Driver<Event>` source, adding it later is not a re-plumb. External clock
sync (Link/MTC) is explicitly out of v1.

### 2.2 Layer pack (`pack.json` + `masks/` + `references/`)

Authored by `python -m wzrd.layerpack`. Layer `id`/`tags`/`group` are
semantic and stable across re-segmentation; mask paths are not. Loader
refuses unknown major versions.

**`source_box` (gig contract C4, landed 2026-10-10).** Optional per layer:
`"source_box": [x0, y0, x1, y1]`, pack pixel space, max-exclusive like
`bbox` — the region's box **padded to a Nano Banana native aspect** by
`extract_color_regions`, i.e. the frame that region's clips are rendered for
(from its `region_surface_image` crop). `build_layerpack` copies it from the
`islands.json` next to the masks, through the same uncrop/scale mapping as
the mask. It **may extend past the canvas** (negative or > size): the
padding is real clip frame that falls off-surface, not an error. Absent →
region placement falls back to `bbox`; empty/inverted → warned, `bbox`
(lenient on purpose: never refuse a show over it). `pack.info` serves it
per layer when present; how clips are placed in it: §2.1.

**Requirement this puts on the authoring tool (v1 §4.1):** stable ids are
only real if `wzrd.layerpack` can *re-import a new SAM2 segmentation onto an
existing pack's identity table* — overlap-match new blobs to existing ids,
surface splits/merges/new regions for human review — rather than emitting a
fresh pack each time. Without reconciliation, every re-shoot silently
invalidates every scene that targets the pack. Optional sidecar
`identity.json` can hold the map; the runtime contract stays "pack ids are
stable, period."

**Identity sidecar — engine slice landed (§5.13, 2026-07-22).**
`<pack_dir>/identity.json` (engine-written, gitignored) holds human-authored
`groups` (group id → member layer ids; same id as a pack group *replaces*
its membership) and `labels` (layer id → human label, overrides the
manifest's). `pack.rs` merges it over the manifest at load — leniently
(stale ids after a re-shoot warn and drop, never refuse boot) — and
`pack.info` + selector resolution serve the merged view. Writes go through
the queued `identity.setGroups` RPC (strict: unknown layer ids are
prescriptive errors), which persists atomically, refreshes the served
`pack.info` snapshot, and re-applies the design scene so group-targeting
bindings re-resolve. This is what lets surface-language commands ("the
*trunk*") resolve for the §2.7 authoring MCP. The UI multi-select half of
§5.13 still trails.

### 2.3 JSON-RPC surface (WS `--ws-addr`, mirrored 1:1 as Tauri commands)

| Method | Kind | Notes |
|---|---|---|
| `pack.info` | inline | Pack snapshot (layers, tags, groups, bbox, centroid, z) with the §2.2 identity sidecar merged in. Refreshed by `identity.setGroups` (served through a swappable `PackInfoCell`). |
| `hello {actor}` | inline | §2.7 — declare this connection's actor (`ui` \| `agent` \| `system`), once per session, never per call. WS connections default `agent`; Tauri direct dispatch is `ui`. Returns `{epoch, rev}`. |
| `changes.list {since_rev?, epoch?}` | inline | §2.7 — change-ring backfill (`{epoch, rev, entries}`). Epoch mismatch or ring wrap → full ring + explicit `note`, never a silently-partial diff. |
| `identity.setGroups {groups?, labels?}` | queued | §2.2/§5.13 — per-key delta into the identity sidecar (`null` removes); strict unknown-id errors listing valid ids. Persists, refreshes `pack.info`, re-resolves design selectors. Reply carries the new `pack` + `{epoch, rev}`. |
| `scene.getState {leg?}` | inline | Last-good scene JSON. §5.6: `leg: "design"` (default — reads follow design) \| `"live"`. Reply carries `{epoch, rev}` so read-modify-write callers can CAS, plus `scene_dir` (what clip paths resolve against — the MCP lists `<scene_dir>/generated/*.mov` from it). |
| `scene.load {json, base_rev?}` | queued | Parse + plan rebuild **on the design leg** (§2.6); new pipelines are probe-gated, so the reply can carry a `probe` report. `base_rev` (§2.7): rejected with "design moved to rev N … re-read and retry" when the design rev moved since the caller's read. On error the previous design plan keeps rendering; live is never touched. A clip that fails to open doesn't fail the load: the reply adds `warnings` (§2.1). Headless single-leg: applies to live as before. |
| `scene.reload` | queued | Re-read from disk (same design-leg routing). |
| `promote {fade_ms?, quantize?}` | queued | §2.6 — crossfade projector live→design over `fade_ms` (default: the §5.4 crossfade-time master, ms), then live adopts design's plan (pointer swap). `quantize: "bar"` (default) starts the fade on the next bar boundary; `"now"` immediately. Pending promote replaced by a newer one; **rejected while a fade is ramping**. |
| `pull` | queued | §2.6 — hard-copy live's scene back into design. Rejected mid-ramp; cancels a pending promote + any in-flight probe apply. |
| `preview.setSource {source}` | queued | §2.6 — `"live"` \| `"design"`: which composite the native preview blits (when the slot doesn't pin one). |
| `preview.file {path \| null}` | queued | Studio monitor (§1b "Studio file preview"): open one HAP `.mov` or PNG/JPEG still (absolute, or relative to the scene dir) for a slot whose params set `file`; `null` closes it. Never touches a leg. Reply `{path, kind: clip\|still, width, height, fps?, frames?, duration_s?, open_ms}`; a non-HAP / unreadable file is an error naming why. |
| `preview.snapshot {width?}` | queued | JPEG of **exactly what the native preview slot shows** (its `PreviewParams`: program view, photo blend, dim — or the file preview), rendered offscreen; default width = slot drawable capped at 1024. Reply `{encoding, width, height, data_b64, params, leg (`file` in Studio), file?, slot_px}`. Errors when no slot is visible. The agent's way to verify the underlay — webview screenshots can't see it. |
| `probe.setThresholds {a_ms, b_ms}` | inline | §2.6 probe bands, A < B, ms of predicted full-res p95. Sidecar-persisted (venue state). |
| `probe.getThresholds` | inline | Current `{a_ms, b_ms}`. |
| `effect.upsert {name, wgsl, descriptor?}` | queued | §2.7 verdict-in-reply: naga-validates **before** touching disk (bad WGSL → line-mapped diagnostics in the reply, nothing written), then writes `effects/<name>/`, rescans the registry (so the watcher echo of the self-write dedupes to nothing), and re-applies the design scene — the reply defers through the §2.6 probe and carries verdict + thumbnail + `{epoch, rev}` when the scene binds the effect. |
| `effect.remove {name}` | queued | |
| `effect.describe {name?}` | queued | §5.5 — input descriptors (type, default, `min`/`max`/`step`/`unit`/`widget`) + `bounds` (§2.4) for one effect, or the whole catalog (built-ins included) when `name` is omitted. |
| `wgsl.validate {source}` | inline | naga diagnostics remapped to user-source lines (drives Monaco squiggles). |
| `telemetry.channels` | inline | The list of live channel names (`telemetry::ALL_CHANNELS`). |
| `perf.snapshot` | inline | §2.9 — the latest `perf` aggregate (same object as the channel). Errors when started with `--perf off` or before the first aggregate (~0.5 s). |
| `perf.setTrigger {ms}` | inline | §2.9 — spike-capture threshold in ms of frame interval, (0, 10000]; default 2× budget = 33.3. Runtime-only. Returns `{ok, trigger_ms}`. |
| `param.set {name, value, leg?}` | **inline** | Writes the target leg's `SliderBank` (§2.6 — `leg` default `"design"`; the UI passes the deck toggle's leg); bound `ui.slider` params update next frame. **No rebuild, no disk write — this is the live knob path.** |
| `param.set {binding, param, value, leg?}` | **inline** | §5.5 override form — pins any *scalar* param on a binding (const or driver output alike) in the target leg's override table; `value: null` clears. Same zero-rebuild property; survives plan rebuilds as long as a scalar param with that name exists (the carry-forward rule). |
| `param.list {leg?}` | inline | `{ sliders, overrides, leg }` — the target leg's knob values + override table (design default). |
| `transport.set {bpm?, resync?, sync?, leg?}` | queued | Live tempo (20–400) with the beat grid held continuous (`Transport::beat_offset` — a tempo change never jumps bar phase); `resync` puts a bar line at "now" (nearest bar); `sync` snaps the design clock onto live's. `leg` omitted = both legs. Not persisted — scene.json's `transport.bpm` stays the cold-boot truth (the UI writes the tapped tempo there too). |
| `master.set {name, value, leg?}` | **inline** | §5.4 — `brightness` \| `speed` \| `saturation` \| `audioListen`, per leg since §2.6 (design default); plus `crossfade` (engine-wide crossfade-time master in seconds, 0–30 — `leg` ignored). Clamped, applied next frame, echoed on the sticky `masters` channel. Not reachable from scene.json by design. |
| `master.list` | inline | Both legs' masters: `{ live: {...}, design: {...} }`. |
| `session.save` | queued | §5.3 — write the session sidecar now (also happens debounced after master/knob changes and on SIGTERM/SIGINT/window close). |
| `alignment.get` | inline | §2.8 — the full document plus `{output: [w,h], points_max, test_pattern}` and the solved `weights` (read-only derived state, so a client can draw the real field without re-solving; never persisted, ignored on input). |
| `alignment.set {enabled?, background?, corners?, points?, flip_h?, flip_v?, fit?}` | **inline** | §2.8 partial merge, validated (exactly 4 corners; ≤64 points; finite coords; radius > 0; `fit` ∈ `stretch` \| `fit` \| `fill`). Rejects with a prescriptive message and the previous alignment keeps rendering. Two behaviours worth knowing: sending `corners` **alone** carries the extra handles with the content (the engine recomputes their dest positions), and a point with **no `anchor`** is anchored at the current field, so adding a handle doesn't move the image. No `base_rev` CAS — one human editor at a time, last-write-wins is right for a drag. |
| `alignment.reset` | inline | §2.8 — identity corners, no handles, black background. Leaves `enabled` and the output transform (`flip_h`/`flip_v`/`fit`) alone. |
| `alignment.setTestPattern {pattern}` | inline | §2.8 — `none` \| `grid` \| `border` \| `corners`, generated in **source** space so it warps with the content, or `white` (the whole raster, unwarped — the lit surface photo). Runtime-only, never persisted. |
| `telemetry.subscribe {channels}` / `.unsubscribe` | per-conn | |

Telemetry channels (all emitted by the engine as of the 2026-07 pass):

| Channel | Rate | Payload |
|---|---|---|
| `preview` | ~15 fps | 320px JPEG of a composite, base64 — the **design** composite in two-leg mode (§2.6 blanket leg rule), live headless. **Demand-gated**: captured only while ≥1 subscriber listens. GPU-downscaled (smooth, not point-sampled), encoded off the render thread. Consumers: MCP `get_preview`, the browser build, remote operators, and in the app only Perform's small CUED·DESIGN monitor, which *samples* it (one-shot grabs ≈1 Hz + on design change, full rate only on hover — [operator-ui.md](operator-ui.md)). The shell subscribes only on webview demand (`telemetry_demand`), never by default. Every in-app room view uses the native underlay (§1b). |
| `fps` | 2 Hz | Honest throughput (frames/wall-second) + p50 frame time. |
| `frame_stats` | 2 Hz | p50/p95/p99 + mask-slice / pipeline / pass counts (counts follow the **design** plan in two-leg mode — §2.6). |
| `drivers` | 10 Hz | Per binding·param: name, source description, live value, affects-count, `overridden` (§5.5 — value then reports the override). Rows come from the **design** plan in two-leg mode. Built only while ≥1 subscriber listens. |
| `audio` | 30 Hz | L/M/H bands + onset envelopes. Built only while ≥1 subscriber listens. |
| `transport` | 30 Hz | `{live: {bpm, beats}, design: {bpm, beats}}` — musical position for the UI's beat lamps and the auto-pilot's bar boundaries. Not sticky; built only while ≥1 subscriber listens. |
| `audio_freshness` | 1 Hz + edges | fresh/stale/down (sticky). |
| `connectivity` | 1 Hz | osc / file_watcher / ws status cells (sticky). |
| `hot_reload` | on event | target, ok, elapsed, message, **`probe`** (§2.6 — `{compiled, predicted_p95_ms, band, thumbnail_b64, verdicts[]}` when a pre-flight probe ran; sticky), plus the §2.7 correlation stamp `{epoch, rev, actor}` (for push consumers — author verdicts arrive in the RPC reply, never via sticky replay). |
| `changes` | on design mutation | §2.7 — one `ChangeEntry` `{epoch, rev, ts_ms, actor, facet, summary}` per design mutation (sticky; ring depth 32, backfill via `changes.list`). Facet map is static in the emit sites: scene applies → `bindings`, effect upserts/removes → `effects`, `identity.setGroups` → `layers`. The webview re-pulls the affected facet on any entry with `actor != "ui"`. |
| `log` | on event | Info+ engine log lines (via the tee logger). |
| `masters` | on `master.set`, promote/pull control copies + startup | §5.4/§2.6 — both legs plus the engine-wide crossfade master: `{live: {brightness, speed, saturation, audioListen}, design: {...}, crossfade}` (sticky). |
| `deck` | on promote/pull/preview transitions + ~10 Hz while ramping | §2.6 snapshot `{promote: idle\|pending\|ramping, mix, fade_ms, quantize, preview_source, two_leg}` (sticky). |
| `perf` | 2 Hz | §2.9 — the frame-profiler aggregate (headroom, per-stage p50/p95/max, per-pass GPU p50/p95 (`gpu`, null when GPU timing is off/unsupported), threads, clips, latency, process, telemetry bytes/s). Not sticky. |
| `perf_spike` | on a frame over the trigger (≤ 1 per 2 s) | §2.9 — that frame ± 5 neighbours with per-stage ms and what ran in each (sticky). |
| `alignment` | at boot, on every accepted mutation, on output resize | §2.8 — the whole document (incl. `flip_h`, `flip_v`, `fit`) plus `weights` and `{output, points_max, test_pattern, solve_ok}` (sticky, so the Align tab hydrates from `lastPayload` and an external camera script sees the UI's edits). |

### 2.4 Effect WGSL contract

User code implements `fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>`
(premultiplied RGBA, additive) with access to `state.*` (time, bar/beat
phase, bpm, audio bands/onsets, resolution), `f_param(N)` / `c_param(N)`
(8 scalar + 4 colour slots), and `sample_mask(uv)`. `fs_main` multiplies the
effect's output by `layer.opacity` (the operator layer fader, §2.5) — user
code never has to honour it. The engine composes
`shaders/effect_prelude.wgsl + body + shaders/effect_main.wgsl` and
naga-validates before pipeline creation; a bad save never blanks the
projector.

**Output bounds — light outside the region is clipped (perf P1,
2026-10-12).** The contract was always "scale your output by `mask`"; the
compositor now cashes it in: every pass of a **region-bounded** effect is
`set_scissor_rect`-ed to its layer's **mask support** — the box of every
*non-zero* mask texel (`LayerGeom::support_uv`, measured from the mask bytes
at pack load, never the manifest's `bbox`, which is thresholded at 128 and
misses soft edges), padded by one mask texel + one composite pixel for the
linear mask filter (`compositor::region_scissor`). Outside that box `mask`
is exactly 0, so a mask-scaled effect is **bit-identical** to the full-canvas
draw — pinned on all built-ins, the 15 library effects (stills + HAP Q) and
the 8 example effects by `region_scissor_is_bit_identical_for_every_shipped_effect`.
An all-zero mask draws nothing. The win is fragments: the gig layout (seven
small regions) shades ~0.7 M instead of 7.4 M pixels per leg. Which effects
are bounded (`EffectDef::bounds`, reported by `effect.describe` as
`"bounds"`):
- **Declared** wins: descriptor.json / inline spec `"bounds": "canvas"`
  (opt out — e.g. an effect that deliberately glows past its region) or
  `"region"` (opt in despite the inference below).
- **Inferred** otherwise (`Bounds::infer`, conservative): `canvas` when the
  body calls `sample_mask(…)` or touches `mask_atlas` (a displaced mask read
  lights pixels whose own mask is 0 — built-in `wobble` is canvas for this
  reason), or never uses its `mask` argument (naga ref count — a
  deliberate full-canvas effect); `region` otherwise. Comments don't count.
- An effect that *uses* `mask` but still emits light where it is 0 (e.g.
  `max(mask, 0.2)`) breaks the contract and is clipped — declare `canvas`.
- The §5.6 probe still shades the full canvas (a pessimistic cost
  estimate); TAKE fades never re-run layer passes, so they inherit the clip.

`sample_mask` samples with an explicit LOD (the atlas has one mip — same
result), so it is legal in non-uniform control flow and an effect may
early-out on `mask == 0` as long as it uses no derivatives
(`textureSample`, `dpdx`/`fwidth`) after the branch.

**Per-layer identity (§5.2, landed 2026-07-11).** When one binding resolves
to N layers, each pass carries its own identity, exposed as prelude
accessors: `layer_seed()` (stable [0,1) hash of the layer *id* — survives
re-segmentation because ids do, D7), `layer_index()` / `layer_count()`
(position within the binding's resolved selection, ascending slice order),
`layer_centroid()` (vec2, uv) and `layer_bbox()` (vec4 `(min_x, min_y,
max_x, max_y)`, uv, max-exclusive). `phase += layer_seed()` desynchronizes
N copies of any cycle; centroid/bbox anchor radial blooms and per-region uv
normalization. Geometry comes from the pack manifest's pixel-space
`bbox`/`centroid` (converted to uv at load) or is computed from the mask
bytes when the manifest omits them (`pack::geom_from_mask`, mirroring
`wzrd.layerpack._bbox_and_centroid`).

Note: `state.audio_*` / `state.onset_*` arrive **pre-scaled by the
audio-listen master** (§5.4) — user WGSL and drivers see the same values.

**The `image` input (gig plan E1, landed 2026-10-10).** A descriptor may
declare **one** input `{ "name": "image", "type": "image" }` (two is a load
error — the prelude has one image slot). It takes no `f_param` slot; the
binding's C5 param (§2.1) supplies the texture. WGSL reads it with
**`sample_image(uv) -> vec4<f32>`**: canvas uv in, premultiplied RGBA out,
transparent outside the placement rect and when nothing is bound. It
samples with an explicit LOD, so it is legal in non-uniform control flow and
at displaced uvs (the library clip effects lean on that — their catalog and
authoring conventions: [effects-library.md](effects-library.md)). Plumbing, for anyone
touching it:

- **Slot:** group 0, binding 4 (`texture_2d<f32>`) + 5 (sampler), in the
  one layer layout every pass and probe item uses (`gpu::layer_bind_group`
  is the only constructor); unbound passes get a 1×1 transparent dummy.
  `LayerParams` grew `image_box` (uv rect, already contain-fitted) and
  `image_meta` (bound flag, encoding, 1/box width, 1/box height — the
  reciprocals spare each `sample_image` tap a divide) → 176 bytes, pinned by
  `layer_params_layout_matches_wgsl`.
- **Colour:** stills and plain HAP/HAP Alpha use sRGB texture formats
  (decode to linear on sample); HAP Q's scaled YCoCg sits in a non-sRGB
  BC3 texture and `sample_image` converts + sRGB-decodes. Measured against
  the PNG path: all three within ~1/255.
- **A clip binds only while current:** `ImageSource::open` decodes the
  *playhead* frame synchronously into staging slot 0, and the stream's
  first `encode_upload` (which precedes every composite that samples it)
  copies it in, so a fresh clip is right from its first frame (no black,
  no frame-0 flash). `PassPlan::build` opens every clip it doesn't carry
  over **in parallel** before its binding loop (`open_new_images` →
  `ImageSource::open_many`, workers ≤ core count; warnings stay in binding
  order) — seven 1080p clips ≈ 3–5 ms instead of ≈ 8 ms serial. The floor
  is the four `mapped_at_creation` staging buffers per clip (~1.3 ms,
  fresh-memory faults); a staging pool would remove it. Each tick
  the plan checks `is_current` — the texture is within
  max(4 frames, 4 render-frames of playhead advance) of the playhead
  (`video::near_playhead`). A stream that sat idle (muted, picked out,
  unrendered design leg) fails it and stays unbound (transparent) for the
  frame or two the decoder needs — never a flash of the old frame; normal
  decode lag at high rates stays inside the tolerance and never blinks.
  The tick judges before that frame's upload, so `encode_uploads`
  re-checks after its copy: a clip the upload just made current gets its
  passes' bind flag patched into the plan's uniform slots (one
  `write_buffer`, transition frames only) and binds **that** frame — an
  unmute or TAKE shows the clip one frame sooner, never a stale one.
  The MCP's `get_preview` one-frame skip predates this and is harmless.
- **Built-in `videoClip`** = `sample_image(uv) × mask × gain` (gain 0–2).
- **Decode — the staging-ring invariant (canonical here since §5.9 landed):** one
  decode thread per (plan, binding) stream does all disk I/O + snappy +
  row padding straight into a mapped slot of a 4-slot `MAP_WRITE` ring; the
  render thread only encodes one `copy_buffer_to_texture` per new frame
  (`PassPlan::encode_uploads`, before `encode_composite`) and re-maps used
  slots after submit (`PassPlan::after_submit`). Decode is demand-driven:
  a leg that doesn't render (idle design, muted/picked-out bindings) asks
  for nothing and decodes nothing. **A frame is decoded once:** the
  decoder remembers it has served the current want's frame (sent, or in
  flight when the want arrived) and never re-decodes it when its slot
  comes back — the render thread only re-asks on a frame change, so a
  30 fps clip on a 60 Hz output used to decode every frame twice.
  Prefetch strides by the median of the last three observed
  frames-per-render-frame, so 8× speed still lands. Falling behind = the
  previous frame stays up. No stream sharing between legs (playheads are
  per leg); dropping a plan drops its streams and their threads exit on
  channel disconnect (never joined on the render thread).
- **Measured (M2, release, 1376×768 @ 30 fps, incl. two incompressible
  528 KB/frame clips and one HAP Q):** seven clips 60.0 fps p95 17.0 ms; a
  5 s TAKE swapping all seven (14 streams) 60.0 fps p95 ≤ 17.2 ms; six
  fabric + one ceiling clip in their regions 60.0 fps p95 17.1 ms, engine
  ≈ 17 % of one core. 120 design reloads: every old decode thread exits,
  RSS plateaus ~170 MB.

### 2.5 Session sidecar + masters (§5.3/§5.4, landed 2026-07-11)

`session.json`, engine-written, next to the scene file (deliberately per
*directory*, not per scene — calibration and masters describe the venue,
and every scene played from that directory shares the physical setup):

```jsonc
{
  "version": 1,
  "projectorCalibration": null,      // 3×3 row-major or null — moved out of scene.json
  "masters": { "brightness": 1.0, "speed": 1.0, "saturation": 1.0, "audioListen": 1.0 },
  "crossfade": 0.5,                  // §5.4 crossfade-time master (seconds) — engine-wide default promote fade
  "params": { "flash_base": 0.35 },  // SliderBank snapshot, by slider name
  "overrides": { "wobble_demo": { "amp": 0.05 } },  // §5.5 per-binding scalar overrides
  "probeThresholds": { "a_ms": 8.0, "b_ms": 14.0 }  // §2.6 probe bands — venue state
}
```

- **Scope rule (two sidecars now):** `scene.json` = what the surface *does*
  (AI + human authored); `session.json` = how *this venue, this night* is set
  (operator only); `alignment.json` (§2.8) = where the light physically
  lands. Both sidecars are engine-written, per-directory venue state, and
  gitignored; no authoring RPC and no scene reload path ever writes either.
  They are separate files because they have different writers (a camera
  script owns alignment content in a way it never owns knob state),
  different size classes (a dense correspondence field lands in alignment
  later) and different lifecycles (alignment is rewritten on every drag).
- **Write policy (engine-owned):** explicit `session.save`, debounced
  ~1.5 s after any `master.set`/`param.set`, and on SIGTERM/SIGINT/window
  close (`signal-hook` flag → `Core::poll_inbound` snapshots and requests a
  host exit — the §5.11 power-blink snapshot). Writes are atomic
  (temp + rename). `session.json` is gitignored.
- **`projectorCalibration` is dead** as of §2.8: the field survives in
  `SessionFile` purely as a boot-time migration source into
  `alignment.json`, is never written by the engine, and can be deleted from
  the struct once no project in the wild still carries one.
- **Masters application (per leg since §2.6):** the *live* leg's
  brightness/saturation ride the final pass uniform (composites
  stay un-mastered; the native preview applies the selected leg's values);
  each leg's speed multiplies **its own** transport's per-frame time
  integration (`time += dt·speed` — bends time, never jumps it, so picks
  and phases stay continuous); each leg's audioListen scales its `audio.*`
  reads (drivers *and* the `state.audio_*` uniform) toward 0. Master values
  are clamped: brightness/saturation 0–2, speed 0–8, audioListen 0–1. The
  sidecar persists the **live** leg's masters/knobs/overrides; both legs
  restore from it at boot.
- **Crossfade-time master (§5.4, engine-wide — *not* per leg):** a single
  operator global holding the **default `promote` fade** in seconds. A
  promote is one engine-wide action, so unlike brightness/speed/etc. it is
  neither duplicated per leg nor copied on promote/pull. Clamped 0–30 s
  (0 = CUT); a `promote` with no explicit `fade_ms` falls back to it. Set
  via `master.set {name:"crossfade"}` (the `leg` arg is ignored), persisted
  in the sidecar, carried on the `masters` channel's top-level `crossfade`.
  The deck panel's FADE fader (logarithmic, CUT at the bottom) drives it.
- **Layer mixer (2026-10-09):** the reserved override params `@opacity`
  (0–1), `@mute` and `@solo` (> 0.5 = on) on a binding are folded by
  `PassPlan::tick` into one per-pass `opacity` (`compositor.rs
  ::layer_opacity`; any soloed binding silences the un-soloed ones) and
  written into `LayerParams.opacity` (the old `_pad0` slot); 0 skips the
  draw. Riding the override table makes them per leg, zero-rebuild,
  sidecar-persisted and promote/pull-copied with no new RPC. The `@` prefix
  can't collide with a WGSL input name. This is the runtime half of roadmap
  §5.7 (no scene.json `layers:` block yet).
- The eventual "show file" (playlist + scenes + session) composes these;
  don't build the umbrella before the §5.6 auto-pilot playlist exists.

### 2.6 Two-deck architecture: design/live legs + promote/pull (§5.6, landed 2026-07-12)

The spec's "two legs, one deck". The engine holds **two `PassPlan` slots**
with per-leg composite targets: `live` drives the projector; `design` is the
AI/operator scratchpad, rendered offscreen only. Two-leg mode is on whenever
a control surface exists (`--ws-addr` — the Tauri host always sets it);
**headless-only runs collapse to a single live leg exactly as pre-two-deck**
(watcher binds live, no design composite allocated, no probe).

- **Blanket leg rule: every authoring and observability surface follows
  design.** `scene.load`/`scene.reload`, the file watcher, effect changes,
  the JPEG `preview` channel, `scene.getState` (default), the `drivers`
  channel and `frame_stats` counts all target/read the design leg. Only the
  projector, the preview toggle's LIVE position and Align's program view
  show live. Live is
  immutable except via `promote`.
- **Both legs tick every frame** (shared transport/audio bus — bar-quantized
  promotes land in sync; §5.2 picks are stateless hashes, so the legs pick
  identically). Design's **render passes are demand-gated** (`Core::
  design_demand`, evaluated *before* the tick): every frame when the native
  preview shows design (deck toggle on DESIGN, or a slot pinning
  `source: design` like Prepare — never the program view) or a promote is
  pending/ramping; when ≥1 `preview` subscriber is the **only** consumer,
  only on JPEG capture frames (`PreviewSampler::capture_due`: interval up,
  nothing in flight — ~15 Hz), unless the design plan draws a clip
  (`PassPlan::draws_clips`), which then renders every frame (a ~4-frame
  playhead stride makes the decoders' prefetch miss → stale frames). A
  stream capture only kicks off on a frame whose source composite was
  rendered (`capture_preview_jpeg(subscribed, fresh)`). An unrendered leg
  still ticks (clock, picks, faders, clip readiness) but skips its uniform
  upload and `after_submit`. An idle design leg costs zero GPU — in the app
  too, since the shell stopped subscribing to `preview` by default
  (2026-10-12).
- **Per-plan uniform buffer.** A `PassPlan` owns one uniform buffer: slot 0
  = `FrameStateGpu`, slot `1+k` = pass k's `LayerParamsGpu`, slots
  `min_uniform_buffer_offset_alignment` (256) apart; each pass's bind group
  points at its slot (built once at plan build). `tick` fills a reused CPU
  `Vec<u8>` (drawing passes only — picked-out / opacity-0 passes are
  skipped and rewritten the tick they draw again) and uploads it with
  **one `write_buffer` per rendered leg**: wgpu-core 22 allocates a staging
  buffer per `write_buffer` call, so per-pass writes cost O(passes)
  allocations (40-pass `layers-heavy`: `tick.live` 0.28 → 0.05 ms p95).
  The final-pass and native-preview uniforms are likewise re-uploaded only
  when they change (`Core::last_final_uniforms` / `last_preview_uniforms`).
- **The native preview's LIVE view shows the TAKE.** Its bind group pairs
  the live composite with design as the mix partner and `preview_uniforms`
  passes the promote `mix`, so Perform's monitor crossfades with the wall
  (mix 0 outside a promote → live only; same shader, no extra pass).
- **Promote = crossfade, then pointer swap.** The final pass
  samples both composites and lerps by a `mix` uniform ramped over
  `fade_ms` (wall time). On completion live **adopts design's already-built
  plan** (zero rebuild on the projector leg) and design rebuilds from the
  same scene JSON (pipelines cache-hit — buffers/bind groups only; clips
  reopen at live's playheads via `Carry::Mirror`). **Clip gate:** every
  ramp starts `gated` — `mix` holds at 0 until `PassPlan::clips_ready()`
  (every clip design draws is current), bounded by `PROMOTE_CLIP_GATE`
  (250 ms, then it takes anyway with a warning). Normally 1–2 frames after
  an idle design leg wakes; it is what keeps a cut TAKE from projecting a
  stale or unbound clip frame.
  Semantically a **copy, not an exchange**: design keeps its content, so
  "promote, push further, promote again" needs no `pull` between rounds.
  Re-entrancy (deterministic — the auto-pilot playlist will drive it):
  pending quantized promote → *replaced* by a newer promote; actively
  ramping → `promote`/`pull` *rejected*. `pull` also cancels a pending
  promote and any in-flight probe apply.
- **Pipeline cache is GC'd, never evicted by key.** User pipeline keys are
  **content-derived** (`file:<path>#<hash>`, `inline:<hash>`), so an edited
  shader gets a fresh cache slot while live keeps drawing the old one.
  After every plan swap the cache retains only keys referenced by *either*
  leg (+ built-ins, the probe calibration shader, in-flight probe keys) —
  this closes the old evict-by-key failure mode where a design edit could
  silently stop live layers drawing (passes skip on cache miss).
- **Pre-flight probe** (`probe.rs`): any pipeline *new to the cache*
  entering design — via `scene.load`, watcher reload, or upsert — is first
  rendered ~60 frames at **half** pack resolution to a scratch target,
  interleaved with live frames (~3.5 ms budget per loop iteration, never a
  stall). Predicted full-res p95 = `overhead + (measured − overhead) × 4`,
  where `overhead` is a once-per-boot calibration run of a trivial shader
  (naive scaling multiplies fixed per-frame cost by the pixel ratio and
  flags fine shaders red). Probe uniforms are **pessimistic**: `audio.*`
  pinned to 1.0, scalars at descriptor `max` where declared — "worst case
  at this venue", not "cost right now". Verdict vs sidecar thresholds
  A < B: green passes; **yellow still enters design, flagged** (the §5.11
  probation window is the live-side net); **red is refused** — the only
  hard gate — and the previous design plan stays. Result rides `hot_reload`
  (`probe` field, incl. a JPEG thumbnail) so the authoring agent
  self-corrects on performance and look. Known residual risk: an
  in-process probe can't contain a shader that *hangs* the device — that's
  §5.11's recovery contract.
- **Design-leg autosave**: every applied design edit debounce-writes the
  draft to `<scene_dir>/.wzrd/design.scene.json` (atomic, gitignored);
  at boot a draft that differs from `scene.json` is restored into the
  design leg (probe-gated like any apply). A crash mid-design can't eat
  the draft.
- **Preview topology: one view, source toggle.** The single native
  `PreviewTarget` samples either composite, each with its own leg's
  brightness/saturation masters — and no §2.8 warp,
  except in Align's program view (§1b). `preview.setSource` flips the leg
  when the slot doesn't pin one; the UI toggle lives in Perform's deck bar.
- **The deck toggle is a full control switch (decision revised 2026-07-12,
  same day it landed).** The original §5.6 design shared one venue-level
  `SliderBank`/`ParamOverrides`/masters/transport across both legs
  ("operator knobs aren't scene content"). First real use disproved it:
  setting speed 4 while previewing DESIGN also sped up the show. Now **each
  leg owns its complete control state** — `Transport` (speed bends time, so
  independence requires per-leg clocks), `Masters`, `SliderBank`,
  `ParamOverrides` — and `param.set`/`master.set` take an optional
  `leg: "design"` (default) | `"live"`; the UI passes the deck toggle's leg
  on every write, and the `drivers`/`frame_stats` channels follow the
  toggle too. **Promote copies design's control state (including the design
  clock) into live at the pointer swap** — what you previewed is exactly
  what goes live, phases and picks included — and design keeps its copies
  (still a copy, not an exchange). **Pull copies live's control state back
  into design.** The `masters` telemetry payload carries both legs
  (`{live, design}`); the session sidecar persists the **live** set (the
  show truth) and both legs boot from it. Single-leg (headless) runs alias
  both legs to one state, so the `leg` param is a no-op there. Consequence
  worth knowing: with per-leg clocks the legs' bar phases can drift apart
  while design speed/tempo differ — promote quantizes on the **live** bar
  boundary (the crowd's musical time), and the clock adoption at swap keeps
  the promoted content continuous with its preview.

### 2.7 Authoring MCP: the AI scene/shader co-author (§5.10, landed 2026-07-22)

`wzrd_mcp/engine_tools.py` exposes **only the authoring slice** of the WS
surface to a local Claude Code session (persistent JSON-RPC client to
`ws://127.0.0.1:9123`, override `WZRD_ENGINE_WS`; optional dep extra
`.[engine]` → `websockets`; the tools only register when it is installed,
so the Modal image never carries them; toggles in `tools_config.json`). Setup
recipe: repo `README.md` § "AI scene authoring".

**Two seats (the core boundary).** The agent authors *structure & shaders*
and operates **only the design leg** — no tool takes a `leg` param. The
human owns live feel (masters, knob overrides), the deck
(**promote/pull**), preview source, and probe thresholds — deliberately NOT
tools, and **no agent `promote`, ever**: the operator flipping to the UI is
the human gate to the projector.

**Facet taxonomy** — one vocabulary for read scope, write verbs, and change
tags: `layers` (backed by `pack.info`), `bindings` (`scene.getState` +
splice + `scene.load`), `effects` (`effect.describe`/`effect.upsert`),
`drivers` (one-shot `drivers`/`audio` telemetry frames). Tool surface: one
read (`get_scene_context {scope?, depth, ids?, since_rev?}` — unscoped
digest is the orient call; `full` requires a scope; single-effect full depth
includes WGSL via `effect.describe {name}`), five facet-bound writes
(`upsert_binding`/`remove_binding` as read-splice-CAS, `upsert_effect`/
`remove_effect`, `set_groups`/`set_labels`, plus the `set_scene` escape
hatch), two utilities (`validate_wgsl`, `get_preview` — strictly one-shot
subscribe/frame/unsubscribe so the §2.6 demand gate holds). Every read is
**self-contained** (status header incl. "engine unreachable since \<ts\>" +
recent changes ride along); author replies carry the probe verdict **as an
image content block** — write → see → fix without touching the projector.
Clips (gig E1d): the `bindings` read also lists the project's HAP clips
(`<scene_dir>/generated/*.mov`, newest first, with the region from each
clip's `.meta.json` sidecar) as ready-to-paste `image.clip` paths — read
from this disk, which the MCP and engine share; apply replies surface clip
`warnings` as `WARNING:` lines; `get_preview` takes the *second* frame so a
just-woken design leg's clips have landed (§2.4).

**The engine cluster behind it** (all in `rpc.rs`/`telemetry.rs`/`core.rs`):

- **Rev + epoch + change ring** (`telemetry::ChangeLog`): monotonic design
  rev bumped by every design mutation, boot epoch so a crash-relaunch can't
  make `since_rev` lie, ring depth 32 broadcast on the sticky `changes`
  channel with `changes.list` backfill — one shape, two transports.
- **Actor identity is per-connection** (`rpc::Actor`): `hello {actor}` once
  per session; WS defaults `agent`, Tauri dispatch passes `ui`, engine-
  internal paths (watcher, autosave restore) record `system`. No per-call
  actor params anywhere.
- **`base_rev` CAS on `scene.load`**, checked on the render thread — a
  splice race is a prescriptive one-turn retry, never a silent clobber.
- **Verdict-in-reply everywhere**: `effect.upsert` naga-validates first,
  then defers its reply through the probe like `scene.load` (see §2.3). No
  author path ever correlates via sticky `hot_reload` replay.
- **Webview reverse-sync**: the webview subscribes to `changes` and, on any
  `actor != ui` entry, re-pulls the affected facet into the Zustand store
  **only** (`App.tsx`) — never into `sceneCommit`'s disk debounce.
- **Persistence — the agent never decides when to save**: every successful
  agent-actor design apply mirrors atomically to
  `<scene_dir>/.wzrd/scene_agent_latest.json` (distinct from the any-actor
  crash autosave); writing agent work into `scene.json` is the operator's
  explicit **ADOPT AGENT SCENE** button in the TopBar (flushes the
  re-synced store through the accept-gated disk path; the watcher echo is
  absorbed by the §3.5 content-equality dedupe).

**Rejected shapes** (don't re-propose): generic `write({scope, payload})`
patch tool; RFC-6902 JSON Patch; overlapping read tools
(`engine_status`/`get_pack`/`get_scene`/`describe_effects` all fold into
scoped `get_scene_context`); whole-scene `set_scene` as the primary write;
embedded-Claude `scene.edit({instruction})`; agent-decided persistence;
per-call actor tagging; author-verdict correlation over sticky telemetry.

### 2.8 Alignment layer: the n-point output warp (§5.14, landed 2026-08-12)

Generalises D9. The operator drags control points until rendered content
lands on the physical surface; the result is persisted per project and
applied by the engine whether or not any UI is running. Four corner handles
by default, extra handles droppable anywhere for local correction.

**Where it sits, and why that placement is not negotiable.** Alignment
describes the *physical install*, like the §5.4 crossfade master — so it
applies **after** the promote crossfade and **after** the masters, to the
projector swapchain only. It is engine-wide, never per leg, never
duplicated, never copied on promote/pull, and never touched by `scene.load`,
agent authoring or an effect reload. It is also **not scene content**: no
MCP tool has a verb for it (same scope rule as §2.5). The native preview
stays **unwarped** in Perform/Prepare (§2.6 convention — the calibration
warp only reads right on the wall; their bind groups hold a 1×1 zero dummy
LUT and `render_preview` clears `adjust.w`). Align's **program view** is the
exception: it runs the projector's own uniforms on the real LUT, indexed by
uv (`fs_preview`) because the preview target is never the LUT's size — the
projector's `fs_main` keeps its exact per-pixel indexing.

**The model.** `W(x) = H⁻¹(x) + R(x)`, mapping dest uv → source uv:

- **Base stays projective.** `H` is the exact unit-square→quad homography of
  the four corners (Heckbert). This is the load-bearing decision: fitting any
  scattered-data interpolator (TPS, MLS, mesh) through four dragged corners
  gives an affine-plus-bending fit, not a perspective one — straight edges
  bow and the image never sits flat on a keystoned wall. Every extra handle
  is a *correction on top of* the projective base.
- **Residual is compactly supported.** Wendland C² RBFs, per-handle radius,
  solved by dense LU (N ≤ 64, microseconds). Locality is the point: a
  mid-frame correction must not slide corners you already dialled in. Note
  per-handle radii make the collocation matrix asymmetric, so Wendland's
  positive-definiteness guarantee (uniform σ only) doesn't cover us —
  invertibility here is empirical, which is why solve failure is a
  first-class rejection path, not a defensive afterthought.
- **Two properties the UI leans on.** A handle created with
  `anchor := W_current(dest)` has coefficient exactly zero, so *adding a
  handle is a no-op on the image* (and removing it is free). And each handle
  stores its dest-space offset from `H(anchor)`, so *a corner drag carries
  the extra handles with the content* — rough in corners first, refine later.

**The LUT is the runtime representation, and the extensibility seam.** An
`Rg32Float` texture sized exactly to the projector swapchain holds
`W(x) − x` (the *offset* — zero means identity, which keeps the
disabled/dummy case trivial). It is rebaked only when the alignment changes
or the swapchain resizes: one fullscreen pass encoded into the *existing*
frame encoder as step 0, no extra submit. Consequences:

- Per-frame cost is **one texel read, independent of handle count** — a
  future camera pass with 500 correspondences costs what 4 corners cost.
- **No stale-LUT window**: a resize marks the bake dirty and the rebake is
  encoded ahead of the final pass in the same frame. Do *not* lean on
  out-of-bounds `textureLoad` returning zero — WGSL leaves that
  implementation-defined, so it is not a portable identity fallback.
- Format choice is deliberate for portability (CLAUDE.md `Features::empty()`):
  `Rg32Float` is core-*renderable*, only blending needs `float32-blendable`;
  and the LUT is read with `textureLoad` at exactly one texel per output
  pixel, so `float32-filterable` is never needed either.
- A camera-driven auto-align uploads a dense field straight into this LUT
  with no analytic model, and nothing downstream changes.

**The Y-flip trap.** `warp_bake.wgsl` and `final_pass.wgsl` share
`fullscreen_vs.wgsl` verbatim *including the Y flip*, because a mismatch
renders as a vertically mirrored warp — plausible enough on a wall to cost
an evening. Two GPU-backed tests in `gpu.rs` pin both halves: the bake
matches the CPU model at every pixel, and the final pass samples through it
in the right direction with the right row.

**Files and policy.** `<scene_dir>/alignment.json`, engine-written, atomic
temp+rename, debounced ~1 s after the last change plus on shutdown (same
`session::touch` mechanism as the sidecar), gitignored. Boot precedence:
the file, else a one-time migration from the legacy calibration matrix,
else identity. Migration direction is the easy thing to get backwards — the
stored matrix is the old shader's **dest→source** map while corners are
*dest positions of source corners*, so migration applies its **inverse** to
the unit square. A file that fails to parse or solve degrades to identity
with a warning rather than failing the boot.

**The background is a light source.** `background` paints dest pixels whose
source falls outside the composite. Any non-black value floods the physical
surface and breaks the additive thesis — it is an alignment aid, not a show
setting, and it persists, so the Align tab shows a warning pill for as long
as it is set.

**Test patterns** (`alignment.setTestPattern`) substitute a generated
pattern for the composite **in source space**, so it warps with the content
and reveals misalignment against physical edges. Runtime-only and never
persisted, so a restart can't leave a grid on the wall.
The exception is **`white`** (2026-10-16, for the show-setup wizard's lit
photo): the whole projector raster at full white — no warp rect, no fit,
no masters, no background — decided at the very end of `shade()` (after
the samples, so they stay in uniform control flow). The photo has to show
everything the lens can light, not the currently aligned rect; that full
throw is what `rectify_surface` detects. Mode id 4; the line patterns are
1–3.

**Output transform: flip + fit** (2026-10-15). Three flat fields on the
document — `flip_h`, `flip_v` (bool, default false) and `fit`
(`"stretch"` default = the old behaviour \| `"fit"` letterbox \|
`"fill"` crop); serde defaults, so older files load unchanged and an unknown
`fit` degrades to stretch. They describe the **projector**, not the warp:
they apply even with `enabled: false` and survive `alignment.reset`. Order
in `fs_main`, per output pixel:

1. **Flip first, in dest space, before the LUT lookup** (integer mirror
   `x → W−1−x` for the texel, `u → 1−u` for the uv). The projector mirrors
   its input, so wall position = M(output pixel); flipping here keeps the
   LUT, the corners and the Align canvas in **wall space** — toggling a flip
   never disturbs the operator's alignment.
2. Warp as before (LUT → warp-source uv `s`).
3. **Fit after the warp**: the composite occupies a centred rect inside
   warp-source space (`alignment::fit_rect` — composite vs swapchain
   aspect, rebuilt on the CPU each frame into the `fit` uniform), composite
   uv `c = (s − offset) / scale`; outside it gets `background`, like any
   out-of-source pixel. Test patterns are generated in composite uv, so they
   land inside the rect.

Align's **program view** (`fs_preview`) applies fit but **not** flip — it
shows wall space, like the canvas drawn over it. The leg previews pass the
identity transform. Pinned by GPU tests in `gpu.rs` (`flip_h` before the
warp puts a translated image's background on the opposite side; letterbox
bars; program view fit-not-flip).

**Control surface.** All three verbs are inline — a drag must not queue
behind the render thread and none of them rebuild a plan. State lives behind
a small mutex in an `Arc<AlignmentState>` shared with the render thread;
writers clone a snapshot out and never hold the lock across work (§1b).

**UI (the Align route, ⌘3 — in the ⚙ Setup menu, not a tab).** SVG over dest space padded ~20% on every side,
because handles are routinely dragged off-screen; canvas units equal
projector pixels so an arrow-key nudge is exactly one output pixel (⇧ = 10)
— the last millimetre of physical alignment, and the one thing a mouse
cannot do. Gestures: drag a handle to move it; **drag inside the quad to pan
the whole image** (a plain corner write, so the extra handles ride along via
the corner-carry rule and nothing reshapes); **click the outline to drop a
handle on the edge** and drag it in the same motion; arrows nudge the
selection, or pan the quad when nothing is selected. Right-click
adds/removes handles and edits radius; corners get **Reset corner** only
(they are structural — removing one would destroy the projective base).

**Edge handles are not extra corners** — only four points can define the
projective base. They are ordinary RBF handles that *start* exactly on the
edge (the click snaps to the drawn boundary, and the engine's no-anchor rule
then anchors them on the source edge), so pulling one bends that side to
follow a wall that isn't straight. Their reach is the handle radius like any
other, and being radially symmetric they pull some interior with them; that
is the honest capability, and bezier edges stay deferred.

**What the canvas draws is the real field.** A source-space grid is mapped
through the model and drawn where it actually lands, evaluated in
`state/warpMath.ts` from the **engine's own solved coefficients** — which is
why `weights` rides along on the alignment payload as read-only derived
state. Re-solving in the UI would have been a second solver and a guaranteed
drift between what the operator sees and what the projector does. The
image under the grid is the native **program view** (§1b): the projector's
exact final pass, local corrections and test pattern included, dimmed, in a
`<NativeSlot>` over the projector rect — image and grid now agree. Without
the underlay (browser, `WZRD_PREVIEW=window`) it falls back to the
`preview` JPEG placed with a CSS `matrix3d` from the corner homography, which
cannot express local corrections (the label says so). Before the grid
existed, dragging a handle changed nothing visible in the UI at all, which
read as a broken feature.

Commits go through `state/alignment.ts`, **not** `sceneCommit.ts`: this is
not scene content, and the engine owns persistence outright, so there is no
save button and no adopt step. Optimistic local state is per-facet, which is
what lets a corner drag keep local corners while accepting the engine's
carried handle positions in the same echo.

**Known caveats** (from the plan's risk list; revisit only if dogfooding
bites): extra handles *can* pull corners — the projective base is exact only
where no residual support reaches, and the default σ = 0.35 reaches a corner
from a quarter of the way in. Fix by shrinking the radius or re-dragging;
if it turns out to bite regularly, pin the corners with four zero-residual
basis functions rather than growing machinery. Folded warps (dragging
handles past each other) are **allowed** — the operator sees the mirroring
immediately on the wall, and fold prevention would cost more than it saves.

**Deliberately not built:** camera-driven auto-alignment (the hooks are the
dense-field upload path and the test patterns; the loop itself is an
external script against this same WS surface, next to `wzrd/align.py` which
already does homography estimation from photos), multi-projector / edge
blending, soft-edge masking, per-region warp, bezier edges.

Headless verification lives in `render-core/tools/align_drag.py` — corner
sweeps, a live handle demo, and `--verify-isolation`, which asserts
`alignment.json` comes out byte-identical across a `scene.load` + `pull`.

### 2.9 Frame profiler: `perf` channel (perf-monitor PM1 2026-10-10, PM2 GPU timing 2026-10-12)

`fps`/`frame_stats` time mark-to-mark, so a 2 ms and a 15 ms frame both read
"60". `perf.rs` times what the frame *spends* its time on and reports
**headroom = 1 − max(cpu.work p95, gpu.total p95) / 16.7 ms** (fixed 60 Hz
budget — the engine doesn't know the projector's refresh; CPU-only when
`gpu` is null). `fps`/`frame_stats` are untouched
(§3.2's honesty contract).

- **Stages** (`perf::Stage`, fixed enum, no strings in the hot path):
  `inbound.cmds`, `inbound.watch`, `inbound.probe`, `inbound.misc`,
  `telemetry`, `acquire` (wait — first in `redraw`, see §4 #9),
  `tick.live` (incl. transport step + promote machine), `tick.design`,
  `encode`, `submit` (incl. handing clip slots back), `present`,
  `preview.present`, `preview.jpeg`, `pace` (wait),
  **`host`** — render-thread time *between* Core calls (winit dispatch,
  TauriHost's size/occlusion polling). `cpu.work = Σ stages − acquire − pace`.
  Each host entry point (`poll_inbound`, `pace_frame`, `redraw`,
  `render_offscreen_frame`) opens with `perf.enter()`, which charges the gap
  since the last span to `host`, so the stages tile the whole frame interval;
  `frame.unaccounted` (interval − Σ stages) is the coverage check — measured
  ≤ 0.015 ms p95 on the smoke scene. **A new cost path adds a stage** (not a
  span inside an existing one) and a matching `Stage::name`.
- **Hot path:** `begin`/`lap`/`end` are clock reads into a fixed array; one
  ring-slot write per frame; atomics only. Allocation, sorting, JSON and the
  registries' mutexes run at the 2 Hz emit inside the `telemetry` stage.
  Measured cost: ~0.7 µs/frame hot path + ~25 µs per aggregate (~1 µs/frame
  amortised) — `cargo test --release perf_overhead -- --ignored --nocapture`
  replays it. `--perf off` (CLI; the Tauri host passes `On`) removes it.
- **Side probes** (process-global, lock-free): each owned thread
  (render, `clip-decode:*`, `ws-conn-*`, `osc-recv`, `engine-telemetry`)
  samples its own `CLOCK_THREAD_CPUTIME_ID` ~1 Hz via `perf::ThreadCpu`
  (`#[cfg(unix)]`; null elsewhere; a slot unsampled > 3 s reads 0 — it's
  parked). `perf::ClipStats` per clip stream: decode ms (decoder thread),
  **stale frames** — wanted frame ≠ the one in the texture after the upload
  decision; a native-rate repeat (`current == wanted`) never counts — and
  ready-queue depth. Streams that never decoded (the idle design copy) are
  counted in `clips_idle`, not listed. `count_telemetry` counts bytes where
  they're already serialized: `ws.rs` (per notification) and the Tauri
  `telemetry_loop` (`json_len`, the webview hop).
- **Latency:** `knob_submit` (`param.set`/`master.set` stamp the oldest
  pending write in `Shared`; the tick picks it up, the submit resolves it
  — ≈ 1 frame since the tick runs after `acquire`, §4 #9),
  `rpc_queue` (`EngineCommand.queued_at` → handled), `osc_age`
  (`AudioFeatures::packet_age_ms` at tick — engine-local clock, not epoch).
- **Process (1 Hz):** CPU % (`CLOCK_PROCESS_CPUTIME_ID`), RSS (macOS
  `proc_pidinfo`, Linux `/proc/self/statm`), wgpu live counts per type
  (`Instance::generate_report`, `num_kept_from_user` — the leak signal) and
  **`gpu_mb`, an inventory** summed from descriptors at sample time
  (`GpuContext::owned_bytes` + every plan image's `ImageSource::gpu_bytes` +
  the preview readback buffer; swapchains estimated at 3 images) — not a
  create-site counter, so a leaked resource shows in the counts, not the MB.
- **Dropped:** presented frames with interval > 1.5× budget (window + total).
- **Spikes:** interval > `trigger_ms` (default 33.3) after the first 60
  frames freezes that frame ± 5 neighbours (stages + `ran: {cmds, reloads,
  probe_steps, clip_opens, rev}`) → sticky `perf_spike` + one `warn` log
  line. At most one capture per 2 s; the rest are `suppressed`. An
  offscreen (occluded / output-off) frame is judged against `trigger_ms` +
  its own 33.3 ms pace, or every paced frame would read as a spike.
- **GPU passes (PM2, `gpu_timing.rs`).** `perf.gpu = {every, frames,
  skipped, total: {p50,p95,max}, passes: {<pass>: {p50,p95,max,n}}}` — passes
  `warp_bake`, `composite.live`, `composite.design`, `final`, `preview`,
  `probe` (`perf::GpuPass`), listed only if they ran in the window (~10 s of
  samples). How it stays zero-impact, and why each choice:
  - **Sampled, 1 frame in 30 by default** (`--gpu-timing N`, Tauri host
    `WZRD_GPU_TIMING`; `0` = off, `1` = every frame). A timed frame costs the
    render thread **~73 µs**: wgpu-core 22 encodes a query *reset* (on Metal a
    blit encoder + fill) ahead of every pass that carries `timestamp_writes`,
    ~30 µs for the first timed pass + ~10 µs per further pass, plus the
    resolve/map. Every-frame timing measurably raised `encode`/`submit` p95
    (~+0.2 ms); 1-in-30 is ~2.4 µs/frame amortised, keeps timed frames < 5 %
    (so they never move `cpu.work` p95), and matches the 2 Hz emit.
    `cargo test --release perf_overhead -- --ignored --nocapture` reprints
    both numbers.
  - **Only existing passes carry timestamps.** Clip **uploads are untimed**:
    `write_timestamp` outside a pass needs `TIMESTAMP_QUERY_INSIDE_ENCODERS`,
    and on Apple Silicon (no blit-boundary sampling) wgpu-hal's Metal backend
    then opens a dummy blit encoder + fill per sample
    (`wgpu-hal-22/src/metal/command.rs::enter_blit`). Live uploads therefore
    sit outside `total`; design uploads land in `composite.design`.
  - **Never blocks.** Frame N's queries (probe, main, preview submits) are
    resolved by frame N+1's main encoder, `map_async`ed after that submit,
    and read once the callback fired — `queue.submit`'s own maintain fires it,
    no extra `poll`. 4 slots; a sampled frame with every slot in flight is
    skipped (`skipped`). Nothing allocates per frame; ticks → ms at the 2 Hz
    aggregate.
  - **`total` = Σ per submit of first-begin → last-end**, not first-to-last
    over the whole frame (that would count the `acquire` vsync wait between
    the probe and main submits). It's the budget truth.
  - **Main-submit passes are charged `end − max(begin, previous pass's
    end)`**, so they tile `total`. Raw stage-boundary stamps overlap on
    Apple's TBDR GPU: the final pass's start-of-vertex fires during the
    composite's fragments — raw `final` read 7.5 ms on a scene whose final
    pass costs 0.09 ms. The **preview blit** (own submit, queued right
    behind the main one, samples its composite) is charged from the main
    span's end and adds only that to `total` — raw, it read p50 8 ms in the
    app. `probe` is raw: the timed probe frames' mean × every probe frame
    that ran (≤ 8 timed per engine frame).
  - **A timed frame runs alone on the GPU.** Otherwise its first pass's
    span counts the previous frame's tail (the app read `composite.live`
    p95 14.4 ms, `gpu.total` p95 29.8 ms — negative headroom at a steady
    60 fps). Offscreen, `wait_turn` drains before acquire (§2.10); on a
    window `GpuContext::drain_before_timed_submit` polls `Maintain::Wait`
    right before the main submit (after encoding, so it waits only for
    what's left; charged to `acquire`). Untimed frames never wait. Verified on a
    visible window 2026-10-12: app `gpu.total` p95 7.4 ms at 60 fps,
    interval p95 17.50 ms with timing on and off, 0 window drops; `preview`
    reads 0.0 (p50 and max).
  - **Metal quirks handled:** a pass with no draws gets no end-of-fragment
    sample (empty composite → dropped for that frame; a frame with nothing
    sane isn't a sample). **wgpu-hal 22 bug:** Metal `copy_query_results`
    passes `NSRange::new(start, end)` — end as *length* — so any resolve not
    starting at query 0 is wrong and overruns. Hence one query set per slot,
    resolved whole (`0..26`) on Metal; other backends resolve per written
    pair so Vulkan never reads an unwritten query. Drop the workaround when
    wgpu is upgraded past the fix.
  - **Read GPU ms knowing the clock.** Apple GPUs downclock when lightly
    loaded: the smoke scene's composite reads ~7.5 ms at 60 fps (GPU ~45 %
    busy) but ~3.4 ms when the engine runs uncapped. At 60 Hz, `gpu.total`
    is honest about *this* frame but pessimistic as a capacity number — the
    GPU clocks up before it runs out. PM3's uncapped bench is the
    capacity measure.
  - **E2 hook:** `Perf::gpu_pass_recent(GpuPass::CompositeLive |
    CompositeDesign, n)` → p50/p95 over the newest n samples — per-leg GPU
    cost that a TAKE crossfade (both legs rendering) doesn't inflate.
- **Consumer:** Debug room → "Performance (frame profiler)" panel
  (`routes/Debug.tsx` `PerfPanel` incl. `GpuTable` + a "gpu total" stat;
  types `PerfPayload`/`PerfGpu`/`PerfSpike` in `api/ipc.ts`, store
  `perf`/`perfSpike`, `perf_spike` hydrated from `lastPayload`). The
  regression gate is §2.10; the graphed panel and `perf.profile` are PM4 in
  [../TODO/perf-monitor.md](../TODO/perf-monitor.md).

What it showed on day one (2026-10-10, M2): smoke scene headroom ~0.93
(work p95 ~1.2 ms); fixture-a with two HAP clips decodes in ~0.1 ms p95,
0 stale. Two real findings are logged in §4 (#9, #10). PM2 (2026-10-12,
release, 60 Hz presenting): **the GPU, not the CPU, sets the headroom** —
smoke `gpu.total` p50/p95 7.7/8.7 ms vs `cpu.work` p95 ~1.7 ms → headroom
~0.48; nine bindings incl. five HAP clips 8.3/9.4 ms. `composite.live` is
nearly all of it (the four procedural effects; one `silk` binding alone
~2.6 ms); `final` ~0.1 ms. GPU timing on (1-in-30) vs off: `cpu.work` p95
and interval p95 within run-to-run noise.


### 2.10 Perf bench + gate: `--bench`, scenarios, `perf_gate.py` (perf-monitor PM3, 2026-10-12)

The regression gate every engine change runs (rule in §6). Three parts:

- **`render-core --bench <scenario.json> --bench-out <report.json>`**
  (`bench.rs`) — a third host beside `WinitHost` and TauriHost: no window,
  no surface. `main.rs` routes `--bench` to its own grammar (`BenchArgs`),
  so the engine `Cli` the Tauri host builds stays untouched.
  `Core::with_topology(cli, two_leg)` picks the §5.6 topology without a WS
  server (scenarios default to two legs, like the operator app);
  `Core::init_gpu_offscreen(w, h, vsync_hz)` builds
  `GpuContext::new_offscreen` (adapter with `compatible_surface: None`,
  final pass into a `Bgra8UnormSrgb` texture at the scenario size, default
  1920×1080). The host then runs the normal `poll_inbound` → `pace_frame` →
  `redraw` contract; `redraw` goes through `GpuContext::acquire_output` /
  `present`, which on a window are exactly the old swapchain calls.
  Offscreen, **`acquire` is the backpressure** (a wait, not work): ≤ 2
  submits in flight (a swapchain's frame latency), and `--paced` /
  `"paced": true` adds a virtual 60 Hz vsync on a fixed grid. Uncapped is
  the default — it measures *cost*; paced is for latency/jitter and stale
  clip frames, which are judged at the show's rate.
- **GPU timing in the bench: 1 frame in 4, each timed frame alone on the
  GPU** (`--gpu-timing`, default `DEFAULT_BENCH_GPU_TIMING` = 4; the
  offscreen output drains the queue before a frame `GpuTimer::recording()`
  says is sampled). Measured why: with two frames in flight Apple's GPU
  starts a frame's first pass while the previous one still draws, and the
  per-submit span counts that wait — smoke read 3.9 ms at two in flight,
  6.4 ms at three, **2.4 ms isolated**, at the same 2.6 ms/frame
  throughput. Draining *every* frame instead lets the GPU idle and
  downclock (4.5 ms). Untimed frames keep the queue full, so the GPU stays
  clocked up. `GpuTimer` itself is unchanged (the getter is read-only); a
  timed frame still costs the render thread ~0.07 ms (§2.9), a constant
  baselines compare like with like.
- **Report** = `Perf::bench_report()`: the Tier-A aggregate over the
  *whole* measured window (`Perf::bench_begin` keeps every `FrameRecord`,
  `GpuFrame` and latency sample — the live rings hold ~5 s), plus clip
  totals (`perf::STALE_TOTAL` diffed over the window, worst decode p95,
  `decodes` + `decode_ms` = frames decoded / decode-thread ms, all streams),
  wgpu resource counts at window start/end, per-method RPC round-trips
  (n, errors, p50/p95/max, skipped), subscriber message counts, and
  machine info (adapter, backend, OS, build profile, git sha + dirty).
- **Teardown never hangs.** After the measured window the timeline is
  cancelled and the worker's queue closed; the host keeps rendering
  (unrecorded, ≤ 10 s) until the worker and every side thread have
  *exited* (`JoinHandle::is_finished`), because they block in
  `rpc::dispatch` on a render-thread reply. A thread still blocked then is
  detached, never joined, and counted in `teardown.stuck_threads`
  (`actions_unfinished` = RPCs left without a reply); the report is written
  before `drop(core)`, and a watchdog exits the process (code 3) if engine
  teardown takes > 20 s. Never key a join on a request counter: an orphaned
  bench once sat 1h43m in `join` on a worker still waiting for its reply.

**Scenario format** (`render-core/perf/scenarios/*.json`; scenes and
fixtures in `perf/scenes/`): `{description, scene, size?, warmup_frames?,
frames, paced?, two_leg? (default true), alignment?, audio?: {lmh,
onset_hz}, subscribe?: [channel], preview?: {size, params}, actor? ("ui"),
timeline?: [...], gate?}`. `preview` holds a native preview slot visible
for the whole run on an offscreen target (`params` = `PreviewParams`).
Paths are relative to the scenario file. Timeline entries are `{at_frame,
method, params, every_frames?, count?}` (frames count from the first
warm-up frame) or `{at_frame, hz, method, params}` (a background thread at
a fixed rate — knob storms); they go through `rpc::dispatch` from a worker
thread, exactly like a WS connection, so they exercise the real `promote` /
`param.set` / `scene.load` paths. Param directives: `{"$alt": [a, b]}`
(by repetition), `{"$scene": file}` (staged scene JSON string for
`scene.load`), `{"$file": file}` (text, e.g. WGSL), `{"$json": file}`,
`{"$path": file}` (absolute path string, e.g. for `preview.file`).
`audio` feeds the engine's own OSC listener on a free port (the real ingest
path). `gate` is read only by the gate script. **The scene is staged**
into a temp dir first (scene + a copy of `effects/` + optional alignment;
pack and `image.clip` paths made absolute; never `session.json` or
`.wzrd/` drafts), so a bench never writes into a real project.

| Scenario | Guards | Mode |
|---|---|---|
| `idle` | fixed floor: final pass + identity warp | uncapped |
| `smoke` | today's typical scene (`examples/phase3_smoke`) | uncapped |
| `layers-heavy` | 40 built-in bindings, audio pinned to 1.0 over OSC | uncapped |
| `clips-7` | seven 1080p HAP clips: decode, **stale = 0** | uncapped |
| `fabric-7` | the gig layout: seven small regions (synthetic `perf/packs/fabric7`) × library waveWarp/edgeGlow on HAP clips — the §2.4 region scissor | uncapped |
| `take-crossfade` | clip-set swap TAKE every 2 s (scene.load → promote 400 ms) | paced |
| `warp-64` | 64-handle alignment + `alignment.set` every 4 frames (rebake) | uncapped |
| `edit-churn` | 300 × `effect.upsert` + 300 × `scene.load`, single leg: resource counts flat, rebuild ms | uncapped |
| `preview-remote` | a `preview` subscriber as design's only consumer (JPEG readback; design renders on capture frames only) | uncapped |
| `preview-clips` | `clips-7` + a `preview` subscriber: a clip-drawing design leg stays per-frame, **stale = 0**, JPEG ≈ 15 Hz | paced |
| `studio-preview` | `clips-7` on LIVE + the Studio monitor (offscreen 960×540 slot, `file: true`) playing an 8th 1080p HAP clip via `preview.file`: LIVE 60 fps, **stale = 0**, no design leg | paced |
| `knob-latency` | `param.set` at 100 Hz → knob→submit p95 | paced |
| `capacity-clips` | info only: largest N clip bindings with headroom ≥ 0 and no stale | uncapped |

Clip scenarios read four 3 s 1080p30 HAP clips with grain (≈ 1 MB/frame,
so the decoder does real work) that the gate encodes once into
`render-core/perf/.media/` (gitignored, like `perf/.work/`). **A new cost
path adds or extends a scenario in the same change.**

**`scripts/perf_gate.py`** (`uv run`, stdlib only): `cargo build
--release`, builds the example pack if missing, runs each scenario 3×,
compares with `render-core/perf/baselines/<adapter>__<hostname>.json`
(committed; `apple-m2__xanders-macbook-pro.json` is the first), prints one
table (scenario × metric, baseline → now, Δ, verdict), exits 1 on FAIL.
Budgets and thresholds are named constants at the top of the script.
Choices the measurements forced, all on 2026-10-12 with other agents and
Chrome on the machine:

- **CPU/GPU costs take the best of the 3 runs, latency and counts the
  median.** Noise only adds time; a median of 3 still flapped (smoke
  `cpu.work` p95 0.15–0.39 ms run to run).
- **Regression = > 15 % and > 0.5 ms for GPU passes, > 1 ms for CPU
  stages, > 2 ms for latency**, stale counts on tracked scenarios
  (> 1.5× and > +10). Every stage/pass is checked; only failures get rows.
- **"No frame > 33 ms" judges render-thread work, not the interval**:
  uncapped, the interval's `acquire` is GPU backpressure, and the desktop's
  GPU stalls ~45 ms for other apps' compositing about one run in five.
- **Each bench run has a wall-clock cap** (`gate.timeout_s`, else 60 s +
  frames at 20 fps): a hung or crashed run is killed and becomes a
  `bench run` FAIL row, and the gate carries on with the next scenario
  (`--update-baseline` refuses to write then). Any run with
  `actions_unfinished` or `teardown.stuck_threads` > 0 FAILs too (a lost
  engine reply). The engine runs under its own default `RUST_LOG`
  (`info,wgpu_core=warn,…` — never plain `info`: wgpu-core logs every
  `Device::maintain` at INFO, ~180 lines/s, which perturbs the timings;
  `fx_gate.py` does the same). The `log` telemetry channel drops
  `wgpu_core`/`wgpu_hal`/`naga` below Warn whatever `RUST_LOG` says.
- A failing scenario re-runs once before it fails; > 10 % *and* > 0.3 ms
  spread prints NOISY. Paced scenarios never judge GPU ms (the GPU
  downclocks at 60 Hz, §2.9). No baseline for the machine → budgets only.
- `--update-baseline --reason "…"` rewrites the baseline and appends
  `{date, machine, sha, reason, deltas}` to `perf/baselines/history.jsonl`.
  A scenario file path argument runs that file judged against its stem's
  baseline — how the gate was proven red (a 24-octave `silk` in a scratch
  copy of `smoke`: `gpu.total` p95 2.46 → 8.45 ms, FAIL on budget and
  regression).

A full run takes **~125–145 s** (incl. ~10 s for the capacity search) and
measures, on the M2 (baseline refreshed 2026-10-12 after the perf pass):
`gpu.total` p95 smoke 2.3 ms, layers-heavy 2.3, preview-remote 2.7 (4.8
before the pass), fabric-7 0.8, warp-64 5.6 (the 64-handle rebake is ~4.1
of it), `knob→submit` p95 ~18 ms (§4 #9; 33.5 before the tick moved), 0
wgpu objects gained over edit-churn, ≥ 64 clip bindings held. GPU p95 rows
spread up to ~40 % run to run even on a quiet machine, and the baseline
keeps the best of 3: a lone p95 FAIL gets an interleaved A/B against the
previous binary before anyone acts on it. Findings it surfaced are §4 #12.

---

## 3. The 2026-07 pass — what was wrong and what changed

This section is the institutional memory of the first "it's laggy and the UI
is dead" review. Keep these invariants.

### 3.1 The 1-second frame stalls (the big one)

**Symptom:** frame p95/p99 ≈ 1017 ms while p50 ≈ 4.8 ms; preview lagging;
whole UI hitching; params seemingly inert.

**Cause:** on macOS, a fully-occluded window's CAMetalLayer throttles to
~1 Hz, so `surface.get_current_texture()` **blocks the render thread for up
to a second per frame**. During a Tauri session the projector window
typically sits behind the operator window, so this was the *steady state*,
not an edge case. Everything rides on the render thread (IPC command drain,
preview capture, hot reload), so one blocked call froze the product.

**Fix (invariant to preserve):** the engine tracks `WindowEvent::Occluded`.
While occluded it never touches the swapchain — `PassPlan::render_offscreen`
renders the composite only (preview + telemetry stay live) at a self-paced
~30 Hz, driven from `about_to_wait` because macOS stops delivering
`RedrawRequested` to occluded windows. On un-occlusion, normal
present resumes. Nothing presents the live composite while occluded, so
`render_offscreen_frame` renders it only for a consumer
(`Core::live_demand_occluded`: native preview on LIVE or the program view,
or a headless JPEG stream); otherwise only ticks. Its clips idle meanwhile
and wake (1–2 transparent frames) on the first visible frame. **Any future render-loop refactor must keep the property:
nothing on the render thread may block on the swapchain of a window that
might be occluded or throttled.**

### 3.1b App Nap collapsed the frame rate to ~9 fps on focus loss

**Symptom:** 60 fps while the engine window was frontmost; ~9 fps the moment
the operator clicked the Tauri shell (window still visible).

**Cause:** macOS App Nap / background timer coalescing. When the engine
process is not the frontmost app, its timers are coalesced to ~100 ms
granularity — the event loop's ~4 ms frame-pacing sleep stretched to
~110 ms per cycle (≈ 9 fps). Occlusion isn't required; losing app focus is
enough, and in the shell workflow the engine is *always* background.

**Fix (invariant):** `lib.rs::hold_latency_critical_assertion()` takes a
process-lifetime `NSProcessInfo` activity assertion
(`NSActivityUserInitiated | NSActivityLatencyCritical`) at startup — the
documented opt-out for real-time AV processes. Never end this activity, and
don't replace the pacing sleep with an NSTimer-backed mechanism without
re-verifying background behaviour (test: present at 60, steal focus via
`osascript`, watch the `fps` channel stay flat).

### 3.2 The FPS pill lied

`fps` was computed as `1000/p50`, reporting "211 fps" while the engine
actually delivered ~18. Now `fps = samples / Σ(frame_times)` — honest
throughput. Percentiles unchanged. If you touch `FpsAccumulator`, keep the
honesty property; the operator trusts this pill mid-show (design spec §10).

### 3.3 Telemetry channels existed but never emitted

`drivers`, `audio`, `connectivity`, `log`, and the frame-stats counts were
declared, typed, and consumed by the UI — but no engine code emitted them.
The Perform/Debug pages looked dead. Emitters now live in
`Core::emit_periodic_telemetry` (drivers 10 Hz, audio 30 Hz, connectivity
1 Hz) and the `main.rs` tee logger. `FrameStats` counts come from
`Core::frame_counts()`. Lesson: **a channel isn't done until something emits
on it; wire UI + emitter in the same change.**

### 3.4 `ui.slider` was a stub and there was no live param path

Every parameter change round-tripped through `scene.load` (full plan
rebuild, buffers + bind groups recreated) or did nothing at all. Now:
`SliderBank` in `drivers.rs`, written by `param.set` **inline on the WS
thread** (no render-thread hop), read per-frame by `ui.slider` drivers.
Knob latency ≤ 1 frame. This is the seed of the design spec's "knobs" —
tune by feel, never re-prompt, never recompile.

### 3.5 Scene-edit echo storms

The UI saved via `scene.load` *and* `write_scene_file`; the file watcher
then fired on our own write → a second full rebuild. And the binding
inspector committed on every keystroke. Fixes: the engine skips watcher
reloads whose content equals the live scene (`reload_scene` dedupe);
the webview routes all mutations through `state/sceneCommit.ts` (optimistic
local state; engine push debounced 150 ms; disk write debounced 800 ms).

### 3.6 Surface canvas was both wrong and slow

The mask overlay used a full-canvas `source-in` fill per layer per preview
frame — which (a) replaced the entire canvas with the last layer's tint
(the "flat teal wash"), and (b) recomposited everything at 15 fps on the
webview main thread. Hit-testing was bbox-based (wrong for overlapping
regions), labels were unreadably small, and clicking wrote a *layer* id into
the *binding* selection field, so selection never reached the inspector.

Now (code: `state/masks.ts` + `components/SurfaceMap.tsx` since the
2026-10-09 redesign): per-layer tinted overlays are composited **once** into offscreen
canvases and blitted per repaint; picking samples a 256px-wide per-layer
alpha map; region names show in a floating hover label, not painted at
centroids (centroid labels landed outside thin/concave regions);
and the store has distinct `selectedLayerId` / `selectedBindingId`.

The canvas↔inspector link runs **both ways**, on the same client-side
selector resolution (`selectorCovers` in `state/scene.ts`, mirroring
`scene.rs::resolve_selector`; `pick` is ignored on purpose — it toggles
which member *draws*, it does not narrow the member set):

- canvas → inspector: a click aims the inspector at the first binding
  covering the picked region.
- inspector → canvas: `hoveredBindingId` in the store (set by the binding
  rows *and* the detail panel) highlights every region that binding
  resolves to, so a `tag`/`group`/`all` selector shows its whole footprint.

Overlays paint in three tiers — hot (pointer, 0.62) / warm (selection,
0.45) / rest (0.22, dimmed to 0.10 while anything is highlighted) — and
coldest-first, so a highlighted region is never buried under an
overlapping neighbour's wash.

### 3.7 Perform route

Superseded by the 2026-10-09 redesign — see [operator-ui.md](operator-ui.md).
Fixes from this pass that still hold: `read_mask_png` no longer pays a `pack.info`
RPC per mask (cached in Tauri state); redundant `request_redraw` removed;
preview JPEG decode moved off the paint path.

### 3.8 The binding inspector authored invalid scenes (2026-07-11)

The structured editor could write params/selectors the engine rejects:
switching an effect kept the old effect's params, "+ param" invented
`param_N` names, changing selector kind dropped `pick` and committed
`{"id": ""}`, and an emptied number field committed `NaN`→`null`. Every
such commit hot-reloads a scene the engine rejects — and a *boot* from
that file leaves a white projector window (no previous plan to fall back
to; the §5.11 "black composite at boot failure" item is the mitigation).
Invariant since the fix: **the inspector only authors what
`effect.describe` declares** — effect switches rebuild params from
declared defaults (same-named params carry over), add-param offers only
declared-but-unset inputs, param type dropdowns are bounded by the
declared input type, and `select.pick` survives kind changes. Keep any
future structured editor on this rule; grep `catalog` in
`routes/Prepare.tsx` (`switchEffect`, `defaultsFor`).

Second invariant, in `sceneCommit.ts`: **disk only ever receives
engine-accepted scenes.** A rejected push stays optimistic in webview
memory (Reload pill shows FAIL) but is never persisted — otherwise one
bad edit survives restarts and every later innocent edit re-writes the
whole corrupted in-memory scene back over any repair (this looped twice
on 2026-07-11 before the gating existed). The file on disk is the
last-good recovery point; don't add a persist path that bypasses the
accept gate.

---

## 4. Known weaknesses (accepted for now)

Ranked; none currently show-stopping at 5–20 layers.

1. **Full plan rebuild on every scene edit.** `scene.load` rebuilds every
   pass, buffer, and bind group. Fine at current scale; will hurt at ~100
   layers × many bindings. The v1 plan (§4.2) already mandates stable
   binding ids as the diff key — implement diff-based rebuild when scenes
   get big enough to notice (see §5.6). Since 2026-10-12 a rebuild is
   *visually* seamless (clips carry over, §2.1 clip continuity); the cost
   is CPU on the render thread only.
1b. **Render-thread hitches on genuinely new content (not edits).** A
   rebuild that introduces a *new* user pipeline compiles it synchronously
   in `apply_design_scene` (Metal compile, tens of ms — a held frame on
   live), and a newly bound *still* decodes its PNG/JPEG synchronously in
   `open_still` (a 4K PNG ≈ 100 ms); new HAP clips open in parallel,
   ≈ 3–5 ms for seven (§2.4). Param/rate/colour/mod edits hit neither. Fix when it bites:
   compile on a worker thread (wgpu `Device` is `Send + Sync`) and decode
   stills off-thread, swapping in when ready.
2. **Polling IPC loops.** The engine WS connection threads sleep 8 ms
   between polls (~8 ms worst-case added RPC latency for *remote* clients,
   constant low CPU). Local Tauri commands are direct dispatch since the
   collapse — no polling on that path. Replace with blocking reads + a
   wake channel only if profiling ever blames it.
3. ~~Preview pipeline is a dead end~~ — **resolved locally by the collapse
   (2026-07-12)**: the Perform hero samples the composite natively (§1b),
   and JPEG capture is demand-gated on subscriber count. The JPEG path
   remains (deliberately) for remote WS clients and the Prepare canvas
   underlay; upgrade it to binary WS frames only if remote operation ever
   becomes a primary workflow.
4. **`thread::sleep` frame pacing.** Works (winit `ControlFlow::Poll` +
   sleep in `about_to_wait`), but a `WaitUntil`-based schedule would be
   cleaner and free the thread for command handling during the sleep.
   Low priority; commands are drained every iteration anyway.
5. ~~**`ui.slider` values are process-lifetime only.**~~ **Resolved
   2026-07-11** by the session sidecar (§2.5): knobs, masters, and §5.5
   overrides persist across restarts. "Write knobs back into scene.json"
   remains a separate explicit authoring action, never implicit.
6. **`post` bindings and `layerRef` (D5 slow path) are parsed but not
   implemented.** Deferred until a real scene needs cross-layer sampling.
   When it lands (v1 §3.6): a consuming binding samples an earlier layer's
   slow-path FBO as a `sampler2D`; forward references in z-order are rejected
   at load. v1 collapses *pass order* (produce-before-consume) and *blend
   order* (z-index) into one linear plan — the escape hatch, if a low-z layer
   ever needs to read a high-z one, is to defer all blending into a final
   composite pass (a localized compositor change, every routed layer already
   renders to its own FBO first).
7. **Fixed WS port 9123.** A stale engine process blocks respawn. Use port
   0 + handshake, or kill-by-pidfile, when it bites.
8. ~~No regression gate~~ — **resolved 2026-10-12**: §2.10 (`--bench` +
   `scripts/perf_gate.py`).
9. ~~Knob → submit is ~2 frames, not ≤ 1~~ — **resolved 2026-10-12** (perf
   P6). Legs used to tick *before* `acquire` — where the frame waits for
   vsync — so a value read at tick reached the GPU one vsync wait later
   (bench 33.3–34.7 ms p95, real swapchain 32.7 ms). `Core::redraw` now
   runs acquire → clocks + promote step → tick both legs → encode → submit;
   the bench's virtual vsync sits in the same `acquire`. `knob-latency`:
   p95 33.3 → ~18 ms, p50 28 → 12 (a write waits ≤ 1 frame for the next
   tick, then ~1 ms tick → submit); budget 22 ms. On a real swapchain (app,
   2026-10-12): 32.7 → 16.7 ms p95. Clip decode lead is
   unchanged (the decoder `want` was already sent in `encode_uploads`,
   after acquire); stale stays 0 on `clips-7`/`preview-clips`. The
   occluded path (`render_offscreen_frame`) has no vsync wait and is
   unchanged. A `SurfaceError` frame now ticks nothing; the next step
   integrates the gap. `OutputFrame` owns its view (an `Arc` offscreen) so
   the acquired frame borrows nothing across the tick.
10. ~~Dev builds drop frames on the preview JPEG~~ — **resolved 2026-10-12**:
   dependencies build optimized in dev (`[profile.dev.package."*"]` in both
   manifests), the shell subscribes to `preview` only on demand, and the
   JPEG is GPU-downscaled + encoded off-thread (`preview.jpeg` p95 24 ms →
   0.2 ms in dev). render-core itself stays at opt-level 1 in dev (fast
   engine rebuilds), so **a show runs a release build** (`pnpm tauri build`,
   or `pnpm tauri dev --release`); in-app CPU numbers taken under `tauri
   dev` overstate the engine's share.
11. **The §2.6 probe renders clip effects without their clip** (found by
   gig F1, 2026-10-12). `probe.rs` builds each item's bind group with no
   image (`layer_bind_group(…, None)`), so an effect with an `image` input
   samples the 1×1 transparent dummy: its thumbnail is always black and
   `predicted_p95_ms` leaves out the real texture reads. Compile + timing
   of the ALU work are still real. Fix: bind the binding's image (or a
   representative still) in probe items.

12. **A clip-swap TAKE is visible** (found by the §2.10 `take-crossfade`
   scenario, 2026-10-12, gated as regression-only until fixed): 4 TAKEs of
   seven 1080p clips at 60 Hz show **~28 stale clip frames** (median) and
   frames of 34–45 ms. The open hitch is mostly gone since clips open in
   parallel (§2.4; 2026-10-12 A/B: `inbound.cmds` max 11.6 → 4.9 ms,
   `tick.live` max 27.8 → 7.4 ms, worst frame 28 → 8 ms), but the stale
   count is unchanged (28): it is the decode trap, not the open. The design leg's clips
   decode nothing until the fade makes design demanded, so the fade's
   first frames want frames that aren't there. This is the E1b/E2 trap in
   [../TODO/gig-2026-11-live-video.md](../TODO/gig-2026-11-live-video.md);
   fixing it should drive the scenario's stale count to 0 — then make it
   `"stale": "zero"`. Related: in two-leg mode an `effect.upsert` compiles
   its pipeline on the render thread (one 116 ms frame seen; #1b).

**Latent safety/scaling risks (from v1 §9 — not yet bitten at 5–20 layers):**
- **Pathological user-WGSL.** `naga` catches syntax/type errors, but an
  infinite compute loop or huge buffer read can still hang a frame. Keep
  effects fragment-only for now; the swap-on-success isolation already stops a
  *bad compile* from blanking the projector, not a *bad-but-valid* shader.
  §5.11's post-swap probation window is the planned mitigation for the
  bad-but-valid case.
- **Mask-atlas upload cost.** Uploading ~100 antialiased masks at load may be
  slow even under the 256-slice cap. Measure before optimizing (lazy slice
  load / tighter packing if it bites).
- **Colour banding under stacked additive blends.** Mitigated today by the
  `Rgba16Float` composite; watch it if flash+glow+floodFill stacks get deep.
- **HMR GPU-resource leaks.** Stale bind groups after a pipeline swap, video
  staging slots from a closed stream (checked for E1: 120 reloads plateau,
  §2.4), FBO targets from a deleted slow-path
  layer, naga handles after effect removal. Swap-on-success gives every
  pipeline a single owner; verify GPU memory *plateaus* under rapid edit/save
  loops rather than climbing — §2.10's `edit-churn` scenario now checks the
  wgpu counts (0 objects gained over 600 edits, single leg).


---

## 5. Structural roadmap — moved

Lives in [../TODO/render-engine-roadmap.md](../TODO/render-engine-roadmap.md).
All **§5.x** references in this doc resolve there (numbering preserved).

---

## 6. Working agreements for agents in this repo

- **Verify headless first.** `cd render-core && cargo run -- --scene
  examples/phase3_smoke.scene.json --windowed --no-osc`. The example pack is
  built by `python ../test.py layerpack`. Then the shell:
  `cd wzrd-app && WZRD_SCENE=... pnpm tauri dev` (the engine compiles in as
  a library — no separate render-core build step for the shell since the
  collapse; without `WZRD_SCENE` it opens the most recent project).
- **Anything a `Core` owns must die with it.** The shell re-roots the
  engine in-process (project switch / restart, §1b), so a new thread,
  socket, watcher or global a `Core` creates must stop and release on
  drop — a set-once global (like `telemetry::global_bus`) keeps pointing
  at the dead engine.
- **Never block the render thread** — not on swapchains of possibly-hidden
  windows (this now covers *two* swapchains: engine window and native
  preview), not on buffer maps (`PreviewSampler` shows the non-blocking
  pattern), not on channels with unbounded senders, and — TauriHost — not
  on tauri window methods that dispatch to the main thread (§1b deadlock
  rule).
- **Swap-on-success everywhere.** Scene loads, effect compiles, plan
  rebuilds: build the new thing completely, validate, then atomically
  replace. A failed edit must never blank the projector or crash the engine.
  (§5.11 extends this with a post-swap performance probation.)
- **Telemetry: emitter + consumer land together.** No declared-but-dead
  channels. And **per-channel policy lives in `telemetry.rs`, never in a
  host** — `is_sticky()` is a function for a reason: the Tauri host used to
  keep its own copy of the sticky list, which silently went stale the next
  time a channel was added and left the Align tab hydrating from nothing. Any
  new "which channels behave how" rule goes next to it.
- **Two edit paths, one state.** Monaco text and structured editors both
  funnel through `sceneCommit.ts` in the UI and `apply_scene_json` in the
  engine. Don't add a third path.
- **`cargo test` in `render-core/` + `pnpm build` in `wzrd-app/` must pass**
  before declaring a change done. There is no CI; you are the CI.
- **Run the perf gate.** After any change under `render-core/` or
  `wzrd-app/src-tauri/` — or a UI change that subscribes to a high-rate
  channel — run `uv run scripts/perf_gate.py` (~2 min, windowless — it
  runs from any shell, nothing needs to be in front) and paste its table
  into your summary. A FAIL is not done. A cost you added on purpose:
  add/extend a scenario that covers it, then `--update-baseline --reason
  "…"`. The webview's own cost is not gated yet (§2.10).
- When you change engine RPC or telemetry shapes, update **both** this doc
  (§2.3) and the TS types in `wzrd-app/src/{api/ipc.ts, state/store.ts}`.

---

## 7. Rejected approaches (don't re-propose)

Settled negative space — each was considered and consciously left out (full
reasoning in the retired v1 plan §7). Re-open only with a concrete scene
that forces it:

- **Browser render path** (Three.js / R3F / TSL / WebGPU-in-browser) — the
  webview never renders the projector; that's the whole reason for D1.
- **DAG-shaped effect graph** — flat per-layer stacks + one `layerRef` slow
  path only (D5). Full DAG only if a real scene forces it.
- **Embedded JS/TS runtime in the core** (`deno_core`, `boa`, `rusty_v8`) —
  core parses JSON only; TS transpile lives in the webview (D13).
- **TypeScript as the canonical scene format** — `scene.json` is canonical;
  `scene.ts` is an optional human ergonomic mirror, never on any hot path.
- **Recording / video export** — out of scope entirely. No frame tap, no
  encoder, no deterministic-clock swap. Use an external screen-recorder if
  ever needed.
- **Hosting the full ISF runtime** — borrow the ISF *descriptor schema* for
  typed effect inputs (D8), not the GLSL runtime.
- **Ableton Link / external clock sync in v1** — the scene's static
  `transport.bpm` covers v1; pre-computed DAW features can arrive over OSC.
- **Live BPM tracking of any kind** (dropped 2026-07-11, roadmap §5.1) —
  the engine never estimates, taps, or follows tempo. The server's
  `/audio/bpm` stream is deliberately ignored; kicks/onsets are the live
  sync mechanism, `transport.bpm` stays a static scene value.
- **A non-projective base for the §2.8 warp** — fitting a scattered-data
  interpolator (thin-plate spline, MLS, mesh) through the four dragged
  corners gives an affine-plus-bending fit, not a perspective one: straight
  edges bow and the image never sits flat on a keystoned wall. The base stays
  the exact 4-corner homography; every extra handle is a correction on top of
  it. This is the load-bearing decision in the alignment layer.
- **Evaluating the warp model per pixel per frame** — an O(N) RBF sum in the
  final pass is >100 M ops/frame at 64 handles and 1080p, for a field that
  only changes when someone drags a mouse. It is baked into an offset LUT
  instead; per-frame cost is one texel read at any handle count.
- **Fold prevention in the warp** — dragging handles past each other produces
  a non-injective map and visible mirroring. The operator sees it instantly
  on the wall; guarding against it would cost more than it saves.
- **Alignment as scene content** — no MCP verb, no `scene.json` field, never
  per leg, never copied on promote/pull. `scene.json` is what the surface
  does; alignment is where the light lands (§2.5 scope rule).
- **Re-solving the warp in the UI** — the Align canvas draws the real field
  from the engine's solved coefficients (`weights` on the alignment payload).
  A second solver in TypeScript would drift from the one that feeds the
  projector, and the canvas exists precisely to be trusted.
- **Engine-side audio signal conditioning** — smoothing, attack/release,
  min/max normalization of `audio.*` all live in the audio server. The
  engine's only audio-tuning surface is effect-strength params on the
  effects themselves (§5.5).
- **Syphon / Spout** — out of v1; reconsider only for a downstream
  Resolume/VDMX integration.
- **Two-WebSocket localhost layout** — one JSON-RPC WS (`--ws-addr`) for
  remote/MCP; Tauri IPC locally.
- **Color-coded mask atlas / per-mask textures** — one `Texture2DArray<R8>`.

Also deliberately *not* built despite looking tempting: a Frostbite-style
**FrameGraph**. The `PassPlan` is a flat, explicit, inspectable pass list —
pass count is small, z-order pins execution, slow-path FBO lifetime is one
frame. Name it a first-class structure for debuggability, don't add transient-
resource aliasing / auto-scheduling until pass count actually outgrows it
(v1 §3.6).
