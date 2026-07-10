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

Four sibling processes, each independently restartable, communicating only
over wires and files:

```
┌────────────────────────────┐   layerpack/ (pack.json + masks/)   ┌──────────────────────────────┐
│ OFFLINE — Python           │ ───────────────────────────────────▶│ render-core (Rust + wgpu)    │
│ wzrd/ + wzrd_mcp/          │                                     │  - projector window (winit)  │
│ photo→detect→align→darken  │                                     │  - mask atlas + pass plan    │
│ →islands→layerpack         │                                     │  - driver bus + slider bank  │
└────────────────────────────┘                                     │  - file watcher hot-reload   │
                                                                   │  - WS JSON-RPC + telemetry   │
┌────────────────────────────┐   OSC /audio/* → udp 127.0.0.1:9000 │    (only with --ws-addr)     │
│ Audio Feature Server       │ ───────────────────────────────────▶│                              │
│ (~/GitHub/Realtime_PyAudio │                                     └──────────────┬───────────────┘
│ _FFT, separate repo)       │                                                    │ ws://127.0.0.1:9123
└────────────────────────────┘                                     ┌──────────────▼───────────────┐
                                                                   │ wzrd-app (Tauri shell)       │
                                                                   │  spawns render-core, proxies │
                                                                   │  RPC + telemetry to a React  │
                                                                   │  webview (Prepare/Perform/   │
                                                                   │  Debug routes)               │
                                                                   └──────────────────────────────┘
```

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
| D9 | Calibration = in-engine 3×3 homography as the final pass, stored in `scene.json`. |
| D13 | `scene.json` is canonical on disk and on the wire. No TS DSL on any critical path. |
| D14 | Python stays Python (offline segmentation, content generation via `wzrd_mcp`). |
| D15 | Effects are user-authored WGSL (inline or `effects/<name>/`), naga-validated, swap-on-success hot-reload. Built-ins are reference implementations, not a ceiling. |
| P4 | Tauri shell ↔ engine as **subprocess over localhost JSON-RPC WS**, not shared-process winit. Phase 7's MCP wrapper connects to the same WS — the shell is not a privileged client. (A staged collapse plan — one process for a lossless preview — lives in `docs/TODO/single-process-collapse.md`. Its Step 1 **landed 2026-07-10**: engine logic now lives in a host-agnostic `core.rs::Core`, `app.rs` is a thin `WinitHost`, and the static spikes are answered — tao has no occlusion event (needs an `NSWindow.occlusionState` poll in a TauriHost), wgpu-on-tao is rwh-0.6-compatible. The full collapse call stays gated on the runtime spikes, the crash-recovery precondition, and the §5.6 design review.) |

Additive blending (`One + One` into an `Rgba16Float` composite, premultiplied
RGBA out of every effect) is the pixel-level embodiment of the thesis — do
not "fix" it to alpha blending.

### Module map

**`render-core/src/`**

| File | Owns |
|---|---|
| `lib.rs` | `Cli` + `run()` — shared entry for the binary and any embedder. |
| `main.rs` | CLI parse + **tee logger** (stderr + `log` telemetry channel via `telemetry::global_bus`). |
| `core.rs` | **Host-agnostic `Core`** (app-collapse Step 1, 2026-07-10): owns GPU context, pass plan, driver bus, OSC, effect registry, watcher, telemetry, WS server, plus the policies every host must share — the §3.1 occlusion invariant and frame pacing. Two-stage init: `Core::new(&cli)` pre-window, `Core::init_gpu(impl Into<wgpu::SurfaceTarget>, w, h)` when the host has a window. Per-frame host contract: `poll_inbound()` → `pace_frame()` → `redraw()` \| `render_offscreen_frame()`. |
| `app.rs` | Thin `WinitHost` — window creation/fullscreen/display selection, winit event delegation into `Core`, exit decisions. The only winit-aware file besides `lib.rs`. |
| `gpu.rs` | wgpu device/surface (takes any `SurfaceTarget` + explicit size — no windowing-crate dependency), mask atlas upload, composite target, pipeline cache, WGSL composer (`prelude + body + main`), homography pipeline. |
| `compositor.rs` | `PassPlan` — scene → ordered layer passes; per-frame `tick()` (driver eval → uniforms); `record_and_submit()` (present path), `render_offscreen()` (occluded path), `driver_rows()` (telemetry snapshot). |
| `drivers.rs` | Driver bus: `const`, `clock.*`, `audio.band/onset`, `ui.slider`; `SliderBank` (live knob values, written by `param.set`); `Transport` (BPM clock). |
| `effects.rs` | Effect registry: built-ins (`tint`, `hueCycle`, `flash`, `wobble`), project-local + inline WGSL, mtime-based rescan. |
| `scene.rs` | `scene.json` parser + selector resolution. |
| `pack.rs` | Layer-pack loader. |
| `osc.rs` | UDP OSC sink for `/audio/lmh` + `/audio/onset/*` → lock-free `AudioFeatures` atomics. |
| `rpc.rs` | JSON-RPC dispatch: inline read-only methods + queued `EngineCommand`s for mutations; `param.set`/`param.list`; `wgsl.validate` with user-source line remapping. |
| `ws.rs` | tungstenite server, thread-per-connection, telemetry fan-out. |
| `telemetry.rs` | `Bus` (bounded per-subscriber channels + sticky replay), `FpsAccumulator` (honest fps + percentiles), `PreviewSampler` (non-blocking composite readback → JPEG @ ~15 fps), payload types. |
| `watch.rs` | notify-based watcher over the scene file + effects dir. |

**`wzrd-app/`** — `src-tauri/src/{lib,engine,rpc}.rs` (spawn + single-IO-thread
JSON-RPC client + command proxies) and `src/` (React: Zustand store,
`state/sceneCommit.ts` debounced commit path, `SurfaceCanvas`, `MonacoPanel`,
`BindingInspector`, `DriverRack`, `AudioStrip`, `PreviewThumbnail`,
`StatusStrip`, three routes).

---

## 2. Contracts (do not break)

### 2.1 `scene.json` (control contract)

```jsonc
{
  "version": 1,
  "pack": "../../test_results/layerpack/pack",   // resolved relative to the scene file
  "transport": { "bpm": 120 },                   // fallback only — live BPM streams from the audio server (§5.1)
  "bindings": [
    {
      "id": "primary_flash",                     // stable — the hot-reload diff key
      "select": { "tag": "leaves" },             // or { "id": ... } | { "group": ... } | { "all": true }; "pick" selectors land with §5.2
      "effect": "flash",                         // built-in | project-local name | { "inline": true, "wgsl": "...", "inputs": [...] }
      "params": {
        "envelope": { "driver": "audio.onset", "band": "low", "decay": 0.15 },
        "base":     { "driver": "ui.slider", "name": "flash_base", "default": 0.1 },
        "color":    "#ffffff"                    // hex string or [r,g,b(,a)] floats
      }
    }
  ],
  "post": [],                                    // parsed, not yet consumed
  "projectorCalibration": null                   // deprecated here — moving to session.json (§5.3); engine reads the sidecar first
}
```

Drivers: `const(value)`, `clock.bars(n)`, `clock.beats(n)`, `clock.phase(rate)`,
`clock.time`, `audio.band(low|mid|high)`, `audio.onset(band, decay)`,
`ui.slider(name, default)`. No `audio.rms`/`audio.fft` scene drivers in v1
(the server emits `/audio/fft` but the engine doesn't consume it; rms doesn't
exist). The server also streams `/audio/bpm` — consumed by the **Transport**
once §5.1 lands, never exposed as a scene driver; scenes lock to tempo
through `clock.*`. Duplicate/empty binding ids are load errors.

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

**Requirement this puts on the authoring tool (v1 §4.1):** stable ids are
only real if `wzrd.layerpack` can *re-import a new SAM2 segmentation onto an
existing pack's identity table* — overlap-match new blobs to existing ids,
surface splits/merges/new regions for human review — rather than emitting a
fresh pack each time. Without reconciliation, every re-shoot silently
invalidates every scene that targets the pack. Optional sidecar
`identity.json` can hold the map; the runtime contract stays "pack ids are
stable, period."

### 2.3 JSON-RPC surface (WS `--ws-addr`, mirrored 1:1 as Tauri commands)

| Method | Kind | Notes |
|---|---|---|
| `pack.info` | inline | Static pack snapshot (layers, tags, groups, bbox, centroid, z). |
| `scene.getState` | inline | Last-good scene JSON. |
| `scene.load {json}` | queued | Parse + full plan rebuild; on error the previous plan keeps rendering. |
| `scene.reload` | queued | Re-read from disk. |
| `effect.upsert {name, wgsl, descriptor?}` | queued | Writes `effects/<name>/`, registry + watcher pick it up. |
| `effect.remove {name}` | queued | |
| `wgsl.validate {source}` | inline | naga diagnostics remapped to user-source lines (drives Monaco squiggles). |
| `param.set {name, value}` | **inline** | Writes the `SliderBank`; bound `ui.slider` params update next frame. **No rebuild, no disk write — this is the live knob path.** |
| `param.list` | inline | Current slider values. |
| `telemetry.subscribe {channels}` / `.unsubscribe` | per-conn | |

Telemetry channels (all emitted by the engine as of the 2026-07 pass):

| Channel | Rate | Payload |
|---|---|---|
| `preview` | ~15 fps | 320px JPEG of the composite, base64. |
| `fps` | 2 Hz | Honest throughput (frames/wall-second) + p50 frame time. |
| `frame_stats` | 2 Hz | p50/p95/p99 + mask-slice / pipeline / pass counts. |
| `drivers` | 10 Hz | Per binding·param: name, source description, live value, affects-count. |
| `audio` | 30 Hz | L/M/H bands + onset envelopes. |
| `audio_freshness` | 1 Hz + edges | fresh/stale/down (sticky). |
| `connectivity` | 1 Hz | osc / file_watcher / ws status cells (sticky). |
| `hot_reload` | on event | target, ok, elapsed, message (sticky). |
| `log` | on event | Info+ engine log lines (via the tee logger). |

### 2.4 Effect WGSL contract

User code implements `fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>`
(premultiplied RGBA, additive) with access to `state.*` (time, bar/beat
phase, bpm, audio bands/onsets, resolution), `f_param(N)` / `c_param(N)`
(8 scalar + 4 colour slots), and `sample_mask(uv)`. The engine composes
`shaders/effect_prelude.wgsl + body + shaders/effect_main.wgsl` and
naga-validates before pipeline creation; a bad save never blanks the
projector. §5.2 extends this contract with per-layer identity
(`layer_seed`, `layer_index`/`layer_count`, `centroid_uv`, `bbox`) so one
binding can vary organically across its resolved layers.

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
present resumes. **Any future render-loop refactor must keep the property:
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
`App::emit_periodic_telemetry` (drivers 10 Hz, audio 30 Hz, connectivity
1 Hz) and the `main.rs` tee logger. `FrameStats` counts come from
`App::frame_counts()`. Lesson: **a channel isn't done until something emits
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

Now: per-layer tinted overlays are composited **once** into offscreen
canvases and blitted per repaint; picking samples a 256px-wide per-layer
alpha map; labels scale with pack resolution and use an outline; the store
has distinct `selectedLayerId` / `selectedBindingId`, and a canvas click
resolves which bindings target the region (client-side selector resolution)
and aims the inspector at the first one.

### 3.7 Perform route

The driver rack is now playable: `ui.slider` rows are real sliders through
`param.set`; literal numbers get an adaptive-range slider + numeric field
through the debounced scene commit; colours get a picker; clock/audio-driven
rows show live read-only bars (bar only rendered for values in [0,1]).
The preview hero fills available height. The rack rows highlight when their
binding is selected.

Smaller fixes riding along: `read_mask_png` no longer pays a `pack.info`
RPC per mask (cached in Tauri state); redundant `request_redraw` removed;
preview JPEG decode moved off the paint path.

---

## 4. Known weaknesses (accepted for now)

Ranked; none currently show-stopping at 5–20 layers.

1. **Full plan rebuild on every scene edit.** `scene.load` rebuilds every
   pass, buffer, and bind group. Fine at current scale; will hurt at ~100
   layers × many bindings. The v1 plan (§4.2) already mandates stable
   binding ids as the diff key — implement diff-based rebuild when scenes
   get big enough to notice (see §5.6).
2. **Polling IPC loops.** The engine WS connection thread and the Tauri
   engine-io thread each sleep 8 ms between polls (~16 ms worst-case added
   RPC latency, constant low CPU). Human-scale traffic doesn't care;
   replace with blocking reads + a wake channel (or async tungstenite) only
   if profiling ever blames it.
3. **Preview pipeline is JPEG→base64→JSON→Tauri event→`<img>`.** ~15 fps at
   320px is cheap, but it's a dead end for a bigger/faster preview. The
   right next step is binary WS frames (skip base64+JSON) and, later, a
   shared-texture path. Also: the sampler captures even with zero preview
   subscribers — the `Bus` knows subscriber counts; gate capture on demand
   when it matters. (The radical alternative — single-process collapse for
   a lossless Resolume-style preview — is staged in
   `docs/TODO/single-process-collapse.md`; the call is gated on its Step-2
   spikes and the §5.6 design review, and the binary-frames work here is
   on hold until that call — roadmap §5.8 has the gate.)
4. **`thread::sleep` frame pacing.** Works (winit `ControlFlow::Poll` +
   sleep in `about_to_wait`), but a `WaitUntil`-based schedule would be
   cleaner and free the thread for command handling during the sleep.
   Low priority; commands are drained every iteration anyway.
5. **`ui.slider` values are process-lifetime only.** Not persisted on exit;
   a restart resets knobs to scene defaults. Resolution is designed: the
   session sidecar (§5.3) snapshots knob + master state across restarts;
   "write knobs back into scene.json" stays a separate explicit authoring
   action, never implicit.
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
8. **No GPU-side timing.** Frame stats are CPU wall-clock; wgpu timestamp
   queries would attribute cost per pass. Add when shader complexity grows.

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
  staging slots from a closed stream, FBO targets from a deleted slow-path
  layer, naga handles after effect removal. Swap-on-success gives every
  pipeline a single owner; verify GPU memory *plateaus* under rapid edit/save
  loops rather than climbing.

---

## 5. Structural roadmap — moved

Lives in [../TODO/render-engine-roadmap.md](../TODO/render-engine-roadmap.md).
All **§5.x** references in this doc resolve there (numbering preserved).

---

## 6. Working agreements for agents in this repo

- **Verify headless first.** `cd render-core && cargo run -- --scene
  examples/phase3_smoke.scene.json --windowed --no-osc`. The example pack is
  built by `python ../test.py layerpack`. Then the shell:
  `(cd render-core && cargo build) && cd wzrd-app && WZRD_SCENE=... pnpm tauri dev`.
- **Never block the render thread** — not on swapchains of possibly-hidden
  windows, not on buffer maps (`PreviewSampler` shows the non-blocking
  pattern), not on channels with unbounded senders.
- **Swap-on-success everywhere.** Scene loads, effect compiles, plan
  rebuilds: build the new thing completely, validate, then atomically
  replace. A failed edit must never blank the projector or crash the engine.
  (§5.11 extends this with a post-swap performance probation.)
- **Telemetry: emitter + consumer land together.** No declared-but-dead
  channels.
- **Two edit paths, one state.** Monaco text and structured editors both
  funnel through `sceneCommit.ts` in the UI and `apply_scene_json` in the
  engine. Don't add a third path.
- **`cargo test` in `render-core/` + `pnpm build` in `wzrd-app/` must pass**
  before declaring a change done. There is no CI; you are the CI.
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
- **Ableton Link / external clock sync in v1** — the server-streamed
  `/audio/bpm` + the scene's fallback BPM cover v1 (§5.1); pre-computed DAW
  features can arrive over OSC.
- **In-engine BPM detection / tap tempo** — the audio server owns tempo and
  streams `/audio/bpm`; the engine follows it (§5.1) and never estimates,
  taps, or smooths tempo itself.
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
