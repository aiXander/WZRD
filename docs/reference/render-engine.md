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
                                    │    borderless native preview child window        │
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
  frame for *both* the engine window and the preview window, upholding
  §3.1 for both swapchains.
- **Render-thread rule (deadlock):** the render thread must never call a
  tauri window method that dispatches to the main thread (`inner_size`,
  `set_*`…) — sync Tauri commands run *on* the main thread and block on
  render-thread replies, so that's a deadlock. Sizes flow in via `Resized`
  events through shared state (`SizeState`).
- **Native preview (Step 3):** a borderless, hidden-by-default tauri
  window (`label: "preview"`) attached as a **macOS child window** of the
  main window (`addChildWindow:ordered:` — follows drags, stacks above the
  webview). React's `NativePreview` component (Perform hero) measures its
  slot and pushes CSS-px bounds through the `preview_set_bounds` command;
  the backend converts to physical screen coords and moves the child
  window; the render thread blits a composite onto it via
  `gpu::PreviewTarget` — the final-pass pipeline with the §2.8 warp off
  (1×1 zero dummy LUT, `adjust.w = 0`).
  Since §2.6 the preview has a LIVE⇄DESIGN source toggle: DESIGN (the
  authoring default) shows the design composite un-mastered (§5.4
  convention); LIVE shows the live composite with the real
  brightness/saturation masters (still no calibration warp — it only
  reads right on the physical surface). Lossless, full-rate, same GPU
  device, zero readback.
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
- **Env knobs:** `WZRD_DISPLAY=<idx>` → engine window borderless
  fullscreen on that monitor (default: decorated window at pack
  resolution); `WZRD_SPIKE=panic|device_loss` → deliberate render-thread
  crash ~5 s after launch (test hook, never in a show).
- **Known warts:** (1) after an engine crash, SIGTERM is a no-op (the
  signal-hook flag's poller is dead) — quit via Cmd+Q/window close, which
  still works; (2) the `NativePreview` placeholder div assumes a
  height-constrained slot when honouring pack aspect; (3) `engine.rs`
  spawns one short-lived helper thread per *queued* RPC so the request
  timeout holds even if the render thread wedges — inline methods dispatch
  on the caller.

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
| `core.rs` | **Host-agnostic `Core`** (app-collapse Step 1, 2026-07-10): owns GPU context, pass plan, driver bus, OSC, effect registry, watcher, telemetry, WS server, session persistence (§5.3 restore-at-boot, debounced write, SIGTERM/SIGINT snapshot → `exit_requested()`), plus the policies every host must share — the §3.1 occlusion invariant and frame pacing. Two-stage init: `Core::new(&cli)` pre-window, `Core::init_gpu(impl Into<wgpu::SurfaceTarget>, w, h)` when the host has a window. Per-frame host contract: `poll_inbound()` → `pace_frame()` → `redraw()` \| `render_offscreen_frame()`. Step-2/3 additions: `control_channel()` (hands an embedding host the WS server's `RpcContext` + command sender — one dispatch path for every consumer), `attach_preview_surface`/`resize_preview_surface`/`set_preview_visible` (native preview; host owns §3.1 for it), `spike_force_device_loss()` (crash-spike test hook). The command channel + `RpcContext` now always exist (headless just never sends). |
| `app.rs` | Thin `WinitHost` — window creation/fullscreen/display selection, winit event delegation into `Core`, exit decisions. The only winit-aware file besides `lib.rs`. |
| `gpu.rs` | wgpu device/surface (takes any `SurfaceTarget` + explicit size — no windowing-crate dependency), mask atlas upload, per-leg composite targets (§2.6: live always, design in two-leg mode), pipeline cache, WGSL composer (`prelude + body + main`), final-pass pipeline (`encode_final`: live×design lerp by the promote mix → masters → §2.8 warp) and `WarpTarget` (the alignment LUT + its bake pipeline). Step 3: keeps `instance`/`adapter` so `PreviewTarget` (second swapchain; §5.6 source toggle — one bind group per leg, LIVE renders with real masters, DESIGN neutral) can attach at runtime; `render_preview()` self-heals `Lost`/`Outdated`. Its test module holds the two GPU-backed guards worth knowing about: `baked_lut_matches_the_cpu_model` and `final_pass_samples_through_the_warp` (both skip cleanly with no adapter), plus a naga parse+validate of the output shaders so a WGSL typo fails on the desk rather than at device creation. |
| `alignment.rs` | §2.8 alignment layer: `AlignmentDoc` (the `alignment.json` schema), `homography_from_corners` (Heckbert) + analytic 3×3 inverse, the Wendland C² kernel and its LU solve, `AlignmentState` (solved doc + dirty stamps, shared with the render thread and the RPC surface), `TestPattern`, bake-uniform packing, and the boot migration from the legacy calibration matrix. Talks to no GPU. |
| `probe.rs` | §5.6 shader pre-flight probe: `ProbeThresholds` (A/B atomics, sidecar-persisted), `ProbeSession` (half-res interleaved probe frames, overhead-calibrated full-res p95 prediction, pessimistic driver values), three-band verdict + JPEG thumbnail. |
| `compositor.rs` | `PassPlan` — scene → ordered layer passes; per-frame `tick(&mut)` (driver eval → uniforms, §5.2 pick re-rolls via `active` flags); `record_and_submit()` (present path), `render_offscreen()` (occluded path), `driver_rows()` (telemetry snapshot). Owns the stable hashes (`fnv1a`/`seed01`/`pick_choice`) behind layer_seed + pick determinism. |
| `drivers.rs` | Driver bus: `const`, `clock.*`, `audio.band/onset`, `ui.slider`; `SliderBank` (live knob values, written by `param.set`); `Masters` (§5.4 operator globals: brightness/speed/saturation/audioListen, atomics written inline by `master.set`); `Crossfade` (§5.4 engine-wide crossfade-time master — default promote fade in seconds, 0–30 s); `ParamOverrides` (§5.5 per-binding scalar override table); `Transport` (BPM clock — integrates `time += dt·speed` per frame so the speed master bends time instead of jumping it); `PickRate` (§5.2 transport-locked pick cadence). |
| `session.rs` | §5.3 session sidecar: `SessionFile` load/save (atomic temp+rename), `session_path` (`session.json` next to the scene), the shared dirty stamp the debounced write keys off. |
| `effects.rs` | Effect registry: built-ins (`tint`, `hueCycle`, `flash`, `wobble`), project-local + inline WGSL, mtime-based rescan. |
| `scene.rs` | `scene.json` parser + selector resolution. |
| `pack.rs` | Layer-pack loader; per-slice uv geometry (`LayerGeom`) from the manifest or the mask bytes (§5.2); §2.2 identity sidecar (`IdentityFile` lenient load/merge, strict `apply_identity_delta`, atomic `save_identity`). |
| `osc.rs` | UDP OSC sink for `/audio/lmh` + `/audio/onset/*` → lock-free `AudioFeatures` atomics. |
| `rpc.rs` | JSON-RPC dispatch: inline read-only + live-tuning methods (`param.set` both forms, `master.set`, `hello`, `changes.list`) + queued `EngineCommand`s for mutations (`scene.load` incl. `base_rev` CAS, `effect.*`, `identity.setGroups`, `session.save`); `wgsl.validate` with user-source line remapping; §2.7 `Actor` (per-connection identity) + `PackInfoCell` (swappable `pack.info` snapshot). |
| `ws.rs` | tungstenite server, thread-per-connection, telemetry fan-out. |
| `telemetry.rs` | `Bus` (bounded per-subscriber channels + sticky replay + `subscriber_count`), `FpsAccumulator` (honest fps + percentiles), `PreviewSampler` (non-blocking composite readback → JPEG @ ~15 fps — **demand-gated since Step 3**: capture only runs with ≥1 `preview` subscriber), §2.7 `ChangeLog` (design rev + boot epoch + change ring), payload types. |
| `watch.rs` | notify-based watcher over the scene file + effects dir. |

**`wzrd-app/`** — `src-tauri/src/{lib,engine,rpc}.rs` (in-process TauriHost —
see §1b: engine window + render thread + telemetry fan-in + preview child
window; `rpc.rs` command wrappers are unchanged from the subprocess era) and
`src/` (React: Zustand store, `state/sceneCommit.ts` debounced commit path,
`SurfaceCanvas` (still consumes the JPEG `preview` channel as its canvas
underlay), `NativePreview` (Step-3 native hero on Perform), `MonacoPanel`,
`BindingInspector`, `DriverRack`, `MastersRow`, `AudioStrip`, `StatusStrip`,
`WarpCanvas` + `state/alignment.ts` (§2.8 — its own commit path, deliberately
not `sceneCommit`) + `state/warpMath.ts` (the §2.8 model on the CPU, fed by the
engine's solved coefficients, for drawing the field and snapping edge clicks),
four routes). Numeric const rows in the driver rack tune through the §5.5
override path (amber = overridden, ↺ clears back to the scene value);
colours still go through the debounced scene commit. §2.7 reverse-sync:
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
`ui.slider(name, default)`. No `audio.rms`/`audio.fft` scene drivers in v1
(the server emits `/audio/fft` but the engine doesn't consume it; rms doesn't
exist). The server also streams `/audio/bpm` — **deliberately ignored**
(live BPM tracking dropped 2026-07-11, roadmap §5.1): `transport.bpm` is a
static scene value, and live musical energy reaches scenes through
`audio.onset`/`audio.band`, not tempo. Duplicate/empty binding ids are load
errors.

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
| `scene.getState {leg?}` | inline | Last-good scene JSON. §5.6: `leg: "design"` (default — reads follow design) \| `"live"`. Reply carries `{epoch, rev}` so read-modify-write callers can CAS. |
| `scene.load {json, base_rev?}` | queued | Parse + plan rebuild **on the design leg** (§2.6); new pipelines are probe-gated, so the reply can carry a `probe` report. `base_rev` (§2.7): rejected with "design moved to rev N … re-read and retry" when the design rev moved since the caller's read. On error the previous design plan keeps rendering; live is never touched. Headless single-leg: applies to live as before. |
| `scene.reload` | queued | Re-read from disk (same design-leg routing). |
| `promote {fade_ms?, quantize?}` | queued | §2.6 — crossfade projector live→design over `fade_ms` (default: the §5.4 crossfade-time master, ms), then live adopts design's plan (pointer swap). `quantize: "bar"` (default) starts the fade on the next bar boundary; `"now"` immediately. Pending promote replaced by a newer one; **rejected while a fade is ramping**. |
| `pull` | queued | §2.6 — hard-copy live's scene back into design. Rejected mid-ramp; cancels a pending promote + any in-flight probe apply. |
| `preview.setSource {source}` | queued | §2.6 — `"live"` \| `"design"`: which composite the native preview blits. |
| `probe.setThresholds {a_ms, b_ms}` | inline | §2.6 probe bands, A < B, ms of predicted full-res p95. Sidecar-persisted (venue state). |
| `probe.getThresholds` | inline | Current `{a_ms, b_ms}`. |
| `effect.upsert {name, wgsl, descriptor?}` | queued | §2.7 verdict-in-reply: naga-validates **before** touching disk (bad WGSL → line-mapped diagnostics in the reply, nothing written), then writes `effects/<name>/`, rescans the registry (so the watcher echo of the self-write dedupes to nothing), and re-applies the design scene — the reply defers through the §2.6 probe and carries verdict + thumbnail + `{epoch, rev}` when the scene binds the effect. |
| `effect.remove {name}` | queued | |
| `effect.describe {name?}` | queued | §5.5 — input descriptors (type, default, `min`/`max`/`step`/`unit`/`widget`) for one effect, or the whole catalog (built-ins included) when `name` is omitted. |
| `wgsl.validate {source}` | inline | naga diagnostics remapped to user-source lines (drives Monaco squiggles). |
| `telemetry.channels` | inline | The list of live channel names (`telemetry::ALL_CHANNELS`). |
| `param.set {name, value, leg?}` | **inline** | Writes the target leg's `SliderBank` (§2.6 — `leg` default `"design"`; the UI passes the deck toggle's leg); bound `ui.slider` params update next frame. **No rebuild, no disk write — this is the live knob path.** |
| `param.set {binding, param, value, leg?}` | **inline** | §5.5 override form — pins any *scalar* param on a binding (const or driver output alike) in the target leg's override table; `value: null` clears. Same zero-rebuild property; survives plan rebuilds as long as a scalar param with that name exists (the carry-forward rule). |
| `param.list {leg?}` | inline | `{ sliders, overrides, leg }` — the target leg's knob values + override table (design default). |
| `master.set {name, value, leg?}` | **inline** | §5.4 — `brightness` \| `speed` \| `saturation` \| `audioListen`, per leg since §2.6 (design default); plus `crossfade` (engine-wide crossfade-time master in seconds, 0–30 — `leg` ignored). Clamped, applied next frame, echoed on the sticky `masters` channel. Not reachable from scene.json by design. |
| `master.list` | inline | Both legs' masters: `{ live: {...}, design: {...} }`. |
| `session.save` | queued | §5.3 — write the session sidecar now (also happens debounced after master/knob changes and on SIGTERM/SIGINT/window close). |
| `alignment.get` | inline | §2.8 — the full document plus `{output: [w,h], points_max, test_pattern}` and the solved `weights` (read-only derived state, so a client can draw the real field without re-solving; never persisted, ignored on input). |
| `alignment.set {enabled?, background?, corners?, points?}` | **inline** | §2.8 partial merge, validated (exactly 4 corners; ≤64 points; finite coords; radius > 0). Rejects with a prescriptive message and the previous alignment keeps rendering. Two behaviours worth knowing: sending `corners` **alone** carries the extra handles with the content (the engine recomputes their dest positions), and a point with **no `anchor`** is anchored at the current field, so adding a handle doesn't move the image. No `base_rev` CAS — one human editor at a time, last-write-wins is right for a drag. |
| `alignment.reset` | inline | §2.8 — identity corners, no handles, black background. Leaves `enabled` alone. |
| `alignment.setTestPattern {pattern}` | inline | §2.8 — `none` \| `grid` \| `border` \| `corners`, generated in **source** space so it warps with the content. Runtime-only, never persisted. |
| `telemetry.subscribe {channels}` / `.unsubscribe` | per-conn | |

Telemetry channels (all emitted by the engine as of the 2026-07 pass):

| Channel | Rate | Payload |
|---|---|---|
| `preview` | ~15 fps | 320px JPEG of a composite, base64 — the **design** composite in two-leg mode (§2.6 blanket leg rule), live headless. **Demand-gated**: captured only while ≥1 subscriber listens. Consumers: remote WS clients + the webview's Prepare canvas underlay. The Perform hero uses the native preview surface instead (§1b). |
| `fps` | 2 Hz | Honest throughput (frames/wall-second) + p50 frame time. |
| `frame_stats` | 2 Hz | p50/p95/p99 + mask-slice / pipeline / pass counts (counts follow the **design** plan in two-leg mode — §2.6). |
| `drivers` | 10 Hz | Per binding·param: name, source description, live value, affects-count, `overridden` (§5.5 — value then reports the override). Rows come from the **design** plan in two-leg mode. |
| `audio` | 30 Hz | L/M/H bands + onset envelopes. |
| `audio_freshness` | 1 Hz + edges | fresh/stale/down (sticky). |
| `connectivity` | 1 Hz | osc / file_watcher / ws status cells (sticky). |
| `hot_reload` | on event | target, ok, elapsed, message, **`probe`** (§2.6 — `{compiled, predicted_p95_ms, band, thumbnail_b64, verdicts[]}` when a pre-flight probe ran; sticky), plus the §2.7 correlation stamp `{epoch, rev, actor}` (for push consumers — author verdicts arrive in the RPC reply, never via sticky replay). |
| `changes` | on design mutation | §2.7 — one `ChangeEntry` `{epoch, rev, ts_ms, actor, facet, summary}` per design mutation (sticky; ring depth 32, backfill via `changes.list`). Facet map is static in the emit sites: scene applies → `bindings`, effect upserts/removes → `effects`, `identity.setGroups` → `layers`. The webview re-pulls the affected facet on any entry with `actor != "ui"`. |
| `log` | on event | Info+ engine log lines (via the tee logger). |
| `masters` | on `master.set`, promote/pull control copies + startup | §5.4/§2.6 — both legs plus the engine-wide crossfade master: `{live: {brightness, speed, saturation, audioListen}, design: {...}, crossfade}` (sticky). |
| `deck` | on promote/pull/preview transitions + ~10 Hz while ramping | §2.6 snapshot `{promote: idle\|pending\|ramping, mix, fade_ms, quantize, preview_source, two_leg}` (sticky). |
| `alignment` | at boot, on every accepted mutation, on output resize | §2.8 — the whole document plus `weights` and `{output, points_max, test_pattern, solve_ok}` (sticky, so the Align tab hydrates from `lastPayload` and an external camera script sees the UI's edits). |

### 2.4 Effect WGSL contract

User code implements `fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>`
(premultiplied RGBA, additive) with access to `state.*` (time, bar/beat
phase, bpm, audio bands/onsets, resolution), `f_param(N)` / `c_param(N)`
(8 scalar + 4 colour slots), and `sample_mask(uv)`. The engine composes
`shaders/effect_prelude.wgsl + body + shaders/effect_main.wgsl` and
naga-validates before pipeline creation; a bad save never blanks the
projector.

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
  The DeckBar's FADE fader (logarithmic, CUT at the bottom) drives it.
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
  projector and the preview toggle's LIVE position show live. Live is
  immutable except via `promote`.
- **Both legs tick every frame** (shared transport/audio bus — bar-quantized
  promotes land in sync; §5.2 picks are stateless hashes, so the legs pick
  identically). Design's **render passes are demand-gated**: they run only
  when the preview toggle sits on DESIGN, ≥1 WS `preview` subscriber exists,
  or a promote is pending/ramping. An idle design leg costs zero GPU.
- **Promote = crossfade, then pointer swap.** The final pass
  samples both composites and lerps by a `mix` uniform ramped over
  `fade_ms` (wall time). On completion live **adopts design's already-built
  plan** (zero rebuild on the projector leg) and design rebuilds from the
  same scene JSON (pipelines cache-hit — buffers/bind groups only).
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
- **Preview topology: one window, source toggle.** The single native
  `PreviewTarget` samples either composite with **that leg's own
  brightness/saturation masters** and no §2.8 warp (the alignment warp only
  reads right on the physical surface) — WYSIWYG for the leg
  you're driving. `preview.setSource` flips it; the UI toggle lives in
  Perform's deck bar.
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
`.[engine]` → `websockets`, tools default-off in `server.py`, enabled in the
local `tools_config.json`, structurally absent from the Modal image). Setup
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
stays **unwarped** (§2.6 convention — the calibration warp only reads right
on the wall), which is why the preview blit binds a 1×1 zero dummy LUT and
writes `adjust.w = 0`.

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

**Control surface.** All three verbs are inline — a drag must not queue
behind the render thread and none of them rebuild a plan. State lives behind
a small mutex in an `Arc<AlignmentState>` shared with the render thread;
writers clone a snapshot out and never hold the lock across work (§1b).

**UI (the Align route, ⌘2).** SVG over dest space padded ~20% on every side,
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
photographic underlay behind the grid is the `preview` JPEG placed with a CSS
`matrix3d` built from the corner homography (a 3×3 maps exactly onto
`matrix3d`, so it comes for free) — CSS cannot express the local
corrections, so the image follows the corner quad only and the label says so.
Grid = truth, image = reference, projector = ground truth. Before the grid
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

Now: per-layer tinted overlays are composited **once** into offscreen
canvases and blitted per repaint; picking samples a 256px-wide per-layer
alpha map; region names live in the status line under the canvas, not
painted at centroids (centroid labels landed outside thin/concave regions);
and the store has distinct `selectedLayerId` / `selectedBindingId`.

The canvas↔inspector link runs **both ways**, on the same client-side
selector resolution (`selectorCovers` in `SurfaceCanvas.tsx`, mirroring
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

The driver rack is now playable: `ui.slider` rows are real sliders through
`param.set`; literal numbers got an adaptive-range slider + numeric field
through the debounced scene commit (since re-routed onto the §5.5 live
override path, 2026-07-11 — scene.json is no longer written implicitly);
colours get a picker; clock/audio-driven rows show live read-only bars (bar
only rendered for values in [0,1]). The preview hero fills available
height, with the §5.4 masters row always visible beneath it. The rack rows
highlight when their binding is selected.

Smaller fixes riding along: `read_mask_png` no longer pays a `pack.info`
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
`BindingInspector.tsx`.

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
   get big enough to notice (see §5.6).
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
  `cd wzrd-app && WZRD_SCENE=... pnpm tauri dev` (the engine compiles in as
  a library — no separate render-core build step for the shell since the
  collapse).
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
