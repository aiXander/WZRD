# WZRD render-engine — system design

## 1. What we are building

A **scene-aware additive projection-mapping engine**.

The "scene" is the segmentation map of a real physical surface (a tree, a rock formation, a building facade). Each segmented region — leaf cluster, trunk, eye, crack, window, pillar — is a first-class addressable layer. Effects bind to layers by *semantic selector* (id / tag / group). Only the regions that change get projected; the rest of the surface stays dark and physically merges with reality.

This is the one thing **no existing VJ tool does**. VDMX, Resolume, MadMapper, TouchDesigner, HeavyM all solve audio, FX hosting, codec decode, projector output, and calibration well — none of them treat per-region semantic segmentation of the physical surface as the central scene primitive. Every other architectural choice in this doc is in service of that single thesis.

### 1.1 End-to-end flow

```
┌─────────────────────────────────────────┐
│ OFFLINE — Python (existing WZRD + new)  │
│  photo → detect → align → darken        │
│        → islands + SAM2 + manual edit   │
│        → wzrd.layerpack                 │
└─────────────────────────────────────────┘
                    │
                    ▼ layerpack/ (scene.json + masks/*.png + refs)
┌─────────────────────────────────────────┐
│ REALTIME — Tauri app                    │
│  ┌────────────────────────────────┐     │
│  │ Rust core (wgpu)               │     │
│  │  loader → Texture2DArray       │     │
│  │  per-layer effect stacks       │     │
│  │  driver bus (audio/MIDI/OSC)   │     │
│  │  composite + homography pass   │     │
│  │  fullscreen output (projector) │     │
│  └────────────────────────────────┘     │
│  ┌────────────────────────────────┐     │
│  │ Webview UI (React + TS)        │     │
│  │  scene authoring, calibration, │     │
│  │  live sliders, audio debug viz │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
                    ▲
                    │ (optional) WebSocket + JSON-RPC
                    │  for remote control / MCP agent
                    ▼
        ┌──────────────────────────┐
        │  MCP / Claude agent      │
        └──────────────────────────┘
```

Two contracts:

1. **Layer pack** — data contract between offline pipeline and runtime (binary assets).
2. **`scene.json` + RPC** — control contract between UI / agent and core. JSON is canonical (D13); the `scene.ts` typed DSL is an optional ergonomic layer for humans that transpiles to JSON before reaching the core.

Everything else is implementation detail behind these two interfaces.

### 1.2 Concrete example scenes

These ground the effect spec — v1 must express all three without writing new shader code per scene.

- **Tree at night.** ~20 leaf clusters, 1 trunk, 1 ground, 1 sky. Leaves: slow per-cluster hue cycle through a palette; all-leaves white flash on bass kick; one random leaf "blooms" (radial gradient from centroid) every 4 bars. Trunk: vertical scroll like sap flow. Ground: green ripple flood-fill from trunk base on kick.
- **Rock formation.** ~8 segmented rocks + background. Each rock independent low-frequency Perlin color drift. A specific rock "wakes up" on a MIDI note (bright pulse propagating across that rock's mask). Background plays a pre-rendered cloud HAP clip masked to its region.
- **Building facade.** Pillars, windows, doors, roof. Building motion (ripples, sway), characters appearing and walking across mask regions, story beats triggered by MIDI cues.

Same engine, same primitives — only the layer pack and bindings change.

---

## 2. Load-bearing decisions

These are committed. Anything not on this list is still flexible.

| # | Decision | Why |
|---|---|---|
| D1 | **Native render core** in Rust + wgpu | Browser hits decoder caps (~4–6 concurrent streams), no HAP, compositor latency floor, no exclusive fullscreen, no zero-copy video→GPU. Native bypasses all four. Same scene file on macOS (Metal) and Linux (Vulkan). |
| D2 | **Tauri app** wraps Rust core + React/TS webview UI | Single binary, native multi-monitor + exclusive fullscreen, native IPC (no localhost WebSocket for the local case), webview hosts the TS toolchain we already want for scene authoring. Remote control still available as a JSON-RPC WebSocket the core opens optionally (MCP, collaboration). |
| D3 | **Layer pack** is the offline↔runtime contract | Versioned (`version: 1`), schema in §4.1. WZRD Python owns segmentation; the runtime never re-segments. |
| D4 | **Masks stored as `Texture2DArray`** (R8 per slice, **256-slice hard cap**) | Real projects sit well under 100 layers; 256 is the ceiling we design for. Single-bind compositing, soft antialiased edges, no UV bleeding. If a scene genuinely needs >256 we'll re-evaluate, not pre-pay. |
| D5 | **Flat per-layer effect stacks, with prior-layer FBOs as bindable inputs** | Stacks stay flat (no DAG), but the compositor exposes any earlier layer's slow-path offscreen FBO to subsequent layers as a `sampler2D` uniform. Enough routing to drive a leaf's displacement from the trunk's sap luminance, without DAG semantics. Full DAG only if a real scene forces it. |
| D6 | **Display-space (px) authoring, normalized UV in shaders** | Authors and LLMs think in pixels; shaders compute in 0–1. |
| D7 | **Selectors over hard-coded ids** in bindings; layer `id`s are *semantic and authored*, not derived from segmentation output | `{ tag: 'leaves' }` survives re-segmentation. Layer `id` ("trunk", "leaf_clusterA") is a semantic name assigned by the human/agent during pack authoring — it persists across re-shoots even if the underlying SAM blob changes, splits, or merges. Mask file paths are an implementation detail; `id`, `tags`, `group`, `parent` are the stable identity surface. Resolved at load time against pack metadata. See §4.1 for the authoring-side contract. |
| D8 | **ISF schema borrowed as effect-declaration spec**, engine runs WGSL natively | ISF's JSON input header is the right shape ("every input is a typed bindable parameter"). We adopt the *spec*, not the GLSL runtime. WGSL is hand-written or transpiled from ISF GLSL via `naga`. |
| D9 | **Calibration = in-engine 4-point homography**, as the final compositor pass | One mental model, one tool. Stored in `scene.json` under `projector_calibration`. |
| D10 | **TypeScript on the JS side** — React UI and the optional `scene.ts` DSL | Authors who prefer typed editing get Monaco autocomplete + typechecked selectors/effect params. Agents and headless clients skip TS entirely and emit `scene.json` directly (D13). TS is an ergonomic human layer, never a critical-path step. |
| D12 | **HAP / HAP-Q is the video path for many concurrent layers**, with H.264/HEVC hardware decode as the fallback for single-stream content | Hardware decoder cap is the bottleneck for the 10×1080p target; HAP sidesteps it by storing GPU-native DXT/BC blocks. Disk-bound, not silicon-bound. |
| D13 | **`scene.json` is the canonical scene format** on disk and on the wire; `scene.ts` is an optional human-only DSL transpiled by the webview | Published JSON Schema is the single contract. Agents, remote clients, MCP, file watcher, the core itself — all read JSON. Headless runs need no UI process, no Vite, no transpile step. `scene.ts` exists purely as an ergonomic typed-editing surface for humans; it never has to exist for the engine to run. Core stays small (no embedded JS runtime, no TS parser inside Rust). |
| D14 | **Existing Python (`wzrd/` + `wzrd_mcp/`) stays in Python**, as a separate process talking MCP over HTTP | The Python code is offline work (segmentation, surface prep, cloud content generation via FAL/Kling/nano-banana). Latency-insensitive, mature, OpenCV/scikit-learn-bound. Porting to Rust is ~months for zero benefit. The Tauri app and the Python MCP server are two cooperating processes. |
| D15 | **Effects are user-authorable WGSL modules, not a fixed library** | The shipped built-ins (`hueCycle`, `flash`, …) are starting points and reference implementations — *not* the boundary of what's expressible. Authors and LLMs drop a `.wgsl` + JSON descriptor into the project's `effects/` folder (or embed WGSL inline in a binding) and the engine hot-loads it. `naga` validates at load; errors surface as messages, not crashes. The LLM's primary creative surface is **writing real shader code**, not picking from a menu of pre-built parameter slots. |

---

## 3. Architecture

### 3.1 Single app, two halves

```
┌────────────────────────────────────────────────────────────┐
│  wzrd-render (Tauri app, single binary)                    │
│                                                            │
│  ┌──────────────────────┐    ┌────────────────────────┐    │
│  │ Rust core (wgpu)     │◀──▶│ Webview (React + TS)   │    │
│  │  - render thread     │ IPC│  - scene.ts editor     │    │
│  │  - audio thread      │    │  - binding/slider UI   │    │
│  │  - decode thread(s)  │    │  - calibration UI      │    │
│  │  - I/O thread        │    │  - audio-debug viz     │    │
│  │  - own native window │    │  - preview thumbnails  │    │
│  │    on projector      │    └────────────────────────┘    │
│  └──────────────────────┘                                  │
│           ▲                                                │
│           │ optional WebSocket :PORT (JSON-RPC 2.0)        │
│           ▼                                                │
└────────────────────────────────────────────────────────────┘
            ▲
            │
   MCP / Claude agent, remote control, collaboration
```

The render core owns *its own* native wgpu window on the projector display — it does not render through the Tauri webview. The webview owns the control UI window on the operator's display. This keeps the projector output free of compositor frames, browser chrome, and UI redraws.

Tauri IPC (`invoke` / `emit`) is the local control channel. The same logical surface is exposed as JSON-RPC over an optional WebSocket for remote control and the MCP agent — same method names, same schemas (D13 implies the UI side is the only thing that transpiles `scene.ts`; remote clients send JSON directly).

### 3.2 Native render core (Rust + wgpu)

**Crate layout** (proposal — re-shape during Phase 2):

```
render-core/
  Cargo.toml
  src/
    main.rs              # Tauri entry, window creation, mainloop
    rpc/                 # method dispatch (Tauri commands + WS JSON-RPC), schemas
    layerpack/           # scene.json parser, mask atlas loader
    compositor/          # render graph, per-layer pass, homography pass
    effects/             # built-in effects, ISF importer, WGSL generator
    drivers/             # clock, audio, midi, osc, ui-bridge
    video/               # ffmpeg-native + hap decoder, vkvideo path
    state/               # canonical scene state + undo stack
```

**Threading model:**

- **Render thread.** wgpu command encoding + submit. 60–120 Hz target. Owns the projector swapchain.
- **Audio thread.** High-priority OS thread (CoreAudio / WASAPI / ALSA via `cpal`). FFT, RMS, onset per callback. Lock-free ringbuffer to render thread.
- **I/O thread.** Tauri command handlers, WS JSON-RPC server, OSC UDP listener, MIDI ingest, file watcher for `scene.ts` reloads (UI re-transpiles on save).
- **Decode thread(s).** One per active video stream (or pool). Push decoded frames into wgpu textures (zero-copy where possible).

State mutation is single-writer: I/O thread parses incoming commands, computes a diff against canonical state, queues the diff for the render thread to apply at frame boundary. Render thread never blocks on locks.

**Render pass sequence per frame:**

1. Drain pending state diffs from I/O thread.
2. Sample driver values (clock advance, audio frame snapshot, latest MIDI/OSC values).
3. For each visible layer in z-order:
   - Bind mask slice index from `Texture2DArray`.
   - Run effect stack (fast path = fused fragment shader; slow path = per-layer offscreen — see §3.6).
   - Blend into composite buffer.
4. Apply optional post-fx.
5. Apply 4-point homography in final fullscreen-quad pass.
6. Present.

### 3.3 Webview control UI

**Stack:** React + TypeScript + Vite, served by Tauri in release builds (bundled into the binary), or `vite dev` in development.

**Responsibilities (all optional — the render core runs without any UI process):**

- Open / switch layer pack.
- Author / edit `scene.ts` (Monaco editor with the typed DSL).
- **Transpile** `scene.ts` → `scene.json` on save (D13), write to disk and/or send to core via IPC.
- Inline WGSL editor + live `naga` validation for project-local effects (D15).
- Live binding editor (UI for "this `rate` parameter is bound to `clock.bars(8)` → drag to `audio.rms()`").
- Calibration mode (4-point corner drag → homography → core).
- Live sliders, mute toggles, scene-pack swap.
- **Audio-debug visualizer** (FFT bars + onset flashes + tunable thresholds). The most important live-tuning surface — built early.
- Preview thumbnails: core renders a downsampled composite, streams jpeg/h264-encoded frames over a telemetry IPC channel at ~15 fps. Not a substitute for looking at the projector, just enough for remote tweaks.

**What it must not do:** carry the actual projector video, or sit on the agent's critical path. Anything at 60 Hz lives in the core. Anything an MCP agent needs to do, it does by writing files or calling RPC directly — never by booting the UI.

Tauri lets us put the projector window and the control window on different displays from one process — the operator's laptop screen runs the webview, the projector display runs the native wgpu fullscreen window.

### 3.4 Scene config — `scene.json` (canonical) + `scene.ts` (optional human DSL)

The scene is one schema with two surfaces:

- **`scene.json`** — the canonical on-disk and on-the-wire format (D13). Strict JSON Schema, versioned, published alongside `rpc.schema.json`. The engine reads this directly. Headless agents, remote clients, the MCP loop, and the file watcher all operate on JSON. **An LLM emits or mutates `scene.json` against the schema — no Vite, no transpile, no UI process required.**
- **`scene.ts`** — *optional* typed DSL for humans. Tauri webview transpiles it to `scene.json` on save. Gives Monaco autocomplete, typechecked selectors, typechecked effect params. Pure ergonomics; the engine never sees `.ts`.

**Canonical `scene.json` (the form the agent actually writes):**

```jsonc
{
  "version": 1,
  "pack": "../packs/tree-2026-05-01/",
  "transport": { "bpm": 120 },
  "bindings": [
    {
      "id": "leaves_hue",
      "select": { "tag": "leaves" },
      "effect": "hueCycle",                       // built-in or project-local
      "params": {
        "palette": ["#0a3", "#3c5", "#7e3", "#5a2"],
        "rate": { "driver": "clock.bars", "n": 8 }
      }
    },
    {
      "id": "leaves_flash",
      "select": { "tag": "leaves" },
      "effect": "flash",
      "params": {
        "color": "#fff",
        "trigger": { "driver": "audio.onset", "band": "low" },
        "decay": 0.15
      }
    },
    {
      "id": "leaf_bloom",
      "select": { "tag": "leaves", "pick": "random_each",
                  "rate": { "driver": "clock.bars", "n": 4 } },
      "effect": "floodFill",
      "params": { "from": "centroid", "color": "#fff", "duration": 1.2 }
    },
    {
      "id": "trunk_sap",
      "select": { "id": "trunk" },
      "effect": "scrollPattern",
      "params": { "pattern": "verticalLines", "speed": 0.05, "color": "#3a2" }
    },
    {
      "id": "leaves_shimmer_custom",
      "select": { "tag": "leaves" },
      "effect": { "inline": true, "wgsl": "fn frag(uv: vec2<f32>, t: f32) -> vec4<f32> { /* ... */ }" },
      "params": { "amp": 0.4, "src": { "layerRef": "trunk_sap" } }
    }
  ],
  "post": [],
  "projectorCalibration": null
}
```

**Optional `scene.ts` mirror (humans only):**

```ts
import pack from '../packs/tree-2026-05-01/scene.json';
import { audio, clock, midi, ui } from '@wzrd/drivers';
import { hueCycle, flash, floodFill, scrollPattern } from '@wzrd/fx';

export default defineScene({
  pack,
  transport: { bpm: 120 },
  bindings: [
    { id: 'leaves_hue',   select: { tag: 'leaves' },
      effect: hueCycle({ palette: ['#0a3','#3c5','#7e3','#5a2'], rate: clock.bars(8) }) },
    { id: 'leaves_flash', select: { tag: 'leaves' },
      effect: flash({ color: '#fff', trigger: audio.onset({ band: 'low' }), decay: 0.15 }) },
    // ...
  ],
});
```

**Properties:**

- Selectors: `{ id }`, `{ tag }`, `{ group }`, `{ all: true }`, optionally `{ pick: 'random_each' | 'random_static', rate? }`. Validated against the layer pack at load.
- Drivers are values, not per-frame callbacks — the runtime steps them.
- Effects reference built-ins by string name, project-local effects by path, or inline a WGSL string (D15).
- Any binding param can take `{ "layerRef": "<other-binding-id>" }` to consume that layer's slow-path FBO as a `sampler2D` input (D5).
- File reload is HMR-ish: on `scene.json` save, the core file-watches, diffs bindings by stable `id`, rebuilds only changed ones. Same path applies when the webview pushes a transpiled `scene.ts`.

Stable binding `id`s are required from day one — they're the diff key for hot-reload. No binary save format; the JSON file *is* the save.

### 3.5 Layer pack — runtime view

Schema is fixed (§4.1). Loader uploads `masks/000.png … N.png` as slices of one `Texture2DArray<R8>`. Metadata parsed into a `Layers` table keyed by `id`, with secondary indices by `tag` and `group` for cheap selector resolution.

Masks are antialiased grayscale — soft-edged. Author tool should dilate / feather slightly during export to avoid hard pixel-grid boundaries between adjacent semantic regions.

**Scope of the mask array.** The `Texture2DArray<R8>` is reserved for static, projector-resolution semantic masks — uniform dimensions, single bind, cheap selector sampling. Any *dynamic* per-layer source (HAP frames, hardware-decoded video, generated content, per-layer FBOs) lives in its own `wgpu::Texture` binding and is paired with the corresponding mask slice in the layer's fragment pass. The array is the mask substrate; it is not a general layer-content store.

### 3.6 Effect model

The engine treats effects as **user-authorable artifacts on disk**, not a fixed library (D15). The shipped built-ins are starting points and reference implementations; the LLM and human authors extend the engine by *writing new effect files*, not by choosing from a menu.

Each effect is:

- A **WGSL fragment-shader module** with a known entry point.
- An **ISF-style JSON descriptor** listing typed inputs (`float`, `color`, `bool`, `image`, `audioFFT`, `event`, `vec2`, `prevLayer`).
- Optionally, a **TS factory wrapper** for the `scene.ts` typed DSL. Pure ergonomics; engine-loaded effects skip this.

**Discovery, in precedence order:**

1. **Inline.** A binding in `scene.json` can carry `"effect": { "inline": true, "wgsl": "..." }` directly. The compositor compiles + caches the pipeline. This is the LLM's lightest authoring path — drop a one-off shader into a scene without touching the filesystem layout.
2. **Project-local.** `<project>/effects/<name>/{shader.wgsl, descriptor.json}` is watched at runtime. Saving a file → `naga` validation → hot pipeline rebuild for any binding using that effect. Validation errors surface as messages, not crashes.
3. **Built-in.** Bundled with the engine binary as a baseline (`hueCycle`, `flash`, `floodFill`, `wobble`, `scrollPattern`, `videoClip`, `glow`, `tint`).

The agent loop is "write a `.wgsl` file (or inline the string) + reference it from `scene.json` + watch the projector update" — no compile step, no engine restart.

**Compositor path selection.** At runtime each layer has an **effect stack** — an ordered list of effect instances. The compositor picks:

- **Fast path:** if all effects in a stack are color-only (no spatial displacement, no neighborhood ops, no `prevLayer` input), fuse them into one fragment shader per layer at scene-load time. No offscreen buffer, no extra blit. Common case (`hueCycle` + `flash` + `scrollPattern`).
- **Slow path:** if any effect needs spatial ops (`wobble`, `glow`, `floodFill` with geodesic propagation) or wants to read another layer's output, render the layer into a per-layer offscreen FBO and run the chain as separate passes. The FBO is retained for the rest of the frame so subsequent layers can sample it.

Effects declare their class (color-only / spatial / consumer-of-prevLayer) in their descriptor so the compositor knows which path to use.

**Inter-layer routing (D5).** Any binding can declare `"params": { "src": { "layerRef": "<earlier-binding-id>" } }`. At composition time the compositor binds that layer's slow-path FBO as a `sampler2D` uniform on the consuming effect. This is the "use the trunk's sap luminance to displace the leaves" path — generative feedback loops without a DAG. The compositor rejects forward references in z-order at load time.

This collapses two orderings into one for v1 — *pass order* (dependency topology: produce before consume) and *blend order* (visual z-index: behind before in front). Forcing them to match keeps the pass plan linear and the rejection rule trivial, at the cost of one real artistic case: a low-z background layer can't read a high-z foreground layer's luminance. The architectural escape hatch, if a scene ever forces it, is already implicit in the slow-path: every routed layer already renders to its own FBO before blending, so "decouple pass order (topological) from blend order (z-index) by deferring all blending into a final composite pass" is a localized change to the compositor, not a re-plumb. Not in v1.

**Internal compile step — pass plan, not FrameGraph.** Scene load is not "interpret bindings in a loop per frame." Bindings compile down to an ordered, explicit **pass plan**: per-layer fast-path pipelines (fused fragments), per-layer slow-path pipelines (offscreen FBO + chained passes), declared `layerRef` dependencies, post-fx passes, and the final homography pass. The render thread executes the plan linearly each frame; the plan is rebuilt on scene/effect hot-reload.

This is deliberately *not* a FrameGraph (Frostbite-style transient-resource aliasing, automatic dependency scheduling, lifetime-derived barrier insertion). Pass count is small (≤ ~100 layers × small stacks), z-order pins execution order, slow-path FBO lifetime is one frame, and dependencies are explicit in the scene. The point of naming the plan as a first-class data structure is **debuggability and predictable rebuild** — every pass, input, and output is inspectable, not derived implicitly per frame. If pass count or routing complexity ever outgrows this, *that* is the moment a real frame graph earns its keep; until then it's infrastructure we'd be paying for without using.

**Pipeline lifecycle under hot-reload.** WGSL edits (project files or inline) follow a strict **swap-on-success** protocol: build the new pipeline → `naga` validate → run one offscreen test draw → only then atomically replace the old pipeline reference in the pass plan. The previous pipeline keeps rendering until the swap commits, so a bad save never blanks the projector. The old `Pipeline + BindGroupLayout + bind groups + sampled texture refs` set is dropped *only after* the swap, which is what wgpu's refcounted handles need to actually free GPU memory rather than retaining a live reference inside a stale bind group. Same pattern applies one level up for whole-scene reloads: build the new pass plan side-by-side, swap atomically at a frame boundary, drop the old. This is the answer to the HMR risk in §9 — every pipeline and pass plan has a single owner and an explicit replacement protocol from day one, not retrofitted after the first leak.

**Built-in effect set for v1 (reference implementations, not the boundary):**

- `hueCycle` — palette + rate → HSL hue modulation.
- `flash` — color + trigger + decay → additive pulse.
- `floodFill` — origin + color + trigger + duration → radial SDF (v0); geodesic via compute shader (v1) respecting mask topology.
- `wobble` — amplitude + frequency → UV displacement.
- `scrollPattern` — pattern fn + speed → procedural texture clipped to mask.
- `videoClip` — `VideoSource` ref (HAP or HW-decoded) → sampled, masked.
- `glow` — kernel size + color → blurred copy, additively blended.
- `tint` — color → constant fill (debug / baseline).

These are deliberately a small set. The expected creative path past v1 is *authoring new effects*, not adding more built-ins.

ISF importer comes after v1's built-ins are proven. Target a documented ISF subset (single-pass, no persistent buffers in v0).

### 3.7 Driver bus

A driver is anything implementing `Driver<T>` — produces a value of type `T` per frame, all sharing `tick(frameTime, audioFrame)`. Any parameter slot of type `T` can accept any `Driver<T>`.

Built-in drivers:

- `clock.bars(n)`, `clock.beats(n)`, `clock.phase(rate)` — BPM-aware transport. `Driver<f32>` in [0,1).
- `audio.rms()`, `audio.band(low|mid|high)` — `Driver<f32>`.
- `audio.onset({ band })` — `Driver<Event>`.
- `audio.fft()` — `Driver<&[f32]>` for spectrum-driven effects.
- `midi.cc(n)`, `midi.note(n)` — `Driver<f32>`.
- `midi.noteOn(n)` — `Driver<Event>`.
- `osc.path('/x/y')` — typed by declared path schema.
- `ui.slider(name, [min, max])` — surfaces a knob in the live UI.
- `const(value)` — wrapper for literal parameters.

OSC is a flat UDP listener inside the core (no WebSocket bridge needed in native). OSC is also the **expected transport for pre-computed audio features** (RMS, bands, onsets, FFT bins) when an external DAW or audio-analysis process is the source — covers headless runs, recorded shows, and external-mixer setups without forcing the core to ingest the raw audio stream.

**Transport.** A single canonical clock owned by the core. BPM sources for v1: manual (UI), tap-tempo, audio-derived (onset-based estimator). External clock sync (Ableton Link, MTC, MIDI clock) is **explicitly not in v1** — defer until a real show demands it.

**Narrative timing in v1 = external sequencer.** The driver bus is fundamentally signal-driven (reactive). For *authored* / repeatable performances — "at bar 128 start the character reveal, at 02:31 trigger the facade collapse," the building-facade scene from §1.2 — v1 leans on an external sequencer (Ableton, Bitwig, Reaper) firing MIDI/OSC events at the right musical or wall-clock time. The DAW is already the right tool for sample-accurate cue scheduling; the engine reacts deterministically per incoming event via `midi.noteOn` / `osc.path`.

An in-engine cue/timeline editor (transport-locked event tracks, quantized cues, record/replay) is a real feature, deferred to Phase 6+ when a confirmed show needs it. The driver bus already accepts arbitrary events from OSC and MIDI, so adding an internal cue source later is one more `Driver<Event>` implementation — not a re-plumb.

### 3.8 Video decode

Two paths, picked per asset:

- **HAP / HAP-Q** for the primary workload (many concurrent 1080p layers). Read DXT/BC blocks from disk, upload to GPU each frame. Disk-bandwidth-bound. No mature Rust HAP crate today — Phase 5 work is either FFI to `hap-cpp` or a small Rust port.
- **H.264 / HEVC hardware decode → zero-copy to GPU texture** for single-stream content. Path: `ffmpeg` via `ffmpeg-next` for demux, OS-native decoder (VideoToolbox / NVDEC / VAAPI) for decode, wgpu interop for the texture. `vkvideo` and `cros-codecs` crates are the strongest references; both early but the architecture is right.

ProRes is not in scope. Software H.264 fallback exists but is documented as "won't hit 60 Hz for >1 stream."

**Decode pipeline — never block the render thread.** Both paths share the same staging strategy:

- Each active stream owns a **ring of N mapped `wgpu::Buffer` staging slots** (N = 3 to start; tune empirically). Slots are allocated at stream-open and recycled, never re-mapped per frame.
- The **decode thread** owns the FFI calls and disk I/O. It reads the next compressed frame (HAP DXT block, or NV12/BGRA from the hardware decoder), writes it into the next free staging slot, and pushes a "slot ready" message to the render thread over a lock-free channel.
- The **render thread**, at frame composition, picks the freshest ready slot for each layer and issues a single `copy_buffer_to_texture` to fill the layer's GPU texture. No FFI, no `map_async` await, no disk syscall on the render thread.
- If decode falls behind, the render thread reuses the last good slot — visible as a stutter, never a frame-stall.

The render thread's only contact with video is "copy buffer → texture." Everything expensive happens on the decode thread, decoupled by the ring. This is the rule that makes "10× 1080p HAP layers" actually feasible; without it the PCIe bus plus async buffer mapping easily costs 1–2 frames of latency.

If the HAP FFI path ever proves too volatile, the fallback is a compute-shader DXT decoder running on the GPU — input becomes a raw bytes buffer, decode runs on the same hardware that consumes the texture, FFI surface shrinks to "read file → memcpy."

### 3.9 Calibration

Final pass in the compositor takes a `mat3` homography uniform and warps composite → projector framebuffer. Identity by default.

UI flow:

1. Toggle calibration mode → core overlays a corner grid on the composite.
2. UI sends 4 source points and 4 target points (set by mouse drag on the preview thumbnail or by physical adjustment).
3. Core computes the 3×3 from the 4 point pairs, applies it, persists back to `scene.ts` under `projectorCalibration`.

**Re-shoot workflow** (projector or scene moved slightly): rather than re-segmenting, run the offline `align` step on a new capture against the original reference photo to get a single homography update. Same field in `scene.ts`. This is the "the projector got bumped" recovery path.

### 3.10 Python ↔ Tauri integration (Pattern A)

The existing Python codebase — `wzrd/` (image/video processing) and `wzrd_mcp/` (FastMCP server wrapping those tools plus FAL/Kling/nano-banana cloud generation) — keeps running unchanged. It is the *offline* half of WZRD: it produces the layer packs and the video/image assets that the realtime render core plays back.

**Two complementary RPC surfaces.** An MCP agent (or the Tauri UI) uses both:

| Surface | Owner | Lives in | Job |
|---|---|---|---|
| **render-core RPC** (Tauri commands locally, JSON-RPC over WS for remote/MCP) | New Rust crate (this doc) | Inside the Tauri binary | Realtime: scene.ts edits, bindings, sliders, calibration, transport |
| **wzrd_mcp** (FastMCP / HTTP) | Existing Python (unchanged) | Separate process — local or Modal | Offline: segment, prepare surface, build layer pack, generate content (`texture_flow`, `kling_v25_image_to_video`, `nano_banana_pro`) |

The two never call each other on the hot path. They communicate via:

1. **Files on disk.** Python writes `layerpacks/` and `assets/`; the Rust core reads them. The layer pack (§4.1) is the formal contract; for generated video/image assets the contract is just "URL or local path."
2. **MCP over HTTP** when the Tauri UI wants to *trigger* offline work on demand. E.g. "Generate clouds for this rock" → UI POSTs to `wzrd_mcp` HTTP endpoint → server returns an S3/CDN URL → UI calls `render-core.binding.add({ effect: videoClip(url) })`. Both Rust and TS have MCP client crates.
3. **Shared agent.** An MCP agent (Claude desktop / Claude Code / bundled chat panel) holds both servers in its tool list and orchestrates them: generate → segment → bind.

**Why this works:**

- Either process can be restarted, upgraded, or moved to a remote machine independently. The expensive Modal-hosted GPU tools (TextureFlow) stay deployed exactly as today.
- The Tauri binary stays slim — no embedded Python, no 200 MB+ runtime, no PyInstaller dance.
- The agent loop already works: Claude already uses `wzrd_mcp` today. Phase 7 adds `render-core` as a *second* MCP surface; the same agent now drives both.

**Development workflow:** start the Python server in one terminal (`python -m wzrd_mcp`), start the Tauri app in another (`pnpm tauri dev`). The Tauri UI is configured with the MCP server URL (default `http://localhost:8787`).

**Fully headless agent run.** The render core ships as a usable **standalone binary** (Phase 2 deliverable, before the Tauri shell exists). An agent writes `scene.json` against the published JSON Schema (D13) and any `effects/*.wgsl` files it needs (D15); the core file-watches and reloads; the projector window updates within one frame budget. **No UI process. No Vite. No transpile step.** The agent's critical path is "write JSON + WGSL → file → projector," with the engine as the sole consumer.

The `scene.ts` typed DSL only matters when a human is in the loop — it's a Monaco/typecheck convenience, not part of the agent contract.

**Sidecar packaging (deferred).** When shipping to non-developer users, Tauri can spawn `wzrd_mcp` as a sidecar binary (built via `pyinstaller`) so the user sees one app. Not in v1 — Pattern A (two processes) is the v1 model.

**What changes in the existing Python:**

- `wzrd/` modules: no changes.
- `wzrd_mcp/server.py`, `tools.py`, `fal_tools.py`: no changes.
- New module: **`wzrd/layerpack.py`** (Phase 1). Takes `wzrd.islands` output + mask PNGs + a tags JSON, emits `scene.json + masks/ + references/`. Also exposed as an MCP tool (`build_layerpack`) so the agent can call it.
- `wzrd_mcp/tools_config.json`: add `build_layerpack` once it exists.

### 3.11 RPC / IPC surface

One logical method set, exposed as Tauri commands locally and as JSON-RPC 2.0 over WebSocket for remote / MCP clients. All methods, params, and result types defined in a single `rpc.schema.json` consumed by both Rust (codegen via `typify` or hand-rolled) and TS (codegen via `json-schema-to-typescript`). One source of truth.

Initial set:

- `pack.load(path)` / `pack.info()` — layer pack management.
- `scene.load(json)` / `scene.applyDiff(diff)` / `scene.getState()` — scene management. JSON is canonical (D13); UI transpiles its TS DSL to JSON before calling these. Agents and remote clients construct JSON directly.
- `effect.upsert({ name, wgsl, descriptor })` / `effect.remove(name)` — register or replace a project-local WGSL effect at runtime (D15). The file watcher uses the same code path under the hood.
- `binding.add(...)` / `binding.update(id, ...)` / `binding.remove(id)` — runtime mutation. UI writes back to `scene.ts` for persistence.
- `param.set(bindingId, paramPath, value)` — live slider tweaks. Ephemeral unless persisted.
- `param.bind(bindingId, paramPath, driverSpec)` — change a binding's driver source.
- `transport.setBpm(n)` / `transport.tap()` / `transport.playPause()`.
- `calibration.set(matrix3)` / `calibration.beginPick()`.
- `telemetry.subscribe(['fft', 'fps', 'preview'])` — opens the telemetry channel.

These same methods are the MCP tool surface in Phase 7. The agent does not get a different API.

---

## 4. Contracts

### 4.1 Layer pack — `scene.json`

```jsonc
{
  "version": 1,
  "projector_resolution": [1920, 1080],
  "source_capture": "references/photo.jpg",
  "surface": "surface.png",
  "layers": [
    {
      "id": "trunk",
      "mask": "masks/001_trunk.png",
      "label": "trunk",
      "tags": ["tree", "structure"],
      "bbox": [820, 400, 1100, 1080],
      "centroid": [960, 760],
      "area_px": 80000,
      "parent": null,
      "z": 1
    }
    // ...
  ],
  "groups": [
    { "id": "leaves", "members": ["leaf_002", "leaf_003"] }
  ]
}
```

Directory:

```
layerpack-2026-05-01-tree/
  scene.json
  surface.png            # dark/aligned surface (for preview overlay)
  masks/
    000_background.png   # antialiased grayscale, projector-resolution
    001_trunk.png
    ...
  references/
    photo.jpg            # original capture (for author UI overlay)
    canny.png            # optional, for canny-aligned video clips
```

Owned by `wzrd.layerpack` Python module (Phase 1). Compatible with `wzrd.islands` output. Consumers must check `version` and refuse unknown majors.

**Identity model — semantic ids, not segmentation-derived blobs (D7).** A layer's `id`, `tags`, `group` membership, and `parent` are *semantic*, assigned during pack authoring, and **stable across re-shoots and re-segmentations**. The mask PNG path is allowed to change (`masks/001_trunk.png` → `masks/004_trunk.png` after a re-segment that produces a different file order) — the `id` is not. This is the contract that lets scene bindings survive re-segmentation: bindings reference `id`/`tag`/`group`, never mask filenames or segmentation-tool blob indices.

This puts a real requirement on `wzrd.layerpack`: it must support **re-importing a new SAM2 segmentation against an existing pack's identity table** (map new blobs to existing ids by mask overlap, surface ambiguous cases — splits, merges, new regions — for human review), not just emit a fresh pack from scratch each time. Without this, every re-shoot silently invalidates every authored scene that targets the pack. The authoring tool can store the identity table as a small `identity.json` alongside `scene.json` if it helps reconciliation, but the runtime contract stays "pack ids are stable, period."

### 4.2 Scene config — JSON Schema (canonical) + TS DSL (optional)

**Canonical:** `scene.schema.json` — strict JSON Schema, versioned with the engine. This is what agents validate against and what the core enforces. See §3.4 for the JSON shape.

**Ergonomic mirror:** TypeScript types for the `scene.ts` DSL, generated from the JSON Schema via `json-schema-to-typescript`. Type signature sketch:

```ts
type Selector =
  | { id: string }
  | { tag: string }
  | { group: string }
  | { all: true }
  | (Selector & { pick: 'random_each' | 'random_static'; rate?: Driver<number> });

type EffectRef =
  | string                                        // built-in or project-local name
  | { inline: true; wgsl: string };               // raw shader embedded in scene

type Binding = {
  id: string;                                     // stable, required for hot-reload
  select: Selector;
  effect: EffectRef;
  params: Record<string, ParamValue>;             // typed by the effect's descriptor
};

type SceneConfig = {
  version: 1;
  pack: string;                                   // path to layer pack
  transport: { bpm: number };
  bindings: Binding[];
  post?: Binding[];
  projectorCalibration?: Mat3 | null;
};
```

Bindings have stable `id`s from day one — required for diff-based hot-reload (§8 note 9). The TS types and JSON Schema must stay in lockstep (codegen, single source of truth).

### 4.3 ISF-style effect descriptor

```jsonc
{
  "name": "hueCycle",
  "category": "color",
  "spatial": false,                  // determines fast/slow path
  "inputs": [
    { "name": "palette", "type": "color[]", "default": ["#000"] },
    { "name": "rate",    "type": "float",   "default": 0.1, "min": 0, "max": 10 }
  ],
  "wgsl": "hue_cycle.wgsl",
  "entry": "main"
}
```

ISF GLSL shaders coming in from `editor.isf.video` get a thin importer that emits this descriptor + a `naga`-transpiled WGSL.

### 4.4 JSON-RPC / IPC contract

JSON-RPC 2.0 method names and schemas in `rpc.schema.json`. Same surface across Tauri commands (local) and WebSocket (remote/MCP).

---

## 5. Phasing

**Guiding principle: get a Rust binary on the projector as fast as possible, with the Tauri/React layer deferred until the core is real.** The webview is a control surface; it isn't load-bearing for "see pixels on the wall." Building it last (a) gives a playable engine in weeks not months, (b) proves the headless agent loop (D13) before any UI exists to lean on, and (c) lets us shape the Tauri layer around a known-good core instead of architecting two halves in parallel.

### Phase 0 — clear the slate (minutes)

The old `render-engine/` browser-Three.js prototype shared neither language, paradigm, nor problem with the new architecture (3D shader sphere in R3F vs. native wgpu 2D mask compositor). Deleted rather than refactored. The one piece worth preserving — the `organicShader` WGSL string — survives as an idea in this doc.

Concretely: `rm -rf render-engine/`, `cargo new render-core` for the standalone binary. No Tauri yet.

### Phase 1 — `wzrd.layerpack` Python module (half-day)

Exports the §4.1 format from `wzrd.islands` output + external mask PNGs + a hand-edited tags file. CLI: `python -m wzrd.layerpack <surface> <masks_dir> --tags tags.json -o pack/`. Smoke test in `test.py`. Wrap as MCP tool `build_layerpack` in `wzrd_mcp/tools.py` so the agent can call it. Blocker for everything downstream.

Only new Python work in the build. Everything else in `wzrd/` and `wzrd_mcp/` stays as-is per D14 / §3.10.

### Phase 2 — Minimal playable Rust core, no UI (1 week)

The fastest path to "see pixels move on the projector." A standalone `render-core` binary with **no Tauri, no webview, no TypeScript on the critical path.**

- CLI: `render-core --pack path/ --scene scene.json`.
- `winit` + wgpu fullscreen window on a chosen display index.
- Layer-pack loader → `Texture2DArray` (R8, 256 slices max — D4).
- JSON Schema for `scene.json` (D13) — strict parsing, helpful errors.
- Compositor: per-layer pass, blend in z-order, homography final pass (identity by default).
- One built-in effect: `tint`. End-to-end through the binding pipeline.
- File watcher on `scene.json` → diff bindings by stable `id` → hot-reload.
- macOS first; verify Linux builds compile.

**Deliverable:** edit `scene.json` in any editor, save, projector updates. Boring on screen (flat tints), but the whole spine — pack loading, mask compositing, scene parsing, hot-reload — is real and the **agent loop is already unblocked**.

### Phase 3 — Effects, drivers, user-WGSL (2 weeks)

Build out the effect model so the agent loop becomes genuinely creative. Still no Tauri.

- Effect discovery from disk (D15): project-local `effects/<name>/{shader.wgsl, descriptor.json}` + inline-WGSL bindings. `naga` validation at load; hot pipeline rebuild on file save.
- Built-in effect set per §3.6 (`hueCycle`, `flash`, `floodFill` v0, `wobble`, `scrollPattern`, `glow`).
- Slow-path FBO routing: `layerRef` params bind earlier layers' offscreen textures as `sampler2D` inputs (D5).
- Driver bus: `clock`, `audio` (cpal — RMS, bands, simple onset), `osc` (rosc — covers MIDI-via-OSC and external feature streams), `ui` (stubbed; wired up properly in Phase 4).
- Optional JSON-RPC WebSocket server for remote control (same method set as future Tauri IPC).

**Deliverable:** §1.2 tree scene runs audio-reactive on the projector with no UI ever opened. An MCP agent (or a human with a text editor) can write/edit `scene.json` and `effects/*.wgsl` and see results immediately. **This is the milestone that proves the architectural thesis.**

### Phase 4 — Tauri shell + webview UI (2 weeks)

Now that the core is real, wrap it with the control UI. The core binary keeps working standalone — Tauri is a second front-end, not a replacement.

- Tauri project scaffolded around the existing core crate (`pnpm create tauri-app`).
- Tauri IPC bridges to the same RPC method set as the WebSocket.
- Webview UI: layer list, binding editor, slider rack, calibration corner drag, **audio-debug visualizer** (the most important live-tuning surface).
- Monaco editor for `scene.ts`; webview transpiles to `scene.json` on save (D13).
- Monaco editor for `effects/*.wgsl` with inline `naga` validation.
- Preview thumbnail stream over IPC.

**Deliverable:** the operator's laptop runs the webview, the projector display runs the native wgpu window, all from one Tauri process.

### Phase 5 — Video (1–2 weeks)

- H.264/HEVC zero-copy decode → wgpu texture via `vkvideo` / VideoToolbox bindings. Single-stream first; verify zero-copy.
- HAP decoder (FFI to `hap-cpp` initially, port to Rust if FFI hurts).
- Ring-buffered staging slots between decode thread and render thread (§3.8) — non-negotiable for hitting the 10× 1080p HAP target without frame-stalls.
- `videoClip` effect wired in.

If HAP-on-base-M2 doesn't hit 10× 1080p @ 60 Hz, that's a real datum and we downgrade the layer-count target.

### Phase 6 — Live polish (ongoing)

- Calibration re-shoot flow (§3.9).
- Multi-pack switching at runtime.
- Single-binary release builds (Tauri bundle).
- Per-OS install/launch story.

### Phase 7 — MCP / Claude integration (1 week, unblocked by Phase 3)

The MCP tool surface *is* the RPC surface. Add an MCP server that proxies the documented method set over the JSON-RPC WebSocket, plus a `scene.edit({ instruction })` tool that round-trips a natural-language change into `scene.json` (and optional new `effects/*.wgsl` files) using Claude. The headless engine from Phase 3 is already the deployment target — Phase 7 is the agent-facing wrapper. Test the round trip locally before integrating into the broader WZRD MCP.

**Reading the timeline.** Phases 0–3 (~3–4 weeks total) produce a playable, agent-driven engine with no Tauri. Phases 4+ are improvements on top of an already-working system. If anything slips, slip Phase 4 onward, not the core.

---

## 6. Spikes — verify before locking in

Each small enough to do as a one-session spike before committing the surrounding decisions.

### 6.1 Tauri + wgpu native window on a non-primary display (Mac + Linux)

Confirm we can have one Tauri app process own (a) a webview window on the operator's display and (b) a native wgpu fullscreen window on the projector display, with no compositor frame on the projector path. `winit` claims to handle this; verify on both OSes.

**Must explicitly test live cross-window interaction**, not just static "two windows up at once." On macOS specifically, exclusive-fullscreen wgpu windows on a secondary display interact badly with focus changes — clicking sliders in the React webview can pull the projector window out of fullscreen, cause Spaces to reshuffle, or introduce frame stutter on the projector path. The spike must drag a webview slider for ~30s while watching the projector window for focus-loss, mode-flip, or stutter. **Fallback plan if true exclusive fullscreen misbehaves:** a borderless, non-resizable, non-decorated `winit` window sized to the projector display's screen bounds. Loses any "real exclusive fullscreen" compositor-bypass wins on macOS, but stays stable under operator interaction — which is the actual production constraint.

### 6.2 HAP-on-Rust reality check

Confirm whether `hap-cpp` (vidvox C++ ref impl) can be FFI'd cleanly into a Rust crate and whether the decode path hits sub-frame latency in a wgpu upload context. If FFI is messy, plan an in-Rust port — HAP itself is small.

### 6.3 ISF GLSL → WGSL via `naga`

Take 3 representative ISF shaders from `editor.isf.video`, run through `naga`, see what breaks. Decides how much ISF support we promise vs "write WGSL natively."

### 6.4 10× 1080p HAP smoke on base M2

The hardware target. If a 200-line Rust+wgpu prototype playing 10 HAP files into 10 quads at 60 Hz doesn't hit, every higher-level assumption needs revisiting before Phase 5.

### 6.5 Headless `scene.json` hot-reload + (later) TS transpile round-trip (D13)

Two paths to validate, in order:

1. **Headless (Phase 2).** Run the standalone `render-core` binary with no UI. Edit `scene.json` in any editor → file watcher fires → core diffs against current state → projector updates within one frame budget. This is the agent's critical path; it must work before the webview exists.
2. **Webview (Phase 4).** Edit `scene.ts` in Monaco → webview transpiles to JSON on save → IPC → same diff/apply path. Same result, ergonomic surface for humans.

If headless works and webview works, the agent loop is unblocked end-to-end.

### 6.6 User-authored WGSL effect hot-reload (D15)

Drop `effects/shimmer/{shader.wgsl, descriptor.json}` into a project folder, bind it from `scene.json`, edit the WGSL file, watch the pipeline rebuild without an engine restart. `naga` validation errors should surface as a UI/CLI message, not a crash, and not blank the projector — the previous good pipeline keeps running until the new one is valid. This is the LLM's primary creative surface; verify it actually feels good before promising it.

---

## 7. Things consolidated away from

Prior docs left a lot of optionality. Flattened here:

- **Three.js / TSL / R3F / WebGPU-in-browser** — out of the render path. The Tauri webview still uses React, but it does not render the projector output.
- **`wgslFn` raw shader blobs as the long-term effect format** — out. Effects are TS-declared, ISF-schema-shaped, WGSL-backed.
- **Hosting full ISF runtime** — out. Borrow the schema, not the runtime.
- **Recording / video export** — out of scope entirely. No frame tap, no encoder, no deterministic-clock driver swap. The engine drives a projector in real time; capturing its output is a job for an external screen-recorder if ever needed.
- **DAG-shaped effect graph** — out for v1. Flat per-layer stacks only.
- **OffscreenCanvas / WebWorker / WebCodecs / browser fullscreen** — moot (no browser renderer).
- **Window Management API** — moot (Tauri + winit pick the display).
- **Color-coded mask atlas / individual mask textures** — out. `Texture2DArray` only.
- **`scene.config.js` (camera + objects + materialNode)** — out. Replaced by `scene.json` (canonical) + optional `scene.ts`.
- **TypeScript as the canonical scene format** — out. `scene.json` is canonical on disk and on the wire (D13); `scene.ts` is an optional human-only ergonomic layer.
- **Embedded JS runtime in the core** (`deno_core`, `boa`, `rusty_v8`) — out. Core only parses JSON; the webview transpiles TS when it's involved (D13).
- **Fixed effect library as the LLM's expressive ceiling** — out. Effects are user-authorable WGSL files (D15); built-ins are starting points, not the boundary.
- **Two-WebSocket localhost layout** — out. Tauri IPC locally; optional single JSON-RPC WS for remote/MCP.
- **Ableton Link / external clock sync in v1** — out. Tap-tempo, manual BPM, and audio-derived BPM cover v1. Pre-computed audio features from a DAW can arrive over OSC if needed. Defer Link until a real show demands it.
- **Syphon** — out of v1. Reconsider if a downstream Resolume/VDMX integration appears.

---

## 8. Open uncertainties

Not blockers for Phase 0–2; flagged for later decision.

1. **Bevy vs raw wgpu.** Bevy gives ECS, schedule, render graph, asset hot-reload. But it pulls in a game-engine worldview the rest of WZRD doesn't share, and Phase 2 wants minimum-viable-binary-this-week. *Lean: raw wgpu for Phase 2, with the option to pull in `bevy_ecs` (the crate, not the engine) if state management gets hairy.* Settle during Phase 2/3.
2. **First-show projector resolution and physical surface.** Affects calibration UX, mask resolution, the 10× layer target. Should be pinned before Phase 5.
3. **macOS-first or Linux-first ship.** Lean macOS first (dev machine), Linux parity later.
4. **HAP-Q vs HAP plain.** HAP-Q better quality, larger files, slower decode. The "10× 1080p" target is for HAP plain; HAP-Q probably caps lower.
5. **ISF importer subset.** No-multipass / no-persistent-buffer is the obvious v0. Audio-input ISFs (`audioFFT`, `audio`) translate naturally to our driver bus. `IMG_PIXEL` / `IMG_NORM_PIXEL` / passthrough macros are a follow-up call.
6. **Audio loopback on macOS without user setup.** Currently requires BlackHole / Loopback. Apple makes a system tap hard. *Lean: document BlackHole, don't bundle a tap.*
7. **Remote control over a tunnel for collaboration / demo.** Trivial to enable later (the WS surface is already remote-ready), but auth/ACL is real work. Not in v1.
8. **Multi-projector / edge-blending.** Not in v1, but the architecture (one composite buffer per output, per-output homography) supports it. Don't accidentally hard-code single-output.
9. **Scene reload granularity.** Hot-swapping a single binding's params is cheap; re-running the entire scene is more honest. The middle path (diff by stable binding `id`, re-init only changed ones) is what Phase 2 targets — every binding has a stable `id` from day one (§4.2).
10. **Inline-WGSL vs file-based effects in practice.** D15 supports both. Open question: which one does the agent gravitate toward, and how much friction does each have? Inline keeps the scene self-contained; file-based gets proper Monaco editing and reuse. Watch behavior in Phase 3, decide if either path needs ergonomic polish.
11. **Shader include / module system.** Effects past v1 will want shared utilities (noise libs, SDF utils, color-space helpers, palette helpers) instead of every WGSL file re-implementing `permute()` and `hsv2rgb()`. WGSL has no native `#include`; the options are a small text-preprocessor pass (`#import noise`, `#import color` resolved against a known module dir) before handing source to `naga`, or stitching modules together via `naga::front::wgsl`. Either approach composes equally with inline-WGSL (preprocess the string before compile) and file-based effects. Not in v1 — but design the effect loader's compile entry point as a single function so wedging this in later is a one-layer change, not a rewrite.

---

## 9. Known risks

Carried forward from prior plan §8, still relevant:

- **Texture-array upload cost.** Uploading ~100 antialiased masks at load time may be slow even under the 256-slice cap (D4). Mitigation: lazy-load mask slices, or pack masks more tightly. Measure before optimizing.
- **User-WGSL validation surface.** D15 lets the LLM ship arbitrary shader code. `naga` catches syntax/type errors at load, but pathological shaders (infinite loops in compute, huge buffer reads, GPU hangs) can still kill a frame. Mitigation: keep effects to fragment-only initially, set conservative pipeline timeouts where the backend supports them, and isolate the previous-good pipeline so a bad load never blanks the projector.
- **Color banding under many additive blends.** 8-bit-per-channel render targets band fast under flash + glow + flood-fill stacks. Plan for an opt-in 16-bit-float intermediate composite buffer.
- **Audio onset detection.** Raw FFT is easy; good onset detection is library work. Plan to swap implementations if the first one feels sluggish. `aubio` (FFI), or a hand-rolled high-frequency-flux detector, are the options.
- **HMR + GPU resource cleanup.** With effects added/removed live, leaks compound fast. Addressed structurally by the swap-on-success pipeline lifecycle in §3.6 — every pipeline / bind group / pass plan has a single owner and an explicit replacement protocol from day one, not retrofitted after the first leak. Specific failure modes to watch: textures held by stale bind groups after a pipeline swap, decoder staging slots from a closed video stream, FBO targets from a deleted slow-path layer, and `naga` module handles retained after an effect is removed from the project. The Phase 6.6 spike must exercise these (rapid edit/save loops on inline WGSL, video stream stop/start, scene reload mid-frame) and verify GPU memory plateaus, not climbs.
- **Layer-pack format evolution.** `version: 1` from day one, loader refuses unknown majors, bump on breaking changes.
- **Calibration drift after physical bumps.** Handled by the re-shoot flow (§3.9), but the workflow itself needs to be smooth enough that users actually use it.

---

## 10. Summary

- **One realtime engine, shipped in two stages.** A standalone Rust+wgpu `render-core` binary (Phases 0–3) becomes playable in ~3–4 weeks with no UI at all. The Tauri shell + React/TS webview (Phase 4+) wraps it later as a control front-end — native fullscreen on the projector display, control UI on the operator's display.
- **One offline service.** Existing Python `wzrd_mcp` (FastMCP) runs separately — local subprocess or Modal — for segmentation, surface prep, and cloud content generation. No port to Rust.
- **Two contracts.** Layer pack (offline data) and `scene.json` + RPC (live control). The optional `scene.ts` DSL is a typed ergonomic mirror of `scene.json`, never canonical.
- **One scene model.** Selectors over semantic regions, flat per-layer effect stacks with inter-layer FBO sampling, ISF-shaped effect schema, drivers as the universal binding source.
- **An LLM-shaped creative surface.** Effects aren't a fixed library — agents and humans write WGSL (inline in `scene.json` or as project-local files) and the engine hot-loads it. The expressive ceiling is "what can be written in a shader," not "what the engine designer pre-built."
- **Two complementary agent surfaces.** `wzrd_mcp` for content generation, `render-core` RPC for realtime playback. A Claude agent uses both: generate → segment → bind → tweak.

The unique architectural commitment — the one thing that makes this *WZRD* and not "a browser VJ tool we'd never finish" — is treating the segmentation of the physical surface as the central scene primitive. Every load-bearing decision exists to make that primitive cheap to author, fast to composite, and resilient to physical re-shoots.
