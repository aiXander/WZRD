# WZRD render-engine — system design (v1, RETIRED)

> **RETIRED 2026-07-04** (moved to `docs/finished/` 2026-07-10). This was the
> original build plan (phases 0–4.2, all landed). It is superseded by
> `docs/reference/render-engine.md` (current system state, contracts, the
> 2026-07 performance/telemetry pass) + `docs/TODO/render-engine-roadmap.md`
> (the structural roadmap). Everything load-bearing was promoted there; this
> file holds only historical rationale behind decisions D1–D15 and the
> phase-by-phase build notes.

> **Implementation status (2026-05-22).** Phases 0–4.2 are landed.
> `render-core/` is a Rust crate (`[lib] + [[bin]]`) whose standalone binary
> still drives the projector headlessly; the new `wzrd-app/` Tauri shell
> spawns the same binary as a sidecar with `--ws-addr 127.0.0.1:9123` and
> proxies the §3.11 RPC surface through Tauri commands into a React +
> TypeScript + Vite + Tailwind webview. Three routes ship — Prepare (surface
> canvas + Monaco editor + binding inspector), Perform (preview hero +
> audio strip + driver rack), Debug (collapsible panels for connectivity,
> render stats, driver bus, hot-reload events, log stream, pack & scene
> state). Phase 4.1's status strip (OSC/Engine/FPS/Reload pills) sits in
> the top bar across every route. Inline `naga`-validated WGSL squiggles
> work in Monaco via `wgsl.validate` IPC. The 4.1 preview thumbnail is
> implemented as a periodic GPU readback of the composite buffer
> (`Rgba16Float` → CPU f16 decode → JPEG → base64) emitted on the
> `preview` telemetry channel at ~15 fps. Audio capture remains in the
> separate Realtime Audio Feature Server (separate Python process);
> `render-core` is a passive OSC sink for `/audio/lmh` + `/audio/onset/*`
> (`audio_refactor_plan.md` kept as the design-rationale paper trail).
> Video paths (Phase 5) and MCP wrapper (Phase 7) are not yet started.
> Slow-path FBO routing (D5, `layerRef`) is intentionally deferred — no
> scene has needed it yet.
>
> **Phase 4 architectural choice — subprocess + WS, not in-process winit.**
> The original spec envisioned Tauri + winit sharing one event loop in one
> process (§6.1). On macOS that's a real spike (NSApp main thread; exclusive
> fullscreen interactions with webview focus changes; cross-window
> ownership). Rather than fight it, Phase 4 lands as **Tauri shell ↔
> `render-core` subprocess over localhost JSON-RPC WebSocket**. This:
>   - keeps the headless agent path (`render-core --scene foo.json`) byte-
>     identical and unblocks every Phase 7 use case from one code path;
>   - reuses the *exact* method set Phase 7 will expose to MCP — the only
>     thing remote MCP needs is to point at the same WS surface;
>   - matches Pattern A (§3.10) — the audio server already lives next door
>     as a sibling process over OSC; render-core becomes a third sibling
>     over JSON-RPC. Same model, same operator mental load.
>   - resolves §6.1 by sidestepping it. The shell never opens the projector
>     window itself; the subprocess does, and the two halves communicate
>     only through frames on a wire.
>
> **Post-Phase-3 correctness fixes (architecture review v1).** Compositor
> blending is now genuinely additive (`One + One`); the composite buffer is
> `Rgba16Float`; effects return premultiplied RGBA. This is the
> "scene-aware *additive* projection-mapping" thesis actually showing up in
> the pixels — the prior `SrcAlpha / OneMinusSrcAlpha` path silently
> replaced layers instead of summing light. The pack manifest was renamed
> from `scene.json` to `pack.json` (legacy reads still work with a warning)
> to stop sharing a filename with the runtime control file. The CPU-side
> mask atlas drops after GPU upload; the effect rescan tracks per-file
> mtimes so a single editor save only invalidates the one pipeline that
> actually changed. The composite texture carries `COPY_SRC` pre-emptively
> for Phase 4's preview readback. **Still open from the review** — most
> notably the explicit `Layer` object between `SceneFile` and `Binding`
> (review #2 + #9): high-impact for scene-authoring ergonomics and the
> live-perform UI in Phase 4, deferred until those land so we don't churn
> the schema twice.

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
                    ▼ layerpack/ (pack.json + masks/*.png + refs)
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
- `audio.band(low|mid|high)` — `Driver<f32>`, sourced from the Realtime Audio Feature Server's `/audio/lmh` OSC stream (autoscaled into ~[0, 1] server-side).
- `audio.onset({ band })` — `Driver<Event>`, sourced from `/audio/onset/{low,mid,high}` triggers, exposed as a decaying envelope on read.
- `midi.cc(n)`, `midi.note(n)` — `Driver<f32>`.
- `midi.noteOn(n)` — `Driver<Event>`.
- `osc.path('/x/y')` — typed by declared path schema.
- `ui.slider(name, [min, max])` — surfaces a knob in the live UI.
- `const(value)` — wrapper for literal parameters.

Audio capture and DSP do **not** live in the engine. The Realtime Audio Feature Server (separate Python process) owns capture, autoscaling, soft gating, per-band Schmitt onset detection, and BPM tracking; `render-core` binds a UDP socket on `127.0.0.1:9000` and forwards the decoded features into atomics that the driver bus reads. See `audio_refactor_plan.md` §3.1 for the auto-detect / auto-recover lifecycle (either process can start, stop, or restart at any time without the other knowing) and §4.1 for the OSC ↔ driver mapping. v1 explicitly does not expose `audio.rms`, `audio.bpm`, or `audio.fft` — the engine surfaces exactly what the server emits, and scene authors pick a band when they want "loudness."

`AudioFeatures` also captures two side-channels Phase 4 telemetry leans on directly: **`sample_rate`** (set on `/audio/meta`, today diagnostic-only) and **`is_fresh(stale_after_ms)`** (true iff a packet arrived inside the window; powers the watchdog log and the Phase 4.1 OSC status pill — `audio_freshness` telemetry is one atomic-load per poll, no new core work needed).

OSC remains a flat UDP listener inside the core for non-audio paths (`osc.path('/x/y')`); the same dispatcher will generalise to a `HashMap<String, f32>` for arbitrary OSC inputs in a follow-up (`audio_refactor_plan.md` §10).

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

**Third sibling process: the audio feature server.** Same Pattern A applies — the Realtime Audio Feature Server (separate Python repo, `Realtime_PyAudio_FFT`) runs as a long-lived localhost process emitting OSC to `127.0.0.1:9000`. `render-core` listens; the Tauri shell surfaces a top-bar status pill that deep-links to the audio server's browser UI for tuning DSP (clicking the OSC pill opens `http://127.0.0.1:8765/`). The engine never embeds audio capture; offline tools, live engine, and the audio server are three independent processes sharing files on disk and localhost protocols.

**Fourth sibling process (as of Phase 4.1): the Tauri shell.** The Tauri shell process spawns `render-core` as a *child* with `--ws-addr 127.0.0.1:9123` and proxies the §3.11 RPC surface through Tauri commands into the React webview. Same Pattern A — restartable independently, communicates only over wires (JSON-RPC over WS in this case). Phase 7's MCP wrapper will connect to the *same* WS surface; the shell isn't a privileged client, just the first one. The standalone `render-core --scene foo.json` (no `--ws-addr`) stays the headless agent target.

**What changes in the existing Python:**

- `wzrd/` modules: no changes.
- `wzrd_mcp/server.py`, `tools.py`, `fal_tools.py`: no changes.
- New module: **`wzrd/layerpack.py`** (Phase 1). Takes `wzrd.islands` output + mask PNGs + a tags JSON, emits `pack.json + masks/ + references/`. Also exposed as an MCP tool (`build_layerpack`) so the agent can call it.
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

### 4.1 Layer pack — `pack.json`

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
  pack.json              # layer-pack manifest (was `scene.json` pre-review-v1; renamed to stop colliding with the runtime control file)
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

### Phase 0 — clear the slate (minutes) ✅ done

The old `render-engine/` browser-Three.js prototype shared neither language, paradigm, nor problem with the new architecture (3D shader sphere in R3F vs. native wgpu 2D mask compositor). Deleted rather than refactored. The one piece worth preserving — the `organicShader` WGSL string — survives as an idea in this doc.

Concretely: `rm -rf render-engine/`, `cargo new render-core` for the standalone binary. No Tauri yet.

### Phase 1 — `wzrd.layerpack` Python module (half-day) ✅ done

Exports the §4.1 format from `wzrd.islands` output + external mask PNGs + a hand-edited tags file. CLI: `python -m wzrd.layerpack <surface> <masks_dir> --tags tags.json -o pack/`. Smoke test in `test.py`. Wrap as MCP tool `build_layerpack` in `wzrd_mcp/tools.py` so the agent can call it. Blocker for everything downstream.

Only new Python work in the build. Everything else in `wzrd/` and `wzrd_mcp/` stays as-is per D14 / §3.10.

### Phase 2 — Minimal playable Rust core, no UI (1 week) ✅ done

The fastest path to "see pixels move on the projector." A standalone `render-core` binary with **no Tauri, no webview, no TypeScript on the critical path.**

- CLI: `render-core --pack path/ --scene scene.json`.
- `winit` + wgpu fullscreen window on a chosen display index.
- Layer-pack loader → `Texture2DArray` (R8, 256 slices max — D4).
- JSON Schema for `scene.json` (D13) — strict parsing, helpful errors.
- Compositor: per-layer pass, blend in z-order, homography final pass (identity by default).
- One built-in effect: `tint`. End-to-end through the binding pipeline.
- File watcher on `scene.json` → diff bindings by stable `id` → hot-reload.
- macOS first; verify Linux builds compile.

**Deliverable shipped:** edit `scene.json` in any editor, save, projector updates. Boring on screen (flat tints), but the whole spine — pack loading, mask compositing, scene parsing, hot-reload — is real and the agent loop is unblocked.

### Phase 3 — Effects, drivers, user-WGSL (2 weeks) ✅ done

Built out the effect model so the agent loop is genuinely creative. Still no Tauri.

**Landed:**

- Effect discovery from disk (D15): project-local `effects/<name>/{shader.wgsl, descriptor.json}` + inline-WGSL bindings. `naga` validation at load; hot pipeline rebuild on file save with swap-on-success (a bad save keeps the previous good pipeline rendering).
- Driver bus (§3.7): `const`, `clock.bars/beats/phase/time`, `audio.band/onset`, `ui.slider` (stub until Phase 4). Audio features ingested over OSC from the standalone Realtime Audio Feature Server (separate Python process) — `render-core` is a passive sink. Lock-free atomic `AudioFeatures` between the OSC recv thread and the render thread.
- Built-in effect catalog: `tint`, `hueCycle`, `flash`, `wobble`. Deliberately a small reference set — past v1 the creative path is *authoring new effects* (D15), not adding built-ins.
- Single `Texture2DArray<R8>` mask atlas, one shared `FrameState` uniform written per frame, one `LayerParams` uniform per binding. Built-in effects share one pipeline with an `effect_id` switch; user effects each get their own pipeline cached by content hash (inline) or file path (project-local).
- WGSL composer: every effect is compiled as `prelude + body + main`, so user code only writes `fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>` and accesses `state.*` / `f_param(N)` / `c_param(N)` / `sample_mask(uv)`.
- File watcher widened to watch the effects directory recursively in addition to the scene file.

**Deliberately deferred (per "don't overdo it" — add when a real scene demands them):**

- Slow-path FBO routing (`layerRef`, D5) — no scene has needed cross-layer sampling yet.
- `floodFill`, `scrollPattern`, `glow` built-ins — easier to author project-local once the use case is concrete.
- Generic OSC paths beyond `/audio/*` (`osc.path('/x/y')`) — the audio-feature OSC sink is landed; widening the dispatcher to a `HashMap<String, f32>` for non-`/audio/*` paths is a small follow-up.
- Optional JSON-RPC WebSocket server for remote control — folded into Phase 7's MCP wrapper.

**Status:** the §1.2 tree scene primitives all work — palette cycle, audio-onset flash, audio-reactive amplitudes, time-driven UV displacement, user-authored shaders. The architectural thesis ("agent edits text, projector responds, no UI") is proven end-to-end.

### Phase 4.1 — Tauri shell, minimum viable UI (~3–5 days) ✅ done

The smallest UI that adds real value over the standalone headless binary. One window, no routes, no structured editors, no panels. The standalone `render-core` binary and the headless `scene.json` + `effects/*.wgsl` agent path stay unchanged — Tauri is an additional front-end, never a replacement.

The two wins over "just edit files in VSCode and watch the projector":

1. **Inline `naga` squiggles in the WGSL editor** — see shader errors at the call site instead of tailing a terminal.
2. **Glanceable liveness** — one strip says "audio is flowing, frames are rendering, the last save compiled."

That's the value proposition. Everything richer — surface canvas with mask overlays, structured binding inspector, driver rack, audio feature strip, multi-panel debug — lands in Phase 4.2, sized against what 4.1 actually exposes as painful.

**Single-window contents:**

- **Monaco editor** for the currently open `scene.json` plus a flat file picker over `<project>/effects/*/{shader.wgsl, descriptor.json}`. Save (⌘S) on a scene file → `scene.load(json)` over IPC; save on an effect file → existing file watcher picks it up. Inline diagnostics via debounced `wgsl.validate(source)` RPC.
- **Surface preview thumbnail** in a corner — downsampled composite from the `preview` telemetry channel at ~15 fps jpeg. Confirms the projector is alive without alt-tabbing. No interactivity, no mask overlays — that's the 4.2 surface canvas.
- **Status strip** (top bar), three pills:
  - **OSC pill.** Green / amber / red on freshness of the audio-feature-server feed (fresh, stale ≥2s, never-heard-from / bind-failed). Click → opens the audio server's localhost browser UI in the system browser. The single most-glanced indicator during a show; unambiguous at 2m. Audio DSP itself (gates, compression, onset thresholds, BPM smoothing) is tuned on the server's own UI — not duplicated here. Backed by `AudioFeatures::is_fresh(2_000)` (already shipped in `osc.rs`) emitted over the `audio_freshness` telemetry channel — no new DSP code.
  - **FPS pill.**
  - **Last-reload outcome.** Compact: `effect 'drift' OK 14ms` / `naga error line 12 col 5 — previous pipeline retained`. Functions as a one-line debug page until 4.2 grows a real one.
- **Open pack / open scene** as native file pickers in the top bar.

**Design principle for the UI surface — carries through 4.2+:** swap-on-success extends to the UI. A bad WGSL save, a malformed scene edit, a failed RPC — never blanks the projector, never opens a modal. Errors surface inline (Monaco markers + the reload pill); the projector keeps its last good frame.

**Scaffolding:**

- `pnpm create tauri-app` rooted alongside `render-core/`. The existing core becomes a library crate plus the standalone `render-core` binary (kept for headless runs).
  - **Crate-split mechanics.** `render-core/Cargo.toml` grows a `[lib]` section (`path = "src/lib.rs"`) alongside the existing `[[bin]]`. `src/lib.rs` re-exports the module roots (`pack`, `scene`, `compositor`, `drivers`, `effects`, `gpu`, `osc`, `watch`) and lifts the current `App` / `ApplicationHandler` impl out of `main.rs` into a `pub fn run(cli: Cli)` entry point — `src/main.rs` becomes a 5-line thin wrapper that parses CLI + delegates. `src-tauri/Cargo.toml` then depends on `render-core` as a path-dep library and calls `render_core::run_with_*` variants from its command handlers. The standalone binary stays buildable + headless-agent-runnable through the split.
- Tauri spawns the wgpu render window via `winit` on the projector display (configurable index, default = secondary if present).
- **IPC bridge** (`src-tauri/src/rpc.rs`): every Tauri command is a thin wrapper around the same dispatch function the future WebSocket will use (§3.11). TS types codegen'd from `rpc.schema.json`.
- **Frontend stack:** React + TypeScript + Vite. Plain Tailwind. No design system, no router (one screen). shadcn/ui can come later.
- **WS lives between Tauri and render-core, not between webview and remote clients.** "Tauri IPC only" still holds at the *webview* boundary (the React app talks `invoke()`); remote/phone access stays out of 4.1.

**RPC additions:**

- `telemetry.subscribe(channels: string[])` / `telemetry.unsubscribe(channels: string[])` — initial channels only: `preview`, `hot_reload`, `audio_freshness`, `fps`. Other channels (`log`, `frame_stats`, `drivers`, full `audio`, `connectivity`) land with their consumers in 4.2.
  - `audio_freshness` carries `{ peer, packet_rate, last_packet_age_ms, state: 'fresh' | 'stale' | 'down' }` — enough to colour the OSC pill, nothing more.
- `wgsl.validate(source: string) → diagnostics` — for inline Monaco squiggles, independent of `effect.upsert`.

Everything else 4.1 needs (`pack.load`, `scene.load`, `effect.upsert`) is already on §3.11.

**Spikes:**

- **Tauri + `winit` cross-window cooperation on macOS secondary display** (carry-over from §6.1). Re-verified on the Tauri-hosted topology, not just `cargo run` standalone. — **Resolved by sidestepping**: 4.1 ships render-core as a sibling subprocess (see top-of-doc status block + below "as built"), so the Tauri process never owns a wgpu window and the cohabitation question is moot. The cross-process boundary is JSON-RPC over localhost WS.
- **Monaco + WGSL.** Community grammar + `naga` diagnostics mapped to Monaco's marker API. — **Done.** Hand-rolled Monarch grammar in `wzrd-app/src/components/MonacoPanel.tsx`; engine-side `wgsl.validate` (`render-core/src/rpc.rs`) runs `naga::front::wgsl::parse_str` on the composed `prelude + body + main` source and remaps spans back into user-source line/column space before returning.

**Deliverable:** open a pack, edit `scene.json` and `effects/*.wgsl` in Monaco with inline naga errors, glance at the status strip during a show, confirm the projector is alive via the corner thumbnail. Headless agent path unchanged.

**As built (2026-05-22).**

- **Crate split landed.** `render-core/Cargo.toml` carries both `[lib]` (`name = "render_core"`) and `[[bin]]`. The lib re-exports nine module roots; `App` + `ApplicationHandler` lifted into `src/app.rs`; `src/main.rs` is now an 18-line CLI wrapper around `render_core::run`. `wzrd-app/src-tauri/` consumes the lib by path-dep.
- **Subprocess + WS, not in-process winit.** The engine spawns its own winit window. The Tauri shell launches `render-core --ws-addr 127.0.0.1:9123` from its `setup` hook. The shell's `src-tauri/src/engine.rs` owns a single I/O thread that does request-reply demux + telemetry fan-out (request envelopes carry a u64 `id`; replies route via per-id oneshot; `telemetry.event` notifications emit on the Tauri `engine:telemetry` event channel).
- **Engine-side WS server.** `render-core/src/ws.rs` (sync `tungstenite`, thread-per-connection). State-mutating methods (`scene.load`, `effect.upsert`, `effect.remove`) queue an `EngineCommand` and the WS worker thread blocks on a reply oneshot until the render thread drains the queue in `about_to_wait`. Read-only methods (`pack.info`, `scene.getState`, `wgsl.validate`, `telemetry.channels`) resolve inline.
- **Telemetry.** `render-core/src/telemetry.rs` defines a clone-able `Bus` with bounded per-subscriber channels + a sticky-replay store for late subscribers. Channels live now: `preview`, `hot_reload`, `audio_freshness`, `fps`, `frame_stats`. `log`, `drivers`, `audio`, `connectivity` exist as channel names + payload types but the engine doesn't yet emit on them (4.2's UI surfaces them as "no data yet" until a follow-up wires emitters).
- **Preview readback.** `PreviewSampler` in `telemetry.rs` schedules `copy_texture_to_buffer` from the composite (`Rgba16Float`) every ~66 ms, maps the buffer, manually decodes f16 to f32 (avoids a `half` dep), downsamples to ~320 px wide, JPEG-encodes (`image::codecs::jpeg`), base64-encodes, and emits on `preview`. `COPY_SRC` was pre-emptively set on the composite texture for exactly this in the post-Phase-3 review.
- **`wgsl.validate` shape.** Picked option (1) from §6.7 (validate-in-core, IPC-bounced). One naga instance, perfect parity with the live pipeline path. Latency is fine under the debounce window.
- **Status strip pills.** OSC / Engine / FPS / Reload land verbatim. Clicking the OSC pill opens `http://127.0.0.1:8765/` (audio server browser UI) via `@tauri-apps/plugin-shell`.

### Phase 4.2 — Authoring + perform + debug UI (~1–2 weeks) ✅ done

Builds the three-route structure on the 4.1 spine, sized against actual pain points from using 4.1. Defer any of the three routes individually if 4.1 covers it well enough.

**Design principles (from `user_design_spec.md`):**

- **Surface-first.** Once a surface canvas exists, the photo + masks + named regions are the primary visual on every page that has one. Panels are chrome.
- **Surface-language.** Layers are addressed by `id` / `tag` / `group` (D7) in user-facing UI. Mask paths, slice indices, blob numbers live only on the Debug page.
- **Two modes, one stripped UI.** Prepare/Perform split is real but both routes run on the laptop in 4.2 — no mobile, no showtime-polish yet.
- **Audio features come from outside; routing happens inside.** Post `audio_refactor_plan.md`, `render-core` is an OSC sink. WZRD's UI is a *routing* tool — which audio feature drives which param on which layer. Don't duplicate the audio server's DSP UI; link out via the OSC pill.

**Top-level navigation:** three routes (`/prepare` ⌘1, `/perform` ⌘2, `/debug` ⌘3), keyboard-switchable. The 4.1 status strip stays in the top bar across all routes.

#### Prepare route

Three columns, ~40 / 35 / 25:

- **Left — surface canvas.** Reference photo with mask overlays as toggleable layers. Clicking a region highlights it and shows its `id` / `tags` / `group` in a small inspector strip. Hovering a binding (right panel) highlights its resolved layer set on the canvas. Read-only; pan + zoom only. No region renaming, no sidecar `identity.json` editing in 4.2.
- **Middle — editor pane.** Same Monaco from 4.1, now with `scene.json` and `effects/*.wgsl` as proper tabs.
- **Right — binding inspector.** Structured editor for the binding currently selected. Monaco stays the source of truth; this panel exposes dropdowns / sliders / pickers that mutate the JSON and write back.

**Binding inspector — visual driver routing.** Per selected binding:

- **Selector row.** Dropdown for selector kind (`id` / `tag` / `group` / `all`), second dropdown populated from the loaded pack. "→ N layers" chip; click to highlight on the surface canvas.
- **Effect row.** Dropdown of built-ins + project-local + "inline WGSL". Inline reveals an embedded textarea or jumps the middle pane to a fresh Effects tab.
- **Param rows** — one per input declared in the effect's descriptor. Editor shape by type: `float` → numeric input + driver picker; `color` → swatch; `bool` → toggle; `color[]` → reorderable swatches; `vec2` / `image` → text input.
- **Driver picker** (for any `float` param): `const(value)`, `clock.bars(n)` / `beats(n)` / `phase(rate)` / `time`, `audio.band(low | mid | high)`, `audio.onset(band, decay)`, `ui.slider(name, [min, max])`. Each driver-bound row shows the driver's *live value* as a small filled bar — confirms the audio is moving while you author.
- **"Add binding"** at the top of the binding list — opens a fresh row defaulting to `{ select: all, effect: tint, color: white }`.

Both surfaces round-trip cleanly: Monaco save → core hot-reload; inspector edit → JSON updated in Monaco → save → core hot-reload. One state.

#### Perform route

- **Top — surface preview** (expanded from 4.1's corner thumbnail).
- **Middle — audio feature strip.** Three vertical bars for `audio.band(low/mid/high)` + three onset-flash indicators with current decay envelopes. Powered by `telemetry.subscribe(['audio'])`. This *is* the post-refactor audio-debug viz — no full FFT in v1 (per `audio_refactor_plan.md` §10), no broadband RMS. Tuning happens on the audio server's UI (click OSC pill).
- **Bottom — driver rack.** Single scrollable list of every driver-bound param in the active scene. Per row: `binding_id · param_name`, source pill (compact: `audio.onset(mid, 0.3)`, `clock.bars(8)`, `ui.slider("warmth")`, `const(0.4)`), live value bar, inline control if applicable (`ui.slider` → knob; `audio.onset` → decay slider; `clock.bars/beats` → `n` stepper; `audio.band` → informational), "→ N layers" affects chip.

All edits fire `param.set(...)` or the driver-replace RPC. Ephemeral by default; explicit "save to scene" writes back to `scene.json`.

#### Debug route

Dev-time tool. Vertical stack of collapsible panels — kept dense because this is the page that lets me debug the build. Designed as a self-contained route that can be cut without touching Prepare or Perform. Gated behind `WZRD_DEBUG_UI` so release builds can hide it.

- **Connectivity.** OSC audio feed (bind addr, peer, packet rate, last-packet age, dispatcher histogram of `/audio/lmh` and `/audio/onset/*` counts, last `/audio/meta`), file watcher, Tauri IPC, `wzrd_mcp` reachability (ping `GET /health` every 5s). Each row green/amber/red.
- **Render stats.** Frame time (p50/p95/p99 over 10s), FPS, mask atlas slice count, active pipeline count, pass-plan length. ~4 Hz updates.
- **Driver bus snapshot.** Live values of every active driver. 30 Hz telemetry channel.
- **Hot-reload events.** Scrollable log of every effect/scene reload attempt with outcome. The panel I'll stare at most while iterating on shaders.
- **Log stream.** Structured `log::*` events from the core. Filter by level + target. Capped at 2000 lines in the UI; disk log unchanged.
- **Pack & scene state.** Read-only dump of currently loaded `pack.json`, active `scene.json`, compiled effects, resolved layer sets per binding.

**RPC additions on top of 4.1:** widen `telemetry.subscribe` channels to include `log`, `frame_stats`, `drivers`, full `audio` (post-refactor `{ band_low/mid/high, onset_low/mid/high }`), `connectivity`. The §3.11 method set (`param.set`, `param.bind`, `binding.*`, `transport.*`, `calibration.set`) is already there.

**Deliverable:** the three-route Tauri app that makes "wire audio band X to param Y on layer Z" a few-clicks operation, with a working Debug page for the build phase.

**As built (2026-05-22).**

- **Three routes live**, keyboard-switchable via ⌘1/⌘2/⌘3 in `wzrd-app/src/App.tsx`. Status strip persists across all three.
- **Prepare.** Three columns 40/35/25:
  - `SurfaceCanvas.tsx` reads the pack via `pack.info`, fetches each layer's mask PNG over a `read_mask_png` Tauri command (b64-encoded), draws masks tinted by per-layer hue on an HTML canvas. Live preview JPEG underlays the masks so the canvas doubles as a live performance view. Hover/click pick layers via bbox lookup. Overlays toggle. Read-only — region renaming + identity.json editing stay deferred to 4.3+ as spec'd.
  - `MonacoPanel.tsx` opens `scene.json` + every `effects/<name>/shader.wgsl` as tabs. Hand-rolled Monarch WGSL grammar registered on mount (Monaco bundles JSON but not WGSL). ⌘S on the scene tab: round-trips through `scene.load(json)` (immediate engine reload) *and* writes to disk so the headless watcher path stays consistent. ⌘S on an effect tab: `effect.upsert` writes to disk + invalidates the pipeline.
  - `BindingInspector.tsx` is the structured editor. Mutations write the modified scene JSON back through `scene.load` AND `write_scene_file` — both Monaco and the inspector edit the *same* scene JSON, no second source of truth. Selector / effect / driver dropdowns implemented; "+ Add" creates a fresh binding skeleton.
- **Perform.** `PreviewThumbnail` (variant=`fill`) + `AudioStrip` + `DriverRack`. The audio strip renders L/M/H band bars and onset-decay indicators directly from the `audio` telemetry payload (when emitted). The driver rack merges scene-parsed rows with live `drivers` telemetry — if `drivers` events arrive, live values overlay the scene rows; otherwise the rack still lists every driver-bound param with static-source info.
- **Debug.** Six collapsible panels per the spec. Each panel is independent — cuttable without touching Prepare/Perform. Log stream caps at 2000 lines (matches the spec); level filter `all` / `warn` / `error`. Pack & scene state dumps are read-only `<pre>` blocks of JSON.
- **State model.** Single Zustand store (`src/state/store.ts`). Every telemetry channel has its own slice plus an append-style buffer where it makes sense (log, hot-reload history). Sticky channels (`hot_reload`, `audio_freshness`, `connectivity`, `fps`) get replayed to new subscribers Rust-side and re-applied Tauri-side, so a route change never wipes the pills.
- **Open as known gaps.**
  - The engine doesn't yet emit `drivers`, `audio`, `log`, `connectivity`, or `frame_stats` regularly — channel names + bus + UI consumers are all wired, just no engine-side emitter beyond `fps` + `frame_stats` percentiles + the `audio_freshness` heartbeat. The UI degrades gracefully ("waiting for…" placeholders). Next on the list for a Phase 4.2 follow-up; both halves are designed so wiring an emitter is a single-file change.
  - `param.set`, `param.bind`, `transport.setBpm`, `calibration.set` from §3.11 are not yet exposed as Tauri commands — the inspector currently mutates `scene.json` directly via `scene.load` + `write_scene_file`. Same effective behaviour, less plumbing; the structured RPCs land when the cue editor in Phase 6+ needs them.

### Phase 4.3+ — UI polish (deferred)

Add against demand, not the spec. Currently deferred:

- **Layer → audio matrix sidebar** (rows = layers, cols = bands; cell intensity = live driver value). Redundant with the binding inspector for everyday work — add only if scenes get dense enough to warrant the transposed view.
- **Driver-rack grouping / filter chips** (`all · audio · clock · ui-slider`).
- **Per-binding modulation depth slider** — would require wrapping driver expressions in `{ driver, depth, base }` in the scene schema. Design after a few real shows expose what shape it needs to take. Today's workarounds: effect params authored with sensible ranges, a `ui.slider` driver wrapping the audio-driven value inside the effect's own WGSL, or tuning the audio server's gain/compression.
- Mobile / phone access (Perform on iPhone over WebSocket).
- Master "audio listen" fader, panic blackout button, scene chooser grid + scene save-as, binding mute toggles, scene crossfade, auto-pilot chain.
- Calibration UI (4-corner drag) + re-shoot flow.
- Region inspector with sidecar `identity.json` editing and re-author flow.
- AI co-author panel (lands properly in Phase 7).
- `scene.ts` Monaco surface (JSON-only through 4.2).
- shadcn/ui or any real design system pass.
- Debug RPC trace panel + manual JSON-RPC console + GPU memory readout.
- FFT-bin spectrum viz / `audio.fft` driver (waits on server-side `osc.send_fft` per `audio_refactor_plan.md` §10).
- Audio-server control passthrough (tunneling the server's gates / compression / onset thresholds through WZRD — explicit anti-feature; "click OSC pill → open server UI" is simpler).
- Recording, video export — out of scope for v1 generally.

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

### 6.1 Tauri + wgpu native window on a non-primary display (Mac + Linux) — resolved by subprocess split

Original question: can one Tauri app process own both (a) a webview window on the operator's display and (b) a native wgpu fullscreen window on the projector display, with no compositor frame on the projector path? `winit` claims to handle it; verify on both OSes — especially the macOS focus-change failure modes (exclusive fullscreen pulled out, Spaces reshuffle, frame stutter when the React webview takes focus).

**Resolution (Phase 4 build, 2026-05-22).** Sidestepped rather than spiked. The Phase-4 implementation runs render-core as a **sibling subprocess** of the Tauri shell and bridges them over a localhost JSON-RPC WebSocket (`127.0.0.1:9123`). The Tauri process never opens a wgpu window; the engine subprocess does. Cross-window-on-one-process is therefore moot for v1. Trade-offs taken:

- **Wins.** Headless agent path stays byte-identical (`render-core --scene foo.json` with no `--ws-addr`). Same RPC surface Phase 7's MCP wrapper uses — single contract, two transports. macOS NSApp / Spaces / focus interactions reduce to "two normal processes each running their own event loop" with no shared state.
- **Costs.** A second process to launch + supervise (the Tauri shell does this in `setup`; window close kills the child). One extra hop in the request path (Tauri command → in-process WS client → render-core WS server → render thread). Latency is unmeasured but the only path that goes through it is human-scale (slider drags, ⌘S saves, debounced WGSL validation).
- **Reopen condition.** If a future scene wants OS-level exclusive fullscreen on a secondary display *and* the borderless-fullscreen fallback (now the default in `App::build_window`) starts hitting limits — re-spike. The borderless mode is what's stable today and matches the §6.1 fallback plan verbatim.

### 6.2 HAP-on-Rust reality check

Confirm whether `hap-cpp` (vidvox C++ ref impl) can be FFI'd cleanly into a Rust crate and whether the decode path hits sub-frame latency in a wgpu upload context. If FFI is messy, plan an in-Rust port — HAP itself is small.

### 6.3 ISF GLSL → WGSL via `naga`

Take 3 representative ISF shaders from `editor.isf.video`, run through `naga`, see what breaks. Decides how much ISF support we promise vs "write WGSL natively."

### 6.4 10× 1080p HAP smoke on base M2

The hardware target. If a 200-line Rust+wgpu prototype playing 10 HAP files into 10 quads at 60 Hz doesn't hit, every higher-level assumption needs revisiting before Phase 5.

### 6.5 Headless `scene.json` hot-reload + (later) TS transpile round-trip (D13)

Two paths to validate, in order:

1. **Headless (Phase 2). ✅ validated.** Run the standalone `render-core` binary with no UI. Edit `scene.json` in any editor → file watcher fires → core diffs against current state → projector updates within one frame budget. This is the agent's critical path; it must work before the webview exists.
2. **Webview JSON round-trip (Phase 4.1/4.2). ✅ validated.** Edit `scene.json` in Monaco → ⌘S → Tauri command `scene_load` → in-process WS client posts `scene.load` over `ws://127.0.0.1:9123` → engine WS server queues an `EngineCommand` → render thread drains it in `about_to_wait` → diff/apply via the same `apply_scene_json` path the file watcher uses → next composite frame reflects the edit. The TS transpile surface (`scene.ts`) is deferred to 4.3+ per "UI polish (deferred)".

If headless works and the JSON round-trip works, the agent loop is unblocked end-to-end.

### 6.6 User-authored WGSL effect hot-reload (D15) ✅ validated (Phase 3)

Drop `effects/<name>/{shader.wgsl, descriptor.json}` into a project folder, bind it from `scene.json`, edit the WGSL file, watch the pipeline rebuild without an engine restart. `naga` pre-validates the composed source (`prelude + body + main`); errors surface as `log::error!` messages, not crashes, and the previous good pipeline keeps rendering until the new one validates. Inline-WGSL bindings (content-hashed pipeline keys) share the same path. Verified end-to-end in `render-core/examples/phase3_smoke.scene.json` + `render-core/examples/effects/drift/`.

### 6.7 `wgsl.validate(source)` IPC shape (Phase 4.1)

Two viable implementations of the inline-WGSL Monaco squiggles:

1. **Validate-in-core.** Tauri command bounces source over IPC → core composes `prelude + body + main`, calls `naga::front::wgsl::parse_str`, maps the `ParseError` span back to a `{line, col, message, severity}` list → returns over IPC. One `naga` (already a core dep), perfect parity with the live pipeline path. Cost: every keystroke (debounced) round-trips through IPC.
2. **Validate-in-webview.** Build `naga` as a WASM module, load it in the renderer process, run synchronous validation in the Monaco worker. Zero IPC latency. Cost: a second `naga` build configuration, divergence risk if the WASM version drifts from the Rust crate version, larger webview bundle.

Lean: pick (1) for 4.1 — the IPC cost is negligible vs. the parity win, and `naga::front::wgsl::parse_str` is already what compositor.rs calls. Only spike (2) if the keystroke-debounce experience genuinely feels laggy at the projector.

Spike acceptance: a deliberately bad shader (missing semicolon, undeclared identifier, wrong return type) surfaces a Monaco marker on the right line within ~150ms of the keystroke, on the same source the engine would compile.

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
6. ~~**Audio loopback on macOS without user setup.**~~ **Resolved.** Audio capture moved out of `render-core` into the standalone Realtime Audio Feature Server (separate Python process). Any system-loopback dance (BlackHole / Loopback / system tap) is now the audio server's problem, and tunable from its own browser UI. `render-core` just listens for OSC features — no mic permission prompt on macOS first run.
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
- **HMR + GPU resource cleanup.** With effects added/removed live, leaks compound fast. Addressed structurally by the swap-on-success pipeline lifecycle in §3.6 — every pipeline / bind group / pass plan has a single owner and an explicit replacement protocol from day one, not retrofitted after the first leak. Specific failure modes to watch: textures held by stale bind groups after a pipeline swap, decoder staging slots from a closed video stream, FBO targets from a deleted slow-path layer, and `naga` module handles retained after an effect is removed from the project. The Phase 6.6 spike must exercise these (rapid edit/save loops on inline WGSL, video stream stop/start, scene reload mid-frame) and verify GPU memory plateaus, not climbs.
- **Layer-pack format evolution.** `version: 1` from day one, loader refuses unknown majors, bump on breaking changes.
- **Calibration drift after physical bumps.** Handled by the re-shoot flow (§3.9), but the workflow itself needs to be smooth enough that users actually use it.

---

## 10. Summary

- **One realtime engine, shipped in two stages.** A standalone Rust+wgpu `render-core` binary (Phases 0–3) is playable with no UI at all. The Tauri shell + React/TS webview (Phase 4+) wraps it as a control front-end — and runs the engine as a *subprocess* over localhost JSON-RPC WebSocket, so the projector window stays owned by the engine and the operator UI is a separate process. Same model the audio feature server uses (§3.10) — three sibling processes, each restartable, communicating over wires.
- **One offline service.** Existing Python `wzrd_mcp` (FastMCP) runs separately — local subprocess or Modal — for segmentation, surface prep, and cloud content generation. No port to Rust.
- **Two contracts.** Layer pack (offline data) and `scene.json` + RPC (live control). The optional `scene.ts` DSL is a typed ergonomic mirror of `scene.json`, never canonical.
- **One scene model.** Selectors over semantic regions, flat per-layer effect stacks with inter-layer FBO sampling, ISF-shaped effect schema, drivers as the universal binding source.
- **An LLM-shaped creative surface.** Effects aren't a fixed library — agents and humans write WGSL (inline in `scene.json` or as project-local files) and the engine hot-loads it. The expressive ceiling is "what can be written in a shader," not "what the engine designer pre-built."
- **Two complementary agent surfaces.** `wzrd_mcp` for content generation, `render-core` RPC for realtime playback. A Claude agent uses both: generate → segment → bind → tweak.

The unique architectural commitment — the one thing that makes this *WZRD* and not "a browser VJ tool we'd never finish" — is treating the segmentation of the physical surface as the central scene primitive. Every load-bearing decision exists to make that primitive cheap to author, fast to composite, and resilient to physical re-shoots.
