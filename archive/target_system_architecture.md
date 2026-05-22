# WZRD — Target System Architecture (research brief)

A concise summary of the architecture WZRD is converging on, intended as input to a deep-research pass looking for prior art, comparable projects, and lessons learned. Distilled from `render-engine/refactor_plan.md` (May 2026 pivot) and `context_to_continue_from.md` (subsequent VDMX / ISF / HAP / native-vs-browser evaluation).

## One-line description

A **scene-aware additive projection-mapping VJ engine**: an offline Python pipeline segments a photograph of the physical projection surface into a hierarchy of labelled regions, and a realtime native render core composites per-region effects driven by audio / MIDI / OSC / clock, output to the projector in exclusive fullscreen — authored by humans or an LLM via a text scene file edited through a browser-based control UI.

## The defining idea

The "scene" is not a 3D world or a flat video. The scene **is the segmentation map of a real physical object** (tree, rock formation, painted face, sculpture). Each segmented region — a leaf cluster, a trunk, an eye, a crack — is a first-class addressable layer that effects bind to by id, tag, group, or selector. Only the changing regions get projected; the unchanged surface stays dark. This is the "additive projection mapping" thesis.

Adjacent existing systems (VDMX, Resolume, MadMapper, TouchDesigner, HeavyM, Millumin, Smode) handle audio, FX hosting, codec decode, projector output, and calibration well — but **none of them treat per-region semantic segmentation of the projection surface as the central scene definition**. That is WZRD's unique contribution. Everything else in the architecture is in service of that.

## Two-part system

### 1. Offline pipeline — Python (existing WZRD toolkit, extended)

- `detect` → extract the projection surface from a photo.
- `align` → reproject re-shoots to the same canonical frame (SIFT/AKAZE + RANSAC + ECC).
- `darken` → CIELAB gradient-weighted histogram equalisation for the dark base layer.
- `islands` → KMeans color clustering + connected components for coarse auto-segmentation.
- External segmentation (SAM2, manual cleanup) merged in.
- **New: `wzrd.layerpack`** — exports a directory containing `scene.json` (layers with id, label, tags, bbox, centroid, parent, z, groups) + `masks/*.png` (binary or anti-aliased, at projector resolution) + reference imagery.
- `subtract_frame` / `subtract_video` / `reproject` remain available for the "produce a loopable mp4 for downstream tooling" path.

The layer pack is the durable contract between offline and realtime. Versioned from day one.

### 2. Realtime render core — native (Rust + wgpu, currently favoured)

- **Layer pack loader.** Mask stack uploaded as a `Texture2DArray` (one R8 slice per layer), metadata parsed into addressable scene graph.
- **Compositor.** 2D, display-space, no 3D camera. Per-layer effect stacks (flat ordered lists) rendered to offscreen buffers, then blended in z-order. Final pass applies a 4-point homography for projector calibration.
- **Effect graph.** Composable small effects: `hueCycle`, `flash`, `floodFill` (geodesic, mask-respecting; compute-shader candidate), `wobble`, `scrollPattern`, `videoClip`, `glow`. Effects are node-graph fragments, not opaque shaders.
- **Effect schema borrows from ISF (Interactive Shader Format).** ISF's JSON-input declaration (float, color, image, audio, audioFFT, event) is the battle-tested shape; adopt as the binding-format spec. ISF runtime is GLSL, so a WGSL transpile path (or shipping a small ISF→WGSL converter via naga) is the bridge to the existing community shader library.
- **Driver bus.** Every effect parameter is bindable to a time-varying signal: `clock.bars(n)` / `clock.beats(n)` / `clock.phase(rate)` (BPM-aware transport); `audio.rms()` / `audio.band(low|mid|high)` / `audio.onset({band})` (Web Audio or native FFT + onset detection à la meyda/aubio); `midi.cc(n)` / `midi.note(n)`; `osc.path('/x/y')`; `ui.slider(...)`.
- **Video decode.** Native (FFmpeg / AVFoundation + wgpu interop) for H.264/HEVC zero-copy to GPU. **HAP / HAP-Q decoder** added because it is the VJ-world standard for many concurrent 1080p layers on a single hardware decoder — CPU does cheap Snappy decompression, GPU receives DXT/BC blocks directly. Target: 10× 1080p concurrent layers on base M2 sustained 60fps with sub-frame latency.
- **Projector output.** Native exclusive fullscreen window on a chosen display, bypassing the OS/browser compositor for one less frame of latency and direct VSync control.
- **Cross-platform parity.** wgpu (Vulkan on Linux, Metal on macOS). Same scene file, same perf on Apple Silicon and Linux dGPU workstations.
- **Recording.** Tap on the live composite stream → per-frame PNG → hardware-encoded mp4 (VideoToolbox on macOS, NVENC/VAAPI on Linux). Deterministic offline-clock driver swap so audio-reactive scenes can be re-rendered repeatedly to the same output.

### 3. Control UI — browser (React/TS) over localhost

- Talks to the native core via WebSocket / JSON-RPC. **Control plane only — no video pixels.**
- Owns: scene authoring (editing `scene.config.js` / `.ts`), binding editor, calibration UI (corner-drag 4-point homography), live sliders & mute toggles, audio-debug visualiser (FFT bars + onset flashes), preview thumbnails (downsampled WS stream).
- Web tech kept because (a) most productive stack for live-tweak surfaces, (b) easiest target for LLM / Claude / MCP authoring, (c) remote control / collaboration / demo collapse to "swap localhost for a tunnel."

## Authoring model

Two artifacts, both checked into git:

1. **Layer pack** (data) — outputs of the offline pipeline.
2. **Scene config** (code) — a text file mapping selectors to effects with driver-bound parameters. Editable by hand or by an LLM. MCP tool integration (Phase 7) exposes high-level operations like `set_binding`, `add_effect`, `remove_layer_fx`.

Example binding (conceptual):

```
select: { tag: 'leaves' }
effect: hueCycle({ palette: [...], rate: clock.bars(8) })

select: { tag: 'leaves', pick: 'random_each', rate: clock.bars(4) }
effect: floodFill({ from: 'centroid', color: '#fff', duration: 1.2 })

select: { id: 'background' }
effect: floodFill({ from: [960, 1000], color: '#0a4',
                    trigger: audio.onset({ band: 'low' }) })
```

Selectors over hard-coded ids → resilient to re-segmentation. Effects as composable fragments → no per-scene shader code. Drivers as plain time-varying values → "everything is bindable" is uniform across audio, MIDI, OSC, clock, and UI.

## Decisions already made (load-bearing)

- **Mask representation:** `Texture2DArray`, one R8 slice per layer. Supports hundreds of layers with soft edges.
- **Effect composition:** flat per-layer stacks, not a DAG. Cheap, matches DSL shape 1:1.
- **Coordinate system:** display-space (px) at authoring level, normalised UV inside shaders.
- **Calibration:** in-engine 4-point homography as a final compositor pass.
- **Segmentation ownership:** WZRD owns the layer pack; the renderer never re-segments.
- **Native render core, browser control UI** is the currently favoured shape (Direction A-hybrid, leaning native). The pure-browser path was rejected primarily on: Chromium's ~4–6 concurrent video-decoder cap, no HAP support, compositor latency floor, and missing exclusive fullscreen.

## Open questions for the research pass

1. **Prior art on segmentation-driven projection mapping.** Who has built a system where per-region semantic masks of the physical surface are the *scene primitive*? Academic projects (CMU's projection mapping research, Disney Research, MIT Media Lab), open-source experiments, art installations?
2. **ISF as a binding-format spec outside the ISF runtime.** Has anyone adopted ISF's JSON input schema as a contract while running a different shader backend (WGSL/Vulkan)? Lessons on the schema's edges (multipass, persistent buffers, audio bins)?
3. **wgpu-based VJ / live-visuals engines.** Native Rust/wgpu projects in the live-visuals space (Notan, Bevy as a VJ host, nannou, vvvv gamma's stride). What did they learn about audio latency, MIDI/OSC ingestion, projector output on Linux?
4. **HAP decoding outside Resolume/VDMX.** Open-source HAP decoders, integration patterns with modern GPU APIs (Vulkan video / Metal compute), comparable codecs (NotchLC, Hap-Q vs DXV).
5. **LLM-authored visual scenes.** Projects shipping an agent loop that edits a live-reloaded scene file (Disco Diffusion-era VJ tools, Hydra, Strudel, TidalCycles + AI). What scene-format shape works best for LLM authoring at sub-second iteration?
6. **Browser-UI-over-localhost + native-core split.** Reference architectures (Resolume Arena's API, TouchDesigner WebClient, Tauri-based creative tools). Pitfalls around state sync, hot reload, and crash recovery between the two processes.
7. **Calibration and re-shoot workflows.** How established projection-mapping tools handle "the projector or scene got bumped" without a full re-segmentation pass — homography re-fit on top of an existing layer pack.
8. **Determinism for recording in a realtime-driven engine.** Patterns for swapping a wall-clock driver for an offline-clock driver while keeping audio-reactive effects re-renderable to identical output.

## Keywords for research seeding

projection mapping, additive projection mapping, scene-aware compositing, segmentation-driven visuals, ISF (Interactive Shader Format), HAP / HAP-Q codec, Syphon / Spout, VDMX, Resolume Arena, MadMapper, TouchDesigner, HeavyM, Millumin, Smode, Hydra, vvvv gamma, nannou, wgpu, naga, Rust creative coding, SAM2 segmentation, agent-authored visuals, MCP tool calls, live coding visuals, audio-reactive shaders, meyda, aubio, Ableton Link, OSC bridge, 4-point homography calibration, WebGPU vs native for media workloads.
