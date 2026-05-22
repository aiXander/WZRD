# WZRD render-engine — context to continue from

Saved 2026-05-03. Conversation context for picking up the VDMX / OSC / GLSL / ISF / Syphon evaluation against the render-engine refactor plan.

## What I was thinking about

Just learned about VDMX (Mac VJ software with great native audio analysis), OSC, GLSL, ISF shaders, and Syphon (GPU texture sharing). Wanted to know:

1. Are these compatible / complementary with the direction in `render-engine/refactor_plan.md`?
2. Is the overall direction sound?
3. How would I start building toward it?

## Files referenced

- `README.md` — WZRD toolkit overview (detect, darken, align, subtract_*, islands, reproject, MCP server).
- `render-engine/refactor_plan.md` — the May 2026 pivot from "3D shader scene that records to mp4" to "realtime, scene-aware 2D compositor driven by a segmentation map of the physical projection surface." Layer pack format, effect-binding DSL, drivers (audio/MIDI/OSC/clock), 7-phase roadmap.

## Compatibility analysis (the agent's take)

**OSC** — already in the plan (§3.2 drivers, §5.3, Phase 5). Trivial fit. ~50-line WebSocket-to-OSC bridge (osc.js or node-osc). Receive from anything: Ableton, TouchOSC on phone, Eurorack with OSC interface, VDMX clock. Move forward to Phase 3 alongside MIDI.

**GLSL** — engine is WebGPU/WGSL but TSL compiles to both, and Three.js WGSL is GLSL-ergonomic. Not a real conflict.

**ISF (Interactive Shader Format)** — the interesting one. ISF = GLSL fragment shader + JSON header declaring inputs (audio FFT, image, float, color, event). It is *exactly* the effect-binding shape sketched in §3.2.

Two consequences:
1. Hundreds of community ISF shaders exist (editor.isf.video, vidvox/ISF-Files on GitHub). Speak ISF → import all of that for free.
2. ISF input declarations are a battle-tested schema for "every parameter is a bindable driver." Reinventing it is wasteful.

JS runtime exists: `interactive-shader-format-js` (WebGL). Porting/wrapping for WebGPU/TSL is non-trivial but the *spec* is what to borrow even if the runtime isn't.

**Syphon** — macOS-only GPU texture sharing. Browser cannot produce or consume Syphon. Needs Electron/Tauri wrap + node-syphon, or a Swift helper that grabs the canvas. Only worth the effort if WZRD plugs into a *larger* live rig (Resolume / MadMapper / VDMX downstream, or pulling a Syphon texture *from* VDMX as a video source). Not on the critical path.

**VDMX audio analysis** — good (FFT, onset, BPM, envelope follower with nice UI). But algorithms aren't magic. Web Audio + a JS onset detector (`meyda`) gets ~80% of the way for free. The hard-to-replicate part is VDMX's *UI for tuning audio analysis live*. Mitigation: build a small audio-debug panel early in the browser engine.

## The bigger architectural fork

VDMX already solves: audio analysis, OSC/MIDI, ISF hosting, Syphon I/O, projector output, calibration, live parameter tweaking, scene switching.

VDMX does NOT solve: scene-aware per-region masking of a real photographed surface — the *unique* WZRD value-add.

Phases 3–5 of the refactor plan are roughly "rebuild VDMX in the browser." Two directions:

### Direction A — Browser engine, augmented
Keep refactor plan. Adopt ISF as the effect format (massive shader library for free). Add OSC early. Skip Syphon.
- More code, but everything is text/code/agent-authorable.
- Matches Phase 7 (LLM-authored scenes), matches API-first instinct.

### Direction B — WZRD as a layer-pack source feeding VDMX
WZRD = offline segmentation + layer pack + small native bridge publishing each mask as Syphon texture (or ISF generator). VDMX hosts shaders, audio, OSC, projector output.
- Way less code, ships in days not weeks.
- But: VDMX state is not a text file an agent can edit, Mac-locked, closed-source dependency, harder to demo remotely, less satisfying.

### Hybrid (probably best)
**Direction A with explicit ISF compatibility layer.** Effect schema is ISF-shaped; effects can be either native TSL nodes *or* imported ISF shaders compiled at load time. Borrow VDMX's ecosystem without adopting VDMX itself.

Trade-off to be honest about: A is more code but right for Phase 7 (agent authoring requires text-defined scenes). B is genuinely the faster path to a working live show. Worth knowing before committing two weeks to A.

## How to start (concrete)

Two cheap spikes BEFORE committing to Phase 2's compositor work:

### Spike 1 — ISF (2–3 hours)
Grab one ISF shader from editor.isf.video. Render in a standalone WebGPU page using `interactive-shader-format-js` (or port the JSON header parser, ~100 lines). Feed it audio FFT and a binary mask as `inputImage`. If it composites correctly with a layer mask, the whole effect-library question collapses: adopt ISF schema as the binding format, inherit the shader library.

### Spike 2 — OSC (1 hour)
Tiny `node-osc` → WebSocket bridge. Send `/test 0.5` from CLI, log it in browser. Confirms driver bus contract end-to-end. Lets Phase 2 dev be controlled with TouchOSC on a phone instead of building UI.

### Then Phase 1 from the plan
`wzrd.layerpack` Python module — half a day, unblocker for everything else. Don't touch the renderer until there's real layer-pack data to test against.

### Skip Syphon
Until there's a concrete reason (e.g. client wants WZRD output as one layer in their Resolume show). When that happens, Electron-wrap it. Don't build for it speculatively.

### Move audio-debug UI earlier
The plan currently underweights this. Web Audio is fine, but tuning "which band fires the flash" in the dark in front of a projector with no visual feedback will be miserable. Add tiny FFT-bars + onset-flash visualizer to `/live` mode in Phase 3, not Phase 5. ~2 hours. Saves a bad demo.

## Hardware reality check — 10× 1080p layers on base M2

Asked: can a base M2 MacBook render ~10 masked 1080p video layers with subtle FX (warp, hue shift) in realtime, low-latency?

GPU work and disk I/O are not the bottleneck:
- 10 × 1080p × a handful of fragment ops at 60Hz ≈ ~1 Gpixel/s vs. tens of Gpixel/s available.
- Memory bandwidth ~10 GB/s vs. ~100 GB/s available.
- Disk ~300 MB/s vs. ~3 GB/s+ SSD.

The bottleneck is **concurrent video decode**. Base M2 has one hardware video decoder; how you wrap it dictates whether 10 streams works:

- **Browser `HTMLVideoElement` → `THREE.VideoTexture`** (refactor_plan §5.4 default): risky at 10 streams. Chromium typically hardware-decodes the first 4–6 video elements and silently falls back to software / drops frames beyond that. ~5–6 streams safe; 10 unreliable, especially HEVC.
- **WebCodecs `VideoDecoder`** (deferred in §5.4): explicit hardware-decoder control, scales to 10 cleanly. More plumbing.
- **HAP / HAP-Q** (the VJ-world standard — what Resolume/VDMX use): GPU-decompressed from disk, no hardware decoder pressure. Built for this exact workload. 10× 1080p HAP-Q is comfortable on base M2.
- **ProRes**: hardware-accelerated on M2 Pro/Max only; base M2 decodes in software. Avoid for the base chip.

Latency: H.264/HEVC pipelines carry ~1–3 frames of decode latency per stream. HAP is essentially zero (per-frame GPU decompress). Sub-frame responsiveness to audio/MIDI pushes toward HAP or pre-loaded short clips as GPU textures.

**Bearing on the A-vs-B fork:** if 10 concurrent layers is a hard requirement for the first show, this is a real datum for Direction B / hybrid. VDMX already solved this problem by adopting HAP; replicating it in-browser means either committing to WebCodecs early or shipping a HAP decoder. Worth deciding before Phase 2 locks in `VideoTexture` as the video path.

## Performance / runtime target (added 2026-05-12)

Whatever direction we commit to (A / B / hybrid), the application must run **natively-fast** and **maximally GPU-accelerated** on both macOS (Apple Silicon, Metal) and Linux (NVIDIA/AMD, Vulkan). Concretely:

- Treat the GPU as the primary compute device end-to-end: video decode, mask sampling, compositing, FX, and output should all live in GPU memory with zero unnecessary CPU round-trips.
- Cross-platform parity matters — same scene file, same perf characteristics, on a MacBook (M-series) and on a Linux workstation/laptop with a discrete GPU.
- If modern browsers (WebGPU + WebCodecs + zero-copy `VideoFrame` → GPU texture) can genuinely match native GPU throughput for our workload (10× 1080p layers, sub-frame audio-reactive latency), the browser path is acceptable.
- If they can't — because of WebCodecs decoder limits, lack of HAP support, video-element fallback to software decode, missing Vulkan/Metal interop, or latency floors — prefer a **native app** (Electron/Tauri with a native render core, or a pure native Rust/Swift/C++ app using wgpu/Metal/Vulkan directly). The deciding factor is sustained realtime perf, not developer convenience.
- Decision should be made *before* Phase 2 locks in the renderer surface. The "hardware reality check" section above (10× 1080p, HAP vs WebCodecs vs VideoTexture) is the first concrete test of this — extend it with a Linux + dGPU equivalent.

## Why browser ≠ native for this workload (added 2026-05-12)

"It's just software on the same hardware" is correct in general — WebGPU is within ~10–30% of native for pure compute/render. The gap that matters here is structural, not a vague slowness, and it stacks specifically on the VJ workload:

1. **Process isolation tax.** Browser renderer process does not own the GPU; every command buffer goes via IPC to a separate GPU process. Fixed per-call cost. Native is in-process.
2. **Mandatory runtime validation.** WebGPU spec requires it for security. Native can compile it out in release.
3. **Video decode pipeline is the real killer.** Native (Metal/VideoToolbox, VA-API + Vulkan video) writes decoded frames straight into GPU memory, zero copy. Browser `HTMLVideoElement` → `VideoTexture` often does CPU readback + re-upload per frame. WebCodecs `VideoFrame` *can* be zero-copy to `GPUTexture` in modern Chromium but coverage is uneven.
4. **Chromium concurrent decoder cap (~4–6 streams).** This is a browser policy decision, not a hardware limit — the base M2 hardware decoder can do more. Past the cap, silent fallback to software decode.
5. **No HAP / HAP-Q.** The codec the VJ world standardised on. CPU does cheap Snappy decompression, GPU receives pre-compressed DXT/BC blocks directly — no hardware video decoder involved, scales with VRAM/disk. Browsers have no HAP support and no path to add one (not standardised). WASM decoder at 10×1080p means doing on the CPU what the GPU would do for free natively.
6. **Compositor latency floor.** Browser fullscreen is borderless-windowed-composited, not exclusive. Goes through Chrome's `viz` compositor → OS compositor → display. Native exclusive fullscreen bypasses the OS compositor — one less frame of latency, direct VSync control. Matters for sub-frame audio reactivity.
7. **No real-time scheduling hints.** Native can ask the OS for RT priority / thread affinity for audio threads. Browser can't.
8. **WebGPU feature subset.** Missing/limited: bindless, mesh shaders, ray tracing, subgroup ops, sparse textures, GPU-driven indirect rendering. Not blockers today, but options you can't reach for later.

Net: for a 3-layer scene you cannot tell the difference. For 10× 1080p + sub-frame reactivity + Linux parity, you hit the Chromium decoder cap, the compositor floor, the missing HAP, and the WebCodecs API edges — each a specific structural limit, not a vague perf gap. This is why every serious VJ tool (Resolume, VDMX, MadMapper, TouchDesigner) is native.

## Current leaning — native core + localhost browser UI (added 2026-05-12)

Direction now favoured: **native render core, browser-based control UI talking to it over localhost** (HTTP/WebSocket).

- Render core = native binary (Rust + wgpu is the strongest candidate; Tauri-wrapped is acceptable if Rust scope is too large). Owns: layer pack loading, mask texture array, video decode (hw + HAP), compositor, FX graph, audio analysis, MIDI/OSC ingestion, projector output (exclusive fullscreen).
- UI = browser app (React/TS, reuse most of the current `render-engine/` work — minus Three/R3F) served from the native core or as a separate dev server. Owns: scene authoring, binding editing, calibration UI, live sliders/mute toggles, audio-debug visualiser, agent-authored config editing. Talks to the core via a small JSON-RPC / WebSocket API.
- The browser UI is *control plane only*. No video pixels go through it. Preview thumbnails are fine (downsampled stream over WS), but the projector output is a native window.

Why this shape:
- Removes every browser ceiling above (decode cap, compositor latency, HAP, exclusive fullscreen) from the render path.
- Keeps the agent-authoring story intact — the scene format is still a text file an LLM can edit; the core just consumes it.
- Lets the UI stay web-tech, which is the most productive stack for live-tweak surfaces and the easiest target for Claude/MCP authoring.
- Cross-platform parity (Mac + Linux) is wgpu's whole job. Single render-core binary per platform; UI is the same everywhere.
- Future remote control / collaboration / web demo all collapse to "swap localhost for a tunnel."

Implications for the refactor plan:
- §5.4 (`VideoTexture`) is dropped; video path is native (FFmpeg/AVFoundation + wgpu interop, plus a HAP decoder).
- §5.5 (browser fullscreen) is dropped; native window owns the projector.
- §5.6 calibration shader stays, just moves into the native core.
- Phase 2's "compositor" is now a wgpu compositor, not TSL. The §3 schema (layer pack + bindings) survives unchanged — that was always the durable contribution.
- ISF spike still useful, but as a *spec to borrow* for the binding format. Running ISF shaders natively means a WGSL transpile path (or shipping naga + a small ISF→WGSL converter). Defer the runtime, adopt the schema now.
- TS migration value increases — the RPC contract between UI and core wants types on both ends.

Open spike before committing:
- Stand up a minimal Rust + wgpu app that loads a layer pack and renders 10× 1080p HAP video layers with a simple FX (hue cycle, additive flash). If sustained 60fps with sub-frame latency on base M2, the architecture is validated. Estimated 1–2 sessions for the spike.

## Open questions for next session

- Decide A vs B vs hybrid before more refactor planning. Hybrid is recommended but commit explicitly.
- If hybrid: define the minimum ISF subset to support (inputs: float, color, image, audio, audioFFT, event — skip multipass on v0?).
- Whether to bring TypeScript migration forward (plan says Phase 3; if ISF schema lands earlier, TS for that schema's types is cheap insurance).
- Confirm projector resolution and physical surface for the first real-world demo before Phase 4 calibration work.
