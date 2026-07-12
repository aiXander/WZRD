# WZRD — app collapse analysis

> **RETIRED 2026-07-12: Steps 2–3 LANDED, plan complete.** Both runtime
> spikes passed (clean teardown; crash containment + relaunch-to-light
> ~160–290 ms ≪ 20 s), all §5 validation criteria met (one process, two
> windows + preview child window, §3.11 identical, WS still serving
> external clients, headless binary unchanged), and the Step-3 native
> preview shipped in the same run — including §6.4's demand-gated JPEG
> capture. Durable residue lives in
> `../reference/render-engine.md` §1/§1b (topology, TauriHost rules,
> spike results, known warts). Leftover prerequisites for the first live
> show on this topology — shader pre-flight probe, design-leg autosave —
> live in `render-engine-roadmap.md` §5.6/§5.11 (they always did).
> Everything below is the historical analysis and decision record.

> **Status 2026-07-12: COMMITTED (operator decision).** Collapse is the
> plan, no longer pending: the two Step-2 runtime spikes (clean GPU
> teardown on Tauri exit; crash-must-not-corrupt-state + relaunch-to-light
> ≤ ~20 s) run as the **first tasks of Step 2** and act as a *fallback
> trigger*, not a gate — if they fail on macOS, revert to the subprocess
> split + §5.8 binary WS frames. Sequencing (also decided): **Steps 2–3
> land before the §5.6 two-deck build**, so the design-leg preview is
> built once, natively, on the final topology. Prerequisite unchanged:
> the shader pre-flight probe must exist before the collapsed build ships
> to a live show. This doc is now an implementation plan (§5), not an
> open decision.

> **Status 2026-07-11: operator decision inputs SETTLED — recommendation
> now leans COLLAPSE, pending only the Step-2 runtime spikes.** Three
> requirements were fixed in conversation with the operator:
> 1. **A rare ~20 s full blackout is acceptable** *if* state restore is
>    total (scene + knobs + masters + calibration + design draft). That's
>    the Resolume recovery formula — autosave + relaunch + recover — and it
>    **relaxes the §6.1 crash-recovery precondition**: in-process recovery
>    is now best-effort, relaunch-with-restore is the accepted backstop.
>    The state machinery is §5.3 (session sidecar, in flight) + the §5.6
>    design-leg autosave.
> 2. **AI-written shaders get a pre-flight probe** before touching any
>    plan (compile → ~60 offscreen frames at reduced res → p95 check →
>    thumbnail; roadmap §5.6). Crashes become rare by construction, not
>    by hope — this counters the "agent shaders make collapse riskier
>    than Resolume" argument at its source.
> 3. **The design-leg preview bar is "decent enough to judge effect
>    quality"** — near-full-res NOT required (§3.3 note). Both collapse
>    and the §5.8 binary-frames fallback can meet it; collapse wins on
>    margin and simplicity rather than being the only option.
>
> **Status 2026-07-10 (evening): Step 1 LANDED, static spikes ANSWERED.**
> The Core/Host split shipped: `render-core/src/core.rs` holds the
> host-agnostic `Core` (GPU, plan, drivers, telemetry, WS, watcher, frame
> pacing, occlusion policy); `app.rs` is now a thin `WinitHost`;
> `GpuContext::new` takes any `wgpu::SurfaceTarget` + explicit size instead
> of a winit `Window`. Standalone binary verified behaviour-identical
> (boot, plan build, §3.11 RPC surface, fps telemetry, occlusion →
> offscreen-at-30 Hz all exercised live). Spike results in §5 Step 2:
> **tao has no occlusion event** (mitigation known, bounded — poll
> `NSWindow.occlusionState`); **wgpu-surface-on-tao is compatible**
> (tao 0.35 + tauri 2.11 both implement rwh 0.6, same version wgpu 22
> uses). Remaining before Step 2 can land: the two *runtime* spikes
> (clean GPU teardown, §6.1 crash-recovery proof) and the §5.6 design
> review. The incremental alternative (binary WS frames, demand-gated
> capture) stays roadmap §5.8 in
> [render-engine-roadmap.md](render-engine-roadmap.md).
>
> **Plan revised 2026-07-10** after the roadmap restructure: the decision
> gate moved from "before Phase 5 (video)" to "as part of the §5.6
> two-deck design" (§3.3 — the design leg is where the preview becomes
> load-bearing), in-process crash recovery became a hard **precondition**
> for Step 2 (§6.1) — *superseded 2026-07-11: relaxed to state-integrity +
> relaunch-≤20 s, see §6.1* — and Step 3b was demoted from peer option to
> feasibility spike (§5 Step 3). **The full collapse commit now hangs on
> the Step-2 runtime spikes + the §5.6 design review only.**

> Working doc, 2026-05-22. Decides whether to keep the Phase-4 subprocess
> split (render-core ↔ Tauri shell over localhost JSON-RPC WebSocket) or
> collapse the two into a single process so the operator-UI preview can
> sample the engine's composite texture directly the way Resolume / MadMapper
> / VDMX do.
>
> The subprocess split has shipped
> end-to-end (Phases 4.1 + 4.2 landed); the preview pipeline works and is
> stable after the 2026-05-22 render-thread fixes (`PreviewSampler` async
> readback + 240 Hz frame cap). The question this doc answers is *whether
> the preview ceiling is high enough for the long-term creative workflow*,
> and *what it actually takes to lift it.*

---

## 1. What we're solving

The Tauri webview's preview thumbnail is currently a ~320 px JPEG at 15 fps,
shipped from the engine subprocess to the webview as base64 over WebSocket.
After the recent fixes it costs the render thread nothing meaningful, but
the **visual ceiling** is fixed by that pipeline:

- Lossy (JPEG q70)
- Low resolution (320 px hardcoded)
- Capped at ~15 fps
- Always a frame or two behind the projector output (readback → encode → IPC → React render)

Resolume's preview is none of those things — it's full-resolution, full-fps,
lossless, and indistinguishable from looking at the projector itself. The
question is: do we want that, and what would it take.

This doc is also the persistent answer to a related question the user asked
at the same time: *why is there a browser at `localhost:1420` AND a "wzrd-app"
window AND something rendering frames?* The architecture below makes that
explicit.

---

## 2. Current architecture (as built, 2026-05-22)

### 2.1 Process topology

Four sibling processes (when running in dev with everything wired up):

```
┌───────────────────────────────────────────────────────────────────────┐
│ Operator's laptop                                                     │
│                                                                       │
│  ┌──────────────────┐    ┌────────────────────────────────────────┐   │
│  │ Vite dev server  │    │ Tauri shell process (wzrd-app)         │   │
│  │ http://:1420     │◀───┤   - WebKit webview window (the UI)     │   │
│  │ (HMR, dev only)  │    │   - engine-io thread (WS client)       │   │
│  └──────────────────┘    │   - request/reply demux + telemetry    │   │
│                          │     fan-in via Tauri events            │   │
│                          └─────────┬──────────────────────────────┘   │
│                                    │ spawns at startup                │
│                                    │ ws://127.0.0.1:9123              │
│                                    ▼                                  │
│                          ┌────────────────────────────────────────┐   │
│                          │ render-core subprocess                 │   │
│                          │   - winit + wgpu native window         │   │
│                          │   - render thread (240 Hz capped)      │   │
│                          │   - WS server (JSON-RPC over text)     │   │
│                          │   - OSC UDP sink :9000 (audio)         │   │
│                          │   - file watcher (scene.json + effects)│   │
│                          └─────────┬──────────────────────────────┘   │
│                                    │ OSC features in                  │
│                                    │ udp://127.0.0.1:9000             │
│                                    ▼                                  │
│                          ┌────────────────────────────────────────┐   │
│                          │ Realtime Audio Feature Server          │   │
│                          │ (separate Python repo)                 │   │
│                          │   - capture, DSP, /audio/lmh, /onset/* │   │
│                          │   - own browser UI on :8765            │   │
│                          └────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

Three of those are "real" — the Tauri shell, the engine, the audio server.
The Vite dev server is *only* a dev-time HMR host; the WebKit webview loads
from `http://localhost:1420` while developing. In a `pnpm tauri build`
release bundle the Vite server is gone and the webview loads from the
bundled `frontendDist` (`wzrd-app/src-tauri/tauri.conf.json:9`). The
`localhost:1420` browser tab the user sees when running `pnpm tauri dev` is
*just the same Vite server in a regular browser* — it's empty because the
React app expects to be hosted inside Tauri's IPC environment. Closing the
browser tab changes nothing.

So in production it's **three** processes, in dev it's three plus the Vite
HMR server.

### 2.2 GPU contexts

Each native window owns its own GPU device:

- The Tauri shell's WebKit webview has its own Metal context that WebKit
  uses to composite HTML. Inaccessible from Rust.
- `render-core` creates its own `wgpu::Instance` + `wgpu::Device` +
  `wgpu::Queue` (`render-core/src/gpu.rs:187-217`). It owns:
  - The swapchain surface for the winit window (the "projector").
  - The composite buffer (`Rgba16Float`, pack-resolution, 1024×768 in the
    smoke scene).
  - The mask atlas (`Texture2DArray<R8>`).
  - The pipeline cache (built-ins + user WGSL effects).

These two contexts cannot share textures. They're in different processes,
running on different (logical) GPU devices with no IPC primitive that maps
GPU memory between them. The kernel has IOSurface on macOS, which is the
underlying mechanism Syphon and similar systems use — but neither WebKit
nor wgpu expose it directly to us, and the webview cannot sample an
IOSurface from a JavaScript `<img>` element regardless.

### 2.3 The preview pipeline (today)

Every 66 ms on the render thread (`render-core/src/telemetry.rs:367-525`,
post-2026-05-22 fixes):

1. **Allocate-once readback buffer.** A `wgpu::Buffer` sized for the full
   composite (`bytes_per_row * src_h`, e.g. 15360 × 768 ≈ 12 MB at 1024×768).
   Allocated lazily and reused across captures; only re-allocated if the
   composite changes shape.

2. **Submit `copy_texture_to_buffer`.** The composite (already pre-emptively
   tagged `COPY_SRC` since the post-Phase-3 review) is copied to the readback
   buffer on the GPU queue. Fast.

3. **Async map.** `buffer.slice(..).map_async(MapMode::Read, cb)` is fired;
   the callback flips an `AtomicU8` from `MAP_PENDING` to `MAP_READY_OK`
   when the data is mappable on the CPU. **The render thread does not
   wait.** It returns from `maybe_capture` immediately.

4. **On a subsequent `maybe_capture` call** (the next frame or two later),
   the render thread checks the atomic. If `MAP_READY_OK`, it:
   - Manually decodes `Rgba16Float` → f32 per sampled pixel (no `half` dep).
   - Downsamples to a 320 px wide RGB8 thumbnail (~76 KB raw).
   - JPEG-encodes at quality 70 via `image::codecs::jpeg::JpegEncoder`
     (~10–20 KB out).
   - Base64-encodes (`base64::engine::general_purpose::STANDARD`,
     ~13–27 KB out).
   - Emits on the telemetry bus channel `preview`
     (`render-core/src/telemetry.rs:200-212`).
   - Unmaps the buffer and clears the in-flight state.

5. **Telemetry bus → WS notification.** The per-subscriber bounded channel
   (`SUBSCRIBER_CAP = 256`) carries the frame to the WS conn thread, which
   wraps it as a JSON-RPC `telemetry.event` notification and writes it to
   the socket.

6. **Tauri shell engine-io thread reads** the notification
   (`wzrd-app/src-tauri/src/engine.rs:236-282`), records sticky channels,
   and forwards every frame as a Tauri `engine:telemetry` event.

7. **React webview** (`wzrd-app/src/App.tsx:60-95`) receives the event,
   updates `useStore.setPreview`, triggers a re-render. The
   `<PreviewThumbnail>` and `<SurfaceCanvas>` components decode the base64
   data URL into an `<img>` / canvas blit.

**Each step exists for a specific reason:**

| Step                | Why it exists                                                 |
| ------------------- | ------------------------------------------------------------- |
| GPU→CPU readback    | Engine and webview are different processes / GPU contexts.    |
| Downsample to 320 px | Design choice — "glanceable thumbnail", not primary surface. |
| JPEG encode         | Raw composite is ~6 MB/frame → 90 MB/s at 15 Hz. JPEG → ~225 KB/s. |
| Base64 encode       | JSON-RPC is text-only; can't put raw binary in a JSON string. |
| WS hop              | Subprocess split — the only way to cross the process boundary.|

None of those are stupid. Each one is the local minimum given the
architecture above it.

### 2.4 Why the architecture is this shape

From the retired v1 design doc's Phase-4 status block (decision carried
forward as P4 in `../reference/render-engine.md`):

> **Phase 4 architectural choice — subprocess + WS, not in-process winit.**
> The original spec envisioned Tauri + winit sharing one event loop in one
> process (§6.1). On macOS that's a real spike (NSApp main thread; exclusive
> fullscreen interactions with webview focus changes; cross-window ownership).
> Rather than fight it, Phase 4 lands as Tauri shell ↔ render-core subprocess
> over localhost JSON-RPC WebSocket.

What the split bought:

- ✅ Headless agent path (`render-core --scene foo.json`) is byte-identical;
  agents and CI deploy one binary, no UI dependency.
- ✅ Same RPC surface MCP will need in Phase 7 — only the transport
  (Tauri command vs raw WS client) differs by consumer.
- ✅ Engine crash doesn't take the UI down; UI crash doesn't take the
  projector down.
- ✅ Sidesteps every macOS NSApp / Spaces / focus issue that an in-process
  winit-inside-Tauri would have hit.

What the split cost:

- ❌ Cannot share GPU textures between engine and UI → preview pipeline
  must serialize pixels through some transport.
- ❌ Subprocess management complexity (spawn, supervise, kill on shell close).
- ❌ One extra hop on every RPC request (Tauri command → in-process WS
  client → engine WS server → render thread).

---

## 3. The problem

The preview ceiling. Concretely:

- **Lossy + low-res.** A JPEG q70 320 px thumbnail is fine for "is the
  projector alive" but kills any subtle visual you actually want to judge
  (gradients, fine masks, subpixel motion, additive blend artifacts).
- **Laggy.** Two-frame latency floor (readback + IPC + React render) at
  best. The user is always looking at a slightly stale composite.
- **Bandwidth-bounded if scaled.** Going to full-res lossless at 60 Hz
  through the JSON-RPC text channel is ~50 MB/s of base64. Tolerable on
  localhost, but the encode/decode cost shifts to be the bottleneck.
- **Doesn't match the obvious comparable.** Resolume / VDMX / MadMapper
  / Arena all have indistinguishable-from-projector preview. Operators
  coming from any of those tools will notice immediately.

### 3.1 What Resolume does

Single process. Single GPU context. The "preview" is **another viewport
sampling the same composite texture** that gets sent to the projector
output. One extra draw call per frame. No readback, no encode, no IPC,
no compression. ~Free.

### 3.2 What we incorrectly claimed earlier

In an earlier conversation thread, the subprocess split was partially
defended on the grounds that *"MCP needs the same WS surface, so the
engine has to expose a WS server, so it might as well live in its own
process."*

That argument is wrong. MCP and process topology are orthogonal:

- **File-based agent path** is already the canonical contract
  (D13 in `../reference/render-engine.md`). LLMs write `scene.json` and
  `effects/*.wgsl`; the file watcher reloads. No WS needed.
- **RPC-based MCP** can connect to a WS server that lives *inside* a
  collapsed (single-process) shell just as easily as one inside a
  subprocess. The WS server is a thread, not a process — its address
  binding and protocol are identical either way.
- **The headless agent binary** (`render-core --scene foo.json`) stays
  exactly as it is — it's a separate `[[bin]]` target that doesn't load
  any Tauri code.

So the *real* and *only* blocker on collapsing is the §6.1 macOS spike:
winit ↔ tao event-loop cohabitation, NSApp main-thread ownership, and
the focus / fullscreen / Spaces interactions between the wgpu window
and the webview window when they share one process.

### 3.3 Where the ceiling actually binds — the design leg (added 2026-07-10)

The Resolume comparison in §3 overstates the *live-leg* pain. WZRD is a
projection-mapping tool: in the core workflow the operator is in the room
with the surface and judges the live output by looking at physical
reality — the thing the preview approximates is standing right there. A
glanceable thumbnail for "is the projector alive / roughly what's playing"
is a defensible live-leg preview indefinitely.

What changes the calculus is roadmap §5.6 (design/live two-deck). The
design leg renders **only to an offscreen composite** — it is never on the
projector, so its preview channel is the *only* way the operator ever sees
it, and promote decisions ("does this draft go to the crowd?") are judged
entirely through it. That is where a 320 px q70 JPEG at 15 fps fails the
design spec: gradients, fine mask edges, and additive blend artifacts are
exactly what you need to check *before* promoting.

Consequence: **the collapse decision is a §5.6 design input, not a
standalone preview optimization.** Make the call while designing the
two-deck — and before building §5.8's design-leg `PreviewSampler`, which
is throwaway work under collapse.

**Preview bar fixed (2026-07-11).** The operator's actual design workflow
is prompting an AI for new shaders and judging drafts on screen before
they go live — so a design-leg preview is unconditionally required, but
the bar is "**decent enough to judge effect quality and decide what to do
next**," not lossless-indistinguishable. Practical floor: roughly half
pack resolution at ~30 fps. That is reachable by the §5.8 incremental
path (binary WS frames) as well as by collapse — so the preview alone no
longer *forces* collapse; it makes some upgrade mandatory and leaves the
topology choice to the recovery/robustness trade-off in §6/§8.

---

## 4. Proposed solution — collapse to a single process

### 4.1 What "collapse" actually means

The Tauri shell process directly owns:

- The webview window (today, unchanged).
- A second native window for the engine output, created via Tauri's window
  API (which uses `tao`, a winit fork) instead of winit directly.
- A `wgpu::Surface` attached to that second window via `raw-window-handle`.
- All of the existing `render-core` core logic — pack loader, compositor,
  driver bus, effect registry, OSC sink, file watcher — running on a
  render thread inside the same process.
- (Optional, unchanged) the WS server on `127.0.0.1:9123` for external
  MCP clients.

What goes away:

- The subprocess spawn (`wzrd-app/src-tauri/src/engine.rs:64-119`).
- The engine-io thread, request/reply demux, telemetry fan-in over WS.
  Tauri commands call into `render-core` core logic directly.
- The preview JPEG / base64 / WS notification path. Replaced with a
  second `wgpu::Surface` (or a child `wgpu::Texture` rendered to a
  Tauri window) sampling the same `composite_texture` the projector
  window samples.

What stays:

- The standalone `render-core` binary as a separate `[[bin]]` target —
  still uses winit, still works headless, still byte-identical for
  agents and CI.
- The WS JSON-RPC surface — still served from inside the collapsed
  process for external MCP consumers and remote operators.
- The whole RPC method set (`§3.11`) — local Tauri commands call the
  same dispatch function the WS server calls.
- The audio feature server as a separate sibling process — unchanged.

### 4.2 Why "MCP needs the same WS" doesn't block collapse

| Use case                          | Subprocess today           | Collapsed                    |
| --------------------------------- | -------------------------- | ---------------------------- |
| Local Tauri UI → engine           | Tauri cmd → WS → engine    | Tauri cmd → direct Rust call |
| Remote MCP client → engine        | direct WS to engine        | direct WS to collapsed app   |
| Headless agent / CI               | run standalone binary      | run standalone binary        |
| File-based LLM workflow           | file watcher in engine     | file watcher in engine       |

All four still work. The collapsed process just exposes a WS server as
one of its threads — exactly the way the subprocess does — for everyone
who isn't the local UI.

### 4.3 What's actually blocking

Three real items, in descending order of risk:

1. **winit ↔ tao event-loop cohabitation on macOS.** Both want to own
   `NSApp` on the main thread. Solution: don't use winit inside the Tauri
   process. Use Tauri's window API (which uses tao) to create the engine
   window, hand its `raw-window-handle` to wgpu. The standalone
   `render-core` binary keeps using winit and is unaffected.

2. ~~**`App` is hard-wired to winit's `ApplicationHandler`.**~~
   **RESOLVED — Step 1 landed 2026-07-10.** The Core/WinitHost split
   exists (`render-core/src/core.rs` + the thin `app.rs`); see §5 Step 1
   for the API residue. What remains of this item is only the
   **TauriHost** itself (Step 2): own a Tauri window's handle, drive
   `Core` on a render thread, bridge Tauri commands to Core's dispatch.

3. **macOS webview focus / Spaces / exclusive-fullscreen interactions.**
   When the React webview takes focus, does the wgpu surface on the
   secondary display lose its exclusive-fullscreen mode? When the user
   command-tabs, do both windows reshuffle as expected? When Spaces
   moves the webview to another space, what happens to the engine
   window?

   The exclusive-fullscreen variant has historical hazards; the
   borderless-fullscreen variant (already the engine's default fallback
   per `render-core/src/app.rs:321-325`) is much better behaved. The
   collapse plan assumes borderless-fullscreen — same as today.

### 4.4 What collapse buys

- **Resolume-style preview.** Bind a small render pipeline in the Tauri
  window that samples `composite_texture` and draws it to a quad. One
  extra draw call. No readback. No encode. No IPC. Lossless,
  full-resolution, full-fps, sub-frame latency. Resize the preview
  window however you want.
- **The preview thumbnail's existence stops being load-bearing on the
  engine.** Engine no longer needs `PreviewSampler`, the
  base64/JPEG/`preview` telemetry channel, or the COPY_SRC tag on the
  composite (well — keep the tag, it's free).
- **One less hop on every Tauri-originated RPC.** Local UI calls run as
  direct Rust function calls instead of serializing through WS. Latency
  goes from ~1 ms to microseconds.
- **No more subprocess supervision.** No spawn, no shutdown, no
  "engine WS at addr never came up within 5s" failure mode
  (`wzrd-app/src-tauri/src/engine.rs:294-300`).

### 4.5 What collapse costs

- **Refactor effort.** Step 1 (Core/Host split) is a half-day of safe,
  useful-on-its-own work. Step 2 (Tauri-embedded host + the wgpu-on-tao
  wiring) is 2-ish days plus an unknown macOS surprise tax. Step 3
  (delete the preview pipeline) is an hour.
- **Two host wrappers to maintain.** WinitHost (for the standalone
  binary) and TauriHost (for the shell). Both delegate to the same Core;
  divergence risk is low.
- **No more subprocess crash isolation — and this collides with §5.11.**
  A `panic!` in the render thread, a wgpu device loss, or a runaway shader
  takes the Tauri shell down with it. Today the shell survives an engine
  crash, and roadmap §5.11 explicitly plans to exploit that ("shell
  supervises the engine child: crash → respawn with the same scene within
  seconds"). Collapse deletes that mechanism. Worse, WZRD's differentiator
  — agent-written WGSL hot-swapped mid-show — makes render-thread trouble
  *more* likely than in Resolume, whose plugins are vetted native code.
  *(2026-07-11: this cost was re-priced and accepted — the pre-flight
  probe attacks the frequency, the autosave/restore contract bounds the
  damage; see the relaxed §6.1.)*
- **Engine GPU context lifetime now tied to Tauri's window lifecycle.**
  The render thread has to start after Tauri's setup hook (when the
  engine window exists) and stop cleanly on app exit.
- **macOS-only spike risk.** `§6.1`'s NSApp / Spaces / focus questions
  are still real — they were sidestepped, not solved. If they bite,
  Linux + Windows builds still work fine and we'd be debugging a
  Mac-specific edge case.

---

## 5. Staged implementation path

Each step is useful on its own and reversible without losing work.

### Step 1 — split `App` into Core + WinitHost — **DONE (2026-07-10)**

Landed as planned; residue for whoever builds Step 2:

- **`render-core/src/core.rs` — `Core`**, host-agnostic. Owns GPU context,
  pass plan, effect registry, transport, OSC, telemetry bus, WS server,
  file watcher, slider bank. Also owns the two policies both hosts must
  share identically: the §3.1 occlusion invariant (`set_occluded` /
  `occluded()` / `render_offscreen_frame`) and frame pacing (`pace_frame`).
- **Two-stage init** (one deviation from the original sketch, deliberate):
  `Core::new(&cli)` does everything pre-window (pack/scene load, OSC, WS
  server — so load errors still fail fast with a non-zero exit before the
  event loop starts), then `Core::init_gpu(target, width, height)` brings
  up wgpu. `target` is `impl Into<wgpu::SurfaceTarget<'static>>` — a winit
  `Arc<Window>`, a tao/tauri window, anything rwh-0.6. Width/height are
  passed explicitly because a raw handle can't be queried for size.
- **Per-frame host contract:** `poll_inbound()` (drain IPC + watcher +
  telemetry heartbeats) → `pace_frame()` → either `redraw()` (returns
  `wgpu::SurfaceError`; on `Lost`/`Outdated` the host queries the window
  size and calls `resize`) or `render_offscreen_frame()` while occluded.
- **`app.rs` is now `WinitHost`** (~170 lines): window creation +
  fullscreen/display selection, event delegation, exit decisions. Owns the
  `Arc<Window>`; `GpuContext` no longer stores a window at all and
  `gpu.rs` has zero winit imports.
- `rpc::handle` now takes `&mut Core`; `lib.rs::run` wires `WinitHost`.

**Validated:** `cargo build` + `cargo test` green, `wzrd-app/src-tauri`
`cargo check` green (subprocess path untouched); live windowed run
exercised boot, plan build, all §3.11 RPC methods over WS, fps telemetry,
and the occlusion → offscreen-30 Hz path (`presenting: false` observed).

### Step 2 — TauriHost + wgpu-on-Tauri-window (~2 days + surprise tax)

**Goal:** Tauri shell process owns both the webview window and the engine
window; `render-core` runs as a library inside that process.

- Add a second Tauri window in `wzrd-app/src-tauri/src/lib.rs` for the
  engine output. Mark it borderless, positionable by display index.
- On Tauri's `setup` hook:
  - Create the engine window.
  - Get its `raw-window-handle` via `tauri::WebviewWindow::raw_window_handle`.
  - Spawn a render thread that creates `Core::new(cli, handle)` and runs
    a redraw loop with the existing 240 Hz cap.
  - The thread receives commands from Tauri commands via an
    `EngineCommand` channel — same shape as today's `cmd_rx` between WS
    server and render thread.
- Replace `EngineHandle::spawn` (which forks a subprocess) with
  `EngineHandle::start_in_process` that boots the render thread.
  Keep the same `request(method, params)` API surface so `rpc.rs`
  doesn't change.
- Keep the WS server thread alive inside Core for external MCP /
  remote operator use. Tauri commands skip it entirely — they call
  `Core::dispatch_rpc(method, params)` directly.
- Headless `render-core` binary is untouched; it still uses
  `WinitHost`.

**Validation:** `pnpm tauri dev` produces two windows from one process
(webview + engine), no subprocess in `ps`. All §3.11 RPC methods work
identically. WS server still binds and accepts external clients.
Headless binary still runs.

**Risk hotspots — spike results (static ones answered 2026-07-10 by
reading the locked dependency sources):**

- **tao occlusion signal: ANSWERED — NO.** tao 0.35.2's `WindowEvent` has
  no `Occluded` variant and its source contains no occlusion handling at
  all (winit grew `Occluded` in 0.27; tao forked earlier). The §3.1
  invariant — never block on the swapchain of a possibly-occluded window —
  therefore needs another trigger in TauriHost. Mitigation is known and
  bounded: poll `NSWindow.occlusionState & NSWindowOcclusionStateVisible`
  once per frame on the render thread (a cheap AppKit property read; the
  `NSWindow` is reachable from the rwh 0.6 `AppKitWindowHandle`'s
  `ns_view.window`), or observe
  `NSWindowDidChangeOcclusionStateNotification`. Budget this into Step 2;
  it is no longer an unknown.
- **wgpu surface from a Tauri window: ANSWERED — YES.** tao 0.35.2's
  `Window` implements `rwh_06::HasWindowHandle` + `HasDisplayHandle` and
  is `Send + Sync`; tauri 2.11's `Window<R>` and `WebviewWindow<R>` both
  implement the same rwh 0.6 traits directly. Both lockfiles resolve a
  single `raw-window-handle 0.6.2` — the version wgpu 22 consumes — so a
  tauri window satisfies `Core::init_gpu`'s
  `impl Into<wgpu::SurfaceTarget<'static>>` bound as-is, no adapter code.
- **winit-vs-tao event-loop conflict: MOOT by construction.** After
  Step 1, `Core` has no winit dependency; the Step-2 plan never creates a
  winit event loop inside the Tauri process.

**Still open — runtime spikes (need an actual embedded TauriHost):**
- Does the engine window cleanly close when Tauri exits, releasing GPU
  resources?
- The §6.1 crash-recovery proof: a deliberately-panicking effect and a
  forced device loss must leave the webview alive.

### Step 3 — wire the Resolume-style preview (~half day)

**Goal:** The Tauri UI's preview panels sample the engine's composite
texture directly.

- Add a third wgpu render pipeline inside Core: a fullscreen-quad
  shader that samples `composite_texture` and writes to a target
  texture or surface.
- Delivery: **plan around (a) — a native preview surface layered over
  the Tauri window.** A small native subwindow/child view positioned
  over the React layout's "preview" slot, rendered to by the same wgpu
  device. UI chrome overlaps via z-order; the tax is layout integration
  with React (position sync on scroll/resize), which is known and
  bounded.
- **(b) — texture-to-webview via a custom Tauri plugin — is demoted to a
  30-minute feasibility spike, not a peer option** (revised 2026-07-10).
  As originally written ("wrap the texture as a video stream the
  `<video>` element can consume") it hand-waved the hard part: there is
  no clean Tauri path from a wgpu texture into a webview `<video>`
  without re-encoding frames — which reintroduces the exact pipeline
  this step deletes. Pursue it only if the spike finds a genuinely
  zero-copy IOSurface→WKWebView route.
- Delete `PreviewSampler` as the *local* preview path. Keep the
  `preview` telemetry channel for remote-operator / MCP clients (the
  only way to see the projector over WS), but emit only on subscriber
  demand — see §6.4.

Remaining Step-3 design detail (position sync mechanics for the native
surface): decide after Step 2 lands and we know what the layout actually
wants to look like.

---

## 6. Preconditions & decisions (revised 2026-07-10)

1. **Crash recovery — RELAXED (2026-07-11), no longer a hard blocker.**
   The operator accepted the Resolume contract: a rare ~20 s blackout is
   fine *if restore is total*. The recovery story is therefore layered,
   and only the last layer is mandatory:
   - *Best effort, in-process:* `catch_unwind` around the render-thread
     loop → rebuild Core (device, plan, last-good scene) while the
     webview stays alive; same path for wgpu device loss. Build it if
     the Step-2 spike shows it's cheap; don't gate the collapse on it.
   - *Accepted backstop, always:* relaunch-with-restore. **The real
     precondition moves to state integrity:** a panic at any moment must
     never corrupt or lose the §5.3 sidecar, the scene file, or the
     §5.6 design-leg autosave (atomic writes: temp file + rename), and
     relaunch-to-light must be proven ≤ ~20 s.
   - *Frequency control:* the §5.6 shader pre-flight probe (compile +
     offscreen timed run + thumbnail before any plan swap) keeps the
     crash rate near zero in the first place; §5.11's probation window
     backs it up at full res. This is what Resolume can't do for
     third-party FFGL — WZRD tests its "plugins" automatically before
     every swap.
   - Residual known-unfixable: a shader that hard-hangs the GPU device
     in-process. Covered by the backstop (relaunch); if it turns out to
     happen in practice, a separate probe *process* with its own Metal
     device is the escalation (works on the single M2 GPU — macOS
     contains most faults to the offending process's command buffers).

2. **Step 3 delivery: RESOLVED — plan around 3a (native surface).**
   3b is a feasibility spike only; see §5 Step 3 for the reasoning.

3. **When: DECIDED (2026-07-12) — collapse first, then §5.6.** The call
   is made: Steps 2–3 land before the two-deck build, so the design-leg
   preview lands natively once. (Historical framing: §3.3 argued the call
   belonged inside the §5.6 design because the design leg is where the
   preview becomes load-bearing — that is what forced the decision now.)

4. **Do we keep the JPEG/base64 preview channel post-collapse? Yes.**
   It's the only way a remote MCP client over WS can see what's on the
   projector. Emit only when a subscriber actually asks for it —
   demand-gating survives either outcome of this decision and is safe
   to build anytime.

---

## 7. Explicitly out of scope

- **Syphon / Spout / cross-process GPU sharing.** Real and well-supported
  in the VJ ecosystem, but the receiver must be native code; a webview
  `<img>` can't subscribe. Would solve the cross-process case but
  requires a native preview surface anyway — at which point we might
  as well collapse.
- **Recording / video export.** Not in v1; not changed by collapse.
- **Multi-projector / edge-blending.** Out of v1; orthogonal to
  collapse.
- **Replacing the WS surface with something other than JSON-RPC.**
  The MCP contract assumes JSON-RPC; keep it. Collapse is only about
  *who hosts* the WS server, not what it speaks.

---

## 8. Decision (committed 2026-07-12; recommendation text of 2026-07-11 below)

**Collapse is COMMITTED** — the operator took the recommendation on
2026-07-12. Execution order: (1) Step-2 runtime spikes as the first
hours (fallback trigger — revert to subprocess + §5.8 binary frames only
if they fail), (2) land Steps 2–3, (3) then build §5.6 two-deck with a
native design-leg preview. Pre-flight probe + design-leg autosave +
atomic sidecar writes ship alongside Step 2 as planned in point 5 below.

**Collapse is now the preferred outcome** — deliberately adopting the
Resolume formula (single process + rare crashes via pre-flight + fast
total-restore recovery + direct-sampled preview) rather than drifting
into it. What changed: the operator fixed the three decision inputs
(status block at top) — blackout tolerance, shader pre-flight, and a
judgeable-not-lossless preview bar — which dissolved the crash-isolation
argument that previously kept the subprocess split ahead.

1. ~~**Do Step 1 now, unconditionally.**~~ **DONE 2026-07-10** — see §5
   Step 1. The engine now has a host-agnostic `Core`, which is also the
   natural home for the two `PassPlan` slots §5.6 needs.

2. **Only the two runtime spikes still gate the commit** (static ones
   answered 2026-07-10: tao occlusion → NO, poll `NSWindow.occlusionState`;
   wgpu-on-tao → YES; winit conflict → moot): clean GPU teardown on Tauri
   exit, and the *relaxed* §6.1 proof — a deliberately-panicking effect
   and a forced device loss must not corrupt saved state, and
   relaunch-to-light must land ≤ ~20 s. These are the first hours of
   Step 2 itself.

3. **Sequencing stays with §5.6:** make the formal call while designing
   the two-deck, land Steps 2–3, then build the design-leg preview
   natively. The §5.8 binary-frames path remains the documented fallback
   — and per the 2026-07-11 preview bar it genuinely suffices — if the
   runtime spikes surprise us on macOS.

4. **Until the call is formalized, still don't build §5.8's binary-WS
   preview frames or design-leg `PreviewSampler`** — throwaway under
   collapse. (Demand-gated capture is fine anytime; it survives either
   outcome as the remote-thumbnail path.)

5. **Prerequisites to schedule alongside Step 2** (both useful under
   either topology, both roadmap §5.6): the shader pre-flight probe and
   the design-leg autosave; plus atomic sidecar writes (§6.1). The
   collapse should not ship before the pre-flight probe exists — it's
   the mechanism that makes single-process acceptable with AI-authored
   shaders.
