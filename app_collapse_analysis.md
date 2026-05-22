# WZRD — app collapse analysis

> Working doc, 2026-05-22. Decides whether to keep the Phase-4 subprocess
> split (render-core ↔ Tauri shell over localhost JSON-RPC WebSocket) or
> collapse the two into a single process so the operator-UI preview can
> sample the engine's composite texture directly the way Resolume / MadMapper
> / VDMX do.
>
> Status: **analysis only, no commitment.** The subprocess split has shipped
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

From `render_engine_architecture.md:25-40` (the Phase-4 status block at the
top of the design doc):

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
  (D13 in `render_engine_architecture.md`). LLMs write `scene.json` and
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

2. **`App` is hard-wired to winit's `ApplicationHandler`.**
   `render-core/src/app.rs:333-446` implements `winit::application::ApplicationHandler`
   directly. To support both hosts, the engine has to split into:
   - **Core** (host-agnostic): GPU context lifecycle, pass plan, compositor,
     driver bus, OSC sink, effect registry, file watcher, telemetry bus,
     WS server. Knows nothing about winit or tao.
   - **WinitHost** (the standalone binary): owns the winit event loop and
     calls into Core on each event. Existing `App` minus the GPU/render
     logic.
   - **TauriHost** (the collapsed shell): owns a Tauri window's
     `raw-window-handle`, drives Core on a render thread, bridges Tauri
     commands to Core's RPC dispatch.

   This is mostly a refactor — every piece exists today, just glued to
   the winit handler.

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
- **No more subprocess crash isolation.** A `panic!` in the render
  thread, a wgpu device loss, or a runaway shader takes the Tauri shell
  down with it. Today the shell survives an engine crash and can
  potentially relaunch it.
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

### Step 1 — split `App` into Core + WinitHost (~half day)

**Goal:** Standalone binary unchanged, but the engine's GPU + render logic
no longer assumes winit.

- New module `render-core/src/core.rs` (working name):
  - Holds `GpuContext`, `PassPlan`, `EffectRegistry`, `Transport`,
    `AudioFeatures`, the bus, the WS server, the file watcher.
  - Public API: `Core::new(cli, window_handle) -> Result<Self>`,
    `Core::resize(w, h)`, `Core::redraw() -> Result<(), wgpu::SurfaceError>`,
    `Core::poll_inbound()` (drain command queue + watcher + emit
    audio_freshness heartbeat).
  - Receives a `raw-window-handle` instead of a `winit::Window` so it
    doesn't care which crate created the window.
- `render-core/src/app.rs` becomes a thin `WinitHost` that owns the
  winit `ApplicationHandler`, creates the winit window, hands its handle
  to `Core::new`, and delegates every event.
- `pub fn run(cli)` in `lib.rs` wires `WinitHost` exactly like today.

**Validation:** standalone binary behaves identically. `cargo run -- --scene
examples/phase3_smoke.scene.json --windowed` produces the same fps, same
hot-reload behavior, same OSC ingest. Tauri subprocess path also unchanged
(it just spawns the same binary).

This step is useful even if we *never* collapse — it cleanly separates
"engine logic" from "winit event loop" and makes the engine more testable.

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

**Risk hotspots to spike first** (each ~30 min):
- Can wgpu actually create a surface from a Tauri window's
  `raw-window-handle` on macOS? (Should — wgpu is `raw-window-handle`-native
  and tao implements the trait.)
- Does winit's `EventLoop::new()` conflict if a tao event loop is already
  running? (Don't call it from inside the Tauri process — that's why we
  use tao's window API, not winit's.)
- Does the engine window cleanly close when Tauri exits, releasing GPU
  resources?

### Step 3 — wire the Resolume-style preview (~half day)

**Goal:** The Tauri UI's preview panels sample the engine's composite
texture directly.

- Add a third wgpu render pipeline inside Core: a fullscreen-quad
  shader that samples `composite_texture` and writes to a target
  texture or surface.
- Two delivery options inside the same window architecture:
  - **(a) Native preview surface as a child of the Tauri window.**
    A small native subwindow positioned over the React layout's
    "preview" slot, rendered to by the same wgpu device. UI controls
    overlap via z-order. Cleanest visual result; trickiest layout
    integration with React.
  - **(b) Render to a shared GPU texture, expose to the webview via
    a custom `tauri-plugin` that wraps the texture as a video stream
    the `<video>` element can consume.** Avoids the layout-integration
    problem at the cost of one Metal `IOSurface` hop. Possibly the
    sweet spot.
- Delete `PreviewSampler` and the `preview` telemetry channel emission
  path. Keep the channel *name* reserved in case a remote-operator
  MCP client still wants a JPEG thumbnail; emit only on subscriber
  demand.

Open design question for Step 3: which delivery option above? Decide
after Step 2 lands and we know what the layout actually wants to look
like.

---

## 6. Open questions / decisions needed before starting

1. **Are we OK losing subprocess crash isolation?** A shader-induced
   GPU hang or wgpu device-loss event currently takes only the engine
   subprocess down; the shell survives. Post-collapse, both go down
   together. In practice: how often does this matter? Modern wgpu
   device loss is rare; user-WGSL is already `naga`-validated before
   compile; the swap-on-success pipeline lifecycle (`§3.6`) means a
   bad shader keeps the previous good pipeline. Probably fine but
   worth confirming.

2. **Native preview surface vs shared-texture-via-plugin (Step 3a vs 3b).**
   3a is straightforward but couples the React layout to native window
   positioning. 3b is cleaner from React's perspective but requires a
   custom Tauri plugin (small) and uses an `IOSurface`-ish path that's
   Mac-specific to start.

3. **When?** Phase 4.2 just landed. Phase 5 (video / HAP) and Phase 7
   (MCP wrapper) are the next planned chunks. Collapse can slot in
   before either — it doesn't block anything — but it's also explicit
   non-load-bearing scope. Doing it before Phase 5 means video decode
   lands on the collapsed architecture. Doing it after means one more
   round of subprocess plumbing for the video path.

4. **Do we keep the JPEG/base64 preview channel at all post-collapse?**
   It's the only way a remote MCP client over WS can see what's on the
   projector. Probably yes, but emit only when a subscriber actually
   asks for it (already a planned optimization independent of collapse).

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

## 8. Recommendation

Do **Step 1** unconditionally — it's a safe refactor, half a day, and
makes the engine cleaner regardless of whether we ever do Step 2. It
unblocks the option without committing to it.

Decide on **Step 2** before starting Phase 5 (video). The video path
is heavier infrastructure than the engine has today; landing it on the
collapsed architecture means one fewer cross-process boundary for
decoded frames to cross.

Treat **Step 3** as the payoff that makes the whole thing worth doing.
If we're not going to use the in-process GPU context for a
Resolume-quality preview, we don't need to collapse — the current
subprocess split is fine.
