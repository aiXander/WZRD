# Performance monitor + perf gate (plan, 2026-10-10)

**Goal.** Every frame's cost is attributable to a named part of the stack
(render-thread stage, GPU pass, binding, side thread, IPC), visible live in
the app, and checked by a scripted **perf gate** every coding agent runs
after a change — so "this feature added 4 ms of GPU" is caught the day it
lands, not on show night.

Status: **PM1 (CPU frame profiler) shipped 2026-10-10** — current state in
[reference/render-engine.md](../reference/render-engine.md) §2.9. PM2–PM4
below are still design. Build order at the bottom.
Current telemetry contracts this extends: [reference/render-engine.md](../reference/render-engine.md)
§2.3 (channels), §3.1/§3.2 (occlusion + honest-fps invariants), §6 (never
block the render thread).

---

## 1. What today's numbers can't tell us

- **`fps` / `frame_stats` measure the frame *interval*, not frame *cost*.**
  `FpsAccumulator::mark` (`telemetry.rs`) times mark-to-mark, which includes
  the `pace_frame` sleep and the swapchain wait. Capped at 60, a 2 ms frame
  and a 15 ms frame both read "60 fps, p50 16.7". **There is no headroom
  number** — the first symptom of a costly feature is already a dropped
  frame. Every gig gate ("seven clips at 60 fps p95") is therefore pass/fail
  with no margin visible.
- **No attribution.** One wall-clock number covers IPC drain, watcher, probe
  steps, driver ticks, encode, submit, swapchain wait, preview blit and JPEG
  readback. No GPU time at all (reference §4 weakness #8).
- **Side threads are invisible.** A clip decoder falling behind shows the
  previous frame (by design — a stutter, never a stall), so `fps` stays
  perfect while the video visibly judders. WS threads, the Tauri telemetry
  fan-in and the webview's event load aren't measured either.
- **No leak signal.** Reference §4 lists HMR GPU-resource leaks as a latent
  risk with "verify GPU memory plateaus" as the check — nothing measures it.
- **No latency numbers** (knob → frame, OSC onset → frame), though "knob
  latency ≤ 1 frame" is a stated invariant (§3.4).

## 2. Measurement model — three tiers

| Tier | Cost | When | Answers |
|---|---|---|---|
| **A. Always-on frame profile** | ≤ 0.1 ms/frame | every frame, every build | "where does the frame go right now, how much headroom is left" |
| **B. Deep profile** (`perf.profile`) | inflates frames while running | on demand, ~2 s | "which *binding* costs what" |
| **C. Bench harness** (`--bench`) | n/a (offline) | after every change, by agents | "did this change make anything slower than the baseline" |

### 2.1 The stage taxonomy (the contract that makes attribution stable)

Fixed enum, never free-form strings in the hot path. New features add a
stage here rather than hiding inside an existing one — **adding a stage for
your feature is part of building the feature.**

**Render-thread CPU stages** (wall time per frame, `Instant` spans):

| Stage | Covers (current code) |
|---|---|
| `inbound.cmds` | `poll_inbound` command drain (`rpc::handle` — scene loads land here) |
| `inbound.watch` | watcher poll + `reload_scene` / `reload_effects` |
| `inbound.probe` | `step_probe` (§5.6 pre-flight probe; budgeted ≤ 3.5 ms) |
| `inbound.misc` | design autosave, audio pill |
| `telemetry` | `emit_periodic_telemetry` + `FpsAccumulator` |
| `tick.live` / `tick.design` | `PassPlan::tick` per leg (driver eval → uniforms) |
| `encode` | warp-bake encode, uploads, composite + final encode |
| `acquire` | `surface.get_current_texture()` — **backpressure, not work**; kept separate so vsync waits never read as cost |
| `submit` / `present` | `queue.submit`, `frame.present` |
| `preview.present` | native preview blit (`present_preview`) |
| `preview.jpeg` | demand-gated JPEG readback (`capture_preview_jpeg`) |
| `pace` | `pace_frame` sleep |

Derived: **`cpu.work` = Σ stages − `acquire` − `pace`**; **headroom =
1 − max(cpu.work, gpu.total) / frame budget** (16.7 ms at 60 Hz). Headroom is
the headline number, not fps.

**GPU passes** (timestamp queries, §2.3): `gpu.warp_bake`, `gpu.uploads`,
`gpu.composite.live`, `gpu.composite.design`, `gpu.final`, `gpu.preview`,
`gpu.probe`; `gpu.total` = first-begin → last-end of the frame.

**Side threads:** per clip stream (`clip-decode:<label>`): decode ms/frame
p50/p95, **stale frames** (render thread wanted frame N, slot not ready →
showed older frame), queue depth. Stale ≠ repeated: a 30 fps clip on a
60 Hz engine *should* show each frame twice (Resolume's content-rate vs
composition-rate distinction) — only "wanted frame not ready" counts. This
is TouchDesigner's `mvreadahead` (movie read-ahead failures). Per thread (render, each decoder, WS
connections, OSC, telemetry fan-in): CPU % of one core.

**Latency stamps:** `knob→submit` (`param.set`/`master.set` write → first
submit that read it; seq number in `SliderBank`/`Masters`), `rpc.queue`
(`EngineCommand` enqueue → handled), `osc→tick` (age of the newest audio
packet at tick time, from `AudioFeatures::last_packet_ms`).

**Dropped frames:** presented frames whose interval exceeds 1.5× the
budget (a missed vsync) — a count, like TouchDesigner's `droppedframes`,
because p95 over 2.5 s hides a single hitch and a single hitch is what the
audience sees.

**Process:** CPU % (`getrusage` delta), RSS, **GPU memory MB** (own
accounting: every texture/buffer the engine creates goes through a counter
with its byte size — portable, unlike Metal's `currentAllocatedSize`),
wgpu live-resource counts per type (`Instance::generate_report()` — buffers, textures, bind groups,
pipelines; the leak signal), telemetry bytes/s per channel (counted on the
bus, and in the Tauri `telemetry_loop` for the webview hop).

### 2.2 Tier A — `render-core/src/perf.rs`

- **Recorder:** `[u32; Stage::COUNT]` of µs per frame, written by a
  `perf.span(Stage::X)` guard; no allocation, no locks, no strings. A ring
  of the last ~300 frames; aggregate at 2 Hz → p50/p95/max per stage.
- **Thread CPU:** each owned thread samples its own
  `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` once per second and publishes into
  an atomic slot. POSIX → identical on macOS and Linux (keeps the
  portability rule; `#[cfg(unix)]`, null elsewhere).
- **Channel `perf`** (2 Hz, not sticky): `{ budget_ms, headroom, cpu:{stage:
  {p50,p95,max}}, gpu:{pass:{p50,p95}} | null, threads:[{name,cpu_pct}],
  clips:[{label,decode_p95,stale}], latency:{knob_submit_p95, rpc_queue_p95,
  osc_age_p50}, process:{cpu_pct, rss_mb, resources:{…}}, telemetry_bps:{…},
  last_change:{epoch,rev,actor,facet} }`. Inline RPC **`perf.snapshot`**
  returns the same object (agents, MCP, scripts). Keep `fps`/`frame_stats`
  untouched — the operator pill's honesty contract (§3.2) stands.
- **Change-correlated alerts.** The engine already stamps every design
  mutation with a rev (§2.7 `ChangeLog`), and promote/clip-open are known
  events. Keep a rolling 30 s per-stage baseline; when a stage's p95 jumps
  > 25 % and > 0.5 ms within 2 s after an event, emit `perf` with
  `alert: {stage, before, after, cause: <change entry | "promote" | …>}` and
  log it. This is the runtime answer to "which change just added strain".
- **Spike capture** (TouchDesigner's Performance Monitor "Frame Trigger"):
  the recorder already holds the last ~300 frames, so when a frame exceeds
  `trigger_ms` (default 2× budget) freeze the breakdown of that frame ± 5
  neighbours, plus what ran in it (commands handled, watcher reloads, probe
  steps, clip opens, change rev), and emit it as a sticky `perf_spike`
  payload + one log line. Aggregates find *steady* load; this finds the
  one-off hitch. `perf.setTrigger {ms}` adjusts it.
- **Switch:** `--perf off|on` (default `on`); the bench measures the
  monitor's own overhead (on vs off) and gates it at ≤ 0.1 ms.

### 2.3 GPU timing

- `wgpu` 22 on Metal exposes `TIMESTAMP_QUERY` +
  `TIMESTAMP_QUERY_INSIDE_ENCODERS` when the GPU supports **stage-boundary**
  counter sampling — Apple Silicon (M2) does; **draw-boundary sampling it
  does not** (verified in `wgpu-hal-22/src/metal/adapter.rs`). So: per-pass
  timing yes, per-draw no. Vulkan/NVIDIA has both.
- Request the two features **only if the adapter offers them**, exactly like
  the BC exception in `gpu.rs::GpuContext::new`; absent → GPU columns are
  `null`, everything else works. Document as the second capability-gated
  exception to `Features::empty()` in CLAUDE.md + reference §1b.
- Mechanics: one `QuerySet` (2 × passes), `timestamp_writes` on each render
  pass, `encoder.write_timestamp` around the upload copies, resolve into a
  ring of 3 `MAP_READ` buffers mapped **non-blocking** (the `PreviewSampler`
  pattern — never `poll(Wait)` on the render thread); results arrive 2–3
  frames late, keyed by frame index. Convert with
  `queue.get_timestamp_period()`.
- Caveat for readers of the numbers: on a tile-based GPU, passes overlap;
  per-pass times are attribution, `gpu.total` is the budget truth.

### 2.4 Tier B — `perf.profile {frames?, leg?}` (per-binding cost)

The composite is **one render pass with one draw per binding**
(`compositor.rs::encode_composite`), and the M2 can't time draws. So the
profile mode, for ~120 frames, **splits the composite into one pass per
binding** (`LoadOp::Load`, same additive blend → pixel-identical output) and
puts timestamps on each, plus CPU `tick` cost per binding. Reply: bindings
ranked by GPU ms with `share_of_composite`, effect name, selector, mask
coverage (px), and any clip it reads. Split passes cost extra tile
load/store on Apple GPUs, so absolute numbers are inflated — the reply
reports the scale factor (Σ split / un-split composite) and the ranking is
what matters. Runs on design by default; on live only with `leg:"live"`.

**When bindings become passes, this goes always-on.** Roadmap §5.15
(effect chains) and §5.16 (feedback + `post`) give some bindings their own
offscreen pass; each such pass gets its own timestamp pair in Tier A, so
per-binding GPU cost becomes free for exactly the bindings that are
expensive. That is how TouchDesigner shows a GPU time per TOP: every TOP is
its own pass.

UI: **Debug room → Perf panel** — a scrolling frame-time graph (CPU work
and GPU total against the budget line, dropped frames and spikes marked —
TouchDesigner's Perform CHOP → Trail CHOP view), stacked bars (CPU row, GPU row) against
the budget line, headroom, threads table, clip decoders, telemetry
bandwidth, alert log, "Profile bindings" button → ranked table. Perform's
FPS pill takes a headroom tint (green > 40 %, yellow > 15 %, red below).
Emitter + consumer land together (§6 working agreement).

## 3. Tier C — the perf gate agents run

### 3.1 Windowless bench mode

`render-core --bench <scenario.json> --bench-out <report.json>`:

- **No window, no surface**: adapter requested with `compatible_surface:
  None`; the final pass renders into an offscreen texture at the scenario's
  output size (default 1920×1080). Runs from any agent shell, no display,
  no focus/occlusion effects, no App Nap.
- **Uncapped** by default (no `pace`, no vsync) → measures *cost*.
  `--paced` runs at 60 Hz to measure jitter and latency.
- Scenario = `{scene, size, warmup_frames, frames, timeline: [{at_frame,
  rpc:{method, params}}]}`. Timeline actions go through `rpc::dispatch`, so
  a scenario exercises the real promote / `param.set` / `scene.load` paths.
- Report = the Tier-A aggregate over the measured window + machine info
  (adapter name, OS, git sha + dirty flag, build profile).

### 3.2 Initial scenario suite — `render-core/perf/scenarios/`

Each one guards something named; scenes reuse `examples/` and
`tests/fixtures/` (HAP clips) where possible.

| Scenario | Guards |
|---|---|
| `idle` | fixed floor: empty scene, final pass + warp only |
| `smoke` | today's typical scene (`examples/phase3_smoke.scene.json`) |
| `layers-heavy` | 40 bindings × built-ins, audio pinned to 1.0 (worst case, like the probe) |
| `clips-7` | the E1 gate scene: seven HAP clips — decode p95, **stale frames** |
| `take-crossfade` | two-leg, promote every 2 s swapping clips (E1b/E2 trap) |
| `warp-64` | 64-handle alignment + an edit storm (rebake cost) |
| `edit-churn` | 300 × `scene.load` / `effect.upsert` → resource counts **plateau**, rebuild ms |
| `preview-remote` | a `preview` subscriber attached (JPEG readback path) |
| `knob-latency` | paced; `param.set` at 100 Hz → `knob→submit` p95 |
| `capacity-clips` | *informational, never FAIL*: adds 1080p HAP clips one at a time until headroom < 0 → "this machine holds N clips at 60 fps" — the number VJs compare (Resolume's layers × resolution benchmarks) |

A feature that adds a new cost path **adds or extends a scenario in the same
change** (clip effects → extend `clips-7`; feedback/post chain → new
`post-chain`, etc.).

### 3.3 `scripts/perf_gate.py` (run with `uv run`)

1. `cargo build --release` in `render-core/`.
2. Run the suite 3× (≈ 10 s per scenario), take the median per metric.
3. Compare against `render-core/perf/baselines/<machine-key>.json`
   (machine key = adapter name + hostname; committed).
4. Print one table: scenario × metric, baseline → now, Δ, verdict; exit 1 on
   any **FAIL**.

**Verdict rules**

- *Regression* (FAIL): a stage/pass p95 rises > 15 % **and** > 0.3 ms
  absolute (both, so tiny numbers don't flap), or latency p95 rises > 2 ms.
- *Absolute budgets* (FAIL regardless of baseline; at 1080p, M2):
  `gpu.total` p95 ≤ 8 ms (half a 60 Hz frame); `cpu.work` p95 ≤ 4 ms; no
  frame > 33 ms in steady scenarios; **stale clip frames = 0**;
  `knob→submit` p95 ≤ 17 ms; resource counts flat after `edit-churn`; perf
  monitor overhead ≤ 0.1 ms.
- *Noise:* a failing scenario is re-run once before failing; > 10 % spread
  across the 3 runs prints **NOISY** (several agents and `cargo` builds share
  this machine — GPU timestamps are far less sensitive to that than CPU
  wall time, so GPU verdicts are the trustworthy ones).
- No baseline for this machine → absolute budgets only (Linux box later).
- `--update-baseline --reason "…"` rewrites the baseline **and** appends
  `{date, sha, reason, deltas}` to `perf/baselines/history.jsonl`, so a
  deliberate cost increase is a recorded decision, not silent drift.

### 3.4 The agent rule (goes into CLAUDE.md + reference §6 when PM3 lands)

> After any change under `render-core/` or `wzrd-app/src-tauri/` — or a UI
> change that subscribes to a high-rate channel — run `uv run
> scripts/perf_gate.py` and paste its table into your summary. A FAIL is not
> done. A cost you added on purpose: add/extend a scenario that covers it,
> then `--update-baseline --reason`.

Not gated (yet): the webview. The WKWebView runs out of process; its load
reaches us as telemetry bandwidth (`telemetry_bps`, measured) and shared
cores/GPU. A dev-only rAF jank meter in the Debug panel covers it until a
real problem shows up.

## 3.5 Prior art (checked 2026-10-10)

| Tool | What it monitors | Taken into this plan |
|---|---|---|
| **TouchDesigner** | Perform CHOP: `fps`, `msec`, `droppedframes`, `mvreadahead`, `gpumemused`, `activeops`. Per-operator CPU + GPU cook time (Info / middle-click). **Probe** palette: live per-node cost overlay, costs ~3.5 ms itself → turn off when done. **Performance Monitor** (Alt+Y): per-frame hierarchical breakdown, "Frame Trigger (ms)" captures only slow frames; its docs admit it's weak for GPU work because TOP times are CPU submit cost | channel set (§2.1), stale clip frames, dropped frames, GPU MB, spike capture (§2.2), trail graph + on-demand-only deep profile (§2.4), CPU vs GPU kept as separate columns |
| **Resolume Arena** | An FPS readout (staff: "don't use it to track performance"), display info; users fall back to GPU-Z / Activity Monitor and forum benchmarks (layers × resolution → fps) | operator surface stays one number (headroom pill); content-rate vs engine-rate distinction; `capacity-clips` |

Neither has a regression gate (Tier C) — they're end-user tools, not
codebases that agents change daily. That part is ours.

## 4. Build order

| Pkg | Scope | Gate |
|---|---|---|
| **PM1** | Done — see [reference/render-engine.md](../reference/render-engine.md) §2.9. Deviations from §2.1–§2.2 that constrain PM2–PM4: a 15th stage `host` (time between Core calls) makes the stages tile the interval; GPU-MB is a descriptor **inventory**, not a create-site counter (leaks show in the wgpu counts); `telemetry_bps` is counted at the WS + webview hops (`{ws, webview}`), not on the bus; spans are `begin/lap/enter/end` tokens (borrowck rules out RAII guards on `&mut Core`). | gate met: headroom 0.93 on smoke; unaccounted ≤ 0.015 ms p95; overhead ~2 µs/frame |
| **PM2** | GPU timestamps (capability-gated), non-blocking readback | GPU columns populated on the M2, `null` with the feature forced off |
| **PM3** | `--bench` windowless mode, scenario suite, `perf_gate.py`, first committed baseline, the agent rule in CLAUDE.md | gate run is green on main, red on a deliberately slow shader |
| **PM4** | `perf.profile` split-pass mode, Debug-room Perf panel, headroom-tinted pill, change-correlated alerts | slow binding tops the profile table; alert names the rev that introduced it |

**Interaction with the gig plan** ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)):
E1's gates ("seven clips at 60 fps p95") become honest with PM1 + `clips-7`.
**E2 (post-swap probation: auto-revert a live swap that blows the frame
budget) should consume PM1/PM2's per-leg GPU time instead of wall-clock
p95** — that also solves its documented trap (a TAKE crossfade raises total
frame time because both legs render; per-pass time on the swapped leg
doesn't).

## Open questions (for Xander)

- Budgets in §3.3 are proposals — confirm 8 ms GPU / 4 ms CPU at 1080p.
- The `knob→submit ≤ 17 ms` budget fails today: PM1 measures ~31 ms p95
  because legs tick before the vsync wait in `acquire` (reference §4 #9).
  Either move the tick after `acquire` before PM3 lands, or set this budget
  to ~34 ms.
- Ordering: PM1 + PM3 before the remaining gig engine work (E1b–E2), or
  after the gig?
