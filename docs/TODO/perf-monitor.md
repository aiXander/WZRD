# Performance monitor + perf gate (plan, 2026-10-10)

**Goal.** Every frame's cost is attributable to a named part of the stack
(render-thread stage, GPU pass, binding, side thread, IPC), visible live in
the app, and checked by a scripted **perf gate** every coding agent runs
after a change — so "this feature added 4 ms of GPU" is caught the day it
lands, not on show night.

Status: **PM1 (CPU frame profiler) shipped 2026-10-10, PM2 (sampled per-pass
GPU timestamps) and PM3 (`--bench` + `scripts/perf_gate.py` regression gate)
shipped 2026-10-12** — current state in
[reference/render-engine.md](../reference/render-engine.md) §2.9–§2.10. PM4
(per-binding profile, graphed panel, alerts) is still design. Build order at
the bottom; open questions at the end.
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
  readback. (GPU time per pass: PM2, shipped.)
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

**GPU passes** (timestamp queries, §2.3; `perf::GpuPass`): `warp_bake`,
`composite.live`, `composite.design`, `final`, `preview`, `probe`;
`total` = Σ per-submit first-begin → last-end. `uploads` is deliberately
untimed (reference §2.9 says why).

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

### 2.3 GPU timing — done (PM2)

Shipped 2026-10-12: `render-core/src/gpu_timing.rs`, see
[reference/render-engine.md](../reference/render-engine.md) §2.9. What it
found that constrains PM3/PM4:

- **A timed frame costs ~73 µs of render-thread CPU** (wgpu-core encodes a
  query reset before every pass with `timestamp_writes`; ~10 µs per extra
  timed pass), so Tier A samples 1 frame in 30. PM4's split-pass profile
  pays ~10 µs CPU per split pass on top of the tile load/store — fine for
  an on-demand mode, but report it.
- **Apple GPUs downclock at 60 fps**: the smoke scene's GPU total reads
  ~7.7 ms at 60 Hz and ~3.7 ms uncapped. PM3's uncapped `--bench` is the
  right place for GPU budgets; don't gate on 60 Hz GPU ms.
- **wgpu-hal 22 Metal resolve bug** (`NSRange` end-as-length) is worked
  around with a per-slot query set resolved from index 0 — any new resolve
  site (PM4's per-binding passes) must do the same until wgpu is upgraded.
- Draw-boundary timing doesn't exist on the M2 (stage-boundary only), so
  per-binding cost still needs PM4's split passes.

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

## 3. Tier C — the perf gate agents run (PM3, done)

Shipped 2026-10-12: `render-core --bench` (windowless, offscreen final
pass), ten scenarios in `render-core/perf/scenarios/`, `scripts/perf_gate.py`
(3 runs, baseline compare, one table, exit 1 on FAIL), the first M2 baseline
in `render-core/perf/baselines/`, and the agent rule in `CLAUDE.md` + reference
§6. Deviations from the original design that constrain PM4: the bench times
1 frame in 4 with each timed frame **alone on the GPU** (two frames in flight
inflate per-submit spans on Apple GPUs); costs take the **best** of 3 runs,
latency/counts the median; regression floors are 0.5 ms GPU / 1 ms CPU (not
0.3); "no frame > 33 ms" judges render-thread work, not the interval;
`edit-churn` runs single-leg (the two-leg probe at uncapped rates refuses
heavy scenes — the probe path is covered by `take-crossfade`). Done — see
[reference/render-engine.md](../reference/render-engine.md) §2.10.

Open from PM3:
- **Not committed yet (2026-10-12).** PM3's edits to `core.rs`, `gpu.rs`,
  `perf.rs`, `lib.rs`, `main.rs` sit on top of PM2's uncommitted diff in the
  same files (and `gpu_timing.rs` is still untracked), so PM3 can't be
  committed alone without sweeping PM2 in or leaving a commit that doesn't
  build. Land PM2 + PM3 together (or after PM2's commit) — ask Xander
  first. PM3's own new files: `render-core/src/bench.rs`, `render-core/perf/`
  (scenarios, scenes, `baselines/`, `.gitignore`), `scripts/perf_gate.py`;
  plus its doc edits (reference §2.10/§4/§6, this file, `CLAUDE.md`,
  afk-test-session B5).
- **Window path not watched on screen** since `redraw` moved to
  `acquire_output`/`present`, and the bench's virtual-vsync knob latency
  isn't cross-checked against a real swapchain yet — both are
  [afk-test-session.md](afk-test-session.md) block B5 (B3 covers the Tauri
  side).
- **`take-crossfade` stale frames are gated regression-only** (~28 stale
  clip frames over 4 TAKEs today — reference §4 #12). When E1b/E2 fix the
  TAKE trap, set its `gate.stale` to `"zero"` and re-baseline.
- **Not gated (yet): the webview.** The WKWebView runs out of process; its
  load reaches us as telemetry bandwidth (`telemetry_bps`) and shared
  cores/GPU. A dev-only rAF jank meter in the Debug panel covers it until a
  real problem shows up.
- **`capacity-clips` reuses 4 clip files**, so it measures decode + GPU,
  mostly from the page cache — a show with N distinct clips also pays disk
  I/O. It hits the search cap (≥ 64) on the M2; raise `gate.capacity.max`
  or use distinct files if the number starts to matter.
- **No Linux baseline yet**: the gate runs budgets-only on a machine without
  one; commit one with `--update-baseline --reason` the first time it runs
  on the Linux box.

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
| **PM2** | Done 2026-10-12 — see reference §2.9. Deviations: uploads untimed (Metal would add blit encoders); sampled 1-in-30 (`--gpu-timing`, `WZRD_GPU_TIMING`); `total` sums per-submit spans; main-submit passes charged from the previous pass's end (TBDR overlap); E2 accessor `Perf::gpu_pass_recent`. | gate met: columns populated on the M2 (smoke `gpu.total` p50 7.7 ms @ 60 Hz); `null` with `--gpu-timing 0` / feature absent (unit tests); on vs off `cpu.work` p95 within noise |
| **PM3** | Done 2026-10-12 — see reference §2.10 and §3 above for the deviations. | gate met: green on main (two consecutive full runs, ~100 s each); red on a 24-octave `silk` (`gpu.total` p95 2.46 → 8.45 ms, FAIL on budget + regression) |
| **PM4** | `perf.profile` split-pass mode, Debug-room Perf panel, headroom-tinted pill, change-correlated alerts | slow binding tops the profile table; alert names the rev that introduced it |

**Interaction with the gig plan** ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)):
E1's gates ("seven clips at 60 fps p95") become honest with PM1 + `clips-7`.
**E2 (post-swap probation: auto-revert a live swap that blows the frame
budget) should consume PM1/PM2's per-leg GPU time instead of wall-clock
p95** — `Perf::gpu_pass_recent(GpuPass::CompositeLive, n)` (at the default
1-in-30 sampling, ~2 samples/s: size the probation window for it, or run
`--gpu-timing` lower during probation) — that also solves its documented trap (a TAKE crossfade raises total
frame time because both legs render; per-pass time on the swapped leg
doesn't).

## Open questions (for Xander)

- **Confirm the gate budgets** — all named constants at the top of
  `scripts/perf_gate.py`, chosen in PM3:
  - `gpu.total` p95 ≤ **8 ms**, uncapped bench, 1080p output (half a 60 Hz
    frame). M2 today: smoke 2.5, preview-remote 4.8, warp-64 5.9.
  - `cpu.work` p95 ≤ **4 ms**. M2 today: ≤ ~1.6 ms everywhere.
  - Worst single frame's render-thread work ≤ **33 ms** in steady scenarios.
  - Stale clip frames = **0** in `clips-7` (`take-crossfade`: regression
    only, see §3).
  - `knob→submit` p95 ≤ **36 ms** — not the planned 17 or the suggested 34:
    the bench measures 33.3–34.7 ms with a 100 Hz knob (ceiling 2 × 16.7 ms
    because legs tick before the vsync wait, reference §4 #9), so 34 failed
    about half the runs. Moving the tick after `acquire` is the real fix.
  - wgpu objects gained over `edit-churn` = **0**.
  - Perf monitor overhead p95 ≤ **0.1 ms**.
  - Regression: > 15 % and > 0.5 ms (GPU) / > 1 ms (CPU) / > 2 ms latency.
- Ordering: PM4 before the remaining gig engine work (E1b–E2), or after the
  gig?
