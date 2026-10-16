# Measuring on this Mac — recipes and traps

Reached from `SKILL.md` step 4 and 5. Every number here was learned the hard
way on 2026-10-12 (M2, on battery, 5 agents building in parallel).

## Judge a change: interleaved A/B, never one gate run

- `perf_gate.py` compares against a baseline taken on a quiet machine. With
  load average ~10 it went red on 10 GPU rows (+15–50 %) for changes that
  cannot touch those passes. A single gate run on a busy machine is noise.
- `scripts/ab.py <out> <rounds> <binA> <binB> <scenario…>` runs the two
  binaries alternately (ABBA) so both eat the same load; read **medians of
  p50s**. 3 rounds separates ≥ 20 % effects; layers-heavy drifts ±10 % on the
  *same* binary, so small effects there need 5–6 rounds and an on/off switch
  inside one binary (an env var) to isolate a single change.
- Keep the before-binary: build unmodified main, `cp target/release/render-core $S/<name>-before`.
  A source tree exported into an old `target/` keeps the old binary
  (timestamps) — `touch` the sources or check the build actually ran.
- Run the full gate and `--update-baseline --reason` only when the machine is
  quiet (`uptime` load < ~3), once, after all merges.
- A p95 FAIL that survives a quiet machine: A/B the *before* binary on that
  scenario first. The baseline stores the best of 3 runs, so it can be an
  outlier (layers-heavy's 1.61 was; the original binary read 2.35). A pass
  whose p50 drops but p95 rises after a whole-pass speedup is usually the
  GPU downclocking on less total work: isolate the one change (build with
  only it reverted) before blaming it (warp_bake 2026-10-12: −10 % on both).

## Bench report fields worth reading

`perf.gpu.passes.composite.live.p50` (the dominant GPU cost),
`perf.gpu.total`, `perf.frame.work.p95` (CPU), `perf.clips.{decodes,
decode_ms, stale}`, `perf.latency.knob_submit.p95`, `teardown.stuck_threads`.
Paced scenarios (`knob-latency`, `take-crossfade`, `preview-clips`) run at a
virtual 60 Hz, where the Apple GPU downclocks: their GPU ms read ~3× the
uncapped ones and are the closer stand-in for a show.

## On-screen numbers

- **fps gates** need the engine window uncovered: a covered window reads
  27–33 ms. Since 2026-10-15 the output window starts **hidden** (the TopBar
  OUTPUT pill turns it on) — launch with `WZRD_OUTPUT=window` (or
  `display:<idx>`) or an fps gate measures offscreen frames. The window then
  centres on the operator's display — move or shrink one so they don't overlap.
- **Windowed GPU timing** is trustworthy since P6 (verified 2026-10-12:
  app `gpu.total` p95 7.4 ms at 60 fps, was 29.8). Paced in-app passes still
  read ~3× the uncapped bench (GPU downclock), so costs come from the bench.
- **App idle cost:** `scripts/appcpu.sh <label> perform live` — per-process
  CPU of `wzrd-app`, its WebKit processes and WindowServer. WebContent is the
  number UI changes move. The `wzrd-app` process includes the engine (dev
  build: render-core at opt-level 1). **Only compare back-to-back runs on
  the same machine state**: under load 11 the same build read WebContent
  3.8 %, quiet it reads 6.7 %. To measure an old build, `git worktree add
  --detach <dir> <sha>`, clone `src-tauri/target` + `node_modules` with
  `cp -Rc`, symlink `projects/`, and run a copy of `appcpu.sh` whose `cd`
  points at the worktree. 2026-10-12 quiet, Perform/LIVE, engine window in
  front: 366aa13 28.9 % / 13.0 % → main ~18 % / ~6.7 % (WindowServer
  27 → 31.5 %, open in perf-monitor.md).
- **Frame spikes during an action:** start `scripts/spikes.py 12 &` first,
  then act; ≥ 500 ms is a fail, ~35 ms on minimize/room switch is normal.
- **Screen recordings:** `ffmpeg -f avfoundation -framerate 60 -capture_cursor 0 -i "1:none" -t 4 -vf "scale=1440:-2,format=yuv420p" -r 60 -c:v h264_videotoolbox …`
  Record the whole screen and crop afterwards: then one recording compares
  the monitor slot and the engine output frame by frame. Brightness curve of
  a crop: `-vf "crop=w:h:x:y,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=y.txt"`.
  A tile of frames (`select='not(mod(n\,8))',tile=6x2`) is the fastest way to
  *see* whether something faded or cut.

## Driving the app unattended

- Helpers: `source scripts/appenv.sh` (here) gives `js`, `room`, `rpc`,
  `snap`, `knobs`, `sidebyside`, `shot`; the repo's `scripts/cgmouse.swift`
  clicks/drags. Hover needs a real `mouseMoved` CGEvent (cgmouse has none:
  a 5-line Swift `CGEvent(mouseEventSource:mouseType:.mouseMoved…)`), and
  only after a click on empty chrome made the window key — the cue
  thumbnail's label turning "live peek" is the check that it landed.
- Counting UI refreshes from a recording: per-frame `signalstats` YDIF on a
  crop, frames with YDIF > 0.3 (YAVG equality counts H.264 noise).
- The first synthetic click after activating the app only makes the main
  window key (the engine output window had focus) — click empty chrome
  first, then verify clicks land with a capture-phase `pointerdown` probe.
- Window min size is ~1100×720 pt; the screen is 1440×900 pt.
- `preview.setSource` changes the operator's deck posture (LIVE vs DESIGN);
  set it back, and restore `projects/fixture-a/session.json` afterwards.
- A raw `param.set` over WS isn't seen by the webview's override table until
  the next hydrate — drive mixer state through the UI when the UI's reaction
  is what you're testing.
