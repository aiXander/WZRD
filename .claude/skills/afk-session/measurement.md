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
  quiet (`uptime` load < ~3, on the charger), once, after all merges.

## Bench report fields worth reading

`perf.gpu.passes.composite.live.p50` (the dominant GPU cost),
`perf.gpu.total`, `perf.frame.work.p95` (CPU), `perf.clips.{decodes,
decode_ms, stale}`, `perf.latency.knob_submit.p95`, `teardown.stuck_threads`.
Paced scenarios (`knob-latency`, `take-crossfade`, `preview-clips`) run at a
virtual 60 Hz, where the Apple GPU downclocks: their GPU ms read ~3× the
uncapped ones and are the closer stand-in for a show.

## On-screen numbers

- **fps gates** need the engine window uncovered: a covered window reads
  27–33 ms. In the Tauri app the engine output window sits *under* the main
  window by default — move or shrink one so they don't overlap.
- **Windowed GPU timing** over-read on 2026-10-12 (app `gpu.total` p95
  29.8 ms at a solid 60 fps); timed window frames are now isolated like the
  bench's, unverified on screen until AFK block B7 passes. Until then trust
  bench GPU numbers over in-app ones for costs.
- **App idle cost:** `scripts/appcpu.sh <label> perform live` — per-process
  CPU of `wzrd-app`, its WebKit processes and WindowServer. WebContent is the
  number UI changes move (8.7 % → 3.8 % for batched, demand-gated telemetry).
  The `wzrd-app` process includes the engine (dev build: render-core at
  opt-level 1).
- **Frame spikes during an action:** start `scripts/spikes.py 12 &` first,
  then act; ≥ 500 ms is a fail, ~35 ms on minimize/room switch is normal.
- **Screen recordings:** `ffmpeg -f avfoundation -framerate 60 -capture_cursor 0 -i "1:none" -t 4 -vf "scale=1440:-2,format=yuv420p" -r 60 -c:v h264_videotoolbox …`
  Record the whole screen and crop afterwards: then one recording compares
  the monitor slot and the engine output frame by frame. Brightness curve of
  a crop: `-vf "crop=w:h:x:y,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=y.txt"`.
  A tile of frames (`select='not(mod(n\,8))',tile=6x2`) is the fastest way to
  *see* whether something faded or cut.

## Driving the app unattended

- Helpers (`js`, `room`, `rpc`, `shot`) are in `docs/TODO/afk-test-session.md`
  block B2's setup; `scripts/cgmouse.swift` clicks/drags.
- The first synthetic click after activating the app only makes the main
  window key (the engine output window had focus) — click empty chrome
  first, then verify clicks land with a capture-phase `pointerdown` probe.
- Window min size is ~1100×720 pt; the screen is 1440×900 pt.
- `preview.setSource` changes the operator's deck posture (LIVE vs DESIGN);
  set it back, and restore `projects/fixture-a/session.json` afterwards.
- A raw `param.set` over WS isn't seen by the webview's override table until
  the next hydrate — drive mixer state through the UI when the UI's reaction
  is what you're testing.
