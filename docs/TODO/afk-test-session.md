# AFK test session — every test that needs the screen, run in one sitting

> **Status: no blocks queued** (2026-10-12: B1–B7 all passed and are
> retired; the perf pass was re-verified on screen at 18:10–18:35 —
> 15 effects at 60 fps, continuity, TAKE fade on the monitor, in-app GPU
> timing, knob→submit p95 16.7 ms, app idle CPU 28.9 → ~18 %). Collects every test that needs the
> engine window **visible, in front, and nobody touching the Mac**: fps gates,
> screen-recorded flash checks, window screenshots. Xander says "I'm AFK",
> one agent runs all blocks below in order, fills in each block's
> **Results**, commits, and reports. Agents with screen-bound tests **append a
> block** (template at the bottom) instead of asking Xander for their own
> session.

## Why these can't run while Xander works

- **macOS throttles a covered window.** The engine's swapchain drops to ~30
  fps when its window is behind another one, so a `perf` frame-interval p95
  reads ~33 ms even for a plain `videoClip` scene (measured 2026-10-12). A
  fully hidden window switches to the offscreen path instead — also not the
  number the projector sees.
- **Recordings are screen captures.** The engine has no recording path and the
  WS `preview` channel (~15 fps) is too slow for flash checks (it misses
  7–8 Hz strobes; see [../reference/effects-library.md](../reference/effects-library.md)),
  so flash gates record the window with ffmpeg avfoundation at 60 fps —
  whatever is on top of the window gets recorded.

## Before Xander leaves (≈ 3 min, Xander still at the Mac)

1. **Power:** plug in the charger (battery can throttle the GPU), Low Power
   Mode off (`pmset -g | grep lowpowermode` → 0).
2. **Quit the WZRD app** and any other GPU-heavy app (they share the GPU and
   skew fps). Check nothing holds the gate ports: `lsof -i :9177 -i :9123`.
3. **Fresh release build** of the engine every block uses:
   `cd render-core && cargo build --release` (other agents edit render-core
   — note `git log -1 --oneline` + `git status --short render-core` in the
   Results so numbers are attributable).
4. **Clean desktop:** leave a normal desktop Space with **no full-screen app**
   on it (engine windows open on the current Space), and hide every other
   app (click the desktop, ⌥⌘H). The terminal running the agent may be hidden
   too — hidden apps keep running.
5. **Smoke check (≈ 1 min):**
   `caffeinate -dims uv run python scripts/fx_gate.py run videoClip --bench projects/fx-bench`
   → `test_results/fx_gate/run.json` must show `videoClip.live.interval_p95`
   ≈ 16.7–17.5 ms, and a frame of `test_results/fx_gate/rec_videoClip.mp4`
   (`ffmpeg -ss 4 -i … -frames:v 1 x.jpg`, then look at it) must show the
   seven clips, not another window. If either fails, fix that before Xander
   leaves — `scripts/winbounds.swift --front` activation is untested.
6. **Accessibility for synthetic input (B2):** the runner's terminal (c11)
   must be allowed in System Settings → Privacy & Security → Accessibility —
   B2 clicks, drags and resizes the app. Check:
   `osascript -e 'tell application "System Events" to get size of window 1 of process "Finder"'`
   prints a size, not an error (Finder needs a window open). Only Xander can
   grant it.
7. Xander leaves. Start **one** `caffeinate -dimsu -t 10800 &` for the whole
   run — wrapping single commands isn't enough: on 2026-10-12 the screen
   locked in a gap between commands and every later recording captured the
   lock screen (nothing can unlock it unattended). Also check System
   Settings → Lock Screen → "Turn display off" / screensaver are long.

## Runbook

Run the blocks in order; total budget is the sum of their durations. Each
block is self-contained: preconditions, command, pass criteria, where results
go, what to do after. If a block fails, record it and **move on** — the
point is one sitting.

| Block | Owner | Duration | What it proves |
|---|---|---|---|

---

## Block template (copy, fill, append above this section, add a runbook row)

```markdown
### Bn — <package>: <what>

**Owner:** <package + link to its plan>. **Duration:** ≈ N min.

**Preconditions:** <bench/project, build, data — must already exist on this Mac>

**Commands** (repo root, wrapped in `caffeinate -dims`):

    <exact commands; no interactive steps; window must come up in front>

**Pass criteria:** <numbers from `perf` / check_frames / a file to look at>

**Results:** _(filled by the runner)_

**After the run:** <commits, doc updates, what to delete from this file>
```

Rules for a block: it must run unattended from the commands alone, finish in
≤ 15 min, never need Xander's input mid-way, and say what "fail" means so the
runner can move on. Prefer `scripts/fx_gate.py`'s helpers (window bounds,
60 fps capture, `perf` summary) over a second harness.
