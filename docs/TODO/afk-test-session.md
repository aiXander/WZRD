# AFK test session — every test that needs the screen, run in one sitting

> **Status: open — nothing run yet.** Collects every test that needs the
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
6. Xander leaves. Wrap the whole run in `caffeinate -dims` so the display
   never sleeps or locks.

## Runbook

Run the blocks in order; total budget is the sum of their durations. Each
block is self-contained: preconditions, command, pass criteria, where results
go, what to do after. If a block fails, record it and **move on** — the
point is one sitting.

| Block | Owner | Duration | What it proves |
|---|---|---|---|
| B1 | gig F1 (clip effects) | ≈ 12 min | 15 library effects: 60 fps × 7 clips, flash rate, preview.jpg |

---

### B1 — F1 clip effects: fps, flash rate, previews

**Owner:** gig package F1 ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)
F1, library in [../reference/effects-library.md](../reference/effects-library.md)).
Gates 1–2 (probe green, dial ranges) already passed 2026-10-12.

**Bench:** `projects/fx-bench/` (gitignored, on this Mac only; its README
says how it was made): the E1 seven-clip layout — one 1376×768 ceiling + six
432×768 fabric strips at a 1376×768 projector — with Hap1 cuts of real
TextureFlow renders. `fx_gate.py` overwrites its `scene.json` and `effects/`.

**Commands** (repo root):

```bash
caffeinate -dims uv run python scripts/fx_gate.py run videoClip --bench projects/fx-bench   # baseline (if not done in the smoke check)
caffeinate -dims uv run python scripts/fx_gate.py run --bench projects/fx-bench             # all 15, ≈ 10 min
caffeinate -dims uv run python scripts/fx_gate.py preview --preview-clip projects/fx-bench/preview.mov   # ≈ 2 min
```

Per effect `run` does: cold boot with the effect on all seven clips → 10 s of
`perf` (live leg only) → 8 s more with the design leg awake (a `preview`
subscription, as in the app) → 8 s 60 fps recording → `check_flash`.

**Pass criteria (per effect):**

- `live.interval_p95` ≤ 17.5 ms (60 fps; the E1 baseline was 17.0–17.2).
  `live+design` is informational — report it, don't fail on it.
- `flash.status == "pass"` (≤ 3 large flashes/s at default dials). The
  baseline `videoClip` recording must pass too, else the clips themselves
  flash and the effect numbers mean nothing.
- `rec.fps` ≈ 60 and `rec.frames` ≈ 480 (a short recording = capture fell
  behind; re-run that effect).
- `library/effects/<name>/preview.jpg` exists and, looked at, shows the
  effect on the temple clip (not black, not another window).

**Results:** _(fill in: commit + render-core state, then one row per effect)_

| Effect | live p95 ms | live+design p95 ms | flashes/s | preview ok |
|---|---|---|---|---|
| videoClip (baseline) | | | | — |

**After the run:**

1. Copy the previews into the fixtures:
   `for p in fixture-a fixture-b; do cp -R library/effects/ projects/$p/effects/; done`.
2. Commit `library/effects/*/preview.jpg` + this file's Results by path.
3. If every row passes: in the gig plan mark F1 done (drain its section to a
   few lines ending in a pointer to effects-library.md, update the status
   line) and delete B1 from this file. A failing effect: fix it (shader cost
   or flash behaviour), re-run that effect only (`fx_gate.py run <name>`) in
   the next AFK session — leave B1 here with only the failing rows.

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
