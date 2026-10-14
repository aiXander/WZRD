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
| B8 | operator-UI sprint 2026-10-14 | ≈ 10 min | the native preview follows resized/swapped panels; Perform layer × deletes |

### B8 — operator-UI sprint (2026-10-14): panel layout × native preview, Perform delete

**Owner:** the 2026-10-14 operator-UI feedback sprint — resizable/swappable
panels and Perform mixer delete ([operator-ui.md](../reference/operator-ui.md)
"Panel layout", "Layer delete"). Both were verified in the browser seat
only; the native underlay is invisible there. (Wheel-on-fader removal needs
no screen: `ui/Fader.tsx` has no wheel listener.) **Duration:** ≈ 10 min.

**Preconditions:** `projects/fixture-a` exists (else
`uv run python wzrd-app/scripts/fixture_projects.py`). Work on a copy so
the fixture's scene and layout stay clean.

**Commands** (repo root, wrapped in `caffeinate -dims`):

    rm -rf projects/b8 && cp -R projects/fixture-a projects/b8 && rm -f projects/b8/ui_layout.json
    export B=test_results/afk-b8 && mkdir -p $B
    (cd wzrd-app && WZRD_SCENE=../projects/b8/scene.json pnpm tauri dev > ../$B/app.log 2>&1 &)
    until grep -q "preview underlay installed" $B/app.log; do sleep 1; done; sleep 5
    source .claude/skills/afk-session/scripts/appenv.sh
    sidebyside
    # 1. Perform: unlock, drag the side column 200 px left, then swap stage/side.
    room perform
    js "import('/src/ui/layoutStore.ts').then(m=>(m.setLayoutLocked(false),'ok'))"
    shot b8-perform-before
    js "(()=>{const e=[...document.querySelectorAll('[class*=cursor-col-resize]')].pop();const r=e.getBoundingClientRect(),o={bubbles:true,pointerId:1,isPrimary:true,buttons:1,clientY:r.top+200};e.dispatchEvent(new PointerEvent('pointerdown',{...o,clientX:r.left+4}));for(const d of [50,100,150,200]){window.dispatchEvent(new PointerEvent('pointermove',{...o,clientX:r.left+4-d}))}window.dispatchEvent(new PointerEvent('pointerup',{...o,buttons:0,clientX:r.left-196}));return 'ok'})()"
    sleep 1; shot b8-perform-resized
    js "import('/src/ui/layoutStore.ts').then(m=>(m.setPaneOrder('perform.cols',['side','main']),m.notifyLayout(),'ok'))"
    sleep 1; shot b8-perform-swapped
    # 2. Prepare: resize the left column (the surface map's native slot must follow).
    room prepare
    js "(()=>{const e=[...document.querySelectorAll('[class*=cursor-col-resize]')][0];const r=e.getBoundingClientRect(),o={bubbles:true,pointerId:1,isPrimary:true,buttons:1,clientY:r.top+200};e.dispatchEvent(new PointerEvent('pointerdown',{...o,clientX:r.left+4}));window.dispatchEvent(new PointerEvent('pointermove',{...o,clientX:r.left+124}));window.dispatchEvent(new PointerEvent('pointerup',{...o,buttons:0,clientX:r.left+124}));return 'ok'})()"
    sleep 1; shot b8-prepare-resized
    # 3. Perform delete (DESIGN): rows before / after clicking the first row's ×.
    room perform
    js "document.querySelectorAll('[aria-label^=\"Delete \"]').length" > $B/rows-before.txt
    js "(document.querySelector('[aria-label^=\"Delete \"]').click(), 'ok')"; sleep 2
    js "document.querySelectorAll('[aria-label^=\"Delete \"]').length" > $B/rows-after.txt
    cat projects/b8/ui_layout.json > $B/ui_layout.json

**Pass criteria:** in `b8-perform-resized`, `-swapped` and
`b8-prepare-resized` the live picture fills the monitor / surface-map frame
exactly: no offset, no stale strip, no picture leaking under another panel.
`ui_layout.json` has `perform.cols/side` ≈ 516 and `order.perform.cols =
["side","main"]`. `rows-after.txt` is one less than `rows-before.txt`
(the deck is on DESIGN after load). If the `cursor-col-resize` selector
misses, record "selector drift" and move on.

**Results:** _(filled by the runner)_

**After the run:** delete `projects/b8`. Pass → delete this block and its
runbook row. A misplaced native picture is a `notifyLayout()` gap — see the
operator-ui.md trap on NativeSlot moves.

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
