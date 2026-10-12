# AFK test session — every test that needs the screen, run in one sitting

> **Status: open — first run 2026-10-12** (B1, B3, B4, B5 passed and are
> retired; B2 has one step left; B6 added for the perf pass). Collects every test that needs the
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
| B2 | native preview underlay | ≈ 5 min | the PROGRAM monitor shows a TAKE crossfade in step with the projector (step 8; the rest passed 2026-10-12) |
| B6 | perf pass 2026-10-12 | ≈ 20 min | fx + continuity gates, app GPU timing and idle CPU, and the UI animation changes on the engine after the perf pass |

---

### B2 — native preview underlay: interaction + window robustness

**Owner:** [native-preview-underlay.md](native-preview-underlay.md) (the
"Manual checks" section; landed 2026-10-12). How it works:
[../reference/render-engine.md](../reference/render-engine.md) §1b
"Native preview — the underlay" and
[../reference/operator-ui.md](../reference/operator-ui.md) (the
`<NativeSlot>` traps + "Driving the Tauri webview itself"). In one line: the
engine's preview is an `NSView` *under* a transparent webview, so the risks
are clicks not reaching the page, HTML not drawing on top, and the image
not tracking its slot when the window changes. Agents already verified
placement, all three rooms, the warp, snapshots, engine restart and the
`WZRD_PREVIEW=window` fallback from screenshots — this block covers what a
still can't.

**Duration:** ≈ 12 min. **Preconditions:** pre-leave step 6 (Accessibility)
done; `projects/fixture-a` exists (`uv run python wzrd-app/scripts/fixture_projects.py`
recreates it); nothing else on ports 1420 / 9123 / 9124.

**Setup** (repo root; everything after this runs inside one `caffeinate -dims` shell):

```bash
B=test_results/underlay && mkdir -p $B
# fixture-a state this block changes — restored in "After the run"
cp projects/fixture-a/alignment.json projects/fixture-a/session.json $B/ 2>/dev/null
cp projects/fixture-a/.wzrd/design.scene.json $B/ 2>/dev/null || true
swiftc -O scripts/cgmouse.swift -o $B/cgmouse && swiftc -O scripts/winbounds.swift -o $B/winbounds
(cd wzrd-app && WZRD_SCENE=../projects/fixture-a/scene.json pnpm tauri dev > ../$B/app.log 2>&1 &)
until grep -q "first frame presented" $B/app.log; do sleep 1; done; sleep 3
grep -q "preview underlay installed" $B/app.log || echo "FAIL: not in underlay mode"
PID=$(pgrep -f target/debug/wzrd-app)
# js '<expr>' → evaluates in the app webview, prints the result (devport + app log)
js() { local t=r$RANDOM; curl -s localhost:9124/cmd/webview_eval -d "$(python3 -c 'import json,sys;print(json.dumps({"js":"(async()=>{let r;try{r=await ("+sys.argv[1]+")}catch(e){r=\"ERR \"+e};await fetch(\"/__shell/cmd/debug_log\",{method:\"POST\",headers:{\"content-type\":\"application/json\"},body:JSON.stringify({msg:\"'$t' \"+(typeof r===\"string\"?r:JSON.stringify(r))})})})()"}))' "$1")" >/dev/null; sleep 1.5; grep -a "\[webview\] $t" $B/app.log | sed "s/.*$t //"; }
room() { js "(localStorage.setItem('wzrd.route','$1'),location.reload(),'ok')" >/dev/null; sleep 3; }
rpc() { uv run -q --with websockets python -c 'import asyncio,json,sys,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    await w.send(json.dumps({"jsonrpc":"2.0","id":1,"method":sys.argv[1],"params":json.loads(sys.argv[2])}))
    while True:
      r=json.loads(await w.recv())
      if r.get("id")==1: print(json.dumps(r.get("result",r.get("error")))[:400]); return
asyncio.run(m())' "$1" "${2:-{\}}"; }
shot() { screencapture -x -o -l "$($B/winbounds $PID | awk '$4>900{print $1;exit}')" $B/$1.png; }
```

Screen point of an element = main window origin from `$B/winbounds $PID
--front` (the row ≥ 900 px wide; x, y in points) + its `getBoundingClientRect()`
centre, with y offset by the title bar: `y_screen = win_y + (win_h −
innerHeight) + rect_y`. Get rects and `innerHeight` with `js`. Activate the
app once first (`$B/winbounds $PID --front`) so clicks land on it.

**Steps.** Steps 1–7 and 9 passed on 2026-10-12 (region pick, dropdown
over the image, click over the monitor, resize, fullscreen, minimize /
restore, 20 fast room switches, Align handle drag) and are retired. Left:

8. **TAKE crossfade on the monitor.** `room perform`, `rpc preview.setSource '{"source":"live"}'`.
   Make design differ: `rpc param.set '{"binding":"facade_silk","param":"@mute","value":1,"leg":"design"}'`.
   Keep the engine output window **uncovered** (a covered engine window
   drops to ~28 fps): shrink/move the windows so the monitor slot and the
   engine output don't overlap. Record the **whole screen** (avfoundation
   `-i "1:none"`, `scale=1440:-2`, 4 s), and 1 s in
   `rpc promote '{"fade_ms":1500,"quantize":"now"}'`. Crop the main-facade
   area of both the monitor and the engine output and run `signalstats`.
   Pass: the monitor's brightness ramps over ~90 frames **in step with the
   engine output**, not one jump. 2026-10-12 result: FAIL — the engine
   output faded correctly but the monitor held the old look for the whole
   fade and jumped at its end (Perform's LIVE source never applied the fade
   mix). Fixed the same day, untested on screen: the LIVE preview now pairs
   live with design as its mix partner (`gpu.rs` `attach_preview`).
10. **Second display** (only if `system_profiler SPDisplaysDataType | grep -c Resolution:` ≥ 2):
    move the window there, `shot display2`. Pass: image sharp and in its
    slot. 2026-10-12: skipped (one display).

Traps learned on 2026-10-12: the first synthetic click after activating the
app only makes the main window key (the engine output window had focus) —
click once on empty chrome first; `alignment.json` may not exist in
fixture-a (copy with `2>/dev/null`, and delete the engine-written one when
restoring); `perf_spike` needs `telemetry.subscribe` (method name), not
`subscribe`.

**Not covered here — needs Xander's eyes (≈ 1 min next time he's at the
Mac):** Prepare's BOTH backdrop vs. the old look; the knob is
`BACKDROP_PARAMS.both.photo_weight` in `wzrd-app/src/components/SurfaceMap.tsx`
(0.15 now). And the gig-length soak (U7) is a rehearsal, not a block.

**After the run:** quit the app and Vite, restore fixture-a, commit the
result here. Step 8 passes → delete the "Manual checks" section of
[native-preview-underlay.md](native-preview-underlay.md) (keep the BOTH
tuning line) and delete B2 from this file.

---

### B6 — perf pass 2026-10-12: re-verify on screen

**Owner:** the 2026-10-12 perf pass (no plan file; current state in
[../reference/render-engine.md](../reference/render-engine.md) §2.4 "Output
bounds", §2.6 design demand, §2.9–§2.10, and
[../reference/operator-ui.md](../reference/operator-ui.md) telemetry demand).
It landed after the screen locked, so nothing of it has been watched on a
real window: region scissor on every layer pass, a per-plan uniform buffer,
the design leg sampled for the JPEG stream, parallel clip opens + one decode
per clip frame, batched demand-gated app telemetry, a sampled cue monitor,
opacity-only glow/lamp animations, and (if P6 landed) the tick moved after
`acquire` + isolated GPU timing on the window path. **Duration:** ≈ 20 min.

**Preconditions:** release build of current main; `projects/fx-bench`;
`projects/fixture-a/perf-clips.scene.json`; nothing on 1420/9123/9124/9177.

**Commands / checks** (repo root, inside `caffeinate -dims`):

1. `uv run python scripts/fx_gate.py run --bench projects/fx-bench` — all 15
   effects: live p95 ≤ 17.5 ms, flash pass, and look at
   `test_results/fx_gate/rec_<fx>.mp4` frames for anything cut off at a
   region edge (the scissor clips output to each region's mask box; bit
   identity is unit-tested, this is the eyes-on check). edgeGlow's
   live+design `headroom_min` was 0.35 before — expect higher.
2. `uv run python scripts/continuity_gate.py --bench projects/fx-bench` —
   status pass (clip opens moved to worker threads, first frame now arrives
   via the staging ring).
3. Re-run the former B3 snapshot (git history of this file, commit
   4c85335^): in the app, `passes.preview` should now read ≲ 0.3 ms and
   `gpu.total` p95 below the 16.7 ms budget at 60 fps (fails today: 29.8).
4. App idle cost in the show posture (Perform, LIVE, engine window in
   front): per-process CPU via `top -l 7 -s 5 -pid <wzrd-app> -pid
   <WebContent…> -pid <WindowServer>`. 2026-10-12 on a loaded machine:
   before wzrd-app 18.9 % / WebContent 8.7 %, after P4 15.2 % / 3.8 %.
   Measure both again on a quiet machine.
5. Screenshots to eyeball (Xander or the runner): CUED·DESIGN thumbnail
   refreshes (~1 Hz, and at full rate while hovered), the armed TAKE glow
   pulses, the beat lamp and Lamp primitive look as before, the TopBar at a
   1100-pt-wide window doesn't overlap.

**Results:** _(filled by the runner)_

**After the run:** all pass → delete B6 and its runbook row. A fail →
describe it in render-engine.md §4 and leave the failing check here.

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
