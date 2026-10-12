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
6. **Accessibility for synthetic input (B2):** the runner's terminal (c11)
   must be allowed in System Settings → Privacy & Security → Accessibility —
   B2 clicks, drags and resizes the app. Check:
   `osascript -e 'tell application "System Events" to get size of window 1 of process "Finder"'`
   prints a size, not an error (Finder needs a window open). Only Xander can
   grant it.
7. Xander leaves. Wrap the whole run in `caffeinate -dims` so the display
   never sleeps or locks.

## Runbook

Run the blocks in order; total budget is the sum of their durations. Each
block is self-contained: preconditions, command, pass criteria, where results
go, what to do after. If a block fails, record it and **move on** — the
point is one sitting.

| Block | Owner | Duration | What it proves |
|---|---|---|---|
| B2 | native preview underlay | ≈ 12 min | clicks/dropdowns over the GPU image, resize/fullscreen/minimize/room-switch robustness, TAKE crossfade + Align drag on the image |
| B6 | perf P6 frame ordering | ≈ 5 min | windowed GPU timing no longer over-reads (preview blit, `gpu.total`); the timed-frame drain costs no vsync; knob→submit ≈ 1 frame on a real swapchain |

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
cp projects/fixture-a/alignment.json projects/fixture-a/session.json $B/
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

**Steps** — each one: act, then check; record pass/fail + a one-line note:

1. **Prepare region pick.** `room prepare`. Take a region's centroid from
   `rpc pack.info` (`layers[].centroid`, **pack pixels** — divide by the
   pack's width/height for uv; pick a region whose centroid lies inside its
   own mask, e.g. a facade, not a thin strip), map it into the surface
   canvas rect (`js "JSON.stringify(document.querySelector('canvas').getBoundingClientRect())"`),
   `cgmouse click` there. Pass: `js "(await import('/src/state/store.ts')).useStore.getState().selectedLayerId"`
   returns that region's id. `shot prepare_pick` — the hover label/selection
   draws over the light.
2. **Dropdown over the image.** `room perform`. Click the TopBar project
   menu button (text `fixture-a`). Pass: `shot perform_menu` shows the menu
   panel drawn *over* the monitor image (not cut off behind it). Press Esc /
   click it again to close.
3. **Click over the monitor.** In Perform, click the centre of the monitor.
   Pass: no error in `$B/app.log`, the app stays responsive (`js "1+1"`
   answers), keyboard focus is still the page (`js "document.hasFocus()"` →
   true).
4. **Resize.** `osascript -e 'tell application "System Events" to tell process "wzrd-app" to set size of window "WZRD" to {1200, 760}'`,
   wait 1 s, `shot resize_small`; then `{1700, 1000}`, `shot resize_large`.
   Pass: in both, the image fills exactly the monitor's black-framed slot —
   no offset, no band, no image outside the frame. Repeat once in
   `room align` (image = the dashed projector rect).
5. **Fullscreen.** `...set value of attribute "AXFullScreen" of window "WZRD" to true`,
   wait 2 s, `shot fullscreen`, then back to `false`. Pass as 4.
6. **Minimize / restore.** `...set value of attribute "AXMinimized" of window "WZRD" to true`,
   wait 3 s, then `false` (or `$B/winbounds $PID --front`), wait 2 s.
   Pass: `$B/app.log` shows `native preview paused` then `presenting`; no
   `perf_spike` ≥ 500 ms in that window (`rpc` can't read channels — use the
   Debug room's "last spike", or subscribe to `perf_spike` like
   `fx_gate.py`'s `Rpc.collect`). `shot restored` matches the slot.
7. **Fast room switching.** 20× `osascript -e 'tell application "System Events" to keystroke "1" using command down'`
   cycling ⌘1 / ⌘2 / ⌘3 with 150 ms gaps, ending on ⌘2 (Align). Pass: no
   `perf_spike` ≥ 500 ms, no panic in the log, `shot rooms_end` shows the
   program view inside the projector rect, and `rpc preview.snapshot '{"width":400}'`
   answers with `"leg": "program"`.
8. **TAKE crossfade on the monitor.** `room perform`, `rpc preview.setSource '{"source":"live"}'`.
   Make design differ: mute its first binding —
   `rpc param.set '{"binding":"<first binding id from scene.getState>","param":"@mute","value":1,"leg":"design"}'`.
   Start a 60 fps recording of the monitor rect (`fx_gate.py`'s `record()`
   recipe: avfoundation crop, 3 s) and, 0.5 s in,
   `rpc promote '{"fade_ms":1500,"quantize":"now"}'`. Pass: the mean
   brightness of the muted region's area falls gradually over ~90 frames,
   not in one step (`ffmpeg -i rec.mp4 -vf "crop=…,signalstats" -f null -`
   or sample frames and look).
9. **Align handle drag.** `room align`. Find the TL handle (the first
   `svg circle.cursor-grab`), `cgmouse drag` it 60 px right and 40 px down.
   Pass: `rpc alignment.get` shows `corners[0]` moved by ≈ (60/fit, 40/fit)
   projector px in uv, and `rpc preview.snapshot '{"width":600}'` (decode
   `data_b64`) shows the image warped to match `shot align_drag`. Then
   `rpc alignment.reset`.
10. **Second display** (only if `system_profiler SPDisplaysDataType | grep -c Resolution:` ≥ 2):
    move the window there (`set position of window "WZRD" to {<x on display 2>, 100}`),
    `shot display2`. Pass: image sharp and in its slot (contentsScale follows
    the display). One display → record "skipped".

Also record: `preview.present` p95 from the Debug room's profiler (or a
`perf` subscription) with Perform open.

**Not covered here — needs Xander's eyes (≈ 1 min next time he's at the
Mac):** Prepare's BOTH backdrop vs. the old look; the knob is
`BACKDROP_PARAMS.both.photo_weight` in `wzrd-app/src/components/SurfaceMap.tsx`
(0.15 now). And the gig-length soak (U7) is a rehearsal, not a block.

**Results:** _(fill in: commit, then one row per step)_

| Step | Pass? | Note / screenshot |
|---|---|---|
| 1 Prepare pick | | |
| 2 Dropdown over image | | |
| 3 Click over monitor | | |
| 4 Resize | | |
| 5 Fullscreen | | |
| 6 Minimize / restore | | |
| 7 Room switching | | |
| 8 TAKE crossfade | | |
| 9 Align drag | | |
| 10 Second display | | |
| `preview.present` p95 | — | |

**After the run:**

1. Quit the app (`kill $PID`; the Vite server too), then restore fixture-a:
   `cp $B/alignment.json $B/session.json projects/fixture-a/` and
   `cp $B/design.scene.json projects/fixture-a/.wzrd/` if it was saved.
2. Commit this file's Results by path (screenshots stay in `test_results/`,
   gitignored).
3. All pass: delete the "Manual checks" section of
   [native-preview-underlay.md](native-preview-underlay.md) (keep the BOTH
   tuning line) and delete B2 from this file. A failure: fix it if it's
   small and re-run that step next session; otherwise leave B2 with only the
   failing steps and describe the failure under the plan's "Manual checks".

---

### B6 — perf P6: windowed GPU timing + knob latency on a real swapchain

**Owner:** perf package P6 — frame ordering in `Core::redraw`
([perf-monitor.md](perf-monitor.md) "Verify on screen"; current state in
[render-engine.md](../reference/render-engine.md) §2.9 and §4 #9).
**Duration:** ≈ 5 min.

Three changes only a visible window proves: (1) a GPU-timed window frame
now drains the queue before its main submit (1 frame in 30) — it must not
drop frames; (2) the preview blit is charged from the main span's end —
`preview` should read the blit, not 8 ms; (3) legs tick after `acquire`, so
`knob_submit_p95` should read ≈ 17–19 ms, not ≈ 33.

**Preconditions:** `projects/fixture-a/perf-clips.scene.json` (four
generative bindings + five HAP clips; recreate by appending five
`videoClip` bindings over `generated/tf_a0{1..5}.mov` to fixture-a's
scene); nothing on ports 1420 / 9123 / 9124.

**Commands** (repo root, inside `caffeinate -dims`):

```bash
B=test_results/p6 && mkdir -p $B
cp projects/fixture-a/alignment.json projects/fixture-a/session.json $B/
snap() { uv run -q --with websockets python -c 'import asyncio,json,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    await w.send(json.dumps({"jsonrpc":"2.0","id":1,"method":"perf.snapshot","params":{}}))
    while True:
      r=json.loads(await w.recv())
      if r.get("id")==1:
        s=r["result"]; g=s["gpu"] or {}
        print(json.dumps({"presenting":s["presenting"],"work_p95":s["frame"]["work"]["p95"],"interval_p95":s["frame"]["interval"]["p95"],"dropped":s.get("dropped"),"latency":s.get("latency"),"passes":g.get("passes"),"total":g.get("total")})); return
asyncio.run(m())'; }
# 100 Hz live brightness wiggle (0.95 <-> 1.0) for 10 s: stamps knob_submit
knobs() { uv run -q --with websockets python -c 'import asyncio,json,websockets
async def m():
  async with websockets.connect("ws://127.0.0.1:9123",max_size=None) as w:
    for i in range(1000):
      await w.send(json.dumps({"jsonrpc":"2.0","id":i,"method":"master.set","params":{"name":"brightness","value":0.95+0.05*(i%2),"leg":"live"}}))
      await asyncio.sleep(0.01)
asyncio.run(m())'; }
for MODE in 30 0; do   # default sampling, then GPU timing off
  (cd wzrd-app && WZRD_GPU_TIMING=$MODE WZRD_SCENE=../projects/fixture-a/perf-clips.scene.json pnpm tauri dev > ../$B/app_$MODE.log 2>&1 &)
  until grep -q "preview underlay installed" $B/app_$MODE.log; do sleep 1; done; sleep 25
  osascript -e 'tell application "System Events" to set frontmost of (first process whose name is "wzrd-app") to true'; sleep 5
  knobs; snap | tee $B/snap_$MODE.json
  pkill -f target/debug/wzrd-app; sleep 2; pkill -f vite; sleep 2
done
cp $B/alignment.json $B/session.json projects/fixture-a/
```

**Pass criteria:** in `snap_30.json`: `passes.preview.p50` ≲ 0.5 ms;
`total.p95` < 16.7 ms (headroom positive); `latency.knob_submit_p95`
≤ 22 ms (both snaps). Between `snap_30` and `snap_0`: `interval_p95` within
~0.5 ms and `dropped.window` not higher with timing on (the drain costs no
vsync). Fail = any of these; record the numbers and move on.

**Results:** _(filled by the runner)_

**After the run:** all pass → delete the "Verify on screen" item in
perf-monitor.md and this block. `preview` still high → the over-read isn't
the TBDR begin stamp; record the numbers under that item. Drops with timing
on → the window drain (`GpuContext::drain_before_timed_submit`) costs a
vsync; record it there and in reference §2.9.

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
