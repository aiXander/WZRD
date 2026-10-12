# Native preview underlay — what's left (sprint plan, 2026-10-12)

**Status:** U0–U6 **landed 2026-10-12** (same day as the plan). Open: the
manual checks below, then **U7** (soak → delete the fallback). Current
state lives in [render-engine.md](../reference/render-engine.md) §1b
("Native preview — the underlay") and [operator-ui.md](../reference/operator-ui.md)
(the `<NativeSlot>` / hole-rule traps, the devport agent hooks).

**Decided with Xander (2026-10-12), still binding:** the underlay plays the
Nov 2026 gig if rehearsal is solid; the ≥ 2 h soak is *not* a precondition
for using it on stage. `WZRD_PREVIEW=window` (the old child window above the
webview; Perform only, Prepare/Align on the JPEG) stays as the stage escape
hatch until the soak has run. Don't let it linger past the gig.

---

## Open

### Manual checks (need a human at the window — never run by an agent)

Agents verified placement, rooms, warp, photo blend, snapshots, engine
restart and the fallback via screenshots + `preview.snapshot` (see Shipped).
What a still can't show:

- Clicks over the image reach the webview (Prepare region pick, Align
  handle drag, Perform monitor) and a dropdown opened over the monitor
  renders on top.
- Window resize / fullscreen toggle / fast room switching: image tracks the
  slot, no stale or misplaced frame (trailing 1–2 frames during a live
  resize is expected — webview and native layer commit separately).
- Minimize/restore and moving to a second display (contentsScale follows on
  the next `place`; check it's sharp on a 1× display).
- A TAKE crossfade visible in the Perform monitor; dragging an Align handle
  — the image trails the grid by a few frames (accepted).
- Prepare BOTH: eyeball against memory of the old canvas; tune
  `BACKDROP_PARAMS.both.photo_weight` in `SurfaceMap.tsx` (0.15 now) if the
  photo reads too dim/bright — tune the weight, not the space (linear
  additive is the physics).

### U7 — Soak, delete the old path

- Run a gig-length session (≥ 2 h, audio on, clips playing, room switching,
  a few TAKEs) on the underlay — a rehearsal or the gig itself. Watch
  `perf_spike` and record `preview.present` p95.
- Then delete: `PreviewHost::Window` + `build_preview_window` +
  `PreviewNsRefs` + the child-window `OcclusionProbe` use in
  `wzrd-app/src-tauri/src/engine.rs`, the `focus` re-push in
  `NativeSlot.tsx`, the `window` branch of `api/previewMode.ts`, and the
  `WZRD_PREVIEW` env knob. Keep `preview_jpeg_demand` — Perform's
  CUED·DESIGN cue monitor and the browser build still use the JPEG.
- Doc pass: drop the fallback from `render-engine.md` §1b and
  `operator-ui.md`, drop this plan's row from `CLAUDE.md`, move this file to
  `docs/finished/`.

### Stretch (not scheduled) — Resolume-style PREVIEW + PROGRAM monitors

Two slots in Perform side by side: design leg and program view. The slot
command already carries an id (`"main"`), so this is additive: one
`PreviewTarget` + view per slot. It would also replace the cue monitor's
JPEG (the last in-app JPEG consumer — today it keeps the design leg and the
readback running in Perform whenever the deck sits on LIVE). Cost: design
composite renders whenever Perform is open. Xander's call after U7.

### Open questions

1. **Should MCP `get_preview` grow a size parameter** (e.g. 768 px) now that
   the readback is cheap? It still returns the 320-px thumbnail; agents that
   need detail can use `preview.snapshot {width}` over WS (no MCP wrapper
   yet — add one if agents keep reaching for it).

---

## Shipped (2026-10-12)

- **U0** — `[profile.dev.package."*"] opt-level = 2` in both Cargo
  manifests (render-core itself opt-level 1 in the app). Clean dev build of
  the shell: 170 s.
- **U1** — the shell's telemetry fan-in and the browser bridge subscribe to
  every channel *except* `preview`; consumers hold a refcounted
  `preview_jpeg_demand` (`usePreviewJpegDemand`). Finding: Perform's
  CUED·DESIGN cue monitor (`DeckPanel.tsx`, shown on LIVE) is a real JPEG
  consumer, so "no design render in Perform on LIVE" holds only with that
  panel hidden; on DESIGN the JPEG is fully idle.
- **U2 spike — go.** wry 0.55 puts the `WKWebView` under
  `NSWindow.contentView` = `WryWebViewParent` (not flipped); a sibling
  `NSView` inserted below it works. `drawsBackground = NO` via KVC on the
  instance works without Tauri's `macos-private-api`, window stays opaque.
  Traps hit: wgpu's Metal backend messages the view's (nil) layer → panic,
  so the view gets its own `CAMetalLayer` up front; and placement must use
  `contentLayoutRect` (the webview frame spans the title bar).
- **U3** — `PreviewParams {source?, program, dim, photo_weight,
  light_weight}`; `fs_preview` entry in `final_pass.wgsl` (projector
  `fs_main` untouched) with uv-indexed LUT for the program view; surface
  photo loaded on preview attach. Test
  `program_preview_samples_through_the_warp_at_another_size`.
- **U4/U5** — `underlay.rs` + `preview_set_slot` / `preview_mode`;
  `<NativeSlot>` in Perform, Prepare (overlay-only canvas) and Align
  (program view, grid moved into the SVG around the projector rect).
  Also fixed: `WarpCanvas` never measured itself when the alignment doc
  arrived after mount (infinite handles).
- **U6** — GPU downscale (`Downscaler`, 3×3 bilinear taps) + `preview-encode`
  worker; `preview.snapshot {width?}` RPC; devport `webview_eval` /
  `debug_log` so agents can drive the app's webview.

Measured in a dev build on the M2 (Before = the original 2026-10-12 measurement; After = fixture-a 1024×768, projector window occluded):

| Measure | Before | After |
|---|---|---|
| `preview.jpeg` p95, consumer attached | ~24 ms | 0.215 ms (5.2 ms after U0 alone) |
| `preview.jpeg` in Perform on DESIGN / Align / Prepare | ~24 ms (always on) | idle (no capture) |
| `preview.present` p95 | — | 0.2–0.5 ms |
| frame headroom | −47 % | +89 % |

Done — see [render-engine.md](../reference/render-engine.md) §1b and
[operator-ui.md](../reference/operator-ui.md).
