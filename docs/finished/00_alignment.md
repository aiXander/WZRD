# Alignment layer — n-point output warp (§5.14)

> **Status: SHIPPED 2026-08-12 (Phases A–D).** Durable content lives in
> [../reference/render-engine.md](../reference/render-engine.md) **§2.8**
> (model, `alignment.json` contract, engine-wide/not-per-leg rule, LUT
> invariant, Y-flip trap, UI behaviour) + §2.3 (RPC + telemetry rows).
> Roadmap entry: [render-engine-roadmap.md](render-engine-roadmap.md) §5.14.
> This husk is retired to `docs/finished/`.

## What shipped

- **Phase A — engine core.** `render-core/src/alignment.rs` (document, Heckbert
  homography + analytic inverse, Wendland C² kernel + LU solve,
  `AlignmentState`, migration from the legacy calibration matrix);
  `shaders/warp_bake.wgsl` + `homography.wgsl` → `final_pass.wgsl`, with the
  fullscreen vertex stage extracted to `shaders/fullscreen_vs.wgsl` so the two
  passes cannot drift on the Y flip; `gpu.rs` `WarpTarget` (Rg32Float offset
  LUT + bake pipeline) and `FinalPassUniforms` replacing `HomographyUniforms`;
  `core.rs` boot load, rebake-on-dirty inside the existing frame encoder,
  debounced persist, shutdown snapshot. `alignment.json` gitignored.
- **Phase B — control surface.** `alignment.get/set/reset`, all inline; sticky
  `alignment` telemetry channel; Tauri commands + typed `ipc.ts` wrappers +
  store slice. Headless driver kept at `render-core/tools/align_drag.py`
  (corner sweep, live handle demo, `--verify-isolation`).
- **Phase C — the Align route.** `routes/Align.tsx` + `components/WarpCanvas.tsx`
  + `state/alignment.ts` (per-facet optimistic state, one push per animation
  frame, engine owns persistence) + `state/warpMath.ts`. Four routes now:
  ⌘1 Prepare / ⌘2 Align / ⌘3 Perform / ⌘4 Debug.
- **Post-dogfood additions (same day).** Three things the first real session
  exposed, all in reference §2.8: the canvas draws the **actual field** as a
  warped grid (the `matrix3d` underlay can only express the corner
  homography, so before this a handle drag changed nothing visible in the UI
  and read as broken — the engine was warping correctly the whole time);
  **pan** by dragging inside the quad or arrowing with nothing selected; and
  **edge handles** placed by clicking the outline. The grid reads the engine's
  solved coefficients (`weights`, added to the payload as read-only derived
  state) rather than re-solving in the UI.
- **Sticky-channel drift bug.** The Tauri host kept its own hardcoded copy of
  the sticky-channel list, so `alignment` — which emits only at boot — never
  reached `last_payload` and the tab sat on its placeholder forever. Fixed by
  making `telemetry::is_sticky()` the single definition both sides call, with
  tests. Worth remembering as a shape: any per-channel policy duplicated in a
  host will drift on the next channel added.
- **Phase D item 12 — test patterns** (`alignment.setTestPattern`), pulled
  forward as planned since manual alignment against a grid beats aligning
  against live content.

## Verification that exists as tests

- `alignment.rs`: identity round-trip, keystone maps corners exactly, the
  no-op-add property, corner drags carrying handles, degenerate corners and
  coincident handles rejected without half-applying, the handle cap as a
  message not a truncation, migration inverting the dest→source matrix.
- `gpu.rs`: naga parse+validate of both output shaders (this caught `meta`
  being a reserved WGSL keyword before it ever reached a device);
  `baked_lut_matches_the_cpu_model` and `final_pass_samples_through_the_warp`
  — the two halves of the Y-flip trap, both GPU-backed, both skipping cleanly
  without an adapter.
- `align_drag.py --verify-isolation`: `alignment.json` byte-identical across
  `scene.load` + `pull` (the §6 "alignment survives everything else" rule).

## Still open (deferred by design)

- **Camera-driven auto-alignment.** Hooks designed for but not built:
  `alignment.setField {width, height, data}` (dense offset upload — the LUT is
  already the runtime representation, so this is a texture upload plus a
  "model = dense" flag) and the capture→detect→solve loop as an external
  script against the existing WS surface, next to `wzrd/align.py`.
  `align_drag.py` is its skeleton.
- **Warped native preview toggle** (`preview.setWarp`) — only if dogfooding
  shows the unwarped preview is confusing on the Align tab. §2.6's convention
  says it shouldn't be.
- **Multi-projector / edge blending.** Nothing forecloses it (the document is
  already a per-output object), but one output was the scope.
- **Handle radius default (0.35) wants one real projector session to tune.**
  Too small feels like a dent, too large like the whole image sliding — and at
  0.35 a handle a quarter of the way in reaches a corner and will pull it. If
  that bites in practice, pin the corners with four zero-residual basis
  functions; don't grow fold-prevention-style machinery.
