# Effects library — clip effects for the live set

The global library of reusable clip effects (gig plan C6/F1, landed
2026-10-12). Effects are authored once in `library/effects/` and **copied**
into a project's `effects/` — a project stays self-contained, never links
back. The engine side (how an `image` input, `sample_image` and HAP decode
work) is [render-engine.md](render-engine.md) §2.1/§2.4; this doc covers the
library: layout, the catalog, the authoring conventions every effect follows,
and how the gates are run.

## Layout and install

```
library/effects/<name>/
  shader.wgsl        # fn effect(uv, mask) -> premultiplied vec4
  descriptor.json    # inputs with min/max/default/step; + "description", "audio" (engine ignores both)
  preview.jpg        # 640×480, a real frame of the effect on a TextureFlow clip
```

- **Install = `cp -R library/effects/<name> <project>/effects/`.** The
  engine picks it up from `<scene_dir>/effects/` (file watcher, D15). There
  is no library browser and no MCP tool yet: those are post-gig
  ([../TODO/vj-toolbox.md](../TODO/vj-toolbox.md) phase 1).
- `wzrd-app/scripts/fixture_projects.py` copies the whole library into
  fixture-a/-b on every rebuild.
- The other C6 folders: `library/recipes/` (TextureFlow style recipes) is
  in [mcp-server.md](mcp-server.md) "Style recipes"; `library/midi/` is
  pinned in [../TODO/gig-2026-11-live-video.md](../TODO/gig-2026-11-live-video.md) C6.

## Catalog (15)

Every effect ends with a `gain` dial (0–2). "Audio" is the dial to put an
`audio.*` driver on (with a `range`, render-engine.md §2.1). Defaults look
right with no audio (`--no-osc`).

| Effect | What it does | Audio dial |
|---|---|---|
| `kaleido` | mirrored-wedge kaleidoscope around the clip centre; `twist` bends it into a spiral (Resolume's Bendoscope) | `zoom` / `twist` |
| `mirror` | `mode` 0 L→R, 1 T→B, 2 quad fold, 3/4 flip H/V (Resolume's Flip); fold axis sways | `sway` |
| `rgbSplit` | R/B channel offset along a spinning direction (Shift RGB) | `amount` |
| `pixelate` | big pixels anchored on the clip box; `dot` makes LED dots with dark gaps | `size` |
| `posterize` | a few flat colour bands, perceptually even | `levels` |
| `edgeGlow` | Sobel outlines lit in the clip's colour (or `tint`), flat areas dark | `boost` |
| `noiseDisplace` | displacement by two drifting value-noise fields | `amount` |
| `strobeGate` | beat-synced soft gate, **hard-capped at 2.5 cycles/s** | `depth` (never `rate`) |
| `hueShift` | hue rotation (offset + per-bar cycle) + saturation | `shift` |
| `lumaKey` | keep bright parts (or dark, `invert`); kills a lifted floor | `threshold` |
| `heat` | thermal false colour from luma; palette scroll | `offset` |
| `waveWarp` | travelling sine-wave ripple | `amp` |
| `dither` | ordered Bayer 8×8 into 2–8 levels (or mono) | `size` |
| `vignette` | fade toward the clip box edges | `radius` |
| `transform` | zoom / rotate / spin / pan, mirror-tiled past the edges | `zoom` |

Not in the library because they need previous frames: Trails, Delay RGB,
feedback looks. Those wait for feedback + `post` (roadmap §5.16, toolbox).

## Authoring conventions (follow them for new effects)

- **One `image` input, one sampling site.** Every shader defines
  `fn fx_in(uv) -> vec4<f32> { return sample_image(uv); }` and reads the
  clip only through it. Post-gig the toolbox re-cuts these as chain filters
  by swapping `sample_image` → `sample_prev` in that one line.
- **Self-contained WGSL.** Helpers (`tri`, `in_box`, `clamp_box`, `hash3`,
  noise) are duplicated per file until `#import` lands (roadmap §5.5).
- **Additive-safe output.** Return premultiplied colour × `mask` (sampled
  at the *output* uv, never the displaced one) × `gain`, the same as the
  built-in `videoClip`. This is also what lets the engine **clip each pass
  to its region's mask box** (render-engine.md §2.4 output bounds): light
  outside it is cut. Don't call `sample_mask` (that makes the effect shade
  the whole canvas — slower), and if an effect must glow past its region,
  declare `"bounds": "canvas"` in its descriptor. Colour ops
  un-premultiply (`rgb / max(a, 1e-4)`),
  work, and re-multiply. Every effect maps black to zero light: posterize,
  dither and heat keep band 0 at black, lumaKey's keyed area is black.
- **Work in clip-box space.** `layer.image_box` is the contain-fitted rect
  the clip lands in; kaleido/mirror/transform/vignette/pixelate anchor on
  it (aspect from `box size × state.resolution`). Folds and zooms
  mirror-repeat (`tri`) so they never open holes.
- **Displacement clamps into the box.** rgbSplit, noiseDisplace and
  waveWarp return 0 for pixels outside the box and clamp displaced reads
  into it — otherwise the box edge pulls in transparent black seams.
- **Musical time** is `bars = state.time * state.bpm / 240` (speeds are
  "per bar"); `state.time` follows the leg transport and speed master.
- **Every dial declares `min`/`max`/`default` (+ `step`, `unit`)** — the
  UI's knobs and the MOD ranges read them through `effect.describe`.
  Mode/toggle dials are floats with `step: 1`.
- **Photosensitivity (lying-down audience).** No effect may flash above 3
  large flashes/s at default dials. strobeGate halves its rate until
  `rate × bpm/60 ≤ limit` so it stays on the beat grid, and all its layers
  gate **in phase** — per-layer desync (`layer_seed`) would multiply the
  flashes the room sees. The cap is 2.5, not 3: `check_flash` counts edges
  in a 1 s window, so a 3 Hz square gate reads 3.5 flashes/s (fail). A
  Python port of the gate through `check_flash` passes at every tempo
  20–400 BPM even at the worst dials (max 3.0/s; defaults max 2.5/s).

## Gates and how to re-run them

`scripts/fx_gate.py` (+ `scripts/winbounds.swift`, compiled on first use)
drives a release `render-core` over WS on port 9177. `--bench` is a scratch
project with a pack and `generated/*.mov`; it gets one binding per layer
(≤ 7, clips round-robin). Outputs go to `test_results/fx_gate/`.

- **`probe`** — `effect.upsert` → `scene.load` binding it on the design
  leg → `effect.upsert` again; the second reply carries the §2.6 probe
  verdict. Probe predictions jitter with machine load (a one-off yellow at
  11 ms re-probed green at 2.5 ms). **The probe thumbnail is always black
  for clip effects**: the probe binds no image (render-engine.md §4 #11).
- **`run`** — per effect a cold boot; `perf` frame-interval p95 over 10 s
  (live only, then with the design leg awake via a `preview`
  subscription), then an 8 s **60 fps screen recording** of the window →
  `check_flash`. Two traps: the engine window must be **visible and in
  front** (a covered window's swapchain is throttled to ~30 fps, so the
  baseline itself reads 33 ms), and a second `telemetry.subscribe` on one
  connection **replaces** the channel set rather than adding to it.
- **`preview`** — one-layer canvas scene, window screenshot → `preview.jpg`.

**Last full run (2026-10-12, M2 on battery, `projects/fx-bench`):** all 15
pass — live p95 16.95–17.26 ms (baseline `videoClip` 17.14), live+design
≤ 17.5, flashes/s ≤ 2.5 (edgeGlow; strobeGate 1.5). Lowest headroom with the
design leg awake: edgeGlow 0.35 (its 9 clip taps), everything else ≥ 0.69.

**Why not record the `preview` channel (~15 fps)?** Simulated square-wave
strobes through `check_flash` (2026-10-12): at 15 fps a 7.5 Hz strobe reads
0 flashes/s and 7–8 Hz read ~2 (false pass), while timing jitter adds +0.5 to
honest rates (a legal 3 Hz reads 3.5, false fail). 30 fps tracks 60 fps.
`check_flash` itself is blind to exactly 10 Hz (and multiples) at any fps,
because it compares each frame with the one 100 ms later.
