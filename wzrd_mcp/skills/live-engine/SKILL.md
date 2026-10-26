---
name: live-engine
description: "Changing what the realtime engine renders now: read the live scene, bind clips and effects to layers, write WGSL, label/group layers. Use when the operator asks to change live visuals."
metadata:
  kind: workflow
---

# Live engine authoring

The realtime engine (render-core) plays a layer pack: masks with semantic ids
(from skill `region-animation`), and a scene of **bindings** — selector →
effect → params. You edit the **design leg**; the operator previews it and
promotes design → live by hand. Promotion, masters, knob overrides and
saving to disk (the ADOPT button) are the operator's; offer, never assume.
Needs the engine running locally (Tauri shell or headless render-core with
`--ws-addr 127.0.0.1:9123`).

## The loop

1. **Orient**: `get_scene_context()` with no args, on every new operator
   command — it reflects their UI edits instantly. It lists layers (ids,
   labels, groups), bindings, effects, driver values, recent changes and the
   current `rev`. Later reads: scope them (`scope=["bindings"]`), drill with
   `ids`, pass `since_rev` to get only what changed.
2. **Name the surface** if layers are unlabeled: `set_labels` / `set_groups`
   so the operator's words ("the trunk", "the canopy") resolve to ids.
3. **Edit one thing at a time**: `upsert_binding(id, binding)` /
   `remove_binding(id)`. A CAS rejection means the design moved — re-read and
   call again.
4. **Read the verdict** in every write reply: probe band green/yellow/red +
   a thumbnail of the design composite. Red = refused for frame cost:
   simplify. No probe ran (cached shaders) → `get_preview` to look.
5. Report what changed in one or two lines and invite the operator to
   promote when it looks right.

## Bindings

```json
{"select": {"group": "canopy"}, "effect": "flash",
 "params": {"envelope": {"driver": "audio.onset", "band": "low", "decay": 0.15},
            "color": "#ffd27a"}}
```

- Selector: exactly one of `id` | `tag` | `group` | `all: true`; optional
  `pick: {"mode": "random_each", "rate": {"driver": "clock.bars", "n": 4}}`
  plays one member at a time (`random_static` = one pick at load, no rate).
- Drivers: `clock.bars(n)`, `clock.beats(n)`, `clock.phase(rate)`,
  `clock.time`, `audio.band(low|mid|high)`, `audio.onset(band, decay)`,
  `ui.slider(name, default)`; or a constant. Tempo is the scene's static
  `transport.bpm`; musical energy comes through `audio.*`.
- Built-in effects: `tint`, `hueCycle`, `flash`, `wobble`, `videoClip`; plus
  project-local ones (see the effects list in the context read).
- `"range": [from, to]` on any driver sweeps the param between the two as
  the driver goes 0 → 1: `{"driver": "audio.onset", "band": "low",
  "decay": 0.2, "range": [0.3, 1.5]}`.
- `"shape"` on the periodic clocks (`clock.bars` / `clock.beats` /
  `clock.phase`) sets the waveform over one cycle: `pingpong` (up and back,
  no jump) or `sine` (same, eased — the smoothest) for smooth textures;
  `saw` (ramp up then jump — the default when omitted), `ramp_down`,
  `square`, `random` (glides to a new random value each cycle; optional
  `"seed"`). Prefer `pingpong`/`sine` unless the operator wants a hard reset
  each cycle: `{"driver": "clock.bars", "n": 4, "shape": "sine",
  "range": [0.2, 1.0]}`. Not valid on audio/slider drivers — invert those
  by swapping the `range` ends.

## Clips on regions

A rendered clip plays through an effect input of type `image`. The
`bindings` read lists the project's clips as ready-to-paste paths, each
with the region it was made for — match the operator's words ("the clip
for the left fabric") to that region.

```json
{"select": {"id": "fabric_left"}, "effect": "videoClip",
 "params": {"image": {"clip": "/Users/x/WZRD/media/inbox/texture_flow_20261024-2114.mov",
                      "placement": "region", "rate": 1.0},
            "gain": 1.0}}
```

- `clip`: a HAP `.mov` (or a `.png`/`.jpg` still) — an absolute path (library
  clips, as `get_scene_context` lists them) or relative to the project folder. Only `.mov` plays as video — an mp4 needs `to_hap` first.
- `placement`: `"region"` (default) fits the clip into each selected
  layer's box — the box the region's clips were rendered for; `"canvas"`
  fits it to the whole frame and the layer mask cuts it out. Both
  contain-fit, never stretch.
- `rate`: playback speed, negative plays backwards; the operator's speed
  master bends it too.
- One binding over a group (`{"group": "fabrics"}`) plays the same clip in
  every member's own box.
- A clip that can't open is a `WARNING:` line in the reply, not a failure:
  read it, fix the path, call again.
- To swap a region's clip, upsert the same binding id with the new path —
  the operator's clip grid does the same on click.

**Clip effects**: any effect with an `image` input plays a clip *through*
it, so the clip responds to the music. The global library has 15
(`library/effects/`: `kaleido`, `mirror`, `rgbSplit`, `pixelate`,
`posterize`, `edgeGlow`, `noiseDisplace`, `strobeGate`, `hueShift`,
`lumaKey`, `heat`, `waveWarp`, `dither`, `vignette`, `transform`); each
descriptor's `description` and `audio` fields say what it does and which
dial to put an `audio.*` driver on. A project only sees the ones copied
into its `effects/` folder (`cp -R library/effects/<name>
<project>/effects/`; the context read lists what's there). Same binding
shape, `"effect": "kaleido"`, plus its dials.

## Writing a WGSL effect

`upsert_effect(name, wgsl, descriptor)` — the body implements
`fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>` returning premultiplied,
additive RGBA (black = no light). Available: `state.*` (time, bar/beat phase,
bpm, audio bands/onsets — already scaled by the operator's audio master),
`f_param(N)` (8 floats) / `c_param(N)` (4 colors) in descriptor input order,
`sample_mask(uv)`, `sample_image(uv)` (the clip of an `image` input, premultiplied, transparent outside its placement box), and per-layer identity `layer_seed()`, `layer_index()`,
`layer_count()`, `layer_centroid()`, `layer_bbox()` — add `layer_seed()` to a
phase to desync copies across layers. Descriptor:
`{"inputs": [{"name": "rate", "type": "float", "default": 1.0, "min": 0, "max": 8}, {"name": "tint", "type": "color", "default": "#ffffff"}]}`;
add `{"name": "image", "type": "image"}` for a clip effect, and follow the
library's conventions: read the clip through one `fx_in(uv)` helper, return
colour × the `mask` argument × a `gain` dial, and black stays black (no light). Copy the nearest
library effect as a starting point. The engine clips each pass to its region's
mask box, so light outside the region is cut; an effect that must glow past it
declares `"bounds": "canvas"` in the descriptor (calling `sample_mask` also
makes it full-canvas, and slower).

`upsert_effect` validates first (line-mapped errors, nothing written). The
probe only runs when the design scene binds the effect, so bind it first to
see a verdict and thumbnail on each iteration.
