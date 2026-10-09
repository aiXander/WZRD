---
name: live-engine
description: "Changing what the realtime engine renders now: read the live scene, bind effects to layers, write WGSL, label/group layers. Use when the operator asks to change live visuals."
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
   call again. `set_scene` only for a first scene from empty or a full rewrite.
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
- Built-in effects: `tint`, `hueCycle`, `flash`, `wobble`; plus project-local
  ones (see the effects list in the context read).

## Writing a WGSL effect

`upsert_effect(name, wgsl, descriptor)` — the body implements
`fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>` returning premultiplied,
additive RGBA (black = no light). Available: `state.*` (time, bar/beat phase,
bpm, audio bands/onsets — already scaled by the operator's audio master),
`f_param(N)` (8 floats) / `c_param(N)` (4 colors) in descriptor input order,
`sample_mask(uv)`, and per-layer identity `layer_seed()`, `layer_index()`,
`layer_count()`, `layer_centroid()`, `layer_bbox()` — add `layer_seed()` to a
phase to desync copies across layers. Descriptor:
`{"inputs": [{"name": "rate", "type": "float", "default": 1.0, "min": 0, "max": 8}, {"name": "tint", "type": "color", "default": "#ffffff"}]}`.

Iterate syntax with `validate_wgsl` (cheap, line-mapped). The probe only runs
when the design scene binds the effect, so bind it first to see a verdict and
thumbnail on each iteration. `remove_effect` deletes the files; remove the
bindings that use it too.
