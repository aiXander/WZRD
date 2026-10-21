---
name: surface-prep
description: "Venue photos → the rectified lit crop and the dark surface, pinned as the show's assets (asset:surface_lit, asset:surface). Use when starting a new surface."
metadata:
  kind: workflow
---

# Surface prep

Goal: the show's surface images, seen from the projector, pinned as named
assets every later step takes as `"asset:<name>"`:

| Asset | What | Used as |
|---|---|---|
| `photo_lit` | photo with the projector showing full white | detection input |
| `photo_ambient` | same view, room lit (optional, better colour) | darkening source |
| `surface_lit` | rectified lit crop at the canvas | segmentation input, TextureFlow `control_input` |
| `surface` | dark colour-corrected surface at the canvas | `image_1` for keyframes, subtraction background |
| `surface_ambient` | rectified aligned ambient photo, undarkened | reference |

## Steps

1. `set_project("<show>", canvas=…)` if not done; pass `project="<show>"` on
   every call below (nothing is remembered between calls).
2. Photos, whole frame: `import_photos(paths=[lit, ambient], conform=False)`
   (or newest from ~/Downloads). Pin them:
   `pin_asset("photo_lit", <path>)`, `pin_asset("photo_ambient", <path>)`.
3. `rectify_surface(lit_photo="asset:photo_lit", ambient_photo="asset:photo_ambient")`
   (no ambient photo → omit it; the lit crop is darkened instead). One run
   lands in `surfaces/`: `<run>_lit.png`, `<run>_surface.png`, `<run>_ambient.png`.
4. `view_media` the three. Wrong quad → re-run with `corners=` (the result's
   `corners`, nudged; normalised TL, TR, BR, BL in the lit photo). Too bright →
   lower `max_brightness` (0.15-0.25); flat → raise `detail_boost`.
5. Pin the keepers: `pin_asset("surface_lit", …)`, `pin_asset("surface", …)`,
   `pin_asset("surface_ambient", …)`. Check with `list_assets()`.
6. Optional — mask what can't reflect: `nano_banana` with
   `list_prompts("mask")`'s template, `image_urls=["asset:surface"]`,
   `aspect_ratio="auto"`, `dest="project/surfaces"`. `view_media`: the surface
   must be pixel-identical, only sky/voids black; then pin it as `surface`.

Done when `surface_lit` and `surface` are pinned and the user agrees they
match the real view.

## Preview tuning (`simulate_view`)

Too dark → raise `projection_strength`; surface too visible → lower
`surface_weight`; dark-but-not-black areas need a little bounce → `ambient`
0.05-0.15.
