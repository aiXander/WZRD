---
name: surface-prep
description: "Raw surface photos → the dark, aligned, masked surface image every workflow builds on. Use when starting a new surface."
metadata:
  kind: workflow
---

# Surface prep

Goal: one image of the surface, seen from the projector, darkened to how it
looks at night, with everything that can't reflect light set to pure black.
This image is `image_1` for every keyframe, the background for subtraction,
and the base for previews.

## Steps

1. `set_project("<surface-or-show-name>")` if not done; pass
   `project="<name>"` on every call below (nothing is remembered between calls).
2. Pick the path by what the user has:

   | Photos | Call |
   |---|---|
   | Night photo with the projector showing a white frame + a day photo | `prepare_surface(night_image, day_image)` — detects the lit area, aligns the day photo to it, darkens |
   | Night photo only (or an already-cropped surface) | `prepare_surface(night_image)` — darken only |
   | A camera on this machine | `capture_camera_snapshot` first, then the above |

   Set `alignment_aids=true` once per session if the user still has to line up
   the projector — it renders an alignment-aid video next to the surface.
3. `view_media` the result. Too bright → lower `max_brightness` (0.15-0.25);
   flat → raise `detail_boost` on `darken_surface`.
4. Mask what can't reflect: `nano_banana` with the "mask non-reflective areas"
   template from skill `image-prompting`, `aspect_ratio="auto"`. Check the
   result with `view_media`: the surface must be pixel-identical, only sky/voids
   black. Keep this masked image as THE surface.

Done when the masked dark surface exists and the user agrees it matches the
real view. Hand its path to the next workflow.

## Preview tuning (`simulate_view`)

Too dark → raise `projection_strength`; surface too visible → lower
`surface_weight`; dark-but-not-black areas need a little bounce → `ambient`
0.05-0.15.
