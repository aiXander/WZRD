---
name: textureflow
description: "Abstract, ambient or looping motion from style images (VJ loops, surface-hugging textures, region fills, logos), and the style recipe library. Load before any texture_flow call."
metadata:
  kind: guide
---

# TextureFlow

TextureFlow turns 1-6 style images into a smooth, morphing animation. Style
images steer textures, colors and content but never appear verbatim. It tells
no story and takes no keyframes — for characters and narrative use skill
`projection-story`. Reach for it for ambient motion, VJ loops, abstract fills,
logos.

## Style recipes (the look library)

A recipe is a saved look: style images + settings that rendered well,
shared across projects (`library/recipes/`). Start from one rather than from
loose images whenever a recipe fits.

- `list_recipes()` → name, when to use it, the look in a few words.
  `list_recipes(name=…)` → the full recipe; `view_media` its `preview`.
- `texture_flow(recipe=<name>, …)` renders with the recipe's images and
  settings as defaults; every argument you pass wins. Always add what
  belongs to the target: `width`/`height` (its aspect), `region`, and
  `control_input`/`diffusion_mask` where a setup below asks for them.
  Recipes saved from a draft carry draft settings (no upscale, 4 steps) —
  for finals pass `use_upscale`/`upscale_esrgan`/`n_steps` explicitly.
- `save_recipe(name, from_clip=<clip>, when_to_use=…)` when the operator
  loves a render — on their word; it lands in every project's library.

## Setups

| Goal | Settings |
|---|---|
| Free abstract loop | `images` only. Often beautiful out of the box. |
| Texture that hugs the surface's shapes | `control_input` = the dark surface (non-reflective areas black), `use_controlnet1=true`, `preprocessor1="Scribble_XDoG_Preprocessor"`, `controlnet_strength1=0.45` (0.35 subtle … 0.65 strong) |
| "AI-skin" the surface itself | `control_input` = surface, `denoise` 0.7-0.9 (keeps its layout and darkness, restyles it) |
| Fill one region's shape, black elsewhere in its box | `diffusion_mask` = its `region_crop_mask`, width/height at its aspect, **no** `control_input` (with one, its pixels show through the black mask area) |
| Animated logo | logo as `control_input`, `use_controlnet1=true`, Canny or Scribble |
| Motion-driven | a simple motion clip/GIF as `control_input` |
| Animated QR code | QR as `control_input`, `preprocessor1="none"` (luminance mode) |
| Follow a dancer's pose | person clip as `control_input`, `preprocessor1="OpenposePreprocessor"` |

Light landing on non-reflective areas (sky, voids) is invisible anyway, so a
full-frame TextureFlow render can usually be projected as-is: preview it with
`simulate_view` against the surface instead of background-subtracting it.

## For the live engine

- **`region=<pack layer id>`** (with `project=<show>`) on every render meant
  for one region: its record files it under that region, and the engine
  places it in the region's box. Render at the region's `aspect_ratio`
  (from `extract_color_regions`) — the engine contain-fits, so a matching
  aspect fills the box.
- Each render also writes the engine-playable HAP clip beside the mp4
  (`clip` in the reply, same render id). Renders land in the media library
  (the Studio target folder, else `inbox/`); `dest=` picks another folder.
- `check_frames(<mp4>)` on every render before it can reach the wall: a
  `flash` fail is never played (the audience lies looking up); `haze`
  means a lifted black floor that washes the surface; `loop_seam` near 1 =
  clean loop.

## Settings

- **Length**: ~3 s per style image (`n_seconds`).
- **Resolution**: keep the target aspect; snap both sides to multiples of 32.
  Finals: long side ~1024, short side ≥384 (16:9 → 1024×576). For a region,
  use the region crop's aspect ratio from `extract_color_regions`.
- **Fast iteration**: `use_upscale=false` AND `upscale_esrgan=false`, 384-512
  px, `n_steps=4-6` (~30 s warm). Finals: upscales on (~6 min warm).
- **Motion**: `motion_scale` 0.9 subtle … 1.25 lively. `mapping_mode` picks the
  motion pattern (`horizontal_stripes_up` reads well in demos; avoid
  `rotating_segments_counter_clockwise` unless asked).
- **Model**: juggernaut (realistic, default), darkSushiMix (soft, pastel,
  creative), protogen (anime, flat color).
- `denoise < 1` without a `control_input` renders near-black — always pair them.
- Keep the returned `seed` to reproduce a render you like.

## Before a live session

`textureflow_warm(minutes=180)` keeps a GPU hot (~$2/h idle);
`textureflow_warm(minutes=0)` when done. After a redeploy the first renders
take 2-3 min extra.
