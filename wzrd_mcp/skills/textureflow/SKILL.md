---
name: textureflow
description: "Abstract, ambient or looping motion from style images (VJ loops, surface-hugging textures, region fills, logos). Load before any texture_flow call."
metadata:
  kind: guide
---

# TextureFlow

TextureFlow turns 1-6 style images into a smooth, morphing animation. Style
images steer textures, colors and content but never appear verbatim. It tells
no story and takes no keyframes — for characters and narrative use skill
`projection-story`. Reach for it for ambient motion, VJ loops, abstract fills,
logos.

## Recipes

| Goal | Settings |
|---|---|
| Free abstract loop | `images` only. Often beautiful out of the box. |
| Texture that hugs the surface's shapes | `control_input` = the dark surface (non-reflective areas black), `use_controlnet1=true`, `preprocessor1="Scribble_XDoG_Preprocessor"`, `controlnet_strength1=0.45` (0.35 subtle … 0.65 strong) |
| "AI-skin" the surface itself | `control_input` = surface, `denoise` 0.7-0.9 (keeps its layout and darkness, restyles it) |
| Fill one region / island, black elsewhere | `diffusion_mask` = the region mask, **no** `control_input` (with one, its pixels show through the black mask area) |
| Animated logo | logo as `control_input`, `use_controlnet1=true`, Canny or Scribble |
| Motion-driven | a simple motion clip/GIF as `control_input` |
| Animated QR code | QR as `control_input`, `preprocessor1="none"` (luminance mode) |
| Follow a dancer's pose | person clip as `control_input`, `preprocessor1="OpenposePreprocessor"` |

Light landing on non-reflective areas (sky, voids) is invisible anyway, so a
full-frame TextureFlow render can usually be projected as-is: preview it with
`simulate_view` against the surface instead of background-subtracting it.

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
