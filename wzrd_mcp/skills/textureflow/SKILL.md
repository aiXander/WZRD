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

## Working with the Studio (the operator's selection and history)

The operator browses and scores renders in the app's Studio room; the hub
remembers every render's settings and their 0-9 score.

- **"These styles" / "these clips" / "put them there"** → `studio_selection()`:
  its `styles` are the style images to pass as `images=`, `items` the
  selected renders, `target` the folder renders land in by default, and
  `project` the open show (pass it as `project=`).
- **Before a batch, check what worked:** `find_renders(style=<a style
  path>, tool="texture_flow")` (or `recipe=`, `tag=`, `min_score=7`).
  Reuse the `args` of the ≥ 7s, vary seeds around them; avoid what scored
  ≤ 3. Unrated = unjudged, not bad. Say what you're reusing.
- **Batches run in the background:** `wait=false` on every render of a
  batch returns a job id at once; keep talking, poll `jobs()` until done,
  then `check_frames` + `view_media` the outputs. A render that's clearly
  wrong mid-way (wrong region, wrong styles) → `cancel_job(job)`.
- Renders land in the Studio target folder, else the library's `inbox/`;
  `dest=` sends them elsewhere (`"project"` = the show's `generated/`).
- `label_render(path, tags=[…])` tags what you see (content, motion, mood);
  the score is the operator's alone. `move_render` / `rename_render` file
  them when asked. You can't delete media — the operator does that in Studio.

## Style recipes (the look library)

A recipe is a saved look: style images + settings that rendered well,
shared across projects (`library/recipes/`). Start from one rather than from
loose images whenever a recipe fits.

- `list_recipes()` → name, when to use it, the look in a few words.
  `list_recipes(name=…)` → the full recipe; `view_media` its `preview`.
- `texture_flow(recipe=<name>, …)` renders with the recipe's images and
  settings as defaults; every argument you pass wins. Always add what
  belongs to the target: `output_width`/`output_height` (its size),
  `region`, and `control_input`/`diffusion_mask` where a setup below asks
  for them. Older recipes may still carry the removed `width`/`height` or
  `cfg` (the server converts or ignores them; your output size wins).
  Recipes saved from a quick test carry its settings (no upscale, few
  steps) — for finals pass `use_upscale`/`upscale_esrgan`/`n_steps` explicitly.
- `save_recipe(name, from_clip=<clip>, when_to_use=…)` when the operator
  loves a render — on their word; it lands in every project's library.

## Setups

| Goal | Settings |
|---|---|
| Free abstract loop | `images` only. Often beautiful out of the box. |
| Texture that hugs the surface's shapes | `control_input` = the dark surface (non-reflective areas black), `use_controlnet1=true`, `preprocessor1="Scribble_XDoG_Preprocessor"`, `controlnet_strength1` (subtle vs strong: the tool's arg docs). For a steadier texture with the shape showing through more, add some `loop_noise` (Settings) |
| "AI-skin" the surface itself | `control_input` = surface, `denoise` 0.7-0.9 (keeps its layout and darkness, restyles it) |
| Fill one region's shape, black elsewhere in its box | `diffusion_mask` = its `region_crop_mask`, `output_width`/`output_height` = its box, **no** `control_input` (with one, its pixels show through the black mask area) |
| Whole frame morphs A → B → A | `mapping_mode="crossfade"`, 2+ images: no regions, each style fills the frame in turn and the loop fades back to the first |
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
  places it in the region's box. Set `output_width`/`output_height` to the
  region's `source_box` size (from `extract_color_regions`; scale it up,
  same aspect, if it's below the tool's minimum) — the engine contain-fits,
  so a matching aspect fills the box.
- Each render also writes the engine-playable HAP clip beside the mp4
  (`clip` in the reply, same render id). Renders land in the media library
  (the Studio target folder, else `inbox/`); `dest=` picks another folder.
- `check_frames(<mp4>)` on every render before it can reach the wall: a
  `flash` fail is never played (the audience lies looking up); `haze`
  means a lifted black floor that washes the surface; `loop_seam` near 1 =
  clean loop.

## Settings

Ranges, defaults and what each choice looks like are in the tool's own arg
docs — they come from the endpoint, so trust them over anything here or
remembered.

- **Size = the surface.** `output_width`/`output_height` is the only size:
  pass the surface's (or region box's) own pixel size. Any aspect looks as
  good as 16:9 and the video comes out at exactly that size. (`width`/`height`
  and snapping to multiples of 32 are gone.)
- **`density` = content scale, not quality.** With `use_upscale` (finals) it
  sets the pixel count of the generating pass, and forms are drawn at a fixed
  pixel size: more density = more, smaller forms (leaves, faces, repeats);
  less = a few big bold ones. Low for flat colour-region layouts and bold
  graphics, higher for intricate detail or a busy scribble input (values per
  look: the tool's docs). Render time grows with it.
- **Without `use_upscale` the output size itself is generated** (half per side
  with `upscale_esrgan`): a big output then means many tiny forms, a slow
  render, or a refusal. Only pair it with a small output size.
- **Length**: ~3 s per style image (`n_seconds`).
- **Guidance**: `nag_scale` (NAG) replaced the `cfg` args — don't send `cfg`.
  Lower = softer, waxier; raising it past the default barely changes anything.
- **`prompt`: leave it empty.** Prompts are hit or miss and pull hard on top of
  the style images; Xander usually prefers an empty prompt. Only with a
  specific reason: a few material/mood words ("molten glass",
  "bioluminescent", "dark background"), never scenes or people — and render
  the same seed with an empty prompt; keep the prompt only if it clearly wins.
- **Style regions**: `mapping_mode` is the pattern that moves the styles
  (`horizontal_stripes_up` reads well in demos; `crossfade` = whole frame,
  see Setups). `style_softness` = width of the blended band between regions
  (fraction of the short side; 0 = hard cut-out borders). Soft borders are
  Xander's pick — keep the default unless the look wants hard edges.
  `style_blend`: `tokens` = borders read as a faded transition from one
  material into the next; `attention` = the classic double-exposure overlap.
- **`ip_adapter_weight_type`**: how the style images' pull splits between
  layout and colour/texture (`style transfer` = their colours and textures,
  less layout; `composition` = only their layout). `linear` is the classic
  look; anything else is a different look — compare at the same seed.
- **Motion**: `motion_scale` sets how much moves.
- **`loop_noise`**: an aesthetic dial, not a quality one. 0 = the classic
  lively morph; higher = a more static texture where the `control_input`'s
  shape stays put and shows through more (you can lower `controlnet_strength1`),
  denser and harder in contrast. Xander rated 0.5 as good as 0; 0.75 he
  disliked (cluttered, jittery). Changes the composition like a new seed.
- **`auto_levels`**: leave it at full stretch for the wall (deep blacks, no
  grey veil); lower only for a deliberately hazy, washed look.
- **`rife_multiplier`**: output fps = 8 × it. Higher = smoother when the
  engine slows the clip down, a bit slower and bigger. Leave it unless asked.
- **Model**: `base_model` — juggernaut for realistic surfaces, darkSushiMix for
  soft creative looks, protogen for flat-color graphics.
- `denoise < 1` without a `control_input` renders near-black — always pair them.
- Keep the returned `seed` (and `request_id`) to reproduce or finish a render.
- **Backend**: `backend="local"` renders on Xander's desktop GPU ($0, slower,
  same args/outputs) only when he asks for it and the desktop is on; default
  is Modal. A local call that fails (desktop off) is never retried on Modal.

## Iterating: tests, then finals

**Draft → pick → finish** — only if the `texture_flow` tool exposes
`finish_from` (production may not have it yet):

1. Draft several seeds with the **final's** args (output size, density,
   upscales on, steps, style settings) plus `draft=true`, `wait=false` each.
   A draft IS the final's first pass (same composition), at about half a
   full render's time; its video comes out at the small first-pass size.
2. `check_frames` + `view_media` them; pick (or let the operator pick in Studio).
3. Finish each pick: `texture_flow(finish_from=<its request_id>)` — no
   images or other inputs. Only the finish args (upscale settings, output
   size at the same aspect, `auto_levels`, `output_codec`, `rife_multiplier`)
   may be set; anything that shapes the first pass (incl. `density`) must
   stay. A full render can be re-finished the same way.
4. Drafts expire 24 h after rendering and live only on the backend that
   rendered them (finish with the same `backend`). Expired → re-render with
   the same args and seed. If the tool still demands `images` for a finish,
   WZRD's finish support isn't built yet — tell the operator.

**Without drafts**: quick tests = `use_upscale=false` AND
`upscale_esrgan=false`, a small output size at the target aspect, low
`n_steps` (~30 s warm). Their composition differs from the final's (another
size = another content scale), so they judge styles and settings, not a
seed. Finals: upscales on.

## Before a live session

`textureflow_warm(minutes=180)` keeps a GPU hot (costs per idle hour; see
the tool); `textureflow_warm(minutes=0)` when done. After a redeploy the
first renders take 2-3 min extra.
