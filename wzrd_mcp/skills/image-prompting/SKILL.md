---
name: image-prompting
description: "Writing nano_banana prompts: keyframes on a fixed surface, reference images, the anti-slop formula, mask and segmentation-map templates. Load before any nano_banana call."
metadata:
  kind: guide
---

# Image prompting for projection (nano_banana)

The model draws one static frame. It knows only what this call gives it: the
references and the prompt. Its output will be *projected on top of the real
surface*, so every unintended change to the background becomes visible light.

## Keyframes on a surface

- **The surface is always `image_1`.** Put the dark surface photo first in
  `image_urls`, every call, even late in a sequence — chained edits drift and
  lose the scene by frame 3-4 when the original isn't re-pinned.
- **Background lock, stated explicitly.** Say that the background from image_1
  must be copied unaltered, at exactly image_1's aspect ratio and framing, and
  that only the subject/scene elements are added on top. Allowed, deliberate
  background changes are lighting events: a window lights up, a door opens, a
  pattern is projected onto a wall area.
- **Name references by label and role**, never by the character's identity:
  "the character in image_2", "the texture in image_3". Let the reference
  carry the look; extra descriptive words bias the output away from it.
- **Scale and pose.** Characters render too big by default: state the size
  relative to a surface feature ("about as tall as the doorway"). Ask for a new
  pose or action that fits the moment, never the reference's pose.
- **Use the surface.** Anchor action to its features: emerging from a window,
  walking along a ledge, roots spreading through cracks. Invent doors/windows
  only where they make sense structurally. Flat, low-detail areas take
  projected textures well.
- **Off-canvas = gone.** A character leaving the frame disappears in reality.
- **Bright, luminous subjects.** Projection is additive light on a dark
  surface: favour glowing, high-luminance subjects with light spill onto the
  surrounding surface; dark subjects emit no light and vanish.
- **One frame, no relations.** Comparative wording ("now she is closer than
  before") only works if the earlier keyframe is passed as a reference; then
  name it ("continue from image_3: …"). Otherwise describe the frame absolutely.
- **Placement**: spell out left/right/top/bottom explicitly; the model
  sometimes mirrors.

## The prompt formula (anti-slop)

Write a dense narrative paragraph, never a request ("Please generate…"):

1. The physical scene: background lock + reference roles.
2. Subject, action, pose, size, placement, light emission and spill.
3. Style grounding — one choice per slot from `references/anti-slop.md`
   (lens + film stock + lighting for photographic; an 8-layer style stack
   for illustrated). Generic quality words ("4k, masterpiece, hyper-realistic,
   ArtStation, glowing, highly detailed") pull toward generic AI gloss.
4. **End with the terminal constraint block** for the chosen mode:
   - Photographic: `ACTIVELY AVOID THESE: No high dynamic range (HDR), no perfect symmetry, no center-weighted composition, no over-smoothed skin, no plastic textures, no vivid artificial saturation, no studio backdrops, and no artificial depth of field blur.`
   - Illustration: `ACTIVELY AVOID THESE: No digital airbrushing, no glowing neon borders, no 3D rendering, no smooth mathematical gradients, no symmetrical perfection, no generic digital illustration aesthetic.`
   - Add the projection line to either: `No changes to the background from image_1 beyond the described lighting events; same framing and aspect ratio.`

Load `references/anti-slop.md` for the lens / film / lighting / art-movement /
medium lists.

## Templates

Segmentation maps are made in the app's Show setup, which keeps the saved
segmentation templates; the templates below are for your own calls.

**Mask non-reflective areas** (surface prep): `image_urls=[surface]`, prompt:
"Keep the surface in image_1 exactly as it is, pixel for pixel, and replace
every area that cannot reflect projector light — sky, open space behind the
structure, glass, distant background — with pure black (#000000). Same framing
and aspect ratio as image_1. ACTIVELY AVOID THESE: no relighting, no texture
changes, no soft gradients at the mask edge."

**Color segmentation map** (region animation), adapt the bracketed parts:
"Generate a segmentation image converting the [target surface description] in
image_1 into N regions of different solid colors for segmentation mapping. Each
region is a flat, uniform, highly saturated color with no gradients or texture —
pure segmentation zones. Everything that is not part of the segmented surface is
pure black. Colors are as distinct from one another as possible, a bold
color-coded map. The output has exactly the same aspect ratio and framing as
image_1 so it overlays it perfectly. ACTIVELY AVOID THESE: No gradients, no
blending between colors, no soft transitions, no texture on the regions or
background, no artistic effects."

## Settings

`resolution="1K"` for exploration, `"2K"` for finals or small detail.
`aspect_ratio` = the locked one (or `"auto"` to inherit image_1's).
`thinking_level="minimal"` is fine for masks and simple edits.
