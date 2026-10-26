---
name: region-animation
description: "Separate animation per surface region (windows, panels, trunk vs canopy) on the regions the operator set up: per-region TextureFlow clips, a story on one oddly shaped region."
metadata:
  kind: workflow
---

# Region animation

The operator splits the surface into regions in the app's Show setup
(segmentation map → regions → layer pack). You animate them.

## Steps

1. **Regions** — `show_info()` lists them: `id` (what `region=` takes),
   `label`, `tags`, `source_box` (the box the engine plays the region's clips
   in). No regions → the operator runs Show setup's Regions step first.
2. **A region's inputs** — `show_info(region=<id>)` adds `region_inputs`:
   `mask` (white = the region) and `shape` (the surface, black outside the
   region), both cut to its box.
3. **Per-region content** (load skill `textureflow`) — for each region:
   `texture_flow(recipe=<name> or images=<style images>, region=<id>,
   output_width/output_height at the box's aspect)`; add
   `diffusion_mask=<mask>` when the shape is far from its box (a round
   window, a pillar), `control_input=<shape>` to hug its forms. Same recipe
   across regions = coherence; a different one = contrast. Run regions in
   parallel (≤6).
4. **Play it** — the engine places each clip in its region's box and the
   layer mask cuts the shape: skill `live-engine` binds clips and effects to
   those layers.

Done when every region has its clip(s). Fewer, larger regions give bolder
animation; more regions give finer control.

## A story on one region

A narrative confined to one oddly shaped region (a creature climbing a tall
pillar) is skill `projection-story` with the region standing in for the
surface:

1. **Keyframes** from that region's `shape` as `image_1`,
   `aspect_ratio="auto"` (the crop already has the box's aspect, so the
   keyframe comes back fitting it), `region=<id>` on every call.
2. **Segments**: `kling_v25_image_to_video` between keyframes, `region=<id>`
   — it renders at the keyframes' nearest aspect, so it still fits the box.
3. **Stitch**: `concat_videos(…, seamless_loop=true)` →
   `subtract_background_video(video=<stitched>, background_image=<shape>)`
   (only the added light) → `to_hap(<subtracted>, region=<id>)`. Stitching
   drops the region, so name it again here — that files the clip under the
   region.
4. **Place it**: bind it with `placement: "region"` (skill `live-engine`).
