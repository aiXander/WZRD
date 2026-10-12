---
name: region-animation
description: "Separate animation per surface region (windows, panels, trunk vs canopy): segmentation map, region masks, layer pack, per-region TextureFlow clips, a story on one oddly shaped region."
metadata:
  kind: workflow
---

# Region animation

## Steps

1. **Surface** — the masked dark surface from skill `surface-prep`.
2. **Segmentation map** — `nano_banana` with the "color segmentation map"
   template from skill `image-prompting` (`image_urls=[surface]`,
   `aspect_ratio="auto"`), naming the regions you want in the prompt ("the six
   windows as separate regions, the facade as one more"). `view_media` it next
   to the surface: borders must follow real edges.
3. **Regions** — `extract_color_regions(image=<map>, surface_image=<surface>)`.
   Each region returns `region_mask` (full canvas), `region_crop_mask` and
   `region_surface_image` (cut to the region's box), `source_box`,
   `aspect_ratio`. Too many tiny regions → raise `min_area_fraction` or lower
   `max_colors`; similar colors merged → lower `delta_e_threshold`.
4. **Layer pack** — `build_layerpack(masks=<region_masks>,
   surface_image=<surface>, mask_ids=<semantic names: "window_left", "facade">,
   tags={layers: {id: {label, tags}}, groups: [...]})`. Semantic ids and labels
   are what lets the operator later say "flash the windows", and the ids are
   what `region=` takes from here on.
5. **Per-region content** (load skill `textureflow`) — for each region:
   `texture_flow(recipe=<name> or images=<style images>, region=<id>,
   width/height at the region's aspect_ratio)`; add
   `diffusion_mask=<region_crop_mask>` when the shape is far from its box (a
   round window, a pillar). Same recipe across regions = coherence; a
   different one = contrast. Run regions in parallel (≤6).
6. **Play it** — the engine places each clip in its region's box (skill
   `live-engine`, clips on regions). Only to export a full-canvas mp4 for
   another player: `reproject_video(video, **source_box)`.

Done when every region has its clip(s) in the project and the pack exists in
`packs/`. Then skill `live-engine` binds clips and effects to those layers.

## A story on one region

A narrative confined to one oddly shaped region (a creature climbing a tall
pillar) is skill `projection-story` with the region standing in for the
surface:

1. **Keyframes** from that region's `region_surface_image` as `image_1`,
   `aspect_ratio="auto"` (the crop already has a native aspect, so the
   keyframe comes back fitting the box), `region=<id>` on every call.
2. **Segments**: `kling_v25_image_to_video` between keyframes, `region=<id>`
   — it renders at the keyframes' nearest aspect, so it still fits the box.
3. **Stitch**: `concat_videos(…, seamless_loop=true)` →
   `subtract_background_video(video=<stitched>,
   background_image=<region_surface_image>)` (only the added light) →
   `to_hap(<subtracted>, region=<id>)`. Stitching drops the region, so name
   it again here — that files the clip in the region's grid row.
4. **Place it**: bind it with `placement: "region"` (it lands contain-fit in
   the region's box, the layer mask cuts the shape), or export it with
   `reproject_video(video, **source_box)`.

Fewer, larger regions give bolder animation; more regions give finer control.
