---
name: region-animation
description: "Separate animation per surface region (windows, panels, trunk vs canopy): segmentation map, region masks, per-region TextureFlow, reprojection, layer pack for the engine."
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
4. **Per-region content** (load skill `textureflow`) — for each region:
   `texture_flow(images=<style images>, diffusion_mask=<region_crop_mask>,
   width/height matching the region's aspect_ratio)`. Same style images across
   regions = coherence; different ones = contrast. Run regions in parallel (≤6).
5. **Back on the canvas** — `reproject_video(video, **source_box)` per region.
   Each output is a full-canvas video, black outside its region.
6. **Layer pack** (for the live engine) — `build_layerpack(masks=<region_masks>,
   surface_image=<surface>, mask_ids=<semantic names: "window_left", "facade">,
   tags={layers: {id: {label, tags}}, groups: [...]})`. Semantic ids and labels
   are what lets the operator later say "flash the windows".

Done when every region has a reprojected video and (if the engine is the
target) the pack exists in `packs/`. Then skill `live-engine` binds effects to
those layers live.

Fewer, larger regions give bolder animation; more regions give finer control.
