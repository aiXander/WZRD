# Island Animation

Segment a projection surface into independent color regions and generate unique animated content for each.

## Pipeline

1. **Extract color regions** — Use `extract_color_regions` to segment the surface into distinct color-based islands. Each island gets its own mask and bounding box.
2. **Generate per-island content** — For each region, use `texture_flow` with the region mask as `diffusion_mask`. This ensures animation only fills the island area while the rest stays black.
3. **Reproject** — For each island, use `reproject_video` to place the generated video back at its original position on the full projection canvas using the `source_box` coordinates.

## Decision points

- Adjust `max_colors` based on surface complexity. Fewer colors = larger regions = bolder animations. More colors = finer segmentation.
- You can use different style images per island for variety, or the same set for visual coherence.
- The `delta_e_threshold` in `extract_color_regions` controls color merging sensitivity — lower values distinguish subtle color differences.
- After reprojection, the separate island videos can be composited together or played as independent VJ layers.

## Output

- Per-island animated videos (projector-ready, black background)
- Per-island reprojected videos (positioned on the full canvas)
- The island videos can be combined via video mixing in VJ software
