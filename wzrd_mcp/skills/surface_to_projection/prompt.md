# Surface to Projection

Full end-to-end pipeline for creating projection-mapped content from surface photos.

## Pipeline

1. **Prepare surface** — Use `prepare_surface` with the night photo (and optional day photo) to detect, align, and darken the projection surface.
2. **Mask non-reflective regions** — Use `nano_banana_pro` to turn sky and other non-reflective areas to pure black. This ensures the projector doesn't waste light on areas that won't reflect it.
3. **Generate content** — Use `texture_flow` with the style images and the masked surface as a controlnet input. This generates animation that follows the surface contours.
4. **Subtract background** — Use `subtract_background_video` to extract only the additive light content from the generated video. The masked surface serves as the background reference.
5. **Simulate view** — Use `simulate_view` to composite the projector output onto the surface image, previewing the real-world viewing experience.

## Decision points

- If the user provides both day and night photos, use the full pipeline (detect + align + darken). If only a night photo, use darken-only mode.
- The TextureFlow controlnet strength (0.35-0.65) can be tuned based on how closely the animation should follow the surface shapes. Start at 0.45.
- If the preview looks too dark, increase `projection_strength` in `simulate_view`. If the surface is too visible, decrease `surface_weight`.
- If the user wants a Kling-style cinematic video instead of TextureFlow, swap step 3 for `kling_v3_image_to_video`.

## Output

The final output includes:
- `projector_output.output_video` — the file to send to the projector
- `preview.simulated_view_video` — preview of the expected real-world appearance
