# Full-Frame Animation

Generate a cinematic projection animation from a surface image using Kling v3 image-to-video.

## Pipeline

1. **Generate video** — Use `kling_v3_image_to_video` with the surface image as the starting frame and a creative motion prompt. Kling produces cinematic, high-quality motion.
2. **Subtract background** — Use `subtract_background_video` to extract only the parts that changed (the animation) from the static surface. This produces purely additive light content.
3. **Simulate view** — Use `simulate_view` to preview how the projection will look on the real surface.

## Decision points

- The prompt should describe motion, not static content — Kling animates from the starting image.
- Duration scales cost: start with 5s for experiments, go up to 10-15s for final content.
- If the subtracted result looks noisy, increase the `threshold` parameter.
- If the surface image hasn't been pre-processed (sky not blacked out), run `nano_banana_pro` first to mask non-reflective regions.

## Compared to TextureFlow

- **Kling** produces photorealistic, cinematic motion — great for creature animations, flowing water, growing vines, atmospheric effects.
- **TextureFlow** produces trippy, morphing abstract textures — great for VJ loops, psychedelic visuals, ambient mood lighting.
- Choose based on the desired aesthetic.

## Output

- `projector_output.output_video` — additive light content for the projector
- `preview.simulated_view_video` — simulated real-world viewing experience
