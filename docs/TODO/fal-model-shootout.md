# FAL model shootout — replace Nano Banana Pro and Kling

> **Goal:** decide, on WZRD-shaped tasks rather than leaderboards, whether the
> two FAL tools in `wzrd_mcp/fal_tools.py` should move to newer models:
> - **Image:** ~~`nano_banana_pro` → Nano Banana 2.1~~ **done 2026-10-09**
> - **Video:** `kling_v25_image_to_video` (Kling v2.5 Turbo Pro) → **MiniMax H3 Max**
>
> Everything is on FAL behind the existing `FAL_KEY`, so this is a small
> harness script plus one paid test run (budget below). The output is a
> side-by-side contact sheet Xander judges, then a one-commit endpoint swap.
> Status: **Part A (image) done** — Nano Banana 2.1 won and is live as the
> `nano_banana` tool. **Part B (video) not started.** Harness:
> `scripts/fal_shootout.py` (image-only so far).

## Why these contenders (research, 2026-10-09)

Artificial Analysis blind-vote leaderboards, prices from fal's model pages:

| Role | Current | Challenger | Board rank | Price on fal |
|---|---|---|---|---|
| Image → video, start→end | Kling v2.5 Turbo Pro — `fal-ai/kling-video/v2.5-turbo/pro/image-to-video` | MiniMax H3 Max — `minimax/h3-max/image-to-video` | Image-to-video board: H3 Max #1 (1195); Kling 3.0 Pro #20 (1055), v2.5 not listed | H3 Max ~$0.08/s at 768p (~$0.40 per 5 s) |

Runners-up worth one optional column if cheap to add: **Muse Image**
(`meta/muse-image/edit`, $0.01/img, edit board #7) as a draft tier, and plain
**MiniMax H3** (`minimax/h3/image-to-video`, $0.05/s at 480p) as a video draft
tier. Skipped on purpose: GPT Image 2.5 max ($0.21/img, top quality, too
expensive as a default), MAI-Image-2.6 (not on fal), Seedance 2.0 (~$0.30/s
on fal), and Veo 3.1 Lite start→end (fal deprecates it on 2026-10-22).

## Part A — image: done 2026-10-09

Nano Banana 2.1 beat Pro in Xander's blind ranking (40 vs 26 pts) and replaced
it as the `nano_banana` tool. Precedent for Part B (confirm with Xander):
the tool got a **model-agnostic name** and the losing model was **removed, not
kept as a fallback** — so the video tool would become e.g. `image_to_video`.
Done — see [reference/hosted-models.md](../reference/hosted-models.md). (It
has since moved from FAL to Google direct; the harness still calls it via FAL.)

## Part B — video tasks (Kling v2.5 vs H3 Max)

Inputs: the Part A outputs (A4 pair, A5 still) and `test_imgs/creatures/dragon.jpg`
or `fire_horse.jpg` for B4. Extend `scripts/fal_shootout.py` (same
skip-if-exists + results.json + contact sheet) rather than a second script.

Endpoint facts (from the schema pages, not yet exercised; re-read
`https://fal.ai/models/<id>/api` before coding):

- **H3 Max:** `prompt` (required), `image_url` (first frame),
  `end_image_url` (last frame; works alone too), `duration` (float, default 5;
  output can run ~0.7 s longer than requested, which matters for loops),
  `resolution` `480P`/`768P`/`1080P` (1080P is upscaled from 768P),
  `prompt_expansion_mode` `disabled`/`balanced`/`quality`, `seed`,
  `target_audio_url` (ignore). No `negative_prompt` / `cfg_scale`. The output
  aspect ratio follows the input image.
- **Kling v2.5 baseline:** the args the current tool sends: `image_url`,
  `tail_image_url`, `duration` `"5"`/`"10"`, `negative_prompt`, `cfg_scale`.

5 s, audio off, **1 sample per model per task** (2 if budget allows). H3 Max
at 768P; Kling at its default.

| # | Task | Inputs | What wins |
|---|---|---|---|
| B1 | Start→end interpolation | A4 pair (from the Part A winner) | smooth, plausible motion that lands exactly on the end frame |
| B2 | Animate the character on the surface | A5 output (start frame only) | the subject moves, **the camera and surface do not** |
| B3 | Seamless loop | the same image as start and end | no visible jump at the loop point (check H3's extra ~0.7 s) |
| B4 | Creative direction | dragon/fire_horse + an ambitious motion prompt | prompt adherence and motion quality |

**WZRD-specific check for every video:** run it through
`subtract_background_video` against its start frame. Camera drift or
global flicker shows up as noise across the whole frame and makes a clip
unusable for projection however good it looks raw. Put the subtracted clip
next to the raw clip on the contact sheet.

## Budget and run order (Part B)

About 8 clips, **$3–5**. Smoke first (one call per endpoint) to confirm arg
names and output shapes → show Xander → full run **only with his OK** (same
rule as TextureFlow's paid matrix). Contact-sheet cells for video: `<video
autoplay loop muted>` raw next to the `subtract_background_video` result.

## After the video verdict

- Swap the video endpoint in `wzrd_mcp/fal_tools.py` and rename
  `kling_v25_image_to_video` to a model-agnostic name. The rename touches
  `tools_config.json`, the skills that name it (`grep -rn kling wzrd_mcp/skills`:
  `video-prompting` — whose arg names (`image_url`/`tail_image_url`) change too —
  and `projection-story`), and `docs/reference/mcp-server.md`. Prompting craft
  for the new model goes in skill `video-prompting`, not the tool docstring.
- Add the video section to [reference/hosted-models.md](../reference/hosted-models.md),
  then retire this plan to `docs/finished/`.

## Open questions

- Add a draft tier (Muse Image, H3 at 480P) as separate tools, or as a
  `quality` arg on the main tool?
- Is it worth adding Kling v3 Standard (the tool exists but is disabled) as a
  third video column, to separate "Kling is old" from "Kling is worse"?
