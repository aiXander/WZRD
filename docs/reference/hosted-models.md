# Hosted models — which remote models the MCP calls, and why

Two vendors, one module each. Toggles/timeouts: `tools_config.json`; how
the tools register and how their prompting craft is split out into skills:
[mcp-server.md](mcp-server.md).

| Tool | Model | Module · key | Status |
|---|---|---|---|
| `nano_banana` | Nano Banana 2.1 (`gemini-nano-banana-2.1`), Google direct | `wzrd_mcp/google_tools.py` · `GEMINI_API_KEY` (fallback `VERTEX_API_KEY`) | default image tool |
| `kling_v25_image_to_video` | Kling v2.5 Turbo Pro | `wzrd_mcp/fal_tools.py` · `FAL_KEY` | default video tool; challenger pending (see below) |
| `kling_v3_image_to_video` | Kling v3 Standard | `wzrd_mcp/fal_tools.py` · `FAL_KEY` | disabled in `tools_config.json` |

Shared: `wzrd_mcp/_keepalive.py` (`with_keepalive` — pings the MCP client with
progress every 15 s while a remote call runs, returns the moment it finishes).

## Image: Nano Banana 2.1 on Google (since 2026-10-09)

**Why this model.** Replaced Nano Banana Pro (`nano_banana_pro`, removed) after
a blind shootout on WZRD-shaped tasks: Xander ranked 4 images per row
(2 models × 2 seeds) over 11 rows; 2.1 scored **40 pts, avg rank 2.18, 8 rows
won** vs Pro's **26 pts, 2.82, 3 won**. Tied on plain compositing and new
poses; 2.1 clearly better at multi-frame story continuity, start/end keyframe
pairs, and **leaving a darkened projection surface untouched** while adding a
subject (the WZRD-critical case). ~4× cheaper and ~2× faster than Pro.

**Why Google direct, not FAL.** Same model, but no upload step (refs go inline
as bytes), no queue polling, photos go to one vendor instead of two, and no
FAL markup (+17.6 %). Probed end to end (`scripts/google_nb_probe.py`,
outputs in `projects/google-nb-tests/`, gitignored): equal quality on the
kasbah case (share of wall pixels changed: FAL 0.12, Google 0.12–0.16).

**How the tool works** (`google_tools.py`):
- `_client()` — Gemini API client if `GEMINI_API_KEY` is set, else Vertex
  (`VERTEX_API_KEY`, project `GOOGLE_CLOUD_PROJECT` defaulting to
  `gen-lang-client-0837456283`, `location="global"`). Keys are passed
  explicitly: the SDK's env auto-detection would let `GOOGLE_API_KEY` win.
  Transport retries (408/429/5xx, httpx errors) are the SDK's own
  `HttpRetryOptions`.
- `_contents()` — per ref, a `"image_N:"` text part (1-based) then the image
  bytes; prompt last. Refs > 2048 px or not JPEG/PNG/WebP are shrunk and
  re-encoded to JPEG in memory, never on disk.
- `_final_image()` — last image part **not** flagged `thought`. No image →
  `_NoImage(reason)`; content-filter reasons retry 2× immediately.
- `num_images` = N concurrent single-image calls; partial failures are
  reported in `info.failed`. Output extension follows the mime type (Gemini
  `.jpg`, Vertex `.png`; never transcoded). `info.est_usd` is computed from
  `usage_metadata` at list price ($1.50/M input, $7.50/M text+thought, $30/M
  image out) — Vertex is priced the same here, unverified.

**Output size (measured 2026-10-10)** — the geometry every WZRD surface
pipeline depends on:
- An **explicit `aspect_ratio`** returns that aspect's native size: 1K =
  1:1 1024×1024 · 2:3 848×1264 · 3:2 1264×848 · 3:4 896×1200 · 4:3 1200×896
  · 4:5 928×1152 · 5:4 1152×928 · 9:16 768×1376 · **16:9 1376×768** · 21:9
  1584×672; 2K/4K are exactly ×2/×4. A reference of another aspect is
  **cover-cropped** to it — a 16:9 surface loses ≈3 px top and bottom
  (1.792 vs 1.778). The table is `wzrd.utils.NANO_BANANA_SIZES`.
- **`"auto"` follows image_1's own aspect**, not the nearest native one: a
  600×1064 region crop came back 768×1362. That is why the tool defaults to
  `"auto"` whenever refs are given (a canvas surface stays the canvas, a
  region crop stays its box) and to the project canvas's aspect for
  text-to-image.
- Either way, a result that doesn't match its reference's size is mapped
  back with `wzrd.utils.uncrop_to_size` (uniform scale + centred bands) —
  **never stretched**: stretching 1376×768 onto 1920×1080 was a constant
  0.78 % vertical drift (≈ 4 px at the frame edges) in every surface edit.

**Traps (from the probe):**
- **Vertex returns draft images** at `medium`: an extra image flagged
  `thought=True` before the final, sometimes of a different scene. Gemini
  returned none. Hence "skip thought parts, keep the last image".
- **Vertex client:** no `location` → 404 (defaults to `us-central1`);
  `location="global"` without `project` → 403. Interactions API 404s on
  Vertex, so the tool uses `models.generate_content` (works on both).
- **Keep the `image_N:` labels 1-based and explicit** — the model's own text
  numbers inputs `image_0`, `image_1`.
- **Refs:** ≤14 enforced (FAL parity); the API takes more, but quality
  collapses past ~10 (duplicated/dropped subjects).
- **`seed` nudges, doesn't reproduce** (same seed → same composition family,
  different image).
- **Chained edits drift.** Feeding earlier outputs back as references (a
  frame-by-frame story) loses the scene by frame 3–4. For sequences, pin the
  scene with the original photo as a reference every time; keep chains short.
- Payload size costs latency, not failure (30 MB inline: accepted, 34 s).

**Setup:** both keys live in Google Cloud project
`gen-lang-client-0837456283` (created by AI Studio, billing linked; keys under
Console → APIs & Services → Credentials) and only in the repo's gitignored
`.env`. The Vertex key is bound to service account `vertex-express@…`,
restricted to "Agent Platform API" (Vertex AI's new name). As of 2026-10-09
there is **no budget alert** on the project (the only cap is the rate tier's
$10 per 10 min) and the tier's RPM wasn't checked (4 concurrent calls were
fine).

**Numbers (2026-10-09):** latency at 1K with 2 refs: `minimal` ~13 s,
`medium` ~18–30 s; 2K `medium` ~40 s; 4K t2i `minimal` ~24 s. Price per
image: $0.034 / $0.050 / $0.113 at 1K / 2K / 4K, plus ~$0.002 per input image;
a 2-ref 1K `medium` call ≈ $0.045. Batch API is half price (≤24 h) — not
wired up. Search grounding (`enable_web_search`) works with image output.

**Open questions** (not blocking):
- **Default `thinking_level`:** `medium` won the shootout; `minimal` tied on
  the kasbah case and is ~30 % faster. Flip once Xander has seen more
  side-by-sides? Google's model card (human Elo) says thinking matters most for
  general editing (1026 vs 980) and barely for character consistency or mask
  edits; it doesn't say whether `minimal` is its "no thinking" mode.
- **Mask-guided edits:** the card's biggest lead over Pro is mask/ink editing.
  Passing a WZRD surface mask as a reference ("only change inside the white
  area") might enforce the untouched-surface rule harder. Untried.
- **`nano_banana_batch`** at half price — only if someone needs >~50 images in
  one go.

## Video: Kling on FAL

`kling_v3_image_to_video` only knows 16:9 / 9:16 / 1:1; its `aspect_ratio`
defaults to the one nearest the start frame (a tall region keyframe stays
tall) instead of a fixed 16:9.

`fal_tools.py` plumbing: `_fal_subscribe` (retry/backoff + keepalive),
`_to_fal_url` (uploads local inputs to FAL's CDN just in time — FAL's workers
can't read this disk), `_run_fal_tool` (downloads results into the project's
`generated/`).

## Re-running the comparisons

- **FAL shootout:** `scripts/fal_shootout.py` (`--smoke` first; `--sheet-only`
  rebuilds the page). Declarative step list, skip-if-exists, `results.json` per
  call, blind ranking page at `projects/fal-shootout/index.html`. A new
  contender is one entry in `MODELS`. It still calls Nano Banana via FAL
  (imports `_fal_subscribe`/`_to_fal_url`), which is fine for comparisons.
- **Google probe:** `uv run --no-project --with google-genai --with pillow
  --with numpy python scripts/google_nb_probe.py` (~18 calls, ~$0.75).
