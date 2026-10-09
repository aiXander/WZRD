# Nano Banana 2.1: call Google directly instead of through FAL

> **Goal:** the MCP's `nano_banana` tool stops calling FAL
> (`google/nano-banana-2.1[/edit]` in `wzrd_mcp/fal_tools.py`) and calls
> Google's API with the `google-genai` SDK. Same model, same tool name, same
> args, so callers and skills (`wzrd_mcp/skills/*`) don't change.
>
> **Why:** going direct removes the upload step (`_to_fal_url` pushes every
> local input to FAL's CDN, while Google takes the bytes inline in the
> request) and the 15 s queue-poll steps of `_fal_subscribe`. It also sends
> photos to one vendor instead of two and saves FAL's flat **+17.6 % token
> markup** ($0.034 vs $0.040 per 1K image).
>
> **Status (2026-10-09):** credentials exist, and both Google backends have
> been probed end to end: 18 calls, about $0.75, every question below answered
> with evidence. **Code not started; this is one sprint.** Kling stays on FAL,
> so `FAL_KEY` stays.

## Sprint: open work, in order

1. **Xander** does the two remaining manual steps (budget alert, rate-limit
   tier; table below). They don't block the build.
2. **Agent** builds `wzrd_mcp/google_tools.py` per the spec below and deletes
   the FAL image path.
3. **Agent** runs the verification list. **Xander** eyeballs the kasbah case,
   the one that matters for WZRD.
4. **Agent** does the doc pass. Promote into `docs/reference/fal-tools.md`
   (rename it, e.g. `hosted-models.md`, since images no longer go via FAL).
   Update `CLAUDE.md` (the `wzrd_mcp/` file list; "FAL_KEY is the only key a
   local session needs" becomes false) and `.env.example`. Then retire this
   plan. Other agents edit `CLAUDE.md` / `.env.example` / `server.py`
   concurrently: re-read before editing.

## Manual steps (Xander)

Everything lives in Google Cloud project **`gen-lang-client-0837456283`**.
AI Studio created it; billing is linked and paid calls work. Keys are managed
under Console → *APIs & Services → Credentials*.

| # | Step | Status |
|---|---|---|
| 1 | **Gemini API key** ("Generative Language API Key", 2024-10-08, restricted to Gemini API) | ✅ in WZRD `.env` as `GEMINI_API_KEY` |
| 2 | **Vertex key** as a fallback ("API key 2", 2026-10-09). Bound to service account `vertex-express@gen-lang-client-0837456283.iam.gserviceaccount.com`, restricted to "Agent Platform API" (Vertex AI's new name). | ✅ in WZRD `.env` as `VERTEX_API_KEY` |
| 3 | **Budget alert:** Console → Billing → Budgets & alerts, e.g. €25/month on this project, alerts at 50/90/100 %. Otherwise there is no spend cap except the tier's $10 per 10 min. | ☐ |
| 4 | **Rate-limit tier:** aistudio.google.com/rate-limit with this project selected. Note the RPM for `gemini-nano-banana-2.1`. Only matters if it's under ~10. Four concurrent calls in the probe were fine. | ☐ |
| 5 | *(Optional)* Both keys were pasted into a Claude Code session, so they sit in a local transcript. Rotate them if that bothers you (Credentials → key → Regenerate), then update `.env`. | ☐ optional |

The keys live **only** in the repo's gitignored `.env` (`.gitignore:27`).
`python -m wzrd_mcp` loads it through `__main__._load_dotenv` for both the
HTTP and `--stdio` transports. `wzrd_mcp/__init__.py` also loads `~/.eve` (old
Eden secrets), which has no `GOOGLE_*`/`GEMINI_*` API key, so the two can't
clash.

## What the probe proved

Re-runnable: `uv run --no-project --with google-genai --with pillow --with
numpy python scripts/google_nb_probe.py`. Every image (drafts included), the
FAL baselines for the same prompts, `results.json` and a viewer
`index.html` are in `projects/google-nb-tests/` (gitignored).

**Backends** (`google-genai` 2.29.0, model ID `gemini-nano-banana-2.1` on both):

| | Gemini API (`GEMINI_API_KEY`) | Vertex (`VERTEX_API_KEY`) |
|---|---|---|
| Client | `genai.Client(api_key=K)` | `genai.Client(vertexai=True, api_key=K, project="gen-lang-client-0837456283", location="global")` |
| Traps | none | No `location`: defaults to `us-central1` → **404**. `location="global"` without `project` → **403 "project None"**. |
| `models.generate_content` | ✅ | ✅ |
| Interactions API (`client.interactions`) | ✅ | ❌ 404 |
| Output | JPEG | PNG |
| Draft images | none seen, even at `high` | **yes at `medium`:** an extra image part flagged `thought=True` |

**Behaviour** (Gemini API unless noted):

| Question | Answer |
|---|---|
| Quality vs FAL on the WZRD-critical case (darkened kasbah + creature, shootout task A5) | **Equal.** Share of surface pixels changed: FAL 0.120 / 0.123; Google `minimal` 0.123, `medium` 0.157, `high` 0.150 (bigger creature, same untouched wall). See `_sheet_kasbah.jpg`. |
| Latency at 1K, 2 refs | `minimal` **13 s** · `medium` 18 s · `high` 18 s. FAL took 15–30 s on the same task. |
| Other latencies | 2K with 2 refs at `medium`: 42 s. 4K text-to-image at `minimal`: 24 s (5504×3072). |
| **Draft images** | At `medium`, Vertex returns a draft image, then the final. The first smoke test's draft was a *different building on a white canvas*; the final was correct. Only the final is billed as image output. Gemini returned none. **Rule: skip `part.thought` parts, keep the last image.** |
| `image_N` references | Interleaving a `"image_N:"` text part before each image works (A1/A5 compose correctly). Trap: the model's own text calls inputs `image_0.png`, `image_1.png` (zero-based), so keep the explicit 1-based labels. |
| Reference count | The API accepts 15 refs with no error, but **quality collapses past ~10**: the 14-ref grid duplicated creatures and dropped others. Enforce ≤14 (parity with FAL) and say "best ≤10" in the docstring. |
| `seed` | Accepted but **not reproducible:** same seed gives the same composition family, different image (mean pixel diff 33 vs 61 for another seed). |
| Search grounding | ✅ `tools=[Tool(google_search=GoogleSearch())]` works with image output: it ran 3 queries and drew today's real Ghent weather. |
| Payload size | A 30 MB inline request was accepted, but took 34 s. Size costs latency, not failure. |
| Tokens per call | Input image ≈ 1,100 tokens. Output image: 1K = 1,120, 2K = 1,680 (+text), 4K ≈ 3,800. Gemini also bills ~300 output-text tokens per call (~$0.002). `medium` thinking ≈ 700 thought tokens (~$0.005). |

**Price per image** (Google list, standard tier):

| | 1K | 2K | 4K |
|---|---|---|---|
| Google direct | $0.034 | $0.050 | $0.113 |
| Google Batch API (≤24 h) | $0.017 | $0.025 | $0.057 |
| FAL (before) | $0.040 | $0.059 | $0.134 |

Add ~$0.002 per input image. A realistic 2-ref 1K call at `medium` is
**≈ $0.044** (FAL ≈ $0.052). Vertex's 2.1 price was not checked, which is one
more reason to prefer the Gemini key.

## Decisions (settled; don't re-litigate)

- **Gemini API key first; Vertex only if `GEMINI_API_KEY` is unset.** It's
  simpler and documented-price, and returns no drafts. The fallback costs
  three lines.
- **Use `client.aio.models.generate_content`, not the Interactions API.**
  Interactions 404s on Vertex; generate_content works on both. Google calls it
  "legacy, fully supported". It's async-native, so no `to_thread`. The proven
  template is `~/Documents/nn&/nano_banana/run_gcp.py` (Nano Banana Pro on
  Vertex since Feb 2026), outside this repo.
- **Delete the FAL image path; no flag, no fallback.** Rollback is a
  `git revert`. FAL plumbing (`_fal_subscribe`, `_run_fal_tool`,
  `_to_fal_url`) stays for Kling and `scripts/fal_shootout.py`.
- **Default `thinking_level="medium"`:** that's what won the 2026-10-09
  shootout on FAL. Expose it as a tool arg. `minimal` matched quality on A5
  and is ~30 % faster, so it's a fine caller choice, and possibly a future
  default once Xander has seen more.

## Implementation spec

**Shape:** one new module of about 150 lines, one small shared helper, one
deletion.

- **`wzrd_mcp/google_tools.py` (new):** the `nano_banana` tool plus three
  private helpers. Register it in `server.py` next to
  `from . import fal_tools` (≈ line 128). Move the shootout comment block over
  with it.
  - `_client()`: `functools.cache`'d. If `GEMINI_API_KEY` is set, use
    `genai.Client(api_key=…)`. Else if `VERTEX_API_KEY` is set, use the Vertex
    client (project from `GOOGLE_CLOUD_PROJECT`, default
    `gen-lang-client-0837456283`, `location="global"`). Else raise `ToolError`
    naming both vars. **Always pass keys explicitly:** the SDK's own env
    auto-detection lets `GOOGLE_API_KEY` silently win.
  - `_contents(prompt, refs)`: per ref *i* (1-based), a `Part(text=f"image_{i}:")`
    then `Part(inline_data=Blob(mime, bytes))`; the prompt goes last. Refs
    resolve through `file_io.resolve_input_async`, so URLs, base64 and local
    paths all work. Re-encode any ref whose long side is > 2048 px to JPEG
    q≈92 in memory (latency; billing is per image regardless). Never write
    the re-encoded copy to disk.
  - `_final_image(resp) -> (bytes, mime)`: the last `inline_data` part with
    `not part.thought`. If there is none, raise `_NoImage(reason)` carrying
    `finish_reason`, `prompt_feedback` and any model text.
- **The tool:** signature unchanged, plus `thinking_level: str = "medium"`.
  `num_images` = N concurrent single-image calls (`asyncio.gather`); one
  image per response is the reliable contract. Save each image via
  `project.output_path("nano_banana", ext)`, with `ext` taken from the mime
  type (Gemini `.jpg`, Vertex `.png`; never transcode). Return today's shape,
  `{"output_images": [...], "info": {...}}`. Add `backend`,
  `thinking_level` and **`est_usd`** to `info`, computed from
  `usage_metadata` (input $1.50/M, text+thought $7.50/M, image $30/M; the
  image share = `candidates_tokens_details` with modality IMAGE). Agents then
  see the cost of what they spend.
- **Arg mapping:**

  | Tool arg | Request |
  |---|---|
  | `resolution` | `ImageConfig(image_size=…)`. `"1K"/"2K"/"4K"`; uppercase K is required. |
  | `aspect_ratio` | `ImageConfig(aspect_ratio=…)`. Omit for `"auto"`. With refs and no ratio, the model keeps the first ref's aspect. |
  | `image_urls` | `_contents`. Reject more than 14; the docstring says quality drops past ~10. |
  | `enable_web_search` | `tools=[types.Tool(google_search=types.GoogleSearch())]` |
  | `seed` ≥ 0 | `GenerateContentConfig(seed=…)`. Docstring: "nudges, doesn't reproduce". |
  | `thinking_level` | `ThinkingConfig(thinking_level=…)` |
  | always | `response_modalities=["TEXT", "IMAGE"]` |

- **Errors and retry:**
  - Transport errors (429, 5xx, timeouts): first check whether the installed
    SDK has `types.HttpOptions(retry_options=types.HttpRetryOptions(...))` and
    use it on the client if so. Otherwise use a small backoff loop (1 s / 2 s
    / 4 s, 3 retries).
  - Other 4xx fail at once with `e.message` (401/403 → "check
    GEMINI_API_KEY").
  - `_NoImage` with a content-filter reason (`SAFETY`, `PROHIBITED_CONTENT`,
    `IMAGE_PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`): retry up to 2×
    immediately (these are often flaky, per `run_gcp.py`), then `ToolError`
    with the reason and the model's text.
  - FAL's `safety_tolerance` has no direct equivalent; use Google's defaults.
- **Keepalive (`wzrd_mcp/_keepalive.py`, new):** lift the 15 s
  `ctx.report_progress` loop out of `_fal_subscribe` into
  `async def with_keepalive(awaitable, ctx, what) -> result`, then use it in
  both `_fal_subscribe` and the new tool. 2K at `medium` took 42 s, and 4K at
  `high` will be longer.
- **Delete from `wzrd_mcp/fal_tools.py`:** `nano_banana`,
  `_NANO_BANANA_EDIT`, `_NANO_BANANA_T2I`, and any image mention in the
  module docstring.
- **Dependencies:** add `google-genai>=2.29` to the `mcp` extra in
  `pyproject.toml`, and to `pip_install` in the parked
  `wzrd_mcp/modal_app.py` for parity.
- **Text:** in the `server.py` instructions (≈ line 99) and the tool
  docstring, change "via FAL" to "via Google (Gemini API)".
  `tools_config.json` is unchanged (timeout 180 s).

**Verified minimal call** (Gemini key, from the probe):

```python
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = await client.aio.models.generate_content(
    model="gemini-nano-banana-2.1",
    contents=[types.Part(text="image_1:"),
              types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img)),
              types.Part(text=prompt)],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="16:9", image_size="1K"),
        thinking_config=types.ThinkingConfig(thinking_level="medium"),
    ),
)
```

## Verification (≈ $0.30)

Run through the real MCP (`python -m wzrd_mcp`, `set_project smoke-google`):

1. Text-to-image, 1K, `1:1` → one `.jpg`, 1024×1024, `info.est_usd` ≈ $0.04.
2. Edit with 2 local refs (shootout A1 prompt), `16:9`, 2K → 2752×1536.
3. **Kasbah case** (shootout A5 prompt, refs
   `test_imgs/morocco/kasbah/input_raw_darkened.jpg` +
   `test_imgs/stories/story_01/character.jpg`). **Xander compares** it with
   `projects/google-nb-tests/_sheet_kasbah.jpg`.
4. `num_images=2` → two files, no drafts saved.
5. With `GEMINI_API_KEY` unset → the Vertex fallback works and returns a
   `.png` without saving a draft.
6. 15 refs → clean `ToolError` before any API call.
7. `ruff check wzrd_mcp/`.

## Open questions (not blocking)

- **Batch API:** a `nano_banana_batch` tool at half price for bulk offline
  generation (≤24 h turnaround). Only if someone needs more than ~50 images
  in one go.
- **Default thinking level:** flip to `minimal` once Xander has seen more
  side-by-sides? The evidence so far is one task, and it was a tie. Against
  flipping: Google's model card
  (deepmind.google/models/model-cards/nano-banana-2-1/, human side-by-side
  Elo) shows thinking matters most for general editing (1026 thinking vs 980
  no-thinking; Pro 939) and barely for character consistency (1028 vs 1021)
  or mask edits (1049 vs 1042). The card doesn't say whether `minimal` is its
  "no thinking" mode.
- **Mask-guided edits:** the card's largest lead over Pro is mask/ink-based
  editing (1049 vs 927). WZRD already has surface masks, so passing one as a
  reference ("only change inside the white area") might enforce the
  untouched-surface rule more firmly than the prompt alone. Try it on the
  kasbah case once the tool exists.
- **Docstring hints from the card's known limitations:** occasional
  left/right confusion (spell out placement), small text blurry at 1K (use
  2K+ when a frame has text), and "possible slowness or timeouts" (the
  retry-on-timeout rule above covers it).
