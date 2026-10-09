# VJ agent — missing tools (analysis + proposals)

> **Status: proposal for Xander's review. Nothing here is built.** Written
> 2026-10-09 after reading every MCP tool, skill and the product spec
> (`docs/reference/user_design_spec.md`). Feedback wanted on the questions in
> §6; once the tiers are agreed, this becomes the build plan.

## TL;DR

The agent can already **make** everything (keyframes, Kling segments,
TextureFlow loops, region masks, layer packs, live WGSL). What it lacks is
**measurement, repair and assembly**:

1. **It has no sensors except its own eyes.** Every quality check is "call
   `view_media`, look at a contact sheet". That costs ~1–2k tokens per image,
   is subjective, and misses the drift that matters most: a 6 px background
   shift is invisible on a 768 px thumbnail but becomes a bright outline of the
   whole building after background subtraction.
2. **When something is slightly off, the only fix is regenerating** (paid, slow,
   no guarantee). A clip that drifts 1% could be warped back onto the surface
   in two seconds locally.
3. **The pieces never come together.** Nothing composites several region
   videos into one projector file, nothing trims/loops/fades, nothing pulls a
   still out of a video, and the engine can't play an mp4 yet. Story output
   and TextureFlow output have no route onto the projector *through WZRD*.
4. **It forgets.** Every asset is a timestamped path that lives only in the
   agent's context. After a long day (or a context compaction) "the good
   version of keyframe 3" is gone.

### Top picks

| # | Tool | One-liner | Effort |
|---|---|---|---|
| 1 | **`check_frames`** | Your drift tool: feed images/videos, get spatial / colour / structure / temporal / projector-safety stats + pass-warn-fail + a fix hint. No pixels to the LLM. | ~1–1.5 days |
| 2 | **`stabilize_to_surface`** | The repair partner: warp frames back onto the surface, colour-match the background, optionally re-paste the original background pixels. | ~1 day |
| 3 | **`composite_videos`** | N videos (region renders, story, textures) → one additive projector file, with per-input gain, offset, mask, loop. | ~0.5 day |
| 4 | **`edit_video`** | Trim / speed / reverse / ping-pong / fade / loop-to-length / crossfade-loop / conform to projector size. | ~0.5 day |
| 5 | **`extract_frames`** | First / last / at-time / N stills out of a video (chain segments, repair end frames, feed TextureFlow). | ~2 h |
| 6 | **Asset memory** (`list_assets`, `mark_asset` + automatic provenance sidecars) | What exists, what made it (prompt, refs, seed, cost), what's approved. | ~1 day |
| 7 | **`import_photos`** | Phone photos (HEIC, EXIF rotation, AirDrop'd into Downloads) → project. | ~2 h |

Tier 2 (§3.2): text-prompted segmentation (SAM), surface report with numbered
zones, frame interpolation/upscale, pack re-segmentation reconcile. Tier 3
(§3.3): audio/beat analysis, closed-loop camera calibration, stopgap player.

---

## 1. What the agent is today

**Purpose.** A creative collaborator for a VJ standing in front of a physical
surface: it reads the surface, proposes stories, generates content that only
*adds light* to that exact surface (the dark stays dark), and authors live
WGSL effects on the realtime engine's design leg. Two halves:

- **Offline content** (Python, files in a project folder): surface prep →
  keyframes → video segments / TextureFlow → background subtraction →
  simulated view.
- **Live engine** (render-core over WebSocket): bind effects to named regions,
  write shaders, label/group layers. The operator promotes to the projector.

**Enabled tools (24), by stage:**

| Stage | Tools |
|---|---|
| Session | `set_project`, `load_skill` |
| Surface | `prepare_surface` (detect+align+darken), *(`detect_*`, `align_images`, `darken_surface`, `capture_camera_snapshot` exist but are disabled)* |
| Generate | `nano_banana` (images, Google), `kling_v25_image_to_video` (FAL), `texture_flow` + `textureflow_warm` (own Modal GPU) |
| Regions | `extract_color_regions`, `reproject_video`, `build_layerpack` |
| Post | `subtract_background_video`, `concat_videos`, `simulate_view` *(`subtract_background_frame` disabled)* |
| Look | `view_media` (thumbnails + video contact sheets, max 6 items) |
| Live engine | `get_scene_context`, `upsert_binding`, `remove_binding`, `upsert_effect`, `remove_effect`, `set_groups`, `set_labels`, `set_scene`, `validate_wgsl`, `get_preview` |

Craft lives in 7 skills (`projection-story`, `surface-prep`,
`region-animation`, `live-engine` workflows; `image-prompting`,
`video-prompting`, `textureflow` guides). Every "is this OK?" step in those
skills is currently `view_media`.

---

## 2. The journey: a VJ arrives on site

Imagined end to end: Xander arrives at a venue at 16:00, show at 22:00. A
facade with six windows and a door, a projector on a stand, a laptop, a phone.
For each step: what happens today → where it hurts → what would help.

### 2.0 Arrive

`set_project("venue-name")`, `textureflow_warm(180)`. Fine. **Missing:** a
running spend counter. By 22:00 the agent will have made ~80 `nano_banana`
calls, ~15 Kling segments and a dozen TextureFlow renders; nobody knows the
bill. `nano_banana` returns `est_usd`, Kling and TextureFlow don't. → folds
into **asset memory** (§3.1 #6).

### 2.1 Capture the surface

Xander photographs the facade with his phone and AirDrops the shots. They
land in `~/Downloads` as **HEIC**, often **portrait-rotated via EXIF**.
`resolve_input` opens files with PIL, which reads neither HEIC (no
`pillow-heif` dependency) nor applies EXIF rotation. The agent has to find the
files, convert them by hand, and may silently work on a sideways photo.
→ **`import_photos`** (§3.1 #7).

`prepare_surface` wants a *night* photo with the projector showing a white
frame. At 16:00 it's daylight, so in practice: one day photo now, the
night/white-frame photo after sunset. Day-only prep works (darken only) but
the photo is from where Xander stood, not from the projector lens. The
alignment layer in the engine (the Align route: drag corners until the light
lands right) absorbs this later, so this is fine, but the skill should say so.

### 2.2 Read the surface and pitch stories

The agent looks at the surface photo and proposes 2–3 story directions using
its features (windows → portals, door → entrance). That's one image, cheap
enough. What's missing is **shared, precise vocabulary**: when the agent writes
a keyframe prompt it says "the second window from the left", and the model
sometimes mirrors or miscounts. A numbered overlay ("zone 3 = window,
x 61–72%, y 22–41%") would give the agent, Xander and the prompt the same
coordinates. → **`surface_report`** (§3.2 #9), tier 2.

### 2.3 Regions (for per-window animation and the live engine)

Today: `nano_banana` draws a flat-colour segmentation map → 
`extract_color_regions` → masks. Pain points:

- **The segmentation map is a generated image, so it drifts too**: borders
  are shifted or wobbly relative to the real window edges. Checking is by eye.
  `check_frames` (map vs surface) catches the shift; a proper segmenter fixes
  the root cause. → **`segment_surface`** (text-prompted SAM, §3.2 #8).
- **Naming is tedious**: the agent gets 7 anonymous masks and has to look at
  each to call it `window_left`. A numbered overlay image returned by
  `extract_color_regions` (small change, not a new tool) lets Xander say
  "1–6 are windows, 7 is the door" in one line.
- **Re-shoots break everything**: if the projector moves tomorrow and the
  surface is re-segmented, the new pack gets fresh ids and every engine scene
  targeting `window_left` breaks. The engine reference already names this as a
  requirement (re-import a new segmentation onto an existing pack's ids by
  overlap). → **`reconcile_layerpack`** (§3.2 #11).

### 2.4 Keyframes

Load `image-prompting`, then `nano_banana` with the surface as `image_1`,
2+ variations per keyframe, `view_media` each batch. For 5 keyframes × 3
variations that's ~15 images viewed, ~20–30k tokens, and the agent still can't
see:

- a **4–15 px shift / 1–2% zoom** of the whole background (the model re-draws
  the scene slightly off-grid; very common with edit models);
- a **slight exposure / white-balance change** across the whole frame;
- **re-rendered background texture** (bricks redrawn, same layout, different
  pixels), which subtraction turns into sparkle;
- **darkened background** (the model "removed" a lamp or shadowed a wall):
  additive light can't darken anything, so that part of the keyframe is
  physically unprojectable;
- **mirroring** (the skill already warns the model sometimes mirrors);
- **drift accumulating along a chain**, K1 → K2 → K3, which the skill fights
  only by re-pinning the surface as `image_1`.

→ **`check_frames`** ranks the variations by background fidelity without
looking; the agent only `view_media`s the survivors for the *creative*
judgement (pose, story, vibe), which is what eyes are for. Near-misses go
through **`stabilize_to_surface`** (`repin=true`) so every keyframe that
enters Kling has a pixel-identical background.

`check_frames` can also report **subject stats** per keyframe (where the added
light is, its area, its brightness) and the **delta between consecutive
keyframes**. The `projection-story` skill says "keep the visual delta between
keyframes modest; the video model struggles with large jumps", and today
nothing measures that. "Subject moves 48% of the frame width between K2 and
K3, Kling will likely morph" is a cheap, useful warning before spending money.

### 2.5 Segments (Kling)

One Kling call per keyframe pair, then `view_media` on each clip's contact
sheet. Failure modes the contact sheet misses or can't quantify:

- **Slow camera drift / zoom** (the #1 reason subtraction fills the whole
  frame with noise; the skill says so explicitly);
- **Global flicker / exposure breathing**;
- **The clip doesn't actually land on the end keyframe** (Kling treats
  `tail_image_url` as a suggestion; a mismatch shows up as a hitch at the
  concat seam);
- **Hard cuts** mid-clip;
- **Frozen clips** (almost no motion; happens with very similar keyframes).

→ `check_frames` in video mode answers all five with numbers and timestamps.
**`stabilize_to_surface`** rescues the drifting ones (warp every frame back,
colour-match the background), often instead of a $0.35–0.70 regeneration.
**`extract_frames`** gives the agent the actual last frame of a clip, either to
start the next segment from it (perfect continuity, no seam) or to see how
far it ended from the intended keyframe.

### 2.6 Assemble the story

`concat_videos` → `subtract_background_video` → `simulate_view`. Gaps:

- **No check of the final projector file.** Things that matter at night and
  that nobody measures: *haze* (dark pixels at 3–8/255 instead of 0 → a
  visible grey rectangle on the wall, which breaks the "dark stays dark"
  thesis), *how much of the frame is lit* (whole-frame noise = drift),
  *light landing outside the reflective surface* (sky, people's faces),
  *brightness* (dim content vanishes), and **photosensitive flashing** (more
  than 3 big flashes per second is a public-safety issue for an
  installation). → `check_frames` projector mode.
- **Seam quality** of the loop and between segments: a numeric seam score
  instead of hoping. → `check_frames` with `loop=true`.
- **No editing**: can't trim a bad first half-second, slow a segment down,
  ping-pong a short clip into a loop, fade in/out, or loop it to 2 minutes.
  → **`edit_video`**.

### 2.7 TextureFlow and regions

Per-region `texture_flow` with `diffusion_mask` → `reproject_video` per
region → **dead end**: you now have six full-canvas videos (black outside
their window) and no tool to combine them into the one file the projector
plays. Same when Xander wants "story on the door + slow texture on the facade".
→ **`composite_videos`** (additive blend, per-input gain/offset/mask, loops
shorter inputs). The `mask` input also covers "use this full-frame TextureFlow
render but only inside the windows".

TextureFlow output is designed to flow but not guaranteed to loop cleanly →
`check_frames(loop=true)` measures the seam; `edit_video(crossfade_loop_s=1)`
fixes it.

### 2.8 Getting it onto the projector

This is the biggest gap, and it isn't an MCP tool: **the engine can't play a
video file yet** (engine roadmap §5.9, *video layers*, open), and WGSL effects
can't sample a still image either. So today a finished story mp4 or TextureFlow
loop has no path onto the projector through WZRD: it would go through
Resolume, QuickTime fullscreen or similar, which bypasses the engine's
alignment warp and masters. See §4.

### 2.9 Showtime

The live-engine slice is in good shape (orient → bind → read verdict → offer
promote). Gaps the journey exposes:

- The agent can't hand **offline assets** to the live engine (§2.8).
- "Bring back the golden look from earlier" needs asset memory with
  approved/rejected state and notes (§3.1 #6).
- Shaders could do far more with **per-pack reference textures**: an edge map
  of the facade lets a WGSL effect run light along the actual brick lines, a
  depth map lets a sweep light the surface as if from the side. The pack
  already ships `references/` (surface, canny edges), but shaders can't read
  them. Engine-side, §4.

### 2.10 Pack up / next venue

Spec goal: "a pack I open next year projects within minutes". Asset provenance
(what prompt / refs / seed made each file) is what makes a show *re-makeable*,
not just re-playable: the same story re-rendered on a new surface means
re-running the same prompts with a new `image_1`.

---

## 3. Proposed tools

Design rules carried over from `docs/reference/mcp-server.md`: every tool goes
through `server.tool()` + a `tools_config.json` entry; local-first (paths in,
paths out); contract in the docstring, craft in skills; dict results become one
compact JSON block. New numeric code goes in `wzrd/` (Linux-portable, ffmpeg
codecs as parameters), MCP wrappers stay thin.

### 3.1 Tier 1 — build first

#### 1. `check_frames` — drift and quality stats, no pixels to the LLM

**Signature (sketch)**

```python
check_frames(
    items: list[str],              # images and/or videos, in order
    reference: str | None = None,  # the surface; default = first image / first frame
    expect_start: str | None = None,  # video: keyframe the clip should start on
    expect_end: str | None = None,    # video: keyframe the clip should end on
    loop: bool = False,            # also score the last→first seam
    surface_mask: str | None = None,  # masked surface: black = non-reflective (enables spill check)
    sample_fps: float = 8,         # video sampling rate
    report_images: bool = False,   # write a diff heatmap of the worst item/frame, return its path
) -> dict
```

Mode is auto-detected per item: **still** (keyframe, seg map), **clip**
(generated video of the surface), **projector** (mostly-black video/still,
i.e. subtraction output, TextureFlow-on-black, composite).

**What it measures**

| Group | Metric | Catches | How (OpenCV/numpy) |
|---|---|---|---|
| Spatial | `shift_px` / `shift_pct`, `scale`, `rotation_deg`, `max_corner_px` (one number: worst displacement of a frame corner) | background moved / zoomed / tilted | feature match + RANSAC homography (the subject is the outlier set), ECC sub-pixel refine — both already in `wzrd/align.py`; phase correlation for fast per-frame video tracking |
| Spatial | `mirrored`, `size_mismatch` (resolution / aspect differs from reference) | flipped output, wrong aspect | match against the flipped reference; compare dims |
| Colour | `bg_delta_e` (mean CIEDE2000 on background pixels), `exposure_shift_L`, `wb_shift_ab` | global brightness / white-balance change | Lab difference on pixels outside the added region; per-channel shift like `estimate_color_shift` in `wzrd/subtract_video.py` |
| Structure | `bg_ssim` (after alignment, background only) | texture re-rendered, details hallucinated | SSIM / edge-map overlap |
| Physics | `added_fraction`, `darkened_fraction` | how much light is added; how much got *darker* than the surface (unprojectable: light can't subtract) | signed luminance diff after alignment |
| Subject | `subject_bbox`, `subject_centroid`, `subject_area`, `subject_brightness` + `delta_prev` (centroid move, area ratio vs previous item) | subject too dark, too big, jumped too far between keyframes | connected components of the added region |
| Temporal (video) | `drift_trend` (linear px/frame) vs `jitter_px`, `flicker` (std of background luminance), `cuts_s` (spike in frame-to-frame diff), `motion` (mean optical flow; `frozen_spans_s`) | slow camera drift, breathing, hard cuts, static clips | per sampled frame series → summary, never the raw series |
| Boundaries | `start_match`, `end_match` (shift + ΔE vs `expect_start/end`), `seam_score` (boundary diff ÷ median frame diff; ≥3 = visible hitch) | Kling missing the tail keyframe, hitches at concat and loop points | first/last frame comparisons |
| Projector | `lit_fraction`, `haze` (share of pixels at 1–8/255), `black_floor`, `mean_lit_brightness`, `spill_fraction` (light where `surface_mask` is black), `flash_events_per_s_max` | grey rectangle on the wall, whole-frame noise, light on sky/people, dim content, photosensitivity risk (>3 large flashes/s) | luminance stats per sampled frame |

**Output (sketch)**, deliberately compact, ~1 KB per item:

```json
{
  "reference": ".../surfaces/prepare_surface_20261009-161203.webp",
  "verdict": "warn",
  "items": [
    {"path": ".../generated/nano_banana_k2_b.jpg", "kind": "still", "verdict": "fail",
     "shift_px": [14.2, -3.1], "shift_pct": 0.74, "scale": 1.012, "bg_delta_e": 4.8,
     "exposure_shift_L": 5.9, "bg_ssim": 0.93, "darkened_fraction": 0.01,
     "subject": {"centroid": [0.63, 0.41], "area": 0.07, "brightness": 0.78},
     "delta_prev": {"centroid_move": 0.12, "area_ratio": 1.3},
     "issues": ["background shifted 14 px right, 1.2% zoom", "exposure +6 L"],
     "fix": "stabilize_to_surface(repin=true) should fix both"},
    {"path": ".../generated/kling_k2_k3.mp4", "kind": "clip", "verdict": "warn",
     "camera": {"max_shift_pct": 0.9, "drift_trend": "right 0.15 px/frame", "jitter_px": 0.4},
     "color": {"max_bg_delta_e": 3.1, "flicker": 0.8},
     "end_match": {"shift_px": 9.8, "delta_e": 6.4},
     "motion": {"mean_flow_px": 2.1, "frozen_spans_s": []}, "cuts_s": [],
     "issues": ["slow drift right, 17 px by the end", "ends 10 px off keyframe K3"],
     "fix": "stabilize_to_surface; or start the next segment from extract_frames(last)"}
  ]
}
```

Each item gets a verdict, plain-language `issues`, and a `fix` hint that names
the next tool (stabilize vs regenerate: shift and colour are fixable,
`bg_ssim` low means the background was redrawn and only regeneration helps).
`report_images=true` writes one diff heatmap of the worst offender so the
agent can `view_media` *one* image when it does need to look.

**Thresholds** start as guesses and get calibrated once on real outputs we
already have: `test_imgs/stories/` (keyframes + Kling clips) and the
`projects/fal-shootout/` runs. Starting points: shift < 0.3% of width pass,
0.3–1% warn, > 1% fail; background ΔE < 2 pass, 2–5 warn, > 5 fail; seam
score ≥ 3 fail; haze > 2% warn; flashes > 3/s fail.

**Cost:** local, < 2 s per image and < 10 s per 5 s clip on the M2 at
`sample_fps=8`. Replaces ~1.5k tokens per viewed image with ~250 tokens of
JSON, and catches what eyes miss.

**Where:** `wzrd/frame_qc.py` (+ CLI) and a wrapper in
`wzrd_mcp/media_tools.py` next to `view_media`.

#### 2. `stabilize_to_surface` — fix instead of regenerate

```python
stabilize_to_surface(
    media: str,                 # image or video
    reference: str,             # the surface (or the keyframe the clip should match)
    geometry: bool = True,      # per-frame homography back onto the reference
    color: bool = True,         # per-channel gain+offset fit on background pixels
    repin: bool = False,        # paste the reference's exact pixels back where nothing was added (feathered)
    smooth: float = 0.5,        # video: temporal smoothing of the per-frame transforms (no jitter)
) -> {"output": path, "before": {...check summary}, "after": {...check summary}}
```

- **Keyframes:** `repin=true` makes the background pixel-identical to the
  surface before the keyframe goes to Kling, so drift can't compound along the
  chain.
- **Clips:** warp every frame back (borders the warp exposes are filled from the
  reference, which is correct by definition) and remove exposure breathing.
  Then `subtract_background_video` is clean.
- Refuses (with the reason) when `bg_ssim` says the background was redrawn,
  since no warp fixes that.
- Reuses `wzrd/align.py` (homography + ECC) and the colour-shift logic in
  `wzrd/subtract_video.py`. Returns before/after numbers from `check_frames`
  so the agent knows it worked without looking.

#### 3. `composite_videos` — many layers → one projector file

```python
composite_videos(
    layers: list[{video, gain=1.0, offset_s=0, mask=None, blend="add"}],
    canvas: str | None = None,          # surface/projector image → size; default first layer
    duration_s: float | None = None,    # default: longest layer; shorter layers loop
    seamless_loop: bool = False,
) -> {"output_video": path, "info": {...}}
```

Additive blend by default (it's the physics of projection; `screen`/`max` as
options against clipping). `mask` multiplies a layer by a region mask, which
also covers "full-frame TextureFlow only inside the windows". This closes the
region-animation dead end (six reprojected region videos → one file) and lets
story + texture + region work share one output. `simulate_view` the result.

#### 4. `edit_video` — the basic NLE verbs

One tool with optional ops applied in a fixed order, rather than eight tools:

```python
edit_video(video, trim_start_s=None, trim_end_s=None, speed=None, reverse=False,
           pingpong=False, fade_in_s=None, fade_out_s=None, gain=None,
           loop_to_s=None, crossfade_loop_s=None, conform_to=None)  # conform_to = canvas image / "1920x1080"
```

`crossfade_loop_s` (blend the tail into the head) turns any clip into a
seamless loop; `pingpong` is the cheap alternative for short ambient clips;
`conform_to` letterboxes/crops to the projector resolution. ffmpeg filters
underneath (`trim`, `setpts`, `reverse`, `fade`, `xfade`), codec as a
parameter.

#### 5. `extract_frames` — stills out of video

```python
extract_frames(video, at: list[float | "first" | "last"] | None = None, count: int | None = None)
  -> {"frames": [paths], "times_s": [...]}
```

Uses: start the next Kling segment from the previous segment's real last frame
(no seam at all); measure how far a clip ended from its keyframe; grab a great
mid-clip moment as a new keyframe; feed video frames to TextureFlow as style
images; make a projector still. Small, but no tool does it today.

#### 6. Asset memory — provenance sidecars + `list_assets` / `mark_asset`

- **Automatic provenance, zero per-tool work:** `logged_tool` already sees
  every tool's args and result. When a result contains project paths, it
  writes `<output>.meta.json` next to each: tool, args (prompt, refs, seed,
  model), parent inputs, timestamp, `est_usd`, and later QC results. Kling
  and TextureFlow get a cost estimate (they return none today).
- **`list_assets(kind=None, status=None, since=None, query=None)`** →
  compact table: path, kind, size/duration, made-by tool, short prompt,
  status, role, note. Plus a project header: counts and **total spend**.
- **`mark_asset(path, status="approved"|"rejected"|None, role="K3", note=...)`**
  so "the approved K3" survives context compaction and the next session, and
  the asset list doubles as the show's storyboard.

#### 7. `import_photos` — the on-site camera roll

```python
import_photos(paths: list[str] | None = None, from_dir="~/Downloads", since_minutes=30)
  -> {"photos": [{path, size, orientation_fixed, taken_at}]}
```

Picks up AirDropped photos, converts HEIC → PNG (`pillow-heif`), applies EXIF
orientation, copies into the project's `captures/`. Also worth fixing
regardless: `resolve_input` should apply EXIF orientation everywhere, or a
portrait phone photo silently enters the pipeline sideways.

### 3.2 Tier 2 — strong value, larger or less certain

| # | Tool | What | Notes |
|---|---|---|---|
| 8 | **`segment_surface`** | Text-prompted masks ("the windows", "the door", "tree trunk") with real edges, same output shape as `extract_color_regions` (masks + `source_box` + aspect) plus a numbered overlay | SAM 3-class model. Host on FAL (verify endpoint/price) or a second class on our Modal app next to TextureFlow. Removes the "generated seg map drifts" problem at the root. Cheaper interim: `refine_masks` that snaps existing mask edges to surface edges (guided filter), local. |
| 9 | **`surface_report`** | Numbers + one numbered overlay: projectable fraction, large calm (flat, reflective) areas for texture fills, openings (dark enclosed rectangles → windows/doors), dominant line directions, zones with % coordinates | Gives the agent, Xander and the image prompts shared coordinates ("zone 3"). Honest caveat: the agent viewing one surface image is cheap, so the value is the shared vocabulary and placement precision, not token savings. |
| 10 | **`interpolate_video` / `upscale_video`** | RIFE/FILM frame interpolation 24 → 48/60 fps; upscale for small region renders | Slow-moving light judders at 24 fps on a 60 Hz projector. FAL hosts both kinds (verify endpoints). |
| 11 | **`reconcile_layerpack`** | Re-import a new segmentation onto an existing pack's layer ids by mask overlap; report splits/merges/new regions for review | Already a stated requirement in the engine reference (stable ids across re-shoots). Without it, a re-segmented surface breaks every scene. |
| — | Small fixes | `extract_color_regions` returns a numbered overlay image; `simulate_view` accepts stills; re-enable `subtract_background_frame` (or fold stills into the video tool) so a keyframe can be previewed as projector output | No new tools. |

### 3.3 Tier 3 — later / only if a real show needs it

- **`analyze_audio(track)`** → BPM, beat grid, section boundaries; plus
  `edit_video(retime_to_bars=…)` so segments last whole bars. Only for
  installations with a fixed soundtrack (a live DJ set doesn't need it; the
  engine handles live audio).
- **Closed-loop calibration**: photograph the real projection, fit
  `simulate_view`'s `projection_strength` / `surface_weight` / `ambient` to
  it, so previews predict this venue and this projector. Needs the camera
  tool or a phone photo.
- **Storyboard sheet**: keyframes in order with captions and segment arrows,
  as one image for Xander to approve the script (or for a client). `view_media`
  mostly covers it.
- **`play_on_projector` stopgap** (mpv fullscreen on display N, looping) until
  the engine plays video. It would fight the engine for the projector window
  and skip the alignment warp, so only if §4's video layers are far off.

### 3.4 Skill changes that go with the tools

- `projection-story` + `video-prompting`: after every keyframe batch and every
  segment, `check_frames` first; auto-discard fails, `stabilize_to_surface`
  the fixable ones; `view_media` only the survivors, for creative judgement.
  Start each segment from the previous one's real last frame
  (`extract_frames`) when continuity matters more than hitting the exact
  keyframe.
- `region-animation`: end with `composite_videos` + `check_frames`
  (projector mode) + `simulate_view`.
- Possibly a new **`on-site`** workflow skill: the 16:00 → 22:00 run sheet
  (import photos → surface → regions → first quick TextureFlow on the wall
  within the first hour, story in parallel, show checklist: warm GPU, final
  projector-mode checks, flashes, haze).

**Tool count:** 24 enabled today; tier 1 adds 7 → 31. The MCP reference doc
judged search-based tool discovery worse UX at ~25 tools; 31 is still fine,
and `edit_video` / `check_frames` are deliberately one tool each instead of
many.

---

## 4. Engine-side gaps (not MCP tools; pointers into the engine roadmap)

1. **Video layers** (engine roadmap §5.9, open: play a video file inside a
   region via hardware decode). This is the missing bridge between the whole
   offline half and the projector. Every story, TextureFlow loop and composite
   currently ends as an mp4 the engine can't show. Worth re-ranking against
   the roadmap's current "next up" (post-shader-swap probation window, WGSL
   `#import`).
2. **Still images / reference textures in WGSL**: let an effect sample a
   pack's `references/` (surface photo, edge map, a depth map) or a generated
   still. Much smaller than video, and a big live-authoring unlock: light that
   follows the real brick edges, a depth-aware sweep, a keyframe still faded
   in on a cue.
3. **Pack reconcile** (§3.2 #11) is half Python, half "the engine already
   tolerates stale ids leniently".

---

## 5. Suggested build order

1. `check_frames` + calibration pass on `test_imgs/stories/` (the thresholds
   are the hard part; the measurements are standard OpenCV).
2. `stabilize_to_surface` (shares 70% of the code with 1).
3. `extract_frames`, `edit_video`, `composite_videos` (ffmpeg-thin, one
   session together).
4. Asset memory + spend counter.
5. `import_photos` + EXIF fix.
6. Skill updates (§3.4) once 1–3 exist; then tier 2 by show need.

---

## 6. Questions for Xander

1. **Thresholds: strict or lenient?** Should `check_frames` fails make the agent
   regenerate on its own, or only report and ask?
2. **How do finished mp4s reach the projector today** (Resolume, QuickTime,
   other)? That decides whether engine video layers (§4.1) should jump the
   engine queue, or whether a stopgap player is enough for now.
3. **Segmentation host:** SAM-class model on FAL (simplest, per-call cost) or
   on our own Modal app next to TextureFlow (one more thing to maintain, no
   per-call markup)?
4. **Asset memory:** sidecar `.meta.json` per file (simple, survives copying
   a folder) or one `assets.json` index per project (one file, easier to read)?
5. **Spend:** just a running total, or a soft per-project budget the agent
   warns at?
6. Anything in the journey (§2) that doesn't match how you actually work on
   site? In particular the daylight-arrival / phone-photo assumption.
