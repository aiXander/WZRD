# TextureFlow on Modal — `deploy/textureflow/`

The TextureFlow ComfyUI graph (SD1.5 + AnimateDiff-LCM + IP-Adapter +
ControlNet, optional latent/ESRGAN upscale, RIFE ×3) running on **our own**
Modal account, no Eden code, keys or S3. Self-contained folder: imports nothing
from `wzrd/` or `wzrd_mcp/`. The WZRD MCP's `texture_flow` and
`textureflow_warm` tools call it — see [The WZRD client](#the-wzrd-client).

## Where things live

| Path | Role |
|---|---|
| `app.py` | The whole Modal app: image build, `download_models`, class `TextureFlow` (GPU memory snapshot + warm-up), `set_warm` / `cool_down` (show-day warmth), `ui` web server, `freeze`, and the `smoke` / `matrix` / `lock` / `warm` local entrypoints. Validation + injection are plain functions (`resolve_args`, `inject`) — importable locally for a GPU-free check. |
| `workflow/` | `api.yaml` (param surface, defaults, node map — **the single source of truth for defaults**), `workflow_api.json` (what ComfyUI executes), `workflow.json` (UI twin for editing), vendored from `edenartlab/workflows@fcbd170` plus the two local fixes under [Local graph edits](#local-graph-edits-vs-eden-fcbd170) — every edit lands in **both** JSONs. `snapshot.json` = ComfyUI core hash + the 13 node packs at pinned hashes (pruned from Eden's workspace snapshot). |
| `models.json` | Every weight: ComfyUI-relative path, public URL (`generate` = made by the fetcher, `null` = Eden-only), sha256, bytes, and the Eden S3 key it was seeded from. |
| `seed_from_eden.py`, `eden_seed_manifest.json` | Provenance of the one-time copy of 15 files from Eden's private S3 (2026-10-09). Never needed again; keys were passed ephemerally and stored nowhere. |
| `constraints.txt` | Exact torch stack (the torch layer, plus every install on the unlocked path). |
| `requirements.lock` | **The install source** for every Python dep after torch (`LOCKED = True`): the `uv pip freeze` of the first working image. Verified 2026-10-09: the locked image's freeze is identical to the lock (242 pins). |
| `test_inputs/` | Smoke style images, generated matrix inputs (shape/mask/regions pngs + mp4s, uppercase-ext copies), `person.jpg` (public-domain Wikimedia "Pyei Sandy.jpg", the pose case's shape input), and `reference/` = Eden's 2026-03-17 renders (`*_00001` = the smoke args, seed 42). |
| `test_results/` | **NEVER run the full ::matrix without Xander's explicit permission (57 GPU renders, real money); use ::smoke or a narrow --only subset.** Gitignored. Every smoke/matrix output + `result.json` + `matrix-<stamp>.json`; `matrix --out <dir>` / `smoke --out <dir>` write into a subfolder instead of over the last run. |

## Modal objects (workspace `aixander`, environment `wzrd`)

App `wzrd-textureflow`; class `TextureFlow` (L40S, one render per container,
`max_containers=3`, scale-to-zero after 300 s, 1 h timeout, GPU memory
snapshot); `set_warm` + `cool_down` (CPU) and Dict `textureflow-warm` (the warm
deadline); volumes
`textureflow-models` (weights, mounted `/weights`) and `textureflow-outputs`
(every render's files under `<request_id>/`, for re-fetch); `ui` = ComfyUI's
browser UI on the same image, behind Modal proxy auth — a browser must send
`Modal-Key`/`Modal-Secret` headers (a proxy-auth token from the dashboard), so
for small graph edits it's simpler to edit `workflow_api.json` + `workflow.json`
by hand and keep them in sync. No secrets.

**Every CLI call needs `--env wzrd`** (profile default is `gigabrain`) and every
client lookup `environment_name="wzrd"`; without it the lookup silently lands in
the wrong environment and fails.

## The `render` contract

`TextureFlow.render(args: dict, files: dict[str, bytes] | None) -> dict`

- `args`: any subset of `api.yaml`'s parameters, plus `output_codec`
  (`"h264"` default | `"hevc10"`; not in api.yaml, which stays Eden-verbatim —
  `EXTRA_PARAMS`/`OUTPUT_CODECS` in `app.py`). It sets VHS `format`/`pix_fmt`/`crf`
  on the two diffusion outputs, 249 and 226. Media params (`images`,
  `control_input`, `diffusion_mask`) are an `http(s)` URL **or** a key into
  `files`. Omitted args get the **api.yaml default**, never the graph literal.
  `seed` omitted / `-1` / `"random"` → random, reported back in `params.seed`.
- `preprocessor1` choices: `Scribble_XDoG_Preprocessor`, `CannyEdgePreprocessor`,
  `DepthAnythingV2Preprocessor`, `AnyLineArtPreprocessor_aux`, **`OpenposePreprocessor`**,
  `none`. `DensePosePreprocessor` (Eden's "human pose") was **replaced on 2026-10-09 and
  is now rejected** by validation — clients must send `OpenposePreprocessor`.
- Validation runs first and rejects with one `ValueError` listing every problem:
  unknown keys, missing `images`, type, range, integer step (`width`/`height`
  320–1280 step 32, `upscale_resolution` 1024–1536 step 64), enum membership,
  1–6 images, media ref neither URL nor `files` key. Undecodable media is
  rejected during input prep — still before ComfyUI runs. Float `step`s are
  not enforced.
- Returns `{request_id, outputs: {output_video, first_pass_video,
  control_signal_preview, mapping_motion_preview} → {filename, bytes, mime},
  params, timings: {prep_s, queue_s, render_s, total_s, cold_start, boot_s,
  restore_s}}`. `boot_s` is the snapshot-time boot + warm-up (the same number on
  every container restored from that snapshot); `restore_s` is this container's
  post-restore hook.
  A missing intermediate is omitted, not an error. ComfyUI errors surface
  verbatim (`execution_error in node N (Type): …`, or the `/prompt` 400 body).

## Load-bearing decisions and traps

- **Reproduce, don't modernize.** ComfyUI `a7a6335b` + node packs at the hashes
  in `workflow/snapshot.json`, `git clone` + `checkout` (comfy-cli can't pin
  core hashes). Torch stack pinned exactly in `constraints.txt`
  (torch 2.10.0+cu130 / xformers 0.0.35 / sageattention 1.0.6). Everything else
  installs from `requirements.lock` *before* the clones, which install no
  requirements of their own. **Refreshing the lock** (after bumping a node pack):
  `LOCKED = False` restores the original resolution — node requirements with
  `uv pip --exclude-newer 2026-03-17T23:59:59Z` (as of Eden's March deploy), then
  Eden's late pins (`transformers==4.49.0`, `diffusers==0.31.0`, …) last, a failing
  requirements file logged not fatal (the only expected one: controlnet_aux's
  nested dinov2 dev file) — then deploy, `::lock`, set it back.
- **Fast boot = GPU memory snapshot** (`GPU_SNAPSHOT`;
  `enable_memory_snapshot` + `experimental_options={"enable_gpu_snapshot": True}`).
  `@modal.enter(snap=True)` starts ComfyUI with `--highvram` (which also switches
  off its dynamic-VRAM loader) and runs one tiny render (`WARMUP_ARGS`: 2 s, 320²,
  4 steps, Scribble ControlNet on generated inputs), so SD1.5 + AnimateLCM +
  IP-Adapter + CLIP-ViT-H + a ControlNet + RIFE are in VRAM when the snapshot is
  taken. The ComfyUI child process and its socket survive restore. Traps:
  **(a)** the `snap=False` hook must `random.seed()` — otherwise every restored
  container draws the same "random" seeds; **(b)** any redeploy discards the
  snapshot, and the first 1–2 cold starts after it *create* one (≈ 100–185 s
  instead of ≈ 40 s) — prime with a render or `textureflow_warm` before a show;
  **(c)** snapshots are per GPU type and deployed apps only (`modal run` /
  `with_options(gpu=…)` boot the slow way, warm-up included); **(d)** don't
  delete volume files a snapshot used. Measured without the warm-up: restored
  cold 44–48 s and a slower first render, so the warm-up stays.
- **Show-day warmth** (`set_warm(minutes)` → `min_containers=1`; 0 = now back
  to zero): the deadline lives in Dict `textureflow-warm` and a spawned
  `cool_down` function sleeps on Modal until it passes, then resets
  `min_containers=0` — so a closed laptop can't leave an L40S on (≈ $2/h idle).
  Calling again moves the deadline; capped at 12 h; a redeploy also resets it.
- **`libcairo2-dev pkg-config build-essential` are required**: `pycairo` (via
  controlnet_aux → svglib) has no Linux wheel; without them controlnet_aux's
  *whole* requirements file fails and mediapipe & co. silently go missing.
- **Weights are symlinked at container start**, not at build: `link_weights()`
  mirrors every file in `/weights` to the same path under `/root/ComfyUI`. Adding
  a weight = put it in the volume (+ `models.json`); no rebuild.
- **`HF_HUB_OFFLINE=1`** in the image: the controlnet_aux annotators
  (DepthAnythingV2-Large, OpenPose body/hand/face from `lllyasviel/Annotators`, MTEED) must come from the volume
  (`custom_nodes/comfyui_controlnet_aux/ckpts/<hf-repo>/<file>`, verified against
  the pinned controlnet_aux source). A missing one fails loudly instead of
  lazy-downloading. RIFE looks in `custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/`.
- **Input normalisation** (in `_normalize`): `LoadRandomImage` (style images)
  only accepts lowercase `.png/.jpg/.jpeg/.bmp/.webp` and silently skips the
  rest, then raises "No valid images" — so extensions are lowercased and
  anything else that decodes (extensionless, `.gif`, `.avif`) is converted to
  png. `Eden_AllMediaLoader` (control input / mask) decodes `.mp4/.mov` as
  video, `.gif` as animation, else a single image; other video containers
  (`.webm`, …) are transcoded to mp4.
- **Per-request isolation**: inputs go to `ComfyUI/input/<request_id>/`; style
  images become `000000_<name>, 000001_<name>…` in one folder (order = list
  order); everything the request wrote to `input/`, `output/`, `temp/` is
  deleted afterwards so warm containers don't grow or collide.
- **ComfyUI validates every combo input**, including models behind untaken
  switch branches: all six ControlNets (remap targets of `preprocessor1` plus the
  depth one on node 182) must exist even if only Scribble is used. The chosen
  preprocessor always runs (node 129 feeds output 131) even with
  `use_controlnet1=false`.
- **Boot ceiling 120 s** (`BOOT_TIMEOUT_S`); a failed boot or a crashed server
  calls `modal.experimental.stop_fetching_inputs()`. Cancellation POSTs
  `/interrupt`.
- **Determinism**: same seed on the same GPU type ≈ identical (SSIM 0.98 between
  two L40S runs). Across GPU types / Eden's March image the frames drift in fine
  detail (SSIM ~0.7) while palette, composition and motion match; Eden's
  reference renders were made on an **A10G** during its build-time test step.
  Xander judged ours visually indistinguishable (2026-10-09).
- Outputs: 249 final (24 fps after RIFE ×3) and 226 first pass (8 fps) are
  **h264 yuv420p crf 19 by default**; `output_codec="hevc10"` restores the graph's
  original HEVC 10-bit (`hvc1` tag — plays in QuickTime/Safari, not in
  Chrome-on-Linux, OpenCV or some HW decoders). 131 control signal and 69
  IP-adapter content map are always h264.
- **ESRGAN runs whenever `upscale_esrgan` is true, even with
  `use_upscale=false`** (api.yaml's `visible_if` is a UI hint only): 384² →
  1920². Fast tests need both off. `Eden_PROD_DEBUG`=true makes extra debug files save (pngs, a params
  json, a texture grid); the handler deletes them and never returns them.

## Local graph edits (vs Eden `fcbd170`)

Both fixed 2026-10-09; the evidence is the `matrix` cases named below.

**1. `denoise < 1` img2imgs from the shape input.** How the first-pass latent
is built: node 409 `LatentCompositeMasked` takes the VAE-encoded shape input
(123) as *destination* and pastes a *source* latent wherever the diffusion mask
(410 → 407 → grow 398 → blur 399) is white; 206 `SetLatentNoiseMask` then lets
the KSampler (205, `denoise` = node 26) touch only those regions. Eden's graph
wired 409's source to the empty latent (120), so `denoise < 1` started from
zeros and came out grey. **409's source is now switch 126** (`denoise` full →
empty latent 120, else → encoded shape input 123), which the diffusion-mask
rewire had orphaned. **Trap:** 126 is driven by 163 `Eden_Compare` against the
literal in node 122; Eden had `"a == b"` vs `1.0000000000000002`, which is False
for the server's injected `1.0` and would have switched *every* default render
to img2img. Now 163 = **`"a >= b"`** and 122 = **`0.999`** — keep the test
tolerant if you touch it. Evidence: `smoke` (denoise 1) unchanged (SSIM 0.973 vs
the pre-fix render); `denoise_0.7` now returns the black-and-white ring nearly
untouched (mean brightness 47 vs 117 for the grey pre-fix render).
**Behaviour to expect** (sweep 2026-10-09): there is a deliberate cliff between
`denoise` 0.99 (img2img from the shape input) and 1.0 (fresh render from an empty
latent). Below 1, ComfyUI runs `int(n_steps / denoise)` total steps and keeps
the last `n_steps`, so at `n_steps = 4` every value in 0.81–0.99 renders
identically; finer control needs more steps. The "shape input" (node 123 encodes it) is
`control_input` fit to size **whether or not `use_controlnet1` is on**, else the
graph's black-video fallback: so `denoise < 1` without a `control_input` renders
near-black, and black diffusion-mask regions show the `control_input` (black only
when none is given). Verified on renders 2026-10-09. The MCP `texture_flow`
docstring spells this out for calling agents; keep it in sync. On a flat graphic, 0.7 is close to
identity; on a photo, 0.7–0.95 restyles progressively while keeping its layout.

**2. "Human pose" = `OpenposePreprocessor`.** `api.yaml` remaps the pose choice
to `control_v11p_sd15_openpose.pth` (no DensePose ControlNet exists in the
stack). Eden fed it `DensePosePreprocessor` — a filled body-part map the
OpenPose ControlNet never saw in training, and a flat purple frame when there's
no person. Now node 129 runs controlnet_aux's `OpenposePreprocessor` (via
`AIO_Preprocessor`, so its defaults apply: body + hands + face) → a stick-figure
skeleton on black. Weights: `body_pose_model.pth`, `hand_pose_model.pth`,
`facenet.pth` from `lllyasviel/Annotators` in the volume. The old DensePose
torchscript is no longer in `models.json` (the file still sits in the volume,
unused). DWPose was the alternative; skipped because its default detector is
`.onnx` and the image has no GPU onnxruntime. Evidence: `pose_person` (a person
in the shape input) and `pre_OpenposePreprocessor` (no person: black control
signal, no error).

## The WZRD client

`wzrd_mcp/tools.py`:
- **`texture_flow`** — `modal.Cls.from_name($TEXTUREFLOW_APP or "wzrd-textureflow",
  "TextureFlow", environment_name=$WZRD_MODAL_ENV or "wzrd")().render.remote.aio`.
  Every arg defaults to `None` and **only caller-set args are sent**, so api.yaml
  stays the single source of defaults (the old client's own defaults disagreed).
  Media args: an http(s) URL passes through; a local path is read into `files`
  under its basename (deduped `name_2.ext`). The returned bytes land via
  `project.output_path("texture_flow_<output>", …)` in the project's
  `generated/`; the reply is those paths plus `seed`, `params`, `request_id`,
  `timings`. No S3, no AWS env. The docstring spells out server defaults and
  the `denoise` / ESRGAN / OpenPose behaviour for calling agents — keep it in sync
  with api.yaml (it lists the `preprocessor1` choices by hand).
- **`textureflow_warm(minutes)`** — calls the deployed `set_warm`.
- `wzrd_mcp/skills/{island_animation,surface_to_projection}` pin
  `use_upscale`/`upscale_esrgan` false, as the old client defaults did; the server
  defaults would make each step a ~5-min full-HD render.

## Performance (L40S)

Smoke args (3 style images, 2.5 s, 384², 4 steps, upscale off), client wall for
one `render.remote()`, measured 2026-10-09 on the deployed app:

| | cold start | warm |
|---|---|---|
| Phase 1 doc baseline | ≈ 86 s | ≈ 34 s |
| Same code re-measured 2026-10-09 | 58–64 s | 24–34 s |
| **GPU snapshot + warm-up (shipped)** | **37–49 s** (restored) | **25–26 s** |

A restored cold start is ≈ the warm render + 12–20 s of Modal scheduling + GPU
restore — near the floor. The first cold start after a deploy (snapshot
creation) takes ≈ 100–185 s.
Warm ComfyUI time for other shapes (arg matrix): 2 s / 320² / 4 steps 12–26 s;
24 s at 320² 102 s; upscale + ESRGAN from 320² (→ 1920²) 90 s; **all api.yaml
defaults** (640², 6 s, 8 steps, latent upscale 1280 + ESRGAN → 1920²) 290–351 s.

**GPU (D2 re-checked 2026-10-09)** — warm render, smoke args at 512²:

| | L40S | A100-40GB |
|---|---|---|
| 512², no upscale | **26 s** | 54 s |
| 512² → latent 1024 + ESRGAN | **92 s** | 128–137 s |

L40S is 1.4–2× faster and cheaper, so it stays. Never Blackwell (the 2025-era
nodes don't run there).
