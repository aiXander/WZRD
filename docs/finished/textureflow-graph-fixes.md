# Sprint: fix TextureFlow's `denoise` and "human pose" graph bugs

Two bugs inherited from Eden's graph, both verified on renders. Mechanics with
node numbers: [reference/textureflow.md → Known graph quirks](../reference/textureflow.md#known-graph-quirks-left-alone--no-graph-edits-in-phase-1).
Everything lives in `deploy/textureflow/`; deploy/test commands in its `README.md`
(every Modal call needs `--env wzrd`). Parent plan: [textureflow-modal.md](textureflow-modal.md).

## Fix 1 — `denoise < 1` must img2img from the shape input

**Bug:** inside white diffusion-mask regions the first-pass sampler (205) starts
from an empty latent, so `denoise < 1` gives grey, washed-out video instead of
inheriting the shape input's colours/darkness (matrix `denoise_0.7`: mean
saturation 35 vs 74 at denoise 1).

**Fix:** in `workflow_api.json` set node **409** (`LatentCompositeMasked`)
`source` from `["120", 0]` (empty latent) to `["126", 0]` (the orphaned switch:
`denoise == 1` → empty latent, else → encoded shape input 123).

**Trap — float equality:** 126 is driven by 163 `Eden_Compare "a == b"`, a plain
Python `==` between node 26 (denoise; the server injects `1.0`) and node 122
(literal `1.0000000000000002`). `1.0 == 1.0000000000000002` is False, so the
switch would pick the shape input on *every default render* and break parity.
Make the test tolerant: e.g. 163 → `"a >= b"` with 122 = `0.999` (check the
exact `COMPARE_FUNCTIONS` names in `eden_comfy_pipelines/logic/logic_nodes.py`).

**Done when:** matrix `denoise_0.7` keeps the shape input's look (dark shape
input → dark-ish output, saturation no longer collapses); `smoke` (denoise 1)
is unchanged — SSIM ≥ 0.97 against `test_results/smoke/output_video.mp4` from
before the change (two identical runs on L40S score 0.98).

## Fix 2 — "human pose" must produce an OpenPose skeleton

**Bug:** `preprocessor1 = DensePosePreprocessor` is remapped (in `api.yaml`) to
the **OpenPose** ControlNet, which was trained on stick-figure skeletons, not
DensePose's coloured body-part map. Without a person DensePose outputs a flat
purple frame and shape guidance vanishes.

**Fix (recommended):** replace the choice with `OpenposePreprocessor` (pure
torch, matches what `control_v11p_sd15_openpose.pth` was trained on). DWPose is
the alternative but its default bbox detector is `.onnx` and the image has no
GPU onnxruntime — skip unless OpenPose quality disappoints.
- `api.yaml`: rename the choice + its `remap` key; label e.g. "Human pose (OpenPose)".
  This changes the public enum — `DensePosePreprocessor` will be rejected; say
  so in the reference doc (the WZRD client is being rewired in parallel).
- Weights (`HF_HUB_OFFLINE=1` means nothing lazy-downloads): controlnet_aux@83463c2e
  loads `body_pose_model.pth`, `hand_pose_model.pth`, `facenet.pth` from HF
  `lllyasviel/Annotators` into `custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/`.
  Confirm the AIO name (`OpenposePreprocessor`) and paths in
  `node_wrappers/openpose.py` + `src/custom_controlnet_aux/open_pose/__init__.py`,
  add rows to `models.json` (with HF sha256 from the `x-linked-etag` header),
  run `::download_models`.
- Keep or drop the DensePose weight row — dropping it is fine once nothing selects it.
- Matrix: rename case `pre_DensePosePreprocessor`, and add a case with a **person**
  in the shape input (any public photo URL of a standing person) — the current
  test shape has none, so it can't prove pose transfer.

**Done when:** the pose case's `control_signal_preview` shows a skeleton and the
output visibly follows the pose; the no-person case doesn't error.

## How to edit the graph

Edit `workflow_api.json` (what runs) **and** mirror the change in
`workflow.json` (the UI twin — links live in its `links` array plus each node's
`inputs[].link`), so the two never diverge. The deployed `ui` endpoint is behind
Modal proxy auth: a browser needs `Modal-Key`/`Modal-Secret` headers (a proxy
auth token from the Modal dashboard), so hand-editing the JSON is the practical
route. After edits: redeploy, then `smoke` + full `matrix` (all cases must still
pass), then the doc pass — move both bugs out of the reference doc's "Known
graph quirks" into its traps/contract, and close this file.
