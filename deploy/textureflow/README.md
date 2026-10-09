# TextureFlow on Modal (`wzrd-textureflow`)

Eden-free GPU endpoint for the TextureFlow ComfyUI graph. Workspace `aixander`,
environment **`wzrd`** — every command passes `--env wzrd` (the profile default
is `gigabrain`). Full contract + traps: `docs/reference/textureflow.md`.

```bash
uv run modal run    --env wzrd deploy/textureflow/app.py::download_models   # fill missing weights (public URLs, sha256-checked)
uv run modal deploy --env wzrd deploy/textureflow/app.py                    # build + deploy
uv run modal run    --env wzrd deploy/textureflow/app.py::smoke             # test.json render -> test_results/smoke/
uv run modal run    --env wzrd deploy/textureflow/app.py::matrix            # every api.yaml arg + rejections -> test_results/
#   ⚠ NEVER run the full ::matrix without Xander's explicit permission (57 GPU renders, real money); use ::smoke or a narrow --only subset.
#   add --deployed to hit the deployed app instead of an ephemeral one; --only pre_,map_ for a subset;
#   --out <dir> writes under test_results/<dir>/ instead of over the last run
uv run modal run    --env wzrd deploy/textureflow/app.py::warm --minutes 120 # keep one GPU container up (0 = cool down now)
uv run modal run    --env wzrd deploy/textureflow/app.py::lock              # refresh requirements.lock (set LOCKED = False first; see reference doc)
#   smoke also takes --gpu A100-40GB --out <dir> (render on another GPU type)
```

Client call:

```python
tf = modal.Cls.from_name("wzrd-textureflow", "TextureFlow", environment_name="wzrd")()
res = tf.render.remote(args={"images": ["style.jpg", "https://…/b.webp"], "n_seconds": 4},  # + "output_codec": "hevc10"
                       files={"style.jpg": Path("style.jpg").read_bytes()})
res["outputs"]["output_video"]["bytes"]  # + first_pass_video, control_signal_preview, mapping_motion_preview
```

## Provenance

- `workflow/` — `api.yaml`, `workflow_api.json`, `workflow.json` vendored from
  `edenartlab/workflows@fcbd170` (`workspaces/wzrd/workflows/texture_flow/`, 2026-03-17), plus two
  local fixes (denoise img2img, OpenPose instead of DensePose); see the reference doc's "Local graph edits".
  `workflow/snapshot.json` is that workspace's `snapshot.json` pruned to the core hash + the 13 node packs the graph uses.
- `models.json` — every weight, its public URL and sha256. `seed_from_eden.py` +
  `eden_seed_manifest.json` record the one-time copy of 15 files from Eden's S3 (2026-10-09).
- `constraints.txt` — the exact torch stack; `requirements.lock` — `pip freeze` of the first working image, now the image's install source.

After any redeploy the GPU memory snapshot is rebuilt by the next 1–2 cold starts (2–3 min each); prime it before live use.
- `test_inputs/` — smoke style images (Eden CloudFront / Xander's GCS), generated matrix inputs,
  and `reference/` = Eden's 2026-03-17 renders (`*_00001` = test.json, seed 42; `*_00002` = test_video.json, random seed).
