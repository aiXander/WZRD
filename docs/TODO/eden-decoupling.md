# Eden decoupling

Goal: run the whole toolkit + MCP server on localhost, with a **project folder
per job** holding every asset, and no dependency on Eden's AWS account, Modal
workspace, or `eve` conventions. Iterative — sprint 1 and the TextureFlow
rehome have landed; what's left is below.

## Open work

### 1. Teardown of the old Eden deployments (**only with Xander's OK**)

Nothing in WZRD calls these any more (TextureFlow runs on our own
`wzrd-textureflow`, see [reference/textureflow.md](../reference/textureflow.md)).
On 2026-10-09 Xander chose to **keep the `edenartlab` apps unchanged for now**:
`comfyui-wzrd-STAGE` (the Eden TextureFlow endpoint) and the stale `wzrd-mcp`
app (see item 3). Don't stop them without a fresh OK; leave Eden's
`comfy-downloads` volume alone regardless.

Our own leftovers were stopped 2026-10-09: `aixander/main` `comfy-textureflow`
and `aixander/wzrd` `wzrd-textureflow-dev`. The `aixander/main` volume
`comfy-models-cache` (the dead app's model cache) was kept — remove it only with
Xander's OK. Don't touch `stitchly-comfyui`.

### 2. Keys

New `FAL_KEY` (Eden's was being borrowed) → `.env`. Nothing else is needed for
a local session; see `.env.example`.

### 3. Modal deployment sink (only if the cloud path is wanted back)

`modal_app.py` is parked: tools now write to a project folder, which in a
container is ephemeral scratch nothing can read afterwards. Reviving it means
giving it a durable sink — a Modal Volume mounted as `WZRD_PROJECTS_ROOT`, or
`file_io.upload` against a self-owned bucket. Its secret name is now
`WZRD_MODAL_SECRET` (default `wzrd-secrets`), no longer `eve-secrets-PROD`.
A stale `wzrd-mcp` app is still deployed in edenartlab/main — kept for now, see item 1.

### 4. Nice-to-haves

- Per-call `project=` override on tools (today it's session-level via
  `set_project`, plus the `WZRD_PROJECT` env var).
- A `build_scene` step that writes `scene.json` next to a freshly built pack,
  so a project folder is engine-loadable without hand-authoring.
- `fullframe_animation/skill.json` calls `kling_v3_image_to_video`, which is
  disabled (v3 has been off in favour of v2.5 since before sprint 1).

## Done — TextureFlow rehome (2026-10-09)

The last hard Eden dependency is gone: `texture_flow` renders on our own
`wzrd-textureflow` (Modal env `wzrd`), takes local paths or URLs and writes
the mp4s into the project with no AWS env. Done — see
[reference/textureflow.md](../reference/textureflow.md).

## Done — sprint 1 (2026-08-12)

Everything except TextureFlow is off Eden. The offline pipeline previously
*could not run locally at all* (every tool hit `KeyError: AWS_ACCESS_KEY_ID`);
it now runs with no cloud credentials.

- **`wzrd_mcp/project.py`** (new) — project folders under `projects/`
  (gitignored; `WZRD_PROJECTS_ROOT` to move). `output_path`/`output_dir`
  reserve destinations *before* the work runs, so libraries write straight into
  the project and sidecar files (alignment-aid videos) land there for free;
  `store`/`download_async` bring in files produced elsewhere. Layout is a
  render-core scene directory on purpose — `scene.json` + `effects/` + `packs/`.
- **`set_project` tool** (`project_tools.py`) — session-level active project,
  documented as step one in the server instructions.
- **All 11 tools in `tools.py`, both FAL tools, and the camera tool** return
  absolute local paths instead of S3 URLs. `file_io.upload` is now called by
  nothing.
- **`build_layerpack` writes a loadable pack** into `<project>/packs/<name>/`.
  It used to upload masks and rewrite `pack.json` with `mask_url` fields —
  which render-core cannot consume, since `pack.rs` resolves masks off local
  disk relative to `pack.json`. That loop is now closed.
- **FAL inputs accept local paths** — `_to_fal_url` uploads them to FAL's CDN
  just in time. Unavoidable (FAL's workers can't read this disk), ephemeral,
  and unrelated to S3.
- **`file_io.py` kept but dormant** — S3 code intact for a future self-owned
  bucket, `boto3` demoted to the optional `s3` extra and imported lazily.
- Eden's `edenartlab-stage-data` bucket default and the `eve-secrets-PROD`
  secret reference are gone.

Verified end to end without credentials: `set_project` → `prepare_surface` →
`extract_color_regions` → `build_layerpack`, with every mask referenced by the
resulting `pack.json` present on disk relative to it. The engine was not
launched on that pack (GUI window); the check was structural.
