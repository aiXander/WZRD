# Eden decoupling

Goal: run the whole toolkit + MCP server on localhost, with a **project folder
per job** holding every asset, and no dependency on Eden's AWS account, Modal
workspace, or `eve` conventions. Iterative — sprint 1 has landed, the rest is
below.

## Open work

### 1. TextureFlow rehome (the last hard dependency)

`texture_flow` is **disabled** in `tools_config.json` and `server.py`. It calls
`modal.Cls.from_name("comfyui-wzrd-STAGE", "ComfyUIPremium")`, which lives in
the **`edenartlab` Modal workspace (env `main`)** — not in `aixander/gigabrain`
— feeds it public input URLs, and gets outputs back as S3 keys in Eden's
bucket. No local refactor can fix that; the deployment has to move first.

Source material, both outside this repo:

| What | Where |
|---|---|
| The workflow itself | `~/Documents/GitHub/Eden/workflows/workspaces/wzrd/workflows/texture_flow/` (`api.yaml`, `workflow_api.json`, `snapshot.json`, model-name swap script) |
| The deploy machinery | `~/Documents/GitHub/Eden/comfyui.py` — ~1950 lines, heavily opinionated, assumes Eden's S3/`eve` conventions end to end |

Work: extract a minimal ComfyUI-on-Modal deploy (no `eve` coupling), deploy to
`aixander/gigabrain`, then in `tools.py` — point `MODAL_APP_NAME`/`MODAL_CLS_NAME`
at it, give inputs the same just-in-time upload treatment FAL inputs get (the
GPU worker can't read local disk either), and set an output sink it can read
back. The download-into-project plumbing on the response side is already
written and will work the moment `_tf_file_url` can resolve.

### 2. Keys

New `FAL_KEY` (Eden's was being borrowed) → `.env`. Nothing else is needed for
a local session; see `.env.example`.

### 3. Modal deployment sink (only if the cloud path is wanted back)

`modal_app.py` is parked: tools now write to a project folder, which in a
container is ephemeral scratch nothing can read afterwards. Reviving it means
giving it a durable sink — a Modal Volume mounted as `WZRD_PROJECTS_ROOT`, or
`file_io.upload` against a self-owned bucket. Its secret name is now
`WZRD_MODAL_SECRET` (default `wzrd-secrets`), no longer `eve-secrets-PROD`.
A stale `wzrd-mcp` app is still deployed in edenartlab/main — tear it down.

### 4. Nice-to-haves

- Per-call `project=` override on tools (today it's session-level via
  `set_project`, plus the `WZRD_PROJECT` env var).
- A `build_scene` step that writes `scene.json` next to a freshly built pack,
  so a project folder is engine-loadable without hand-authoring.
- `wzrd_mcp/skills/*/skill.json` still say "URL" in their input descriptions
  (harmless — paths chain fine — but stale), and three steps call tools that
  are currently disabled: `texture_flow` in `island_animation` and
  `surface_to_projection`, `kling_v3_image_to_video` in `fullframe_animation`
  (v3 has been off in favour of v2.5 since before this sprint).

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
