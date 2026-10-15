# Backlog — small, low-priority work

Loose items with no plan of their own. None blocks the gig
([gig-2026-11-live-video.md](gig-2026-11-live-video.md)) or the post-gig
toolbox ([vj-toolbox.md](vj-toolbox.md)); pick one up when it's in your way.
Promote an item into a real plan if it grows past a few hours.

## Setup

- **Own `FAL_KEY`.** Not in `.env` or the shell (checked 2026-10-10), so the
  `kling_v25_image_to_video` tool and `scripts/fal_shootout.py` can't run. The
  key Eden lent us is gone; Xander creates one on fal.ai → `.env`
  (`.env.example` lists it).

## MCP server (`wzrd_mcp/`)

- **Image draft tier.** Muse Image (`meta/muse-image/edit` on FAL, ~$0.01/img,
  #7 on the Artificial Analysis edit board) as a cheap draft next to
  `nano_banana` — a separate tool or a `quality` arg on the main one. Current
  model choices: [reference/hosted-models.md](../reference/hosted-models.md).
- **Revive the cloud MCP deploy (only if wanted).** `modal_app.py` is parked:
  in a container the project folder is ephemeral scratch. Reviving it needs a
  durable sink — a Modal Volume mounted as `WZRD_PROJECTS_ROOT`, or
  `file_io.upload` against a self-owned bucket. Its secret is
  `WZRD_MODAL_SECRET` (default `wzrd-secrets`).

## Engine (`render-core/`)

- **Native DXV playback (post-gig).** Xander's existing library (`~/Visuals`,
  ≈ 199 GB) is almost all Resolume **DXV** (`DXD3`); the engine plays HAP only
  (gig plan D2), so the Studio plan converts a DXV clip to HAP beside it on
  "make playable". DXV3 is DXT1/DXT5 under Resolume's own compression;
  ffmpeg's `dxv.c` (~1,200 lines) is the reference to port, and after
  decompression it reuses HAP's BC upload path. Reverses D2 deliberately.

## TextureFlow endpoint (`deploy/textureflow/`)

- **Modernize the pinned stack (optional).** Only if the pinned stack rots or
  new nodes are wanted. Edit the graph by hand in both JSONs, or in the
  deployed `ui` web server (needs Modal proxy-auth headers in the browser);
  re-vendor. Landscape (Oct 2026): ComfyUI core v0.39 with a V3 node API;
  AnimateDiff-Evolved / Advanced-ControlNet v1.6 need a modern core;
  **IPAdapter_plus is maintenance-only** (riskiest pack); eden_comfy_pipelines
  has a "Modernize" commit (2026-09-25). Cheap win any time: swap the five
  1.45 GB ControlNet `.pth` for `comfyanonymous/ControlNet-v1-1_fp16_safetensors`
  (update `api.yaml`'s remap + nodes 134/182). Bumping a node pack means
  refreshing `requirements.lock` — procedure in
  [reference/textureflow.md](../reference/textureflow.md).
