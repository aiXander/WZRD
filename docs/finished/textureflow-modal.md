# TextureFlow on our own Modal account — remaining work

> **Goal:** the local WZRD MCP's `texture_flow` tool renders on our own Modal
> deployment (`wzrd-textureflow`, env `wzrd`), boots fast, and stays
> reproducible. Phases 1–3 are **done**; only the optional Phase 4 is open.
> Current state, contract, traps and numbers: [reference/textureflow.md](../reference/textureflow.md).
> The old Eden / `aixander/main` leftovers are listed there too (hands off
> without Xander's OK).

## Phase 4 — Modernize (later, optional, open)

Only if the pinned stack rots or new nodes are wanted. Edit the graph by hand in both JSONs, or in the deployed `ui` web server (needs Modal proxy-auth headers in the browser); re-vendor. Landscape (Oct 2026): ComfyUI core v0.39 with a V3 node API; AnimateDiff-Evolved / Advanced-ControlNet v1.6 need a modern core; **IPAdapter_plus is maintenance-only** (riskiest pack); eden_comfy_pipelines has a "Modernize" commit (2026-09-25). Cheap win any time: swap the five 1.45 GB ControlNet `.pth` for `comfyanonymous/ControlNet-v1-1_fp16_safetensors` (update `api.yaml`'s remap + nodes 134/182). Bumping a node pack means refreshing `requirements.lock` (`LOCKED = False` → deploy → `::lock` → back; see the reference doc).

## Decisions (Xander, 2026-10-09) — still binding

| # | Decision |
|---|---|
| D1 | Everything WZRD lives in Modal environment **`wzrd`** (workspace `aixander`); every CLI call passes `--env wzrd`, every lookup `environment_name="wzrd"`. |
| D2 | GPU **L40S** (re-benchmarked vs A100-40GB: L40S 1.4–2× faster). **Never Blackwell** (B200/RTX-PRO-6000) — the 2025-era nodes won't run there. |
| D3 | Output codec `h264` by default, `output_codec="hevc10"` for the graph's original HEVC 10-bit. |
| D4 | Weights were copied once from Eden's S3; nothing may depend on Eden keys at runtime. Public URLs in `models.json` are the fallback. |
| D5 | Keep all three base checkpoints. |

## Done

- **Phase 1 (2026-10-09)** — `deploy/textureflow/` deployed as `wzrd-textureflow`: pinned ComfyUI + 13 node packs, 21 sha256-verified weights in volume `textureflow-models`, `render(args, files)` with server-side api.yaml defaults + validation, smoke parity with Eden's reference, full arg matrix.
- **Phase 2 (2026-10-09)** — `texture_flow` MCP tool rewired (local paths → `files`, URLs pass through, only caller-set args sent, mp4s into `generated/`, `seed` surfaced, all S3 code gone) and re-enabled; server-side `output_codec`.
- **Phase 3 (2026-10-09)** — GPU memory snapshot with a warm-up render (restored cold start 37–49 s vs 58–64 s re-measured baseline), `set_warm`/`textureflow_warm` show-day switch with a Modal-side timer, image installed from `requirements.lock`, L40S confirmed over A100.

Done — see [reference/textureflow.md](../reference/textureflow.md).
