# TextureFlow — the WZRD side

TextureFlow (SD1.5 + AnimateDiff-LCM + IP-Adapter + ControlNet, optional latent/ESRGAN upscale, RIFE ×3 →
a looping video) renders on our own Modal account (workspace `aixander`, environment **`wzrd`**). **All
endpoint code lives in the separate `textureflow-lab` repo** (`~/Documents/GitHub/Gigabrain/VJ/textureflow-lab`);
WZRD only calls it by name from the MCP. Nothing in this repo deploys anything to Modal.

## The two apps

| App | Source (textureflow-lab) | Role |
|---|---|---|
| **`textureflow`** | `deploy.py` — the lab's plain-Python pipeline over its vendored node tree | **Live.** The lab's CI redeploys it on every push to the lab's `main` (lint → smoke render → deploy → check → rollback on failure), so WZRD always renders with the lab's latest merged pipeline. Contract, CI chain, snapshot, rollback: the lab's `docs/reference/deploy.md`. |
| `wzrd-textureflow` | `comfyui_app/app.py` — the original ComfyUI graph on the frozen pins | **Fallback**, deployed and idle (scale-to-zero, costs nothing). It was this repo's `deploy/textureflow/` until 2026-10-21, then moved to the lab whole (redeploy, weights download, smoke/matrix, history: the lab's `docs/reference/comfyui-app.md`). Switch to it with `TEXTUREFLOW_APP=wzrd-textureflow` in WZRD's env: no deploy needed. |

Both take the same contract: `TextureFlow.render(args, files) -> {request_id, outputs: {output_video,
first_pass_video, control_signal_preview, mapping_motion_preview} → {filename, bytes, mime}, params,
timings}`, plus `set_warm(minutes)`. Defaults and validation live server-side in `api.yaml`.

**Every Modal CLI call needs `--env wzrd`** (the profile default is `gigabrain`) and every lookup
`environment_name="wzrd"`; without it the lookup silently lands in the wrong environment and fails. The
`textureflow-models` weights volume and `textureflow-outputs` (every render's files under `<request_id>/`)
are shared by both apps; the lab's `comfyui_app/app.py::download_models` is the only writer of the weights.

## The WZRD client

`wzrd_mcp/tools.py`:
- **`texture_flow`** — `modal.Cls.from_name($TEXTUREFLOW_APP or "textureflow", "TextureFlow",
  environment_name=$WZRD_MODAL_ENV or "wzrd")().render`. Every arg defaults to `None` and **only
  caller-set args are sent**, so api.yaml stays the single source of defaults. Media args: an http(s) URL
  passes through; a local path is read into `files` under its basename (deduped `name_2.ext`, photos made
  upright first). `output_video` lands in the media library (`library.output_path`, `~/WZRD/media`, Studio
  target folder else `inbox/`) with an engine HAP clip beside it; the three previews go to
  `.wzrd/intermediates/<job>/`; the reply is those paths plus `seed`, `params`, `request_id`, `timings`.
  Cost = `timings.total_s` (+ `restore_s` on cold starts) × `_TF_USD_PER_S` (RTX PRO 6000 list price),
  checked against the daily spend cap first. The docstring spells out server defaults and the `denoise` /
  ESRGAN / OpenPose behaviour for calling agents — keep it in sync with the lab's `reference/api.yaml`
  (it lists the `preprocessor1` choices by hand). `recipe=<name>` fills unset args and style images from a
  saved look in `library/recipes/` (merge rules: [mcp-server.md](mcp-server.md) "Style recipes").
- **Two backends, picked per call — `backend="modal" | "local"`** (default `$TEXTUREFLOW_BACKEND`, unset =
  `modal`; never sent to the server, like `region`). **No auto-routing and no fallback, ever** (Xander's
  decision): a failed local call is never retried on Modal. `local` = the job server on Xander's desktop
  RTX 3090 over Tailscale (`TEXTUREFLOW_LOCAL_URL`, `TEXTUREFLOW_LOCAL_TOKEN`; server + its wire contract v1
  live in the lab, `docs/TODO/job-server.md` there). Client: `wzrd_mcp/textureflow_local.py` — `GET /v1/health`
  (2 s; off/silent → ToolError "unreachable — is the desktop on?", before the spend check, nothing sent
  anywhere) → multipart submit (`args` JSON part + one `file` part per `files` key) → long-poll status
  (stage → `log_progress` + MCP progress, under `with_keepalive`) → result → download each output `url` into
  `bytes`, giving **Modal's exact `render` dict**, so the library write / previews / record / HAP code is
  shared. Same args, outputs, `params.seed`, timings keys (`queue_s` = real local queue wait,
  `restore_s` null). Local is $0: `_tf_estimate` (and `jobs.ESTIMATORS`, so `/api/estimate`) → 0,
  `paid: False`, `est_usd {}`, no `spend` line, record `args.model = "textureflow-local (TextureFlow, <gpu>)"`.
  A 400 passes the server's validation text through verbatim; a worker that is `down` accepts the job and
  fails it on the first poll; `booting` counts as up. Cancel: `jobs.set_remote("local|<job id>")` →
  `POST /v1/jobs/<id>/cancel` (also for lost jobs, `jobs._cancel_remote`). Test: `python test.py textureflow`
  (Modal mocked + a fake aiohttp job server).
- **`textureflow_warm(minutes)`** — calls the deployed `set_warm`: keeps one GPU container up until a
  deadline that a Modal-side `cool_down` enforces (so a closed laptop can't leave a GPU on, ≈ $3/h idle);
  0 = cool down now; capped at 12 h. `backend="local"` only health-checks the desktop (always warm while
  its service runs): `{"warm": worker != "down", "backend": "local", gpu, worker, busy, queue}`, Modal untouched.
- The `run-sheet` and `textureflow` skills render drafts with `use_upscale`/`upscale_esrgan` false; the
  server defaults would make each draft a ~5-min full-HD render. **ESRGAN runs whenever `upscale_esrgan`
  is true, even with `use_upscale=false`.**

## Show-day traps

- **A redeploy discards the GPU memory snapshot.** For `textureflow` the lab's CI check render retakes it
  right after each deploy; still, prime the endpoint (a render or `textureflow_warm`) before a show, and
  don't merge to the lab's `main` during one — that ships to production minutes later.
- **Rollback** of a bad live deploy: either `TEXTUREFLOW_APP=wzrd-textureflow` here (instant, no deploy),
  or `uv run modal app rollback textureflow --env wzrd` from the lab.
- Timings, GPU choice and per-stage costs: the lab's `docs/reference/benchmarks.md` (on RTX PRO 6000 a 15 s
  HD render ≈ 166 s / ≈ $0.14 after the RIFE bump; a small draft ≈ 20–40 s).
