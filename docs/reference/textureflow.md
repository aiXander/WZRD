# TextureFlow — the WZRD side

TextureFlow (SD1.5 + AnimateDiff-LCM + IP-Adapter + ControlNet, optional latent/ESRGAN upscale, RIFE ×4 →
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
timings}`, plus `set_warm(minutes)`. Defaults and validation live server-side in the lab's
`tflab/spec.yaml`, which `textureflow` also publishes (`spec()`, below); `wzrd-textureflow` validates
against its own frozen `api.yaml` (today the same 20 args + `output_codec`).

**Every Modal CLI call needs `--env wzrd`** (the profile default is `gigabrain`) and every lookup
`environment_name="wzrd"`; without it the lookup silently lands in the wrong environment and fails. The
`textureflow-models` weights volume and `textureflow-outputs` (every render's files under `<request_id>/`)
are shared by both apps; the lab's `comfyui_app/app.py::download_models` is the only writer of the weights.

## The spec-driven tool (`wzrd_mcp/tf_spec.py`)

The endpoint owns its inputs. The lab's `tflab/spec.yaml` holds every arg's type, default, range,
choices (+ `choice_notes`), a UI `label`, an agent-facing `description` (what it does visually, traps),
a model `summary` and a semver `version`; the deployed `textureflow` serves it from a CPU function
`spec()` → `{app, git_sha, spec}` (breaking-change rule: the lab's `docs/reference/deploy.md`).

- **At import** `tools.py` builds `texture_flow` from `tf_spec.current()`: one `Optional[T] = None`
  param per spec arg (array → `list[str]`, media → `str`), WZRD's own hand-written args (`region`,
  `allow_over_cap`, `hap`, `recipe`, `backend`, `ctx`; `project`/`dest`/`wait` injected by `tool()`)
  merged in; docstring = spec `summary` + WZRD's `_TF_INTRO` (media library, recipes, backends) +
  `Args:` rendered by `tf_spec.describe` — description, then `One of: "a" (note), …`, `Range lo-hi, step
  s.`, `Default d.` from the structured fields. `_TF_NOTES` adds WZRD meaning (`images`: required unless
  a recipe supplies them). Seeded 2026-10-21; schema diffed against the old hand-written tool: same 27
  names, types, defaults, `additionalProperties: false` — only descriptions changed.
- **Source order, never blocking on Modal:** the cache `~/WZRD/media/.wzrd/textureflow_spec/<app>.json`
  (the last `spec()` reply of `$TEXTUREFLOW_APP`) → the committed snapshot `wzrd_mcp/textureflow_spec.json`.
  A malformed cache logs and falls back. The fallback `wzrd-textureflow` has no `spec()`: it always uses
  the snapshot.
- **Refresh:** `__main__` logs which spec loaded, runs `check_recipes()` (one warning per saved-recipe arg
  the spec doesn't know), and starts `refresh_in_background()` (daemon thread: `modal.Function.from_name(
  app, "spec", environment_name=env).remote()` → atomic cache write). A changed spec is logged and **takes
  effect on the next hub start** (no live re-registration). `NotFoundError` = an app without `spec()`.
- **Snapshot upkeep:** after a lab spec change ships, `uv run python -m wzrd_mcp.tf_spec --snapshot`
  (from the deployed app) or `--snapshot <lab>/tflab/spec.yaml`, and commit it. `python -m wzrd_mcp.tf_spec`
  prints what's loaded.
- **A new arg or choice needs no WZRD code** — only the craft skill (`wzrd_mcp/skills/textureflow/SKILL.md`)
  and, if it changes a look a lot, the one-line recipe summary (`recipe_tools._look`). Example, lab spec
  2.3.0: `mapping_mode` choice `flow_noise`, args `style_paths` (per-region style sequences) + `style_hold`;
  the tool gains them on the first hub start after the lab deploys it.
- **Studio:** `GET /api/textureflow/spec` (= `tf_spec.public()`: the loaded spec + `source`, `app`,
  `target`). `RenderPanel.tsx` reads its knobs' ranges, steps and defaults from it (length, motion, shape
  strength inside the curated 0.35–0.65 sweet spot, region size snap, whether Draft can send `draft`);
  until it loads, TextureFlow can't be started.
- Test: `python test.py textureflow` §9 + `spec_sources` (schema = spec + WZRD args, recipe arg dropped
  with a warning, cache vs snapshot, refresh, fallback app).

## The WZRD client

`wzrd_mcp/tools.py`:
- **`texture_flow`** — `modal.Cls.from_name($TEXTUREFLOW_APP or "textureflow", "TextureFlow",
  environment_name=$WZRD_MODAL_ENV or "wzrd")().render`. Every spec arg defaults to `None` and **only
  caller-set args are sent**, so the server stays the single source of defaults. Media args: an http(s) URL
  passes through; a local path is read into `files` under its basename (deduped `name_2.ext`, photos made
  upright first). `output_video` lands in the media library (`library.output_path`, `~/WZRD/media`, Studio
  target folder else `inbox/`) with an engine HAP clip beside it; the three previews go to
  `.wzrd/intermediates/<job>/`; the reply is those paths plus `seed`, `params`, `request_id`, `timings`.
  Cost = `timings.total_s` (+ `restore_s` on cold starts) × `_TF_USD_PER_S` (RTX PRO 6000 list price),
  checked against the daily spend cap first. Args and their docs are generated (above). `recipe=<name>` fills unset args and style images from a
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
- **Draft → pick → finish** (lab spec 2.2.0: args `draft` + `finish_from`; server side = the lab's
  `docs/reference/deploy.md` "Draft bundles"). A draft (`draft=true`, the final's args) renders only the
  final's first pass; every 2.2.0 render (draft or full) leaves a bundle on the backend that rendered it for
  24 h, and `texture_flow(finish_from=<request_id>)` runs only the upscale / ESRGAN / RIFE steps from it.
  WZRD's side, all in `_texture_flow`:
  - **The id is kept.** The mp4 is named `texture_flow_<request_id>.mp4` (the id carries the timestamp), and
    every record gets `record.textureflow = {request_id, backend, stage (full|draft|finish), draft: {id,
    expires_at} | null, params (resolved args incl. seed), finished_from?}` (via `_provenance.record`).
    The reply adds `draft` / `finished_from` / `finish_warnings` when the server sends them. Pre-2.2.0
    servers send no `info.stage` / `info.draft`: stage `full`, `draft: null`, nothing else changes.
  - **A finish** takes no `images` / `control_input` / `diffusion_mask` / `recipe` (refused); the source is
    looked up by `json_extract(record, '$.textureflow.request_id')` (`_tf_source`). It runs on the
    **source record's backend** (a contradicting `backend=` is refused; no record → the usual default), its
    record's `parent` = the draft's render id (`provenance` accepts a render id, so it survives a deleted
    draft mp4), and its record `args` = the draft's args + the finish args, so More like this and the
    Settings list work. Estimates (`jobs.ESTIMATORS`): draft ≈ $0.10, finish ≈ $0.10 (full − first pass),
    full $0.20, quick $0.03; a local finish $0.
  - The tool's docstring gains a draft → finish paragraph (`_TF_FINISH_INTRO`) only when the loaded spec
    has `finish_from`; before that the args don't exist in the tool at all (they arrive with the spec, on
    the hub restart after the lab deploys 2.2.0).
  - Studio: the Render panel's Draft sends `draft: true` (spec ≥ 2.2.0, else the old low-steps/no-upscale
    draft); a draft's record gets **Upscale · N h** (→ `finish_from`) or, once expired, **re-draft** (same
    args + seed + `draft`). [operator-ui.md](operator-ui.md) "Render panel".
  - Test: `python test.py textureflow` §1 (pre-2.2.0 record) and §10 (a 2.2.0 tool: Modal + local draft →
    finish, routing, refusals, lineage after the draft's mp4 is deleted).
- Without `draft` (spec < 2.2.0), quick tests are `use_upscale`/`upscale_esrgan` false; the server defaults
  would make each a full-HD render. **ESRGAN runs whenever `upscale_esrgan` is true, even with
  `use_upscale=false`.**

## Show-day traps

- **A redeploy discards the GPU memory snapshot.** For `textureflow` the lab's CI check render retakes it
  right after each deploy; still, prime the endpoint (a render or `textureflow_warm`) before a show, and
  don't merge to the lab's `main` during one — that ships to production minutes later.
- **Rollback** of a bad live deploy: either `TEXTUREFLOW_APP=wzrd-textureflow` here (instant, no deploy),
  or `uv run modal app rollback textureflow --env wzrd` from the lab.
- Timings, GPU choice and per-stage costs: the lab's `docs/reference/benchmarks.md` (on RTX PRO 6000 a 15 s
  HD render ≈ 166 s / ≈ $0.14 after the RIFE bump; a small draft ≈ 20–40 s).
