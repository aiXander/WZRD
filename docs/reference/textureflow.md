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
  (≈ 1 s; stage → `log_progress` + MCP progress + the queue's progress, under `with_keepalive`) → result → download each output `url` into
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

## The render queue (`tf_queue.py`, `tf_eta.py`)

Every `texture_flow` call is **tracked** by `tf_queue` from the moment it passed its checks until its files are
saved; the decisions behind it (Q1–Q8) and the hub API contract live in
[../TODO/textureflow-render-queue.md](../TODO/textureflow-render-queue.md) while lanes are open.

- **Two lanes, two owners.** *Modal* — the hub owns the order: the call parks on `acquire_modal(job)` and the
  dispatcher (an asyncio task, `_run`/`_step`, 1 s tick) grants the lane's head a slot while fewer than the
  operator's **Parallel GPUs** setting run (`settings.tf_modal_slots`, 1–3, default 1; `TEXTUREFLOW_MODAL_SLOTS` only
  seeds it; `POST /api/tf/modal`). A spawned Modal call can't be reordered, which is why jobs wait here. Lowering
  the setting never cancels a running job. *Local* — the **desktop job server owns its queue**: the call submits as
  soon as `/v1/health` answers (`hold_local` holds it in the hub only while unreachable, phase `waiting_backend`), and
  `lane_pos` = the server's `queue_position` (other clients' jobs included). Reorder goes to the server's
  `/v1/queue/order` (our jobs take our slots in the new order; foreign jobs keep theirs); `/api/tf/queue` merges
  the server's `/v1/queue` (foreign jobs as `{foreign: true, …}`) with the hub's held jobs.
- **Status `queued`** = `jobs.lane_rank` is set (schema v3): Modal = lane index, local = server queue position,
  held = `1_000_000 + i`; NULL once rendering. `jobs._job_dict` adds `backend`, `lane_pos`, `pinned`, `progress`
  for texture_flow (live from `tf_queue.progress()` in the hub, else the last persisted `jobs.progress` JSON).
- **Moves** (`/api/jobs/move`): Modal → local takes the job out of the hub lane (`acquire_modal` returns
  `"local"`, the body re-routes); local → Modal cancels the queued server job, `textureflow_local.collect` sees
  `cancelled` + `moved_away` and raises `tf_queue.Moved`, and the body spawns on Modal from its in-memory `files`.
  A move to Modal passes the spend cap (`Entry.est`). A finish is pinned. The record's `args.backend` says where it
  really rendered.
- **Desktop lifecycle:** the server process always runs; its GPU worker boots on demand and exits after 10 idle
  min by itself (health `worker: idle`; `waiting_gpu` while a lab batch holds `gpu.lock` — the lab's
  `docs/reference/local-gpu.md`). The hub never stops it. Only when the desktop answers SSH but the service
  doesn't (`service_down`) does it run `textureflow-ctl start` (`/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=4
  $TEXTUREFLOW_LOCAL_SSH '~/.local/bin/textureflow-ctl start'`, absolute ssh because the app-spawned hub has no
  shell PATH; `~` because a non-login PATH lacks `~/.local/bin`), at most once a minute while jobs are held, or on
  `POST /api/tf/local {action: "start"}`. **Without `TEXTUREFLOW_LOCAL_SSH`** nothing can wake the desktop, so a
  local call keeps the old fail-fast health check (≈ 2 s, before the spend check). Local states:
  `LocalLane.refresh` (health ≤ 2 s cached, concurrent callers share one probe; ssh status ≤ 20 s).
- **Old-server tolerance:** the `meta` part (`{client: "wzrd", label}`) 400s on a server older than the queue
  routes → resent without it, remembered (`textureflow_local._meta_ok`); no `/v1/queue` → the local view falls back
  to the hub's own jobs and local reorder answers 400.
- **Hub restart (Q8):** `HubServer.startup` → `tf_queue.adopt_all()` re-adopts unfinished local jobs whose pid is
  dead (`jobs.remote = local|<server id>`), as hub tasks running `tools.adopt_local` (follow → save → record →
  `jobs.finish`). Modal jobs still queued in the hub read `lost` (persisting that lane is open).
- **Progress.** Local: the v1 status poll at `wait=1` (the long-poll returns early only on state/stage changes;
  step ticks need the poll) → `tf_queue.local_status`. Modal: when the deployed `spec()` reply lists
  `capabilities: ["progress_dict"]` (`tf_spec.capabilities()` — a fetched reply or the cache, **never the committed
  snapshot**, which would claim it for the fallback app), `render.spawn(..., progress_key=<job id>)` and a 1 s poll
  of Modal Dict `textureflow-progress` (env `wzrd`), popped at the end; otherwise estimate-only. Phases
  `queued → waiting_backend (held) → starting → rendering → downloading (local) → saving`. Server stage names:
  a finished lap reports its own name (the *next* lap runs), ticks report `KSampler` (first pass, then the latent
  pass), `esrgan`, `rife` — `tf_eta.Tracker.on_stage` maps them.
- **ETA (`tf_eta.py`, Q6):** per-lap `c + k·x` (x = pixel-frames: sample × steps, latent pass at
  `max(upscale_resolution²·9/16, first pass)`, ESRGAN/RIFE/encode at output size), least squares per backend over
  priors + this library's records (`record.textureflow.stage_s` + sizes, Q7; refit ≤ 1/min). Priors: 3090 =
  `wzrd_mcp/tf_eta_3090_samples.json` (the lab's 272-render export) + one built-in bench (the only ESRGAN point);
  Modal = two RTX PRO 6000 anchors from the lab's `benchmarks.md`. Live pct = done laps + running lap × step/total
  (or elapsed/predicted ≤ 95 %), remaining × the observed speed ratio. Constant overheads (`OVERHEAD_S`) are
  guesses, not refit — tiny renders read long (a 2 s local test: predicted ≈ 13 s, real ≈ 5 s).
- **Records (Q7):** `record.textureflow` also keeps `stage_s`, `gpu`, `first_pass_size`, `output_size`,
  `n_frames`, `n_output_frames`, `steps`, `timings`. Library items carry `tf: {stage, request_id, backend,
  expires_at}` (`library._tf_brief`) for the grid's draft badge.
- Test: `python test.py textureflow` §11–§16 (Modal lane + slots setting, reorder/cancel/move/pinned through the
  real `/api` routes on an ASGI client; the fake server's `/v1/queue` with a foreign job; a legacy server; hold +
  fake ssh start; a fake progress Dict; re-adoption) + `eta_sanity`.

## Show-day traps

- **A redeploy discards the GPU memory snapshot.** For `textureflow` the lab's CI check render retakes it
  right after each deploy; still, prime the endpoint (a render or `textureflow_warm`) before a show, and
  don't merge to the lab's `main` during one — that ships to production minutes later.
- **Rollback** of a bad live deploy: either `TEXTUREFLOW_APP=wzrd-textureflow` here (instant, no deploy),
  or `uv run modal app rollback textureflow --env wzrd` from the lab.
- Timings, GPU choice and per-stage costs: the lab's `docs/reference/benchmarks.md` (on RTX PRO 6000 a 15 s
  HD render ≈ 166 s / ≈ $0.14 after the RIFE bump; a small draft ≈ 20–40 s).
