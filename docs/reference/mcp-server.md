# The WZRD MCP server (`wzrd_mcp/`)

FastMCP server exposing the `wzrd` toolkit, the hosted generation models,
TextureFlow and (locally) the engine authoring slice to an agent — Claude Code
over streamable HTTP (`127.0.0.1:8787/mcp`), Claude Desktop over stdio. The
HTTP process is also the **Studio hub** (next section): it owns the media
library database, the render jobs and the spend cap, and serves the app's
JSON API (`/api/*`) beside `/mcp`. Pins:
**FastMCP 4 / MCP SDK 2.x** (`pyproject.toml` `mcp` extra; the parked
`modal_app.py` image carries the same pins). Same patterns as the motherbrain
MCP (`~/Documents/GitHub/agent_frameworks/motherbrain/docs/reference/mcp_server.md`),
minus everything multi-tenant: no identity, no auth on the local path.

## Context budget: three levels of disclosure

The agent carries ~25 tools, so craft knowledge stays out of the always-on
context and loads on demand. This replaced the old Mastra agent's 250-line
system prompt (`~/Documents/GitHub/mastra/src/mastra/agents/wzrd.ts`), whose
persona, workflow and prompt formulas now live in the skills.

| Level | Where | Cost |
|---|---|---|
| 1. Always on | `server.py::_INSTRUCTIONS` — what WZRD is, the projection physics, the local-file contract, the one-line skills index (generated) | ~650 tokens |
| 2. On demand | `skills/<name>/SKILL.md` — one workflow or craft guide | loaded per job |
| 3. Deeper | `skills/<name>/references/*.md` — named by the SKILL.md body | loaded per branch |

Tool descriptions carry only the **contract** (args, ranges, traps, cost) and
point at a skill for craft ("load skill `image-prompting` before writing a
prompt"). Recipes, use-case lists and prompt formulas belong in skills, never
in a docstring.

## Skills

Agent Skills format: `skills/<name>/SKILL.md` with YAML frontmatter `name`
(must equal the folder), `description` (**quote it** — a `: ` inside an
unquoted value breaks YAML and the fallback parser silently drops
`metadata`), and `metadata.kind: workflow | guide`. Served three ways from
one source (`skill_catalog.py` scans, `skill_tools.py` serves):

- **`load_skill(name, file?)` tool** — works on every client. The skills index
  is appended to its description too, for clients that hide server
  instructions.
- **Resources** `skill://<name>/SKILL.md`, `skill://<name>/_manifest`,
  `skill://<name>/{path*}` — FastMCP's `SkillsDirectoryProvider`, the standard
  skills-over-MCP shape.
- **Prompts** — one per `kind: workflow` skill (`run_sheet`,
  `projection_story`, `surface_prep`, `region_animation`, `live_engine`);
  Claude Code lists them as `/mcp__wzrd__<name>`. They only tell the agent
  to load the skill and start.

Current skills: workflows `run-sheet` (gig S1: venue photos → canvas →
regions → pack → clip library from recipes → show checklist; the entry
point for a venue day, delegating to the others), `projection-story` (the
old Mastra agent's keyframe → video → loop flow), `surface-prep` (photos →
`rectify_surface` → pinned `surface_lit` / `surface`; `run-sheet` steps 2–4
chain assets through segmentation and regions),
`region-animation` (+ "A story on one region": keyframes from a region's
`region_surface_image` with `aspect_ratio="auto"`, `region=` on every call,
`to_hap(region=)` after stitching), `live-engine` (+ C5 clip bindings and
the effects library); guides `image-prompting` (+ `references/anti-slop.md`),
`video-prompting`, `textureflow` (+ style recipes, `region=`,
`check_frames`). Skills name tools directly, so **renaming or
swapping a tool means grepping `skills/` too** (e.g. the pending Kling →
MiniMax video swap touches `video-prompting` and `projection-story`).

## The tool registration stack

Every tool goes through `server.tool(title=, tags=, read_only=, destructive=,
idempotent=, open_world=)` — never `@mcp.tool` by hand:

`mcp.tool(name=fn.__name__, output_schema=None, timeout=<tools_config>, annotations=ToolAnnotations(...))`
→ `_text_result.jsonify_result` → `_log.logged_tool` → body.

- **Text-only results.** A dict result becomes ONE compact JSON text block;
  without this FastMCP ships text + `structuredContent` (the payload twice).
  List results (engine replies, `view_media`) pass through as content blocks
  (`fastmcp.utilities.types.Image` for images). `test.py mcp` fails any tool
  that advertises an output schema.
- **`logged_tool` owns logging, errors, scope and records**: call/result/error
  lines on stderr (never stdout — stdio transport), request timing, any
  non-`ToolError` exception re-raised as `ToolError(str(exc))`, the
  `asset:<name>` expansion (next section), the call's
  scope (contextvars: `project.call_project`, `library.current_dest`,
  `jobs.current_job`), a `jobs` row for render tools, and the render record
  of every output of a dict result (below). Tool bodies carry no
  try/except/log boilerplate; raise `ToolError` for an actionable refusal.
- **Injected params.** `tool(project_arg=True)` adds `project=` (venue,
  region and generator tools); `tool(render=True)` adds `dest=`, `wait=`,
  the tag `render` and a job row per call (`nano_banana`, `texture_flow`,
  `kling_*`, `to_hap`); `dest_arg=True` adds `dest=` alone (`extract_frames`,
  `concat_videos`). They are spliced into the published signature and
  stripped before the body runs, so bodies never see them.
- **Annotations** are MCP behaviour hints: `open_world` = paid/remote call
  (nano_banana, kling, texture_flow), `destructive` = deletes or overwrites
  user-visible state (engine `remove_*`, `set_scene`), `read_only` = no side
  effects.
- **Structured args must be sayable.** A bare `dict` param publishes an
  object schema with no properties, and a grammar-constrained decoder can only
  emit `{}`. Declare the keys: `Annotated[dict[str, Any], Field(json_schema_extra=SCHEMA)]`
  (see `_BINDING_SCHEMA` in `engine_tools.py`, `_LAYERPACK_TAGS_SCHEMA` in
  `tools.py`).
- **Optional inputs are `Optional[...] = None`**, not `""` sentinels.
  FastMCP 4 publishes `additionalProperties: false` on every input schema.
- **Docstring `Args:`** are lifted by FastMCP into per-parameter schema
  descriptions; keep them one line, contract only.

## Operator gate: `tools_config.json`

The single source of per-tool `{enabled, timeout}`. A registered tool without
an entry runs enabled at 120 s and warns at boot. Disabled tools are removed
after import; one that survives removal **fails the boot**. Engine tools only
register when `websockets` is installed (`.[engine]` extra), so a cloud image
never carries them. The parked Modal image mounts `tools_config.json` and
`skills/` explicitly — `add_local_python_source` ships `.py` files only.

## The project folder and its locked canvas

`project.py`: a project (a **show**) is one folder under `projects/` that
doubles as a render-core scene directory (`scene.json`, `effects/`,
`packs/<pack>/`). Venue tools reserve their destination first
(`output_path`/`output_dir`) so libraries write straight into it;
generators write to the media library instead (next section), unless
`dest="project"` (→ `generated/`) or `dest="project/<folder>"` (→ that
folder, one slug segment — e.g. segmentation renders).

**The folder map** (`project.LAYOUT`, show-setup G4) is fixed so the UI can
list folders and whoever wrote a file, it shows up in the right place:
`captures/` raw photos · `surfaces/` rectify runs · `segmentation/` colour
maps · `regions/<run>/` mask runs · `styles/` · `generated/` · `video/` ·
`packs/` · `prompts/` · `effects/`. `project.describe(name)` returns it
(abs path + what) with the show's assets; `set_project`, `list_assets` and
`/api/project` all reply with it. `project.set_project` creates every
folder of the map up front (empty dirs don't count as content for the
canvas lock), so Finder, Studio and the wizard see the same slots; the
UI's `MediaGallery` still treats a missing folder (an older show) as empty.

**Named assets** (`assets.py`, `<project>/assets.json`, contract A1): a show
names its key files once — `ROLES` are the well-known ones the setup wizard
pins: `photo_lit`, `photo_ambient`, `surface_lit` (rectified lit crop:
segmentation + ControlNet input), `surface` (dark surface: Nano Banana
`image_1`, subtraction background), `surface_ambient`, `segmentation`. One
name → one file; the hub is the only writer (`/api/project/assets`, MCP
`pin_asset`; temp + rename under a lock). Pinning indexes the file and
stores its render id + fingerprint, so resolution follows a moved file:
stored path on disk → the id's / fingerprint's current file in
`library.db` → `exists: false`. A "Save as" copy keeps the source show's
absolute paths, so resolution first tries the same relative path inside
*this* show. **`asset:<name>` works as any media argument of any tool**:
`logged_tool` expands every str arg (and str in a list arg) that starts with
`asset:` against the call's project *before* the body runs and before the
args are captured, so bodies and render records only ever see real paths;
an unknown name is a `ToolError` listing the show's assets. Unpinning never
touches the file — an agent still can't delete.

**Prompt templates** (`prompts.py`, contract P1): `<kind>/<name>.json`
(`{schema, kind, name, prompt, notes, updated, from}`) in two scopes — WZRD
(`library/prompts/`, `WZRD_PROMPTS_ROOT` overrides, resolved like
`WZRD_RECIPES_ROOT`) and the show (`<project>/prompts/`). Listing puts the
show's first, `default` first per scope. Shipped: `segmentation/default`
(the old `prompts.yaml` text, region count as `{n_regions}`) and
`mask/default` (skill `image-prompting`'s non-reflective mask). `{n_regions}`
is the one placeholder and the *caller* fills it with a plain replace —
prompts may hold other braces. MCP: `list_prompts`, `save_prompt` (scope
`"wzrd"` only on Xander's word).

**`rectify_surface`** (`tools.py` wrapper → `wzrd/prepare_surface.py::rectify_surface`,
portable): lit photo (projector full white, whole frame) + optional ambient
photo + optional manual `corners` → one **run** in `surfaces/`:
`<run>_lit.png` (rectified lit crop), `<run>_surface.png` (darkened ambient
aligned to the lit crop, else the darkened lit crop), `<run>_ambient.png`
(aligned, undarkened; absent without ambient), `run` =
`YYYYmmdd-HHMMSS[-n]`. Always exactly the canvas (legacy show: 1920x1080),
the whole quad rectified. Manual corners (normalised TL, TR, BR, BL in the
lit photo — `wzrd/detect.py`'s `corners=` override) skip detection and are
used as given, no margin; the reply's `corners` is the quad actually
rectified, so auto → nudge → manual round-trips. Why three files, not one:
the lit crop and the dark surface are different pins with different jobs.
`prepare_surface` stays for old callers. `extract_color_regions` entries
also carry `color` (the map's hex) and `area_fraction`, so a UI can show a
swatch per region.

**Which project a call uses — no per-chat memory.** A call's `project=`,
else the show open in the app (the Studio selection's `project`), else
`WZRD_PROJECT`, else `"default"`. `set_project(name)` creates/opens a show
and locks its canvas but sets nothing sticky; its reply says to pass
`project=` from then on. Why: Claude Code speaks MCP 2026-07-28, which is
sessionless — no `Mcp-Session-Id`, no per-chat marker in headers or
`_meta` (captured 2026-10-15) — so the hub can't tell two chats apart, and
a shared sticky value would let one chat redirect another's output. Decided
with Xander 2026-10-15 (amends the Studio plan's S14).

**`project.json`** (gig contract C1, written by the MCP, read by the app):
`{ "schema": 1, "canvas": [w, h] }` (an old `spend_cap_usd` key is ignored). Absent = a
**legacy project**: tools size outputs from their own args as before.

- **The canvas** is the one pixel frame every surface, segmentation map, mask
  and pack in the project is authored in. `set_project(canvas=…)` sets it
  once — an aspect name (→ its 1K size) or an exact Nano Banana native size
  (`project.native_canvases()`; 2K/4K allowed). Same value again is a no-op;
  a different one, or a canvas for a folder that already holds files, is an
  error. Why Nano Banana's sizes: its surface edits then come back at
  exactly the canvas, so nothing is ever cropped or rescaled ("no drift by
  construction"). The engine scales canvas → projector raster and the
  4-corner alignment absorbs the residual (1.792 vs 1.778 for 16:9).
- **With a canvas**, the sizing args of `detect_projection_surface`,
  `darken_surface`, `prepare_surface`, `reproject_video` and
  `build_layerpack` are refused (`_canvas()` in `tools.py` names the
  offending args); outputs are the canvas exactly. Detection rectifies the
  whole lit quad to the canvas (`detect(output_size=…)`, corners untouched —
  the quad *is* the projector raster). `extract_color_regions` /
  `build_layerpack` refuse a `surface_image` that isn't canvas-sized.
  `nano_banana` takes its default resolution tier and t2i aspect from it.
- **The aspect rule (D8), everywhere:** an image a model made from a
  reference is mapped back into the reference's frame by **undoing the
  model's crop** — `wzrd.utils.uncrop_to_size` / `uncrop_array_to_size` /
  `uncrop_box`: uniform scale to fit, centred, bands filled from the
  reference or black. Used by `align_to_reference` (subtraction), islands
  (seg map → surface/canvas), layerpack (masks → surface frame), reproject
  (clip → its region box). Photos *into* a canvas are the other direction:
  `conform_to_size` (cover-crop + resize). Plain resizes remain only where
  the mapping really is a raster scale (surface/canvas → pack/projector
  resolution). Measured on real keyframes: residual 4.1 px → 0.12–0.19 px.
- **Region boxes** (`source_box` from `extract_color_regions`) are padded to
  a Nano Banana native aspect (`islands.STANDARD_ASPECT_RATIOS` =
  `NANO_BANANA_SIZES`), and the `region_surface_image` crop has the box's
  aspect. A story or texture for one region (a tall pillar) is rendered from
  that crop with `aspect_ratio="auto"` → it fits its box back within ~1 px.
- **Photos arrive upright.** `file_io.resolve_input` — the input side of
  every tool — swaps a HEIC, or any photo with an EXIF rotation tag, for a
  decoded, rotated temp copy (`wzrd.utils.upright_copy`; `pillow-heif` is a
  core dependency). PIL ignores EXIF rotation and OpenCV can't read HEIC, so
  without this an iPhone portrait reached tools sideways. TextureFlow's
  byte packer and Kling's aspect pick apply it too (they read files
  directly). `import_photos` brings photos into `captures/`: the newest in
  `~/Downloads` (AirDrop; arrival = birthtime) or given paths, upright, and
  `conform=True` cover-crops them to the canvas — `conform=False` for
  night/day detection pairs, which need the whole frame.
- **Fresh projects boot.** The engine refuses to start without a scene +
  pack, so `build_layerpack` writes a starter `<project>/scene.json`
  (`pack` → this pack, bpm 120, no bindings) when none exists — opened with
  `"x"`, never overwriting one. It also carries each mask's `source_box`
  (from the `islands.json` beside it) into `pack.json` (contract:
  [render-engine.md](render-engine.md) §2.2).
- **`extract_color_regions` outputs:** `region_mask` is the full-canvas mask
  (what `build_layerpack` takes); `region_crop_mask` / `region_surface_image`
  are cut to the region's box (TextureFlow `diffusion_mask`, Nano Banana
  reference). Before 2026-10-10 `region_mask` pointed at the crop, so packs
  built from it stretched crops over the whole canvas.

## The hub, the media library and render records (Studio H1)

**The hub** (`hub.py`, `python -m wzrd_mcp`): one HTTP process per machine,
bound to `127.0.0.1:8787`, standalone from the engine. `hub.lock`
(`{pid, port, started}` in the library's `.wzrd/`) makes it a singleton —
a second start that finds a hub answering `/api/hello` exits 0, so the
app's spawn race is harmless; a stale lock is taken over. The lock is
released inside uvicorn's graceful shutdown (`__main__.HubServer`):
uvicorn ≥ 0.29 re-raises the SIGTERM it caught once done, so nothing after
`run()` (nor `atexit`) runs on a signal. Shutdown also ends open
`/api/jobs` long-polls at once (`hub.STOPPING`), so a restart takes ~1 s. It exits after
`--idle-exit` s (default 3600) with no request, no in-flight call and no
running job; the long-lived `GET /mcp` stream a connected client holds open
doesn't count (`hub.Activity`). The app spawns it detached
(`<abs uv> run --directory <repo> --extra mcp --extra engine --extra islands
python -m wzrd_mcp`); Claude Code attaches with
`claude mcp add --transport http wzrd http://127.0.0.1:8787/mcp`. A
`--stdio` server shares the database (WAL) but its jobs die with it.

**`/api/*`** (`hub.py`, Starlette `custom_route`s on the same app): thin
JSON over `library.py` / `jobs.py`, errors `{error}` with 400/500.
`hello`, `roots` (+`/remove`), `library/{list, item, index, label, move,
rename, delete, delete_scored, signal, tags}`, `thumb?id=`, `selection`,
`jobs` (long-poll on `version`, or `since`), `jobs/item`, `jobs/cancel`,
`render` (`{tool, args, count}` → each a `wait=false` call of the render
tool, `by: "app"`), `estimate`, `spend`, `recipes`, and for show setup:
`project` (`?project=` → folder map, assets, roles, canvas choices,
`has_scene`, existing shows), `project/assets` (pin / `path: null`
unpins), `tool` (`{tool, args}` → any non-render tool run synchronously,
its JSON back — refuses `render`-tagged tools ("use /api/render") and
`destructiveHint` ones, so the app never needs a route per setup tool),
`prompts` (GET/POST), `trash`, `trash/restore`, `trash/flush`. The exact
shapes are the route functions — read `hub.py`; the webview's typed client
for the show-setup routes is `wzrd-app/src/api/project.ts`.
Destructive operations (delete) exist **only** here, never as MCP tools: an
agent can't delete by construction.

**Deletes are soft** (show-setup T1/G6): `library.delete` (and
`delete_scored`/`purge`, which go through it) moves each file to
`<media_root>/.wzrd/trash/<t_id>/<name>` — a rename on the media root's
device, else a hidden `.wzrd-trash/` beside the file — marks its `files`
row `missing`, and logs a `trash` row; the reply adds `trashed: [{id,
path}]`. A hub thread (`hub.start_trash_flusher`) flushes entries older
than `library.TRASH_TTL_S` (30 min) every 60 s, overdue ones at start.
Flush unlinks and sets `purged` once the render has no file left; restore
moves the file back (`_restored` before the extension if the path is
taken) and re-indexes it — the embedded tag brings the same render id, and
thumbnails were never deleted (they're keyed by render id). Why: Xander
wanted accidental deletes recoverable; records were already kept (S7).

**The library** (`library.py`): home `~/WZRD/media` (`WZRD_MEDIA_ROOT`),
new renders in the Studio *target folder* (selection `target`) else
`inbox/`, extra **roots** indexed in place (never moved). `.wzrd/` holds
`library.db`, `thumbs/<render id>.jpg` and `intermediates/<job>/`
(TextureFlow's preview videos — scratch, not library items). Generators
reserve paths with `library.output_path`; `dest="project"` sends one call's
outputs to the project's `generated/`, `dest="project/<folder>"` to that
project folder, an absolute folder sends them there.
`logged_tool` resolves the folder **once, at call start** and pins it as an
absolute path (`library.current_dest`): a `wait=false` render that finishes
after the operator opened another Studio folder still lands where it was
started (S1).

**`library.db`** (SQLite, WAL; the migrations at the top of `library.py`
are the schema — v1 Studio, v2 the trash; version in `PRAGMA user_version`,
migrations run on open). The hub writes; the app reads it `mode=ro` to
browse while the hub is down (the Rust reader ignores `user_version`).

| Table | Key | Holds |
|---|---|---|
| `renders` | `r_<ulid>` (a WZRD render) or `fp_<fingerprint>` (a foreign file) | the creation record (JSON), denormalised `tool/created/project/region/job/parent`, labels `score` (0–9, Xander only) / `tags` / `agent_tags` / `notes`, `signals` (deck adds, plays, live seconds), `purged`, `thumb` (abs path) |
| `files` | absolute `path` | `dir` (abs), `render_id`, `variant` (`master`/`hap`), `fp`, size/mtime, probe (`kind`, `codec`, `w/h/frames/fps/duration_s`), `last_seen`, `missing` |
| `jobs` | `j_<ulid>` | render-tool calls: args, `by`, `pid`, `started`/`ended` (epoch s), `ok`, `error`, `outputs`, `est_usd`, `duration_s`, `cancelled`, `dest`, `size_key`, `remote`, `project` |
| `roots`, `selection`, `settings` | | extra roots (home row `home=1`; the projects root is always a row too — show-setup G9, so Studio's tree and the Rust reader show every show's folders; `remove_root` refuses it); the Studio selection `{styles, items, target, project}`; `daily_cap_usd` |
| `trash` (v2) | `t_<ulid>` | soft-deleted files: `path` (where it lived), `trash_path`, `render_id`, `size`, `deleted`, `flushed` (NULL while restorable) |

**Identity lives in the file** (`identity.py`): a render's creation record
is embedded once at creation as the `wzrd` tag — mp4/mov: format metadata
via `ffmpeg -map 0 -c copy -movflags use_metadata_tags` (+~350 bytes, no
re-encode; render-core's MOV reader opens tagged HAP, checked by bench);
png: an iTXt chunk; jpg: a COM segment `wzrd=<json>`. webp/gif/webm/heic
carry no tag and are linked by fingerprint. **Fingerprint** =
`<size>-<sha1(first + last 1 MiB)[:20]>`. A path is never an identity:
labels are keyed by render id, so a Finder rename/move keeps them, and the
record outlives a deleted file.

- **Record** (the tag JSON): `{schema: 1, id, variant, tool, args, inputs:
  [{name, path, fp}], parent: <render id>|null, created, est_usd,
  duration_s, job, project, canvas, region, source_box}`.
- **Written by `logged_tool` → `provenance.record_outputs`** after the body
  returns: an output = an absolute media path in the result, modified
  during the call, not an input, not under `.wzrd/`. Never written by tool
  bodies; `save_recipe` is exempt (recipe files aren't renders). A tool
  that knows more returns a private `_provenance` dict: `est_usd` keyed per
  output path, `args` extras, `region`/`source_box`, `parent`, `paid`,
  `variant_of: {hap path: source path}` — that HAP shares the source's id —
  and `record` (extra top-level record fields: `rectify_surface` stores the
  rectified quad as `corners` + `manual_corners`, so the UI can reload a
  trial's quad and slider args from its record).
- **Indexing on open** (`library.list_folder`): unchanged files (same
  size + mtime) are free; new/changed ones are probed on a 4-thread pool
  and show as `pending` until the next listing. Identity order: embedded
  tag → a known fingerprint (a moved untagged file) → a legacy `.meta.json`
  sidecar (absorbed, left on disk, never written into the file) → foreign
  (`fp_…`). A row whose path vanished but whose fingerprint reappears
  elsewhere follows the file. Vanished files get `missing = 1`; a trash
  flush sets `purged` once a render has no file left.

**Jobs** (`jobs.py`, MCP tools `jobs` / `cancel_job` in `studio_tools.py`):
every render-tool call is a `jobs` row. `wait=false` runs the body as an
asyncio task in the hub (`jobs.detach`) and replies `{job, status:
"running", dest, eta_s}` once the call is *submitted* — it passed its
checks (`check_spend` signals it; `to_hap` calls `jobs.submitted()`) — or
straight away with the error/result if it ends first, so a cap refusal or
bad `region=` still fails synchronously. Remote calls register with
`jobs.set_remote` (`modal|<call id>` via `.spawn` + `.get`; `fal|<endpoint>|<request id>`
via `fal_client.submit` + `handle.get`) so `cancel` stops the billing side
first, then the task; Nano Banana has no remote handle. A job whose process
died without `ended` reads **lost** (pid check on read); cancelling a lost
job still cancels its recorded remote call. ETA = median `duration_s` of
the last 20 successful jobs with the same tool + size args (`SIZE_ARGS`).
Pre-call estimates live in `jobs.ESTIMATORS` (registered by the tool
modules; shared by `check_spend` and `/api/estimate`). A job's `dest` is
the folder its outputs land in — `to_hap` records its source's folder.

**The agent's Studio tools** (`studio_tools.py`, S18 permissions): it may
read the selection (`studio_selection` — "these styles" = its `styles`,
renders land in its `target`), search the render database
(`find_renders`: filters tool / min_score / tag / style — a path matches by
fingerprint wherever the image moved, a word by file name — / recipe /
project / text; best-scored first; `args` shows only what differs from the
tool's published defaults; purged records stay as evidence), tag
(`label_render` → `agent_tags`, notes appended `[agent] …`; the score is
Xander's alone — `library.label(agent=True)` refuses one), and move/rename
(`move_render` / `rename_render`, a render's mp4 + HAP together), and fill
a show's decks (`add_to_deck` writes `<project>/decks.json`, contract L6 in
the Studio plan; the app is the other writer and compare-and-swaps on the
file's mtime, so both write temp + rename). It cannot delete: deletion
exists only on `/api`. Skills `textureflow` and
`run-sheet` teach the loop: selection → `find_renders` before a batch →
`wait=false` + `jobs` → `check_frames`.

**Spend cap** (`jobs.py`): daily and global, default $20/day
(`settings.daily_cap_usd`; `/api/spend` or `set_project(daily_cap_usd=…)`
on Xander's word). Today = local midnight on, the sum of `est_usd` over
`jobs` — a running job counts with its pre-estimate (`check_spend` writes
it onto the job), so a parallel batch can't overshoot. Paid tools call
`provenance.check_spend(<pre-estimate>, allow_over_cap)` before the remote
call; their result carries `spend: "$x of $cap today"` and a warning from
80 %. Pre-estimates vs recorded cost: Nano Banana tier price × images
(recorded: token usage), Kling list price × seconds, TextureFlow $0.20 /
$0.03 without upscales (recorded: billed container seconds × the endpoint GPU's list price, `_TF_USD_PER_S`: $3.03/h RTX PRO 6000;
`backend="local"` — the desktop GPU — is $0 and unpaid, [textureflow.md](textureflow.md)).
Prices: [hosted-models.md](hosted-models.md). `project.json`'s old
`spend_cap_usd` is ignored.

**`region=<layer id>`** on `nano_banana`, `kling_*`, `texture_flow`,
`to_hap`: `provenance.region_meta` looks the id up in the pack the call's
project's `scene.json` points at (else the newest `packs/*/pack.json`) and
records its `source_box` (else `bbox`). Unknown id or no pack → refused
before any spend.

## Playable clips (Studio S8)

The engine plays HAP only, so a render that should reach the wall gets a
HAP `.mov` **beside its source, same stem, same render id** (`variant:
"hap"` — labels are shared; the mp4 stays the master, the HAP is a
disposable cache). `media_tools.make_clip`: a source in a temp dir (a
downloaded URL), under `.wzrd/` or in an unwritable folder sends the HAP to
the library instead; re-running replaces the render's own `.mov` (written
temp + rename, so the engine playing the old one keeps its inode) but never
overwrites a foreign file of that name. A DXV `.mov` gets `<stem>_hap.mov`.
`texture_flow` and `kling_*` run it after download (`hap=True`;
`attach_clip` sets `variant_of`); `to_hap` does it for any video
(`region=` files a source that records none). No `.thumb.jpg` is written
any more — thumbnails live in the library cache. The ffmpeg side is
`wzrd/clips.py`: sides padded (centred, black) to multiples of 4, fps and
frame count kept, audio dropped; codec is a parameter. `extract_frames`
pulls stills (`"first"`, `"last"`, seconds, or `count` evenly spaced) into
the library. HAP is 5–10× an H.264's size: move mp4s, make HAP where it
plays.

**Legacy (pre-Studio) clips** in `<project>/generated/` with C2
`.meta.json` sidecars still work everywhere a record is read: recipes walk
the sidecar chain, `get_scene_context` lists them, indexing absorbs them.

## Style recipes (gig P7, contract C6)

The global TextureFlow look library: `library/recipes/<name>/{recipe.json,
style_N.jpg, preview.jpg}` in this repo (`WZRD_RECIPES_ROOT` overrides —
D16: the default is what a Finder-launched app sees). Code:
`recipe_tools.py`. Hand-curated and agent-saved recipes share the folder.

- **`recipe.json`**: `{schema: 1, name, when_to_use, notes, prompt, images:
  [file names beside it, or URLs], args: {…}, source: {clip, project, seed,
  size, region, rendered, saved} | null}`. `args` is the *look* only:
  `_TARGET_ARGS` strips media (`images` sits beside it; `control_input` /
  `diffusion_mask` belong to the region), `width`/`height`, `seed` (kept in
  `source`), and WZRD-local args. `prompt` = the image prompt the style
  images were made with, lifted from their own sidecars when known.
- **`save_recipe(name, from_clip, when_to_use, …)`** walks the clip's
  render record (the library database, else a legacy sidecar chain;
  `parent` links to the `texture_flow` call — a HAP shares its mp4's
  record directly), copies the style
  images downscaled to 768 px JPEG (IP-Adapter sees 224 px; ~100 kB each,
  so they're committed) plus a 640 px preview frame. Folder names are
  slugified lowercase; written via a temp dir + rename; an existing name
  needs `overwrite=true`.
- **`texture_flow(recipe=…)`**: `apply_recipe` merges recipe args as
  defaults under the caller's non-null args (`images` too) and refuses a
  shaped recipe (`use_controlnet1` / `denoise < 1`) without a
  `control_input` — that would render near-black. `images` is optional
  only with a recipe. The *merged* args go into the record (`_provenance`
  `args`), so a clip rendered from a recipe can be saved as a recipe again.
- **`list_recipes(name?)`**: a compact row per recipe (`look` = a one-line
  summary of the args), or one recipe in full with absolute paths.
- `library/recipes/.gitignore` keeps video out of git. Seeded with one
  example, `tree-of-life` (the TextureFlow smoke render's Eden style images
  and args — a demo look, not venue-tuned); `projects/` held no real
  `texture_flow` sidecars to seed from on 2026-10-12.

## File map

| File | Role |
|---|---|
| `server.py` | FastMCP instance + instructions, `tool()` stack, config gate, module imports |
| `skill_catalog.py` / `skill_tools.py` | skills scan + index / `load_skill`, resources provider, prompts |
| `tools.py` | wzrd wrappers: subtraction, surface prep (`rectify_surface`, `prepare_surface`), regions, `texture_flow`/`textureflow_warm`, `build_layerpack`, `simulate_view` |
| `media_tools.py` | `view_media` (downscaled images; one contact sheet per video), `concat_videos` (→ `wzrd/concat.py`), `to_hap` / `extract_frames` + the C3 clip helpers (→ `wzrd/clips.py`), `check_frames` (→ `wzrd/check_frames.py`: numeric clip QC — flash rate, loop seam, haze, lit share, still-vs-surface shift; thresholds and their reasons in the module docstring) |
| `google_tools.py`, `fal_tools.py` | hosted models — [hosted-models.md](hosted-models.md) |
| `engine_tools.py` | engine authoring slice — [render-engine.md](render-engine.md) §2.7 |
| `project.py`, `project_tools.py` | the local project folder (a render-core scene dir), `LAYOUT`, `project.json` + canvas, which project a call uses; tools `set_project`, `import_photos`, `pin_asset`, `list_assets`, `list_prompts`, `save_prompt` |
| `assets.py`, `prompts.py` | named project assets (`assets.json`, `asset:` expansion, resolution); prompt templates (WZRD + show scope) |
| `_text_result.py`, `_log.py`, `_keepalive.py` | result encoding; logging/errors/timing + call scope, injected params, job rows, record hook; 15 s progress pings during remote calls (all paid tools, TextureFlow included) |
| `hub.py` | the hub: `hub.lock`, idle exit, trash flusher, `/api/*` routes |
| `library.py` | media root + roots + selection, `library.db` schema/migrations, indexing, labels, move/rename, soft delete + trash, generator output routing (`dest`) |
| `identity.py` | fingerprint, the embedded `wzrd` tag (read/write), probe, thumbnails, ULIDs |
| `jobs.py` | job rows, status (lost = no `ended` + dead pid), ETA, estimators, `wait=false` detach + cancel, the daily spend cap |
| `studio_tools.py` | the agent's view of the Studio: `jobs`, `cancel_job`, `studio_selection`, `find_renders`, `label_render`, `move_render`, `rename_render` |
| `provenance.py` | render records for every output (`record_outputs`), legacy sidecar reads, `region=` → pack layer box |
| `recipe_tools.py` | style recipe library: `list_recipes`, `save_recipe`, `apply_recipe` (the `texture_flow(recipe=)` merge) |
| `local_tools.py` | `capture_camera_snapshot` (off by default) |
| `file_io.py`, `auth.py`, `modal_app.py`, `upload.py` | input resolution; parked cloud path |

## Adding things

- **A tool**: write the async body in the fitting module, decorate with
  `@tool(...)`, add a `tools_config.json` entry, mention it in the skill whose
  workflow uses it. A new module is imported in `server.py`.
- **A skill**: a folder with a quoted-description `SKILL.md`; it appears in the
  index, `load_skill`, resources and (if `kind: workflow`) prompts with no code
  change. Restart the server; clients re-read instructions on connect.
- Verify: `uv run python test.py mcp` (boot, text-only results, all three skill
  surfaces, prompts match workflows). Point `WZRD_MEDIA_ROOT` at a temp dir in
  tests so they don't write into the real library.

## Considered, not done

- **Session-scoped tool visibility** (FastMCP `ctx.enable_components`: a skill
  load unhides its tool group via `tools/list_changed`). Hidden tools the
  agent can't see are a discoverability risk, client support for the
  notification varies, and Claude Code already defers MCP tool schemas.
- **MCP Tasks** (`task=True`, spec background tasks) for the multi-minute
  renders: progress keepalives cover today's clients; revisit when Claude
  clients poll tasks.
- **Search transforms** (`BM25SearchTransform`): replaces the catalog with
  search + call proxies — worse UX at ~25 tools.
