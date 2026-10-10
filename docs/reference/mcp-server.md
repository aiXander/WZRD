# The WZRD MCP server (`wzrd_mcp/`)

FastMCP server exposing the `wzrd` toolkit, the hosted generation models,
TextureFlow and (locally) the engine authoring slice to an agent — Claude Code
over streamable HTTP (`:8787/mcp`), Claude Desktop over stdio. Pins:
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
- **Prompts** — one per `kind: workflow` skill (`projection_story`,
  `surface_prep`, `region_animation`, `live_engine`); Claude Code lists them as
  `/mcp__wzrd__<name>`. They only tell the agent to load the skill and start.

Current skills: workflows `projection-story` (the old Mastra agent's
keyframe → video → loop flow), `surface-prep`, `region-animation`,
`live-engine`; guides `image-prompting` (+ `references/anti-slop.md`),
`video-prompting`, `textureflow`. Skills name tools directly, so **renaming or
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
- **`logged_tool` owns logging, errors and provenance**: call/result/error
  lines on stderr (never stdout — stdio transport), request timing, any
  non-`ToolError` exception re-raised as `ToolError(str(exc))`, and the C2
  sidecars for every dict result (next section). Tool bodies carry no
  try/except/log boilerplate; raise `ToolError` for an actionable refusal.
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

`project.py` keeps every output of a session in one folder under
`projects/` that doubles as a render-core scene directory (`scene.json`,
`effects/`, `packs/<pack>/`, plus `surfaces/`, `generated/`, `regions/`,
`video/`, `captures/`). Tools reserve their destination first
(`output_path`/`output_dir`) so libraries write straight into it.

**`project.json`** (gig contract C1, written by the MCP, read by the app):
`{ "schema": 1, "canvas": [w, h], "spend_cap_usd": 20 }`. Absent = a
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

## Provenance sidecars and the spend cap

**Sidecar** (gig contract C2): `<output>.meta.json` next to every media file
(`.png .jpg .jpeg .webp .gif .mp4 .mov .webm`) a tool call created inside the
active project — `{ "schema": 1, "tool", "args" (non-null client args +
tool extras: model, used seed), "inputs": [local paths resolved, URLs],
"parent": path|null (default: first local input), "created": ISO UTC,
"est_usd", "region": layer id|null, "source_box": [x0,y0,x1,y1]|null }`.
Reserved for the post-gig toolbox: `"rating"`. Readers ignore unknown keys;
the app's clip grid reads `region`, `source_box`, `est_usd`, `tool`,
`created` and the parent's sidecar (`wzrd-app/src-tauri/src/clips.rs`).

- Written by `logged_tool` → `provenance.write_sidecars` after the body
  returns, never by tool bodies. An output = an absolute media path in the
  result, inside the project, modified during the call, not an input.
  Failure to write is logged, not raised.
- A tool that knows more returns a private `_provenance` dict (popped before
  the agent sees the result): `est_usd` **keyed per output path** (so
  TextureFlow's three preview videos count $0 and the cost lands once, on
  `output_video`), `args` extras, `region`/`source_box`, `paid`.
- **`region=<layer id>`** on `nano_banana`, `kling_*`, `texture_flow`:
  `provenance.region_meta` looks the id up in the pack the project's
  `scene.json` points at (else the newest `packs/*/pack.json`) and records
  its `source_box` (else `bbox` — the engine's own fallback). Unknown id or
  no pack → refused before any spend. The id is only recorded; it doesn't
  change what the model is asked to do.

**Spend cap (D10)**: `spend_cap_usd` in `project.json` (default $20, also
for legacy projects; `set_project(spend_cap_usd=…)` — only on Xander's
word). Spent = sum of `est_usd` over the project's sidecars (rglob per
call; fine at hundreds of files). Each paid tool calls
`provenance.check_spend(<pre-estimate>, allow_over_cap)` before its remote
call and refuses with the running total if it would cross the cap; after
the call its result carries `spend: "$x of $cap"` plus a `spend_warning`
from 80 %. Pre-estimates vs recorded cost: Nano Banana tier price ×
images (recorded: from token usage), Kling list price × seconds (recorded:
the same), TextureFlow $0.20 / $0.03 without upscales (recorded: billed
container seconds × $1.95/h L40S). Prices: [hosted-models.md](hosted-models.md).

## File map

| File | Role |
|---|---|
| `server.py` | FastMCP instance + instructions, `tool()` stack, config gate, module imports |
| `skill_catalog.py` / `skill_tools.py` | skills scan + index / `load_skill`, resources provider, prompts |
| `tools.py` | wzrd wrappers: subtraction, surface prep, regions, `texture_flow`/`textureflow_warm`, `build_layerpack`, `simulate_view` |
| `media_tools.py` | `view_media` (downscaled images; one contact sheet per video), `concat_videos` (→ `wzrd/concat.py`), `check_frames` (→ `wzrd/check_frames.py`: numeric clip QC — flash rate, loop seam, haze, lit share, still-vs-surface shift; thresholds and their reasons in the module docstring) |
| `google_tools.py`, `fal_tools.py` | hosted models — [hosted-models.md](hosted-models.md) |
| `engine_tools.py` | engine authoring slice — [render-engine.md](render-engine.md) §2.7 |
| `project.py`, `project_tools.py` | the local project folder (a render-core scene dir), `project.json` + canvas, `set_project` |
| `_text_result.py`, `_log.py`, `_keepalive.py` | result encoding; logging/errors/timing + sidecar hook; 15 s progress pings during remote calls |
| `provenance.py` | C2 sidecars, spend cap, `region=` → pack layer box |
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
  surfaces, prompts match workflows).

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
