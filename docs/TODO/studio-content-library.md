# Studio: content generation, the media library and clip decks (sprint plan)

> **Status: agreed with Xander 2026-10-15 (grilling session); building. H1, H2, H3 and P1 done.**
> Every decision below is settled. Don't re-litigate them; ask Xander if
> a build finding contradicts one. **Deadline:** everything lands by
> **7 Nov 2026**, decks included, before the gig's dress-rehearsal week
> ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)). **H1 lands
> before ≈ 24 Oct**, when the venue photos arrive and library rendering starts,
> so that every venue render is recorded.
>
> This plan supersedes the gig plan's contracts **C2** (provenance sidecar) and
> **C3** (clip files), its per-project spend cap (D10), and its
> `clipgrid.json` grid (D12). It also absorbs [vj-toolbox.md](vj-toolbox.md)
> phase 4 (in-app generation). Current state it builds on:
> [../reference/mcp-server.md](../reference/mcp-server.md) (project
> folder, sidecars, playable clips, recipes, spend cap) and
> [../reference/operator-ui.md](../reference/operator-ui.md) (rooms, clip
> grid, projects, native preview slots). Read this whole file before starting a
> package.

## What Xander wants

Making content and playing it live are two activities, and the app only
serves the second. Xander will **render and review content from inside the
app**, not mainly from a chat terminal:

1. browse folders of style images and of past renders, including his Resolume
   library (`~/Visuals`, ≈ 199 GB, almost all **DXV**);
2. render images, TextureFlow clips and Kling videos, from a form in the app
   or from a Claude Code session; watch the jobs; cancel the bad ones;
3. **label** renders when he feels like it: a 0–9 score, free tags (moods,
   themes) and notes. Labelling is **opt-in, never mandatory**;
4. **delete freely** to save disk, but keep every render's settings and labels,
   so a database builds up for hill-climbing aesthetics ("which settings
   score ≥ 7 with this style?");
5. make the good clips playable (HAP) **in place**, with no duplicate files;
6. pull clips from the library, including past projects, into a show's
   **decks** (Resolume's stacks of clips) and play them.

## The model

| | **Media library** | **Show** (a project, as today) |
|---|---|---|
| What | every file WZRD made, plus folders indexed in place | one venue's set: pack + regions, `scene.json`, effects, **decks**, looks, masters, alignment |
| Where | home `~/WZRD/media/` (new renders) + **extra roots** indexed in place (`~/Visuals`, `projects/*/generated`, …) | `projects/<name>/`, unchanged |
| UI | new **Studio** room | Prepare / Align / Perform |

- **Shows reference library files in place**, by absolute path. Nothing is
  copied. Deleting a file a show uses breaks that slot, and that is Xander's
  responsibility, as in Resolume. **Never refuse a delete.** Missing files
  are a per-slot warning (the engine already does this) and get fixed by the
  on-demand **Relink missing** button (package D3).
- **A render's identity travels inside the file.** At creation the hub
  embeds a `wzrd` metadata tag in the media file itself: the render id and the
  full creation record. Labels live in a database keyed by that id, **not by
  path**. Renaming, moving and new subfolders never break a record, and the
  record outlives a deleted file. Foreign files (DXV, anything not made by
  WZRD) are identified by a content **fingerprint** instead.
- **The MCP server is the hub.** It is one long-lived HTTP process that owns
  the database, runs render jobs, holds the Studio selection and applies the
  spend cap. The app and every Claude Code session talk to it. Later the
  agent's view of the whole app (and a camera) hangs off the same hub, so
  **don't design anything that assumes the agent and the app can't talk**.

## Decisions (settled 2026-10-15)

| # | Decision | Why |
|---|---|---|
| S1 | **Library home `~/WZRD/media/`**, override `WZRD_MEDIA_ROOT`. New renders land in the folder open in Studio (the *target folder*), else `inbox/` | A Finder-launched app sees no env vars, so the default is what runs |
| S2 | **Extra roots indexed in place.** Nothing is ever moved into the library; Xander's Resolume compositions keep working. WZRD may *add* files to a root (a HAP conversion beside its source) but never moves or rewrites existing ones unless he asks (move/rename/delete in Studio) | Moving `~/Visuals` would break his Resolume projects |
| S3 | **Index on open.** A folder is indexed (fingerprint + probe + thumbnail) the first time Studio opens it; an **"Index this folder"** button indexes a whole tree for search. No always-on background scanner | 199 GB of DXV; pay only for what's looked at |
| S4 | **Identity in the file, labels in the database** (contracts L2, L3). The hub writes the `wzrd` tag once at creation and never rewrites media for a label change. **`.meta.json` sidecars are retired** for new renders; old sidecars are absorbed on first index and left on disk | Rewriting a 30 MB HAP per keypress is slow and races the engine reading it; Xander wants fewer files |
| S5 | **Labels are opt-in**: score `0–9` (higher is better, digit keys; absent = unrated), free **tags** with autocomplete from existing tags, **notes**. Bulk **"delete media scored ≤ N in this folder"**: the records stay | Fast enough to use mid-gig; never a chore |
| S6 | **Implicit signals logged separately** and never overwriting the score: added to a deck, played live, seconds on live | Free extra data for hill-climbing |
| S7 | **Delete = media gone, record kept** (`purged` set, thumbnail kept in the cache). Never refused, even when a show uses the file | Xander's call, as in Resolume |
| S8 | **Make playable = HAP beside the source** (`<stem>.mov` next to `<stem>.mp4`, or next to a DXV original). The mp4 stays as the master; the HAP is a disposable cache ("free space" drops HAP that no deck uses). The HAP carries the same render id (`variant: "hap"`), so labels are shared | No duplicate *masters*; HAP is 2–10× the mp4 and regenerates in seconds |
| S9 | **DXV plays via conversion only** for now: make it playable → HAP beside it. Native DXV decode in the engine is post-gig ([backlog.md](backlog.md)). The engine stays HAP-only (gig D2) | Xander mostly makes new content for the first gig |
| S10 | **Preview monitor in Studio, Resolume-style**: static thumbnails in the grid (no animated thumbs, no autoplay); the selected item plays in one **preview monitor**. HAP plays **natively** in a native slot (the engine decodes it, the same underlay mechanism as Perform's monitor); mp4 plays in the webview's `<video>` in the same frame (WebKit decodes H.264 in hardware); DXV and other codecs show their thumbnail plus "Make playable to preview" | Resolume plays everything through its own engine. Ours decodes HAP only, so mp4 needs the webview |
| S11 | **The MCP server is the hub**: a single instance, streamable HTTP on `127.0.0.1:8787` (not `0.0.0.0` when spawned by the app), standalone from the engine so they can't crash together | The app and the agents need one owner of jobs and the database |
| S12 | **The app spawns the hub** if `GET /api/hello` doesn't answer as a WZRD hub; a lock file prevents a second copy. The hub **outlives the app** (in-flight renders finish) and **shuts itself down after 1 h idle** (no running job, no request). Claude Code sessions need the app opened first; no stdio bridge for now | Xander renders from the app; simplest correct lifecycle |
| S13 | **App ↔ hub over a small JSON API** (`/api/*` Starlette routes beside `/mcp` on the same server, sharing the Python implementations). Agents use MCP tools. Destructive operations (delete, purge) exist **only on `/api`** | Simpler than an MCP-client crate in Rust, and the agent can't delete by construction |
| S14 | **Generators are project-free**: `nano_banana`, `texture_flow`, `kling_*`, `to_hap`, `extract_frames`, `concat_videos` write to the library. Venue tools (`import_photos`, `prepare_surface`, `extract_color_regions`, `build_layerpack`) keep writing into the project. ~~The active project is per MCP session~~ **Amended 2026-10-15 (build finding, Xander's call): no per-chat project.** MCP 2026-07-28 (what Claude Code speaks) is sessionless, so the hub can't tell chats apart. Every project-resolving call passes `project=`; omitted = the show open in the app (selection `project`). `set_project` creates/opens a show and locks its canvas, nothing sticky. `region=` resolves against the call's project. A `dest="project"` arg covers the in-between cases (a segmentation map) | Several parallel Claude sessions share one hub; a shared sticky project would let them redirect each other |
| S15 | **Spend cap is daily and global**: default $20/day, warn at 80 %, the app asks before exceeding (agents get today's refusal + `allow_over_cap`). Summed from the jobs table. "$ today" sits in the jobs pill | Renders no longer belong to a project; a runaway batch is a same-day risk |
| S16 | **Rendering from the app = a Render panel** in Studio: recipe picker, style tray, target region, key knobs (strength, length, seed), count, cost estimate, Go. Plus **More like this** (same args, new seeds), **Make playable** and **Cancel** for running jobs. Chat for creative work stays a Claude Code terminal; in-app chat is post-gig ([vj-toolbox.md](vj-toolbox.md) phase 7) | Covers the repetitive "4 more like this" loop without typing |
| S17 | **Render tools can return immediately** (`wait=false` → job id); `jobs()` reports status; jobs can be **cancelled** (Modal `FunctionCall.cancel`, FAL cancel). Default for agents stays blocking | A full TextureFlow is 5–6 min; blocking freezes the chat |
| S18 | **Agent permissions**: it may label (its tags are stored as **agent tags**, and it never sets or overwrites Xander's score), move and rename library files, fill decks, and bind on design. It may **not** delete, and never takes anything live | Taking live stays a human act (gig D10) |
| S19 | **Decks = Resolume's**: named stacks of clips per show; rows = the show's regions (+ `_unassigned`), columns = slots; one deck shown at a time; switching decks never changes what plays. **No extras for the MVP**: no persistent clips, hidden decks, deck import or colours; clips only (no effect slots) | Xander: "none of these are needed for MVP" |
| S20 | **Naming**: **Studio** = the room where clips are made. **Deck** = a clip stack only. The engine area keeps its **design / live** panels, so UI copy and docs stop calling that area "deck". Wire names (`deck.*` RPC fields, `DeckPanel.tsx`) don't change | Two meanings of "deck" in one booth UI would trip him up |
| S21 | **Save = autosave + "Save as…"** (fork the show: scene, effects, decks, looks, session, alignment, pack; clips stay referenced, not copied). No dirty/revert model | A forgotten Save is a live-rig failure |
| S22 | **Style images are just images in library roots.** The **style tray** is part of the Studio selection; "used n×, average score" comes from the database (records list their inputs by fingerprint) | One browser, no second "styles" concept |

## Contracts (pinned so the lanes can run in parallel)

Build against these, not against another lane's code. Changing one means
editing this section in the same commit. On landing, each moves into
[../reference/mcp-server.md](../reference/mcp-server.md) (L1–L5) or
[../reference/operator-ui.md](../reference/operator-ui.md) (L6) and
leaves a pointer here.

**L1–L5 landed with H1 (2026-10-15)** — the library home + roots, the
embedded `wzrd` tag, the `library.db` schema, the hub (lock, idle exit,
`/api/*` routes) and the Studio selection now live in
[../reference/mcp-server.md](../reference/mcp-server.md) "The hub, the media
library and render records". Deltas from the pinned text: the record's
`inputs` entries also carry `path`; the jpg tag is a COM segment
`wzrd=<json>`; webp/gif/webm/heic carry no tag (linked by fingerprint);
fingerprint = `<size>-<sha1(first + last 1 MiB)[:20]>`; `library.db` adds
a `settings` table (`daily_cap_usd`) and denormalised columns; the hub's
idle check ignores a client's open `GET /mcp` stream; the app spawns it
with `--extra mcp --extra engine --extra islands`. `/api` names follow the
app agent's list (`library/delete_scored`, `move {paths, dest}`).

**L6 — decks** (`<project>/decks.json`, replaces `clipgrid.json`)

```json
{"schema": 1, "active": "Intro",
 "decks": [{"name": "Intro",
            "rows": {"<region id>": [{"path": "/abs/clip.mov", "id": "r_…", "fp": "…"}, null],
                     "_unassigned": []}}]}
```

Slot paths are **absolute** (library files). `id`/`fp` are what Relink
matches on. C5 clip bindings in `scene.json` may now be absolute too: the
engine already accepts them (`compositor.rs` does `scene_dir.join(clip)`,
which keeps an absolute path), but `engine_tools.py`'s binding docs and
validation still say "relative to the scene dir", so update them. Deck
names are unique per show; order = array order.

## Work packages

Lanes: **PY** (`wzrd_mcp/`, `wzrd/`), **SHELL** (`wzrd-app/src-tauri/`),
**UI** (`wzrd-app/src/`), **ENG** (`render-core/`). Collision hot spots:
`tools_config.json` (append only) and `src-tauri/src/{lib,rpc}.rs`
(serialise A1, R1, D1). The perf-gate rule (CLAUDE.md) applies to every
SHELL/ENG package.

| Order | PY | SHELL + UI | ENG | Needs |
|---|---|---|---|---|
| 1 (**by ≈ 24 Oct**) | **H1** hub + library + identity | — | — | — |
| 2 | **H2** jobs: non-blocking + cancel | **A1** hub client + library backend | **P1** file preview source | H1 (A1, P1 need only L1–L4) |
| 3 | **H3** agent tools + skills | **A2** Studio room | — | A1; A2 uses P1 |
| 4 | — | **R1** Render panel | — | H2, A2 |
| 5 | (deck tool in D1) | **D1** decks | — | A1 |
| 6 | — | **D2** Save as, **D3** Relink | — | D1 |

Rough agent time with review loops: H1 ≈ 2 d, H2 ≈ 1 d, H3 ≈ 0.5 d, A1 ≈ 2 d,
P1 ≈ 1 d, A2 ≈ 2 d, R1 ≈ 1.5 d, D1 ≈ 2 d, D2 + D3 ≈ 1 d. That's ≈ 13 days of
agent time, parallel across lanes.

### H1 — the hub, the library database, identity in the file (PY) — **done 2026-10-15**

Shipped: `wzrd_mcp/hub.py` (lock, idle exit, `/api/*`), `library.py`
(`~/WZRD/media`, roots, `library.db`, indexing on open, labels,
move/rename/delete), `identity.py` (fingerprint, the `wzrd` tag), `jobs.py`
(job rows, daily cap), `provenance.record_outputs` replacing sidecars;
generators route to the library (`dest=`), venue tools stay in the project,
`project=` per call (S14 as amended); `recipe_tools` and `get_scene_context`
read the database with legacy-sidecar fallback; `fixture_projects.py` writes
tagged, registered clips. Gate passed: a real Claude Code session's
`texture_flow` + `nano_banana` landed in `inbox/` tagged, with render rows
and job start/end pairs; the TF HAP shares the mp4's id and the release
engine opens tagged HAP (bench, 0 stale); Finder rename + move kept the
record, Finder delete left it `missing = 1`; the daily cap refused at the
cap; `build_layerpack` still wrote into the project. Still owed to the app
lane: `clips.rs` reads the database instead of sidecars (query sent to the
app agent). Done — see
[../reference/mcp-server.md](../reference/mcp-server.md).

### H2 — non-blocking jobs + cancel (PY) — **done 2026-10-15**

Shipped: `wait=false` on every render tool (`jobs.detach`: replies with
`{job, dest, eta_s}` once the call passed its checks), MCP tools `jobs` /
`cancel_job` (`studio_tools.py`), Modal `spawn` + FAL `submit` so cancel
stops the remote side, lost detection, ETA medians, `/api/render`,
`/api/estimate`, `/api/jobs/cancel`; TextureFlow's keepalive came with H1.
Gate passed: a Claude Code session queued 4 TextureFlows with
`wait=false`, kept talking, all 4 landed (61–77 s each); one cancelled
12 s in left no file and Modal reported the call cancelled; a hub killed
mid-render left its job `lost`, and cancelling it stopped the Modal call.
Done — see [../reference/mcp-server.md](../reference/mcp-server.md) "Jobs".

### H3 — agent tools + skills (PY) — **done 2026-10-15**

Shipped: `studio_selection`, `find_renders`, `label_render` (agent tags,
never the score), `move_render`, `rename_render` in `studio_tools.py`;
skills `textureflow` (new "Working with the Studio" section) and
`run-sheet` (selection, `find_renders` before a batch, `wait=false`,
`project=`). Gate passed with cold Claude Code sessions and a hand-seeded
selection: "make two renders with these styles" rendered the two selected
style images into the target folder with `wait=false`; "what worked with
style_3?" cited the 8 and 7 (slow-bloom) against the 3 (muddy) and 2
(flicker). Done — see [../reference/mcp-server.md](../reference/mcp-server.md)
"The agent's Studio tools".

### A1 — hub client + library backend (SHELL) — **done 2026-10-15**

The app attaches to or spawns the hub (detached, absolute `uv` + extras,
own process group), shows a hub lamp with Start/Restart, proxies `/api`
(`hub_call`), browses `library.db` read-only with the hub's JSON shapes,
and serves media as asset-protocol URLs (devport `/file` for the browser
seat). Gate evidence: cold start spawned once, a relaunch attached, the hub
survived the app process being killed; a 500-image folder lists in
≈ 13 ms; idle exit verified on a throwaway hub (`--idle-exit 5`). Not yet
seen: a *render* finishing after an app quit (needs H2's `/api/render`; the
hub is a separate process group, so nothing in the app can stop it).
Done — see [../reference/operator-ui.md](../reference/operator-ui.md)
"Studio + the hub".

### P1 — file preview source (ENG + SHELL) — **done 2026-10-15**

`preview.file {path|null}` plays one HAP clip / still into the native
preview when the slot's id is `"studio"` (`PreviewParams.file`); no leg
renders for it; the source and its decode thread drop on `null` or when
the slot leaves file mode. Bench `studio-preview`: LIVE 60 fps, stale 0.
Still owed by A2: the on-screen check (AFK block) once the Studio room
mounts the slot. Done — see
[../reference/render-engine.md](../reference/render-engine.md) §1b
"Studio file preview".

### A2 — the Studio room (UI) — **done 2026-10-15**, on-screen check owed

Studio is the first tab (⌘0 — ⌘1/⌘2 stay Prepare/Perform so show keys
don't move): tree · grid · monitor + record · style tray, the plan's keys,
jobs pill + hub lamp in the TopBar. Gate evidence: 20 scores by keyboard
(40 keys) all landed in `library.db`; an agent's `to_hap` over MCP showed
as a running tile 0.4 s after the call and as its result 0.9 s after it
ended; tray ⇄ `selection.styles` both ways; mp4 plays in the monitor.
**Owed:** the native HAP monitor + LIVE 60 fps on screen —
[afk-test-session.md](afk-test-session.md) block B9. Done — see
[../reference/operator-ui.md](../reference/operator-ui.md) "Studio + the
hub".

### R1 — Render panel + buttons (SHELL + UI)

- The S16 Render panel in Studio. Recipes come from `list_recipes`, with
  their args as defaults; the cost estimate comes from the hub before Go.
- **More like this** on any WZRD render (same args, new seeds, count).
- **Make playable** on mp4 and DXV, for a single item or a whole folder.
- All of these run as `wait=false` jobs, visible in the jobs pill like
  agent jobs.

**Gate:** with no chat and no terminal:
- render 4 clips from a recipe plus 2 tray styles for a region;
- cancel one;
- make 2 playable;
- 4 variations of a favourite.

### D1 — decks (SHELL + UI + PY tool)

- `decks.json` (L6) replaces `clipgrid.json`.
- Deck tabs (create, rename, reorder, two-click delete) in Prepare's drawer
  and Perform's compact grid.
- Click a slot = bind on design via `sceneCommit` (C5, absolute path); T
  takes it live, as today.
- **Add to deck** from Studio, or drag a render onto a slot. A non-HAP file
  gets "Make playable" first (R1). The row = the render's `region` if this
  show has it, else `_unassigned`.
- The gig's "new from agent" lane becomes an **incoming strip**: the newest
  playable renders whose `project` is this show; click = add to the active
  deck and bind on design.
- Log S6 signals: added to a deck; played live (seconds, from TAKE
  telemetry).
- MCP tool `add_to_deck(path, deck?, region?)` for the agent (S18).
- Update `engine_tools.py`'s C5 docs and validation for absolute clip paths.
- Migration, once on open: `clipgrid.json` + `generated/*.mov` → one deck
  "Main" with absolute paths; `fixture_projects.py` writes the new shape.

**Gate:**
- six clips into "Intro", six into "Drop";
- in Perform, switch decks, click + T;
- reopen the show → decks intact;
- a clip from `projects/fixture-b` plays in fixture-a's deck with no copy
  made.

### D2 — Save as (S21)

Project menu → "Save as…" → name → fork (clips referenced, not copied) →
open the fork. **Gate:** the fork opens with the same wall, decks and looks;
editing it leaves the original untouched.

### D3 — Relink missing

- A button in the project menu (and on a missing slot's warning).
- For every missing path in `decks.json` and `scene.json`, search in order:
  1. a `files` row with the same render id;
  2. the same fingerprint among indexed files;
  3. walk the roots for files of the same **size** (a cheap `stat`), then
     fingerprint only those.
- Rewrite the paths and report found / not found.
- On demand only, never a background process.

**Gate:** move a show's clip folder in Finder → the slots warn → Relink fixes
every one in one click.

## Not in this sprint

- In-app chat ([vj-toolbox.md](vj-toolbox.md) phase 7). Its home is the
  Studio room.
- A stdio bridge so a Claude session can start the hub without the app.
- Native DXV decode in the engine ([backlog.md](backlog.md)).
- Deck extras: persistent clips, hidden decks, importing a deck from another
  show, colours ([vj-tool-feature-survey.md](vj-tool-feature-survey.md)
  top pick 8). Effect slots in decks.
- Statistics beyond `find_renders` (which args correlate with high scores).
  Build them once a few hundred records are labelled.
- "Pack show" (copy a show's clips next to it to take it to another
  machine).
