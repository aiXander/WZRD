# Studio — content generation, the media library, and named clip decks (sprint plan)

> **Status: proposed 2026-10-12 — waiting for Xander's review. Nothing built.**
> Decisions marked **(confirm)** are open until he signs off; the rest are
> the recommendation. Supersedes [vj-toolbox.md](vj-toolbox.md) phase 4
> (in-app generation: MCP client, job tray, ratings) and changes how the
> gig's clip grid gets its clips ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)
> D10/D12, contract C3). Current state it builds on:
> [../reference/mcp-server.md](../reference/mcp-server.md) (project folder,
> provenance sidecars, playable clips, recipes),
> [../reference/operator-ui.md](../reference/operator-ui.md) (rooms, clip
> grid, projects). Read this whole file before starting a package.

## What Xander asked for (2026-10-12)

Making content and playing it live are two different activities, and the
app only serves the second. Before and during a gig Xander:

1. browses folders of **style images** on the Mac and picks some;
2. chats with the agent (Claude Code + the WZRD MCP server) to render
   images, TextureFlow clips, Kling videos;
3. watches which **render jobs** are running, sees results as they land
   (small player), sorts renders into **folders**, converts the good clips
   to the engine's format (HAP);
4. keeps **metadata for every render** — settings, style images, prompts,
   and whether it was good — because most TextureFlow renders come out bad
   and that history is how future agents pick better settings;
5. then pulls clips from that library into a live show, organised in
   **named decks** like Resolume.

"Saving a project" means saving the **show**: effects, what's in the
decks, looks, masters. The content library is not a project, just a UI
over folders on disk.

## The model: two worlds, and disk is the bus

| | **Media library** | **Show** (a project) |
|---|---|---|
| What | every render, clip and still ever made; style-image folders | one live set at one venue: pack + regions, `scene.json`, effects, **decks**, looks, masters, alignment |
| Where | one folder tree, default `~/WZRD/media/` (L1) + read-only style folders anywhere | `projects/<name>/`, unchanged |
| Organised by | folders Xander makes, ratings, tags | regions × decks |
| UI | new **Studio** room | Prepare / Align / Perform, as today |
| Saved | always — files *are* the state | autosaves as today; "Save as" forks a show (S8) |

A clip crosses from library to show by **import = copy** into
`<project>/clips/` (an APFS clone, so no extra disk). The show never points
into the library, so reorganising the library can't break a show. This is
the same rule the effects library uses (gig contract C6: "projects get
copies, never links").

**Disk is the bus.** The agent (in a terminal) and the app never talk to
each other directly. They meet in files under the library root:

- the **render record** = each file's provenance sidecar, extended (L2);
- the **job journal** — one line when a render starts, one when it ends (L3);
- the **Studio selection** — what Xander has selected in the app right now (L4).

So the app works with no chat open, the agent works with no app open, and
a crash on either side loses nothing. It's also why the in-app chat can
come later without redesigning anything (S10).

## How it looks in practice

Show night, between sets. Xander opens **Studio** (the projector keeps
playing LIVE; the TopBar keeps TAKE and the health lamps).

1. **Style tab:** his style folders (`~/Pictures/styles/…`) as a thumbnail
   grid. He clicks three images; they go into the **style tray** along the
   bottom. Each image shows how often it has been used and how many of
   those renders he kept.
2. He opens the folder `wintercircus/fabric` as the **target folder**,
   switches to his Claude Code terminal and types: *"4 TextureFlows for the
   left fabric with these styles, colder than last time."* The agent calls
   `studio_selection` (gets the three paths + target folder) and
   `find_renders` (sees which settings he kept and rejected for those
   styles), then queues four `texture_flow` jobs (S5).
3. Four **running-job tiles** appear in the folder at once, with elapsed
   time against the usual duration for that kind of job ("1:40 of ~5:00").
   Results replace the tiles as they land.
4. **Culling:** arrow keys move, space plays the clip in the inspector,
   `X` rejects, `K` keeps, `F` favourites, `N` adds a note ("too busy at the
   hem"). Rejects drop out of view but their record stays.
5. A favourite: **"More like this"** queues 4 renders with the same settings
   and new seeds, no chat needed (package A3). Or he says "save that as a
   recipe" in chat.
6. **"Add to deck"** on a kept clip: it's made HAP if needed, cloned into the
   open show, and placed in the active deck's row for the region it was made
   for. In Perform: deck tabs `Intro | Drift | Drop`; click a slot = it plays
   on that region in DESIGN; **T** takes it live, as today.

## Decisions

| # | Decision | Why |
|---|---|---|
| S1 | **One media library root**, default `~/WZRD/media/`, override `WZRD_MEDIA_ROOT` (read by the MCP *and* the app). Not in the repo: media is big and not versioned. Style folders are a separate **read-only** list in the app's `settings.json` (`style_roots`). WZRD never writes into them **(confirm the default path)** | Both processes must agree with no shared config. A Finder-launched app sees no env vars, so the default is what runs (gig D16's rule) |
| S2 | **The render record is the sidecar** (C2, extended by L2): rating, notes, tags, input hashes, job id, duration, project. **A record is never deleted.** "Purge" deletes the media and keeps sidecar + thumb | Bad renders are the learning signal. Deleting the file must not delete what was learned |
| S3 | **Ratings: reject / unrated / keep / favourite** (`-1 / absent / 1 / 2`), keyboard-first culling. No 1–5 stars **(confirm)** | Fast enough to rate everything mid-gig, which matters more than resolution. Stars get skipped |
| S4 | **A job journal** written by the MCP for every render-class tool: `start` + `end` (outputs or error). The app's job view and ETAs come from it. Failed jobs live only here, with their args + error | No MCP progress plumbing, and it covers agent-started and button-started jobs alike. TextureFlow sends no progress at all today |
| S5 | **Render tools can return immediately** (`wait=false` → job id). The work keeps running in the MCP server; results land in the library and the journal; `jobs()` reports status. Default stays blocking | A full TextureFlow is 5–6 min. A blocking call freezes the chat for the whole batch, which is exactly the gig moment |
| S6 | **Generator output goes to the library, venue authoring stays in the project.** `nano_banana`, `texture_flow`, `kling_*`, `to_hap`, `extract_frames`, `concat_videos` default to the Studio target folder, else `inbox/`. `import_photos`, `prepare_surface`, `extract_color_regions`, `build_layerpack` keep writing into the project. A `dest="project"` arg covers the in-between cases (a segmentation map for `extract_color_regions`); the `run-sheet` skill uses it. `region=` still resolves against the **active project's** pack, and the record stores `project` | Content outlives a show; the pack and its sources are venue facts |
| S7 | **Spend cap stays per project** (gig D10), summed from journal `end` lines tagged with that project, not from an rglob of project sidecars | Renders no longer live in the project folder; the journal is cheaper to sum anyway |
| S8 | **Save = the show.** Autosave stays (crash safety, all files already autosave). Add **"Save as…"** = fork the show (scene, effects, decks, looks, session, alignment, pack, cloned clips) under a new name. No dirty/revert model **(confirm — or do you want an explicit Save with "unsaved changes"?)** | An explicit save adds a "forgot to save" failure for no gain on a live rig. A fork is what "save a version of tonight's set" needs |
| S9 | **Decks = named pages of the clip grid**, per show, Resolume's meaning. Rows stay **regions** (gig D12), columns are slots. One deck is shown at a time; switching decks changes what the grid shows, never what's playing | That's how Resolume decks behave, and it keeps D12's surface-language rows |
| S10 | **Naming collision:** today "deck" also means the DESIGN \| LIVE area (`DeckPanel.tsx`, the engine's `deck.preview_source`, the design spec's "two legs, one deck"). From now on **"deck" = a named clip page** in UI copy and docs; the DESIGN \| LIVE area becomes the **TAKE panel**, the two sides stay **legs**. Wire names (`deck.*` RPC fields) don't change **(confirm)** | Two meanings of "deck" in one booth UI would be a constant trip hazard |
| S11 | **Chat stays a Claude Code terminal** for this sprint (gig D11). The in-app chat comes later, in the Studio room. How is [vj-toolbox.md](vj-toolbox.md) phase 7: either through Foreman's daemon, or by reusing its *pattern* (the real `claude` CLI in a PTY, shown with xterm.js). Foreman itself can't be embedded: it's a browser app with no public API (checked 2026-10-12) | The disk-bus design makes the chat's location irrelevant to everything in this plan |
| S12 | **Studio is project-independent but project-aware**: it works with any show open, shows a "made for `<project>` · `<region>`" chip on each render, and "Add to deck" targets the open show | Content is made against a venue's regions, but browsing and rating isn't |
| S13 | **Local media in the webview through one helper** (`mediaUrl(path)`): Tauri's asset protocol scoped to the library root, style roots and projects root; in the browser bridge a devport route serves the same files. Previews play the **mp4** (WebKit can't play HAP); one player at a time, never autoplay a grid, paused when the room is hidden | Today the webview can't show local video at all (thumbs are base64 over IPC, which won't scale to a 500-image style folder). Video decode competes with the engine on show night |

## Contracts (pinned so lanes can run in parallel)

Same rule as the gig plan: build against these, not against another lane's
code; changing one means editing this section in the same commit.

**L1 — library layout**

```
<media root>/                  ~/WZRD/media or $WZRD_MEDIA_ROOT
  inbox/                       default landing folder
  <Xander's folders>/…         free-form, any depth
  .wzrd/
    jobs.jsonl                 L3
    selection.json             L4
    thumbs/                    style-image thumbnail cache (app-owned, disposable)
```

An **item** = files sharing a stem in one folder: `<stem>.mp4`, `<stem>.mov`
(HAP), `<stem>.png`, `<stem>.thumb.jpg`, and each media file's
`.meta.json`. Moving or renaming an item moves the whole bundle. Folder
names are free; nothing in the library is addressed by folder meaning.

**L2 — render record (sidecar schema 1, extended)**: today's C2 keys
(`tool, args, inputs, parent, created, est_usd, region, source_box`) plus

| Key | Written by | Meaning |
|---|---|---|
| `job` | MCP, at creation | id linking to L3 |
| `duration_s` | MCP | wall time of the call |
| `project`, `canvas` | MCP | which show's pack/canvas it was made against (`null` if none) |
| `input_hashes` | MCP | `{path: sha1}` of every local input, so a style image is still recognised after its folder moves |
| `rating` | app / `rate_render` | `-1` reject, `1` keep, `2` favourite; absent = unrated |
| `notes`, `tags` | app / `rate_render` | free text; list of strings |
| `purged` | app | ISO time the media was deleted; record + thumb stay |

The MCP writes a sidecar once, at creation, and never rewrites it. Only the
last four keys change afterwards, by an atomic read-modify-write (temp +
rename). Readers ignore unknown keys.

**L3 — job journal** (`jobs.jsonl`, append-only, one JSON object per line,
written under an exclusive lock)

```
{"ev":"start","job":"j_…","t":ISO,"tool":"texture_flow","args":{…},"project":"…","dest":"<abs dir>","est_usd":0.2,"pid":123,"by":"agent"|"app"}
{"ev":"end","job":"j_…","t":ISO,"ok":true,"outputs":[abs paths],"est_usd":0.19,"duration_s":301}
{"ev":"end","job":"j_…","t":ISO,"ok":false,"error":"…"}
```

A `start` with no `end` whose `pid` is dead is shown as **lost**. The ETA
is the median `duration_s` of the last 20 ok jobs of the same tool with
the same size args. `by` is set from the MCP caller (the app passes `by:
"app"` through its client).

**L4 — Studio selection** (`selection.json`, written by the app on every
change, read by the `studio_selection` tool):
`{"schema":1,"updated":ISO,"styles":[abs paths],"items":[abs paths],"target":abs dir|null,"project":name|null}`.

**L5 — decks** (`<project>/decks.json`, replaces `clipgrid.json`):
`{"schema":1,"active":"Intro","decks":[{"name":"Intro","rows":{"<region id>":["clips/a.mov",null,"clips/b.mov"],"_unassigned":[…]}}]}`.
Paths are scene-dir-relative, so C5 clip bindings are unchanged. Deck
names are unique per show; order = array order.

**L6 — show clips**: `<project>/clips/<stem>.mov` + its sidecar + thumb,
cloned from the library at import (macOS `clonefile`, plain-copy fallback).
The sidecar keeps the library path it came from (`imported_from`) for
"show in library". `generated/` stays readable for old projects (migration
in D1).

## Work packages

Lanes: **PY** (`wzrd_mcp/`, `wzrd/`), **SHELL** (`wzrd-app/src-tauri/`),
**UI** (`wzrd-app/src/`). Packages in different lanes run in parallel;
SHELL and UI pair up per feature. Collision hot spots: `tools_config.json`
(append), `src-tauri/src/{lib,rpc}.rs` (serialise A1/A3/D1). The perf gate
rule (CLAUDE.md) applies to every SHELL package.

| Order | PY | SHELL + UI | Needs |
|---|---|---|---|
| 1 | **M1** library routing + records + journal | **A1** library backend + media URLs | contracts only |
| 2 | **M2** selection + learning tools + skills; **M3** non-blocking jobs | **A2** Studio room | A2 needs A1 |
| 3 | — | **A3** MCP client + buttons | M1, A2 |
| 4 | — | **D1** decks + import | A1 (A2 for the Studio button) |
| 5 | — | **D2** Save as | D1 |

### M1 — library routing, extended records, job journal (PY)

`project.py`: `media_root()` (S1), `library_output_path/dir(tool, ext,
dest)` resolving target folder (from L4) → `inbox/`; S6 routing in the
listed tools plus `dest=`. `provenance.py`: L2 keys at creation;
`check_spend` sums L3 (S7). `_log.py` `logged_tool`: L3 `start`/`end` for
tools tagged `render` in `tools_config.json`. Fix the fixture script's fake
`prompt` arg on `texture_flow` (the real tool has none) while there.
**Gate:** a `texture_flow` and a `nano_banana` from a Claude session land in
`inbox/` with L2 records and a start/end pair; a forced failure writes
`ok:false` + error; the spend cap still refuses at the cap for that
project; `build_layerpack` still writes into the project.

### M2 — selection + learning tools, skills (PY)

Tools: `studio_selection()`; `find_renders(tool?, rating?, tag?, style?,
recipe?, project?, text?, limit)` → compact rows (path, rating, notes, the
args that differ from the tool's defaults, style image names); `rate_render(path,
rating?, note?, tags?)` for triage in chat. Skills: `textureflow` and
`run-sheet` learn "these styles" → `studio_selection`, "check the archive
before a batch" → `find_renders` (kept vs rejected settings for these
styles/recipe), and S6's `dest`. **Gate (no UI needed — hand-write
`selection.json`):** a cold Claude session told "two renders with these
styles" uses the selected paths and the target folder; asked "what worked
with style X" it cites kept and rejected records.

### M3 — non-blocking render jobs (PY) (S5)

`wait=false` on `texture_flow`, `kling_*`, `nano_banana`: register an
asyncio task in the server process, return `{job, dest, eta_s}` at once;
`jobs(job?)` → status/outputs from L3. Add `with_keepalive` to the blocking
`texture_flow` path (it has none; a 6-min silent call risks a client
timeout). Trap: under stdio the server dies with the Claude session, which
takes its running jobs with it → those show as **lost** (L3), not as
success. **Gate:** queue 4 TextureFlows, keep chatting, all 4 land; kill the
session mid-render → the app shows them lost.

### A1 — library backend (SHELL)

Commands: `library_tree`, `library_list(folder)` (items per L1 + their L2
records + running jobs targeting the folder), `library_move/rename/mkdir`
(whole bundles), `render_rate` (L2 key write), `render_purge`,
`selection_set` (L4), `styles_list(root, folder)` with a background thumb
cache (`.wzrd/thumbs/`, keyed by path + mtime), a journal tail (read from
the last offset, emit `library:jobs`). Asset protocol + capability scope
(S13) and the devport media route for the browser bridge. Poll the open
folder every 2 s like the clip grid, only while Studio is mounted.
**Gate:** a 500-image style folder lists in < 1 s cold-cache-excluded and
thumbs fill in without blocking; moving an item moves all its files; a
browser-bridge page shows the same thumbs and plays an mp4.

### A2 — the Studio room (UI)

A fifth room, first in the order (**Studio** · Prepare · Align · Perform,
Debug stays last). The name is open **(confirm)**. Layout:

```
┌ TopBar: rooms · TAKE + health (unchanged) · jobs pill (running n / $ today) ┐
│ Library tree  │  folder grid: thumbs, rating badge, NEW,   │  Inspector:      │
│ ─────────     │  running-job tiles (elapsed / ETA),        │  player (mp4)    │
│ Style roots   │  "show rejects" toggle                     │  tool, args,     │
│               │  [Renders | Styles] tabs                   │  style images,   │
│               │                                            │  rating, notes,  │
│               │                                            │  made for … chip │
├───────────────┴── style tray (selected styles) · target folder ─────────────┤
```

Keys (only while no input is focused): arrows, space = play, `X`/`K`/`F`
rate, `N` note, `M` move to a folder, `T` set target folder. Jobs pill
opens a tray of running/finished/failed/lost jobs with their args. No
telemetry subscriptions; the native preview underlay is hidden in this
room. **Gate:** cull 20 renders by keyboard in < 1 min; an agent render
appears as a running tile within one poll and is replaced by its result;
the style tray round-trips into `selection.json`; LIVE keeps 60 fps
while the room is open with a clip previewing (perf gate table + an AFK
block in [afk-test-session.md](afk-test-session.md) for the on-screen part).

### A3 — the app as an MCP client: buttons (SHELL + UI)

Folds in toolbox T4a. The shell attaches to a running `python -m
wzrd_mcp` on `:8787` or spawns one (the `WZRD_AUDIO` spawn pattern in
`src-tauri/src/lib.rs`); calls go through the Rust side (an MCP client
crate, e.g. `rmcp`), never from the webview. Buttons: **Make playable**
(`to_hap`), **More like this** (same args, new seeds, count, with
`wait=false`), **Render with tray** (recipe picker + count + cost estimate
→ `texture_flow` with the tray's styles and the target folder). Their jobs
show through L3 like the agent's. **Gate:** two clips made playable and four
variations of a favourite, no chat and no terminal.

### D1 — decks + import (SHELL + UI)

`decks.json` (L5) replaces `clipgrid.json`; the grid shows the active
deck's rows instead of scanning `generated/`. Deck tabs: create, rename,
reorder, delete (two-click), in Prepare's drawer and Perform's compact
grid. **Import** (L6): from Studio's "Add to deck" or by dragging a render
onto a slot; non-HAP → `to_hap` first via A3 (until A3 lands: refuse with
"make playable first"). Row = the render's `region` if that region exists
in this show, else `_unassigned`. The gig's **"new from agent" lane**
becomes an **incoming strip**: new playable library renders whose
`project` is this show, newest first; click = import into the active
deck + bind on design. Migration (once, on open): `generated/*.mov` +
`clipgrid.json` → clips stay where they are, one deck **"Main"** lists them;
`fixture_projects.py` writes the new shape. **Gate:** six clips into
"Intro", six into "Drop"; in Perform switch decks, click + T; reopen the
show → decks intact; rename the library folder they came from → the show
still plays.

### D2 — Save as (UI + SHELL) (S8)

Project menu → "Save as…" → name → fork per S8 (clips cloned) → open the
fork. **Gate:** the fork opens with the same wall, decks and looks, and
editing it leaves the original untouched.

## Fit with the gig (≈ 21 Nov)

M1–M3, A1–A3 are additive: they don't touch Perform or the engine. **D1
changes the clip grid the gig plays from**, so it lands and is rehearsed by
**≈ 7 Nov** (before the dress rehearsal week), or it waits until after the
gig and the gig plays on today's grid. Rough size, agent time with review
loops: M1+M2 ≈ 1.5 days, M3 ≈ 1 day, A1+A2 ≈ 3–4 days, A3 ≈ 1.5 days, D1 ≈
2 days, D2 ≈ 0.5 day.

## Not in this sprint

- In-app chat (S11) — sketch in [vj-toolbox.md](vj-toolbox.md) phase 7.
- Cancelling a running job (Modal and FAL support it; add once M3 is used).
- Statistics beyond `find_renders` (which args correlate with keeps).
  Do it once there are a few hundred rated records.
- Effects in deck slots (Resolume mixes generators and clips). The open
  question already in vj-toolbox.
- Importing Xander's existing loose renders (`projects/tf-phase2-test`,
  the shootout folders): drop them into the library by hand. They have no
  records, and that's fine.
