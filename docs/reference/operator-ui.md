# Operator UI — the `wzrd-app` webview

> How the human-facing control surface is organised and why. Engine
> contracts it talks to (RPC, telemetry, legs, alignment) live in
> [render-engine.md](render-engine.md) — §2.x pointers below go there. The
> north star is [user_design_spec.md](user_design_spec.md). Redesigned
> 2026-10-09 (Resolume-style revamp); the previous rack/inspector UI is gone.

## The three rules every control obeys (`src/state/actions.ts`)

1. **Hands go where the deck points.** The DESIGN | LIVE switch
   (`deck.preview_source`, §2.6) is the *control leg*: layer faders,
   mute/solo and knob pins write that leg. The whole room re-tints with it —
   `<html data-deck>` flips the `--deck` CSS token (cyan DESIGN, red LIVE),
   so the mode is visible from across the booth.
2. **Masters are the room, not a leg.** `setMaster` writes **both** legs;
   `App.tsx::lockLegs` copies live's masters onto design and calls
   `transport.set {sync}` at hydrate. With identical speed masters the two
   clocks then stay in lockstep, so a TAKE (which copies design's control
   state + clock into live) never jumps the wall's brightness, speed or bar
   grid. Don't reintroduce per-leg master writes.
3. **Structure is authored in DESIGN.** Effect choice, region targeting,
   colours, modulation routing/ranges are scene edits → `sceneCommit.ts` →
   design leg (engine blanket rule). On LIVE those controls render read-only
   with an "edit in DESIGN" affordance; scalar faders stay playable as pins.

Live writes are **throttled (leading + trailing), never debounced**
(`actions.ts::throttled`, 25 ms). The old rack used a 25 ms *debounce*,
which never fired mid-drag (pointer events arrive every ~16 ms) — faders only
"landed" when the hand paused. That was half of "the sliders don't do much";
the other half was guessed ranges (e.g. `scale: 50` on a 0.5–8 control).
Every scalar now uses the effect descriptor's `min`/`max` (`LayerParams.tsx
::rangeFor`, extended to include an out-of-range scene value).

## Rooms (`src/routes/`, ⌘0 Studio · ⌘1 Prepare · ⌘2 Perform; Align ⌘3 + Debug ⌘4 via ⚙)

| Room | What it's for | Key files |
|---|---|---|
| **Studio** | Make, browse and label content (studio plan A2): library roots + folder tree, a folder grid of static thumbs, the preview monitor + the focused item's record and labels, the style tray. Full map: "Studio + the hub" below. | `Studio.tsx`, `components/studio/*`, `state/studio.ts` |
| **Prepare** | Build looks on DESIGN: layer list + region list, the surface map (photo / light / both, "Show regions" switch), inspector (target, effect cards, dials; Duplicate / Delete / deselect × in its header), bottom drawer = **clip decks** or code (Monaco, lazy). Entering it switches hands to DESIGN. Two-click layer creation: pick a region → "+ New"/"Light it up". | `Prepare.tsx`, `LayerList.tsx` (layer + region lists), `LayerInspector.tsx`, `SurfaceMap.tsx`, `ClipGrid.tsx`, `LayerParams.tsx mode="author"`, `MonacoPanel.tsx` |
| **Align** | §2.8 warp; side panel with the "projector bumped?" recovery hint, test patterns, background (lit-background warning). | `Align.tsx`, `WarpCanvas.tsx`, `state/alignment.ts` |
| **Perform** | The booth: monitor (native preview, deck-coloured frame, PREVIEW/PROGRAM label), a slim looks strip + auto-pilot under it, one mixer panel with a Layers \| Clips tab (`MixerPanel`, tab in localStorage `wzrd.perform.mixerTab`; only the visible tab mounts, so the clip poll runs only on Clips), selected layer's dials, deck panel (switch, cue monitor, TAKE/CUT/PULL, FADE, BAR/NOW), masters + BLACK/FLASH/FREEZE + room audio. | `Perform.tsx`, `DeckPanel.tsx`, `LayerStack.tsx`, `ClipGrid.tsx compact`, `LayerParams.tsx`, `LooksBank.tsx`, `MastersPanel.tsx` |
| **Debug** | Engine internals (unchanged content, restyled), plus the plain "Performance (frame profiler)" readout of the `perf` / `perf_spike` channels — headroom, stage p95 table, threads, clips, last spike ([render-engine.md](render-engine.md) §2.9). | `Debug.tsx` |

`TopBar.tsx` is shared: project menu (`ProjectMenu.tsx`), rooms (Studio | Prepare | Perform as tabs — Align is a once-per-gig room, so it and Debug live in the ⚙ **Setup** menu with the panel-layout controls; while you're in one it shows as an extra active pill; a stored `wzrd.route=align` boots into Perform), compact DESIGN|LIVE + TAKE outside Perform,
tempo (beat lamps from the `transport` channel, BPM double-click/scroll,
TAP, `1` = bar line here), the **jobs pill** (running renders · $ today,
`JobsPill.tsx`) and **hub lamp** (`HubLamp.tsx`) — both "Studio + the hub"
below — health (Audio / FPS / reload verdict / Engine —
a restart button while the engine is down), ADOPT for agent edits, help
overlay (`?`), toasts. While the engine is down a red banner under the bar
(`EngineDownBanner`) shows the reason and one "Restart engine" button.

## Projects + engine restart (gig A1)

A **project** is a folder with a `scene.json` under the projects root
(`WZRD_PROJECTS_ROOT`, else `projects_root` in settings, else
`<repo>/projects` — the MCP's default). The engine's scene dir is fixed at
boot, so **opening a project re-roots the in-process engine** (shutdown +
boot on the same windows, ~150–250 ms; host side in
[render-engine.md](render-engine.md) §1b "Re-rooting"). Each project brings
its own pack, effects, looks, `alignment.json` and `session.json`
(masters, knobs) back. "Restart engine" is the same move on the current
scene — the crash recovery.

- Shell: `src-tauri/src/projects.rs` (`projects_list`, `project_current`,
  `project_open`, `engine_restart`). A failed open re-opens the previous
  project and returns the error; a failed boot leaves the shell up with
  the reason (`engine_status.last_error`) instead of dying.
- **Per-machine `settings.json`** in the Tauri app config dir
  (`~/Library/Application Support/build.gigabrain.wzrd/`): `recent`
  (scene paths + `opened_ms`), optional `projects_root`; unknown keys are
  preserved (MIDI's machine default lands here later). With no
  `WZRD_SCENE` / `--scene` the app boots the most recent project — a
  Finder launch works.
- Webview: opening takes two clicks (arm → "OPEN", the wall goes dark for
  a moment). `actions.ts::openProject` first **settles** in-flight scene
  commits into the old project (`sceneCommit.ts::settleCommits`); every
  hydrate then starts from `store.resetProject()` + `dropPendingCommits()`
  + `resetActionState()`, so nothing of the previous engine leaks in.
  Hydrate fires on the engine coming back (status edge) and on the
  shell's `engine:project` event, deduped (`App.tsx::rehydrate`).
- **Save as… (studio D2, S21)** — the only explicit save; everything else
  autosaves. Project menu → name (default `<show>-copy`) →
  `actions.ts::saveProjectAs` settles scene commits + `session.save`, then
  `fork.rs::project_save_as` copies the show into a **sibling folder**:
  every root `*.json` (scene files, session, alignment, decks, ui_layout,
  project), `effects/`, `packs/`, `looks/` and the design draft
  `.wzrd/design.scene.json`; media (`generated/`, …) are **not** copied —
  every relative `"clip"` in the copied JSON becomes the absolute path
  into the original, so both shows play the same files. Temp folder +
  rename (no half fork); an existing name refuses. Then it opens the fork
  (same scene file name). Like any open, both legs boot from the saved
  scene, so the fork's wall = the design that was on screen.
- **Relink missing (studio D3)** — project menu button (shows the missing
  count) or click a ⚠ deck slot. `relink.rs::relink_missing {scene}` looks
  at every deck slot, every `"clip"` in the design scene the UI passes and
  in `looks/*.json`; per missing file: same render id (same extension) in
  `library.db` → same fingerprint → a walk of the library roots + the
  projects root (stat everything, fingerprint only size matches) → for a
  file the library never saw (no id/fp), its file name. It rewrites
  `decks.json` (under the decks write lock) and the looks itself; the
  scene map comes back to the UI and goes through `sceneCommit` (the one
  scene-write path), so LIVE picks it up on the next TAKE. The folders the
  files turned up in are then listed through the hub (indexing them), so
  slots get thumbnails back. Measured: a moved 6-clip folder → 7 refs
  fixed in 72 ms (walk); re-moved → library lookups in 2 ms.
- Test fixtures: `wzrd-app/scripts/fixture_projects.py` builds
  `projects/fixture-a` + `fixture-b` (example pack, real HAP clips carrying
  the embedded `wzrd` tag and registered in `library.db`, a "Main" deck);
  `--add-clip <project> --region <id>` simulates an agent render landing.

## Projector output — the OUTPUT pill (2026-10-15)

Resolume's Output menu: the engine output window is created **hidden** and
the operator turns it on from the TopBar **OUTPUT** pill (pill → display =
two clicks). Hidden reads as occluded, so the engine renders offscreen and
presents nothing (render-engine.md §3.1) — off is free, on is instant, no
engine restart.

- Shell: `src-tauri/src/output.rs` (`output_displays` / `output_state` /
  `output_set {target: off|window|display, display}` / `output_options` /
  `output_identify`; pushes `output:state` + `output:displays`). Webview:
  `components/OutputPill.tsx` + `state/output.ts`; the browser seat reaches
  the same commands through devport.
- **Display target** = borderless window covering that display at
  `NSMainMenuWindowLevel + 1` (above its menu bar and Dock), shown with
  `orderFrontRegardless` so key focus — and every UI shortcut — stays on
  the operator window. The display holding the UI is disabled in the list
  (a borderless above-menu-bar window there would bury the UI with no way
  back); use **Window** for single-screen testing.
- **Persistence:** the target is never saved — output is off at every
  launch. Per machine (`settings.json` → `output`): last display **by
  name** (indices shuffle on replug; duplicates get ` (2)`), `hide_cursor`,
  `keep_awake` (an `NSActivityIdleDisplaySleepDisabled` assertion held only
  while live). **Flips + fit** live in the project's `alignment.json`
  (render-engine.md §2.8 — they describe the install) and are written via
  `state/alignment.ts::setOutputTransform`, not by `output.rs`.
- A watcher thread re-lists displays every 2 s; a vanished target display
  turns the output **off** (never onto the operator's screen) and the pill
  goes red with `lost`. A display not in its native mode (CoreGraphics
  `kDisplayModeNativeFlag`) gets an amber dot and a "set it to native"
  note.
- Closing the output window (window mode) = output off, not quit.
- Scripted launches: `WZRD_OUTPUT=window` or `display:<idx>`
  (`WZRD_DISPLAY=<idx>` still works) turn it on at boot — AFK fps gates
  need this, a hidden window measures offscreen frames.

## Clip decks (studio plan D1, 2026-10-15; was the gig's clip grid)

Resolume's clip stacks, per show, in `<project>/decks.json` (contract L6:
`{"schema":1,"active":name,"decks":[{"name","rows":{region|"_unassigned":
[{path,id,fp}|null…]}}]}`). Slots reference **library files in place by
absolute path** (+ render id + fingerprint for Relink) — a clip from
another show's folder plays without a copy. One deck is shown at a time;
**switching decks never changes what plays**. `ClipGrid.tsx` (Prepare's
drawer, Perform's compact Clips tab) + `state/decks.ts` + shell
`src-tauri/src/decks.rs`.

- **Rows = the pack's regions + Unassigned; each row is a radio group**
  (one clip per region on DESIGN). Leading ■ = NONE/STOP. Click a slot =
  `state/clips.ts::bindClip(absPath, region)` (sets `params.image.clip` on
  the region's first `image`-input binding, else appends `clip_<region>`
  with `videoClip`, via `commitSceneMutation`); click the playing one or ■
  = `stopClip` (a `videoClip` binding is removed, any other effect just
  loses `image`). TAKE as usual. DESIGN = design-colour ring, LIVE = red
  top bar (`clipsByRegion` over both legs; relative C5 paths are made
  absolute by `absClip` before comparing).
- **Deck tabs:** create (+), rename (double-click), reorder (drag),
  delete (✕ then "delete?" within 2.5 s; full grid only; a show keeps ≥ 1).
- **Filling:** Studio "+ Deck" button / `D` key (the selection, each into
  its render's region row if this pack has it, else Unassigned; a non-HAP
  video queues `to_hap` and the slot points at the predicted `.mov`, so it
  shows missing until the job lands); the **INCOMING strip** = the newest
  playable renders whose `project` is this show and that aren't in the
  active deck (`library_incoming`, polled 4 s) — click = add + play on
  DESIGN, drag = file on a row; the agent's `add_to_deck` MCP tool. Drag a
  slot to another row / slot to re-file; ✕ on hover removes it (the file
  stays).
- **Two writers, one file.** The app writes the whole doc with a
  compare-and-swap on the file's mtime (`decks_write {doc, baseMtimeMs}` →
  `"changed"` if the hub's `add_to_deck` wrote in between → re-read,
  re-apply the op, retry once). While a grid is mounted the file is
  re-read every 2 s (refcounted `useDecksLivePoll`), so an agent's add
  shows up without a reload. `decks_read` also returns `missing` (slot
  paths not on disk) → amber dashed "⚠ missing" tile.
- **Migration, once:** a show with no `decks.json` gets one deck "Main"
  from the old `clipgrid.json` filing + `generated/*.mov` (region from the
  embedded tag / old sidecar), absolute paths, written on first read. The
  hub's `add_to_deck` refuses a show that still needs it. Fingerprints
  (`decks.rs::fingerprint`) match the hub's byte-for-byte (`<size>-<sha1
  of first + last MiB>[:20]`, tested).
- **Signals (S6)** to the hub's `/api/library/signal`: `deck_add` on add;
  `played` + seconds when a clip leaves the LIVE scene (`usePlayedSignals`,
  App-level). An engine restart or show switch resets the clocks without
  a signal (it blanks the live scene, which isn't "left the wall").
- Commands: `decks_read`, `decks_write`, `deck_slots {paths}` (id + fp for
  new slots), `library_items {paths}` (thumbs/labels for slots),
  `library_incoming {project, limit}`; all have devport twins.
  `wzrd-app/scripts/fixture_projects.py` writes a "Main" deck directly.

## Studio + the hub (studio plan A1/A2, 2026-10-15)

The content side of the app. The plan with its decisions S1–S22 and
contracts L1–L6 is [../TODO/studio-content-library.md](../TODO/studio-content-library.md);
the hub's own side (routes, `library.db` schema, jobs) is
[mcp-server.md](mcp-server.md).

- **The hub** is `python -m wzrd_mcp` on `127.0.0.1:8787` (`/mcp` for
  agents, `/api/*` JSON for the app), a separate process that owns
  `~/WZRD/media/.wzrd/library.db`, jobs, the Studio selection and spend.
  `src-tauri/src/hub.rs`: at startup a monitor thread calls `GET
  /api/hello` — a WZRD answer = attach, a foreign answer = red lamp "port
  taken" (never spawn), nothing + a live `hub.lock` pid = wait, nothing
  else = **spawn detached**: absolute `uv` (`WZRD_UV`, else
  `~/.local/bin/uv`, `/opt/homebrew/bin/uv`…; a Finder launch has no
  `PATH`) `run --directory <repo> --extra mcp --extra engine --extra
  islands python -m wzrd_mcp`, own process group, output appended to
  `.wzrd/hub.log`, a reaper thread so no zombie. The hub **outlives the
  app** and exits itself after 1 h idle; the monitor re-checks every 3 s
  and emits `hub:status`, but never respawns on its own — "Start hub" /
  "Restart hub" (SIGTERM the pid, wait ≤ 30 s for it and the port —
  uvicorn drains open long-polls — then SIGKILL, then spawn) do. Repo dir:
  `WZRD_REPO`, else this checkout (compile-time). Commands: `hub_status`,
  `hub_ensure`, `hub_restart`, `hub_call {method, path, body}` — the thin
  `/api` proxy (a dependency-free loopback HTTP/1.1 client, chunked
  replies handled, 60 s timeout for the job long-poll).
- **Reads work with the hub down.** `src-tauri/src/library.rs` opens
  `library.db` **read-only** (`rusqlite`, bundled SQLite) and returns the
  hub's exact JSON shapes (`library._item`, `/api/library/list`,
  `/api/library/item`, tags), so the webview has one `LibraryItem` type.
  `api/hub.ts::listFolder` asks the hub when it's up (that call also
  indexes what's new — S3 "index on open") and the reader otherwise.
  Writes (label, move, delete, roots, selection) are hub-only.
- **Media are URLs, never base64.** Tauri's asset protocol
  (`protocol-asset` feature, scope granted at runtime by
  `library::allow_scope` to the library home, every root and the projects
  root — re-run by `library_roots`) and, for the browser seat, the
  devport's `GET /file?path=` (same scope check, one `Range` so `<video>`
  seeks). `api/hub.ts::mediaUrl` picks. 500 indexed items list in ≈ 13 ms
  (hub) / 11 ms (reader); thumbs lazy-load.
- **State:** `state/hub.ts` (its own small store — the hub outlives
  project switches): lamp status (Tauri event; the browser seat polls
  `hub_status`), the job feed (a refcounted long-poll on `/api/jobs`: the
  Studio room and the pill hold it; 2 s while a job runs, else up to 20 s,
  returning early on any change), and **`selection.project` = the open
  show's folder name**, written on every show switch and whenever the hub
  comes up — agent tools called without `project=` act on it.
  `state/studio.ts`: open folder, listing (2 s poll while Studio is
  mounted), grid selection + cursor, "show ≤ N", monitor item, and the
  mirror of the hub's L5 selection. The **open folder is the target
  folder** (S1, written on open); grid selection → `items`; the tray →
  `styles` (debounced 300 ms; the poll reads it back, so an agent's
  edit shows in the tray; a pending write blocks the read-back). Labels
  are optimistic (`pendingLabels` survive a poll until the hub echoes).
- **The room:** `Split` `studio.cols` = `FolderTree` (roots, lazy tree,
  "+ Add folder…" = a new root indexed in place, "Index this folder" =
  deep index) · `MediaGrid` (static thumbs, score/codec/tag badges, a
  dashed tile per file still being indexed, a **job tile** per running
  job whose `dest` is this folder with elapsed/ETA, "show ≤ N", "Delete
  ≤ N…" = dry run then confirm, **Trash (n)** = `RecentlyDeleted`, a
  hover ⤢ = Lightbox) · `Monitor` + `RecordPanel` (score buttons, **"In
  show" pins** — the open show's asset names pointing at this file, ×
  unpins, "★ Pin to show as…" = `NamePrompt` with the roles →
  `pinAsset(openShow, name, path)` — tags (agent tags ✦, read-only),
  notes, made-for chip, settings, inputs, "+ Styles"); `StyleTray` along
  the bottom with "Renders land in". Tiles drag out as `application/x-wzrd-paths` (the tray takes
  them). No telemetry subscriptions.
- **Render panel (R1)** — the side tab next to Record
  (`RenderPanel.tsx`, tab remembered in `localStorage wzrd.studio.side`):
  TextureFlow (recipe from `/api/recipes`; the tray's styles override the
  recipe's), Image (`nano_banana`) and Video (`kling_v25_image_to_video`,
  5/10 s). "Made for" region → size = the region's aspect, long side 640
  (step 32) and `region` in the args; TextureFlow's **Shape input**
  (`AssetPicker` over the open show's assets, `surface_lit` / `surface`
  first → `control_input`) with "Use as shape (ControlNet)" (→
  `use_controlnet1` + `controlnet_strength1`, fader 0.35–0.65, default
  0.45) and **Mask** (→ `diffusion_mask`), each sent only when set (those
  two rows use a `<div>` group, not `Field`'s `<label>` — a label forwards
  every click inside it to its first button); Draft = `n_steps 4`, no upscale;
  count 1–12. `/api/estimate` (debounced) shows ≈ cost vs today's spend;
  over the cap → confirm → `allow_over_cap`. Go = `POST /api/render`
  (`wait=false` jobs, `by: app`), landing in the open folder — the hub
  pins it at submit. Job tiles carry ✕ = `/api/jobs/cancel`.
  **More like this ×N** (record footer, for texture_flow / nano_banana /
  kling) re-runs the **job's submitted args** (`library_record.job_args`
  — the record's own args carry provenance-only keys like `model` that
  the tool rejects) minus `seed`. **Make playable** (record footer, or the
  folder's "Make playable (n)" for every video without a HAP sibling)
  queues `to_hap`; the `.mov` lands beside the source. `+ Deck <name>`
  adds to the open show's active deck (see "Clip decks").
- **Monitor (S10):** HAP → `<NativeSlot id="studio">` + `previewFile(path)`
  (engine `preview.file`, render-engine.md §1b — no leg renders);
  mp4/webm → `<video>`; stills → `<img>`; DXV/other → thumb + "Make
  playable". Unmount sends `preview.file null`.
- **Keys** (only while no input has focus and no modal / lightbox is
  open; Studio's own handler in `Studio.tsx`, and `shortcuts.ts` returns
  early in Studio so T/M/digits never TAKE/mute/select layers): arrows
  move (⇧ extends; ↑↓ step a grid row via `gridColumns()`), space = play
  in the monitor, `F` / Enter = full-screen `Lightbox` over the visible
  items (← → move the cursor with it; HAP/DXV show their thumb — the
  monitor plays them), `0–9` score, `−`
  clears it, `T` tags (autocomplete from every tag in use, Tab completes),
  `N` notes (⌘Enter), `D` add to the open show's active deck, `M` move (folders the tree knows, or a typed
  absolute path), `⌫` delete media — **soft** (show-setup G6: the hub
  moves the files to a trash for 30 min; the toast carries **Undo**, the
  grid's Trash button restores later; records kept — S7; a
  multi-selection needs a second ⌫ within 2 s), Esc deselects. Keys act on the selection,
  else the cursor item (`studio.ts::targets`). Measured: 20 scores = 40
  keys, all landed in `library.db` in order.

## Shared media components (`src/ui/media/`, show-setup M1, 2026-10-16)

One set of media building blocks for Studio and the show-setup wizard
(`components/setup/`), so a tile, a viewer, a delete and a pin behave the
same everywhere. Exact props live in each file's header.

| File | What |
|---|---|
| `types.ts` | `MediaRef` (`path` + optional name/kind/thumb/size/`LibraryItem`), `toMediaRef`, `isImagePath`, `kindOf`, `thumbOf`, `sameDir` |
| `modalState.ts` | `modalOpen()` refcount (`useModalLock`) + the **Esc layer stack** (`useEscapeLayer`) |
| `Modal.tsx` | portal dialog: title bar + `headerRight`, scrolling body, `footer`; Esc closes, backdrop only with `closeOnBackdrop` |
| `Lightbox.tsx` | full-window viewer: ← → step, Esc, F fit / 1:1 (drag pans), Space = play/pause or hold-to-flip the `compareWith` blend; `actions(ref)` in its top bar |
| `MediaTile.tsx` | static thumb + name, `badges`, hover ⤢ / ★ / ✕ (each only when its handler is given), ★ asset chips (`pins`), selected ring |
| `JobTile.tsx` | a running job's tile (elapsed / ETA / ✕ cancel); Studio's `MediaGrid` uses it too |
| `MediaGallery.tsx` | a folder (`listFolder`, 2 s poll while mounted; hub down = reader, no ★/✕) or fixed `items`; job tiles for jobs whose `dest` is the folder; Lightbox; ★ → `NamePrompt` → `pinAsset`; ✕ → soft delete + Undo |
| `AssetPicker.tsx` | select-like button → popover: the show's named assets (`suggest` first; unpinned / missing greyed), the newest 24 files in the show's working folders (from `ProjectInfo.layout`), "Choose file…" (Tauri only) |
| `NamePrompt.tsx` | popover: name field (`^[a-z0-9][a-z0-9_-]{0,47}$`, typed lower-case) + suggestion chips; Enter submits |
| `useFileDrop.ts` | Finder drops via Tauri's `onDragDropEvent` (absolute paths); a stack — only the newest *enabled* user gets hover + drops; no-op in the browser seat |
| `RecentlyDeleted.tsx` | "Trash (n)" → popover: thumb, name, folder, "until hh:mm", Restore / Restore all / Empty now (two-click); polls 5 s only while open |
| `trash.ts` | `softDelete(paths)` = `/api/library/delete` + a toast with **Undo** (restores exactly the `trashed` ids that call returned); `useTrash.version` re-counts the Trash button |

Plus `state/projectAssets.ts`: one `ProjectInfo` per show (`GET
/api/project` — assets, roles, folder map, canvas), polled every 3 s while
any component holds `useProjectInfo(show)` (refcounted), so an agent's
`pin_asset` shows up live; after an app-side pin call
`refreshProjectInfo(show)`. `pinsByPath(info)` = path → asset names.

- **Why a refcount + layer stack instead of each dialog catching keys:**
  room shortcuts are window listeners registered once at boot; the only
  cheap, reliable gate is a module flag they check first (`shortcuts.ts`,
  Studio's handler). Esc is a single capture-phase window listener that
  calls the **topmost** layer and stops the event, so a Lightbox over a
  Modal closes alone. An open `<Popover>` (marked `data-popover`) beats
  every layer: its own listener closes it, the next Esc reaches the layer.
- **Z-order:** Modal `z-[52]` < Lightbox `z-[58]` < Popover `z-[60]` (a
  NamePrompt opened from a lightbox's ★ sits on top) < toasts `z-[90]` (an
  Undo stays clickable over a dialog). Studio's `KeyPrompt` is `z-50`.
- **Toasts can carry one action** (`store.toast(text, tone, {action:
  {label, run}, ms})`, 10 s by default with an action) — rendered by
  `TopBar.tsx::Toasts`.
- Everything is portalled to `<body>`, a sibling of the page — never an
  ancestor painting over a `<NativeSlot>` (the hole rule); a modal simply
  covers the native image while open.

## Show setup wizard (`components/setup/`, show-setup W1, 2026-10-16)

The venue-day steps before anything renders, as one `Modal`
(`SetupWizard.tsx`, mounted once in `App.tsx` as `SetupHost`). Entry
points: project menu **New show…** / **Show setup…**, ⚙ Setup → **Show
setup**. State: `state/setup.ts` — `openSetup({project?, step?, newShow?})`
/ `closeSetup()`, the wizard's **own `project`** (default = the open show;
reopening on the same show resumes the last step), and per-show drafts
(the last surface run + corners + sliders, the segmentation prompt fork +
settings, the last region extract + typed ids/labels) so jumping between
steps loses nothing. It passes `project=` on every call; the engine keeps
playing the open show. Steps are chips (✓ = that step's assets are pinned
in `assets.json`, read through `useProjectInfo`, so an agent's pins tick
them too), freely clickable once a show is picked; footer Back / Next.

| Step | File | Calls |
|---|---|---|
| Show | `ShowStep.tsx` | pick a show (`existing_projects`) or name + canvas → `runTool('set_project', {name, canvas})`; a legacy show without a canvas can lock one; pinned-asset thumbs + the folder map (`info.layout`) |
| Photos | `PhotosStep.tsx` | slots **Lit** / **Ambient**: Finder drop (`useFileDrop`; drops go to the targeted slot, two files = lit + ambient), Choose file…, Newest from Downloads (`import_photos` without paths, count 1), or the `captures/` gallery → `import_photos {conform: false}` + `pinAsset(photo_lit \| photo_ambient)`. **Project white** = `setTestPattern('white')` on the open show's output (hint + display buttons while OUTPUT is off); turned off when the step unmounts (leave the step / close the wizard) |
| Surface | `SurfaceStep.tsx`, `QuadEditor.tsx` | the lit photo with a 4-corner quad (18 px grab radius, ⇧ = ¼ speed, a 4× loupe while dragging, arrow keys nudge a clicked corner by one photo pixel). Sliders = `rectify_surface`'s own defaults. **Run** → `rectifySurface` (corners only when moved by hand; an auto run's returned corners fill the quad; "Auto-detect" forgets manual corners). Outputs side by side (Lightbox compares against the lit crop), **Use this run** pins `surface_lit` + `surface` (+ `surface_ambient`), ★ pins one. **Trials** = `surfaces/` filtered to `*_surface.png`; a click reloads its outputs + settings from `libraryRecord(id).record.args` (auto-detected corners aren't in the record, so an auto trial loads without a quad) |
| Segmentation | `SegmentationStep.tsx` | input `AssetPicker` (default `surface_lit`); prompt = a fork of `listPrompts('segmentation', project)` (show scope first, then WZRD), `{n_regions}` from the stepper via `fillPrompt`, Save to project / Save to WZRD (`NamePrompt` → `savePrompt`), Reset; count 1–8, 1K/2K, thinking minimal/medium; estimate + spend-cap confirm exactly like the Render panel; **Generate** → `startRender('nano_banana', {…, aspect_ratio: 'auto', dest: <show>/segmentation}, count)`; a `MediaGallery` of that folder shows the job tiles, ★ pins `segmentation` |
| Regions | `RegionsStep.tsx` | `segmentation` + `surface` pickers, the `extract_color_regions` sliders → `runTool` → an overview (surface + every mask tinted in its colour, hover a card to single one out — canvas compositing only, never `getImageData`: media URLs are another origin, the canvas is tainted) + a card per region (shape swatch, area %, **id** slug-unique, label) → **Build layer pack** = `runTool('build_layerpack', {masks, mask_ids, surface_image, source_capture: photo_lit, tags: {layers: {id: {label}}}, pack_name})`; result note, then **Open show** (two clicks, `actions.ts::openProject(info.scene_path)`) once the show has a scene |

## Mechanisms worth knowing

- **Layer mixer = reserved overrides.** Opacity / mute / solo are §5.5
  overrides under `@opacity` / `@mute` / `@solo` on the binding (engine
  folds them into the per-pass `layer.opacity`, render-engine.md §2.5). So
  they're per leg, zero-rebuild, persisted in `session.json`, copied by
  TAKE/PULL. The store mirrors both legs' override tables
  (`store.overrides`, hydrated from `param.list`, patched optimistically by
  `actions.pin`). Solo click is exclusive, ⇧-click additive.
- **Pins vs scene values.** In Perform, dragging a scalar *pins* it on the
  control leg (amber `PINNED`, tick at the scene value, double-click
  releases). In Prepare (`mode="author"`) the same fader writes scene.json.
  `bakeDesignPins` (⌘S, and every Save look) writes design pins into the
  scene and clears them.
- **Modulation** is a driver with `range: [from, to]` and, for clocks, a
  `shape` (render-engine.md §2.1 "Clock shapes"). `MOD_SOURCES` in
  `actions.ts` maps VJ words (Kick/Snare/Hats = `audio.onset`
  low/mid/high, Bass/Mids/Highs = `audio.band`, **Tempo** = `clock.bars`/
  `clock.beats`, **Free** = `clock.phase`) to driver specs; `modSourceOf`
  matches on driver + band only. New clock routings default to
  `DEFAULT_MOD_SHAPE = 'pingpong'` at 4 bars / 8 s (the engine default
  without `shape` stays `saw`, for old scenes). UI: the row's `ModChip`
  opens `ModEditor.tsx` — a live scope (canvas fed by `useStore.subscribe`
  + rAF while open, no React render per frame, no new channel demand),
  source grid, 6 shape glyphs, cycle (`TEMPO_RATES` / `FREE_RATES`), onset
  decay, ⇄ invert (swaps the range ends), remove. `RangeFader` edits
  rest→peak (gradient dim→bright) with the live value riding inside.
  Structure → DESIGN only, like every range/source edit.
- **Layer delete** (`actions.ts::deleteLayer` / `deleteLayerOnHands`):
  × on every row (Prepare list, Perform mixer), the inspector header, and
  Delete/⌫. On LIVE it deletes nothing and moves hands to DESIGN (a
  structural edit). An in-memory undo stack (20) restores at the old
  index (⌘Z, the panel's ↶ Undo) — it only undoes layer deletes, it is
  not a scene undo. Cleared on project switch (`resetActionState`).
- **Selection** (Prepare): clicking a region selects it + the first layer
  lighting it; clicking it again steps through its layers, then
  deselects; clicking the selected layer row again, an empty map spot, the
  margin round the canvas or Esc (`actions.ts::clearSelection`) clears
  both. Perform's `LayerStack` always re-selects a layer, so Esc only
  deselects in Prepare.
- **Code drawer saves per file** (`MonacoPanel.tsx`): ⌘S in the editor /
  the Save button saves the *open tab* — a shader through `effectUpsert`
  (engine validates before writing; red strip with line-mapped errors,
  amber on a yellow probe), scene.json through `commitSceneNow`. Dirty ●
  per tab; unsaved shader buffers live in a module store per project, so
  closing the drawer keeps them. One Monaco model per file (`path`).
- **Panel layout** (`ui/layoutSplit.tsx`, `ui/layoutStore.ts`,
  hand-rolled, no dependency): each `Split` has one `fill` pane; others
  have a CSS default (`'248px'`, `'36%'`) until dragged, then a stored px
  size. Dragging an edge resizes the pane on the far side from the fill;
  double-click an edge = default. Unlocked, a ⠿ grip drags a pane onto a
  sibling to swap (keyed, no remount). **Locked by default** (⚙ Setup →
  Lock layout; an amber "LAYOUT UNLOCKED · LOCK" chip shows while off;
  Reset per room / all). Persisted per project in
  `<project>/ui_layout.json` (`{schema:1, locked, sizes:{"perform.cols/side":340},
  order:{"perform.cols":["side","main"]}}`, absent key = default) via
  `ui_layout_read/write {scenePath}` (`src-tauri/src/layout.rs`, atomic,
  devport twins, debounced 400 ms, flushed before a project switch). Not
  in `project.json`: the MCP rewrites that from another process, and one
  writer per file keeps the locked canvas safe.
- **TAKE arming** compares design vs live scene JSON (`designDiffers`,
  key-sorted, ignoring `transport` because TAP moves both legs live) **and**
  the two legs' override tables in the store (mixer `@mute`/`@opacity`/
  `@solo` + pins; neutral entries ignored) — a design-only mute is a
  pending TAKE. Only overrides the store knows about count: it's patched by
  `actions.pin` and re-read by `hydrateOverrides` (hydrate, take, pull), so
  a `param.set` from an external client (MCP, raw WS) isn't seen until the
  next hydrate.
  Live JSON is re-read after every ramp completes (`deck` channel → idle).
- **Looks** are plain scene files in `<scene_dir>/looks/<name>.json` + a
  `.jpg` thumbnail (Tauri commands `looks_list/look_read/look_write/
  look_delete` in `src-tauri/src/rpc.rs`). Click = cue into DESIGN, double
  click / ▶ = cue + TAKE (`recallLook` awaits the design apply first via
  `commitSceneNow`, so GO never fades to the previous design).
- **Auto-pilot** (`state/autopilot.ts`) walks the looks bank every N bars:
  recall 2 beats before the boundary, bar-quantized TAKE lands on the line.
  UI-driven stopgap; it stops when the app closes (the engine playlist in
  the roadmap §5.6 supersedes it).
- **Shared mask assets** (`state/masks.ts`): each region's mask is loaded
  once → tinted overlay, 256-px pick map, silhouette thumbnail (the region
  glyphs in every list) + the pack's `surface.png` for the map's photo mode.
- **The native image lives *under* the page** (the underlay, host side in
  [render-engine.md](render-engine.md) §1b). `<NativeSlot params>`
  (`NativeSlot.tsx`) is a hole: it measures itself and the engine draws
  there, beneath a transparent webview, so anything inside or over it
  (mask overlays, the warp grid, dropdowns, modals) composites on top.
  Three users, differing only in `PreviewParams`: Perform's monitor
  (`NativePreview.tsx` — the deck-toggle leg), Prepare's `SurfaceMap`
  (design + surface photo per the PHOTO/LIGHT/BOTH toggle, BOTH's photo
  weight 0.15 linear ≈ the old 0.42 sRGB alpha; the canvas only draws
  region overlays, cleared transparent) and Align's `WarpCanvas` (the
  program view — the projector's exact output incl. local corrections,
  dimmed 0.55, over the projector rect). The mode is resolved once at boot
  (`api/previewMode.ts`): `underlay`, `window` (`WZRD_PREVIEW=window`, the
  old child window — Perform only, Prepare/Align draw the JPEG) or `none`
  (browser — JPEG everywhere, approximate). Components branch on
  `nativeUnderlay()`. The JPEG `preview` channel is demand-gated like
  every streaming channel (next bullet); never hold it by default — that
  kept the JPEG readback and the design leg running in every room.
- **Telemetry is demand-driven and batched** (`api/telemetryDemand.ts`,
  shell `engine.rs::ChannelDemand` + `telemetry_loop`). Sticky channels
  (`deck`, `masters`, `fps`, `hot_reload`, `changes`, `alignment`, …) and
  `log` always flow. The streaming ones — `preview`, `audio`, `transport`,
  `drivers`, `frame_stats`, `perf` — reach the webview only while a mounted
  component holds `useTelemetryDemand(channel, active)` (or
  `demandChannel` imperatively): Tempo → `transport`, AudioPanel →
  `audio`, a driven `LayerParams` row → `drivers`, Debug → `frame_stats` +
  `drivers` + `perf`, JPEG fallbacks → `preview`. The webview refcounts
  and sends the shell the *union* (`telemetry_demand {channels}`, once per
  microtask, only on change — a reload can't leak demand). The engine
  skips the work behind an unsubscribed channel (`audio`/`transport`/
  `drivers` payloads, the JPEG readback + design leg). A component that
  shows a streaming channel without demanding it just sees stale data.
  The shell emits `engine:telemetry` as an array, at most once per ~33 ms
  (one Tauri emit = one webview eval); within a batch a streaming channel
  keeps only its newest frame. Select primitives from these slices
  (`LayerParams` reads the row's `value`, not the row object; Tempo the
  integer beat) or every frame re-renders the subscriber.
- **The cue monitor samples design, it doesn't stream it**
  (`DeckPanel.tsx::CueMonitor`, shown on LIVE). It holds `preview` demand
  just long enough for one fresh JPEG: on mount, ~250 ms after any design
  change (applied `hot_reload`, scene, design pins, cue, take/pull phase)
  and every 1 s; hovering it peeks at full rate. Streaming it made the
  engine render the whole design leg + a 15 fps readback for a 118 px
  thumbnail during the whole show.
- **Idle-cost rules for animations:** animate only `opacity`/`transform`.
  Glows are pre-painted shadows whose opacity moves (`.arm-glow` on the
  armed TAKE — the element must not clip, so its fill lives in an inner
  wrapper; `Lamp`); meters scale with `transform`, no CSS transition; the
  beat lamps re-render once per beat and decay via `animate-beat`.
- **The hole rule: nothing between a `<NativeSlot>` and the page root may
  paint a background over it**, or the image silently vanishes. `html`,
  `body`, `#root` go transparent under `:root[data-underlay]`; a `Panel`
  holding a slot takes `hole` (no `bg-panel`); `.dim-grid` stays off the
  slot (Align draws its grid in the SVG, around the projector rect). In dev
  the slot walks its ancestors on mount, `console.error`s and outlines any
  that paint a background red. Siblings may paint freely.

## Browser bridge (dev harness)

The same Vite build runs in a plain browser: `src/api/transport.ts` picks
Tauri `invoke`/`listen` inside the shell, else `wsBridge.ts` — JSON-RPC to
the engine's WS (`ws://127.0.0.1:9123`, `?ws=` overrides; declares
`hello {actor:"ui"}`) plus, for everything the shell does itself (files,
looks, projects, clips, engine restart, hub + library), the shell's **debug-build-only
HTTP twin** `src-tauri/src/devport.rs` on `127.0.0.1:9124`
(`WZRD_DEVPORT`), reached through Vite's `/__shell` proxy. It calls the
same Rust command functions on the main thread (the `hub_*` /
`library_*` twins run on the connection thread instead — a hub call can
block for seconds — and `GET /file?path=` serves library media), so a browser seat follows
project switches (the old Vite file middleware was pinned to
`WZRD_SCENE`). A switch shows up in the browser as an engine-socket
reconnect → hydrate. Use it to drive and screenshot the UI with browser
tooling while `pnpm tauri dev` runs. Two seats are independent: edits from
one aren't mirrored into the other's store (both declare actor `ui`).
Browser-tooling traps (Claude in Chrome): CDP's synthetic mouse drag
doesn't fire HTML5 drag-and-drop — exercise the deck grid's drop handlers
with dispatched `DragEvent`s; and the screenshot's coordinate frame can
differ from the page's CSS viewport (seen: 1456 vs 1543 px), so
coordinate clicks miss — click via element refs (`find`). In dev, the
modules are importable from the page (`await import('/src/state/store.ts')`)
to read store state as evidence.

**Driving the Tauri webview itself (agents).** The native underlay is
invisible to the browser seat and to webview screenshots. Two devport hooks
cover it without taking the operator's keyboard focus: `POST
/cmd/webview_eval {js}` runs a script in the app's own webview (e.g.
`import('/src/state/store.ts').then(m => m.useStore.getState().setRoute('align'))`
to switch rooms — not `localStorage` + reload: a stored `align` route boots
into Perform; the afk-session `room` helper does this), and `POST /cmd/debug_log {msg}` (callable from that script via
`/__shell/cmd/debug_log`) writes `[webview] …` into the app log — the way
to read a value back. The engine's `preview.snapshot` RPC returns exactly
what the native slot shows; `screencapture -l <windowid>` grabs the real
window (underlay included) even when it is covered.

## Traps

- **Anything that can move a `<NativeSlot>` without resizing it must call
  `layoutStore.notifyLayout()`** (fires `wzrd:layout`, which NativeSlot
  listens to) — a ResizeObserver misses a pure move. `layoutSplit` already
  covers drags, swaps and panes mounting/unmounting.
- **Preview-slot configs and engine commands travel on different
  channels** (`preview_set_slot` → the host's preview-cfg channel; RPCs
  like `preview.file` → the engine command queue, drained first). React
  StrictMode's dev double-mount sends a hidden "main" slot config between
  `preview.file` calls, so the engine only drops the Studio file when a
  *visible* non-file slot takes over — never on a hidden one (2026-10-15).
- **`tauri dev` doesn't watch `render-core/`**: after an engine edit,
  `touch wzrd-app/src-tauri/src/lib.rs` to make it rebuild.
- **`import('/src/…')` from a page can get a second module instance**
  after Vite HMR updated that module (`?t=` URLs): calling `setRoute` on it
  does nothing. Drive the UI through DOM clicks, or reload first.
- **Monaco `onChange` must route by the changed model's URI, never by
  `activeTab`**: on a tab switch `@monaco-editor/react` can fire a change
  for the new (transient `loading`) model through the previous render's
  closure — routing by tab wrote "// loading…" into `sceneJson` and blanked
  every layer (found 2026-10-14). Same family: `onMount` runs once, so
  anything registered there (the ⌘S command) calls through a ref.
- **The full-frame "background" region** means almost every map click hits
  a region; deselect relies on clicking the selected thing again or Esc.

- **No emitted `vite.config.js`.** Vite resolves `.js` before `.ts`; a
  `tsc -b`-emitted copy silently pinned a stale config (found 2026-10-09).
  `tsconfig.node.json` now emits to `node_modules/.tmp/`.
- **Monaco is bundled, not CDN-loaded** (`monacoSetup.ts`: editor core +
  JSON only, local workers, `wzrd` theme). The CDN default rendered a blank
  editor offline — venues are often offline.
- `pnpm add` under a running dev server breaks Vite's resolved paths;
  restart `pnpm tauri dev`.
- A Rust edit makes `tauri dev` restart the app before the old process
  has let go of its ports: the dev port retries its bind, but the engine
  WS (9123) can lose the race — the app then comes up with the engine
  down and the banner's "Restart engine" fixes it.
- Colour defaults from `effect.describe` arrive as float arrays — convert
  with `LayerParams.tsx::toHex` before writing a scene.
- **A page reload skips React unmount cleanups**, so a `<NativeSlot>` never
  gets to hide itself; `initPreviewMode` resets the slot to hidden at every
  boot. Same family: an effect that only measures on mount misses a
  component that renders a "waiting" stub first (`WarpCanvas` re-measures
  once the alignment doc arrives — before that fix, loading straight into
  Align drew infinitely large handles).

## Keyboard (single keys, ignored while typing)

D / L hands · T take · ⇧T cut · P pull · B blackout · F (hold) flash · Z
freeze · Tab tap · `` ` `` bar line · ↑↓ / 1–9 select layer · M / S mute /
solo · [ ] opacity · ⇧1–9 cue look (⇧⌥ = GO) · ⌘S save design (inside the
code drawer: save the open file) · Delete/⌫ delete selected layer · ⌘Z
restore last deleted layer · Esc deselect (Prepare) · ⌘0 Studio · ⌘1
Prepare · ⌘2 Perform · ⌘3 Align · ⌘4 Debug · ? help. Studio has its own
single keys (see "Studio + the hub"); none of the above fire there.
**While a modal or lightbox is open nothing here fires** (not even ⌘0–4;
`ui/media/modalState.ts::modalOpen()`, show-setup G11) — digits in a
dialog must not select layers, T must not TAKE.
Faders: press = jump, ⇧-drag = fine, double-click = reset. **No mouse
wheel** — the wheel always scrolls the panel (a future MIDI bridge drives
faders through the same `actions.ts` calls; gaps listed in
[midi-controller.md](../TODO/midi-controller.md)).
