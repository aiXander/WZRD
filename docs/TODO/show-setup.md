# Show setup: photos → surface → segmentation → regions

> **Status: landed 2026-10-16** (packages B1 backend, M1 shared media
> components, W1 wizard + `white` test pattern), gone through end to end
> on screen with `test_imgs/night.jpg` + `day.jpg` (show `setup-test`).
> Open: the on-site checks below. The decisions still constrain work on
> these parts; once they've survived a venue day, fold the ones that
> matter into the reference docs and retire this file.
>
> Current state lives in the reference docs:
> [../reference/mcp-server.md](../reference/mcp-server.md) (folder map,
> named assets + `asset:` refs, prompt templates, soft delete/trash,
> `rectify_surface`, the `/api/project` · `/api/tool` · `/api/prompts` ·
> `/api/trash` routes; typed client `wzrd-app/src/api/project.ts`) and
> [../reference/operator-ui.md](../reference/operator-ui.md) ("Shared media
> components", "Show setup wizard", Studio's lightbox/undo/pins) +
> [../reference/render-engine.md](../reference/render-engine.md) §2.8
> (`white`).

## What Xander wanted

The venue-day steps before anything renders, as a popup flow run once per
show from either room, that an agent can drive equally (results show up
in the UI whoever made them): **Photos** (lit = projector full white,
ambient = room lit) → **Surface** (rectify + align + darken, sliders,
re-run, pin the best run) → **Segmentation** (Nano Banana on the lit crop,
prompt forked from a WZRD template, save to project / to WZRD, batches) →
**Regions** (colour map → masks → named regions → layer pack). Shared
media components everywhere: tile, full-screen viewer, soft delete (30 min
trash), named assets that show up in dropdowns.

## Decisions (stated defaults, 2026-10-16 — not grilled)

| # | Decision | Why |
|---|---|---|
| G1 | The wizard is a modal (`SetupWizard`) from the project menu and ⚙ Setup; steps freely jumpable, ✓ when their assets are pinned; it works on any show and passes `project=` on every call — the engine keeps playing the open show | Once-per-show work; a fresh show has no scene to open |
| G2 | Named assets in `<project>/assets.json`, written only by the hub; one name → one file; well-known names `photo_lit`, `photo_ambient`, `surface_lit`, `surface`, `surface_ambient`, `segmentation` | One writer per file; per show, copied by Save as |
| G3 | `asset:<name>` works as any media argument of any tool (expanded in `logged_tool`) | UI and agent name things the same way |
| G4 | Fixed folder map per show (`project.LAYOUT`), created by `set_project` | The UI lists folders; whoever wrote a file, it shows |
| G5 | `rectify_surface` = lit (+ ambient) (+ manual corners) → one run of `<run>_lit/_surface/_ambient.png`; `prepare_surface` stays for old callers | Two pins need two files; manual corners rescue a bad detection |
| G6 | Deletes are soft everywhere: trash, flushed after 30 min, restorable | Recover accidental deletes |
| G7 | Prompt templates: WZRD scope `library/prompts/<kind>/`, project scope `<project>/prompts/<kind>/`; `{n_regions}` is the one placeholder | Iterate in the UI, keep the good ones |
| G8 | Setup tools run via `POST /api/tool` (sync); renders stay on `/api/render` jobs | One generic route |
| G9 | The projects root is always a library root | Studio shows every show's folders |
| G10 | No Rust shell changes for this; the engine change is the `white` pattern | Don't restart Xander's app |
| G11 | An open modal blocks room shortcuts (`modalOpen()`) | Digits in a viewer must not score; T must not TAKE |

## Open

- **On-site checks (first real venue day):** "Project white" on a real
  projector (OUTPUT on → the whole raster white); Finder drag-drop of
  photos into the Photos step (Tauri `onDragDropEvent` — untested; the
  browser seat can't do it); auto-detection on a real night photo (the
  test pair is synthetic).
- **Later, not scheduled:** a "mask non-reflective areas" step between
  Surface and Segmentation (the `mask/default` prompt is shipped for it);
  in-app agent chat ([vj-toolbox.md](vj-toolbox.md) phase 7) can drive the
  same steps — they're plain tool calls.
