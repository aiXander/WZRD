---
name: run-sheet
description: "Venue day, photos to show: locked canvas, regions, layer pack, a clip library from style recipes, the pre-show checklist. Use when setting up WZRD at a venue or preparing a show."
metadata:
  kind: workflow
---

# On-site run sheet

From a handful of venue photos to a project the operator can play: every
surface split into named regions, each region with a row of passing clips in
the app's clip grid, the show checklist ticked. Work through the steps in
order; each ends on a check you can see.

The operator plays the show. You prepare it: you render and file clips, bind
on the design deck only when asked, and taking anything live is theirs.

## 1. Project and canvas

`set_project(name="<venue>-<date>", canvas="<projector aspect>")` — the
canvas is locked once, so ask the projector's aspect if you don't know it
(most are 16:9 → 1376×768). Check: the reply shows the canvas and today's
spend against the daily cap (raise it only on the operator's word).

**From here on, pass `project="<venue>-<date>"` on every call** that takes
it (import, surface, regions, pack, every render, `to_hap`). Nothing is
remembered between calls; a call without it works on whatever show the app
has open.

## 2. Photos → the surface

Ask what the photos are: one shot from the projector's position, or a
night/day pair (a white frame projected at night + the same view by day).

- One shot: `import_photos()` (newest from ~/Downloads, AirDrop lands there)
  or `import_photos(paths=[…])` — upright and conformed to the canvas.
- Night/day pair: `import_photos(paths=[night, day], conform=False)`.

Then follow skill `surface-prep` to the masked dark surface. Check: you
`view_media`'d the surface and the operator agrees it matches the room.

## 3. Regions

Every surface the show treats separately is one region: each fabric panel,
the ceiling, each wall section, a pillar.

1. Agree the region list with the operator and give each a short semantic
   id they'll say out loud: `fabric_left`, `fabric_mid`, `ceiling`,
   `wall_back`.
2. Segmentation map: `nano_banana` with the "color segmentation map"
   template from skill `image-prompting`, `image_urls=[surface]`,
   `aspect_ratio="auto"`, naming every region in the prompt.
   `view_media` it beside the surface: borders on real edges.
3. `extract_color_regions(image=<map>, surface_image=<surface>)`. Match each
   returned region to its id by looking at its `region_surface_image`.

Check: one region per agreed surface, none missing, no slivers (raise
`min_area_fraction` if there are).

## 4. Layer pack

`build_layerpack(masks=<region_mask per region>, surface_image=<surface>,
mask_ids=<the ids, same order>, tags={layers: {id: {label, tags}}, groups:
[…]}, pack_name="<venue>")`. Group what moves together (`fabrics`,
`walls`). It writes the starter `scene.json` too. Check: the reply lists
every id; the operator opens the project from the app's project menu and
sees the regions on the surface map.

Copy the clip effects into the project so the clips can be played with:
`cp -R library/effects/* <project_dir>/effects/` (repo root; the project dir
is in `set_project`'s reply). No shell → ask the operator to run it.

## 5. The clip library

Load skill `textureflow` (settings, and the style recipe library).

1. `list_recipes()`; `view_media` the previews of the likely ones. If the
   operator has styles in the Studio tray, `studio_selection()` gives them.
   `find_renders(recipe=<name>)` / `find_renders(style=<path>)` shows what
   scored well (and badly) with them before. Agree with the operator which
   recipes go on which regions — one recipe set across regions reads as one
   show; a contrasting recipe marks a region out.
2. **Drafts first**, one per region × recipe:
   `texture_flow(recipe=<name>, region=<id>, project=<show>, width=…,
   height=…, use_upscale=false, upscale_esrgan=false, wait=false)` with
   width/height at the region's `aspect_ratio` (from step 3), snapped to
   multiples of 32, short side ≥ 384. ≤ 6 jobs in flight; poll `jobs()`.
3. `check_frames(<clip mp4>)` on every result, and `view_media`. A `flash`
   fail never goes on the wall — drop it. `haze` warns: re-render darker, or
   plan a `lumaKey` effect on that region.
4. Show the operator a contact sheet per region; finals for what they pick
   (drop the two upscale args; vary `seed`, `mapping_mode`, `motion_scale`
   for variety within a recipe). Aim for 3–5 passing clips per region.
5. A new look the operator loves → `save_recipe(name, from_clip=<clip>,
   when_to_use=<mood, motion, which surfaces it reads on>)`, on their word.

Every render lands in the media library (`~/WZRD/media`: the folder open in
Studio, else `inbox/`) with its HAP clip beside it, recorded with the show
and its region (`project=` + `region=`), so the app files it for this show.
Watch the `spend` line in each reply (today's spend, all shows) and stop at
the 80 % warning to ask.

Check: every region's row holds ≥ 3 clips that passed `check_frames`.

## 6. Show checklist

Run through it with the operator before doors and report each line:

| | Check |
|---|---|
| ☐ | Alignment done in the app's Align room (operator, at the venue) |
| ☐ | Every clip on the grid passed `check_frames` — no flash fails |
| ☐ | Audio server running and the guitar moves `audio.*` (operator) |
| ☐ | `textureflow_warm(minutes=<set length + 30>)` ~10 min before doors, if clips will be topped up live; `textureflow_warm(minutes=0)` after the show |
| ☐ | Spend: `set_project()` reports it — headroom for live top-ups |
| ☐ | A first look prepared on design (skill `live-engine`), if asked |

Done when every row is ticked or the operator has waived it.

## During the show

Top-ups: `texture_flow(recipe=…, region=…)` at draft-or-final settings the
operator asks for → `check_frames` → it appears in the grid flagged NEW.
Binding a clip or an effect follows skill `live-engine`, on design only.
