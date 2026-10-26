---
name: run-sheet
description: "Venue day, once the operator has set the show up in the app: read the show, build a clip library per region from style recipes, run the pre-show checklist. Use when preparing a show."
metadata:
  kind: workflow
---

# On-site run sheet

From a set-up show to one the operator can play: each region with a row of
passing clips, the show checklist ticked. Work through the steps in order;
each ends on a check you can see.

The operator sets the show up and plays it. You prepare the content: you
render and check clips, bind on the design deck only when asked, and taking
anything live is theirs.

## 1. The show

The operator runs the app's Show setup (project menu → New show…): show +
canvas, lit/ambient photos, the rectified surface, the segmentation map, the
regions and the layer pack. Agree the region list with them first — every
surface the show treats separately (each fabric panel, the ceiling, a wall
section, a pillar) with a short id they'll say out loud: `fabric_left`,
`ceiling`, `wall_back`.

Then `show_info(project="<show>")`: canvas, regions (ids, labels, boxes),
pinned assets. **From here on, pass `project="<show>"` on every call** that
takes it. Nothing is remembered between calls; a call without it works on
whatever show the app has open. Layers the operator will name as a set
("the fabrics") get a group: `set_groups` (skill `live-engine`).

Copy the clip effects into the show so the clips can be played with:
`cp -R library/effects/* <project_dir>/effects/` (repo root; `project_dir`
is in `show_info`). No shell → ask the operator to run it.

Check: one region per agreed surface, each with its id.

## 2. The clip library

Load skill `textureflow` (settings, and the style recipe library).

1. `list_recipes()`; `view_media` the previews of the likely ones. If the
   operator has styles in the Studio tray, `studio_selection()` gives them.
   Agree with the operator which
   recipes go on which regions — one recipe set across regions reads as one
   show; a contrasting recipe marks a region out.
2. **Drafts first**, one per region × recipe (a few seeds each):
   `texture_flow(recipe=<name>, region=<id>, project=<show>,
   output_width=…, output_height=…, draft=true, wait=false)` with the
   output size = the region box's own pixel size (`source_box` in
   `show_info`) and every
   other arg as the final should have it — a draft is the final's first
   pass. If the tool has no `draft` arg yet (production not on it), use
   `use_upscale=false, upscale_esrgan=false` instead. Modal renders 3 at a
   time (more queue); ≤ 6 jobs in flight; poll `jobs()`.
3. `check_frames(<clip mp4>)` on every result, and `view_media`. A `flash`
   fail never goes on the wall — drop it. `haze` warns: re-render darker, or
   plan a `lumaKey` effect on that region.
4. Show the operator a contact sheet per region; finals for what they pick
   = **finishes, not re-renders**: `texture_flow(finish_from=<the draft's
   request_id>, wait=false)`, nothing else (or the operator clicks Upscale
   on the draft in Studio). Drafts live 24 h. Without `draft`, finals are
   re-renders without the two upscale args. Variety comes from more drafts
   (`seed`, `mapping_mode`, `motion_scale`). Aim for 3–5 passing clips per region.
5. A new look the operator loves → `save_recipe(name, from_clip=<clip>,
   when_to_use=<mood, motion, which surfaces it reads on>)`, on their word.

Every render lands in the media library (`~/WZRD/media`: the folder open in
Studio, else `inbox/`) with its HAP clip beside it, recorded with the show
and its region (`project=` + `region=`), so the app files it for this show.
Watch the `spend` line in each reply (today's spend, all shows) and stop at
the 80 % warning to ask.

The operator files clips into the show's decks in Studio (`D` / "+ Deck")
and takes them live.

Check: every region has ≥ 3 clips that passed `check_frames`.

## 3. Show checklist

Run through it with the operator before doors and report each line:

| | Check |
|---|---|
| ☐ | Alignment done in the app's Align room (operator, at the venue) |
| ☐ | Every clip in the decks passed `check_frames` — no flash fails |
| ☐ | Audio server running and the guitar moves `audio.*` (operator) |
| ☐ | Spend: `jobs()` reports today's — headroom for live top-ups |
| ☐ | A first look prepared on design (skill `live-engine`), if asked |

Done when every row is ticked or the operator has waived it.

## During the show

Top-ups: `texture_flow(recipe=…, region=…)` at draft-or-final settings the
operator asks for → `check_frames` → it lands in the Studio target folder.
Binding a clip or an effect follows skill `live-engine`, on design only.
