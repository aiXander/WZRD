# Merged regions — one shader across several regions

**Status: spec, agreed direction 2026-10-26, not started.** Nothing here is
built. Current engine contracts this extends: [render-engine.md](../reference/render-engine.md)
§2.1 (selectors, `pick`, clip placement) and §2.2 (per-layer identity,
`source_box`, region scissor); UI rules: [operator-ui.md](../reference/operator-ui.md).

## The problem

Xander often puts the same shader on two regions in Prepare and wants them to
behave as **one surface**: one pattern, one clip, laid out across the combined
area. Today that is impossible, even with one binding over both regions,
because the compositor builds one pass per resolved layer and gives each pass
its **own** frame (`compositor.rs`, the `for (layer_index, slice)` loop in
plan build):

| Per-pass value | Today | Consequence |
|---|---|---|
| `layer_bbox()` / `layer_centroid()` | the region's own | patterns scale/centre per region (noiseDisplace, silk, embers, contours, scanwave) |
| `layer_seed()`, `layer_index()`/`layer_count()` | per region | cycles desync on purpose |
| `image_box` (C5 `placement: "region"`) | the region's own `source_box` | **one clip is fit separately into each region** — the same video twice |
| mask slice + scissor | the region's own | light stays inside the region — **keep this** |

Merging = replace the first three rows with one shared frame computed over
the member set; keep the fourth.

## Design

### M1 — engine: `merge` on the selector

```jsonc
"select": { "id": ["facade_left", "facade_right"], "merge": true }
"select": { "group": "facade", "merge": true }        // works with tag/group too
```

1. **`id` accepts a string or a list of strings** (`scene.rs::SelectorSpec`,
   serde one-or-many). Keeps the "exactly one of `all`/`id`/`tag`/`group`"
   rule. Ad-hoc lists exist because the UI can't create groups (nothing in
   `wzrd-app/src` calls `identity.setGroups`), and auto-creating groups would
   be a second write path (identity.json) for one gesture. Errors: empty
   list, unknown id in the list (name it, like the single-id error). Whatever
   re-serializes scenes (design-draft autosave, any serde round-trip) must
   write the shape it read — a single id stays a string.
2. **`"merge": true`** (default false) — the binding still builds **one pass
   per member** (same draw count as today), but every pass gets the merged
   frame:
   - `bbox_uv` = union of member `bbox_uv`.
   - `centroid_uv` = mask-area-weighted mean of member centroids. Needs a
     per-layer mask area: add `area` to `pack::LayerGeom`, measured from the
     mask bytes at pack load (they're dropped after GPU upload — `core.rs`
     clears `mask_atlas`), next to `support_uv`.
   - `image_box` = `contain_uv` into the **union of member `region_box_uv()`**
     (each member's `source_box`, else its `bbox`). One stream is already
     shared by all passes of a binding, so the clip now spans the regions and
     each region reveals its part.
   - `layer_seed` = `seed01` of the sorted member ids joined (stable across a
     layer rename, changes only when membership does); `layer_index = 0`,
     `layer_count = 1`.
   - **Unchanged per pass:** `slice` (own mask), scissor
     (`region_scissor(own support_uv)` — tighter than a union box, so no
     wasted fragments in the gap between far-apart regions), `z`.
3. **`merge` + `pick` is a load error** (picking one member of a merged
   surface is meaningless). `merge` on a one-member selection is legal and a
   no-op (keeps the UI simple).
4. Hot-reload / Carry / TAKE need nothing new: a `merge` flip is an ordinary
   design `scene.load` rebuild, and `Carry::Replace` keys clip streams by
   (binding id, file), which doesn't change.

**Rejected: a true single draw with a union mask.** Needs either a union
slice in the mask atlas (CPU bytes are gone after upload → GPU blit + atlas
re-create + rebind on every membership change) or a prelude ABI change
(`sample_mask` looping over N slices). It would also shade the whole union
box including the gap. The only visible difference from M1: effects that
read `sample_mask` at displaced uvs (`wobble`) would see across the seam.
Revisit only if that ever matters.

**Known limit (documented, not handled):** if two merged masks overlap, the
overlap is drawn twice — additive, so twice as bright. Segmented regions are
disjoint, so this needs a hand-drawn overlap to show.

### M2 — MCP mirror

`wzrd_mcp/engine_tools.py`: the binding JSON schema's `select` (≈ line 73)
gets `id: string | string[]` and `merge: boolean`; `_resolve_selector`
(≈ line 341) resolves lists; the scene-context summary (≈ line 407) shows
`merge`. One line in the authoring skill that names the pattern ("two regions
as one surface → `merge: true`").

### M3 — operator UI (Prepare)

All writes through `sceneCommit` as today; no new commit path.

- **`state/scene.ts`:** `selectorCovers` (the TS mirror of
  `resolve_selector`) and `targetLabel` handle list ids; the label reads
  "Facade L + Facade R" and appends "· merged" when `merge` is on. Update the
  `Binding` type.
- **Inspector "Lights up → Region"** (`LayerInspector.tsx`): the single
  region `<select>` becomes chips + "+ region" (×-remove on each chip; the
  last chip can't be removed). One region writes a plain string id.
- **Surface map:** with a layer selected and its selector in Region mode,
  **shift-click** a region toggles it in that layer's id list. Plain click
  keeps its current meaning.
- **Members switch** (shown whenever the selection has > 1 member):
  `Each separately | As one | One / 4 bars | One` — "As one" writes
  `merge: true` and clears `pick`; choosing a pick mode clears `merge`.
  (Today's "All at once" is renamed "Each separately".)
- **"Merge with ▾"** in the inspector header, listing the other layers that
  use the **same effect**: picking one folds it into this layer — this
  layer's id, params and modulations win; the selectors flatten to a list of
  the union of both layers' resolved region ids (a tag/group selector loses
  its dynamic membership — stated default); `merge: true`; the other layer is
  removed through the existing delete path, so ⌘Z brings it back (and the
  Members switch splits this one). Chosen over list multi-select because the
  store holds one `selectedBindingId` and every panel assumes it.
- **Clip drops** (`state/clips.ts`): today a drop on a region finds the clip
  binding by `b.select?.id === region`. Make it "an image binding whose
  selector covers this region"; if that binding is merged, the drop replaces
  its clip (the new clip spans the merged surface).

### M4 — tests and gates

- `scene.rs`: list id parses; single id round-trips as a string; empty list,
  unknown id in list, `merge` + `pick` → errors.
- `compositor.rs`: a merged two-member binding builds two passes with equal
  `bbox_uv`/`centroid_uv`/`layer_seed`/`image_box` (the union values), own
  `slice` and own scissor; unmerged stays as today.
- An offscreen render check: a merged pair with `videoClip` shows one clip
  placed in the union box (each region shows its part), not two copies.
- TS: `selectorCovers` / `targetLabel` on list ids.
- Perf gate: add a merged binding to a bench scenario in
  `render-core/perf/scenarios/`; the cost should be flat (same pass count,
  same scissors) — expect PASS without a baseline update.

## Order

M1 → M2 and M3 in parallel (M3 can stub against the engine's error reply
until M1 lands) → M4 alongside each. On landing: §2.1/§2.2 of
render-engine.md gain the `merge`/list-id contract, operator-ui.md the
Members/Merge-with/shift-click rules; this file drains and retires.

## Open questions

- **Rendering a clip *for* a merged surface.** The union box isn't at a Nano
  Banana native aspect, so a clip rendered for one member is letterboxed
  (contain) in the union. A Studio render target "merged surface" (crop the
  surface image to the padded union box) would fix it — out of scope until
  it's felt on a real show.
