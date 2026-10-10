# MIDI controller — setup routine, mapping profiles, robust resolution

> **Status: agreed spec, nothing built yet** (Xander, 2026-10-10). Canonical
> plan for hardware-controller support; [vj-toolbox.md](vj-toolbox.md) phase 6
> and roadmap §5.12 point here. App-only work (`wzrd-app/`) — no engine or
> MCP changes. Current UI rules it must obey:
> [../reference/operator-ui.md](../reference/operator-ui.md).
>
> **Schedule:** owned by the gig plan
> ([gig-2026-11-live-video.md](gig-2026-11-live-video.md)): optional in gig
> week 4 in the app lane after the clip grid (package A2), first to cut;
> otherwise post-gig. It uses two gig pieces: the per-machine
> `settings.json` from package A1 and the `library/` folder (contract C6).

## What Xander wants

1. Plug a MIDI controller into the Mac (today: an Akai "mini" — APC mini mk2
   or MPK mini mk3; the design must not care which).
2. A setup routine that walks through every button / knob / fader and binds
   each to a control in the UI.
3. Save that as a file that **auto-loads on project start** and is
   **reusable across projects**.
4. **Robust to UI change:** when a later UI renames or removes a control, the
   mapper warns (visibly, so it can be fixed) and keeps every other mapping
   working. It never crashes, never silently drops a mapping.

## Decisions

| # | Decision | Why |
|---|---|---|
| M1 | **MIDI input in Rust** (`midir` in `src-tauri/`), forwarded to the webview as a Tauri event `midi:message {port, bytes, t}`. The mapper itself runs in the webview | WKWebView has no Web MIDI. Targets are operator actions, which live in the webview |
| M2 | **Every MIDI write goes through `src/state/actions.ts`** — the mapper never calls RPC directly | Gets the three UI rules for free: hands follow the DESIGN/LIVE control leg, masters write both legs, live writes are throttled (25 ms leading + trailing) |
| M3 | A **control registry** (`src/state/controls.ts`): one typed table of every MIDI-addressable control with a **stable string id**, kind, range, `aliases`, and an `apply(value)` that calls `actions.ts`. Mappings reference registry ids, never component names, labels or DOM paths | This is the robustness seam: the UI can be rebuilt freely as long as the registry ids survive or are aliased |
| M4 | **Two files: a reusable profile + a thin project overlay** (format below). Profiles live in the repo's `library/midi/` (gig D16; `WZRD_MIDI_ROOT` overrides), git-versioned like effects and recipes; the project file only names its profile and adds project-specific mappings | Reuse across projects is the default; per-project tweaks don't fork the profile |
| M5 | **Prefer positional targets in profiles** ("mixer layer 3 opacity", "selected layer's dial 2", "look slot 5") over id targets ("binding `fabric_left` opacity"). Id targets are allowed but go in the project overlay | Positional targets mean something in every project; ids only exist in one |
| M6 | **Unresolvable mappings are kept, flagged and inert** — never deleted, never thrown. One toast per load ("3 MIDI mappings need attention"), an amber MIDI lamp in the TopBar health row, and a fix-it list in the MIDI panel | Requirement 4 |
| M7 | **Soft takeover** (pickup) is the default for absolute knobs/faders bound to a value that already exists: the hardware takes over only once it crosses the current value; the panel shows the gap | Design spec §4: "a knob never jumps under my finger". Matters doubly after TAKE copies design's values live |
| M8 | The mapper's MIDI handler **never throws**: each message's resolve + apply is wrapped; failures are logged (rate-limited) and counted on the control's row | A bad mapping mid-set must not stop the other forty |

## Control registry (`src/state/controls.ts`)

```ts
type ControlKind = 'continuous' | 'trigger' | 'momentary' | 'toggle';
interface ControlDef {
  id: string;            // stable forever, e.g. 'master.brightness'
  kind: ControlKind;
  label: string;         // shown in the picker; free to change
  group: 'Masters' | 'Deck' | 'Mixer' | 'Selected layer' | 'Looks' | 'Tempo' | 'FX' | 'Project';
  range?: [number, number];
  aliases?: string[];    // old ids that resolve here (renames)
  get?(): number | boolean | null;   // current value — needed for soft takeover + LED feedback
  apply(v: number | boolean): void;  // calls actions.ts only
}
```

Initial set — everything the keyboard shortcuts (`state/shortcuts.ts`) and
the Perform room already do, so MIDI never has a capability the UI lacks:

| Group | Ids (examples) | Kind |
|---|---|---|
| Masters | `master.brightness`, `master.speed`, `master.saturation`, `master.audioListen`, `deck.fade` (crossfade time) | continuous |
| Deck | `deck.take`, `deck.cut`, `deck.pull`, `deck.hands.toggle`, `deck.quantize.toggle` | trigger |
| FX | `fx.blackout` (toggle), `fx.flash` (momentary), `fx.freeze` (toggle) | — |
| Tempo | `tempo.tap`, `tempo.barline` | trigger |
| Mixer (positional) | `mixer.<1–8>.opacity` (continuous), `mixer.<1–8>.mute` / `.solo` (toggle), `mixer.<1–8>.select` | — |
| Selected layer | `selected.dial.<1–8>` (the selected layer's Nth scalar dial, in descriptor order), `selected.next` / `.prev`, `selected.mute`, `selected.solo` | — |
| Looks | `look.<1–9>.cue`, `look.<1–9>.go`, `autopilot.toggle` | trigger / toggle |
| Project (overlay only) | `binding:<id>.opacity`, `param:<binding id>/<param>` — generated from the current scene, not hand-listed | continuous |

If the clip grid (gig package A2) has landed, the initial set also has
**Grid (positional)**: `grid.<row 1–8>.<slot 1–8>` (trigger — click that
slot) and `grid.<row>.new` (bind the row's newest clip). Later controls
register here as they ship (transition picker, post chain…) — adding to the
registry is the whole job of "make X MIDI-mappable".

**The keyboard shortcuts may later move onto the same registry** (one table
for keys and MIDI) — optional, not part of this plan.

## File formats

**Profile** — `library/midi/<name>.midi.json` (reusable):

```jsonc
{
  "schema": 1,
  "name": "apc-mini-vj",
  "device": { "match": "APC mini" },          // substring of the MIDI port name
  "controls": [                                // the hardware inventory, from the walkthrough
    { "id": "fader1", "label": "Fader 1", "msg": "cc",   "channel": 0, "number": 48, "type": "absolute" },
    { "id": "pad11",  "label": "Pad 1·1", "msg": "note", "channel": 0, "number": 0,  "type": "button" },
    { "id": "enc1",   "label": "Knob 1",  "msg": "cc",   "channel": 0, "number": 70, "type": "relative-2c" }
  ],
  "mappings": [
    { "control": "fader1", "target": "mixer.1.opacity", "takeover": "pickup" },
    { "control": "fader9", "target": "master.brightness", "range": [0, 1], "curve": "lin" },
    { "control": "pad11",  "target": "look.1.go" },
    { "control": "pad12",  "target": "fx.flash", "behavior": "momentary" }
  ]
}
```

**Project overlay** — `<scene_dir>/midi.json` (app-owned, like `looks/`):

```jsonc
{ "schema": 1, "profile": "apc-mini-vj",
  "mappings": [ { "control": "enc1", "target": "param:fabric_left/speed" } ] }
```

Overlay mappings win per hardware control. Hardware is described once (in
the profile's `controls`); mappings reference control ids, so re-learning a
control's CC number never touches the mappings.

**Per-mapping options:** `range` (sub-range of the target), `invert`,
`curve` (`lin` | `log`), `takeover` (`pickup` default | `jump`),
`behavior` for buttons (`trigger` | `momentary` | `toggle`), `sensitivity`
for relative encoders, `threshold` (default 64) when a knob drives a trigger.

**Round-trip rule:** unknown fields and unknown mapping options are
preserved on save (a newer app's file opened in an older app loses nothing).
A file with a higher `schema` than the app knows loads what it understands
and warns.

## Resolution and robustness (M3, M6, M8)

On load, on project switch, and whenever the scene or catalog changes, every
mapping is resolved to one of:

| State | Meaning | What happens |
|---|---|---|
| `ok` | target id (or one of its `aliases`) exists, kinds compatible | active |
| `migrated` | resolved through an alias | active; next save rewrites the new id; info line in the panel |
| `unknown-target` | id not in the registry (renamed without alias, or removed) | inert, kept, warning |
| `stale` | project target whose binding/param no longer exists in the scene | inert, kept, warning — comes back to life if the binding returns |
| `incompatible` | control/target kinds can't combine (below) | inert, kept, warning |
| `no-device` | profile's device not connected | inert; TopBar lamp grey, not amber |

**Kind compatibility:** absolute/relative → continuous ✓; button →
trigger/momentary/toggle ✓; absolute knob → trigger ✓ via `threshold`;
button → continuous ✓ only with an explicit `range` (press = max, release =
min, or toggle between them); anything else = `incompatible`.

**Guards in the code, not just the docs:**
- A unit test asserts every id in the shipped starter profiles resolves
  `ok` — renaming a registry id without adding an alias fails the test.
- Malformed JSON: profile not loaded, warning toast, file left untouched (a
  write never overwrites a file that failed to parse — save to
  `<name>.midi.json.bak-<ts>` instead).
- Device unplugged mid-show: inert, toast, auto-reconnect by port-name match
  when it returns. Nothing else changes.

## The setup routine (UI)

A **MIDI panel** (TopBar MIDI lamp → click; also from Debug ⚙). Five steps:

1. **Connect** — list MIDI input ports (`midi_list_ports` Tauri command), pick
   one; a live monitor shows raw incoming messages. If a starter profile's
   `device.match` fits, offer it ("APC mini detected — start from the APC
   mini template?").
2. **Inventory: "touch every control"** — move every knob and fader, press
   every button once. Each new (msg, channel, number) becomes a row; type is
   inferred (CC sweeping many values → `absolute`; CC only 0/127 or note
   on/off → `button`; values clustered at 1/127 or 63/65 → relative
   encoder). Rows can be renamed ("Fader 1") and the type corrected. Done
   when Xander says so; controls can be added later.
3. **Assign walkthrough** — iterate the inventory one control at a time
   ("Fader 1 — what should this do?"), with skip / back / clear:
   - **Click-to-assign:** the app enters target-pick mode; every mappable
     UI element is outlined, click one to bind it. Elements declare their
     registry id via a `data-control="master.brightness"` attribute, added
     once in the shared primitives (`ui/Fader.tsx`, `ui/primitives.tsx`) and
     passed by each call site — this is the only UI-wide change.
   - **Or pick from a list:** searchable, grouped by registry `group`.
   - The kind check runs at assign time (incompatible = refused with the
     reason, not saved as broken).
   - **Reverse learn, any time outside the routine:** ⌥-click any mappable
     control → "MIDI learn" → move hardware. Same as Resolume's learn.
4. **Test** — the monitor shows control → target → value per message, the
   soft-takeover gap ("fader at 0.20, target at 0.65 — move up to pick up"),
   and per-row error counts.
5. **Save** — name the profile; checkboxes "use in this project" (writes the
   overlay) and "default for this device on this machine".

The panel's second tab is the **health list**: every mapping with its
resolution state; non-`ok` rows have a "re-pick target" button that drops
into click-to-assign for that one control.

## Auto-load order (project start and every project switch)

1. The project's `midi.json` → its named profile (+ overlay).
2. Else the machine default for a connected device (stored in the app's
   per-machine `settings.json`, next to the recent-projects list from gig
   package A1).
3. Else, if a controller is connected: one toast "MIDI controller detected —
   no mapping (open MIDI setup)". Nothing else.

## LED feedback (phase 2)

Controllers with addressable LEDs (APC mini mk2: RGB pads via note-on
velocity; MPK mini pads: not addressable) can show state: mute/solo lit,
active look, TAKE armed, deck colour (cyan DESIGN / red LIVE, matching the
room tint). Profile gains `"feedback": {"<control>": {"on": <vel>, "off":
<vel>}}`; values come from each registry control's `get()`. Output port via
`midir` too. Kept out of phase 1 because it's device-specific.

## Build order

1. **Rust bridge** — `midir` input in `src-tauri/`, port list + hot-plug,
   `midi:message` event, Tauri commands for profile/overlay file I/O
   (mirror `looks_list`/`look_read`/`look_write` in `src-tauri/src/rpc.rs`).
   Browser-bridge mode: the dev harness in Chrome can use Web MIDI behind the
   same `src/api/transport.ts` split, so scripted UI tests can drive MIDI.
2. **Registry + resolver + mapper** — `controls.ts`, pure resolver (no React)
   with **vitest** unit tests (the app has no test runner yet; add vitest,
   scoped to `src/state/`): aliases, unknown, stale, incompatible,
   round-trip preservation, malformed file.
3. **Setup routine + health list + TopBar lamp + ⌥-click learn.**
4. **Starter profiles** for APC mini mk2 and MPK mini mk3 (inventory +
   sensible default mappings), committed straight into `library/midi/`.
5. **Phase 2:** LED feedback.

**Gates:** map a full controller from scratch in < 10 min; restart the app
→ the mapping is live with no clicks; rename one registry id with an alias →
`migrated`, without → exactly that row turns amber and everything else still
plays; unplug/replug mid-set → recovers with no restart; a test sequence
through macOS's IAC virtual MIDI port (sent by a small `mido` script) drives
masters, TAKE and a look recall end to end.

## Non-goals

- **MIDI as a scene modulation source** (a `midi.cc` / `midi.note` driver in
  `scene.json`, the "rock wakes up on a MIDI note" target scene in reference
  render-engine.md) — engine work, separate; this plan is operator control
  only.
- MIDI clock / tempo sync (live BPM tracking is a rejected approach,
  reference §7).
- OSC control surfaces (TouchOSC etc.) — would reuse the registry later.
- Multiple controllers at once — v1 supports one active profile; the format
  doesn't block more (`device.match` per profile).
