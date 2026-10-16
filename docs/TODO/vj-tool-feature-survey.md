> Status: research survey 2026-10-12. Input for build specs, waiting for Xander's review. Nothing built.

# Steal the shaping layer, keep the surface

**Bottom line:** the established VJ tools put most of their live-play value in one layer that WZRD barely has yet: the layer that **shapes and routes control**. That means macro dials that drive several parameters at once, response curves on any modulation, tempo-synced LFO shapes, recall that leaves some channels alone, and an autopilot with randomness and per-item durations. WZRD already matches or beats them on the structural side: preview/program (its DESIGN | LIVE legs), crash-safe shader swaps, a baked warp, and an agent that authors but never performs. The best borrowings are therefore **small UI and mapping features on top of the engine that already exists**, not new engine subsystems. Two caveats run through the catalog. First, several popular features clash with things the engine docs have already ruled out: Ableton Link and tempo following, recording, and audio smoothing inside the engine. Those are tagged ⛔ below rather than quietly proposed again. Second, the Studio plan (Xander's first pass at borrowing: media library, Studio room, named clip decks) gets the deck model right. Resolume suggests four refinements to it, listed in the Top picks.

---

## How to read this

**Surveyed:** Resolume Arena/Avenue 7.x (through 7.28, Sep 2026) and TouchDesigner 2025 in depth. MadMapper 6, HeavyM, Millumin 5, VDMX, Synesthesia, Smode/Notch/disguise where they add something. Plus practitioner sources (VJ blogs, the TouchDesigner forum, Resolume forum summaries). Reddit was unreachable.

**WZRD status tags**

| Tag | Meaning |
|---|---|
| ✅ | have it today |
| 🟡 | partly there |
| 📋 | already planned (the plan is named in words) |
| 🟦 | in the **Studio plan** (media library + Studio room + named clip decks; agreed 2026-10-15, building before the gig) |
| ❌ | gap: nobody has planned it |
| ⛔ | conflicts with something the engine docs list as **rejected** (re-open only deliberately) |
| (?) | I'm not sure of WZRD's current state |

**Relevance** is High / Med / Low for *this* instrument: Mac-first, one projector, additive (only lit regions get light), an agent authors in DESIGN while Xander performs LIVE.

**Plans referenced:** the **gig plan** (six-week build for the ~21 Nov live-guitar lie-down concert), the **toolbox plan** (post-gig order: effects library → effect chains/feedback/post → TAKE transitions → setup room → MIDI), the **engine roadmap** (engine designs, not scheduled), the **MIDI spec** (on hold since today), the **perf-monitor plan**.

---

## Top picks: 14 borrowings worth a build spec

1. **Macro dials, i.e. "one dial, many parameters"** (Resolume Dashboard ([doc](https://resolume.com/support/en/dashboard))). You drag any parameter onto a dial. Each link has its own in/out range, invert, and *which part of the dial's travel* drives it. Toggles and dropdowns can be linked too, so one dial can step through effect variants. **Why:** commercial clip packs ship this way because it makes complex content playable with one hand. It also lets a small MIDI controller cover a big show. **WZRD:** ❌. Today every dial maps to exactly one effect parameter.

2. **Response curves on any modulation** (Resolume envelopes ([doc](https://resolume.com/support/en/envelopes))). A small curve editor sits between the source (beat, kick, MIDI) and the parameter. It has ease/hold/bounce presets, and its x-axis is "phase", not seconds. **Why:** this separates "what drives it" from "how it responds". Kick → *punchy* flash vs kick → *swelling* glow becomes one curve change. **WZRD:** ❌. The `range: [rest, peak]` mapping is strictly linear. **Watch out:** the engine docs reject *audio smoothing / attack-release in the engine*, because that belongs in the audio server. A curve on the mapped range is a mapping shape, not signal conditioning, but say so explicitly in the spec.

3. **Tempo-synced LFO shapes with min/max on any dial** (Resolume timeline/BPM modes, VDMX LFO, TouchDesigner Beat CHOP). Practitioners' most consistent advice: hand knobs are "too intense" for beat-matching, and "far better to have a tempo synced LFO" ([Meissen](https://matthiasmeissen.substack.com/p/twid-15-vj-system-in-touchdesigner)). **WZRD:** ✅ (landed 2026-10-14). Clock drivers take a `shape` (`pingpong` default for new UI routings, `sine`, `saw`, `ramp_down`, `square`, smooth `random`), and the modulation popover picks shape + cycle (¼–16 bars tempo-locked, or 1–64 s free) with a live scope — [operator-ui.md](../reference/operator-ui.md) "Modulation", [render-engine.md](../reference/render-engine.md) §2.1 "Clock shapes". Still open: a `square` duty cycle and response curves.

4. **Audio drives *speed*, not value** (Resolume FFT "+/−" mode ([doc](https://resolume.com/support/en/parameter-animation)), Synesthesia loudness-accelerated time ([doc](https://app.synesthesia.live/docs/ssf/audio_uniforms.html))). Loud music makes a drift *move faster* instead of making a brightness jump. **Why:** this is the cure for the design spec's fear that the crowd will "feel a metronome". Motion builds with the music without flashing. **WZRD:** ❌. Since time in WZRD already integrates (the speed master bends time without jumping), a per-binding "audio bends this binding's clock" fits the existing model.

5. **Look recall that leaves chosen things alone** (Synesthesia presets with per-channel Scene/Meta/Media recall ([doc](https://app.synesthesia.live/docs/manual/presets.html)), Resolume "Ignore Column Trigger" + "Lock Content" ([doc](https://resolume.com/support/en/layers)), CuryCue "tracking" cues that store only what changed ([repo](https://github.com/netzzy/curycue))). **WZRD:** ❌. Recalling a look replaces the whole design scene. Proposed shape: per-look "recall effects / clips / both", plus a per-layer **hold** pin, so the fabric clip survives a look change.

6. **Autopilot that isn't a metronome** (Resolume autopilot ([doc](https://resolume.com/support/en/autopilot)), Synesthesia playlists). Add per-look duration in bars, a **"Bag"** random mode (each look once, then reshuffle, never repeating back to back), **countdown pies on thumbnails** showing what fires next, and later **advance on a musical drop** instead of a timer. **WZRD:** 🟡. The autopilot walks the looks bank *in order*, one look every N bars, and dies with the app window. An engine-side playlist is 📋 in the toolbox plan.

7. **Random within ranges, plus Undo** (Synesthesia bank/group buttons ([doc](https://app.synesthesia.live/docs/manual/controls.html))). One button randomises every dial of a layer inside its declared range, and Undo reverts it. **Why:** this is the cheapest "surprise me" (design spec: "surprise me, sometimes"), and the descriptor ranges already make it safe. **WZRD:** ❌.

8. **Studio-plan decks: four refinements from Resolume** (Xander 2026-10-15: none of the four in the Studio MVP; revisit after the gig) ([decks](https://resolume.com/support/en/decks), [clips](https://resolume.com/support/en/clips)). The Studio plan's "decks = named pages of the clip grid, switching never changes what's playing" is exactly Resolume's model. Add: **(a) persistent clips**, which show on every deck (e.g. a base wash for one region); **(b) closed/hidden decks + a jump-to-deck menu** (Resolume added this in 7.18 because decks overflow); **(c) import a deck from another show** (Resolume's "master composition" workflow: keep a library show of pre-built decks and pull one in mid-show); **(d) colour per deck/slot** for low-light reading.

9. **Effects in clip-grid slots, and a mix fader per effect** (Resolume "effect clips" + per-effect opacity/blend ([doc](https://resolume.com/support/en/effects)), SonomikaTD effect stacks per cell ([asset](https://derivative.ca/community-post/asset/sonomikatd-touchdesigner-vj-system/75106))). The toolbox plan's open question, "should the clip grid also hold effect slots?", gets a clear *yes* from Resolume and SonomikaTD. When the planned effect chains land, give every stage a **mix** (dry/wet) fader the compositor applies, so no shader has to implement its own mix.

10. **Chases across regions** (TouchDesigner Beat CHOP "Multiples + Shift Step" phase-staggered channels ([doc](https://docs.derivative.ca/Beat_CHOP)), VDMX multi-track step sequencer of values *and colours* ([doc](https://vdmx.vidvox.net/tutorials/overview-of-available-data-sources-in-vdmx))). One beat fans out into N regions, each a fraction of a cycle later, or a step grid picks which region group lights on each bar. **Why:** this is the lighting-desk "chase", and additive mapping is basically lighting. **WZRD:** 🟡. `pick` selectors (one random member per cycle) and per-layer identity/seed in WGSL exist, but there is no operator-level "chase across this group" control.

11. **Align room hardening** (Resolume Advanced Output ([modifiers](https://resolume.com/support/en/modifiers)), TouchDesigner Stoner ([doc](https://docs.derivative.ca/Palette:stoner)) and CamSchnappr TouchOSC remote ([doc](https://docs.derivative.ca/Palette:camSchnappr))). Add **undo/redo**, select a whole row/column of handles, lock handles, a numbered-grid test pattern, and a **phone remote to nudge corners while standing at the wall**. **Why:** the design spec's "calibration recovery in under a minute". **WZRD:** 🟡. 4-corner + n-point warp, 1/10 px arrow nudge, whole-quad drag and grid/border/corners test patterns all exist. There is no undo (?), and no remote.

12. **Global "meta" controls** (Synesthesia Meta bank ([doc](https://app.synesthesia.live/docs/manual/controls.html))). WZRD's masters already cover Brightness, Saturation, Speed and "how much the surface listens", and are already kept out of looks (Synesthesia makes the same call). Worth adding: **Audio Speed** (how fast audio-driven motion runs), **Limit Colours** (remap the whole output to a two-colour palette, strong on coloured surfaces), **Mirror**. Skip **Invert**: in additive projection it lights the dark.

13. **Outline / edge effects generated from region shapes** (HeavyM Border / Line / Snake effects that animate a shape's outline ([1.7 notes](https://www.heavym.net/?p=3135))). **Why:** in additive mapping a lit outline merges with the physical edge. It's the most "surface is the instrument" effect family there is. **WZRD:** 🟡. The seed library has `edgeGlow` over clips, but no per-region distance-to-edge accessor for shaders (?).

14. **Projector-view structured-light scan** (MadMapper Space Scanner ([guide](https://madmapper.com/files/13-Space%20Scanner%20Guide-1.pdf))). The projector scans the scene pixel by pixel through a camera, so you get the surface *as the projector sees it*. **Why:** segmentation masks would be born in projector space, so most of the alignment step disappears. **WZRD:** ❌. Camera auto-alignment is deferred in the engine roadmap, and the projector camera is parked in the gig plan. This is a big post-gig bet, not a quick win.

---

## Feature catalog by area

### Clip launching & decks

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Region × slot grid | [Resolume](https://resolume.com/support/en/clips) layers × columns | rows stack, columns are slots | ✅ rows are named regions | High |
| Click = safe | Resolume: name-handle selects, thumbnail fires; 7.22 split column select from trigger ([blog](https://resolume.com/blog/27125)) | prevents accidental fires | ✅ click binds on DESIGN only, T takes it live | — |
| Named decks (pages of clips) | [Resolume](https://resolume.com/support/en/decks) | switch page, playing clips untouched | 🟦 | High |
| Persistent clip on every deck | [Resolume](https://resolume.com/support/en/clips) | flagged clip appears in same slot on all decks | ❌ refine Studio plan | Med |
| Hidden decks + jump menu | Resolume 7.18 ([blog](https://resolume.com/blog/22892)) | close decks, list all in a menu | ❌ refine Studio plan | Med |
| Import a deck from another show | [Resolume](https://resolume.com/support/en/decks) "master composition" | double-click a deck in a library show | ❌ refine Studio plan | Med |
| Replace clip, keep its effect | [Resolume](https://resolume.com/support/en/clips), Millumin swap-in-place ([log](https://help.millumin.com/docs/general/change-log/)) | drop new file onto a clip | ✅ clicking a slot swaps the clip inside the region's existing effect | — |
| Trigger styles: toggle / piano (momentary) | [Resolume](https://resolume.com/support/en/clips) | piano plays only while held | ❌ | Med |
| "Play on selected region" target | Resolume "Active Layer" | one clip bank feeds whichever layer is selected | 🟡 drag to re-file a clip | Med |
| Per-slot beat snap | [Resolume](https://resolume.com/support/en/clips) | composition default, per-clip override | 🟡 global BAR/NOW on TAKE | Med |
| Lock content per layer | [Resolume](https://resolume.com/support/en/layers) | can't eject/replace, can still tweak | ❌ (see Top pick 5) | Med |
| Blank slot = clear region | Resolume 7.22 empty column stops all | deliberate "go dark" slot | ❌ | Med |
| Clip colours | [Resolume](https://resolume.com/support/en/clips), Millumin colour filter | pastel tag per clip/deck/layer | ❌ | Med |
| Multi-select + batch move | Resolume 7.22 | shift-select, drag several | ❌ | Low |
| Thumbnail from current frame | [Resolume](https://resolume.com/support/en/clips) | set thumb incl. effects | 🟡 sidecar thumbs; looks get a jpg | Low |
| Fader start / pick-up | [Resolume](https://resolume.com/support/en/clips) | clip restarts (or resumes) when faded up | ❌ | Low |

### Transport & tempo

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Tap / type / scroll BPM | [Resolume](https://resolume.com/support/en/bpm), HeavyM | — | ✅ TAP, double-click, scroll | — |
| Resync to bar one | Resolume, HeavyM | press on the downbeat | ✅ bar-line key | — |
| Hold-to-nudge tempo | [Resolume](https://resolume.com/support/en/bpm) | temporary speed-up/down while held | ❌ only correction tool without a clock | Med |
| Round tap to whole BPM | requested on Resolume forum ([t=21632](https://resolume.com/forum/viewtopic.php?t=21632)) | — | ❌ (trivial) | Low |
| Phrase indicator (16 beats) | [Resolume](https://resolume.com/support/en/bpm) | outline flashes every 16 beats | 🟡 beat lamps (?) | Low |
| Ableton Link / MIDI clock | Resolume, TD, HeavyM, MadMapper | follow an external tempo | ⛔ rejected for v1; live guitar has no DAW anyway | Low |
| Clip length in beats, ½× / 2× | [Resolume](https://resolume.com/support/en/parameter-animation) BPM Sync | loop stretches to N beats | ❌ clips play at a rate, not a beat length | Med |
| Clip playhead as a mod source | Resolume "Clip Position" | parameter follows the playhead | ❌ | Med |
| Phase-staggered beat channels | [TD Beat CHOP](https://docs.derivative.ca/Beat_CHOP) | N copies, each shifted 1/N cycle | 🟡 (Top pick 10) | High |
| Global pause / stop | Resolume 7.22 ([blog](https://resolume.com/blog/27125)) | pause all transports | 🟡 FREEZE (Z) | — |
| Elapsed / set clock panel | Resolume Clip Time panel ([doc](https://resolume.com/support/en/clip-time-panel)) | big clock, undockable | ❌ | Low |

### Transitions & crossfade

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Timed crossfade + time master + cut | Resolume layer transitions | 0–10 s duration | ✅ TAKE, FADE, CUT, bar-quantized | — |
| Transition modes (wipes, random) | [Resolume](https://resolume.com/support/en/layers); Random mode per trigger | pick a mode per layer/clip | 📋 TAKE transitions (engine roadmap + toolbox plan) | High |
| Transitions as shader files | Millumin ISF column transitions ([log](https://help.millumin.com/docs/general/change-log/)) | custom transition = shader | 📋 same plan, which this validates | — |
| Per-look transition override | Resolume 7.25 per-clip transitions ([blog](https://resolume.com/blog/34172)) | clip carries its own | 📋 same plan | Med |
| Random transition | Resolume | different mode each trigger | ❌ add to the plan | Low |
| Autopilot fires early so the fade lands on the line | Resolume 7.27 ([blog](https://resolume.com/blog/34561)) | — | ✅ recall 2 beats early, TAKE on the bar | — |
| Momentary "peek" | Resolume crossfader "Cut" (returns on release) ([doc](https://resolume.com/support/en/composition)) | hold to show B, release returns | ❌ hold-to-show design on the wall | Med |
| Hand-ridden crossfader (T-bar) | [Resolume](https://resolume.com/support/en/composition) A/B bus, curve | performer rides the mix | ❌ TAKE is timed only | Med |
| Crossfade position as mod source | Resolume "Crossfader Phase" | — | ❌ | Low |
| Video bumper | Resolume 7.22 | a clip plays during the transition | ❌ | Low |

### Scenes, cues, presets, looks

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Saved whole-show snapshots | Resolume columns, Synesthesia presets | — | ✅ looks bank, click = cue, double = GO | — |
| Slots on keys / pads | Synesthesia FavSlots | assign preset to slot, map it | ✅ ⇧1–9 cue, ⇧⌥ GO | — |
| Edit without touching output | Millumin "Silent Edition", HeavyM "edited but not played" ([HeavyM](https://help.heavym.net/hc/articles/9172061701138)) | — | ✅ the DESIGN leg (stronger) | — |
| Masters excluded from presets | [Synesthesia](https://app.synesthesia.live/docs/manual/controls.html) | recall never changes brightness | ✅ masters live outside the scene | — |
| Partial recall / hold layer | Synesthesia channels, Resolume ignore-column-trigger, CuryCue tracking | — | ❌ (Top pick 5) | High |
| Morph between looks | TDMorph ([asset](https://derivative.ca/community-post/asset/tdmorph/73268)) | interpolate *parameters* with a curve | 🟡 TAKE crossfades *images*, not params | Med |
| Randomise + undo | [Synesthesia](https://app.synesthesia.live/docs/manual/controls.html) | per bank/group | ❌ (Top pick 7) | High |
| Ordered cue list, GO + "standby next" | Smode ([doc](https://www.smode.io/doc/ref/animation/cue-list.htm)), CuryCue | pick next without playing it, GO fires | 🟡 DESIGN is the armed cue, TAKE is GO; no ordered list | Med |
| Effect presets (lightning menu) | HeavyM ([help](https://help.heavym.net/hc/sections/360004684840-Managing-Group-Effects)), [Resolume](https://resolume.com/support/en/effects) one-shot presets | effect turns on looking good; saved variants | ❌ | Med |
| Copy / paste effect stack | [Resolume](https://resolume.com/support/en/effects) | paste effects onto other layers | 🟡 tag/group selectors cover "all leaves" | Med |
| Slot remembers its last tweak | Millumin 5.14 Auto States | per-column state memory | ❌ looks reload as saved | Low |
| Setlist notes | Resolume Notes panel, disguise timeline notes | annotate the show | ❌ | Low |

### Parameters & modulation

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Effect declares its own typed dials | Resolume Wire, Notch exposed params ([manual](https://manual.notch.one/2026.1/en/docs/workflows/working-with-media-servers/exposed-parameters/)), Synesthesia scene controls | min/max/step/widget per input | ✅ descriptor-driven dials | — |
| Foldable dial groups | Resolume 7.21 ([blog](https://resolume.com/blog/24836)), Synesthesia | `group` per param → folding sections | ❌ cheap descriptor field | Med |
| Stable param id ≠ display name | [Notch](https://manual.notch.one/2026.1/en/docs/workflows/working-with-media-servers/exposed-parameters/) | maps survive renames | 🟡 params keyed by name (?) | Med |
| Reset / fine drag | Resolume right-click reset | — | ✅ double-click reset, ⇧-drag fine | — |
| Type a value (math allowed) | [Resolume](https://resolume.com/support/en/parameters) | `1920/3` | ❌ | Low |
| Modulation range (rest → peak) | Resolume Dashboard in/out | — | ✅ | — |
| Response curves on modulation | [Resolume envelopes](https://resolume.com/support/en/envelopes) | ease presets, x = phase | ❌ (Top pick 2) | High |
| LFO shapes, tempo-locked | Resolume, VDMX, TD | sine/tri/square/random, rate × | ✅ 2026-10-14 (Top pick 3) | High |
| Audio drives rate | [Resolume](https://resolume.com/support/en/parameter-animation), [Synesthesia](https://app.synesthesia.live/docs/ssf/audio_uniforms.html) | loudness speeds a clock | ❌ (Top pick 4) | High |
| Per-modulation gain + fall | Resolume FFT, [TD audioAnalysis](https://docs.derivative.ca/Palette:audioAnalysis) | boost; how fast it decays | 🟡 onset has `decay`; smoothing ⛔ in engine (audio-server job) | Med |
| Richer musical signals | [Synesthesia](https://app.synesthesia.live/docs/ssf/audio_uniforms.html): hits, presence, intensity, fade-in/out, toggle/random-on-beat | pre-normalised, zero setup | ❌ belongs in the audio server per engine rule | Med |
| Macro dials | [Resolume Dashboard](https://resolume.com/support/en/dashboard), VDMX Control Surface | one dial → many params | ❌ (Top pick 1) | High |
| Dial pick-up on assign | Resolume 7.23 ([blog](https://resolume.com/blog/30807)) | takes current value, no jump | 📋 MIDI spec (soft takeover) | — |
| Engine state as mod source | [VDMX](https://vdmx.vidvox.net/tutorials/overview-of-available-data-sources-in-vdmx) "everything is a source" | opacity, playhead, crossfade, autopilot step | ❌ | Med |
| Step sequencer (values, colours) | [VDMX](https://docs.vidvox.net/vdmx_plugins.html) | multi-track, per step | 🟡 `pick` selectors | Med |
| LFO start conditions | [Resolume](https://resolume.com/support/en/parameter-animation) | restart on load / clip / column trigger | ❌ global phase-locked clocks are a fine default | Low |
| Eyedropper | [Resolume](https://resolume.com/support/en/parameters) samples any pixel incl. output | — | ❌ sampling the *site photo* would be the WZRD twist | Med |
| Meta controls | [Synesthesia](https://app.synesthesia.live/docs/manual/controls.html) | reactivity, audio speed, limit colours, mirror | 🟡 (Top pick 12) | Med |

### Effects UX

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Preview effect before commit | [Resolume](https://resolume.com/support/en/effects) double-click → Preview Output | — | ✅ DESIGN leg + cue monitor | — |
| Library browser, drag onto layer | [Resolume](https://resolume.com/support/en/effects), favourites tab | — | 📋 toolbox plan (library panel) | High |
| Quick search to add anywhere | Resolume 7.19 ([blog](https://resolume.com/blog/23542)), Millumin Quick-Find | Spotlight-style | 📋 toolbox plan has search; make it global | Med |
| Ordered stack, bypass per effect | [Resolume](https://resolume.com/support/en/effects), SonomikaTD | drag to reorder | 📋 effect chains (toolbox plan) | High |
| Per-effect mix fader | [Resolume](https://resolume.com/support/en/effects) opacity + blend per effect | — | ❌ add to chains spec | Med |
| Trails / feedback / global post | Resolume, TD | — | 📋 toolbox plan (with "dark stays dark" clip) | High |
| Effects in grid slots | [Resolume](https://resolume.com/support/en/effects) effect clips | trigger like a clip | ❌ open question in toolbox plan, answer yes | Med |
| One toggle, good defaults | [HeavyM](https://help.heavym.net/hc/sections/360004684840-Managing-Group-Effects) per group | no parameter panel first | ✅ "Light it up" two-click layer | — |
| Fork a stock effect | Resolume Wire stock effects | open and edit | ✅ effects are plain WGSL in the project | — |
| Rename / recolour instances | [Resolume](https://resolume.com/support/en/effects) | same effect, different jobs | ❌ | Low |
| Render-scale / freeze heavy effect | [SonomikaTD](https://derivative.ca/community-post/asset/sonomikatd-touchdesigner-vj-system/75106) | perf escape hatch | ❌ | Med |
| Outline effects from shapes | HeavyM | — | 🟡 (Top pick 13) | High |

### Control mapping (MIDI / keyboard / remote)

All 📋 rows are in the **MIDI spec (on hold)**. The research mostly confirms it.

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Logical control layer | TD Device Mapper (`s1`, `b1`) ([doc](https://docs.derivative.ca/MIDI_Device_Mapper_Dialog)) | maps reference "slider 1", not CC 48 | 📋 hardware inventory + control ids | — |
| MIDI learn | [Resolume](https://resolume.com/support/en/midi-shortcuts); TD has none | click control, move hardware | 📋 ⌥-click learn | High |
| Target scope: by position / this item / selected | [Resolume](https://resolume.com/support/en/midi-shortcuts) | — | 📋 positional + id + selected-layer dials | — |
| Button modes toggle / piano / **range / value**; **velocity** | [Resolume](https://resolume.com/support/en/midi-shortcuts) | range = jump between two values | 🟡 spec has trigger/momentary/toggle; add value, range, velocity | Med |
| Relative encoders, "fake relative" | Resolume | absolute knob behaves relative | 🟡 spec has relative; add fake-relative | Low |
| Double-assigned shown red | [Resolume](https://resolume.com/support/en/midi-shortcuts) | — | ❌ add to spec | Med |
| Mapping survives updates | practitioner pain ([forum p=108782](https://resolume.com/forum/viewtopic.php?p=108782)) | — | 📋 aliases + "kept, flagged, inert" rule | High |
| LED feedback | Resolume five clip states | — | 📋 spec phase 2; must also update from UI/agent changes and show the leg colour | High |
| Starter profiles | practitioners: APC40 mk2, Launch Control XL 3, MIDI Fighter Twister ([LIME](https://limeartgroup.com/best-midi-controllers-of-vjs-visual-artists/)) | — | 🟡 spec plans APC mini mk2 + MPK mini; consider a Twister | Med |
| Editable keyboard map | [Resolume](https://resolume.com/support/en/keyboard-shortcuts) | presets, duplicates red | 🟡 fixed keys; spec's optional unification | Low |
| API + snapshot endpoint | Resolume 7.26 REST monitors snapshot ([blog](https://resolume.com/blog/34484)) | — | ✅ WS JSON-RPC + `preview.snapshot` | — |
| Phone / tablet remote | Resolume custom `index.html` remote ([doc](https://resolume.com/support/en/restapi)), CamSchnappr TouchOSC | — | 🟡 browser bridge exists but is dev-only | Med |
| OSC address per control (`?` query, `+`/`-` relative) | [Resolume](https://resolume.com/support/en/osc) | — | ❌ OSC surfaces are a non-goal in the MIDI spec | Low |

### Output, projection mapping, alignment

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Corner pin + inner warp grid | [Resolume](https://resolume.com/support/en/output-transformation), [TD Stoner](https://docs.derivative.ca/Palette:stoner), MadMapper | perspective base, mesh on top | ✅ same structure (validated) | — |
| Bake warp, drop the editor | [TD Stoner](https://docs.derivative.ca/Palette:stoner) | displacement map | ✅ baked offset LUT | — |
| Input side vs output side | Resolume input selection / output transformation, MadMapper | — | ✅ clip placement box (Prepare) vs Align room | — |
| Nudge 1 / 10 px, whole-quad drag | [Resolume](https://resolume.com/support/en/input-selection) | — | ✅ | — |
| Undo / redo in the warp | Resolume, Stoner | — | ❌ (?) | High |
| Row/column select, handle lock, zoom/pan | [Stoner](https://docs.derivative.ca/Palette:stoner) (Ctrl+r/c, `l`), Resolume space-drag | — | ❌ (?) | Med |
| Don't reset warp when grid changes | Kantan's documented trap ([doc](https://docs.derivative.ca/Palette:kantanMapper)) | — | ✅ handles are corrections on top (?) | — |
| Test-pattern sequence | [Jaze](https://www.baptistejaze.com/en/projector-calibration/projector-test-patterns): grid → crosshatch → numbered grid → greys | — | 🟡 grid / border / corners | Med |
| Panic: kill output | [Resolume](https://resolume.com/support/en/screens) Ctrl-Shift-D | — | ✅ B blackout | — |
| Projector shutter | Millumin PJLink | hardware blackout | ❌ | Low |
| Per-output colour trims | [Resolume](https://resolume.com/support/en/screens) RGB/contrast per screen | white-balance projector vs surface | 🟡 brightness/saturation masters only | Med |
| Hand-fix a mask (feathered bezier) | [MadMapper](https://madmapper.com/madmapper/features), Kantan soft edges | fix AI segmentation by hand | ❌ gig plan: "only if needed" | Med |
| Never light outside regions | Resolume output masks (crop by default) | — | ✅ by construction (additive) | — |
| Projector-view scan | [MadMapper Space Scanner](https://madmapper.com/files/13-Space%20Scanner%20Guide-1.pdf) | — | ❌ (Top pick 14) | High |
| Display pick, cursor hide, no sleep | [TD Window COMP](https://docs.derivative.ca/Window_COMP) | — | ✅ OUTPUT pill (display pick, cursor hide, keep awake, identify — 2026-10-15) | — |

### Monitoring, safety, show reliability

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Preview vs program | Resolume Preview Output | — | ✅ PREVIEW/PROGRAM frame, cue monitor | — |
| Unmistakable live mode | practitioners: preview/live confusion is a top beginner trap ([vjacademy](https://vjacademy.info/resolume-arena-guide)) | — | ✅ whole room tints red | — |
| Show lock (block accidents) | Resolume forum wish ([t=26851](https://resolume.com/forum/viewtopic.php?t=26851)), CuryCue show mode, TD locked Perform | — | 🟡 LIVE makes structure read-only | Med |
| Bad shader never reaches the wall | TD Engine COMP reload-on-crash ([doc](https://docs.derivative.ca/Engine_COMP)) | — | ✅ compile probe + last-good; frame-time watchdog on hold | — |
| Inactive scenes cost nothing | TD pain: 4–5 audio scenes kill fps ([forum](https://forum.derivative.ca/t/reducing-complexity-cook-load-for-live-setup/273176)) | — | ✅ demand-gated design leg; keep measuring | — |
| Per-layer cost overlay | [TD Probe](https://docs.derivative.ca/Palette:probe) | own vs children cost, 10-frame history | 📋 perf-monitor plan (graphed panel, still design) | Med |
| Undo for edits, not triggers | [Resolume](https://resolume.com/support/en/undo) | performance acts aren't edits | ❌ for design edits (?) | Med |
| Safety look on one key | practitioners: "something on screen beats nothing" ([VIDVOX](https://vdmx.vidvox.net/tutorials/vj-travel-kit-whats-in-our-bag)) | — | ❌ pin a look as panic fallback | Med |
| Tape-loop buffer of the last bars | Meissen's self-built TD tool | covers dead time | 🟡 FREEZE holds one frame; ⛔-adjacent (no frame tap rule) | Low |
| Pre-flight check | practitioners: "freeze the rig" | outputs, controllers, audio | 🟡 health lamps | Med |
| Hide accident-prone controls | Resolume 7.22 hide speed sliders | — | ❌ | Low |

### Workspace, layout, library

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Media browser, favourites, thumbnails | [Resolume](https://resolume.com/support/en/clips) | — | 🟦 Studio room | High |
| Folders synced from Finder | Millumin 5.12 | — | 🟦 library is folders on disk | — |
| Moved media still plays | SonomikaTD file relinking | — | 🟦 import = copy, so nothing to relink | — |
| Filter by colour / rating | Millumin Quick-Find | — | 🟦 ratings; colour tags ❌ | Low |
| Save a version of the show | — | — | 🟦 "Save as" fork | — |
| Dockable panels, layout presets | [Resolume](https://resolume.com/support/en/layouts) | — | ✅ partial 2026-10-14: resizable + swappable panels per room, locked by default, saved per project (no named presets) | Low |
| Auto-generated control panel | TD Parameter COMP ([doc](https://docs.derivative.ca/Parameter_COMP)) | panel from a param list, "can't break bindings" lock | ✅ dials from descriptors | — |

### Automation & autopilot

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| Chain of looks every N bars | Resolume, HeavyM (tempo-stepped) | — | ✅ in order, app-bound | — |
| Survives app close | — | — | 📋 engine playlist (toolbox plan) | High |
| Random modes Other / Any / Bag | [Resolume](https://resolume.com/support/en/autopilot) | Bag = all once, reshuffle | ❌ | High |
| Per-item duration, loop count | Resolume column duration, [Synesthesia](https://app.synesthesia.live/docs/manual/presets.html) playlist | — | ❌ global N bars only | High |
| Countdown on thumbnails | Resolume 7.22 | shows what fires next | ❌ | Med |
| Advance on drop / breakdown | [Synesthesia](https://app.synesthesia.live/docs/manual/presets.html) Auto-Transition | — | ❌ needs an energy signal from the audio server | Med |
| Manual trigger wins over autopilot | [Resolume](https://resolume.com/support/en/autopilot) priority rule | — | 🟡 (?) | Med |
| Follow actions (when X ends, go to Y) | MadMapper 6 markers ([CDM](https://cdm.link/madmapper-6-brings-timelines-markers-clips-more/)), Resolume clip actions | — | ❌ | Low |
| Bad entry skipped and logged | design spec | — | 📋 engine playlist | High |

### Remote, API, agent

| Feature | Who | How it works | WZRD | Rel. |
|---|---|---|---|---|
| MCP server for *building*, not playing | **Resolume 7.26** Arena + Wire MCP: "mostly used for building... not so much during live performances" ([blog](https://resolume.com/blog/34484)) | Claude/Codex build compositions | ✅ same split; WZRD adds the probe-gated design leg | — |
| Agent tools for look banks / morphs | third-party TouchDesigner MCP ([Glama](https://glama.ai/mcp/servers/lucasmaher-hash/touch-designer-mcp/tools/create_preset_morph)) | `create_look_bank`, `create_preset_morph` | ✅ looks exist; morph ❌ | — |
| Live change feed | Resolume WebSocket API ([doc](https://resolume.com/support/en/restapi)) | subscribe to changes | ✅ `changes` ring + actor | — |

---

## Naming & model clashes

| Word | In Resolume / others | In WZRD today | Recommendation |
|---|---|---|---|
| **Deck** | a *page* of clips ("records in a DJ's bag") | the DESIGN \| LIVE area, plus "two legs, one deck" in the design spec | The Studio plan already proposes "deck = clip page" and renaming the DESIGN \| LIVE area to the **TAKE panel**. The research backs that. Confirm it before decks ship, because Resolume users will read "deck" the Resolume way. |
| **Layer** | a row that holds clips | a *binding* (region + effect) in the mixer, while clip-grid rows are *regions* | Two different "rows" in one booth. Either make mixer rows = regions (the engine roadmap's still-open "layer object") or name them differently on screen. |
| **Column** | fires every clip in it at once (a scene) | slot position in the grid | If decks get column triggers, they overlap with **looks**. Pick one "whole-scene" concept. Looks are the better one (they include effects). |
| **Group** | Resolume: a mini-composition of layers; HeavyM: shapes sharing effects | region identity group (a selector) | WZRD matches HeavyM, which is fine. Avoid adding Resolume-style layer groups under the same word. |
| **Scene** | Synesthesia: one shader; TD: a container | the whole `scene.json` | Fine internally. In the UI, "look" already covers saved scenes. |
| **Cue** | Smode/lighting: an armed step, GO fires it | "cue a look into DESIGN" | Consistent. If a cue list arrives, keep "cue = arm, TAKE = GO". |
| **Preview / Program, Take / Cut** | broadcast vision-mixer terms | same | Already aligned with the industry. Keep. |
| **Master** | Resolume: composition opacity; Synesthesia: brightness | the whole operator-owned row | Fine. Keep "masters" plural for the row. |

---

## What nobody has (WZRD's differentiation)

- **Surface-language layers.** Every tool surveyed addresses pixels, slices or shapes. None has named, AI-segmented regions as the unit of play. HeavyM's effects-per-shape-group is the closest, and it is hand-drawn.
- **Additive by construction.** In other tools, "dark stays dark" is a blend-mode choice (Resolume's default Add) that any effect can break. In WZRD it's the model, and the planned post chain clips to the region union.
- **An agent that authors into a probe-gated design leg.** Resolume's 7.26 MCP is the closest convergence and explicitly targets building, not live play. But it has no private design leg, no compile/frame probe, and no "agent never promotes" rule.
- **Region-aware transitions** (regions reveal from their centroids, or cascade in seed order) are planned. Nobody ships transitions that know the geometry.
- **Content made for a region, with a learning record.** No tool ties generated clips to the region they were made for, plus ratings and settings history (the Studio plan's render records).
- **No surveyed tool shows generative-AI authoring** inside the app. Live AI in VJing is StreamDiffusion in TouchDesigner, which trades fidelity and stability for responsiveness ([IIHQ](https://interactiveimmersive.io/blog/artificial-intelligence/using-streamdiffusion-with-touchdesigner-tips-and-tricks-for-advanced-users/)). WZRD's "AI preps, human performs" avoids that trade.

---

## Practitioner signals

| Signal | Implication for WZRD |
|---|---|
| Tempo-synced LFOs beat hand knobs for rhythm ([Meissen](https://matthiasmeissen.substack.com/p/twid-15-vj-system-in-touchdesigner)) | Top pick 3 before more audio features |
| Biggest fears: a crash on show day, deck switches freezing with big files, **MIDI mappings lost after an update** ([forum summaries](https://resolume.com/forum/viewtopic.php?p=108782)) | The MIDI spec's "kept, flagged, inert" rule is the right priority. Rehearse deck switching with real HAP clips |
| External clock drifts; Link preferred; manual re-sync keeps you on beat "around 80%" of a night ([t=20840](https://resolume.com/forum/viewtopic.php?t=20840)) | With no clock (a deliberate choice), hold-to-nudge plus bar-resync are the whole toolkit. Make them great |
| Two-device rig: clip grid + knob unit; endless encoders; LED feedback so eyes stay on the wall ([LIME](https://limeartgroup.com/best-midi-controllers-of-vjs-visual-artists/)) | Macro dials + relative encoders + a hardware sign of which leg the knobs drive |
| A coloured pad says "loaded", not *which* clip | On-screen mirror of the controller grid, or Stream Deck thumbnails later |
| "Freeze the rig" before a show; backups: "better something on screen than nothing" ([VIDVOX](https://vdmx.vidvox.net/tutorials/vj-travel-kit-whats-in-our-bag)) | Pre-flight check + one-key safety look |
| At the venue: placement → optics → warp; coarse-to-fine; recalibrate daily on moving surfaces ([Jaze](https://www.baptistejaze.com/en/blog/calibration-mistakes-mapping)) | Fabric moves. Align undo + quick corner-only re-fit matter more than more handles |
| Beginner mistakes: too bright, too fast, off-beat; "there's a lot of power in darkness" ([STVinMotion](https://www.stvinmotion.com/three-beginner-vj-mistakes/)) | Additive is on the right side of this. An autopilot with per-look durations helps pacing |
| TD performers find building a perform UI a huge separate project ([TD forum](https://forum.derivative.ca/t/do-you-really-use-perform-mode/271810)) | WZRD's ready booth UI over a separate engine is a selling point |
| Grid laid out by energy: calm left, peak right | Deck/slot order could carry an energy hint for the autopilot |

Evidence quality: forum material comes from search summaries, not direct reads. There is no usage data or vote ranking anywhere. Treat these as directional.

---

## Deliberately skipped

| Skipped | Why |
|---|---|
| DMX / Art-Net pixel mapping, fixtures, LED scanner, moving heads, lasers | lighting-console territory, not projection |
| Denon StageLinQ, Pioneer Pro DJ Link | DJ track sync; the gig is live guitar |
| SMPTE / LTC / MTC timecode, timelines with montage tracks | no pre-timed show; maybe later for installations |
| Multi-projector edge blending, black-level compensation | one projector on the base M2; deferred in the engine roadmap |
| 3D object calibration (CamSchnappr), 3D stage visualisers (disguise) | design spec: "not a 3D scene editor" |
| Syphon / Spout / NDI / capture cards | rejected for v1; only for a future Resolume hand-off |
| Recording, offline render export | rejected in the engine docs |
| Audio playback, audio effects, VST | WZRD isn't an audio tool |
| TouchDesigner two-way "Bind" between peers | WZRD's single commit path for scene edits is deliberate |
| Calendar (ICS) scheduling, understudy backup machines | installation / big-show features; revisit for installs |
| Leap Motion, QR/OCR inputs, vision effects (depth, background removal) | no fit with an offline-segmented surface |
| 10-bit / HDR output, Windows fullscreen-exclusive mode | low value / not Mac |

---

## Sources

**Resolume:** [Layers](https://resolume.com/support/en/layers) · [Clips](https://resolume.com/support/en/clips) · [Decks](https://resolume.com/support/en/decks) · [Composition](https://resolume.com/support/en/composition) · [Autopilot](https://resolume.com/support/en/autopilot) · [BPM](https://resolume.com/support/en/bpm) · [Parameter animation](https://resolume.com/support/en/parameter-animation) · [Parameters](https://resolume.com/support/en/parameters) · [Envelopes](https://resolume.com/support/en/envelopes) · [Dashboard](https://resolume.com/support/en/dashboard) · [Effects](https://resolume.com/support/en/effects) · [Undo](https://resolume.com/support/en/undo) · [Layouts](https://resolume.com/support/en/layouts) · [Keyboard](https://resolume.com/support/en/keyboard-shortcuts) · [MIDI](https://resolume.com/support/en/midi-shortcuts) · [OSC](https://resolume.com/support/en/osc) · [REST](https://resolume.com/support/en/restapi) · [Link](https://resolume.com/support/en/link) · [Screens](https://resolume.com/support/en/screens) · [Input selection](https://resolume.com/support/en/input-selection) · [Output transformation](https://resolume.com/support/en/output-transformation) · [Modifiers](https://resolume.com/support/en/modifiers) · [Clip time](https://resolume.com/support/en/clip-time-panel) · releases [7.18](https://resolume.com/blog/22892) [7.19](https://resolume.com/blog/23542) [7.21](https://resolume.com/blog/24836) [7.22](https://resolume.com/blog/27125) [7.23](https://resolume.com/blog/30807) [7.24](https://www.resolume.com/blog/32999) [7.25](https://resolume.com/blog/34172) [7.26](https://resolume.com/blog/34484) [7.27](https://resolume.com/blog/34561) [7.28](https://resolume.com/blog/34630)

**TouchDesigner:** [Perform mode](https://docs.derivative.ca/Perform_Mode) · [Window COMP](https://docs.derivative.ca/Window_COMP) · [Parameter COMP](https://docs.derivative.ca/Parameter_COMP) · [Binding](https://docs.derivative.ca/Binding) · [MIDI mapper](https://docs.derivative.ca/MIDI_Device_Mapper_Dialog) · [Beat CHOP](https://docs.derivative.ca/Beat_CHOP) · [audioAnalysis](https://docs.derivative.ca/Palette:audioAnalysis) · [Lag CHOP](https://docs.derivative.ca/Lag_CHOP) · [Stoner](https://docs.derivative.ca/Palette:stoner) · [Kantan](https://docs.derivative.ca/Palette:kantanMapper) · [CamSchnappr](https://docs.derivative.ca/Palette:camSchnappr) · [Engine COMP](https://docs.derivative.ca/Engine_COMP) · [Probe](https://docs.derivative.ca/Palette:probe) · [TDMorph](https://derivative.ca/community-post/asset/tdmorph/73268) · [CuryCue](https://github.com/netzzy/curycue) · [SonomikaTD](https://derivative.ca/community-post/asset/sonomikatd-touchdesigner-vj-system/75106) · [TD MCP (Glama)](https://glama.ai/mcp/servers/lucasmaher-hash/touch-designer-mcp/tools/create_preset_morph)

**Others:** MadMapper [features](https://madmapper.com/madmapper/features) · [Space Scanner](https://madmapper.com/files/13-Space%20Scanner%20Guide-1.pdf) · [CDM v6](https://cdm.link/madmapper-6-brings-timelines-markers-clips-more/) — HeavyM [effects](https://help.heavym.net/hc/sections/360004684840-Managing-Group-Effects) · [sequences](https://help.heavym.net/hc/articles/9172061701138) · [1.7](https://www.heavym.net/?p=3135) — Millumin [change log](https://help.millumin.com/docs/general/change-log/) — VDMX [data sources](https://vdmx.vidvox.net/tutorials/overview-of-available-data-sources-in-vdmx) · [plugins](https://docs.vidvox.net/vdmx_plugins.html) — Synesthesia [controls](https://app.synesthesia.live/docs/manual/controls.html) · [presets](https://app.synesthesia.live/docs/manual/presets.html) · [audio uniforms](https://app.synesthesia.live/docs/ssf/audio_uniforms.html) — Smode [cue list](https://www.smode.io/doc/ref/animation/cue-list.htm) — Notch [exposed params](https://manual.notch.one/2026.1/en/docs/workflows/working-with-media-servers/exposed-parameters/)

**Practitioners:** [Meissen TD VJ system](https://matthiasmeissen.substack.com/p/twid-15-vj-system-in-touchdesigner) · [VIDVOX travel kit](https://vdmx.vidvox.net/tutorials/vj-travel-kit-whats-in-our-bag) · [LIME controllers](https://limeartgroup.com/best-midi-controllers-of-vjs-visual-artists/) · [Baptiste Jaze calibration](https://www.baptistejaze.com/en/blog/calibration-mistakes-mapping) · [test patterns](https://www.baptistejaze.com/en/projector-calibration/projector-test-patterns) · [STVinMotion](https://www.stvinmotion.com/three-beginner-vj-mistakes/) · [vjacademy](https://vjacademy.info/resolume-arena-guide) · [TD forum: Perform mode](https://forum.derivative.ca/t/do-you-really-use-perform-mode/271810) · [TD forum: cook load](https://forum.derivative.ca/t/reducing-complexity-cook-load-for-live-setup/273176) · Resolume forum (via search summaries): [mappings lost](https://resolume.com/forum/viewtopic.php?p=108782), [Link/resync](https://resolume.com/forum/viewtopic.php?t=20840), [tap rounding](https://resolume.com/forum/viewtopic.php?t=21632), [UI lock](https://resolume.com/forum/viewtopic.php?t=26851) · [IIHQ StreamDiffusion](https://interactiveimmersive.io/blog/artificial-intelligence/using-streamdiffusion-with-touchdesigner-tips-and-tricks-for-advanced-users/)
