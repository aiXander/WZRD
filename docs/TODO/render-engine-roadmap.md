# Render engine — structural roadmap

> The forward plan for `render-core/` + `wzrd-app/`. Current system state,
> contracts, and invariants live in
> [../reference/render-engine.md](../reference/render-engine.md) — references
> of the form §1–§4, §6, §7 point there. Section numbers here keep the
> **§5.x** form so cross-references in the reference doc stay valid.
>
> Ordered by leverage toward `../reference/user_design_spec.md`. Each item is
> scoped so an agent can pick it up standalone. **General rule: engine first,
> headless-verifiable, UI second.** §5.1–§5.6 are resolved (landed or
> dropped); the single-process collapse + §5.6 two-deck **LANDED
> 2026-07-12**; **§5.10 authoring MCP + the §5.13 engine slice LANDED
> 2026-07-22**; **§5.14 alignment layer LANDED 2026-08-12** — contracts in
> [../reference/render-engine.md](../reference/render-engine.md)
> §1/§1b/§2.2/§2.6/§2.7/§2.8.
> **This file holds engine designs; it does not schedule them.** Order and
> hand-off packages live in two plans: **gig (Nov 2026)** —
> [gig-2026-11-live-video.md](gig-2026-11-live-video.md): §5.9 video, then
> §5.11's probation window, plus the §5.11 "restart engine" button with the
> project switcher; **post-gig** — [vj-toolbox.md](vj-toolbox.md): §5.5
> `#import` → §5.15 chains → §5.16 feedback + `post` → §5.17 TAKE
> transitions → the §5.6 auto-pilot playlist. §5.7 (layer object), §5.8 and
> the rest of §5.11 / §5.12 are unscheduled.

### 5.1 Live transport: music-locked BPM — DROPPED (2026-07-11)

Out of scope. Live musical energy arrives as discrete `audio.onset` events
(already consumed); `transport.bpm` stays a static scene value driving
`clock.*` phase, and the audio server's `/audio/bpm` stream stays ignored by
`osc.rs`. Revisit only if a real scene proves onsets + static clock
insufficient.

### 5.2 Per-layer variation + `pick` selectors — LANDED (2026-07-11)

See reference §2.1 (`pick` grammar) + §2.4 (per-layer identity accessors).
Picks are stateless — a pure hash of (binding id, transport cycle) via
`compositor::pick_choice` — so no RNG state crosses the §5.6 legs.
`phase3_smoke.scene.json`'s `pick_bloom` binding demos both.

### 5.3 Operator-owned state: the session sidecar — LANDED (2026-07-11)

See reference §2.5 (full contract). `session.rs` owns load/save (atomic
temp+rename; gitignored; per *directory*/venue, not per scene); debounce
drained in `Core::poll_inbound`; SIGTERM/SIGINT → snapshot + graceful exit.
Calibration has since moved out of the sidecar entirely — see §5.14 /
reference §2.8. The
"show file" umbrella (playlist + scenes + session) still waits for §5.6's
auto-pilot playlist.

### 5.4 Masters row + crossfade-time master — LANDED (2026-07-11 / 2026-07-22)

See reference §2.5. `Masters` atomics in `drivers.rs` — brightness/saturation
in the final pass, speed as `Transport` time integration (bends
time, never jumps), audioListen scales every `audio.*` read; RPC
`master.set`/`master.list`, persisted in the sidecar, always-visible Perform
row. The `promote` crossfade fade is an engine-wide master
(`drivers::Crossfade`, seconds, 0–30, `0`=CUT) — *not* per-leg, never copied
on promote/pull; a `promote` with no explicit `fade_ms` falls back to it.
DeckBar exposes it as a logarithmic FADE fader (CUT at bottom of travel).

### 5.5 Params first-class: descriptor knobs + overrides — LANDED (2026-07-11)

See reference §2.3. Descriptors carry `min`/`max`/`step`/`unit`/`widget` per
input (`effect.describe`); `param.set {binding, param, value}` pins any scalar
via the `ParamOverrides` table consulted in `tick()` (keyed (binding id, param
name), outside the plan → survives rebuilds; `null` clears). scene.json is
never written implicitly. Audio conditioning stays server-side.

**Still open — shared WGSL utilities via `#import` (specced 2026-07-22;
fast-follow immediately after §5.10, which multiplies the duplication — an
authoring agent re-implements `permute()`/`hsv2rgb()` in every effect it
writes).** WGSL has no native include; the fix is a small text preprocessor
in the single compile entry point (`gpu::compose_shader` /
`build_effect_pipeline` — still a one-layer change):

- **Syntax:** a line `#import <module>` (e.g. `#import noise`), processed
  before `naga` ever sees the source. Recursive (lib modules may import
  each other), cycle-rejected, deduped — each module inlined at most once
  per composed shader.
- **Resolution order:** project-local `<effects_dir>/_lib/<module>.wgsl`
  first, then engine-shipped built-ins
  (`render-core/shaders/lib/<module>.wgsl`, embedded via `include_str!`) —
  projects can shadow built-ins. Unknown module = prescriptive compile
  error listing the available modules.
- **Line remapping must survive:** the composer already remaps naga
  diagnostics past the prelude; extend the flat offset to a real source
  map so an error reports `_lib/noise.wgsl:12`, and `wgsl.validate` (and
  the §5.10 `validate_wgsl` tool) inherit it.
- **Cache + probe correctness fall out for free:** the §2.6
  content-hashed pipeline keys hash the **post-preprocess** source —
  editing a lib module changes every dependent effect's key, so dependents
  re-enter the cache as new pipelines (and get probed) while live keeps
  drawing the old ones. The watcher adds `_lib/` to its watch set; a lib
  change marks all user effects dirty (the mtime rescan in `effects.rs`
  already exists).
- **MCP exposure (§5.10 landed — wire in when this ships):** lib modules
  live in the `effects` facet, ids namespaced `_lib/<module>`;
  `upsert_effect({name: "_lib/noise", wgsl})` writes a module (no
  descriptor, no probe of its own — dependents re-probe via the key
  change).

### 5.6 Design/Live legs + Promote/Pull (the two-deck architecture) — LANDED (2026-07-12)

Full contract in reference **§2.6** (legs + blanket leg rule, promote/pull +
re-entrancy, demand-gated design rendering, content-hashed pipeline keys +
cross-leg GC, pre-flight probe, design autosave, preview source toggle) and
§2.3 (RPC verbs + `deck` channel + `hot_reload.probe`). Residue worth knowing:

- Live-freeze is via **content-derived pipeline keys** + a retain-set GC (an
  edited shader gets a new cache slot; live keeps the old one) — not eviction
  guards.
- The deck toggle is a **full control switch**: per-leg
  transport/masters/knobs/overrides, copied design→live on promote and
  live→design on pull. (The "shared knobs, no copy" idea died same-day — design
  speed 4× also sped the show.)
- Probe frames interleave with live (~3.5 ms/loop-iteration budget, calibrated
  per boot). Headless (no `--ws-addr`) = single live leg, byte-identical to the
  pre-two-deck engine.

**Still open (deferred out of §5.6 scope by design, 2026-07-12): the
engine-side auto-pilot playlist** (spec §13). A UI-driven stopgap landed
2026-10-09 (looks bank + AUTO every N bars in Perform — see
[../reference/operator-ui.md](../reference/operator-ui.md)); it dies with
the app window, which is why the engine version below still matters — a thin scheduler composing existing
verbs, no new engine code: `scene.load` → design, wait for build/probe OK
(the `scene.load` reply already carries the probe verdict), `promote
{quantize:"bar"}`, dwell, repeat, skip entries that fail. Ship it once
promote has been proven solid at a real show. (The §5.4 **crossfade-time
master** — "promote fade default" as an operator master — landed separately
on 2026-07-22; see §5.4.)

### 5.7 Layer object + intensity (carried over from v1 review #2/#9)

**Runtime half LANDED 2026-10-09:** per-binding opacity / mute / solo as
reserved `@opacity`/`@mute`/`@solo` overrides folded into `layer.opacity`
(reference §2.5), driven by the Perform layer mixer. **Still open:** the
scene.json `layers:` block (authored intensity that survives into looks and
headless runs), z-override, and per-*region* (vs per-binding) addressing.

Introduce an explicit `Layer` concept between scene and bindings: per-region
**intensity** (how hard its bindings light it), **mute**, and z-override,
addressable as `layers: { "trunk": { "intensity": 0.8 } }` in scene.json and
tweakable via the live param path (§5.5). The deck UI in Perform then lists
layers, not raw bindings — matching how the performer thinks (spec: "the
deck reads in surface-language").

### 5.8 Preview & render-loop upgrades

- **Collapse LANDED 2026-07-12; spikes passed; the subprocess fallback is
  retired.** The native preview surface exists (reference §1b), and
  demand-gated capture landed with it (`Bus::subscriber_count` gates
  `PreviewSampler`). The binary-frames and design-leg-`PreviewSampler`
  items are **dead** — revive only if remote (WS) operation ever becomes a
  primary workflow and the JPEG thumbnail stops sufficing there.
- Optional, still open: `WaitUntil` scheduling; GPU timestamp queries into
  `frame_stats`; adaptive frame cap (drop to 30 Hz when no audio + no
  clock-driven bindings are active).

### 5.9 Video layers — scope decided 2026-10-10, built as gig package E1

**Scope (gig plan D1–D6):** **HAP only** — plain HAP (BC1) required, HAP Q
if cheap, no alpha variant; no H.264/HEVC path and **no compute-shader DXT
fallback**. Request `TEXTURE_COMPRESSION_BC` only when the adapter offers it;
without it `videoClip` refuses with a clear error and the rest of the engine
runs (the one documented exception to `Features::empty()`, recorded in
reference §1b when it lands). Video is an effect **`image` input** read via
`sample_image(uv)`, so any WGSL effect can modulate it; two per-binding
placements, **canvas** (full-frame clip cut by the mask) and **region** (clip
scaled into the layer's padded `source_box`, `bbox` fallback). MP4 → HAP
transcode is MCP-side, never in the engine. The scene/pack shapes (clip
binding, stateless playhead, `source_box`) are pinned as contracts **C4/C5**
in [gig-2026-11-live-video.md](gig-2026-11-live-video.md); build steps and
gates are its package **E1**.

**The load-bearing invariant** (v1 §3.8, still correct): each stream owns a
**ring of N mapped `wgpu::Buffer` staging slots** (N≈3, allocated at
stream-open, recycled). A **decode thread** does all demux, snappy and disk
I/O and pushes "slot ready" over a lock-free channel; the render thread's
only contact with video is one `copy_buffer_to_texture` from the freshest
ready slot. If decode falls behind, reuse the last good slot — a stutter,
never a frame-stall. Watch the HMR/cleanup failure mode (reference §4
safety/scaling risks): staging slots from a closed stream must drop.

**E1 design pass (2026-10-10, before step b):**

- **Demuxer: a hand-written ISO-BMFF/QuickTime sample-table reader** in
  `video.rs` (~250 lines: `moov/trak/mdia` → `mdhd` timescale, `hdlr` =
  `vide`, `stsd` fourcc + width/height, `stts`/`stsc`/`stsz`/`stco|co64` →
  one (offset, size, start time) per frame) + the pure-Rust **`snap`**
  crate. No ffmpeg link (Linux build), and not the `mp4` crate — it only
  models the sample entries it knows (`avc1`, `hev1`, …) and HAP's
  `Hap1`/`HapY` are not among them. HAP frames are parsed per the Vidvox
  spec: section header (3-byte size + type, 0 → 4-byte extended size),
  compressor none/snappy/chunked ("complex": decode-instruction container
  with per-chunk compressor + size tables). Accepted: `Hap1` (BC1, sRGB),
  `Hap5` (HAP Alpha, BC3 sRGB — same code path, so it comes free), `HapY`
  (HAP Q: non-sRGB BC3, scaled-YCoCg → RGB → linear in `sample_image`).
  Refused with a named error: `HapM`/`HapA`/`Hap7`, any other codec, and
  dimensions that aren't a multiple of 4 (BC blocks; P5 pads).
- **No stream sharing between legs.** A stream belongs to one *(plan,
  binding)*; the N passes of a multi-layer binding share its one frame
  texture, each placed in its own box. Why: playheads are per leg (per-leg
  clocks + speed masters, §2.6), so live and design can want different
  frames of the same file; the gate's worst case (a TAKE that swaps all
  seven clips) has nothing to share anyway; and because the plan owns its
  streams, the promote pointer swap carries them into live — a clip never
  reopens or jumps at TAKE. Design's post-promote rebuild opens fresh
  streams that cost nothing until design renders (decode is
  demand-driven, below). Cost accepted: duplicate decode while design
  *renders* the same clip live is showing.
- **Bind-group slot (landed in step a):** group 0, binding 4
  `texture_2d<f32>` + binding 5 sampler, in the one layer layout every
  pass and probe item uses; passes without an image bind a 1×1
  transparent dummy. Placement rect + encoding ride `LayerParams`
  (`image_box`, `image_meta`).
- **Staging ring per stream:** N = 4 `MAP_WRITE | COPY_SRC` buffers,
  rows padded to 256 B. Slot cycle: *free* (mapped, decode thread writes
  it, unmaps, sends `ready{slot, frame}`) → *ready* (render thread holds
  it) → one `copy_buffer_to_texture` into the stream's frame texture →
  after submit, `map_async` → callback hands it back as *free*. Stale
  ready slots recycle the same way without a copy.
- **Demand-driven decode:** each rendered frame the render thread computes
  the wanted frame and `try_send`s `want{frame, step}` (bounded channel —
  never blocks); the decode thread decodes `frame, frame+step, frame+2·step`
  into free slots, where `step` is the observed frames-per-render-frame
  (≥ 1, signed by rate) so 8× speed still prefetches the right frames. No
  wants → no work: a design leg nobody watches decodes nothing. Upload
  picks the ready slot nearest the wanted frame without passing it; none
  → keep the current frame (a stutter, never a stall).
- **Playhead (C5 confirmed as written):** `frame = sample at ((leg
  transport seconds × rate) rem_euclid duration)`, read from the leg's own
  integrated clock, so the speed master bends clips and promote's clock
  copy keeps them continuous across TAKE.
- **BC:** `TEXTURE_COMPRESSION_BC` is requested iff the adapter offers it
  (`GpuContext::bc_supported`); without it every HAP open fails with a
  named per-binding warning and stills keep working.
- **Teardown:** dropping a stream drops its channels; the decode thread
  exits on disconnect and is never joined on the render thread; it holds
  the staging buffers, so they drop with it.

**Later, not scheduled:** H.264/HEVC hardware decode (VideoToolbox / NVDEC /
VAAPI) for single-stream content; stills and pack `references/` through the
same `image` input. ProRes out of scope.

### 5.10 Authoring MCP — the AI scene/shader co-author — LANDED (2026-07-22)

Done — full contract in [../reference/render-engine.md](../reference/render-engine.md)
**§2.7** (tool surface, facet taxonomy, the six-piece engine cluster:
rev/epoch change ring + sticky `changes` channel, per-connection actor via
`hello`, probe-deferred `effect.upsert`, `base_rev` CAS, webview
reverse-sync + ADOPT button, agent mirror file) and §2.2/§2.3 (identity
sidecar, RPC additions). Python side: `wzrd_mcp/engine_tools.py`
(`.[engine]` extra, default-off in `server.py`, on in the local
`tools_config.json`); setup recipe in the repo `README.md`. Decisions that
constrain future work: Claude Code is the author (no embedded-LLM tool); no
live operator controls as tools and **no agent `promote`, ever**; no
agent-decided persistence; rejected shapes listed in §2.7.

Still open (small): `get_scene_context`'s `layers` digest could fold in the
§5.5 `_lib` modules once the `#import` preprocessor lands (`_lib/<module>`
ids in the `effects` facet — see §5.5).

### 5.11 Reliability hardening (spec §9: "trust it to stay up")

**Accepted recovery contract (operator, 2026-07-11) — the Resolume
formula.** A rare full blackout of ~20 s is acceptable *provided restore is
total*: relaunch comes back with the same scene, knobs, masters,
calibration (§5.3 sidecar) and in-flight design draft (§5.6 design-leg
autosave). This is the yardstick every architecture choice measures
against — supervised subprocess respawn (~2 s) comfortably beats it;
single-process relaunch-with-restore meets it. Defense in depth, in order:
(1) the §5.6 pre-flight probe keeps bad shaders out of both legs,
(2) the probation window below catches full-res budget blowers,
(3) snapshot/restore bounds the damage of whatever still gets through.

- ~~Tauri shell supervises the engine child~~ — superseded by the **landed**
  collapse (2026-07-12): the mechanism is **relaunch-with-restore** per the
  contract above. Spike results: the render loop already runs under
  `catch_unwind` (a panic — incl. wgpu device-loss fatal errors — kills
  the engine thread, not the webview; `engine:status` reports it), state
  files stayed byte-identical through both crash modes, and measured
  relaunch-to-light was ~160–290 ms — the ≤20 s contract has huge margin.
  Still open here: an in-app "restart engine" button (today: quit +
  relaunch) — **gig package A1**, built with the project switcher, which
  re-roots the engine with the same respawn
  ([gig-2026-11-live-video.md](gig-2026-11-live-video.md) D13) — and the
  post-crash wart that SIGTERM is a no-op (reference §1b), which stays in
  this hardening pass.
- Engine startup is already last-good-tolerant (bad scene → previous plan);
  extend to "scene fails at boot → black composite + hot-reload watch"
  instead of exit.
- Headless autostart recipe (launchd/systemd) documented for installations.
- ~~Slider/master state snapshot on SIGTERM~~ — **done** (landed with §5.3,
  2026-07-11: `signal-hook` flag → sidecar snapshot → graceful exit).
- **Post-swap probation window — pulled into the gig as package E2**
  ([gig-2026-11-live-video.md](gig-2026-11-live-video.md) D15, which adds the
  TAKE-crossfade false-positive trap). The MCP multiplies AI shader churn by an order of
  magnitude, and yellow-verdict shaders can still be promoted to live —
  the probe gates *entry to design*, the probation window is the only
  live-side net. `naga` proves a shader *compiles*; a
  valid-but-pathological one can still blow the frame budget (reference §4
  risk list). Extend swap-on-success: after any pipeline/plan swap, watch
  frame time for ~30 frames; if p95 blows the budget and the swap is the
  delta, auto-revert to the retained previous pipeline and emit a
  `hot_reload` failure (`reverted: frame budget`). All the machinery exists
  (retained old pipeline, `FpsAccumulator`, sticky telemetry) —
  swap-on-success becomes swap-on-*performs*.
- **Headless status file.** The file-based agent path has no way to read
  compile errors short of tailing stderr — but the spec has the AI
  "self-correct from its own compile errors without me refereeing." After
  every reload attempt, mirror the sticky `hot_reload` (+ `connectivity`)
  payloads to `<scene_dir>/.wzrd/status.json`. Agent loop: write file →
  read status → fix. Keep it a byte-for-byte mirror of the telemetry
  payloads — one shape, two transports.

### 5.12 UI polish backlog (post-structural)

Calibration UI — **superseded and shipped** as the §5.14 alignment layer
(n-point warp + the Align route); see reference
[§2.8](../reference/render-engine.md#28-alignment-layer-the-n-point-output-warp-514-landed-2026-08-12).
Still open from the old line: the **re-shoot recovery path** (v1 §3.9) —
when the projector/surface is bumped, run the offline `align` step on a new
capture against the original reference photo and push the resulting corners
through `alignment.set`, no re-segmentation. That is now an external script
against the existing WS surface, not engine work.
Layer deck panel in Perform — **landed 2026-10-09** with the UI redesign
(see [../reference/operator-ui.md](../reference/operator-ui.md)), as did a
looks bank (saved scenes) standing in for "scene save-as + chooser".
Still open: MIDI controller mapping (WKWebView has no Web MIDI — needs a
Rust-side bridge; full spec in [midi-controller.md](midi-controller.md));
virtualized log list; region renaming + `identity.json` editing (→ §5.13,
delivered inside the Site room, vj-toolbox phase 5). In-app AI chat is
parked — §5.10 settled Claude Code as the author; revisit on top of
[vj-toolbox.md](vj-toolbox.md) phase 4's MCP client.

### 5.13 Identity sidecar: group authoring + region renaming — ENGINE SLICE LANDED (2026-07-22)

**Engine slice done** (as step 0 of §5.10) — see reference **§2.2**:
`identity.json` load/merge in `pack.rs` (lenient at load, strict on write,
labels included from day one) + the queued `identity.setGroups
{groups?, labels?}` RPC (per-key delta, `null` removes; persists, refreshes
`pack.info`, re-resolves design selectors). Agent-side authoring works today
via the §5.10 `set_groups`/`set_labels` MCP tools.

**Still open — the UI half:** multi-select on the Surface canvas (⌘/⇧-click
accumulates a selection; pixel-accurate picking already exists) → "New group
from selection" → commits via `identity.setGroups`; region renaming in
Prepare writes `labels` the same way. Distinct from `sceneCommit.ts` — this
writes the sidecar, not the scene. Built as the "name + group" step of the
Site room ([vj-toolbox.md](vj-toolbox.md) phase 5), reusable from Prepare.

Non-goals: no group nesting, no per-group colour/metadata beyond membership,
no offline `wzrd.layerpack` group-reconciliation (that rides the broader
identity-table re-import work in §2.2).

### 5.14 Alignment layer: the n-point output warp — LANDED (2026-08-12)

Full contract in reference **§2.8** (model, `alignment.json`, the
engine-wide/not-per-leg rule, the LUT invariant, the Y-flip trap, UI
behaviour) and §2.3 (three RPC verbs + the sticky `alignment` channel).
Supersedes the §5.12 calibration-UI line and generalises **D9**. Residue
worth knowing here:

- The base stays **projective** (4-corner homography); extra handles are
  compactly-supported Wendland corrections *on top of it*. Fitting an
  interpolator through the corners instead was considered and rejected —
  straight edges bow under keystone.
- Runtime representation is a baked **offset LUT**, so per-frame cost is one
  texel read regardless of handle count. That is also the seam a dense
  camera-solved field uploads into later.
- Not scene content, not per leg, no MCP verb.

**Still open (deliberately deferred):** camera-driven auto-alignment
(`alignment.setField` dense upload + the external capture→detect→solve loop,
skeleton in `render-core/tools/align_drag.py`); a warped native preview
toggle if the unwarped one proves confusing on the Align tab;
multi-projector / edge blending. The default handle radius (0.35) wants one
real projector session to tune — see the corner-drift caveat in §2.8.

### 5.15 Per-binding effect chains (added 2026-10-10)

Today one binding = one effect over its regions; stacking two bindings on a
region only *adds* light, it can't *process* (no "clip → kaleido → strobe").
Add a short **linear chain per binding** — the "flat per-layer stacks" the
v1 D5 decision always allowed (reference §7 rejects a DAG, not this).

- **Scene shape:** `"effect"` may become an array of stages, each
  `{ "effect": <name|inline>, "params": {…}, "id": <stable stage id> }`;
  the single-effect form stays valid (a one-stage chain). Stage 0 is a
  *source* (a generator or `videoClip`), later stages are *filters*.
- **Filter contract:** filters implement the same `fn effect(uv, mask)` but
  get a prelude accessor `sample_prev(uv)` — the previous stage's output.
  Descriptors gain `"kind": "source" | "filter" | "transition"` (§5.17
  uses the third) so UI, MCP and library can tell them apart; a filter at
  stage 0 is a load error.
- **Gig clip effects migrate here:** gig package F1's ten clip effects read
  the clip themselves via `sample_image` (one effect per binding). With
  chains they become `kind: "filter"` stages — swap `sample_image` →
  `sample_prev` and move the clip to a `videoClip` stage 0. Old one-effect
  clip bindings stay valid (a source that reads an image).
- **Execution — decide in the design pass, leaning (a):** (a) each stage
  renders into a pooled per-leg intermediate texture (region `source_box`-
  sized when the binding is region-placed, canvas-sized otherwise), the
  last stage composites additively with the mask as today — predictable
  cost, each stage keeps its own uniform block and its own probe; (b)
  compile-time fusion (stages renamed `stage_k`, `sample_prev` rewritten to a
  call of `stage_{k-1}` at a displaced uv) — no textures, but cost multiplies
  with every tap a filter takes. (a) is also the FBO machinery the unbuilt
  `layerRef` slow path (reference §4 item 6) needs; build it once.
- **Live knobs per stage:** overrides key on `(binding id, stage id,
  param)`; a reserved per-stage `@bypass` override (like `@mute`, reference
  §2.5) toggles a stage with zero rebuild. Content-hashed pipeline keys,
  probe and the cross-leg GC (reference §2.6) apply per stage unchanged.
- **MCP:** `upsert_binding` accepts the chain form; `get_scene_context`
  digests it. Skill `live-engine` teaches source vs filter.

Gate: a region-placed clip → kaleido → strobe chain on 7 regions at 60 fps
on the M2; bypassing a stage mid-show is glitch-free; headless loads the
same scene.

### 5.16 Feedback texture + the global `post` chain (added 2026-10-10)

Two primitives most classic VJ effects need. Both build on §5.15's filter
contract.

- **Feedback:** a chain may opt in (`"feedback": true` on the binding) to a
  per-binding, **per-leg** previous-frame texture, read via
  `sample_feedback(uv)` in any stage. Enables trails, echo, smear, slit-scan.
  One extra texture per opted-in binding; cleared on plan rebuild of that
  binding. Trap to settle: on promote, live's feedback continues from its own
  history (not design's) — note it in reference §2.6 when it lands.
- **`post` chain:** `scene.json`'s `post: []` is parsed but never consumed
  (reference §4 item 6). Consume it as a chain of filters over each leg's
  composite, run before the final pass (live×design mix → masters → warp),
  so masters still sit on top and stay operator-owned. **Dark stays dark:**
  post output is multiplied by the union of all region masks unless an
  effect's descriptor opts out (`"spill": true`) — a blur or glow must not
  light the wall between regions by default. Per leg like everything else;
  TAKE copies design's post chain live.
- The probe covers post stages exactly like binding stages.

Gate: trails on one fabric region + a global glow post, dark gaps between
regions measured black (`check_frames` lit-fraction unchanged within 0.1 %).

### 5.17 TAKE transitions (added 2026-10-10)

The only transition today is the promote crossfade: the final pass lerps
live × design by the promote mix (reference §2.6). Generalise the lerp into
a **transition shader**: `fn transition(uv, mix) -> vec4` with
`sample_live(uv)` / `sample_design(uv)` accessors, chosen per TAKE.

- **Region-aware by default** — the point of doing this in WZRD rather than
  as a generic wipe: the transition can read the mask atlas + per-layer
  geometry (centroid, bbox, seed) so "each region reveals from its centroid"
  or "regions flip one by one in seed order" are expressible. Only pixels
  whose design and live values differ need to animate; unchanged regions
  stay put.
- Ships with: dissolve (today's lerp, the default), luma wipe, region
  cascade, centroid reveal. Transitions are `kind: "transition"` effect files
  — library items, AI-writable, probe-gated like any effect.
- **Operator control:** the transition is an engine-wide master next to the
  §5.4 FADE master (never per leg, never copied on promote/pull), with an
  optional per-`promote` override so a look in the looks bank can carry its
  own transition. `promote {transition?}` param; `deck` channel reports it.

Gate: CUT, dissolve and region cascade all bar-quantized at 60 fps; a broken
transition shader falls back to dissolve, never a black wall.
