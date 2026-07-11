# Render engine — structural roadmap

> The forward plan for `render-core/` + `wzrd-app/`. Current system state,
> contracts, and invariants live in
> [../reference/render-engine.md](../reference/render-engine.md) — references
> of the form §1–§4, §6, §7 point there. Section numbers here keep the
> **§5.x** form so cross-references in the reference doc stay valid.
>
> Ordered by leverage toward `../reference/user_design_spec.md`. Each item is
> scoped so an agent can pick it up standalone. **General rule: engine first,
> headless-verifiable, UI second.** §5.1–§5.5 are resolved (landed or
> dropped, 2026-07-11); §5.6 (two-deck) is the next big structural one.
>
> Related open decision (uncommitted): single-process collapse for a
> lossless preview — [single-process-collapse.md](single-process-collapse.md).

### 5.1 ~~Live transport: music-locked BPM~~ — DROPPED (2026-07-11)

Live BPM tracking is out of scope for now. BPM is a slowly-changing smoothed
float with little value in this engine — the live musical energy that matters
arrives as discrete events (kicks/onsets via `audio.onset`), which the engine
already consumes. `transport.bpm` stays what it is today: a static scene
value driving `clock.*` phase. The audio server's `/audio/bpm` stream stays
deliberately ignored by `osc.rs`. Section number retained so §5.x
cross-references stay valid; revisit only if a real scene proves onsets +
static clock insufficient.

### 5.2 Per-layer variation + `pick` selectors — LANDED (2026-07-11)

Implemented — see [../reference/render-engine.md](../reference/render-engine.md)
§2.1 (`pick` grammar + strictness) and §2.4 (per-layer identity accessors:
`layer_seed()`, `layer_index()`/`layer_count()`, `layer_centroid()`,
`layer_bbox()`). Residue worth knowing: picks are stateless — a pure hash
of (binding id, transport cycle) via `compositor::pick_choice`, rate
restricted to `clock.*` (`drivers::PickRate`) — so no RNG state exists to
carry across the §5.6 legs; all member passes stay in the plan and a
re-pick just flips `active` flags. `phase3_smoke.scene.json`'s `pick_bloom`
binding demos both features (inline WGSL using `layer_centroid()` +
`layer_seed()`, re-picked every 2 bars).

### 5.3 Operator-owned state: the session sidecar — LANDED (2026-07-11)

Implemented — see [../reference/render-engine.md](../reference/render-engine.md)
§2.5 for the full contract (sidecar shape, scope rule, write policy, read
precedence). Residue worth knowing: `session.rs` owns load/save (atomic
temp+rename; `session.json` is gitignored); the sidecar is per *directory*
(venue), not per scene; the debounce rides a shared epoch-ms `AtomicU64`
touched by the WS thread and drained in `Core::poll_inbound` (~1.5 s);
SIGTERM/SIGINT land via a `signal-hook` flag → snapshot + graceful host
exit, which is §5.11's power-blink item done early. Reference §4 weakness 5
(knob persistence) is resolved. `projectorCalibration` in scene.json is a
deprecated read-only fallback. The "show file" umbrella (playlist + scenes
+ session) still waits for §5.6's auto-pilot playlist.

### 5.4 Masters row — LANDED (2026-07-11)

Implemented per the original sketch — see reference §2.5. `Masters` atomics
in `drivers.rs`; brightness/saturation in the final homography pass (the
preview deliberately shows the un-mastered composite); speed as per-frame
`Transport` time integration (bends time, never jumps); audioListen scales
every `audio.*` read including the `state.audio_*` uniform. RPC
`master.set`/`master.list` + sticky `masters` telemetry; persisted in the
sidecar; always-visible row in Perform (double-click a label to reset).
**Crossfade-time master still pending** — it only means something once
§5.6's promote exists; add it there.

### 5.5 Params first-class: descriptor knobs + overrides — LANDED (2026-07-11)

Implemented — see reference §2.3. Descriptors (user *and* built-in) carry
`min`/`max`/`step`/`unit`/`widget` per input, served by `effect.describe`;
`param.set {binding, param, value}` pins any scalar param via the
engine-side `ParamOverrides` table consulted in `tick()` (zero rebuild,
`value: null` clears, `drivers` telemetry flags `overridden`). Carry-forward
on regenerate is inherent: the table is keyed (binding id, param name) and
lives outside the plan, so overrides survive every rebuild where the name
still resolves to a scalar. The driver rack's numeric rows now tune through
this path; scene.json is never written implicitly. Audio conditioning stays
server-side (settled) — effect strength is just a scalar input on the
effect, now live-tweakable via the override path.

**Still open (forward concern, v1 §8.11): shared WGSL utilities.** Effects
past v1 will want shared noise/SDF/colour-space/palette helpers instead of
every file re-implementing `permute()`/`hsv2rgb()`. WGSL has no native
`#include`; the fix is a small `#import`-style text preprocessor run before
`naga`. The compile entry point is still the single `gpu::compose_shader` /
`build_effect_pipeline` pair, so wedging it in stays a one-layer change.

### 5.6 Design/Live legs + Promote/Pull (the two-deck architecture)

The single biggest structural feature (spec: "two legs, one deck"). Sketch:

- Engine holds **two `PassPlan` slots**: `live` (drives the projector) and
  `design` (renders only to an offscreen composite consumed by a second
  preview channel). All authoring RPCs (`scene.load`, `effect.upsert`,
  param edits) target `design` by default; `live` is immutable except via:
  - `promote {fade_ms}` — crossfade the projector output from the live
    composite to the design composite (two composites already exist as
    textures; the final pass lerps), then **copy** design's scene into the
    live slot. **Promote is a copy, not a swap:** design keeps its content
    (now identical to live), so the performer keeps iterating on the same
    idea — the scratchpad never snaps back to the pre-promote look, and
    "promote, push further, promote again" needs no manual `pull` between
    rounds.
  - `pull` — hard-copy live's scene back into design (the explicit reverse).
- **Knob/override carry across legs:** apply the §5.5 carry-forward rule
  (param name + type still match → tuned value survives) on both promote and
  pull — a leg crossing must never make a knob jump under the operator's
  finger (spec §4).
- Both plans tick from the same transport/audio so a promote lands in sync.
- The AI/agent only ever sees the design leg (spec §5: drafts never reach
  the crowd). The file-watcher path binds to design too; headless-only runs
  (no WS) collapse to a single live leg exactly as today.
- Prerequisite: diff-based plan rebuild (reference §4 weakness 1) is *not*
  required, but memory doubles per plan — acceptable.

This also creates the natural home for the **auto-pilot playlist**
(spec §13): a queue of saved scenes promoted on a bar/minute schedule with
per-entry dwell, skipping entries that fail to build.

**Shader pre-flight probe (operator requirement, 2026-07-11).** The design
leg shares the process and GPU with the live leg, so a pathological
AI-written shader entering *design* still stalls the *live* output. Gate it:
on any `effect.upsert` / watcher reload targeting design, the engine (a)
naga-compiles, (b) renders the effect ~60 frames to a scratch offscreen
target at reduced resolution (half/quarter of pack res), (c) measures p95
frame time and scale-checks it against the full-res budget, and only then
swaps the pipeline into the design plan. The probe result —
`{compiled, p95_ms, thumbnail}` — goes out on `hot_reload` telemetry and
the §5.11 status file, so the authoring agent can self-correct on
*performance and look*, not just compile errors, before the operator ever
sees the draft. Runs on a single M2 GPU: same device, probe frames
sequenced between live frames (a few ms/frame of budget during the burst).
Known residual risk: an in-process probe cannot contain a shader that hangs
the GPU device — that class is covered by the §5.11 recovery contract
(relaunch + restore), and a separate probe *process* with its own Metal
device stays available as optional hardening if hangs show up in practice.
Defense order: pre-flight keeps bad shaders out of both legs; the §5.11
probation window catches what slips through at full res.

**Design-leg autosave.** Once design state lives behind RPC edits (not just
the scene file), a crash mid-design must not eat the draft: debounce-write
the design leg's scene to `<scene_dir>/.wzrd/design.scene.json` (~every few
seconds / on every applied edit), and offer it for restore on next boot.
Together with the §5.3 sidecar this is the Resolume-style "reload and
continue in under a minute" contract — see §5.11.

**Design input:** the single-process-collapse call is made *here*, as part
of this design — the design leg is only ever visible through its preview,
which is what makes the preview ceiling load-bearing. Operator inputs
settled 2026-07-11 (see
[single-process-collapse.md](single-process-collapse.md) §6 + §8): the
design preview must be **decent enough to judge effect quality** (not
necessarily lossless/full-res), and a **rare ~20 s full blackout is
acceptable** provided state restore is total — which shifts the §8
recommendation toward collapse, pending the Step-2 runtime spikes.

### 5.7 Layer object + intensity (carried over from v1 review #2/#9)

Introduce an explicit `Layer` concept between scene and bindings: per-region
**intensity** (how hard its bindings light it), **mute**, and z-override,
addressable as `layers: { "trunk": { "intensity": 0.8 } }` in scene.json and
tweakable via the live param path (§5.5). The deck UI in Perform then lists
layers, not raw bindings — matching how the performer thinks (spec: "the
deck reads in surface-language").

### 5.8 Preview & render-loop upgrades

- **Decision gate (updated 2026-07-11):** the single-process collapse call
  ([single-process-collapse.md](single-process-collapse.md)) is made as part
  of the §5.6 design. Step 1 (Core/Host split) landed 2026-07-10, static
  spikes answered; the crash-recovery precondition was **relaxed** by the
  operator's accepted recovery contract (§5.11: rare ~20 s blackout OK if
  restore is total), so only the Step-2 runtime spikes gate the commit and
  the recommendation now leans collapse. Until the call is made, **hold**
  the binary-frames and design-leg `PreviewSampler` items below
  (throwaway under collapse); demand-gated capture is safe anytime
  (survives either outcome as the remote-thumbnail path).
- Binary preview frames over WS (drop base64+JSON), larger preview when the
  Perform route is focused — **only if collapse is rejected**. Note the
  operator's design-preview bar is "decent enough to judge effect quality,"
  not lossless — so this path (e.g. ~half-res at 30 fps) is a genuinely
  sufficient fallback, not a compromise that fails the spec.
- Design-leg preview channel (second `PreviewSampler` on the design
  composite, required by §5.6) — **only if collapse is rejected**; under
  collapse the design leg gets a native preview surface instead.
- Capture gated on subscriber demand — either way.
- Optional: `WaitUntil` scheduling; GPU timestamp queries into
  `frame_stats`; adaptive frame cap (drop to 30 Hz when no audio + no
  clock-driven bindings are active).

### 5.9 Video layers (v1 Phase 5, unchanged in spirit)

Two paths: **HAP/HAP-Q** for many concurrent 1080p layers (disk-bandwidth-
bound; FFI to hap-cpp or a small Rust port — the fallback if FFI hurts is a
compute-shader DXT decoder), and **H.264/HEVC hardware decode** (VideoToolbox
/ NVDEC / VAAPI via `ffmpeg-next` + wgpu interop) for single-stream content.
ProRes out of scope; software decode "won't hit 60 Hz for >1 stream."

The load-bearing invariant (v1 §3.8 has the full design — it remains correct):
each stream owns a **ring of N mapped `wgpu::Buffer` staging slots**
(N≈3, allocated at stream-open, recycled). A **decode thread** does all FFI +
disk I/O and pushes "slot ready" over a lock-free channel; the render thread's
only contact with video is one `copy_buffer_to_texture` from the freshest
ready slot. If decode falls behind, reuse the last good slot — a stutter,
never a frame-stall. Entry point: a `videoClip` effect whose descriptor
declares an `image` input bound to a `VideoSource`. Watch the HMR/cleanup
failure mode (reference §4 safety/scaling risks): staging slots from a
closed stream must drop.

### 5.10 MCP wrapper (v1 Phase 7)

A thin MCP server proxying the WS method table (§2.3) plus a
`scene.edit({instruction})` tool that round-trips natural language into
scene.json + effects/*.wgsl via Claude. The engine surface is already
agent-shaped; the wrapper adds no new engine code. With §5.6 landed, the
agent gets `design`-leg-only access for free.

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

- Tauri shell supervises the engine child: crash → respawn with the same
  scene within seconds, log the event, sticky `connectivity` reflects it.
  (Under collapse this bullet is replaced by relaunch-with-restore per the
  contract above.)
- Engine startup is already last-good-tolerant (bad scene → previous plan);
  extend to "scene fails at boot → black composite + hot-reload watch"
  instead of exit.
- Headless autostart recipe (launchd/systemd) documented for installations.
- ~~Slider/master state snapshot on SIGTERM~~ — **done** (landed with §5.3,
  2026-07-11: `signal-hook` flag → sidecar snapshot → graceful exit).
- **Post-swap probation window.** `naga` proves a shader *compiles*; a
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

Calibration UI (4-corner drag → homography written to the session sidecar
§5.3, <1 min recovery — spec §8; plus the **re-shoot recovery path** from
v1 §3.9: when the projector/surface is bumped, run the offline `align` step
on a new capture against the original reference photo to get a single
homography update — no re-segmentation);
layer deck panel in Perform (per §5.7); virtualized log list; driver-rack
grouping/filter chips; scene save-as + scene chooser grid; region renaming +
`identity.json` editing in Prepare (→ §5.13); AI co-author chat panel (with §5.10).

### 5.13 Identity sidecar: group authoring + region renaming (small sprint)

Bindings already resolve `select: { group }` (D7), and the Binding Inspector
already offers a group dropdown — but nothing in the UI can *create* a group
or assign layers, so the dropdown is empty unless `wzrd.layerpack` authored
groups offline. The fix is the first slice of the `identity.json` sidecar
(§2.2): human-authored, pack-adjacent metadata overlaid at load time, keeping
`pack.json` machine-authored and "pack ids stable, period." Groups are a
property of the *surface*, not a performance, so they belong here — **not** in
scene.json (per-scene) and **not** by rewriting the pack.

Scope (deliberately thin):
- **Engine:** load `<pack_dir>/identity.json` if present; merge its `groups`
  over `pack.groups` before serving `pack.info`. One new queued RPC
  `identity.setGroups {groups}` writes the sidecar and re-emits `pack.info`.
  Same slot later carries region renames (the §5.12 backlog item — build the
  reader/writer once, add a `labels` map when renaming lands).
- **UI:** multi-select on the Surface canvas (⌘/⇧-click accumulates a
  selection; already have pixel-accurate picking) → "New group from selection"
  → commits via `identity.setGroups`. Editing membership = re-select + save.
  Distinct from `sceneCommit.ts` — this writes the sidecar, not the scene.

Non-goals: no group nesting, no per-group colour/metadata beyond membership,
no offline `wzrd.layerpack` group-reconciliation (that rides the broader
identity-table re-import work in §2.2).
