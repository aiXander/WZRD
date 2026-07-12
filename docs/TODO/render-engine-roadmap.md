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
> dropped, 2026-07-11). The single-process collapse (Steps 2–3) **LANDED
> 2026-07-12** — both runtime spikes passed; current topology + residue in
> [../reference/render-engine.md](../reference/render-engine.md) §1/§1b.
> **Next up: §5.6 (two-deck) — all gating operator decisions are now made
> (2026-07-12); the section is execution-ready with an ordered build
> sequence.** Its design-leg preview builds on the native preview surface
> that now exists. Hard prerequisite before the collapsed build plays a live
> show: the §5.6 shader pre-flight probe (step 4 of the sequence).

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

The single biggest structural feature (spec: "two legs, one deck"). All the
gating operator decisions are now made (2026-07-12) — this section is
**execution-ready**; the ordered build sequence is at the end. A design-review
pass (2026-07-12) folded in the amendments marked below: pointer-swap promote,
cross-leg pipeline-cache guard, probe calibration + pessimistic drivers,
demand-gated design rendering, the blanket leg rule, and promote re-entrancy.

- Engine holds **two `PassPlan` slots**: `live` (drives the projector) and
  `design` (renders only to its own offscreen composite). Each leg owns its
  own composite target (memory doubles — acceptable). All authoring RPCs
  (`scene.load`, `effect.upsert`, param edits) target `design` by default;
  `live` is immutable except via:
  - `promote {fade_ms, quantize}` — crossfade the projector output from the
    live composite to the design composite (two composites already exist as
    textures; the final pass lerps), then **copy** design's scene into the
    live slot. **Implementation (amendment 2026-07-12): the copy is a
    plan-pointer swap, not a live-slot rebuild.** On ramp completion, live
    adopts design's already-built `PassPlan` (zero rebuild, zero hitch on
    the projector leg); design's plan is then rebuilt from the same scene
    JSON in the background — the expensive part lands on the leg nobody is
    projecting. Safe because picks are stateless (§5.2: a pure hash of
    binding id + transport cycle, so the rebuilt design plan picks
    identically) and pipelines cache-hit (no recompiles).
    **Quantize (operator decision 2026-07-12):**
    `quantize: "bar" | "now"`, default `"bar"` — the fade *starts* on the
    next bar boundary so the visual change lands on a downbeat (both legs
    tick the same transport, so the boundary is well-defined); `"now"`
    starts immediately. Surface it as a toggle next to the promote control.
    **Promote is semantically a copy, not an exchange:** design keeps its
    content (now identical to live), so the performer keeps iterating on the
    same idea — the scratchpad never snaps back to the pre-promote look, and
    "promote, push further, promote again" needs no manual `pull` between
    rounds. (The pointer swap above is invisible to the operator — both legs
    end up holding the same content.)
  - `pull` — hard-copy live's scene back into design (the explicit reverse).
  - **Re-entrancy (amendment 2026-07-12), part of the RPC contract:** a
    *pending* bar-quantized promote (fade not yet started) is **replaced**
    by a newer `promote`; while a fade is actively ramping, `promote` and
    `pull` are **rejected** with an error until the ramp completes. The
    auto-pilot playlist will hit these paths programmatically — the rule
    must be deterministic, not UI polish.
- **Shared `pipeline_cache` needs a cross-leg eviction guard (amendment
  2026-07-12).** Pipelines are cached per effect key in `gpu.pipeline_cache`;
  `effect.remove` evicts keys, and `PassPlan` passes *silently skip* on a
  cache miss (`compositor.rs`). Safe today with one plan — but with two
  legs, an eviction driven by a *design* edit can remove a pipeline the
  *live* plan still references, and live layers stop drawing mid-show with
  no error. Eviction must consult both plans' live `pipeline_key` sets:
  evict only when unreferenced by either leg. Lands in build step 1.
- **Design-leg rendering is demand-gated (amendment 2026-07-12); ticking is
  not.** Both legs tick every frame (cheap; keeps transport/pick state
  coherent), but design's render passes only run when something consumes
  the result: the preview toggle is on DESIGN, ≥1 WS `preview` subscriber
  exists (the `Bus::subscriber_count` gate the JPEG sampler already uses),
  or a promote fade is in flight (both composites must be current while the
  final pass lerps). Otherwise the design leg costs zero GPU — the probe
  exists to protect live headroom; the architecture shouldn't spend half of
  it by default.
- **Knob/override carry across legs — resolved to "no copy needed"
  (operator decision 2026-07-12):** the §5.5 `SliderBank` + `ParamOverrides`
  tables stay **single, shared, venue-level** state (keyed by slider name /
  by binding id + param name), consulted by *both* legs' `tick()`. Operator
  knobs aren't scene content, so a tuned value is automatically identical on
  both legs — carry-forward across promote/pull falls out for free, and a
  knob can never jump under the operator's finger (spec §4). No per-leg
  override table, no copy step on promote/pull. (Edge case, accepted: if
  design and live momentarily hold a same-id binding with *different*
  effects mid-edit, the shared override applies to both — which is exactly
  the intended carry-forward, not a bug.)
- Both plans tick from the **same shared transport/audio bus** (already how
  the driver bus is built) so a bar-quantized promote lands in sync.
- The AI/agent only ever sees the design leg (spec §5: drafts never reach
  the crowd). **Blanket leg rule (amendment 2026-07-12): every authoring
  and observability surface follows design** — the file watcher, the JPEG
  `preview` channel, the Prepare canvas underlay, `scene.getState`, the
  `drivers` channel, and `frame_stats` pass counts. Only the projector and
  the preview toggle's LIVE position show live. `scene.getState` gains an
  optional `leg: "design" | "live"` param (default `design`) so `pull` is
  verifiable over RPC and the §5.10 MCP wrapper stays unambiguous.
  Divergence between legs is transient (promote/pull re-converge them), so
  the rule costs nothing in practice. Headless-only runs (no WS) collapse
  to a single live leg exactly as today — watcher binds live.
- Prerequisite: diff-based plan rebuild (reference §4 weakness 1) is *not*
  required.

This also creates the natural home for the **auto-pilot playlist**
(spec §13): a queue of saved scenes promoted on a bar/minute schedule with
per-entry dwell, skipping entries that fail to build. **Deferred (operator
decision 2026-07-12): explicitly out of §5.6 scope** — it ships as a thin
follow-up layer once promote is proven solid live. Design constraint on
§5.6 so the deferral stays cheap: the playlist must need no new engine
verbs — a scheduler composing existing calls (`scene.load` → design, wait
for build/probe OK, `promote {quantize:"bar"}`, dwell, repeat, skip on
failure).

**Shader pre-flight probe — a load *predictor* (operator decision
2026-07-12).** The design leg shares the process and GPU with the live leg,
so a pathological AI-written shader entering *design* still stalls the *live*
output. The probe answers one question before a shader is allowed in: *how
much load will this put on the GPU at full resolution and full fps — will it
drown the framerate?* The gate covers **any new pipeline entering design,
whatever the entry path** (amendment 2026-07-12) — `effect.upsert`, watcher
reloads, *and* `scene.load` (a scene can pull in new effects; the auto-pilot's
"wait for build/probe OK" step already assumes this). The engine
(a) naga-compiles, (b) renders the effect ~60 frames to a scratch offscreen
target at **half** pack resolution, (c) measures p95 frame time and
**scales it up to a predicted full-res p95** (fragment cost scales with
pixel count).

**Scaling must calibrate out fixed overhead (amendment 2026-07-12).** Naive
`predicted ≈ measured × full_px/probe_px` assumes the measured time is all
fragment work; at reduced res a probe frame is dominated by fixed per-frame
cost (encoder submit, scheduling, an unsaturated GPU), and multiplying that
overhead by the pixel ratio predicts red for shaders that are actually fine.
Fix: probe a trivial shader once at boot to measure the fixed floor, then
`predicted = overhead + (measured − overhead) × full_px/probe_px`. Half res
(4× ratio) keeps the correction small — quarter res (16×) would amplify any
calibration error, which is why it was dropped.

**Probe with pessimistic driver values (amendment 2026-07-12).** A shader
whose cost scales with an audio-driven param (iteration counts, ray steps)
probes cheap in silence and blows the budget on the first drop. During probe
frames, drivers evaluate pessimistically: `audio.*` pinned to 1.0, scalar
params at their descriptor `max` where one exists (current value otherwise).
The probe answers "worst case at this venue", not "cost at this instant".

**Three-band verdict against two operator-set thresholds A < B** (predicted
full-res p95, in ms; budget = 16.6 ms @ 60 Hz):

- **predicted < A → green** — passes clean into the design plan.
- **A ≤ predicted ≤ B → yellow** ("heavy but doable") — **still swaps into
  design** so the operator/agent can iterate on the look, but flagged; the
  §5.11 full-res probation window is the live safety net if it's later
  promoted.
- **predicted > B → red** — refused entry to design entirely (same as a
  compile error); the previous pipeline stays. This is the only hard gate.

So the probe is **advisory in the middle band, hard gate above B** — not a
binary pass/fail. **A and B live in the session sidecar** (§2.5 / §5.3),
not scene.json: they describe *this GPU + projector*, which is venue-level
state alongside calibration and masters, so they persist per-venue and are
set from the GUI. Defaults ≈ A 8 ms / B 14 ms, tunable. New RPC
`probe.setThresholds {a_ms, b_ms}` (inline, sidecar-persisted) +
`probe.getThresholds`.

The probe result — `{compiled, predicted_p95_ms, band, thumbnail}` — goes
out on `hot_reload` telemetry and the §5.11 status file, so the authoring
agent self-corrects on *performance and look*, not just compile errors,
before the operator ever sees the draft. Runs on a single M2 GPU: same
device, probe frames sequenced between live frames (a few ms/frame of budget
during the burst). Known residual risk: an in-process probe cannot contain a
shader that *hangs* the GPU device — that class is covered by the §5.11
recovery contract (relaunch + restore), and a separate probe *process* with
its own Metal device stays available as optional hardening if hangs show up
in practice. Defense order: pre-flight keeps red shaders out of both legs;
the §5.11 probation window catches yellow shaders that blow the budget at
full res once promoted.

**Design-leg autosave.** Once design state lives behind RPC edits (not just
the scene file), a crash mid-design must not eat the draft: debounce-write
the design leg's scene to `<scene_dir>/.wzrd/design.scene.json` (~every few
seconds / on every applied edit), and offer it for restore on next boot.
Together with the §5.3 sidecar this is the Resolume-style "reload and
continue in under a minute" contract — see §5.11.

**Preview topology — RESOLVED (operator decision 2026-07-12):
single window, source toggle (Resolume-style).** The collapse already
landed the native preview surface (`gpu::PreviewTarget` blitting a composite
onto a child window — reference §1b). The two-deck build keeps **one**
preview window in the GUI with a **LIVE ⇄ DESIGN toggle** that selects
*which composite the single `PreviewTarget` samples* — not a second window,
not a second pane. Consequences that fall out of the toggle:
- **LIVE** shows the live composite with **real masters applied**
  (brightness/saturation as the crowd sees them) — but *not* literally "the
  presented frame" (swapchain frames aren't sampled; `PreviewTarget` re-runs
  the homography pipeline against a composite) and **not** the calibration
  warp: the warp only looks right on the physical surface — in a flat
  preview window it's just keystone distortion. Mechanically: live composite
  + non-neutral master uniforms + identity homography (amendment 2026-07-12).
- **DESIGN** shows the scratch composite un-mastered (the §5.4 convention the
  current preview already follows), so preview colour = raw effect colour.
- Mechanically: the design leg gets its own composite; `PreviewTarget` gains
  a one-line source select (which composite view its bind group samples) and
  its master uniforms flip with the toggle (real masters on LIVE, neutral on
  DESIGN), re-bound when the toggle flips. The delivery mechanism
  (child-window blit, demand-gated readback) is unchanged. Both runtime
  spikes passed, so the subprocess fallback (§5.8) is retired.

---

### 5.6 execution order (engine-first, headless-verifiable)

Each step is independently landable and testable over RPC before any UI
exists. Ship in this order:

1. **Two composites + two `PassPlan` slots.** Move composite ownership out of
   the single `GpuContext.composite_*` into per-leg targets; `Core` holds
   `live` + `design` plans. Includes the **cross-leg `pipeline_cache`
   eviction guard** (evict only when unreferenced by either leg) — the
   silent-skip-on-cache-miss failure mode must be closed before two plans
   exist. Headless still constructs only `live`. Verify: headless output
   byte-identical to today (design leg dormant).
2. **Authoring RPCs target `design`; `live` frozen.** Route `scene.load`,
   `effect.*`, param edits, and the file watcher at the design plan, per the
   blanket leg rule (reads/telemetry follow design; `scene.getState` gains
   the `leg` param). Design's render passes are **demand-gated** from the
   start (preview toggle / WS subscriber / fade in flight). Confirm over WS
   that editing never disturbs the live composite and an idle design leg
   adds zero GPU work. Shared `SliderBank`/`ParamOverrides` already feed
   both `tick()` calls — verify a knob reads identically on both legs.
3. **`promote {fade_ms, quantize}` + `pull`.** Final pass samples both
   composites + a `mix` uniform ramped over `fade_ms`; `quantize:"bar"`
   defers the ramp start to the next transport bar boundary, `"now"` starts
   immediately. On ramp completion, live adopts design's built plan by
   **pointer swap** and design rebuilds in the background (promote) /
   live→design scene copy (pull). Implement the re-entrancy rule (pending
   quantized promote replaced; mid-ramp promote/pull rejected). Verify a
   bar-quantized promote lands on the downbeat with no frame hitch and the
   knobs don't jump.
4. **Pre-flight probe + `probe.set/getThresholds`.** Half-res 60-frame
   render with overhead-calibrated scaling and pessimistic driver values
   (see the probe amendments above) → predicted full-res p95 → three-band
   verdict; gate **every new pipeline entering design** (upsert, watcher,
   `scene.load`) on red, flag on yellow; emit `{compiled, predicted_p95_ms,
   band, thumbnail}` on `hot_reload`. Thresholds in the session sidecar.
   This is the **hard prerequisite before any live show** — land it before
   UI go-live.
5. **Design-leg autosave** to `<scene_dir>/.wzrd/design.scene.json` (see the
   autosave paragraph above) + restore-on-boot offer.
6. **UI last:** the LIVE⇄DESIGN preview toggle, the Promote control with the
   bar/now quantize toggle + `pull`, and a threshold-A/B settings control.

### 5.7 Layer object + intensity (carried over from v1 review #2/#9)

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

- ~~Tauri shell supervises the engine child~~ — superseded by the **landed**
  collapse (2026-07-12): the mechanism is **relaunch-with-restore** per the
  contract above. Spike results: the render loop already runs under
  `catch_unwind` (a panic — incl. wgpu device-loss fatal errors — kills
  the engine thread, not the webview; `engine:status` reports it), state
  files stayed byte-identical through both crash modes, and measured
  relaunch-to-light was ~160–290 ms — the ≤20 s contract has huge margin.
  Still open here: an in-app "restart engine" button (today: quit +
  relaunch), and the post-crash wart that SIGTERM is a no-op (reference
  §1b) — both fold into this hardening pass.
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
