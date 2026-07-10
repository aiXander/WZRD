# Render engine — structural roadmap

> The forward plan for `render-core/` + `wzrd-app/`. Current system state,
> contracts, and invariants live in
> [../reference/render-engine.md](../reference/render-engine.md) — references
> of the form §1–§4, §6, §7 point there. Section numbers here keep the
> **§5.x** form so cross-references in the reference doc stay valid.
>
> Ordered by leverage toward `../reference/user_design_spec.md`. Each item is
> scoped so an agent can pick it up standalone. **General rule: engine first,
> headless-verifiable, UI second.** §5.1–§5.3 are small and show-critical —
> the pre-first-show path; §5.6 is the big structural one.
>
> Related open decision (uncommitted): single-process collapse for a
> lossless preview — [single-process-collapse.md](single-process-collapse.md).

### 5.1 Live transport: music-locked BPM + phase resync (next)

Smallest item, most show-critical. Today `transport.bpm` is a static number
typed into `scene.json` before the music started; every `clock.*`-driven
binding is synced to that guess, and the only live correction path is a
scene edit (full plan rebuild). Meanwhile the audio server already tracks
tempo and emits `/audio/bpm ,f` per block on the same OSC feed as
`/audio/lmh` (the engine currently ignores it — `osc.rs` drops `/audio/bpm`
on the floor).

- **BPM is an input, never an estimate.** `osc.rs` decodes `/audio/bpm` into
  `AudioFeatures` (one more atomic); `Transport` follows it while the feed
  is fresh (same freshness window as the OSC pill) and falls back to the
  scene's `transport.bpm` when it isn't (`--no-osc`, server down/stale).
  The engine must never compute, estimate, or tap tempo itself — the audio
  server owns tempo (§7), including any smoothing or plausibility gating of
  the BPM stream.
- **Invariant — the transport integrates phase.** `phase += bpm/60 · dt` per
  frame, never `phase = f(bpm, wall_time)`. With BPM continuously varying at
  runtime, deriving phase from absolute time would make every `clock.*`
  driver jump on each BPM update. (The §5.4 masters "speed" control
  multiplies this same integration step, for the same reason.)
- **`transport.resync`** — inline RPC on the zero-rebuild path (like
  `param.set`): "the downbeat is *now*", zeroing bar/beat phase at call
  time. A correct BPM at the wrong phase still fires the every-4-bars bloom
  off-beat; this is the one-button fix. Companion inline read
  `transport.state` → `{bpm, source: "live"|"fallback", bar_phase,
  beat_phase}`.
- Telemetry: fold `bpm` + `source` into the existing `audio` channel payload
  (emitter + UI consumer land together — §3.3 lesson). UI: BPM readout +
  SYNC button in the status strip / Perform.

### 5.2 Per-layer variation + `pick` selectors (the organic-look gap)

When a binding selects 20 leaf clusters, all 20 passes currently get
byte-identical uniforms — every leaf animates in perfect lockstep, which
reads mechanical, and the tree scene's "one random leaf blooms every 4 bars"
is inexpressible. The workaround (author 20 near-duplicate bindings) defeats
the selector model. Two additions:

- **Per-layer identity in the WGSL contract.** Extend `LayerParamsGpu` (+
  the prelude mirror) with `layer_seed` (stable hash of the layer *id* —
  stable across re-segmentation because ids are, D7), `layer_index` /
  `layer_count` within the binding's resolved selection, `centroid_uv`, and
  `bbox` (uv-space). Surface them to user WGSL as prelude accessors
  documented in §2.4. One `hueCycle` binding plus `phase += layer_seed` then
  desynchronizes 20 leaves for free; a bloom radiates from `centroid_uv`.
  Pass-plan build work only — the uniform struct has padding headroom, no
  new bind groups.
- **`pick` selectors (reinstated from v1).** The selector grammar grows
  `{ "tag": "leaves", "pick": { "mode": "random_each" | "random_static",
  "rate": { "driver": "clock.bars", "n": 4 } } }`. `random_each` re-picks
  one member of the resolved set each time the rate driver wraps;
  `random_static` picks once at scene load. Seed the RNG from the transport
  bar counter so runs are deterministic and the design-leg preview (§5.6)
  picks the same layer its promote will.

Litmus: with this landed, the selection semantics of all three target scenes
are expressible without per-layer binding duplication.

### 5.3 Operator-owned state: the session sidecar

`scene.json` is the AI-writable composition, but venue-physical and
performance state currently leaks into it (`projectorCalibration`) or
evaporates on restart (`SliderBank` — reference §4 weakness 5). The spec's
AI "emits the complete effect each turn"; the day an agent rewrite drops the
calibration field, projector alignment dies mid-show — exactly the failure
the masters design already guards against. Calibration is also *per-venue*
while a scene is *per-artwork*: replaying last summer's scene at a new venue
currently drags a stale homography along (spec §12).

One sidecar — `session.json` next to the scene, engine-written, never on the
AI's editing surface:

```jsonc
{
  "version": 1,
  "projectorCalibration": null,      // 3×3 row-major or null — moved out of scene.json
  "masters": { "brightness": 1.0, "speed": 1.0, "saturation": 1.0, "audioListen": 1.0 },
  "params": { "flash_base": 0.35 }   // SliderBank + §5.5 override snapshot
}
```

- **Read precedence:** sidecar first; `projectorCalibration` in `scene.json`
  stays readable as a deprecated fallback (warn on use, never written back).
- **Write policy:** engine-owned. Written on explicit save, debounced after
  master/knob changes, and on SIGTERM — which gives §5.11 its power-blink
  snapshot for free. `scene.load` and the file watcher never touch it; no
  RPC on the design/authoring surface writes it.
- **Scope rule:** `scene.json` = what the surface *does* (AI + human
  authored); `session.json` = how *this venue, this night* is set (operator
  only).
- The eventual "show file" (playlist + scenes + session) composes these;
  don't build the umbrella before the auto-pilot playlist (§5.6) exists.

Resolves reference §4 weakness 5: knobs persist via the sidecar, and "write
knobs back into scene.json" stays a separate explicit authoring action,
never implicit.

### 5.4 Masters row (engine-level, operator-owned)

A small set of global controls the AI can never touch (spec: "masters are
mine alone"): overall **brightness**, **speed** (global time scale),
**saturation**, **audio-listen** (scales every `audio.*` driver toward 0),
and later **crossfade time**. Implementation: engine-owned `Masters` struct,
applied (a) in the final homography pass for brightness/saturation, (b) as
multipliers in `Transport`/`AudioFeatures` reads for speed/listen — speed
multiplies the §5.1 phase-integration step (`phase += speed · bpm/60 · dt`),
never a scaled absolute clock, so a speed change bends time instead of
jumping it. RPC: `master.set {name, value}` + a `masters` telemetry channel.
Persisted in the session sidecar (§5.3), **not** inside `scene.json` — the
AI edits scene.json; it must not be able to reach the masters. UI: an
always-visible row in Perform.

### 5.5 Params become first-class: descriptor-driven knobs

The design spec's core loop — "the AI grows the dials, I play them" —
needs parameters that are addressable and typed without recompiling:

- **Extend effect descriptors** with UI metadata per input: `min`, `max`,
  `step`, `unit`, `widget` (slider/knob/toggle/palette). Built-ins get
  descriptors too. `pack.info`-style `effect.describe(name)` RPC (or fold
  into `pack.info`'s sibling `scene.describe`) so the UI/agent can render
  controls without guessing ranges.
- **Address params as `binding.param`, not just global slider names.**
  `param.set { binding, param, value }` should override *any* scalar param
  (const or driver output scaling), stored in an engine-side override table
  consulted by `tick()` — same zero-rebuild property as the slider bank.
  Keep name-keyed `ui.slider` for scene-authored shared knobs.
- **Value semantics on regenerate:** when the AI rewrites an effect, carry
  forward user-tuned values wherever the param name + type still match
  (spec §4: "a knob never jumps under my finger; my hand-tuning carries
  forward").

**Audio conditioning stays server-side (settled).** Every `audio.*` value
arrives already smoothed and min/max-bounded by the audio server; the engine
adds no attack/release, normalization, or per-param signal conditioning
(§7). The one audio-tuning knob that belongs engine-side is **effect
strength** — how hard a given audio value drives a given visual parameter.
That's an ordinary scalar input on the effect itself (author effects as
`base + strength · audio_value`, with `strength` declared in the
descriptor), which this item's override path makes live-tweakable — no
special driver wrapper, no schema change.

Related forward concern (v1 §8.11): effects past v1 will want **shared WGSL
utilities** (noise, SDF, colour-space, palette helpers) instead of every file
re-implementing `permute()`/`hsv2rgb()`. WGSL has no native `#include`; the
fix is a small `#import`-style text preprocessor run before `naga`. Keep the
effect loader's compile entry point a single function so wedging this in is a
one-layer change, not a rewrite.

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

**Design input:** the single-process-collapse call is made *here*, as part
of this design — the design leg is only ever visible through its preview,
which is what makes the preview ceiling load-bearing. See
[single-process-collapse.md](single-process-collapse.md) §3.3 + §8 and the
§5.8 decision gate.

### 5.7 Layer object + intensity (carried over from v1 review #2/#9)

Introduce an explicit `Layer` concept between scene and bindings: per-region
**intensity** (how hard its bindings light it), **mute**, and z-override,
addressable as `layers: { "trunk": { "intensity": 0.8 } }` in scene.json and
tweakable via the live param path (§5.5). The deck UI in Perform then lists
layers, not raw bindings — matching how the performer thinks (spec: "the
deck reads in surface-language").

### 5.8 Preview & render-loop upgrades

- **Decision gate (2026-07-10):** the single-process collapse call
  ([single-process-collapse.md](single-process-collapse.md)) is made as part
  of the §5.6 design. Its Step 1 (Core/Host split) **landed 2026-07-10**
  and the static spikes are answered; only the runtime spikes + the
  crash-recovery precondition remain. Until the call is made, **hold**
  the binary-frames and design-leg `PreviewSampler` items below
  (throwaway under collapse); demand-gated capture is safe anytime
  (survives either outcome as the remote-thumbnail path).
- Binary preview frames over WS (drop base64+JSON), larger preview when the
  Perform route is focused — **only if collapse is rejected**.
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

- Tauri shell supervises the engine child: crash → respawn with the same
  scene within seconds, log the event, sticky `connectivity` reflects it.
- Engine startup is already last-good-tolerant (bad scene → previous plan);
  extend to "scene fails at boot → black composite + hot-reload watch"
  instead of exit.
- Headless autostart recipe (launchd/systemd) documented for installations.
- Slider/master state snapshot on SIGTERM — implemented as the session
  sidecar write (§5.3) — so a power blink comes back close to where it was.
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
`identity.json` editing in Prepare; AI co-author chat panel (with §5.10).
