# Audio refactor plan — replace in-process capture with OSC ingest

Status: **implemented (2026-05-22).** All file-by-file changes in §5 are landed in `render-core/` (`src/audio.rs` removed; `src/osc.rs` shipped; `Cargo.toml` deps swapped; `AudioState` → `AudioFeatures`; CLI flags `--no-osc` / `--osc-addr` in `main.rs`; `state.audio_rms` dropped from the shader prelude; smoke-test scene patched). Kept around as the design-rationale paper trail referenced from `render_engine_architecture.md` §3.7, §3.10, and the status block. The "deferred" items in §10 are still open.

## 1. Goal

`render-core` should not capture or analyse audio. All audio features come from the
**Realtime Audio Feature Server**
(`/Users/xandersteenbrugge/Documents/GitHub/Realtime_PyAudio_FFT`) running on
localhost, emitting pre-processed features over OSC/UDP. `render-core` just
listens to that stream and exposes the values to drivers and the `FrameState`
uniform.

Architectural payoff:

- One audio pipeline across the whole stack (browser viz, TouchDesigner, VJ tool,
  custom receivers) — tune once, every consumer agrees.
- No mic permission prompt on macOS, no BlackHole/Loopback setup inside the
  engine binary, no DSP code maintained in two languages.
- The audio server already does what `audio.rs` does today, *better*: peak
  follower + soft noise gate + tanh compression, Schmitt-trigger onset with
  per-band sensitivity / refractory / slow-τ, BPM tracking — all live-tunable
  from the server's browser UI.

Non-goals: replacing MIDI/OSC drivers in general (those don't exist yet), or
ingesting the full FFT spectrum (deferred — see §10).

---

## 2. Current state — what gets deleted

Live today in `render-core/`:

- `src/audio.rs` (348 lines) — entire file. cpal input stream + Hann-windowed
  rFFT + 3-band energy + spectral-flux onset detector + atomic `AudioState`.
- `Cargo.toml`: `cpal = "0.15"`, `rustfft = "6.2"`.
- `main.rs`: `--no-audio` flag, `try_spawn`/`AudioCapture` plumbing, the
  `_audio_capture` field that keeps the stream alive.
- `README.md`: "Audio" section + BlackHole/Loopback paragraph.
- `render_engine_architecture.md`: §3.7 driver description, §8 open question #6
  ("Audio loopback on macOS without user setup"), Phase 3 audio bullet.

The `AudioState` type, the `AudioBand` enum, and the driver names (`audio.rms`,
`audio.band`, `audio.onset`) **survive** — they're the public API the scene
schema and effect WGSL already depend on. Only the *fill mechanism* changes.

---

## 3. Target architecture

```
┌───────────────────────────────────────────────┐
│  Realtime Audio Feature Server (Python)       │
│  - PortAudio capture                          │
│  - 3-band IIR + autoscale + onset + BPM       │
│  - OSC out, default 127.0.0.1:9000 (UDP)      │
└──────────────────┬────────────────────────────┘
                   │ /audio/lmh, /audio/onset/{low,mid,high},
                   │ /audio/meta, (/audio/bpm + /audio/fft ignored in v1)
                   ▼
┌───────────────────────────────────────────────┐
│  render-core (Rust)                           │
│  ┌─────────────────────┐                      │
│  │ osc.rs              │  UDP recv thread     │
│  │  - rosc decode      │  writes atomics      │
│  │  - dispatch by addr │                      │
│  └─────────┬───────────┘                      │
│            ▼                                  │
│  ┌─────────────────────┐                      │
│  │ AudioFeatures       │  shared Arc<…>       │
│  │  (was AudioState)   │  atomic snapshot     │
│  └─────────┬───────────┘                      │
│            ▼                                  │
│  drivers.rs ─ FrameState uniform ─ effects    │
└───────────────────────────────────────────────┘
```

The OSC thread is the *only* writer; the render thread is the *only* reader.
No FFT, no DSP, no cpal stream.

### 3.1 Connection lifecycle (auto-detect, auto-recover)

The engine and the audio server are independent processes. **Either may start,
stop, or restart at any time without the other knowing.** That has to be a
first-class property of `osc.rs`, not an afterthought.

Concrete behaviour:

- **Engine starts before server.** UDP bind succeeds on a free port whether
  anyone is sending to it or not. The recv thread spins on `recv_from` with a
  short read timeout; while no packets arrive, all atomics stay at their
  initial zero. Renderer keeps running. **No retry loop, no "wait for server"
  startup path — just a socket sitting there ready.**
- **Server starts later.** The very first OSC packet hits the already-bound
  socket and gets dispatched. Atomics start updating on the next packet.
  Effects bound to `audio.band(…)` / `audio.onset(…)` start reacting within
  one frame of the first matching message. **Zero handshake, no scene reload
  required.**
- **Server stops mid-run.** Packets stop arriving. Atomics freeze at their
  last value. Onset envelopes (already decaying off the last timestamp via
  `exp(-dt/τ)`) naturally fade to zero. Nothing in the engine notices or
  cares; the render loop keeps drawing.
- **Server restarts mid-run.** First post-restart packet just dispatches —
  same path as "server starts later." No reconnect logic, because there's no
  *connection* to reconnect (UDP is connectionless).
- **Server is on another machine.** Same code path; user passes
  `--osc-addr 0.0.0.0:9000` so we bind all interfaces, and the server's
  `osc.destinations[]` is pointed at this machine's IP.

The only state we need to track for the user's benefit is **packet freshness**
— "have I heard from the server recently?" — so we can log a clear status
transition. Implementation: `AudioFeatures::last_packet_ms` is bumped on
every dispatched message; a small watchdog inside the recv thread (or the
main loop's `about_to_wait`) logs one line on transition between fresh ↔
stale, with a 2-second hysteresis:

```
[info]  OSC: connected (first packet from 127.0.0.1:9000)
... silence for >2s ...
[warn]  OSC: stale (no packets for 2.1s) — is the audio server running?
... packets resume ...
[info]  OSC: connected (packets resumed)
```

One log per transition, never per packet. Nothing else changes — drivers and
effects don't observe connection state, they just read whatever values the
atomics hold.

**Crash isolation.** UDP bind failure (port already in use, permission
denied) logs a warning and drops the engine into the same "no audio" mode as
`--no-osc`. Render loop never blocks on the socket; the recv thread can
panic and the renderer keeps drawing zeros (though `rosc::decoder` parse
errors are non-fatal anyway and just get logged at `trace`).

---

## 4. OSC ↔ AudioFeatures mapping

The server's contract (from its README):

| OSC address          | Args                         | Rate          | Maps to                                                                 |
|----------------------|------------------------------|---------------|-------------------------------------------------------------------------|
| `/audio/lmh`         | `low:f mid:f high:f`         | ~187 Hz       | `AudioFeatures.band_{low,mid,high}` (atomic store, relaxed)             |
| `/audio/onset/low`   | `1:i`                        | rising edge   | `stamp_onset_ms(&onset_low_ms)` — same envelope semantics as today      |
| `/audio/onset/mid`   | `1:i`                        | rising edge   | `stamp_onset_ms(&onset_mid_ms)`                                         |
| `/audio/onset/high`  | `1:i`                        | rising edge   | `stamp_onset_ms(&onset_high_ms)`                                        |
| `/audio/meta`        | sr, blocksize, n_fft_bins, … | startup/edit  | logged once; sample rate stored for diagnostic; bumps `last_packet_ms`  |
| `/audio/bpm`         | `bpm:f`                      | ~187 Hz       | **ignored in v1** (§10). Decoded + dropped; bumps `last_packet_ms`.     |
| `/audio/fft`         | `n` floats                   | ~94 Hz        | **ignored in v1** (§10). Server defaults this off anyway.               |

### 4.1 Driver surface — expose what the server emits, no synthesised fields

The driver bus exposes exactly the OSC features that arrive — no derived /
synthesised values. The author or agent picks what to bind from raw inputs;
the engine doesn't pre-mix them into things like "broadband RMS" or "overall
energy."

Concrete consequence: **`audio.rms` is removed from the driver bus.** It had
no clean OSC equivalent and would have been a confusing synthesised fallback.
The `examples/phase3_smoke.scene.json` `flash.base` param is rewired to
`audio.band(low)` so the same pulse-with-baseline behaviour stays visible.

The v1 audio driver set is therefore:

- `audio.band(low | mid | high)` — `/audio/lmh` floats, already `~[0, 1]`.
- `audio.onset(band, decay)` — `/audio/onset/<band>` triggers, decayed on read
  by `exp(-dt/τ)` (unchanged from today).

No `audio.rms`, no `audio.bpm`, no `audio.fft`. If a scene needs "loudness,"
the author picks the band that matters; the LLM does the same.

### 4.2 Onset envelopes — read-side decay matches OSC trigger model

The server sends one int per onset (rising edge only). The current
`onset_envelope(band, decay_seconds)` already computes `exp(-dt / τ)` from a
stored millisecond timestamp — that's *exactly* the right shape for OSC
trigger arrival. We write `stamp_onset_ms(&slot)` on every
`/audio/onset/<band>` message instead of running spectral-flux detection
internally. **Zero shader changes for `onset_*`.** The `flash` effect, the
`audio.onset` driver, and user WGSL using `state.onset_*` keep working
byte-for-byte.

### 4.3 BPM — explicitly out of v1

`scene.transport.bpm` stays authoritative for `clock.bars(n)` /
`clock.beats(n)`. `/audio/bpm` is decoded so it doesn't log as "unknown
address" but the value is discarded. If/when auto-locking the transport to
the audio server is genuinely needed, the design path is clear (see §10) —
not worth the cognitive overhead of an `audio.bpm` driver or a
`followAudioBpm` flag right now.

---

## 5. File-by-file changes

### 5.1 `Cargo.toml`

```diff
-cpal = "0.15"
-rustfft = "6.2"
+rosc = "0.10"
```

`rosc` (the OSC parser crate, ~maintained) decodes packets; the UDP socket is
plain `std::net::UdpSocket`. No async runtime needed for a single recv loop on
its own thread. Roughly +1 dep, -2 deps.

### 5.2 Delete `src/audio.rs` entirely

Replace with `src/osc.rs`. Keep `AudioFeatures` (the renamed `AudioState`) and
`AudioBand` co-located in the new file — they're the only types that survive.

Proposed layout for `src/osc.rs` (~150 lines, vs 348 today):

```rust
//! OSC receiver — pulls per-frame audio features from the standalone
//! Realtime Audio Feature Server (Python, separate process). See
//! /Users/xandersteenbrugge/Documents/GitHub/Realtime_PyAudio_FFT/README.md
//! for the wire contract. We bind a UDP socket on the configured port,
//! decode rosc packets on a dedicated thread, and write into AudioFeatures
//! via relaxed atomic stores. The render thread never blocks.
//!
//! Lifecycle: socket is bound at startup whether or not the server is
//! running. Packets flow when the server is up; atomics freeze when it's
//! down. Restart in either direction is invisible to the rest of the
//! engine — see §3.1.

pub struct AudioFeatures {
    sample_rate: AtomicU32,  // from /audio/meta, diagnostic only
    band_low: AtomicU32,     // /audio/lmh args[0], already ~[0,1]
    band_mid: AtomicU32,     //                args[1]
    band_high: AtomicU32,    //                args[2]
    onset_low_ms: AtomicU64, // engine timestamp when /audio/onset/low arrives
    onset_mid_ms: AtomicU64,
    onset_high_ms: AtomicU64,
    last_packet_ms: AtomicU64, // bumped on every dispatched OSC message
    start: Instant,
}

impl AudioFeatures {
    pub fn new() -> Arc<Self> { /* zero-init */ }
    pub fn band(&self, band: AudioBand) -> f32 { /* unchanged */ }
    pub fn onset_envelope(&self, band: AudioBand, decay_seconds: f32) -> f32 { /* unchanged */ }
    /// True if a packet arrived in the last `stale_after_ms` ms. Powers the
    /// fresh ↔ stale watchdog log in §3.1.
    pub fn is_fresh(&self, stale_after_ms: u64) -> bool { /* … */ }
}

pub enum AudioBand { Low, Mid, High }  // unchanged
impl AudioBand { pub fn parse(s: &str) -> Option<Self> { /* unchanged */ } }

pub struct OscListener { _join: JoinHandle<()> }

pub fn try_spawn(state: Arc<AudioFeatures>, addr: SocketAddr) -> Option<OscListener> {
    match spawn(state, addr) {
        Ok(l) => { log::info!("OSC listening on {addr} (waiting for packets…)"); Some(l) },
        Err(e) => { log::warn!("OSC disabled: {e:#}"); None },
    }
}

fn spawn(state: Arc<AudioFeatures>, addr: SocketAddr) -> Result<OscListener> {
    let socket = UdpSocket::bind(addr)?;
    socket.set_read_timeout(Some(Duration::from_millis(250)))?; // allow clean shutdown + watchdog tick
    let handle = thread::Builder::new().name("osc-recv".into()).spawn(move || {
        let mut buf = [0u8; 8192]; // rosc max packet
        let mut was_fresh = false;
        loop {
            match socket.recv_from(&mut buf) {
                Ok((n, _)) => {
                    match rosc::decoder::decode_udp(&buf[..n]).map(|(_, p)| p) {
                        Ok(rosc::OscPacket::Message(msg)) => dispatch(&state, &msg),
                        Ok(rosc::OscPacket::Bundle(b))  => dispatch_bundle(&state, &b),
                        Err(e) => log::trace!("osc decode: {e}"),
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                    // Tick: no packet this interval, check watchdog edge.
                }
                Err(e) => { log::warn!("osc recv: {e}"); thread::sleep(Duration::from_millis(50)); }
            }
            // Watchdog: one log line per fresh ↔ stale transition (§3.1).
            let fresh = state.is_fresh(2_000);
            if fresh != was_fresh {
                if fresh {
                    log::info!("OSC: connected (packets arriving)");
                } else {
                    log::warn!("OSC: stale (no packets for 2s) — is the audio server running?");
                }
                was_fresh = fresh;
            }
        }
    })?;
    Ok(OscListener { _join: handle })
}

fn dispatch(state: &AudioFeatures, msg: &rosc::OscMessage) {
    state.mark_packet();  // updates last_packet_ms
    match msg.addr.as_str() {
        "/audio/lmh" => { /* read 3 floats, store band_{low,mid,high} */ }
        "/audio/onset/low"  => { state.stamp_onset_ms(&state.onset_low_ms);  }
        "/audio/onset/mid"  => { state.stamp_onset_ms(&state.onset_mid_ms);  }
        "/audio/onset/high" => { state.stamp_onset_ms(&state.onset_high_ms); }
        "/audio/meta"       => { /* read sr, log once on change, store sample_rate */ }
        "/audio/bpm" | "/audio/fft" => { /* v1: decoded so logs stay clean, value dropped (§10) */ }
        other => log::trace!("unhandled OSC addr {other}"),
    }
}
```

Behavioural parity with today:

- `AudioFeatures::rms/band/onset_envelope` keep the same f32 return semantics
  the driver bus already calls — drivers.rs needs **zero changes** to its
  evaluation logic, only the import path (`crate::audio::*` → `crate::osc::*`).
- Atomic ordering stays `Relaxed` everywhere — same correctness story (no
  cross-field invariants).

### 5.3 `src/drivers.rs`

Import change + drop the `AudioRms` driver variant:

```diff
-use crate::audio::{AudioBand, AudioState};
+use crate::osc::{AudioBand, AudioFeatures};
 ...
 pub enum DriverSpec {
     Const(f32),
     ClockBars { n: f32 },
     ClockBeats { n: f32 },
     ClockPhase { rate: f32 },
     ClockTime,
-    AudioRms,
     AudioBand(AudioBand),
     AudioOnset { band: AudioBand, decay: f32 },
     UiSlider { name: String, default: f32 },
 }
 ...
 match driver {
     ...
-    "audio.rms" => Ok(Self::AudioRms),
     "audio.band" => ...
     "audio.onset" => ...
     ...
+    "audio.rms" => bail!("audio.rms removed in OSC refactor — bind audio.band(\"low|mid|high\") instead"),
     other => bail!("unknown driver {other:?}"),
 }
```

…and every `&AudioState` becomes `&AudioFeatures`. The explicit error for
`audio.rms` is worth the two lines — if a stale scene uses it, the message
points straight at the fix instead of the generic "unknown driver."

No `audio.bpm` driver in v1 (§4.3). `audio.band`, `audio.onset` are the
entire audio driver surface.

Update the file-level docstring: the comment about "consumed as decaying
envelopes by `audio.onset`" is now even more accurate — the only event source
in v1 is OSC trigger arrival.

### 5.4 `src/compositor.rs` and `src/gpu.rs`

`FrameState` uniform layout today carries `audio_rms` + `audio_low/mid/high` +
`onset_low/mid/high`. The refactor drops `audio_rms` (§4.1):

```diff
 #[repr(C)]
 #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
 pub struct FrameState {
     pub time: f32,
     pub bar_phase: f32,
     pub beat_phase: f32,
     pub bpm: f32,
-    pub audio_rms: f32,
     pub audio_low: f32,
     pub audio_mid: f32,
     pub audio_high: f32,
     pub onset_low: f32,
     pub onset_mid: f32,
     pub onset_high: f32,
     pub resolution: [f32; 2],
     ...
 }
```

Rewire the source in the same diff:

```diff
-use crate::audio::AudioState;
+use crate::osc::AudioFeatures;

-pub fn tick(&self, gpu: &GpuContext, transport: &Transport, audio: &AudioState) {
+pub fn tick(&self, gpu: &GpuContext, transport: &Transport, audio: &AudioFeatures) {
     let ctx = transport.frame_context(audio);
     ...
-    audio_rms: audio.rms(),
-    audio_low: audio.band(crate::audio::AudioBand::Low),
+    audio_low: audio.band(crate::osc::AudioBand::Low),
     audio_mid: audio.band(crate::osc::AudioBand::Mid),
     audio_high: audio.band(crate::osc::AudioBand::High),
     ...
```

Shader prelude DOES change (§5.6) — `state.audio_rms` is gone. The
`FrameState` uniform layout bump is fine because there are no real user WGSL
effects deployed yet — the project-local `effects/drift/` shader in
`examples/effects/drift/` doesn't reference `audio_rms`, and that's the only
on-disk user effect. The bundled `phase3_smoke.scene.json`'s inline
`ripple` shader doesn't reference it either.

### 5.5 `src/main.rs`

CLI surface:

```diff
-/// Skip audio capture entirely. Useful for headless tests or boxes
-/// without an input device.
-#[arg(long)]
-no_audio: bool,
+/// Disable OSC ingest entirely. The engine still runs (clocks tick, drivers
+/// return 0 for audio fields) but no audio features arrive.
+#[arg(long)]
+no_osc: bool,
+
+/// UDP address the OSC receiver binds to. Must match the server's
+/// `osc.destinations[*].port`. Default 127.0.0.1:9000.
+#[arg(long, default_value = "127.0.0.1:9000")]
+osc_addr: SocketAddr,
```

Wiring:

```diff
-let audio_state = AudioState::new();
-let audio_capture = if cli.no_audio {
-    log::info!("audio capture disabled (--no-audio)");
-    None
-} else {
-    try_spawn(Arc::clone(&audio_state))
-};
+let audio_state = AudioFeatures::new();
+let osc_listener = if cli.no_osc {
+    log::info!("OSC ingest disabled (--no-osc)");
+    None
+} else {
+    osc::try_spawn(Arc::clone(&audio_state), cli.osc_addr)
+};
```

Field rename: `_audio_capture: Option<AudioCapture>` → `_osc_listener: Option<OscListener>`.

Module mount: `mod audio;` → `mod osc;`.

### 5.6 Shaders — `src/shaders/effect_prelude.wgsl`

Drop the `audio_rms` field to mirror the `FrameState` Rust struct (§5.4):

```diff
 struct State {
     time: f32,
     bar_phase: f32,
     beat_phase: f32,
     bpm: f32,
-    audio_rms: f32,
     audio_low: f32,
     audio_mid: f32,
     audio_high: f32,
     onset_low: f32,
     onset_mid: f32,
     onset_high: f32,
     ...
 };
```

Update the comment block near the audio fields to mention "values arrive
over OSC from the Realtime Audio Feature Server (see audio_refactor_plan.md
§3.1)."

Check the bundled `flash` built-in (`src/shaders/builtin_effects.wgsl`) — if
its comment references `audio.rms`, retarget to `audio.band(low)`.

### 5.7 Scene examples

`examples/phase3_smoke.scene.json` — patch the one `audio.rms` reference:

```diff
 "id": "primary_flash",
 "params": {
     "envelope": { "driver": "audio.onset", "band": "low", "decay": 0.18 },
-    "base":     { "driver": "audio.rms" },
+    "base":     { "driver": "audio.band", "band": "low" },
     "color":    "#ffffff"
 }
```

`audio.band(low)` is the closest semantic replacement for what the
mic-driven RMS used to do for this binding (both "track the low end
loudness"). The other audio drivers in the file (`audio.onset`, `audio.band`)
keep working unchanged.

### 5.8 `render-core/README.md`

Replace the "Audio" section. New copy (sketch):

> ## Audio
>
> `render-core` does not capture audio. It expects the Realtime Audio Feature
> Server (`audio-server`, see
> `/Users/xandersteenbrugge/Documents/GitHub/Realtime_PyAudio_FFT`) to be
> running and emitting OSC to `127.0.0.1:9000`. The render thread reads
> features from a lock-free atomic snapshot updated by a dedicated UDP recv
> thread.
>
> Quick start:
>
> ```bash
> # Terminal 1
> audio-server
>
> # Terminal 2
> cargo run -- --scene examples/phase3_smoke.scene.json --windowed
> ```
>
> Override the bind address with `--osc-addr 0.0.0.0:9000` (listen on all
> interfaces — useful if the audio server runs on another machine), or skip
> ingest with `--no-osc` for headless tests.
>
> Drivers `audio.band("low|mid|high")` and `audio.onset(band, decay)` map
> onto the server's `/audio/lmh` and `/audio/onset/{low,mid,high}` streams.
> No `audio.rms`, no `audio.bpm`, no `audio.fft` in v1 — the engine exposes
> exactly what the server emits; scene authors and the agent decide how to
> mix them into effects.
>
> The render-core auto-detects the server: the UDP socket is bound at
> startup regardless of whether the server is running, and packets flow
> as soon as the server starts. Stopping the server freezes the last
> values; restarting it resumes mid-flight. Status transitions log one
> line each.

Delete the "BlackHole/Loopback" paragraph entirely.

### 5.9 `render_engine_architecture.md`

- §3.7 driver list: replace the `cpal + rustfft` description with "OSC ingest
  from the Realtime Audio Feature Server, schema `/audio/lmh`, `/audio/onset/*`,
  `/audio/bpm`." Strike "Audio captured via cpal + rustfft + three-band
  spectral-flux onset detection" from the Phase 3 summary.
- §8 #6 ("Audio loopback on macOS without user setup") — resolved. Delete the
  whole bullet; the audio-loopback question now belongs to the audio server.
- §9 Known risks — drop the "Audio onset detection" bullet (it's the server's
  problem now, and is genuinely solved there).
- Phase 3 "deferred" list — `rosc` ingest is no longer deferred; mark it
  landed.
- Add a short note in §3.10 (Python ↔ Tauri integration) acknowledging the
  audio server as a *third* sibling process (alongside `wzrd_mcp` and the
  future Tauri shell). Same Pattern A — files on disk + localhost protocols,
  never embedded.

---

## 6. Compatibility & semantic deltas

Things that *change observably*:

- **`audio.rms` removed.** Scenes referencing it now log a clear error
  (`"audio.rms removed in OSC refactor — bind audio.band("low|mid|high") instead"`)
  and the binding is skipped; the engine keeps running. Only the bundled
  `phase3_smoke.scene.json` uses it today, patched in this same PR.
- **`state.audio_rms` removed from the shader prelude.** No user WGSL
  references it today (verified against `examples/effects/drift/` and the
  inline `ripple` shader). Any future user shader that needs "overall
  energy" picks a band explicitly.
- **Onset behaviour changes for the better.** Today: spectral-flux on the
  cpal mic signal, threshold `0.04` hard-coded. After: whatever the audio
  server's per-band Schmitt detector says — tunable live from the server's
  browser UI. Acceptance is "do kick/snare/hat trigger cleanly in
  `phase3_smoke`."
- **No mic permission prompt on macOS first run.** Direct UX win.
- **"Audio offline" failure mode changes** from "mic blocked / no input
  device" to "server not running on `127.0.0.1:9000`." Easier to diagnose
  (`lsof -iUDP:9000` or open the server's browser UI), and the §3.1
  watchdog log says so directly.
- **No coupling to engine startup order.** Either process can start first;
  see §3.1. Today the engine has to be started after BlackHole or mic
  permission is sorted — that ordering constraint goes away.

Things that *don't* change:

- Scene JSON schema. `audio.band` / `audio.onset` keep working byte-for-byte.
- Effect WGSL prelude for `state.audio_low/mid/high` and `state.onset_*`.
- Hot-reload behaviour on `scene.json` / `effects/`.
- The composite + homography pass plan.

---

## 7. Open decisions to confirm before coding

Resolved during plan review:

- ~~`audio.rms`~~ — **removed entirely.** Engine exposes raw incoming
  features; author picks the band. §4.1.
- ~~BPM driver / transport coupling~~ — **out of v1.** `/audio/bpm` decoded
  and dropped. §4.3 / §10.
- ~~Connection diagnostic~~ — **yes, with 2 s hysteresis.** Built into the
  recv loop in §5.2.

Still open:

1. **Bind address default.** `127.0.0.1:9000` matches the server's default
   destination. Single `--osc-addr SOCKET_ADDR` flag (accepts both
   `127.0.0.1:9000` and `0.0.0.0:9000`), or split host/port? Recommend
   one `SocketAddr` flag.
2. **`/audio/fft` — drop, store, or expose?** Recommend drop in this PR;
   revisit when a scene asks for it (§10). Server defaults to FFT off anyway.
3. **`AudioFeatures` rename — do it or leave the type as `AudioState`?**
   Cosmetic; `AudioFeatures` reads better since we no longer own audio
   state, we just snapshot features. Recommend rename in this PR — touches
   ~5 import sites.

---

## 8. Test plan

Smoke tests run by hand; there is no test runner here today. The first four
cases are the headline auto-detect / auto-recover acceptance from §3.1 —
explicitly verify each one.

1. **Engine starts before server (cold).** `cargo run -- --scene
   examples/phase3_smoke.scene.json --windowed` with `audio-server` not
   running. Expected: UDP bind succeeds, `OSC listening on 127.0.0.1:9000 (waiting for packets…)`
   logged, projector renders with `audio.*` drivers returning 0. Clocks
   animate normally. After 2 s, watchdog log: `OSC: stale …`. Engine does
   not crash and does not block.
2. **Server starts after engine.** With (1) still running, launch
   `audio-server` in another terminal and play audio. Expected: within one
   packet, `primary_flash` (band=low onset) and `audio_drift` (band=mid)
   start reacting. Watchdog log: `OSC: connected (packets arriving)`.
   **No engine restart, no scene reload, no extra step needed.**
3. **Server stops mid-run.** Kill `audio-server` while engine is rendering
   reactively. Expected: bands freeze at last value, onset envelopes decay
   naturally to zero via the existing `exp(-dt/τ)` read-side decay. After
   2 s, watchdog log: `OSC: stale …`. Engine does not crash.
4. **Server restarts mid-run.** Restart `audio-server`. Expected: features
   resume within one packet, watchdog: `OSC: connected (packets resumed)`.
   No engine action needed.
5. **Engine starts after server.** Launch `audio-server` first, then engine.
   Expected: same as (2) but watchdog goes straight to `connected` with no
   `stale` interlude.
6. **`--no-osc`.** Engine starts, no socket bound, all audio drivers read 0,
   no watchdog logs.
7. **`--osc-addr 127.0.0.1:9001` mismatch.** Engine binds 9001, server emits
   to 9000 — expected: no features arrive; user misconfiguration, but the
   engine still runs and the watchdog flags `stale` after 2 s.
8. **Port already in use.** Start a second `render-core` against the same
   `--osc-addr`. Expected: second instance logs `OSC disabled: …address in use…`,
   keeps running with zeroed features (same path as `--no-osc`).
9. **GPU memory steady state.** Run for 15 minutes with audio active.
   Expected: no growth (no per-packet allocation — fixed 8 KB recv buffer,
   atomics written in place).
10. **No mic permission prompt on macOS first launch.** Verify on a clean
    user account or after revoking Terminal's mic permission.

Add a tiny `osc.rs` unit test for `dispatch()`: hand-craft `OscMessage`
instances for each address, assert atomics update and `last_packet_ms`
advances. Doesn't need a socket.

---

## 9. Rollout

Single PR, single commit. Scope is small enough (≈300 lines net deleted,
≈200 lines added):

1. `Cargo.toml`: drop `cpal`, `rustfft`; add `rosc`.
2. Delete `src/audio.rs`, create `src/osc.rs`.
3. Rename `AudioState` → `AudioFeatures`, `try_spawn` → `osc::try_spawn`,
   `AudioCapture` → `OscListener`. Mechanical.
4. `src/main.rs`: swap CLI flag (`--no-audio` → `--no-osc` + `--osc-addr`),
   swap module mount, swap spawn call.
5. `src/drivers.rs`, `src/compositor.rs`: update imports. Optionally add
   `DriverSpec::AudioBpm`.
6. `render-core/README.md`: rewrite "Audio" section, drop BlackHole paragraph,
   add server-startup quick start.
7. `render_engine_architecture.md`: update §3.7, §3.10, §8 #6, §9 audio risks,
   Phase 3 summary.

Manual smoke tests from §8.

No deprecation period for `--no-audio` — it's an internal binary, scenes don't
reference it, and the flag is replaced by a strictly better one.

---

## 10. Deliberately deferred

- **BPM in any form (`audio.bpm` driver, `/audio/bpm` ingest, transport
  auto-follow).** The server already streams BPM but no scene needs it yet,
  and the design tradeoffs (BPM jitter wobbling integrated bar/beat phase
  vs. auto-locking the clock to the music) deserve a real use case before
  picking a side. When a show wants it: store `/audio/bpm` in an atomic,
  add `audio.bpm` driver (raw value, no clock coupling) as the cheap path;
  or add `transport.followAudioBpm: true` to `scene.json` for the
  full auto-lock path (one conditional in `Transport::elapsed_sec`, possibly
  with a freeze-when-stale policy so the phase doesn't jolt during silence).
- **FFT bin streaming (`/audio/fft`).** The server can emit 128 (configurable)
  per-bin floats. Today no scene needs them. When the first one does:
  store as `Vec<AtomicU32>` of length `n_fft_bins` (from `/audio/meta`),
  expose as `audio.fft` driver returning a slice + add a storage-buffer
  binding to the `FrameState` so shaders can index bins. Server-side knob is
  `osc.send_fft: true` + `fft.enabled: true` in its `configs/main.yaml`.
- **Pre-autoscale raw RMS** (`low_raw / mid_raw / high_raw`). Only available
  over the server's WebSocket. If a scene needs it, add a `/audio/lmh_raw`
  channel server-side (the server's own README v1.1 list).
- **`set_*` control messages back to the server.** All knobs are tunable
  from the server's own browser UI; tunneling them through `render-core`
  and into Tauri is a Phase 4+ concern (and only if the operator workflow
  demands it — arguably "open the audio server's localhost UI in another
  tab" is simpler).
- **MIDI / generic OSC paths beyond `/audio/*`.** The architecture doc's
  §3.7 promises `midi.cc(n)`, `osc.path('/x/y')` — not in this PR. Once
  `osc.rs` exists, generalising the dispatcher to populate a
  `HashMap<String, f32>` for non-`/audio/*` paths is a 30-line follow-up.
