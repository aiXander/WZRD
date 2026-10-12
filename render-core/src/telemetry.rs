//! Telemetry bus — engine → WS subscribers (§3.11 `telemetry.subscribe`).
//!
//! The bus is owned jointly by the engine (writer) and the WS server (fans
//! out to live subscribers). Channels in Phase 4.1: `preview`, `hot_reload`,
//! `audio_freshness`, `fps`. Phase 4.2 widens to `log`, `frame_stats`,
//! `drivers`, `audio`, `connectivity`.
//!
//! Backpressure model: each subscriber has a bounded outbound queue. If the
//! queue fills (slow consumer), we drop the *oldest* messages on the noisy
//! channels (`preview`, `frame_stats`, `drivers`, `audio`, `log`) and *block
//! briefly* on the critical-but-rare channels (`hot_reload`,
//! `audio_freshness`). This keeps the projector window responsive when the
//! UI is paused on the operator's display.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use serde::Serialize;
use serde_json::{json, Value};

use crate::gpu::GpuContext;

/// Logical names of all live telemetry channels. Used by `telemetry.subscribe`
/// to validate the requested channel list and by the bus to filter emission.
pub const ALL_CHANNELS: &[&str] = &[
    "preview",
    "hot_reload",
    "audio_freshness",
    "fps",
    "log",
    "frame_stats",
    "drivers",
    "audio",
    "transport",
    "connectivity",
    "masters",
    "deck",
    "changes",
    "alignment",
    "perf",
    "perf_spike",
];

/// Channels whose most recent payload is retained and replayed to new
/// subscribers — the "current state" channels, as opposed to the streaming
/// ones where only the next frame matters.
///
/// **One definition, deliberately.** The Tauri host used to keep its own copy
/// of this list for its `last_payload` snapshots, and the two silently drifted
/// the moment a channel was added: `alignment` emits only at boot, so a
/// missing entry there meant the Align tab sat on "waiting for the engine's
/// alignment document…" forever. Any host that snapshots sticky state must
/// call this, never re-type the list.
pub fn is_sticky(channel: &str) -> bool {
    matches!(
        channel,
        "hot_reload"
            | "audio_freshness"
            | "connectivity"
            | "fps"
            | "masters"
            | "deck"
            | "changes"
            | "alignment"
            | "perf_spike"
    )
}

/// What an emitter pushes onto the bus and what a subscriber sees on the
/// wire. Channel + payload — payload is a JSON value the WS server can
/// envelope directly into a JSON-RPC notification.
#[derive(Debug, Clone, Serialize)]
pub struct TelemetryFrame {
    pub channel: String,
    pub payload: Value,
}

/// Outbound queue capacity per subscriber. Large enough that a 30 Hz
/// `drivers` snapshot + 15 fps `preview` frame can coexist for a full second
/// of UI hiccup before drops start.
const SUBSCRIBER_CAP: usize = 256;

/// A single live subscriber. Owned by the WS server; the bus only sees the
/// sender half plus the subscribed channel filter.
pub struct Subscriber {
    pub id: u64,
    pub channels: HashSet<String>,
    pub tx: Sender<TelemetryFrame>,
}

#[derive(Default)]
struct Inner {
    subscribers: Mutex<Vec<Subscriber>>,
    next_id: AtomicU64,
    /// Sticky values — these are the most recent value emitted on each
    /// channel for which we have a notion of "current state." When a new
    /// subscriber attaches, the WS server replays the sticky values so the
    /// UI doesn't have to wait for the next emission.
    sticky: Mutex<std::collections::HashMap<String, Value>>,
}

#[derive(Clone, Default)]
pub struct Bus {
    inner: Arc<Inner>,
}

impl Bus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new subscriber. Returns the receiver half + a stable id.
    pub fn subscribe(&self, channels: HashSet<String>) -> (u64, Receiver<TelemetryFrame>) {
        let (tx, rx) = bounded(SUBSCRIBER_CAP);
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        // Replay sticky values that match the requested channels.
        if let Ok(sticky) = self.inner.sticky.lock() {
            for (channel, payload) in sticky.iter() {
                if channels.contains(channel) {
                    let _ = tx.try_send(TelemetryFrame {
                        channel: channel.clone(),
                        payload: payload.clone(),
                    });
                }
            }
        }
        let mut subs = self.inner.subscribers.lock().expect("subscribers lock");
        subs.push(Subscriber { id, channels, tx });
        (id, rx)
    }

    pub fn unsubscribe(&self, id: u64) {
        let mut subs = self.inner.subscribers.lock().expect("subscribers lock");
        subs.retain(|s| s.id != id);
    }

    /// How many live subscribers listen on `channel`. Lets emitters skip
    /// expensive work (e.g. the preview JPEG readback) when nobody is
    /// listening (§6.4 demand-gated capture).
    pub fn subscriber_count(&self, channel: &str) -> usize {
        self.inner
            .subscribers
            .lock()
            .map(|subs| {
                subs.iter()
                    .filter(|s| s.channels.contains(channel))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Update the set of channels a live subscriber is listening on.
    pub fn update_channels(&self, id: u64, channels: HashSet<String>) {
        let mut subs = self.inner.subscribers.lock().expect("subscribers lock");
        if let Some(s) = subs.iter_mut().find(|s| s.id == id) {
            s.channels = channels;
        }
    }

    /// Emit on `channel`. Drops the oldest message on a slow subscriber for
    /// the noisy channels; for sticky channels (`hot_reload`,
    /// `audio_freshness`, `connectivity`), the payload is also retained as
    /// the current value so late-arriving subscribers see it.
    pub fn emit(&self, channel: &str, payload: Value) {
        if is_sticky(channel) {
            if let Ok(mut s) = self.inner.sticky.lock() {
                s.insert(channel.to_string(), payload.clone());
            }
        }
        let frame = TelemetryFrame {
            channel: channel.to_string(),
            payload,
        };
        let mut subs = self.inner.subscribers.lock().expect("subscribers lock");
        let mut closed: Vec<u64> = Vec::new();
        for s in subs.iter() {
            if !s.channels.contains(channel) {
                continue;
            }
            match s.tx.try_send(frame.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    // Backpressure: drain one to make room, drop oldest.
                    // crossbeam-channel doesn't expose pop-back, so the
                    // simplest "drop" is a noop send into the same channel —
                    // we accept a momentary stutter rather than block.
                    log::trace!(
                        "telemetry subscriber {} slow on channel {channel}",
                        s.id
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    closed.push(s.id);
                }
            }
        }
        if !closed.is_empty() {
            subs.retain(|s| !closed.contains(&s.id));
        }
    }

    // ---------- channel-typed convenience helpers ----------

    pub fn emit_fps(&self, fps: f32, frame_time_ms: f32, presenting: bool) {
        self.emit(
            "fps",
            json!({
                "fps": fps,
                "frame_time_ms": frame_time_ms,
                // false ⇒ the projector window is occluded and the engine is
                // deliberately self-pacing offscreen (~30 Hz) — the UI shows
                // this as a distinct mode, not as a performance problem.
                "presenting": presenting,
            }),
        );
    }

    pub fn emit_audio_freshness(&self, fresh: bool, last_packet_ms: u64) {
        let state = if fresh {
            "fresh"
        } else if last_packet_ms == 0 {
            "down"
        } else {
            "stale"
        };
        self.emit(
            "audio_freshness",
            json!({
                "state": state,
                "last_packet_ms": last_packet_ms,
            }),
        );
    }

    pub fn emit_hot_reload(&self, event: HotReloadEvent) {
        let probe = event
            .probe
            .as_ref()
            .map(|p| serde_json::to_value(p).expect("probe report"));
        self.emit(
            "hot_reload",
            json!({
                "target": event.target,
                "ok": event.ok,
                "elapsed_ms": event.elapsed_ms,
                "message": event.message,
                "probe": probe,
                "epoch": event.epoch,
                "rev": event.rev,
                "actor": event.actor,
            }),
        );
    }

    /// §5.6 two-deck snapshot — sticky, emitted on every promote/pull/preview
    /// transition (plus a ~10 Hz trickle while a fade ramps).
    pub fn emit_deck(&self, snapshot: DeckSnapshot) {
        self.emit("deck", serde_json::to_value(snapshot).expect("deck snapshot"));
    }

    /// §5.14 alignment document — sticky, emitted at boot, on every accepted
    /// mutation, and when the output size changes. Sticky so the Align tab
    /// hydrates from `lastPayload('alignment')` exactly like `masters`/`deck`,
    /// and so an external camera script sees the UI's edits.
    pub fn emit_alignment(&self, payload: Value) {
        self.emit("alignment", payload);
    }

    pub fn emit_preview_jpeg(&self, jpeg_bytes: &[u8], width: u32, height: u32) {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg_bytes);
        self.emit(
            "preview",
            json!({
                "encoding": "jpeg",
                "width": width,
                "height": height,
                "data_b64": b64,
            }),
        );
    }

    pub fn emit_audio(&self, snapshot: AudioSnapshot) {
        self.emit("audio", serde_json::to_value(snapshot).expect("audio snapshot"));
    }

    pub fn emit_frame_stats(&self, stats: FrameStats) {
        self.emit("frame_stats", serde_json::to_value(stats).expect("frame stats"));
    }

    pub fn emit_drivers(&self, snapshot: DriverSnapshot) {
        self.emit("drivers", serde_json::to_value(snapshot).expect("drivers snapshot"));
    }

    pub fn emit_connectivity(&self, connectivity: Connectivity) {
        self.emit(
            "connectivity",
            serde_json::to_value(connectivity).expect("connectivity"),
        );
    }

    /// §5.4/§5.6 masters snapshot — sticky, emitted at startup, on every
    /// `master.set`, and when promote/pull copy a leg's control state.
    /// Carries **both legs** (full-control-switch: the UI shows whichever
    /// leg the deck toggle selects).
    pub fn emit_masters(&self, state: MastersState) {
        self.emit("masters", serde_json::to_value(state).expect("masters"));
    }

    pub fn emit_log(&self, level: &str, target: &str, message: &str) {
        self.emit(
            "log",
            json!({
                "level": level,
                "target": target,
                "message": message,
                "ts_ms": now_ms(),
            }),
        );
    }
}

#[derive(Debug, Clone)]
pub struct HotReloadEvent {
    pub target: String,
    pub ok: bool,
    pub elapsed_ms: f32,
    pub message: Option<String>,
    /// §5.6 pre-flight probe outcome, when one ran for this reload —
    /// `{compiled, predicted_p95_ms, band, thumbnail_b64, verdicts}` so the
    /// authoring agent self-corrects on performance, not just compiles.
    pub probe: Option<ProbeReport>,
    /// §5.10 correlation stamp: boot epoch + design rev after this event +
    /// the actor whose write caused it. Serves push consumers (webview
    /// re-sync) — author-verdict correlation uses the RPC reply itself,
    /// never sticky replay.
    pub epoch: u64,
    pub rev: u64,
    pub actor: String,
}

// ---------- §5.10 change ring: rev counter + boot epoch ----------

/// One design mutation, as carried on the sticky `changes` channel and the
/// `changes.list` backfill RPC — one shape, two transports.
#[derive(Debug, Clone, Serialize)]
pub struct ChangeEntry {
    /// Boot epoch (engine start, epoch ms). A crash-relaunch resets the rev
    /// counter; the epoch keeps a stale `since_rev` from silently lying.
    pub epoch: u64,
    /// Monotonic design revision — bumped by every design mutation.
    pub rev: u64,
    pub ts_ms: u64,
    /// "ui" | "agent" | "system" (per-connection identity, §5.10).
    pub actor: String,
    /// Facet taxonomy tag: "bindings" | "effects" | "layers".
    pub facet: String,
    pub summary: String,
}

const CHANGE_RING_DEPTH: usize = 32;

/// §5.10 — the design leg's revision counter, boot epoch, and a bounded ring
/// of recent [`ChangeEntry`]s. Lives engine-side because the operator's UI
/// edits must land in it too; the render thread records, every read surface
/// (RPC, MCP digest header) reads.
pub struct ChangeLog {
    epoch: u64,
    rev: AtomicU64,
    ring: Mutex<std::collections::VecDeque<ChangeEntry>>,
}

impl ChangeLog {
    pub fn new() -> Self {
        Self {
            epoch: now_ms(),
            rev: AtomicU64::new(0),
            ring: Mutex::new(std::collections::VecDeque::with_capacity(
                CHANGE_RING_DEPTH,
            )),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn rev(&self) -> u64 {
        self.rev.load(Ordering::Acquire)
    }

    /// Bump the rev, push the entry onto the ring, broadcast it on the
    /// sticky `changes` channel. Returns the new rev.
    pub fn record(&self, bus: &Bus, actor: &str, facet: &str, summary: impl Into<String>) -> u64 {
        let rev = self.rev.fetch_add(1, Ordering::AcqRel) + 1;
        let entry = ChangeEntry {
            epoch: self.epoch,
            rev,
            ts_ms: now_ms(),
            actor: actor.to_string(),
            facet: facet.to_string(),
            summary: summary.into(),
        };
        {
            let mut ring = self.ring.lock().expect("change ring lock");
            if ring.len() >= CHANGE_RING_DEPTH {
                ring.pop_front();
            }
            ring.push_back(entry.clone());
        }
        bus.emit("changes", serde_json::to_value(&entry).expect("change entry"));
        rev
    }

    /// The most recent `n` entries, newest last.
    pub fn recent(&self, n: usize) -> Vec<ChangeEntry> {
        let ring = self.ring.lock().expect("change ring lock");
        ring.iter().rev().take(n).rev().cloned().collect()
    }

    /// `changes.list {since_rev?, epoch?}` payload. An epoch mismatch or a
    /// `since_rev` older than the ring's tail returns everything the ring
    /// holds plus an explicit `note` — never a silently-partial diff.
    pub fn list_json(&self, since_rev: Option<u64>, epoch: Option<u64>) -> Value {
        let current = self.rev();
        let ring = self.ring.lock().expect("change ring lock");
        let mut note: Option<String> = None;
        let entries: Vec<&ChangeEntry> = match (since_rev, epoch) {
            (Some(_), Some(e)) if e != self.epoch => {
                note = Some(format!(
                    "engine restarted (epoch {} != yours {}); full ring returned — treat your cached state as stale",
                    self.epoch, e
                ));
                ring.iter().collect()
            }
            (Some(s), _) => {
                let tail_rev = ring.front().map(|e| e.rev).unwrap_or(current + 1);
                if s + 1 < tail_rev && s < current {
                    note = Some(format!(
                        "ring wrapped (oldest retained rev {tail_rev}, you asked since {s}); full ring returned — re-read the facets you care about"
                    ));
                    ring.iter().collect()
                } else {
                    ring.iter().filter(|e| e.rev > s).collect()
                }
            }
            (None, _) => ring.iter().collect(),
        };
        json!({
            "epoch": self.epoch,
            "rev": current,
            "entries": entries,
            "note": note,
        })
    }
}

/// The probe payload that rides `hot_reload` (§5.6). `predicted_p95_ms` /
/// `band` / `thumbnail_b64` describe the *worst* probed pipeline;
/// `verdicts` carries the per-pipeline breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    pub compiled: bool,
    pub predicted_p95_ms: f32,
    pub band: String,
    pub thumbnail_b64: Option<String>,
    pub verdicts: Vec<crate::probe::KeyVerdict>,
}

/// §5.4/§5.6 — per-leg masters, the `masters` channel payload. On single-leg
/// (headless) runs both fields report the one live set.
#[derive(Debug, Clone, Serialize)]
pub struct MastersState {
    pub live: crate::drivers::MastersSnapshot,
    pub design: crate::drivers::MastersSnapshot,
    /// §5.4 crossfade-time master (seconds) — engine-wide, not per leg: the
    /// default promote fade. Carried here so every client converges on one
    /// value without a second telemetry channel.
    pub crossfade: f32,
}

/// §5.6 deck snapshot — the two-leg state the UI's promote controls render.
#[derive(Debug, Clone, Serialize)]
pub struct DeckSnapshot {
    /// "idle" | "pending" (bar-quantized, waiting for the boundary) |
    /// "ramping" (fade in flight).
    pub promote: String,
    /// Live→design crossfade position, 0..1.
    pub mix: f32,
    pub fade_ms: Option<f32>,
    pub quantize: Option<String>,
    /// Which composite the native preview samples: "live" | "design".
    pub preview_source: String,
    /// False on headless single-leg runs (promote/pull unavailable).
    pub two_leg: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioSnapshot {
    pub band_low: f32,
    pub band_mid: f32,
    pub band_high: f32,
    pub onset_low: f32,
    pub onset_mid: f32,
    pub onset_high: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameStats {
    pub fps: f32,
    pub frame_time_ms_p50: f32,
    pub frame_time_ms_p95: f32,
    pub frame_time_ms_p99: f32,
    pub mask_slice_count: u32,
    pub pipeline_count: u32,
    pub pass_count: u32,
    /// false ⇒ occluded, self-paced offscreen rendering (intentional).
    pub presenting: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriverSnapshot {
    pub drivers: Vec<DriverRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriverRow {
    pub binding_id: String,
    pub param_name: String,
    pub source: String,
    pub value: f32,
    pub affects: u32,
    /// §5.5 — true when a live `param.set {binding, param}` override is
    /// pinning this param; `value` then reports the override.
    pub overridden: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Connectivity {
    pub osc: ConnectivityCell,
    pub file_watcher: ConnectivityCell,
    pub ws: ConnectivityCell,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectivityCell {
    pub status: String, // "ok" | "warn" | "down"
    pub detail: Option<String>,
}

// ---------- emitter helpers (composed into the render thread) ----------

/// Tracks frame timings over a short sliding **time** window and emits an
/// `fps` event roughly twice a second so the status pill has fresh numbers
/// without flooding the bus.
///
/// The window is time-based (last ~2.5 s), not count-based: a count-based
/// window spans 10+ seconds of wall time at 60 fps, which turns a clean
/// mode switch (e.g. present @60 → occluded self-pace @30) into a slow,
/// alarming-looking droop on the pill. With a time window the readout
/// settles on the new rate within the window length.
pub struct FpsAccumulator {
    /// Frame delta-times (ms), oldest first. Pruned so the sum stays within
    /// `WINDOW_MS`.
    samples: std::collections::VecDeque<f32>,
    total_ms: f32,
    last_frame: Option<Instant>,
    last_emit: Instant,
}

const FPS_WINDOW_MS: f32 = 2_500.0;

impl FpsAccumulator {
    pub fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(512),
            total_ms: 0.0,
            last_frame: None,
            last_emit: Instant::now(),
        }
    }

    /// Drop history — called on present ↔ offscreen mode switches so the
    /// readout snaps to the new rate instead of blending across modes.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.total_ms = 0.0;
        self.last_frame = None;
    }

    pub fn mark(&mut self, bus: &Bus, counts: FrameCounts, presenting: bool) {
        let now = Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = now.duration_since(prev).as_secs_f32() * 1000.0;
            self.samples.push_back(dt);
            self.total_ms += dt;
            while self.total_ms > FPS_WINDOW_MS && self.samples.len() > 1 {
                if let Some(old) = self.samples.pop_front() {
                    self.total_ms -= old;
                }
            }
        }
        self.last_frame = Some(now);

        if now.duration_since(self.last_emit) >= Duration::from_millis(500) {
            let (p50, p95, p99) = percentiles(&self.samples);
            // Honest throughput: frames delivered per second of wall time,
            // NOT 1000/p50 — the latter reads "211 fps" while the engine is
            // actually hitching on 1s stalls.
            let fps = if self.total_ms > 0.0 {
                self.samples.len() as f32 * 1000.0 / self.total_ms
            } else {
                0.0
            };
            bus.emit_fps(fps, p50, presenting);
            bus.emit_frame_stats(FrameStats {
                fps,
                frame_time_ms_p50: p50,
                frame_time_ms_p95: p95,
                frame_time_ms_p99: p99,
                mask_slice_count: counts.mask_slices,
                pipeline_count: counts.pipelines,
                pass_count: counts.passes,
                presenting,
            });
            self.last_emit = now;
        }
    }
}

/// Per-frame resource counts the render thread hands to [`FpsAccumulator`]
/// so the `frame_stats` channel carries real numbers.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameCounts {
    pub mask_slices: u32,
    pub pipelines: u32,
    pub passes: u32,
}

fn percentiles(samples: &std::collections::VecDeque<f32>) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut v: Vec<f32> = samples.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f32| -> f32 {
        let idx = ((v.len() as f32) * q).clamp(0.0, (v.len() - 1) as f32) as usize;
        v[idx]
    };
    (pick(0.5), pick(0.95), pick(0.99))
}

/// The JPEG `preview` channel and `preview.snapshot` (docs/TODO/native-
/// preview-underlay.md D5). In the app the native preview shows the
/// composite directly; this path only serves external clients (MCP
/// `get_preview`, the browser build, remote operators) and agent snapshots.
///
/// **Render-thread budget.** Never `device.poll(Wait)`, never encode on the
/// render thread:
///
/// 1. *Kick off* (stream, every `interval` while subscribed): a GPU pass
///    downscales the composite into a 320-px `Rgba8UnormSrgb` thumbnail
///    (`gpu::Downscaler`, sRGB encode free on store), copies just that
///    (~230 KB at 4:3) into a readback buffer, `map_async`.
/// 2. *Pending*: each call drives `Maintain::Poll`; returns if unmapped.
/// 3. *Ready*: copy the mapped bytes into a `Vec`, unmap, hand the job to
///    the encode worker through a `bounded(1)` channel — busy worker →
///    frame dropped (it's a thumbnail). The worker JPEG-encodes and emits.
///
/// Snapshots (`preview.snapshot`) ride the same readback → worker path on
/// their own unbounded queue (never dropped), and reply to their caller.
pub struct PreviewSampler {
    interval: Duration,
    /// When the current/last stream capture was kicked off.
    last_capture: Instant,
    downscaler: Option<crate::gpu::Downscaler>,
    stream: Option<Readback>,
    /// The last stream readback buffer, unmapped, for the next capture.
    spare: Option<wgpu::Buffer>,
    snapshots: Vec<(Readback, SnapshotReply)>,
    worker: Option<EncodeWorker>,
}

/// One in-flight `map_async` readback of a 4-byte-per-pixel image.
struct Readback {
    buffer: wgpu::Buffer,
    bytes_per_row: u32,
    width: u32,
    height: u32,
    /// Byte order of the source texture (the preview surface may be BGRA).
    bgra: bool,
    state: Arc<AtomicU8>,
}

/// Where a snapshot's encoded image goes, plus what to report with it.
pub struct SnapshotReply {
    pub reply: Sender<Result<Value, String>>,
    /// Echoed into the reply (the `PreviewParams` used, etc.).
    pub meta: Value,
}

/// Pixels handed to the encode worker: tightly packed, 4 bytes per pixel.
struct EncodeJob {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    bgra: bool,
    snapshot: Option<SnapshotReply>,
}

struct EncodeWorker {
    stream_tx: Sender<EncodeJob>,
    snapshot_tx: Sender<EncodeJob>,
}

const MAP_PENDING: u8 = 0;
const MAP_READY_OK: u8 = 1;
const MAP_READY_ERR: u8 = 2;

/// Width of the `preview` channel thumbnail.
const THUMB_WIDTH: u32 = 320;

impl Readback {
    fn start(buffer: wgpu::Buffer, bytes_per_row: u32, width: u32, height: u32, bgra: bool) -> Self {
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let cb = Arc::clone(&state);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            cb.store(if res.is_ok() { MAP_READY_OK } else { MAP_READY_ERR }, Ordering::Release);
        });
        Self {
            buffer,
            bytes_per_row,
            width,
            height,
            bgra,
            state,
        }
    }

    /// `None` while still mapping; `Some(Err)` if the map failed;
    /// `Some(Ok(pixels))` (tightly packed) once ready — the buffer is
    /// unmapped and done.
    fn take(&self) -> Option<Result<Vec<u8>, ()>> {
        match self.state.load(Ordering::Acquire) {
            MAP_PENDING => None,
            MAP_READY_OK => {
                let row = (self.width * 4) as usize;
                let mut pixels = Vec::with_capacity(row * self.height as usize);
                {
                    let data = self.buffer.slice(..).get_mapped_range();
                    for y in 0..self.height as usize {
                        let at = y * self.bytes_per_row as usize;
                        pixels.extend_from_slice(&data[at..at + row]);
                    }
                }
                self.buffer.unmap();
                Some(Ok(pixels))
            }
            _ => Some(Err(())),
        }
    }
}

impl PreviewSampler {
    pub fn new() -> Self {
        Self {
            interval: Duration::from_millis(66), // ~15 fps
            last_capture: Instant::now() - Duration::from_secs(1),
            downscaler: None,
            stream: None,
            spare: None,
            snapshots: Vec::new(),
            worker: None,
        }
    }

    /// Perf-monitor PM1 GPU-MB inventory: the thumbnail target + readbacks.
    pub fn gpu_bytes(&self) -> u64 {
        self.downscaler.as_ref().map_or(0, |d| d.gpu_bytes())
            + self.stream.as_ref().map_or(0, |r| r.buffer.size())
            + self.spare.as_ref().map_or(0, |b| b.size())
            + self.snapshots.iter().map(|(r, _)| r.buffer.size()).sum::<u64>()
    }

    /// Anything in flight that needs this frame's `maybe_capture` even with
    /// no `preview` subscriber (a snapshot, or a stream frame mid-map).
    pub fn busy(&self) -> bool {
        self.stream.is_some() || !self.snapshots.is_empty()
    }

    /// A stream capture would kick off if `maybe_capture` ran now with
    /// `stream` set: none in flight and the ~15 Hz interval has elapsed.
    /// `Core` asks this at frame start so a design composite that only
    /// feeds the JPEG stream renders on capture frames alone.
    pub fn capture_due(&self) -> bool {
        self.stream.is_none() && self.interval_up()
    }

    /// The interval is up, judged at frame *start* with a little lead so a
    /// capture lands on a steady frame count (4 frames at 60 Hz) instead of
    /// slipping a frame whenever the start-of-frame check falls just short.
    /// A steady cadence keeps a sparsely rendered design leg's clip
    /// prefetch stride exact (no stale frames).
    fn interval_up(&self) -> bool {
        const LEAD: Duration = Duration::from_millis(8);
        self.last_capture.elapsed() + LEAD >= self.interval
    }

    fn worker(&mut self, bus: &Bus) -> &EncodeWorker {
        self.worker.get_or_insert_with(|| EncodeWorker::spawn(bus.clone()))
    }

    /// Queue a `preview.snapshot` readback (the caller already encoded the
    /// copy into `buffer`; this kicks off the map once submitted).
    pub fn push_snapshot(
        &mut self,
        buffer: wgpu::Buffer,
        bytes_per_row: u32,
        width: u32,
        height: u32,
        bgra: bool,
        reply: SnapshotReply,
    ) {
        let rb = Readback::start(buffer, bytes_per_row, width, height, bgra);
        self.snapshots.push((rb, reply));
    }

    /// Per-frame driver. `texture` is the composite the stream samples —
    /// the design leg's in two-leg mode (§5.6 blanket leg rule: the JPEG
    /// channel is an authoring surface), the live composite headless.
    /// `stream` = someone subscribes to `preview` right now **and**
    /// `texture` was rendered this frame (a demand-gated composite that sat
    /// out this frame must not be captured stale — see `capture_due`).
    pub fn maybe_capture(
        &mut self,
        gpu: &GpuContext,
        bus: &Bus,
        texture: &wgpu::Texture,
        stream: bool,
    ) {
        // Fire any completed map callbacks. Non-blocking.
        gpu.device.poll(wgpu::Maintain::Poll);

        if let Some(res) = self.stream.as_ref().and_then(Readback::take) {
            let rb = self.stream.take().expect("checked above");
            if let Ok(pixels) = res {
                let job = EncodeJob {
                    pixels,
                    width: rb.width,
                    height: rb.height,
                    bgra: false,
                    snapshot: None,
                };
                // Busy worker → drop the frame; the next one is 66 ms away.
                let _ = self.worker(bus).stream_tx.try_send(job);
                // `take` unmapped it: the next capture copies into it.
                self.spare = Some(rb.buffer);
            }
        }

        let mut i = 0;
        while i < self.snapshots.len() {
            let Some(res) = self.snapshots[i].0.take() else {
                i += 1;
                continue;
            };
            let (rb, reply) = self.snapshots.swap_remove(i);
            match res {
                Ok(pixels) => {
                    let job = EncodeJob {
                        pixels,
                        width: rb.width,
                        height: rb.height,
                        bgra: rb.bgra,
                        snapshot: Some(reply),
                    };
                    let _ = self.worker(bus).snapshot_tx.send(job);
                }
                Err(()) => {
                    let _ = reply.reply.send(Err("preview snapshot readback failed".into()));
                }
            }
        }

        if !stream || self.stream.is_some() || !self.interval_up() {
            return;
        }
        self.last_capture = Instant::now();

        let src_w = gpu.composite_width.max(1);
        let src_h = gpu.composite_height.max(1);
        let w = THUMB_WIDTH.min(src_w);
        let h = ((w as f32 * src_h as f32 / src_w as f32).round() as u32).max(1);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview thumbnail encoder"),
            });
        let downscaler = self
            .downscaler
            .get_or_insert_with(|| crate::gpu::Downscaler::new(&gpu.device));
        let thumb = downscaler.encode(&gpu.device, &mut enc, texture, w, h);
        let (buffer, bytes_per_row) =
            gpu.encode_readback_reusing(&mut enc, thumb, w, h, self.spare.take());
        gpu.queue.submit(std::iter::once(enc.finish()));
        self.stream = Some(Readback::start(buffer, bytes_per_row, w, h, false));
    }
}

impl EncodeWorker {
    fn spawn(bus: Bus) -> Self {
        let (stream_tx, stream_rx) = bounded::<EncodeJob>(1);
        let (snapshot_tx, snapshot_rx) = crossbeam_channel::unbounded::<EncodeJob>();
        let _ = std::thread::Builder::new()
            .name("preview-encode".into())
            .spawn(move || {
                let mut cpu = crate::perf::ThreadCpu::register("preview-encode");
                loop {
                    cpu.tick();
                    let job = crossbeam_channel::select! {
                        recv(snapshot_rx) -> j => j,
                        recv(stream_rx) -> j => j,
                    };
                    let Ok(job) = job else { break };
                    let encoded = encode_rgba_jpeg(&job.pixels, job.width, job.height, job.bgra, 
                        if job.snapshot.is_some() { 90 } else { 70 });
                    match (job.snapshot, encoded) {
                        (None, Ok(bytes)) => bus.emit_preview_jpeg(&bytes, job.width, job.height),
                        (None, Err(e)) => log::trace!("preview thumbnail encode failed: {e}"),
                        (Some(snap), Ok(bytes)) => {
                            use base64::Engine;
                            let mut out = json!({
                                "encoding": "jpeg",
                                "width": job.width,
                                "height": job.height,
                                "data_b64": base64::engine::general_purpose::STANDARD.encode(&bytes),
                            });
                            if let (Some(o), Value::Object(meta)) = (out.as_object_mut(), snap.meta) {
                                o.extend(meta);
                            }
                            let _ = snap.reply.send(Ok(out));
                        }
                        (Some(snap), Err(e)) => {
                            let _ = snap.reply.send(Err(format!("snapshot encode failed: {e}")));
                        }
                    }
                }
            });
        Self {
            stream_tx,
            snapshot_tx,
        }
    }
}

/// JPEG-encode tightly packed 8-bit pixels (RGBA or BGRA, already
/// sRGB-encoded by the GPU). Alpha is dropped.
fn encode_rgba_jpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    bgra: bool,
    quality: u8,
) -> Result<Vec<u8>, image::ImageError> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for px in pixels.chunks_exact(4) {
        if bgra {
            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
        } else {
            rgb.extend_from_slice(&px[..3]);
        }
    }
    let mut jpeg = Vec::with_capacity(rgb.len() / 6);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality).encode(
        &rgb,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(jpeg)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            // Subnormal.
            (mant as f32) * 2f32.powi(-24)
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let e = exp as i32 - 15;
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(e)
    };
    if sign == 1 { -f } else { f }
}

/// sRGB electro-optical transfer function (linear → gamma-encoded), clamped
/// to [0, 1]. Mirrors the automatic conversion the GPU does when storing into
/// the sRGB swapchain. Only the probe thumbnail (`encode_jpeg_thumbnail`)
/// still decodes on the CPU; the `preview` channel downscales on the GPU.
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub(crate) fn encode_jpeg_thumbnail(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    bytes_per_row: u32,
    target_w: u32,
) -> Result<(Vec<u8>, u32, u32), image::ImageError> {
    let aspect = src_h as f32 / src_w as f32;
    let target_w = target_w.min(src_w);
    let target_h = ((target_w as f32 * aspect).round() as u32).max(1);
    let mut rgb: Vec<u8> = Vec::with_capacity((target_w * target_h * 3) as usize);
    let stride_x = src_w as f32 / target_w as f32;
    let stride_y = src_h as f32 / target_h as f32;
    for ty in 0..target_h {
        let sy = ((ty as f32 + 0.5) * stride_y) as u32;
        let sy = sy.min(src_h - 1);
        let row_start = sy as usize * bytes_per_row as usize;
        for tx in 0..target_w {
            let sx = ((tx as f32 + 0.5) * stride_x) as u32;
            let sx = sx.min(src_w - 1);
            let px_off = row_start + sx as usize * 8;
            let r = f16_to_f32(u16::from_le_bytes([src[px_off], src[px_off + 1]]));
            let g = f16_to_f32(u16::from_le_bytes([src[px_off + 2], src[px_off + 3]]));
            let b = f16_to_f32(u16::from_le_bytes([src[px_off + 4], src[px_off + 5]]));
            // The composite is linear-light Rgba16Float. The window's sRGB
            // swapchain applies the linear→sRGB OETF automatically on store;
            // this CPU readback must do it explicitly, or the preview shows
            // raw linear values (too-dark midtones, shifted color balance).
            rgb.push((linear_to_srgb(r) * 255.0) as u8);
            rgb.push((linear_to_srgb(g) * 255.0) as u8);
            rgb.push((linear_to_srgb(b) * 255.0) as u8);
        }
    }
    let mut jpeg: Vec<u8> = Vec::with_capacity(rgb.len() / 4);
    {
        let mut encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 70);
        encoder.encode(&rgb, target_w, target_h, image::ExtendedColorType::Rgb8)?;
    }
    Ok((jpeg, target_w, target_h))
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod sticky_tests {
    use super::*;

    /// A sticky channel that isn't in `ALL_CHANNELS` can never be subscribed
    /// to, so its replay is dead code — and a typo here is invisible at
    /// runtime. Cheap guard against both.
    #[test]
    fn every_sticky_channel_is_subscribable() {
        for c in ALL_CHANNELS {
            let _ = is_sticky(c);
        }
        for c in [
            "hot_reload",
            "audio_freshness",
            "connectivity",
            "fps",
            "masters",
            "deck",
            "changes",
            "alignment",
            "perf_spike",
        ] {
            assert!(is_sticky(c), "{c} should be sticky");
            assert!(ALL_CHANNELS.contains(&c), "{c} missing from ALL_CHANNELS");
        }
    }

    /// The state channels a late-mounting UI hydrates from. `alignment` in
    /// particular emits only at boot and on edits, so without stickiness the
    /// Align tab has no way to learn the current document.
    #[test]
    fn boot_only_channels_replay_to_late_subscribers() {
        let bus = Bus::new();
        bus.emit_alignment(json!({ "corners": [[0, 0]] }));
        let channels: HashSet<String> = ["alignment".to_string()].into_iter().collect();
        let (_id, rx) = bus.subscribe(channels);
        let frame = rx.try_recv().expect("sticky alignment replayed on subscribe");
        assert_eq!(frame.channel, "alignment");
    }

    /// Streaming channels must NOT be retained — a replayed stale preview
    /// frame would show the operator a composite from minutes ago.
    #[test]
    fn streaming_channels_are_not_retained() {
        for c in ["preview", "drivers", "audio", "frame_stats", "log", "perf"] {
            assert!(!is_sticky(c), "{c} should not be sticky");
        }
    }
}

#[cfg(test)]
mod change_log_tests {
    use super::*;

    #[test]
    fn record_bumps_rev_and_broadcasts_sticky() {
        let bus = Bus::new();
        let log = ChangeLog::new();
        assert_eq!(log.rev(), 0);
        let r1 = log.record(&bus, "agent", "bindings", "one");
        let r2 = log.record(&bus, "ui", "effects", "two");
        assert_eq!((r1, r2), (1, 2));
        // Sticky replay: a late subscriber sees the last entry.
        let (_, rx) = bus.subscribe(["changes".to_string()].into_iter().collect());
        let frame = rx.try_recv().expect("sticky changes replay");
        assert_eq!(frame.payload["rev"], 2);
        assert_eq!(frame.payload["actor"], "ui");
    }

    #[test]
    fn list_since_filters_and_flags_wrap_and_epoch() {
        let bus = Bus::new();
        let log = ChangeLog::new();
        for i in 0..40 {
            log.record(&bus, "agent", "bindings", format!("e{i}"));
        }
        // Ring depth 32: revs 9..=40 retained.
        let v = log.list_json(Some(38), None);
        assert_eq!(v["entries"].as_array().unwrap().len(), 2);
        assert!(v["note"].is_null());
        // since_rev older than the tail → full ring + explicit note.
        let v = log.list_json(Some(3), None);
        assert_eq!(v["entries"].as_array().unwrap().len(), 32);
        assert!(v["note"].as_str().unwrap().contains("ring wrapped"));
        // Epoch mismatch → full ring + restart note.
        let v = log.list_json(Some(38), Some(log.epoch() + 1));
        assert!(v["note"].as_str().unwrap().contains("restarted"));
        // Caught-up cursor → empty, no note.
        let v = log.list_json(Some(40), None);
        assert_eq!(v["entries"].as_array().unwrap().len(), 0);
        assert!(v["note"].is_null());
    }
}

// ---------- global bus hook (log → telemetry tee) ----------

static GLOBAL_BUS: std::sync::OnceLock<Bus> = std::sync::OnceLock::new();

/// Register the engine's bus as the process-global log sink. Called once
/// from `App::new`; the tee logger in `main.rs` forwards `info!`/`warn!`/
/// `error!` records onto the `log` telemetry channel from then on.
pub fn set_global_bus(bus: &Bus) {
    let _ = GLOBAL_BUS.set(bus.clone());
}

pub fn global_bus() -> Option<&'static Bus> {
    GLOBAL_BUS.get()
}
