//! Tier-A always-on frame profiler (perf-monitor plan PM1).
//!
//! `fps` / `frame_stats` time mark-to-mark, which includes the pacing sleep
//! and the swapchain wait — a 2 ms frame and a 15 ms frame both read "60".
//! This module times **what the frame spends its time on**: every
//! render-thread stage of the per-frame host contract (`poll_inbound` →
//! `pace_frame` → `redraw` | `render_offscreen_frame`) is a fixed [`Stage`],
//! and `cpu.work = Σ stages − acquire − pace` gives the headroom number
//! that fps can't.
//!
//! **Hot-path rules** (reference §6 "never block the render thread"): the
//! per-frame path is clock reads into a fixed `[u32; Stage::COUNT]`, one
//! ring slot write, and a handful of relaxed atomics — no strings, no
//! allocation, no locks. Everything that allocates (percentiles, JSON, the
//! registries' mutexes) runs at the 2 Hz aggregation inside the
//! `telemetry` stage, like every other emitter.
//!
//! Side threads report through process-global, lock-free slots:
//! [`ThreadCpu`] (each owned thread samples its own
//! `CLOCK_THREAD_CPUTIME_ID` ~1 Hz), [`ClipStats`] (decode ms + stale
//! frames per clip stream) and [`count_telemetry`] (bytes per channel at
//! the WS and webview hops). Their registries are only locked when a
//! thread/clip registers and at aggregation.
//!
//! GPU pass times (PM2) arrive from `gpu_timing.rs` as [`GpuFrame`]s of raw
//! ticks; `gpu` is `null` when GPU timing is off or unsupported.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::telemetry::{Bus, ALL_CHANNELS};

/// Frame budget the headroom number is measured against (60 Hz). The
/// projector's real refresh isn't known to the engine; every gig gate is
/// phrased at 60 fps.
pub const BUDGET_HZ: f64 = 60.0;
/// Frames held in the ring — ~5 s at 60 Hz. Aggregates and spike
/// neighbourhoods both read from it.
pub const RING_FRAMES: usize = 300;
/// Frames either side of a spike frozen into the `perf_spike` payload.
const SPIKE_NEIGHBOURS: usize = 5;
/// Boot frames (shader compiles, first uploads) never trigger a spike.
const SPIKE_ARM_AFTER_FRAMES: u64 = 60;
/// At most one spike capture per this window; the rest are counted.
const SPIKE_MIN_GAP: Duration = Duration::from_secs(2);
/// Command kinds remembered per frame for the spike report.
const CMDS_PER_FRAME: usize = 4;
const EMIT_EVERY: Duration = Duration::from_millis(500);
/// Resource counts + GPU-MB inventory + RSS are refreshed at 1 Hz.
const SLOW_EVERY: Duration = Duration::from_secs(1);
/// A thread slot not re-sampled for this long reports 0 % (it's parked
/// on a blocking call, so it isn't burning CPU).
const THREAD_STALE_MS: u64 = 3_000;

// ---------------------------------------------------------------------------
// Stage taxonomy (plan §2.1) — the contract that keeps attribution stable.
// A new feature adds a stage here rather than hiding inside an existing one.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    /// `poll_inbound` command drain (`rpc::handle` — scene loads land here).
    InboundCmds,
    /// Watcher poll + `reload_scene` / `reload_effects`.
    InboundWatch,
    /// `step_probe` (§5.6 pre-flight probe).
    InboundProbe,
    /// Design autosave, audio pill, session/alignment debounced writes.
    InboundMisc,
    /// `emit_periodic_telemetry`, `FpsAccumulator`, this module's emit.
    Telemetry,
    /// Transport step + promote state machine + live `PassPlan::tick`.
    TickLive,
    /// Design `PassPlan::tick`.
    TickDesign,
    /// Warp bake, clip uploads, composites, final pass — command encoding.
    Encode,
    /// `surface.get_current_texture()` — backpressure, not work.
    Acquire,
    /// `queue.submit` + handing consumed clip slots back.
    Submit,
    /// `frame.present()`.
    Present,
    /// Native preview blit.
    PreviewPresent,
    /// Demand-gated JPEG readback.
    PreviewJpeg,
    /// `pace_frame` sleep — idle, not work.
    Pace,
    /// Render-thread time *between* Core calls: the host's event-loop
    /// iteration (winit dispatch) or loop glue (TauriHost size/preview
    /// polling, occlusion `msg_send`). Work, but not Core's.
    Host,
}

impl Stage {
    pub const COUNT: usize = 15;
    pub const ALL: [Stage; Stage::COUNT] = [
        Stage::InboundCmds,
        Stage::InboundWatch,
        Stage::InboundProbe,
        Stage::InboundMisc,
        Stage::Telemetry,
        Stage::TickLive,
        Stage::TickDesign,
        Stage::Encode,
        Stage::Acquire,
        Stage::Submit,
        Stage::Present,
        Stage::PreviewPresent,
        Stage::PreviewJpeg,
        Stage::Pace,
        Stage::Host,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::InboundCmds => "inbound.cmds",
            Stage::InboundWatch => "inbound.watch",
            Stage::InboundProbe => "inbound.probe",
            Stage::InboundMisc => "inbound.misc",
            Stage::Telemetry => "telemetry",
            Stage::TickLive => "tick.live",
            Stage::TickDesign => "tick.design",
            Stage::Encode => "encode",
            Stage::Acquire => "acquire",
            Stage::Submit => "submit",
            Stage::Present => "present",
            Stage::PreviewPresent => "preview.present",
            Stage::PreviewJpeg => "preview.jpeg",
            Stage::Pace => "pace",
            Stage::Host => "host",
        }
    }

    /// `acquire` (vsync backpressure) and `pace` (deliberate sleep) are
    /// time the frame waits, not work it does.
    pub fn is_work(self) -> bool {
        !matches!(self, Stage::Acquire | Stage::Pace)
    }
}

/// GPU passes timed with timestamp queries (PM2, `gpu_timing.rs`). Only
/// passes that already exist — the clip upload copies stay untimed (see
/// `gpu_timing.rs` for why).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuPass {
    /// §5.14 warp LUT bake (edit-triggered, rare).
    WarpBake,
    CompositeLive,
    CompositeDesign,
    /// Projector final pass (presented frames only).
    Final,
    /// Native preview blit (its own submit).
    Preview,
    /// §5.6 pre-flight probe frames (their own submits) — must stay last:
    /// its query pairs follow the fixed passes.
    Probe,
}

impl GpuPass {
    pub const COUNT: usize = 6;
    pub const ALL: [GpuPass; GpuPass::COUNT] = [
        GpuPass::WarpBake,
        GpuPass::CompositeLive,
        GpuPass::CompositeDesign,
        GpuPass::Final,
        GpuPass::Preview,
        GpuPass::Probe,
    ];

    pub fn name(self) -> &'static str {
        match self {
            GpuPass::WarpBake => "warp_bake",
            GpuPass::CompositeLive => "composite.live",
            GpuPass::CompositeDesign => "composite.design",
            GpuPass::Final => "final",
            GpuPass::Preview => "preview",
            GpuPass::Probe => "probe",
        }
    }
}

/// One timed GPU frame in raw timestamp ticks (× `period_ns` → ns at
/// aggregation). Arrives 2–3 frames after the CPU frame it belongs to.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuFrame {
    pub ticks: [u32; GpuPass::COUNT],
    /// Bit per [`GpuPass`] that ran (and was timed) this frame.
    pub ran: u8,
    /// Σ over the frame's submits of first-begin → last-end.
    pub total: u32,
}

/// GPU timing state as `Perf` sees it — `None` in `Perf` = `gpu: null`.
struct GpuTiming {
    period_ns: f32,
    every: u32,
    frames: Ring<GpuFrame, RING_FRAMES>,
    /// Sampled frames that found every readback slot in flight.
    skipped: u64,
}

impl GpuTiming {
    /// Samples the aggregate reads: ~10 s worth at the sampling stride
    /// (20 at the default 1-in-30), never fewer than 16, at most the ring.
    fn window(&self) -> usize {
        (2 * RING_FRAMES / self.every.max(1) as usize).clamp(16, RING_FRAMES)
    }

    /// Newest `n` samples (newest first).
    fn recent(&self, n: usize) -> impl Iterator<Item = &GpuFrame> {
        (0..n.min(self.frames.len())).filter_map(move |i| self.frames.back(i))
    }
}

// ---------------------------------------------------------------------------
// Per-frame record + ring
// ---------------------------------------------------------------------------

/// What ran in a frame besides rendering — the spike report's "why".
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameEvents {
    pub cmds: [&'static str; CMDS_PER_FRAME],
    pub cmd_count: u16,
    pub reloads: u8,
    pub probe_steps: u8,
    pub clip_opens: u16,
    /// Design rev at frame end (§2.7 change ring).
    pub rev: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameRecord {
    pub stages_us: [u32; Stage::COUNT],
    /// Wall time since the previous frame's end (0 for the first frame).
    pub interval_us: u32,
    pub presenting: bool,
    /// The monitor's own cost in this frame (frame close + any aggregation).
    pub monitor_us: u32,
    pub events: FrameEvents,
}

impl FrameRecord {
    pub fn work_us(&self) -> u32 {
        Stage::ALL
            .iter()
            .filter(|s| s.is_work())
            .map(|s| self.stages_us[*s as usize])
            .sum()
    }

    pub fn total_us(&self) -> u32 {
        self.stages_us.iter().sum()
    }

    /// Interval time no stage covered (host glue, event-loop dispatch).
    /// The PM1 gate: this stays under 0.2 ms.
    pub fn unaccounted_us(&self) -> u32 {
        self.interval_us.saturating_sub(self.total_us())
    }
}

/// Fixed-capacity ring, allocated once.
struct Ring<T: Copy + Default, const N: usize> {
    buf: Box<[T; N]>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> Ring<T, N> {
    fn new() -> Self {
        Self {
            buf: Box::new([T::default(); N]),
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, v: T) {
        self.buf[self.head] = v;
        self.head = (self.head + 1) % N;
        self.len = (self.len + 1).min(N);
    }

    /// `back(0)` is the newest element.
    fn back(&self, i: usize) -> Option<&T> {
        (i < self.len).then(|| &self.buf[(self.head + N - 1 - i) % N])
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).rev().filter_map(move |i| self.back(i))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// p50 / p95 / max over µs samples, reported in ms. Sorts `scratch` in
/// place (reused across calls — aggregation-time only).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pct {
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
}

impl Pct {
    pub fn of(scratch: &mut [u32]) -> Self {
        if scratch.is_empty() {
            return Self::default();
        }
        scratch.sort_unstable();
        let pick = |q: f64| {
            // Nearest-rank: the smallest sample with ≥ q of the set at or
            // below it.
            let rank = ((q * scratch.len() as f64).ceil() as usize).clamp(1, scratch.len());
            scratch[rank - 1]
        };
        Self {
            p50: pick(0.50) as f32 / 1000.0,
            p95: pick(0.95) as f32 / 1000.0,
            max: *scratch.last().expect("non-empty") as f32 / 1000.0,
        }
    }

    fn json(self) -> Value {
        json!({ "p50": round3(self.p50), "p95": round3(self.p95), "max": round3(self.max) })
    }
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn us(d: Duration) -> u32 {
    d.as_micros().min(u32::MAX as u128) as u32
}

/// GPU ticks → µs (aggregation time only).
fn ticks_us(ticks: u32, period_ns: f64) -> u32 {
    (ticks as f64 * period_ns / 1000.0).round().min(u32::MAX as f64) as u32
}

// ---------------------------------------------------------------------------
// Shared handle (RPC surface + inline writers)
// ---------------------------------------------------------------------------

/// The cross-thread half: inline RPC writers stamp knob writes into it,
/// `perf.setTrigger` writes the trigger, `perf.snapshot` reads the latest
/// aggregate. Lives in `RpcContext`.
pub struct Shared {
    enabled: bool,
    /// Monotonic µs of the oldest knob write not yet picked up by a tick
    /// (0 = none pending).
    knob_pending_us: AtomicU64,
    trigger_us: AtomicU32,
    /// Latest 2 Hz aggregate — written by the render thread at emit time
    /// (the same cadence it already takes the bus lock at).
    snapshot: Mutex<Option<Value>>,
}

impl Shared {
    pub fn new(enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            enabled,
            knob_pending_us: AtomicU64::new(0),
            trigger_us: AtomicU32::new((2.0 * 1_000_000.0 / BUDGET_HZ) as u32),
            snapshot: Mutex::new(None),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// `param.set` / `master.set` landed. Keeps the *oldest* pending write
    /// so the latency sample is the worst one the next frame resolves.
    pub fn knob_written(&self) {
        if self.enabled {
            let _ = self.knob_pending_us.compare_exchange(
                0,
                mono_us(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
        }
    }

    pub fn trigger_ms(&self) -> f32 {
        self.trigger_us.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// `perf.setTrigger {ms}`.
    pub fn set_trigger_ms(&self, ms: f32) -> Result<f32, String> {
        if !ms.is_finite() || ms <= 0.0 || ms > 10_000.0 {
            return Err("perf.setTrigger: ms must be a finite number in (0, 10000]".into());
        }
        self.trigger_us.store((ms * 1000.0) as u32, Ordering::Relaxed);
        Ok(self.trigger_ms())
    }

    /// `perf.snapshot`.
    pub fn snapshot(&self) -> Result<Value, String> {
        if !self.enabled {
            return Err("perf monitor is off (engine started with --perf off)".into());
        }
        self.snapshot
            .lock()
            .ok()
            .and_then(|s| s.clone())
            .ok_or_else(|| "no perf aggregate yet — the first one lands ~0.5 s after the first frame".into())
    }
}

/// Process-wide monotonic µs, nonzero (0 is the "none" sentinel above).
fn mono_us() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64 + 1
}

fn wall_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Process-wide switch for the side-thread probes (clip decode timing,
/// thread CPU, telemetry byte counts). Set by `Core::new` from `--perf`.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Render-thread recorder
// ---------------------------------------------------------------------------

/// A started span. `None` when the monitor is off, so a disabled monitor
/// costs one branch per call site.
#[derive(Clone, Copy)]
pub struct Span(Option<Instant>);

/// A spike waiting for its trailing neighbours before it is frozen.
struct PendingSpike {
    frames_after: usize,
}

/// One frozen spike: the trigger frame ± neighbours, oldest first.
struct SpikeCapture {
    frames: [FrameRecord; 2 * SPIKE_NEIGHBOURS + 1],
    len: usize,
    /// Index of the trigger frame inside `frames`.
    at: usize,
    trigger_ms: f32,
    ts_ms: u64,
}

/// Small fixed sample ring for latency metrics.
type Samples = Ring<u32, 128>;

pub struct Perf {
    on: bool,
    shared: Arc<Shared>,
    budget_us: u32,

    cur: FrameRecord,
    last_frame_end: Option<Instant>,
    /// When the last span closed — the start of a possible host gap.
    last_mark: Option<Instant>,
    frames: Ring<FrameRecord, RING_FRAMES>,
    frames_total: u64,
    dropped_total: u64,

    knob_inflight_us: u64,
    knob_submit: Samples,
    rpc_queue: Samples,
    osc_age: Samples,

    pending_spike: Option<PendingSpike>,
    ready_spike: Option<Box<SpikeCapture>>,
    last_spike_at: Option<Instant>,
    spikes_total: u64,
    spikes_suppressed: u64,

    clip_opens_seen: u64,
    render_cpu: ThreadCpu,
    process: ProcessSampler,
    telemetry: TelemetryRates,

    last_emit: Instant,
    last_slow: Option<Instant>,
    /// Slow-path values the host supplies at 1 Hz (needs the GPU context).
    resources: Value,
    gpu_mb: Option<f64>,
    rss_mb: Option<f64>,
    /// PM2 — `None` until a `GpuTimer` exists (`perf.gpu` is then `null`).
    gpu: Option<Box<GpuTiming>>,
    /// PM3 `--bench` — the whole measured window (`None` outside a bench).
    bench: Option<Box<BenchLog>>,

    scratch: Vec<u32>,
}

/// PM3 `--bench`: every record of the measured window, kept whole — the
/// ring only holds ~5 s and the latency rings 128 samples. Allocated once
/// at [`Perf::bench_begin`]; pushes stay allocation-free up to `frames`.
struct BenchLog {
    frames: Vec<FrameRecord>,
    gpu: Vec<GpuFrame>,
    gpu_skipped_at_start: u64,
    knob_submit: Vec<u32>,
    rpc_queue: Vec<u32>,
    stale_at_start: u64,
    started: Instant,
}

impl Perf {
    pub fn new(on: bool) -> Self {
        set_enabled(on);
        Self {
            on,
            shared: Shared::new(on),
            budget_us: (1_000_000.0 / BUDGET_HZ) as u32,
            cur: FrameRecord::default(),
            last_frame_end: None,
            last_mark: None,
            frames: Ring::new(),
            frames_total: 0,
            dropped_total: 0,
            knob_inflight_us: 0,
            knob_submit: Ring::new(),
            rpc_queue: Ring::new(),
            osc_age: Ring::new(),
            pending_spike: None,
            ready_spike: None,
            last_spike_at: None,
            spikes_total: 0,
            spikes_suppressed: 0,
            clip_opens_seen: clip_opens(),
            render_cpu: ThreadCpu::register("render"),
            process: ProcessSampler::default(),
            telemetry: TelemetryRates::default(),
            last_emit: Instant::now(),
            last_slow: None,
            resources: Value::Null,
            gpu_mb: None,
            rss_mb: None,
            gpu: None,
            bench: None,
            scratch: Vec::with_capacity(RING_FRAMES),
        }
    }

    pub fn on(&self) -> bool {
        self.on
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// Open a span inside a Core call (nested in an entry span's frame).
    #[inline]
    pub fn begin(&self) -> Span {
        Span(self.on.then(Instant::now))
    }

    /// Open a span at a **host entry point** (`poll_inbound`, `pace_frame`,
    /// `redraw`, `render_offscreen_frame`): the time since the previous span
    /// closed is the host's, charged to [`Stage::Host`]. This is what makes
    /// the stages tile the whole frame interval.
    #[inline]
    pub fn enter(&mut self) -> Span {
        if !self.on {
            return Span(None);
        }
        let now = Instant::now();
        if let Some(t) = self.last_mark {
            self.add(Stage::Host, now.duration_since(t));
        }
        Span(Some(now))
    }

    /// Close `span` into `stage` (accumulates — a stage may run in several
    /// places in one frame).
    #[inline]
    pub fn end(&mut self, stage: Stage, span: Span) {
        if let Some(t) = span.0 {
            let now = Instant::now();
            self.add(stage, now.duration_since(t));
            self.last_mark = Some(now);
        }
    }

    /// Close `span` into `stage` and open the next one with the same clock
    /// read, so back-to-back stages tile the frame without gaps.
    #[inline]
    pub fn lap(&mut self, stage: Stage, span: Span) -> Span {
        match span.0 {
            Some(t) => {
                let now = Instant::now();
                self.add(stage, now.duration_since(t));
                self.last_mark = Some(now);
                Span(Some(now))
            }
            None => span,
        }
    }

    #[inline]
    fn add(&mut self, stage: Stage, d: Duration) {
        let slot = &mut self.cur.stages_us[stage as usize];
        *slot = slot.saturating_add(us(d));
    }

    // ----- events + latency stamps (render thread) -----

    /// One queued `EngineCommand` about to be handled: its kind (for the
    /// spike report) and how long it waited in the queue.
    pub fn note_cmd(&mut self, kind: &'static str, queued: Duration) {
        if !self.on {
            return;
        }
        let ev = &mut self.cur.events;
        if (ev.cmd_count as usize) < CMDS_PER_FRAME {
            ev.cmds[ev.cmd_count as usize] = kind;
        }
        ev.cmd_count = ev.cmd_count.saturating_add(1);
        self.rpc_queue.push(us(queued));
        if let Some(b) = self.bench.as_mut() {
            b.rpc_queue.push(us(queued));
        }
    }

    pub fn note_reload(&mut self) {
        self.cur.events.reloads = self.cur.events.reloads.saturating_add(1);
    }

    pub fn note_probe_step(&mut self) {
        self.cur.events.probe_steps = self.cur.events.probe_steps.saturating_add(1);
    }

    /// Before the legs tick: pick up the oldest knob write this tick reads,
    /// and sample the age of the newest OSC packet (`None` = none yet).
    pub fn before_tick(&mut self, osc_packet_age_ms: Option<u64>) {
        if !self.on {
            return;
        }
        if self.knob_inflight_us == 0 {
            self.knob_inflight_us = self.shared.knob_pending_us.swap(0, Ordering::AcqRel);
        }
        if let Some(age) = osc_packet_age_ms {
            self.osc_age.push((age.min(u32::MAX as u64 / 1000) * 1000) as u32);
        }
    }

    /// After the submit that carried the ticked values.
    pub fn after_submit(&mut self) {
        if self.knob_inflight_us != 0 {
            let lat = mono_us().saturating_sub(self.knob_inflight_us);
            self.knob_submit.push(lat.min(u32::MAX as u64) as u32);
            if let Some(b) = self.bench.as_mut() {
                b.knob_submit.push(lat.min(u32::MAX as u64) as u32);
            }
            self.knob_inflight_us = 0;
        }
    }

    /// Close the frame: interval, dropped-frame + spike detection, ring
    /// push. Called once at the end of `redraw` / `render_offscreen_frame`.
    pub fn end_frame(&mut self, presenting: bool, rev: u64) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        let opens = clip_opens();
        self.cur.events.clip_opens = opens.saturating_sub(self.clip_opens_seen).min(u16::MAX as u64) as u16;
        self.clip_opens_seen = opens;
        self.cur.events.rev = rev;
        self.cur.presenting = presenting;
        let first = self.last_frame_end.is_none();
        self.cur.interval_us = self.last_frame_end.map_or(0, |t| us(now.duration_since(t)));
        self.last_frame_end = Some(now);
        self.render_cpu.tick_at(now);

        self.close_record(first, now);
        self.cur = FrameRecord::default();
        // The close itself is monitor cost, carried into the next frame;
        // the host gap starts after it.
        let done = Instant::now();
        self.cur.monitor_us = us(done.duration_since(now));
        self.last_mark = Some(done);
    }

    /// Ring push + detectors. Split out (and clock-free) for the tests.
    fn close_record(&mut self, first: bool, now: Instant) {
        let record = self.cur;
        if first {
            // No interval to judge; keep it out of the ring.
            return;
        }
        self.frames_total += 1;
        if record.presenting && record.interval_us > self.budget_us * 3 / 2 {
            self.dropped_total += 1;
        }
        self.frames.push(record);
        if let Some(b) = self.bench.as_mut() {
            b.frames.push(record);
        }

        if let Some(p) = self.pending_spike.as_mut() {
            p.frames_after += 1;
            if p.frames_after >= SPIKE_NEIGHBOURS {
                self.pending_spike = None;
                self.freeze_spike(SPIKE_NEIGHBOURS);
            }
        } else {
            let trigger = self.shared.trigger_us.load(Ordering::Relaxed);
            if self.frames_total > SPIKE_ARM_AFTER_FRAMES && record.interval_us > trigger {
                let recent = self
                    .last_spike_at
                    .is_some_and(|t| now.duration_since(t) < SPIKE_MIN_GAP);
                if recent {
                    self.spikes_suppressed += 1;
                } else {
                    self.last_spike_at = Some(now);
                    self.spikes_total += 1;
                    self.pending_spike = Some(PendingSpike { frames_after: 0 });
                }
            }
        }
    }

    /// Copy the trigger frame (`after` frames back from the newest) and its
    /// neighbours out of the ring. Fixed-size copy, no allocation beyond
    /// the one box.
    fn freeze_spike(&mut self, after: usize) {
        let mut cap = SpikeCapture {
            frames: [FrameRecord::default(); 2 * SPIKE_NEIGHBOURS + 1],
            len: 0,
            at: 0,
            trigger_ms: self.shared.trigger_ms(),
            ts_ms: wall_ms(),
        };
        let oldest = after + SPIKE_NEIGHBOURS;
        for back in (0..=oldest).rev() {
            if let Some(r) = self.frames.back(back) {
                if back == after {
                    cap.at = cap.len;
                }
                cap.frames[cap.len] = *r;
                cap.len += 1;
            }
        }
        self.ready_spike = Some(Box::new(cap));
    }

    // ----- GPU timing (PM2) -----

    /// A `GpuTimer` exists: from now on `perf.gpu` reports. `period_ns`
    /// converts ticks; `every` is the sampling stride (1 = every frame).
    pub fn set_gpu_timing(&mut self, period_ns: f32, every: u32) {
        if self.on {
            self.gpu = Some(Box::new(GpuTiming {
                period_ns,
                every,
                frames: Ring::new(),
                skipped: 0,
            }));
        }
    }

    /// One read-back GPU frame (render thread, after a submit).
    pub fn push_gpu(&mut self, frame: GpuFrame) {
        if let Some(g) = self.gpu.as_mut() {
            g.frames.push(frame);
            if let Some(b) = self.bench.as_mut() {
                b.gpu.push(frame);
            }
        }
    }

    /// A sampled frame found every readback slot in flight — not timed.
    pub fn note_gpu_skipped(&mut self) {
        if let Some(g) = self.gpu.as_mut() {
            g.skipped += 1;
        }
    }

    /// p50/p95/max (ms) of `pass` over the newest `frames` timed GPU frames
    /// in which it ran — `None` without GPU timing or samples. The per-leg
    /// composite accessor the E2 probation window reads
    /// (`GpuPass::CompositeLive` / `CompositeDesign`): unlike frame
    /// interval, a TAKE crossfade rendering both legs doesn't move it.
    pub fn gpu_pass_recent(&mut self, pass: GpuPass, frames: usize) -> Option<Pct> {
        let g = self.gpu.as_ref()?;
        let bit = 1u8 << pass as u8;
        let ns = g.period_ns as f64;
        self.scratch.clear();
        self.scratch.extend(
            g.recent(frames)
                .filter(|f| f.ran & bit != 0)
                .map(|f| ticks_us(f.ticks[pass as usize], ns)),
        );
        (!self.scratch.is_empty()).then(|| Pct::of(&mut self.scratch))
    }

    /// The `perf.gpu` object, or `null`.
    fn gpu_json(&mut self) -> (Value, Option<Pct>) {
        let Some(g) = self.gpu.as_ref() else {
            return (Value::Null, None);
        };
        let ns = g.period_ns as f64;
        let window = g.window();
        self.scratch.clear();
        self.scratch.extend(g.recent(window).map(|f| ticks_us(f.total, ns)));
        let n = self.scratch.len();
        let total = (n > 0).then(|| Pct::of(&mut self.scratch));
        let mut passes = Map::new();
        for pass in GpuPass::ALL {
            let bit = 1u8 << pass as u8;
            self.scratch.clear();
            self.scratch.extend(
                g.recent(window)
                    .filter(|f| f.ran & bit != 0)
                    .map(|f| ticks_us(f.ticks[pass as usize], ns)),
            );
            if self.scratch.is_empty() {
                continue;
            }
            let count = self.scratch.len();
            let mut row = Pct::of(&mut self.scratch).json();
            row["n"] = json!(count);
            passes.insert(pass.name().into(), row);
        }
        let json = json!({
            "every": g.every,
            "frames": n,
            "skipped": g.skipped,
            "total": total.map(Pct::json),
            "passes": passes,
        });
        (json, total)
    }

    // ----- aggregation (2 Hz, inside the `telemetry` stage) -----

    /// True when the 1 Hz slow inputs (resource counts, GPU-MB) are due;
    /// the host then calls [`Perf::set_slow_inputs`] before [`Perf::emit`].
    pub fn slow_due(&self) -> bool {
        self.on && self.last_slow.is_none_or(|t| t.elapsed() >= SLOW_EVERY)
    }

    pub fn set_slow_inputs(&mut self, resources: Value, gpu_bytes: Option<u64>) {
        self.resources = resources;
        self.gpu_mb = gpu_bytes.map(|b| (b as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0);
        self.rss_mb = rss_bytes().map(|b| (b as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0);
        self.last_slow = Some(Instant::now());
    }

    pub fn emit_due(&self) -> bool {
        self.on && self.last_emit.elapsed() >= EMIT_EVERY
    }

    /// Emit a pending spike (sticky `perf_spike` + one log line) and, when
    /// due, the `perf` aggregate. `last_change` is the newest §2.7 change
    /// entry, if any.
    pub fn emit(&mut self, bus: &Bus, last_change: Option<Value>) {
        if !self.on {
            return;
        }
        let t0 = Instant::now();
        if let Some(spike) = self.ready_spike.take() {
            let (payload, line) = spike_payload(&spike, self.budget_us, self.spikes_suppressed);
            log::warn!("{line}");
            bus.emit("perf_spike", payload);
        }
        if self.emit_due() {
            self.last_emit = t0;
            let payload = self.aggregate(last_change);
            if let Ok(mut s) = self.shared.snapshot.lock() {
                *s = Some(payload.clone());
            }
            bus.emit("perf", payload);
        }
        self.cur.monitor_us = self.cur.monitor_us.saturating_add(us(t0.elapsed()));
    }

    // ----- PM3 bench window -----

    /// Start recording the measured window (drops any previous one).
    /// `frames` sizes the buffers so the frame path never reallocates.
    pub fn bench_begin(&mut self, frames: usize) {
        if !self.on {
            return;
        }
        self.bench = Some(Box::new(BenchLog {
            frames: Vec::with_capacity(frames + 16),
            gpu: Vec::with_capacity(frames + 16),
            gpu_skipped_at_start: self.gpu.as_ref().map_or(0, |g| g.skipped),
            knob_submit: Vec::with_capacity(frames * 2 + 16),
            rpc_queue: Vec::with_capacity(1024),
            stale_at_start: STALE_TOTAL.load(Ordering::Relaxed),
            started: Instant::now(),
        }));
    }

    /// Frames recorded since [`Perf::bench_begin`].
    pub fn bench_frames(&self) -> usize {
        self.bench.as_ref().map_or(0, |b| b.frames.len())
    }

    /// The Tier-A aggregate over the whole measured window (same stage /
    /// pass / percentile definitions as the `perf` channel), then stop
    /// recording. `null` when no window was recorded.
    pub fn bench_report(&mut self) -> Value {
        let Some(b) = self.bench.take() else { return Value::Null };
        let mut scratch: Vec<u32> = Vec::with_capacity(b.frames.len().max(b.knob_submit.len()));
        let mut pct = |vals: &mut dyn Iterator<Item = u32>| {
            scratch.clear();
            scratch.extend(vals);
            (!scratch.is_empty()).then(|| Pct::of(&mut scratch))
        };
        let mut cpu = Map::new();
        for stage in Stage::ALL {
            let p = pct(&mut b.frames.iter().map(|r| r.stages_us[stage as usize])).unwrap_or_default();
            cpu.insert(stage.name().into(), p.json());
        }
        let row = |p: Option<Pct>| p.map_or(Value::Null, Pct::json);
        let frame = json!({
            "interval": row(pct(&mut b.frames.iter().map(|r| r.interval_us))),
            "work": row(pct(&mut b.frames.iter().map(FrameRecord::work_us))),
            "unaccounted": row(pct(&mut b.frames.iter().map(FrameRecord::unaccounted_us))),
            "monitor": row(pct(&mut b.frames.iter().map(|r| r.monitor_us))),
        });
        let gpu = match self.gpu.as_ref() {
            None => Value::Null,
            Some(g) => {
                let ns = g.period_ns as f64;
                let mut passes = Map::new();
                for pass in GpuPass::ALL {
                    let bit = 1u8 << pass as u8;
                    let mut it = b
                        .gpu
                        .iter()
                        .filter(|f| f.ran & bit != 0)
                        .map(|f| ticks_us(f.ticks[pass as usize], ns));
                    let n = b.gpu.iter().filter(|f| f.ran & bit != 0).count();
                    if let Some(p) = pct(&mut it) {
                        let mut r = p.json();
                        r["n"] = json!(n);
                        passes.insert(pass.name().into(), r);
                    }
                }
                json!({
                    "every": g.every,
                    "frames": b.gpu.len(),
                    "skipped": g.skipped.saturating_sub(b.gpu_skipped_at_start),
                    "total": row(pct(&mut b.gpu.iter().map(|f| ticks_us(f.total, ns)))),
                    "passes": passes,
                })
            }
        };
        let lat = |p: Option<Pct>, n: usize| {
            p.map_or(Value::Null, |p| {
                let mut r = p.json();
                r["n"] = json!(n);
                r
            })
        };
        let knob = lat(pct(&mut b.knob_submit.iter().copied()), b.knob_submit.len());
        let queue = lat(pct(&mut b.rpc_queue.iter().copied()), b.rpc_queue.len());
        let dropped = b
            .frames
            .iter()
            .filter(|r| r.presenting && r.interval_us > self.budget_us * 3 / 2)
            .count();
        let (streams, decode_p95) = clip_decode_summary();
        let seconds = b.started.elapsed().as_secs_f64();
        json!({
            "frames": b.frames.len(),
            "seconds": (seconds * 1000.0).round() / 1000.0,
            "fps": ((b.frames.len() as f64 / seconds.max(1e-6)) * 10.0).round() / 10.0,
            "budget_ms": round3(self.budget_us as f32 / 1000.0),
            "cpu": cpu,
            "frame": frame,
            "dropped": dropped,
            "gpu": gpu,
            "latency": { "knob_submit": knob, "rpc_queue": queue },
            "clips": {
                "streams": streams,
                "decode_p95": decode_p95.map(round3),
                "stale": STALE_TOTAL.load(Ordering::Relaxed).saturating_sub(b.stale_at_start),
            },
        })
    }

    fn pct_of(&mut self, f: impl Fn(&FrameRecord) -> u32) -> Pct {
        self.scratch.clear();
        self.scratch.extend(self.frames.iter().map(&f));
        Pct::of(&mut self.scratch)
    }

    fn pct_samples(scratch: &mut Vec<u32>, ring: &Samples) -> Option<Pct> {
        if ring.len() == 0 {
            return None;
        }
        scratch.clear();
        scratch.extend(ring.iter().copied());
        Some(Pct::of(scratch))
    }

    /// The `perf` payload (plan §2.2). Public for `perf.snapshot` parity and
    /// the tests.
    pub fn aggregate(&mut self, last_change: Option<Value>) -> Value {
        let budget_ms = self.budget_us as f32 / 1000.0;
        let mut cpu = Map::new();
        for stage in Stage::ALL {
            let p = self.pct_of(|r| r.stages_us[stage as usize]);
            cpu.insert(stage.name().into(), p.json());
        }
        let work = self.pct_of(FrameRecord::work_us);
        let interval = self.pct_of(|r| r.interval_us);
        let unaccounted = self.pct_of(FrameRecord::unaccounted_us);
        let monitor = self.pct_of(|r| r.monitor_us);
        let n = self.frames.len();
        let presenting = self.frames.back(0).is_some_and(|r| r.presenting);
        let dropped_window = self
            .frames
            .iter()
            .filter(|r| r.presenting && r.interval_us > self.budget_us * 3 / 2)
            .count();
        let (gpu, gpu_total) = self.gpu_json();
        // Whichever side is closer to the budget sets the headroom.
        let busiest = gpu_total.map_or(work.p95, |g| g.p95.max(work.p95));
        let headroom = if n == 0 { None } else { Some(round3(1.0 - busiest / budget_ms)) };

        let knob = Self::pct_samples(&mut self.scratch, &self.knob_submit).map(|p| round3(p.p95));
        let queue = Self::pct_samples(&mut self.scratch, &self.rpc_queue).map(|p| round3(p.p95));
        let osc = Self::pct_samples(&mut self.scratch, &self.osc_age).map(|p| round3(p.p50));
        let (clips, clips_idle) = clip_rows(&mut self.scratch);

        json!({
            "budget_ms": round3(budget_ms),
            "headroom": headroom,
            "presenting": presenting,
            "frames": n,
            "cpu": cpu,
            "frame": {
                "interval": interval.json(),
                "work": work.json(),
                "unaccounted": unaccounted.json(),
                "monitor": monitor.json(),
            },
            "dropped": { "window": dropped_window, "total": self.dropped_total },
            "gpu": gpu,
            "threads": thread_rows(),
            "clips": clips,
            "clips_idle": clips_idle,
            "latency": {
                "knob_submit_p95": knob,
                "rpc_queue_p95": queue,
                "osc_age_p50": osc,
            },
            "process": {
                "cpu_pct": self.process.sample(),
                "rss_mb": self.rss_mb,
                "gpu_mb": self.gpu_mb,
                "resources": self.resources,
            },
            "telemetry_bps": self.telemetry.sample(),
            "last_change": last_change,
            "spikes": {
                "trigger_ms": round3(self.shared.trigger_ms()),
                "total": self.spikes_total,
                "suppressed": self.spikes_suppressed,
            },
        })
    }
}

fn frame_json(r: &FrameRecord, offset: i64) -> Value {
    let mut stages = Map::new();
    for s in Stage::ALL {
        let v = r.stages_us[s as usize];
        if v > 0 {
            stages.insert(s.name().into(), json!(round3(v as f32 / 1000.0)));
        }
    }
    let ev = &r.events;
    let cmds: Vec<&str> = ev.cmds.iter().take(ev.cmd_count as usize).copied().collect();
    json!({
        "offset": offset,
        "interval_ms": round3(r.interval_us as f32 / 1000.0),
        "work_ms": round3(r.work_us() as f32 / 1000.0),
        "presenting": r.presenting,
        "stages": stages,
        "ran": {
            "cmds": cmds,
            "cmd_count": ev.cmd_count,
            "reloads": ev.reloads,
            "probe_steps": ev.probe_steps,
            "clip_opens": ev.clip_opens,
            "rev": ev.rev,
        },
    })
}

/// The worst *work* stage of a frame (waits don't explain a spike unless
/// nothing else does — then `acquire`/`pace` is the answer).
fn worst_stage(r: &FrameRecord) -> (Stage, u32) {
    let pick = |work_only: bool| {
        Stage::ALL
            .iter()
            .filter(|s| !work_only || s.is_work())
            .map(|s| (*s, r.stages_us[*s as usize]))
            .max_by_key(|(_, v)| *v)
            .expect("non-empty")
    };
    let w = pick(true);
    let any = pick(false);
    if w.1 * 2 >= any.1 { w } else { any }
}

fn spike_payload(cap: &SpikeCapture, budget_us: u32, suppressed: u64) -> (Value, String) {
    let hit = &cap.frames[cap.at];
    let (stage, stage_us) = worst_stage(hit);
    let frames: Vec<Value> = cap.frames[..cap.len]
        .iter()
        .enumerate()
        .map(|(i, r)| frame_json(r, i as i64 - cap.at as i64))
        .collect();
    let ev = &hit.events;
    let cmds: Vec<&str> = ev.cmds.iter().take(ev.cmd_count as usize).copied().collect();
    let line = format!(
        "perf spike: {:.1} ms frame (trigger {:.1} ms) — worst stage {} {:.1} ms; ran: cmds [{}], reloads {}, probe steps {}, clip opens {}, rev {}",
        hit.interval_us as f32 / 1000.0,
        cap.trigger_ms,
        stage.name(),
        stage_us as f32 / 1000.0,
        cmds.join(", "),
        ev.reloads,
        ev.probe_steps,
        ev.clip_opens,
        ev.rev,
    );
    let payload = json!({
        "ts_ms": cap.ts_ms,
        "frame_ms": round3(hit.interval_us as f32 / 1000.0),
        "trigger_ms": round3(cap.trigger_ms),
        "budget_ms": round3(budget_us as f32 / 1000.0),
        "worst_stage": stage.name(),
        "worst_stage_ms": round3(stage_us as f32 / 1000.0),
        "frames": frames,
        "suppressed": suppressed,
    });
    (payload, line)
}

// ---------------------------------------------------------------------------
// Thread CPU (each owned thread samples itself)
// ---------------------------------------------------------------------------

struct ThreadSlot {
    name: String,
    /// CPU % of one core × 100; `u32::MAX` = unavailable on this platform.
    pct_x100: AtomicU32,
    sampled_ms: AtomicU64,
}

fn thread_registry() -> &'static Mutex<Vec<Weak<ThreadSlot>>> {
    static REG: OnceLock<Mutex<Vec<Weak<ThreadSlot>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Owned by a thread; call [`ThreadCpu::tick`] from its loop. Samples
/// `CLOCK_THREAD_CPUTIME_ID` at most once a second. Dropping it removes
/// the thread from the `perf` payload.
pub struct ThreadCpu {
    slot: Arc<ThreadSlot>,
    last: Option<(u64, Instant)>,
}

impl ThreadCpu {
    pub fn register(name: impl Into<String>) -> Self {
        let slot = Arc::new(ThreadSlot {
            name: name.into(),
            pct_x100: AtomicU32::new(if cfg!(unix) { 0 } else { u32::MAX }),
            sampled_ms: AtomicU64::new(0),
        });
        if let Ok(mut reg) = thread_registry().lock() {
            reg.retain(|w| w.strong_count() > 0);
            reg.push(Arc::downgrade(&slot));
        }
        Self { slot, last: None }
    }

    #[inline]
    pub fn tick(&mut self) {
        if enabled() {
            self.tick_at(Instant::now());
        }
    }

    fn tick_at(&mut self, now: Instant) {
        // The baseline is taken lazily on the owning thread — a `Core`
        // built on one thread and driven from another must not mix clocks.
        match self.last {
            None => {
                if let Some(cpu) = thread_cpu_ns() {
                    self.last = Some((cpu, now));
                }
            }
            Some((cpu0, t0)) if now.duration_since(t0) >= Duration::from_secs(1) => {
                if let Some(cpu) = thread_cpu_ns() {
                    let wall = now.duration_since(t0).as_nanos().max(1) as f64;
                    let pct = (cpu.saturating_sub(cpu0) as f64 / wall * 100.0).clamp(0.0, 400.0);
                    self.slot.pct_x100.store((pct * 100.0) as u32, Ordering::Relaxed);
                    self.slot.sampled_ms.store(wall_ms(), Ordering::Relaxed);
                    self.last = Some((cpu, now));
                }
            }
            _ => {}
        }
    }
}

fn thread_rows() -> Value {
    let now = wall_ms();
    let Ok(mut reg) = thread_registry().lock() else { return json!([]) };
    reg.retain(|w| w.strong_count() > 0);
    let rows: Vec<Value> = reg
        .iter()
        .filter_map(Weak::upgrade)
        .map(|s| {
            let raw = s.pct_x100.load(Ordering::Relaxed);
            let sampled = s.sampled_ms.load(Ordering::Relaxed);
            let pct = if raw == u32::MAX {
                Value::Null
            } else if sampled != 0 && now.saturating_sub(sampled) > THREAD_STALE_MS {
                json!(0.0)
            } else {
                json!((raw as f32 / 100.0 * 10.0).round() / 10.0)
            };
            json!({ "name": s.name, "cpu_pct": pct })
        })
        .collect();
    Value::Array(rows)
}

#[cfg(unix)]
fn clock_ns(clock: libc::clockid_t) -> Option<u64> {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: plain POSIX call writing into a stack timespec.
    let rc = unsafe { libc::clock_gettime(clock, &mut ts) };
    (rc == 0).then(|| ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
}

#[cfg(unix)]
fn thread_cpu_ns() -> Option<u64> {
    clock_ns(libc::CLOCK_THREAD_CPUTIME_ID)
}

#[cfg(not(unix))]
fn thread_cpu_ns() -> Option<u64> {
    None
}

#[cfg(unix)]
fn process_cpu_ns() -> Option<u64> {
    clock_ns(libc::CLOCK_PROCESS_CPUTIME_ID)
}

#[cfg(not(unix))]
fn process_cpu_ns() -> Option<u64> {
    None
}

/// Resident set size. macOS: `proc_pidinfo(PROC_PIDTASKINFO)`; Linux:
/// `/proc/self/statm`; elsewhere `None`.
#[cfg(target_os = "macos")]
fn rss_bytes() -> Option<u64> {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: the buffer is a correctly-sized, zeroed proc_taskinfo.
    let rc = unsafe {
        libc::proc_pidinfo(
            std::process::id() as libc::c_int,
            libc::PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    (rc == size).then_some(info.pti_resident_size)
}

#[cfg(target_os = "linux")]
fn rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (page > 0).then(|| pages * page as u64)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn rss_bytes() -> Option<u64> {
    None
}

#[derive(Default)]
struct ProcessSampler {
    last: Option<(u64, Instant)>,
    pct: Option<f32>,
}

impl ProcessSampler {
    /// CPU % of one core since the previous call (multi-core can exceed 100).
    fn sample(&mut self) -> Option<f32> {
        let now = Instant::now();
        let cpu = process_cpu_ns()?;
        if let Some((cpu0, t0)) = self.last {
            let wall = now.duration_since(t0).as_nanos().max(1) as f64;
            self.pct = Some(((cpu.saturating_sub(cpu0) as f64 / wall * 1000.0).round() / 10.0) as f32);
        }
        self.last = Some((cpu, now));
        self.pct
    }
}

// ---------------------------------------------------------------------------
// Clip decoder stats
// ---------------------------------------------------------------------------

const DECODE_SAMPLES: usize = 64;

/// Per clip stream. The decode thread writes `decode_us`; the render thread
/// writes `stale` + `queue`. Lock-free on both sides.
pub struct ClipStats {
    label: String,
    decode_us: [AtomicU32; DECODE_SAMPLES],
    decode_n: AtomicU64,
    stale: AtomicU64,
    stale_reported: AtomicU64,
    queue: AtomicU32,
}

static CLIP_OPENS: AtomicU64 = AtomicU64::new(0);
/// Stale frames across every stream ever opened — the PM3 bench diffs it
/// over its window (streams come and go with plan rebuilds).
static STALE_TOTAL: AtomicU64 = AtomicU64::new(0);

fn clip_opens() -> u64 {
    CLIP_OPENS.load(Ordering::Relaxed)
}

fn clip_registry() -> &'static Mutex<Vec<Weak<ClipStats>>> {
    static REG: OnceLock<Mutex<Vec<Weak<ClipStats>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

impl ClipStats {
    /// Called once per clip open (plan build, off the per-frame path).
    pub fn register(label: impl Into<String>) -> Arc<Self> {
        CLIP_OPENS.fetch_add(1, Ordering::Relaxed);
        let stats = Arc::new(Self {
            label: label.into(),
            decode_us: [const { AtomicU32::new(0) }; DECODE_SAMPLES],
            decode_n: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            stale_reported: AtomicU64::new(0),
            queue: AtomicU32::new(0),
        });
        if let Ok(mut reg) = clip_registry().lock() {
            reg.retain(|w| w.strong_count() > 0);
            reg.push(Arc::downgrade(&stats));
        }
        stats
    }

    /// Decode thread: one frame's read + decompress + slot write.
    pub fn record_decode(&self, d: Duration) {
        if enabled() {
            let i = self.decode_n.fetch_add(1, Ordering::Relaxed) as usize % DECODE_SAMPLES;
            self.decode_us[i].store(us(d), Ordering::Relaxed);
        }
    }

    /// Render thread: the playhead wanted a frame the ring didn't have, so
    /// an older frame stayed up. A native-rate repeat (30 fps clip on a
    /// 60 Hz engine wanting the same frame twice) is *not* stale.
    pub fn mark_stale(&self) {
        self.stale.fetch_add(1, Ordering::Relaxed);
        STALE_TOTAL.fetch_add(1, Ordering::Relaxed);
    }

    /// Render thread: decoded frames waiting to be uploaded.
    pub fn set_queue(&self, depth: usize) {
        self.queue.store(depth as u32, Ordering::Relaxed);
    }
}

/// PM3 bench: (active streams, worst per-stream decode p95 in ms) over each
/// stream's last [`DECODE_SAMPLES`] decodes. Read-only — unlike
/// [`clip_rows`] it doesn't advance the per-window stale counters.
fn clip_decode_summary() -> (usize, Option<f32>) {
    let Ok(reg) = clip_registry().lock() else { return (0, None) };
    let mut scratch = Vec::with_capacity(DECODE_SAMPLES);
    let mut active = 0;
    let mut worst: Option<f32> = None;
    for c in reg.iter().filter_map(Weak::upgrade) {
        let n = (c.decode_n.load(Ordering::Relaxed) as usize).min(DECODE_SAMPLES);
        if n == 0 {
            continue;
        }
        active += 1;
        scratch.clear();
        scratch.extend(c.decode_us[..n].iter().map(|a| a.load(Ordering::Relaxed)));
        let p95 = Pct::of(&mut scratch).p95;
        worst = Some(worst.map_or(p95, |w| w.max(p95)));
    }
    (active, worst)
}

/// One row per *active* clip stream. A stream that has never decoded and
/// never gone stale is idle (typically the design leg's copy while nothing
/// consumes the design composite) — counted, not listed, so the two legs'
/// same-named streams don't read as duplicates.
fn clip_rows(scratch: &mut Vec<u32>) -> (Value, usize) {
    let Ok(mut reg) = clip_registry().lock() else { return (json!([]), 0) };
    reg.retain(|w| w.strong_count() > 0);
    let mut idle = 0;
    let rows: Vec<Value> = reg
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|c| {
            let active = c.decode_n.load(Ordering::Relaxed) > 0 || c.stale.load(Ordering::Relaxed) > 0;
            idle += usize::from(!active);
            active
        })
        .map(|c| {
            let n = (c.decode_n.load(Ordering::Relaxed) as usize).min(DECODE_SAMPLES);
            scratch.clear();
            scratch.extend(c.decode_us[..n].iter().map(|a| a.load(Ordering::Relaxed)));
            let p = Pct::of(scratch);
            let stale = c.stale.load(Ordering::Relaxed);
            let prev = c.stale_reported.swap(stale, Ordering::Relaxed);
            json!({
                "label": c.label,
                "decode_p50": (n > 0).then(|| round3(p.p50)),
                "decode_p95": (n > 0).then(|| round3(p.p95)),
                "stale": stale.saturating_sub(prev),
                "stale_total": stale,
                "queue": c.queue.load(Ordering::Relaxed),
            })
        })
        .collect();
    (Value::Array(rows), idle)
}

// ---------------------------------------------------------------------------
// Telemetry bandwidth (counted where the bytes are already serialized)
// ---------------------------------------------------------------------------

/// Which hop a telemetry byte count was taken at.
#[derive(Debug, Clone, Copy)]
pub enum Hop {
    /// WS notifications to remote clients (`ws.rs`).
    Ws = 0,
    /// The Tauri `telemetry_loop` → webview event hop.
    Webview = 1,
}

const MAX_CHANNELS: usize = 32;
static TELEMETRY_BYTES: [[AtomicU64; MAX_CHANNELS]; 2] =
    [[const { AtomicU64::new(0) }; MAX_CHANNELS], [const { AtomicU64::new(0) }; MAX_CHANNELS]];

/// Count `bytes` delivered on `channel` at `hop`. Unknown channels are
/// ignored (they can't be subscribed to anyway).
pub fn count_telemetry(hop: Hop, channel: &str, bytes: usize) {
    if !enabled() {
        return;
    }
    if let Some(i) = ALL_CHANNELS.iter().position(|c| *c == channel) {
        if i < MAX_CHANNELS {
            TELEMETRY_BYTES[hop as usize][i].fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }
}

/// Serialized size of a JSON value without allocating the string — for
/// hops (the Tauri webview event) that serialize internally.
pub fn json_len(v: &Value) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut c = Count(0);
    let _ = serde_json::to_writer(&mut c, v);
    c.0
}

#[derive(Default)]
struct TelemetryRates {
    last: Option<([[u64; MAX_CHANNELS]; 2], Instant)>,
}

impl TelemetryRates {
    fn sample(&mut self) -> Value {
        let now = Instant::now();
        let mut cur = [[0u64; MAX_CHANNELS]; 2];
        for (h, row) in TELEMETRY_BYTES.iter().enumerate() {
            for (i, a) in row.iter().enumerate() {
                cur[h][i] = a.load(Ordering::Relaxed);
            }
        }
        let out = match self.last {
            Some((prev, t0)) => {
                let secs = now.duration_since(t0).as_secs_f64().max(1e-3);
                let hop = |h: usize| {
                    let mut m = Map::new();
                    for (i, name) in ALL_CHANNELS.iter().enumerate().take(MAX_CHANNELS) {
                        let d = cur[h][i].saturating_sub(prev[h][i]);
                        if d > 0 {
                            m.insert((*name).into(), json!((d as f64 / secs).round() as u64));
                        }
                    }
                    Value::Object(m)
                };
                json!({ "ws": hop(0), "webview": hop(1) })
            }
            None => json!({ "ws": {}, "webview": {} }),
        };
        self.last = Some((cur, now));
        out
    }
}

// ---------------------------------------------------------------------------
// GPU resources: wgpu live counts + own byte inventory
// ---------------------------------------------------------------------------

/// Live wgpu objects per type (`Instance::generate_report`) — the leak
/// signal: these must plateau under edit churn.
pub fn wgpu_resources(instance: &wgpu::Instance, backend: wgpu::Backend) -> Value {
    let Some(report) = instance.generate_report() else { return Value::Null };
    if !matches!(
        backend,
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 | wgpu::Backend::Gl
    ) {
        return Value::Null;
    }
    let hub = report.hub_report(backend);
    json!({
        "buffers": hub.buffers.num_kept_from_user,
        "textures": hub.textures.num_kept_from_user,
        "texture_views": hub.texture_views.num_kept_from_user,
        "samplers": hub.samplers.num_kept_from_user,
        "bind_groups": hub.bind_groups.num_kept_from_user,
        "render_pipelines": hub.render_pipelines.num_kept_from_user,
        "shader_modules": hub.shader_modules.num_kept_from_user,
        "command_buffers": hub.command_buffers.num_kept_from_user,
        "query_sets": hub.query_sets.num_kept_from_user,
    })
}

/// Bytes a texture occupies (all mips, all layers, all samples) from its
/// own descriptor — portable, unlike Metal's `currentAllocatedSize`.
pub fn texture_bytes(t: &wgpu::Texture) -> u64 {
    let format = t.format();
    let (bw, bh) = format.block_dimensions();
    let block = format.block_copy_size(None).unwrap_or(4) as u64;
    let layers = t.depth_or_array_layers().max(1) as u64;
    let mut total = 0u64;
    for mip in 0..t.mip_level_count() {
        let w = (t.width() >> mip).max(1);
        let h = (t.height() >> mip).max(1);
        total += w.div_ceil(bw) as u64 * h.div_ceil(bh) as u64 * block;
    }
    total * layers * t.sample_count().max(1) as u64
}

// ---------------------------------------------------------------------------
// Tests: the aggregator logic only — never real timing numbers (flaky).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(stages: &[(Stage, u32)], interval_us: u32) -> FrameRecord {
        let mut r = FrameRecord {
            interval_us,
            presenting: true,
            ..Default::default()
        };
        for (s, v) in stages {
            r.stages_us[*s as usize] = *v;
        }
        r
    }

    /// Feed a synthetic record through the detectors (no clocks involved).
    fn push(p: &mut Perf, r: FrameRecord) {
        p.cur = r;
        p.close_record(false, Instant::now());
        p.cur = FrameRecord::default();
    }

    #[test]
    fn stage_names_are_unique_and_complete() {
        let names: std::collections::HashSet<_> = Stage::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(names.len(), Stage::COUNT);
        for (i, s) in Stage::ALL.iter().enumerate() {
            assert_eq!(*s as usize, i, "Stage::ALL must be in discriminant order");
        }
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut v: Vec<u32> = (1..=100).map(|i| i * 1000).collect();
        let p = Pct::of(&mut v);
        assert_eq!((p.p50, p.p95, p.max), (50.0, 95.0, 100.0));
        let mut one = vec![7000];
        let p = Pct::of(&mut one);
        assert_eq!((p.p50, p.p95, p.max), (7.0, 7.0, 7.0));
        assert_eq!(Pct::of(&mut []), Pct::default());
    }

    #[test]
    fn work_excludes_acquire_and_pace_and_unaccounted_is_the_gap() {
        let r = frame(
            &[
                (Stage::InboundCmds, 100),
                (Stage::TickLive, 200),
                (Stage::Encode, 300),
                (Stage::Acquire, 5_000),
                (Stage::Pace, 10_000),
                (Stage::Present, 50),
            ],
            16_000,
        );
        assert_eq!(r.work_us(), 650);
        assert_eq!(r.total_us(), 15_650);
        assert_eq!(r.unaccounted_us(), 350);
        // Stages can't be negative time; a total over the interval clamps.
        assert_eq!(frame(&[(Stage::Encode, 20_000)], 16_000).unaccounted_us(), 0);
    }

    #[test]
    fn ring_keeps_the_newest_and_iterates_oldest_first() {
        let mut r: Ring<u32, 4> = Ring::new();
        for i in 1..=6 {
            r.push(i);
        }
        assert_eq!(r.len(), 4);
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5, 6]);
        assert_eq!(r.back(0), Some(&6));
        assert_eq!(r.back(4), None);
    }

    #[test]
    fn dropped_frames_count_presented_intervals_over_one_and_a_half_budgets() {
        let mut p = Perf::new(true);
        let budget = p.budget_us;
        push(&mut p, frame(&[], budget)); // on time
        push(&mut p, frame(&[], budget * 3 / 2)); // exactly 1.5× — not dropped
        push(&mut p, frame(&[], budget * 2)); // missed vsync
        let mut hidden = frame(&[], budget * 3);
        hidden.presenting = false; // offscreen self-pacing is never "dropped"
        push(&mut p, hidden);
        assert_eq!(p.dropped_total, 1);
        let agg = p.aggregate(None);
        assert_eq!(agg["dropped"]["window"], 1);
        assert_eq!(agg["frames"], 4);
    }

    #[test]
    fn aggregate_reports_headroom_from_work_p95() {
        let mut p = Perf::new(true);
        let budget_ms = p.budget_us as f32 / 1000.0;
        for _ in 0..100 {
            // 4 ms of work, the rest waiting on vsync.
            push(&mut p, frame(&[(Stage::Encode, 4_000), (Stage::Acquire, 12_000)], 16_667));
        }
        let agg = p.aggregate(None);
        assert_eq!(agg["frame"]["work"]["p95"], 4.0);
        assert_eq!(agg["cpu"]["acquire"]["p50"], 12.0);
        let headroom = agg["headroom"].as_f64().unwrap() as f32;
        assert!((headroom - (1.0 - 4.0 / budget_ms)).abs() < 1e-3);
        // No `GpuTimer` (`--gpu-timing 0`, or the adapter lacks
        // TIMESTAMP_QUERY) → the GPU columns are null, headroom is CPU-only.
        assert!(agg["gpu"].is_null());
    }

    fn gpu_frame(passes: &[(GpuPass, u32)], total: u32) -> GpuFrame {
        let mut f = GpuFrame {
            total,
            ..Default::default()
        };
        for (pass, t) in passes {
            f.ticks[*pass as usize] = *t;
            f.ran |= 1 << *pass as u8;
        }
        f
    }

    #[test]
    fn gpu_pass_names_are_unique_and_complete() {
        let names: std::collections::HashSet<_> = GpuPass::ALL.iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), GpuPass::COUNT);
        for (i, p) in GpuPass::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i, "GpuPass::ALL must be in discriminant order");
        }
        assert!(GpuPass::COUNT <= 8, "GpuFrame::ran is a u8 bitmask");
    }

    #[test]
    fn gpu_columns_fill_and_headroom_takes_the_busier_side() {
        let mut p = Perf::new(true);
        let budget_ms = p.budget_us as f32 / 1000.0;
        p.set_gpu_timing(1.0, 1); // 1 ns per tick (Apple Silicon)
        for _ in 0..100 {
            push(&mut p, frame(&[(Stage::Encode, 4_000), (Stage::Acquire, 12_000)], 16_667));
            p.push_gpu(gpu_frame(
                &[(GpuPass::CompositeLive, 6_000_000), (GpuPass::Final, 1_000_000)],
                10_000_000,
            ));
        }
        p.push_gpu(gpu_frame(&[(GpuPass::WarpBake, 500_000)], 500_000));
        p.note_gpu_skipped();
        let agg = p.aggregate(None);
        let gpu = &agg["gpu"];
        assert_eq!(gpu["every"], 1);
        assert_eq!(gpu["frames"], 101);
        assert_eq!(gpu["skipped"], 1);
        assert_eq!(gpu["total"]["p95"], 10.0);
        assert_eq!(gpu["passes"]["composite.live"]["p95"], 6.0);
        assert_eq!(gpu["passes"]["composite.live"]["n"], 100);
        assert_eq!(gpu["passes"]["warp_bake"]["n"], 1);
        assert!(gpu["passes"].get("composite.design").is_none(), "passes that never ran are omitted");
        // GPU 10 ms > CPU work 4 ms → the GPU sets the headroom.
        let headroom = agg["headroom"].as_f64().unwrap() as f32;
        assert!((headroom - (1.0 - 10.0 / budget_ms)).abs() < 1e-3);
        // E2's per-leg accessor reads the same samples.
        let live = p.gpu_pass_recent(GpuPass::CompositeLive, 10).unwrap();
        assert_eq!(live.p95, 6.0);
        assert!(p.gpu_pass_recent(GpuPass::CompositeDesign, 10).is_none());
    }

    #[test]
    fn gpu_timing_stays_null_with_the_monitor_off() {
        let mut p = Perf::new(false);
        p.set_gpu_timing(1.0, 1);
        p.push_gpu(gpu_frame(&[(GpuPass::Final, 1_000)], 1_000));
        assert!(p.aggregate(None)["gpu"].is_null());
        assert!(p.gpu_pass_recent(GpuPass::Final, 10).is_none());
        set_enabled(true); // process-global; leave it on for other tests
    }

    #[test]
    fn spike_freezes_trigger_frame_with_neighbours_once() {
        let mut p = Perf::new(true);
        let trigger = p.shared.trigger_us.load(Ordering::Relaxed);
        for _ in 0..SPIKE_ARM_AFTER_FRAMES + 10 {
            push(&mut p, frame(&[(Stage::Encode, 1_000)], 16_667));
        }
        let mut hit = frame(&[(Stage::InboundCmds, 40_000)], trigger + 1);
        hit.events.cmds[0] = "scene.load";
        hit.events.cmd_count = 1;
        push(&mut p, hit);
        assert!(p.ready_spike.is_none(), "waits for the trailing neighbours");
        for _ in 0..SPIKE_NEIGHBOURS {
            push(&mut p, frame(&[(Stage::Encode, 1_000)], 16_667));
        }
        let cap = p.ready_spike.take().expect("spike frozen after 5 trailing frames");
        assert_eq!(cap.len, 2 * SPIKE_NEIGHBOURS + 1);
        assert_eq!(cap.at, SPIKE_NEIGHBOURS);
        let (payload, line) = spike_payload(&cap, p.budget_us, 0);
        assert_eq!(payload["worst_stage"], "inbound.cmds");
        assert_eq!(payload["frames"][SPIKE_NEIGHBOURS]["offset"], 0);
        assert_eq!(payload["frames"][0]["offset"], -(SPIKE_NEIGHBOURS as i64));
        assert_eq!(payload["frames"][SPIKE_NEIGHBOURS]["ran"]["cmds"][0], "scene.load");
        assert!(line.contains("inbound.cmds"));
        assert_eq!(p.spikes_total, 1);

        // A second hitch inside the min gap is counted, not captured.
        push(&mut p, frame(&[], trigger + 1));
        assert_eq!((p.spikes_total, p.spikes_suppressed), (1, 1));
        assert!(p.pending_spike.is_none());
    }

    #[test]
    fn spikes_stay_disarmed_during_boot_and_obey_set_trigger() {
        let mut p = Perf::new(true);
        push(&mut p, frame(&[], 500_000)); // first-frame shader compile
        assert_eq!(p.spikes_total, 0);
        for _ in 0..SPIKE_ARM_AFTER_FRAMES {
            push(&mut p, frame(&[], 16_667));
        }
        p.shared.set_trigger_ms(100.0).unwrap();
        push(&mut p, frame(&[], 50_000));
        assert_eq!(p.spikes_total, 0, "50 ms is under a 100 ms trigger");
        push(&mut p, frame(&[], 150_000));
        assert_eq!(p.spikes_total, 1);
        assert!(p.shared.set_trigger_ms(0.0).is_err());
        assert!(p.shared.set_trigger_ms(f32::NAN).is_err());
    }

    #[test]
    fn worst_stage_prefers_work_unless_a_wait_dominates() {
        let r = frame(&[(Stage::Encode, 3_000), (Stage::Acquire, 5_000)], 40_000);
        assert_eq!(worst_stage(&r).0, Stage::Encode);
        let r = frame(&[(Stage::Encode, 1_000), (Stage::Acquire, 30_000)], 40_000);
        assert_eq!(worst_stage(&r).0, Stage::Acquire);
    }

    #[test]
    fn knob_latency_resolves_the_oldest_pending_write_at_submit() {
        let mut p = Perf::new(true);
        p.shared.knob_written();
        let first = p.shared.knob_pending_us.load(Ordering::Relaxed);
        p.shared.knob_written(); // a later write keeps the older stamp
        assert_eq!(p.shared.knob_pending_us.load(Ordering::Relaxed), first);
        p.before_tick(None);
        assert_eq!(p.shared.knob_pending_us.load(Ordering::Relaxed), 0);
        assert_eq!(p.osc_age.len(), 0, "no OSC packet yet → no age sample");
        p.after_submit();
        assert_eq!(p.knob_submit.len(), 1);
        p.after_submit(); // nothing pending → no sample
        assert_eq!(p.knob_submit.len(), 1);
    }

    #[test]
    fn disabled_monitor_records_nothing() {
        let mut p = Perf::new(false);
        let s = p.begin();
        p.end(Stage::Encode, s);
        p.end_frame(true, 0);
        p.end_frame(true, 0);
        assert_eq!(p.frames.len(), 0);
        assert!(p.shared.snapshot().is_err());
        set_enabled(true); // process-global; leave it on for other tests
    }

    /// Not a test — a measurement. Replays one frame's worth of monitor
    /// calls (the same span/lap/end sequence `poll_inbound` + `pace_frame` +
    /// `redraw` make) and the 2 Hz aggregation, and prints the cost. Run:
    /// `cargo test --release perf_overhead -- --ignored --nocapture`.
    /// Never asserts a number (shared machine, flaky by nature).
    #[test]
    #[ignore]
    fn perf_overhead() {
        let bus = Bus::new();
        let mut p = Perf::new(true);
        let frames = 200_000u32;
        let t = Instant::now();
        for _ in 0..frames {
            // poll_inbound
            let s = p.enter();
            let s = p.lap(Stage::InboundCmds, s);
            let s = p.lap(Stage::InboundWatch, s);
            let s = p.lap(Stage::InboundProbe, s);
            let s = p.lap(Stage::InboundMisc, s);
            let s = p.lap(Stage::Telemetry, s);
            p.end(Stage::InboundMisc, s);
            // pace_frame
            let s = p.enter();
            p.end(Stage::Pace, s);
            // redraw
            let s = p.enter();
            p.end(Stage::TickLive, s);
            p.before_tick(None);
            let s = p.begin();
            let s = p.lap(Stage::TickLive, s);
            p.end(Stage::TickDesign, s);
            let s = p.begin();
            let s = p.lap(Stage::Acquire, s);
            let s = p.lap(Stage::Encode, s);
            let s = p.lap(Stage::Submit, s);
            let s = p.lap(Stage::Present, s);
            p.after_submit();
            let s = p.lap(Stage::Submit, s);
            let s = p.lap(Stage::PreviewPresent, s);
            let s = p.lap(Stage::Telemetry, s);
            p.end(Stage::PreviewJpeg, s);
            p.end_frame(true, 0);
        }
        let per_frame_ns = t.elapsed().as_nanos() as f64 / frames as f64;
        // A full GPU ring, so the aggregate below pays for the `gpu` columns.
        p.set_gpu_timing(1.0, 1);
        for i in 0..RING_FRAMES as u32 {
            let mut f = GpuFrame { total: 900_000 + i, ..Default::default() };
            for pass in [GpuPass::CompositeLive, GpuPass::Final, GpuPass::Preview] {
                f.ticks[pass as usize] = 300_000 + i;
                f.ran |= 1 << pass as u8;
            }
            p.push_gpu(f);
        }
        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            p.last_emit = Instant::now() - EMIT_EVERY;
            p.emit(&bus, None);
        }
        let emit_us = t.elapsed().as_micros() as f64 / n as f64;
        println!(
            "perf monitor: {per_frame_ns:.0} ns/frame hot path; {emit_us:.0} µs per 2 Hz aggregate \
             (= {:.1} µs/frame amortised at 60 Hz)",
            emit_us * 2.0 / 60.0
        );
        gpu_timing_overhead();
    }

    /// PM2's share of [`perf_overhead`]: the engine-shaped frame (3 main
    /// passes + a preview submit, each drawing at 1080p) on a real device,
    /// GPU timing on vs off, interleaved in rounds so machine drift hits
    /// both. Reports render-thread CPU per frame (encode + submits + timer
    /// calls) and CPU + GPU drain (the extra GPU work: per-pass query
    /// resets, the resolve, the readback copy).
    fn gpu_timing_overhead() {
        use crate::gpu_timing::{GpuTimer, Rig};
        let Some(rig) = Rig::new() else {
            println!("gpu timing: no adapter with TIMESTAMP_QUERY — skipped");
            return;
        };
        let mut perf = Perf::new(true);
        let mut timer = GpuTimer::new(&rig.device, &rig.queue, rig.backend, 1);
        perf.set_gpu_timing(timer.period_ns(), timer.every());
        let (rounds, per_round) = (20, 200);
        let (mut cpu, mut all) = ([Duration::ZERO; 2], [Duration::ZERO; 2]);
        for _ in 0..20 {
            rig.frame(None, &mut perf); // warm-up
            rig.device.poll(wgpu::Maintain::Wait);
        }
        for _ in 0..rounds {
            for (k, timed) in [false, true].into_iter().enumerate() {
                for _ in 0..per_round {
                    let t = Instant::now();
                    cpu[k] += rig.frame(timed.then_some(&mut timer), &mut perf);
                    rig.device.poll(wgpu::Maintain::Wait);
                    all[k] += t.elapsed();
                }
            }
        }
        let per = |d: Duration| d.as_nanos() as f64 / 1000.0 / (rounds * per_round) as f64;
        println!(
            "gpu timing, per timed frame: render-thread cpu {:.1} → {:.1} µs (+{:.1}); \
             cpu + gpu drain {:.1} → {:.1} µs (+{:.1}); {} gpu frames read back. \
             At the default 1-in-{}: +{:.1} µs/frame cpu amortised",
            per(cpu[0]),
            per(cpu[1]),
            per(cpu[1]) - per(cpu[0]),
            per(all[0]),
            per(all[1]),
            per(all[1]) - per(all[0]),
            perf.gpu.as_ref().map_or(0, |g| g.frames.len()),
            crate::DEFAULT_GPU_TIMING,
            (per(cpu[1]) - per(cpu[0])) / crate::DEFAULT_GPU_TIMING as f64,
        );
        set_enabled(true);
    }

    #[test]
    fn telemetry_channel_table_fits() {
        assert!(ALL_CHANNELS.len() <= MAX_CHANNELS);
        assert!(ALL_CHANNELS.contains(&"perf") && ALL_CHANNELS.contains(&"perf_spike"));
    }
}
