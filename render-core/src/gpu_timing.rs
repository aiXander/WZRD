//! Per-pass GPU timestamps (perf-monitor PM2) — the `gpu` half of the
//! `perf` channel.
//!
//! **Zero-impact rules** (the profiler must never cost the show a frame):
//!
//! - Timestamps ride on render passes that already exist
//!   (`timestamp_writes`); nothing splits, adds a pass or opens an encoder to
//!   be timed. The clip **upload copies are not timed**: on Apple Silicon a
//!   timestamp outside a pass (`write_timestamp`) makes wgpu-hal's Metal
//!   backend open a dummy blit encoder + fill per sample (no
//!   blit-boundary counter sampling), which is exactly the extra GPU
//!   structure this module refuses.
//! - Never blocks the render thread. A frame's queries are resolved by the
//!   **next** frame's main encoder (every pass of frame N — probe, main,
//!   preview — has been submitted by then), mapped with `map_async` after
//!   that submit, and read once the callback fired. `queue.submit` already
//!   runs a non-blocking maintain, so no extra `poll` is needed. When every
//!   slot is in flight the frame simply isn't timed.
//!   The one deliberate wait: a *sampled* frame (1 in `every`) runs alone
//!   on the GPU — offscreen `wait_turn` drains before acquire, a window
//!   drains right before the main submit
//!   (`GpuContext::drain_before_timed_submit`) — or its first pass's span
//!   counts the previous frame's tail. Untimed frames never wait.
//! - Allocation-free per frame: the query set and buffers are created once
//!   (`GpuTimer::new`); raw tick deltas go into a fixed [`GpuFrame`].
//!   Conversion to ms and percentiles happen in the 2 Hz `perf` aggregate.
//! - Off means off: with `--perf off`, `--gpu-timing 0` or an adapter
//!   without `TIMESTAMP_QUERY`, no `GpuTimer` exists, no feature is
//!   requested and `perf.gpu` is `null`.
//!
//! **What the numbers mean.** `total` = first-begin → last-end *within each
//! submit*, summed over the frame's submits (main encoder, preview blit,
//! probe frames) — so CPU-side gaps between submits (the vsync wait in
//! `acquire`, probe pacing) never read as GPU time. It is the budget truth.
//! Per pass, passes in the **main** submit are charged
//! `end − max(begin, previous pass's end)`: on a tile-based GPU (Apple) the
//! next pass's start-of-vertex stamp fires while the previous pass's
//! fragments still run — measured on the M2, raw `final` read 7.5 ms on a
//! scene whose final pass costs 0.2 ms, because it absorbed the composite.
//! Charged this way the passes tile the span (Σ ≈ total). Preview and probe
//! passes are their own submits: probe reports raw begin → end, the preview
//! blit is charged from the main span's end (it queues right behind it and
//! samples its composite; raw, its early begin stamp read ~8 ms for a
//! ~0.3 ms blit in the app). A pass with no
//! draws (an empty composite) gets no end-of-fragment sample on Metal and
//! is dropped from that frame. Live clip uploads precede the first timed
//! pass, so they sit outside `total`; design uploads fall between two timed
//! passes and land in `composite.design`'s charge.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::perf::{GpuFrame, GpuPass, Perf};
#[cfg(test)]
use crate::perf;

/// Readback slots: one recording + up to three resolving/mapping. Results
/// land 2–3 frames late; a fourth slot keeps a slow map from skipping
/// frames in steady state.
const SLOTS: usize = 4;
/// Probe frames timed per engine frame. A probe step runs as many probe
/// frames as fit in its ~3.5 ms budget; the rest are counted and the timed
/// mean is extrapolated over them.
const PROBE_PAIRS: u32 = 8;
/// One begin/end pair per non-probe pass, then the probe pairs.
const FIXED_PAIRS: u32 = GpuPass::COUNT as u32 - 1;
const PAIRS: u32 = FIXED_PAIRS + PROBE_PAIRS;
const QUERIES_PER_SLOT: u32 = PAIRS * 2;
/// Per-pair resolves (non-Metal) give each pair its own block: resolve
/// destinations must be 256-aligned, and the pairs written in a frame are
/// rarely contiguous. Metal resolves the slot contiguously (see
/// [`GpuTimer::encode_resolve`]); this size fits both layouts.
const PAIR_STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
const SLOT_BYTES: u64 = PAIRS as u64 * PAIR_STRIDE;
/// A pass longer than this is a garbage sample (unwritten query), not a
/// measurement.
const MAX_SANE_NS: f64 = 1.0e9;

const MAP_IDLE: u8 = 0;
const MAP_PENDING: u8 = 1;
const MAP_OK: u8 = 2;
const MAP_ERR: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Passes of the current frame write here.
    Recording,
    /// Frame closed; the next main encoder resolves it.
    AwaitResolve,
    /// Resolve + copy encoded; `map_async` goes out after that submit.
    Resolved,
    /// `map_async` in flight (`map` flips when the callback fires).
    Mapping,
}

struct Slot {
    readback: wgpu::Buffer,
    state: SlotState,
    map: Arc<AtomicU8>,
    /// Bit per pair written this frame (fixed passes, then probe pairs).
    written: u16,
    probe_timed: u8,
    /// Probe frames run this frame, timed or not.
    probe_total: u16,
}

pub struct GpuTimer {
    /// One query set per slot, so every slot's queries start at index 0
    /// (the Metal resolve workaround needs that).
    sets: [wgpu::QuerySet; SLOTS],
    resolve: wgpu::Buffer,
    slots: [Slot; SLOTS],
    /// Metal: resolve each slot as one `0..QUERIES_PER_SLOT` range. Others:
    /// one resolve per written pair.
    resolve_whole: bool,
    /// Time 1 frame in `every` (1 = every frame).
    every: u32,
    frame_no: u64,
    /// Slot recording the current frame; `None` = this frame isn't timed.
    cur: Option<usize>,
    /// Ticks → ns (`Queue::get_timestamp_period`).
    period_ns: f32,
}

impl GpuTimer {
    /// Features to request for GPU timing, if the adapter offers them.
    pub const FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

    /// Allocate the query sets + buffers (once). The device must have been
    /// created with [`Self::FEATURES`]. `every` ≥ 1.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, backend: wgpu::Backend, every: u32) -> Self {
        let sets = std::array::from_fn(|i| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some(&format!("perf gpu timestamps {i}")),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES_PER_SLOT,
            })
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf gpu timestamp resolve"),
            size: SLOT_BYTES * SLOTS as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|i| Slot {
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("perf gpu timestamp readback {i}")),
                size: SLOT_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: SlotState::Free,
            map: Arc::new(AtomicU8::new(MAP_IDLE)),
            written: 0,
            probe_timed: 0,
            probe_total: 0,
        });
        let mut timer = Self {
            sets,
            resolve,
            slots,
            resolve_whole: backend == wgpu::Backend::Metal,
            every: every.max(1),
            frame_no: 0,
            cur: None,
            period_ns: queue.get_timestamp_period(),
        };
        timer.open_slot();
        timer
    }

    pub fn period_ns(&self) -> f32 {
        self.period_ns
    }

    pub fn every(&self) -> u32 {
        self.every
    }

    /// True when the current frame is sampled (a slot is recording). Read
    /// by the PM3 bench's offscreen output, which runs sampled frames alone
    /// on the GPU (`gpu.rs` `OffscreenOutput::wait_turn`).
    pub fn recording(&self) -> bool {
        self.cur.is_some()
    }

    /// Bytes this module holds on the GPU (perf `gpu_mb` inventory).
    pub fn gpu_bytes(&self) -> u64 {
        self.resolve.size()
            + self.slots.iter().map(|s| s.readback.size()).sum::<u64>()
            + QUERIES_PER_SLOT as u64 * SLOTS as u64 * wgpu::QUERY_SIZE as u64
    }

    /// `timestamp_writes` for `pass` in the current frame, or `None` (frame
    /// not sampled, no free slot, pass already timed this frame, probe pairs
    /// used up).
    pub fn writes(&mut self, pass: GpuPass) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let i = self.cur?;
        let slot = &mut self.slots[i];
        let pair = if pass == GpuPass::Probe {
            slot.probe_total = slot.probe_total.saturating_add(1);
            if slot.probe_timed as u32 >= PROBE_PAIRS {
                return None;
            }
            slot.probe_timed += 1;
            FIXED_PAIRS + slot.probe_timed as u32 - 1
        } else {
            pass as u32
        };
        if slot.written & (1 << pair) != 0 {
            return None;
        }
        slot.written |= 1 << pair;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.sets[i],
            beginning_of_pass_write_index: Some(pair * 2),
            end_of_pass_write_index: Some(pair * 2 + 1),
        })
    }

    /// Start of the frame's **main** encoder: resolve the previous frame's
    /// slot into it, then one copy into that slot's readback buffer.
    ///
    /// **wgpu-hal 22 Metal bug:** `copy_query_results` builds
    /// `NSRange::new(range.start, range.end)` — the end index passed as the
    /// *length* — so any resolve not starting at query 0 resolves the wrong
    /// range and overruns its destination. On Metal the slot is therefore
    /// resolved whole, `0..QUERIES_PER_SLOT` (length = end, correct), and
    /// unwritten pairs are ignored via `written`. Elsewhere each written pair
    /// resolves on its own, so Vulkan never reads a query nobody wrote.
    pub fn encode_resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(i) = self.slot_in(SlotState::AwaitResolve) else {
            return;
        };
        let slot = &mut self.slots[i];
        let base_b = i as u64 * SLOT_BYTES;
        if self.resolve_whole {
            encoder.resolve_query_set(&self.sets[i], 0..QUERIES_PER_SLOT, &self.resolve, base_b);
        } else {
            for pair in 0..PAIRS {
                if slot.written & (1 << pair) != 0 {
                    encoder.resolve_query_set(
                        &self.sets[i],
                        pair * 2..pair * 2 + 2,
                        &self.resolve,
                        base_b + pair as u64 * PAIR_STRIDE,
                    );
                }
            }
        }
        encoder.copy_buffer_to_buffer(&self.resolve, base_b, &slot.readback, 0, SLOT_BYTES);
        slot.state = SlotState::Resolved;
    }

    /// After the main encoder's submit: hand finished readbacks to `perf`,
    /// then start the map for the slot resolved this frame.
    pub fn after_submit(&mut self, perf: &mut Perf) {
        for i in 0..SLOTS {
            if self.slots[i].state != SlotState::Mapping {
                continue;
            }
            match self.slots[i].map.load(Ordering::Acquire) {
                MAP_OK => {
                    let frame = self.read_slot(i);
                    self.slots[i].readback.unmap();
                    self.free(i);
                    // Nothing sane in it (only empty passes) — not a sample.
                    if frame.ran != 0 {
                        perf.push_gpu(frame);
                    }
                }
                MAP_ERR => self.free(i),
                _ => {}
            }
        }
        if let Some(i) = self.slot_in(SlotState::Resolved) {
            let slot = &mut self.slots[i];
            slot.state = SlotState::Mapping;
            slot.map.store(MAP_PENDING, Ordering::Relaxed);
            let cb = Arc::clone(&slot.map);
            slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                cb.store(if res.is_ok() { MAP_OK } else { MAP_ERR }, Ordering::Release);
            });
        }
    }

    /// Close the frame: its slot waits for the next main encoder's resolve,
    /// and the next frame gets a free slot if it's sampled and one exists.
    pub fn end_frame(&mut self, perf: &mut Perf) {
        // A resolve that never happened (no main encoder ran) is dropped.
        if let Some(i) = self.slot_in(SlotState::AwaitResolve) {
            self.free(i);
        }
        if let Some(i) = self.cur.take() {
            if self.slots[i].written != 0 {
                self.slots[i].state = SlotState::AwaitResolve;
            } else {
                self.free(i);
            }
        }
        self.frame_no += 1;
        if self.frame_no % self.every as u64 == 0 && !self.open_slot() {
            perf.note_gpu_skipped();
        }
    }

    fn open_slot(&mut self) -> bool {
        let Some(i) = self.slot_in(SlotState::Free) else {
            return false;
        };
        self.slots[i].state = SlotState::Recording;
        self.cur = Some(i);
        true
    }

    fn slot_in(&self, state: SlotState) -> Option<usize> {
        self.slots.iter().position(|s| s.state == state)
    }

    fn free(&mut self, i: usize) {
        let s = &mut self.slots[i];
        s.state = SlotState::Free;
        s.map.store(MAP_IDLE, Ordering::Relaxed);
        s.written = 0;
        s.probe_timed = 0;
        s.probe_total = 0;
    }

    /// Mapped readback → one [`GpuFrame`] of tick deltas.
    fn read_slot(&self, i: usize) -> GpuFrame {
        let slot = &self.slots[i];
        let data = slot.readback.slice(..).get_mapped_range();
        let stride = if self.resolve_whole { 2 * wgpu::QUERY_SIZE as u64 } else { PAIR_STRIDE };
        let pair_ticks = |pair: u32| -> Option<(u64, u64)> {
            let at = (pair as u64 * stride) as usize;
            let begin = u64::from_le_bytes(data[at..at + 8].try_into().ok()?);
            let end = u64::from_le_bytes(data[at + 8..at + 16].try_into().ok()?);
            sane(begin, end, self.period_ns).then_some((begin, end))
        };
        collect_frame(slot.written, slot.probe_timed, slot.probe_total, pair_ticks)
    }
}

fn sane(begin: u64, end: u64, period_ns: f32) -> bool {
    // 0 = never written; u64::MAX = Metal's counter error value.
    begin != 0 && end != u64::MAX && end >= begin && ((end - begin) as f64 * period_ns as f64) < MAX_SANE_NS
}

/// Fold one slot's pair timestamps into a frame record: main-submit passes
/// charged from the previous pass's end (see the module doc), probe
/// extrapolated from its timed frames, and the per-submit spans summed into
/// `total`. Clock-free (the tests drive it directly).
fn collect_frame(
    written: u16,
    probe_timed: u8,
    probe_total: u16,
    pair_ticks: impl Fn(u32) -> Option<(u64, u64)>,
) -> GpuFrame {
    let mut f = GpuFrame::default();
    let clamp = |t: u64| t.min(u32::MAX as u64) as u32;
    // Main-encoder span: first begin → last end over its passes, which run
    // in `GpuPass` order (warp bake, live, design, final).
    let mut span: Option<(u64, u64)> = None;
    for pass in GpuPass::ALL {
        if pass == GpuPass::Probe {
            continue;
        }
        let pair = pass as u32;
        if written & (1 << pair) == 0 {
            continue;
        }
        let Some((b, e)) = pair_ticks(pair) else { continue };
        f.ran |= 1 << pass as u8;
        if pass == GpuPass::Preview {
            // Its own submit, queued right behind the main one (every main
            // pass precedes it in `ALL`, so `span` is complete here). The
            // blit samples a composite the main submit draws, so its real
            // work can't start before the main span ends — but on a TBDR GPU
            // its begin stamp fires while the main fragments still run
            // (measured in the app: a ~0.3 ms blit read 8 ms raw). Charged
            // from the main span's end, like the main passes tile; `total`
            // gets only the part outside the main span.
            let from = span.map_or(b, |(_, main_end)| b.max(main_end));
            let own = clamp(e.saturating_sub(from));
            f.ticks[pass as usize] = own;
            f.total = f.total.saturating_add(own);
        } else {
            let from = span.map_or(b, |(_, prev_end)| b.max(prev_end));
            f.ticks[pass as usize] = clamp(e.saturating_sub(from));
            span = Some(span.map_or((b, e), |(sb, se)| (sb.min(b), se.max(e))));
        }
    }
    if let Some((b, e)) = span {
        f.total = f.total.saturating_add(clamp(e - b));
    }
    // Probe frames: each is its own submit; time the first PROBE_PAIRS and
    // extrapolate the mean over every probe frame that ran.
    let (mut sum, mut n) = (0u64, 0u64);
    for k in 0..probe_timed as u32 {
        let pair = FIXED_PAIRS + k;
        if written & (1 << pair) == 0 {
            continue;
        }
        if let Some((b, e)) = pair_ticks(pair) {
            sum += e - b;
            n += 1;
        }
    }
    if n > 0 {
        let probe = clamp(sum * probe_total.max(n as u16) as u64 / n);
        f.ticks[GpuPass::Probe as usize] = probe;
        f.ran |= 1 << GpuPass::Probe as u8;
        f.total = f.total.saturating_add(probe);
    }
    f
}

/// Test/bench rig: a headless device with `TIMESTAMP_QUERY` (or `None`
/// where the adapter doesn't offer it) and a 1080p target, driving the
/// engine's per-frame call order: resolve → composite + final passes →
/// submit → `after_submit` → preview submit → `end_frame`. Every pass draws
/// one fullscreen triangle — Metal samples end-of-fragment only for a pass
/// with fragment work (an empty pass resolves its end stamp as 0), and every
/// engine pass draws.
#[cfg(test)]
pub(crate) struct Rig {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub backend: wgpu::Backend,
    view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
}

#[cfg(test)]
impl Rig {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        if !adapter.features().contains(GpuTimer::FEATURES) {
            return None;
        }
        let backend = adapter.get_info().backend;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu_timing test device"),
                required_features: GpuTimer::FEATURES,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gpu_timing test target"),
            size: wgpu::Extent3d { width: 1920, height: 1080, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                     let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                     return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
                 }
                 @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.1, 0.2, 0.3, 1.0); }"
                    .into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Some(Self { device, queue, backend, view, pipeline })
    }

    fn pass(&self, encoder: &mut wgpu::CommandEncoder, ts: Option<wgpu::RenderPassTimestampWrites<'_>>) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: ts,
        });
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }

    /// One engine-shaped frame. Returns the CPU time of the render-thread
    /// part (encode + submits + timer calls); the GPU drain after it is
    /// the caller's.
    pub fn frame(&self, mut timer: Option<&mut GpuTimer>, perf: &mut Perf) -> std::time::Duration {
        let t0 = std::time::Instant::now();
        let mut enc = self.device.create_command_encoder(&Default::default());
        if let Some(t) = timer.as_deref_mut() {
            t.encode_resolve(&mut enc);
        }
        for pass in [GpuPass::CompositeLive, GpuPass::CompositeDesign, GpuPass::Final] {
            self.pass(&mut enc, timer.as_deref_mut().and_then(|t| t.writes(pass)));
        }
        self.queue.submit(std::iter::once(enc.finish()));
        if let Some(t) = timer.as_deref_mut() {
            t.after_submit(perf);
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.pass(&mut enc, timer.as_deref_mut().and_then(|t| t.writes(GpuPass::Preview)));
        self.queue.submit(std::iter::once(enc.finish()));
        if let Some(t) = timer.as_deref_mut() {
            t.end_frame(perf);
        }
        t0.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End to end on a real device: timestamps written by passes in two
    /// submits per frame, resolved a frame later, mapped without blocking,
    /// and landing in `perf.gpu`. Also runs every call under wgpu
    /// validation. Skips where the adapter lacks TIMESTAMP_QUERY.
    #[test]
    fn real_device_frames_reach_the_perf_gpu_columns() {
        let Some(rig) = Rig::new() else {
            eprintln!("no adapter with TIMESTAMP_QUERY — skipping");
            return;
        };
        let mut perf = Perf::new(true);
        let mut timer = GpuTimer::new(&rig.device, &rig.queue, rig.backend, 1);
        perf.set_gpu_timing(timer.period_ns(), timer.every());
        assert!(timer.writes(GpuPass::Final).is_some());
        assert!(timer.writes(GpuPass::Final).is_none(), "one pair per pass per frame");
        timer.end_frame(&mut perf); // that pair is never resolved (no main encoder) — dropped
        for _ in 0..12 {
            rig.frame(Some(&mut timer), &mut perf);
            rig.device.poll(wgpu::Maintain::Wait); // test only: let maps land
        }
        let agg = perf.aggregate(None);
        let gpu = &agg["gpu"];
        assert!(gpu["frames"].as_u64().unwrap() >= 8, "{gpu}");
        for name in ["composite.live", "composite.design", "final", "preview"] {
            assert!(gpu["passes"][name]["p50"].as_f64().unwrap() > 0.0, "{name}: {gpu}");
        }
        assert!(gpu["total"]["p50"].as_f64().unwrap() > 0.0);
        perf::set_enabled(true);
    }

    #[test]
    fn query_layout_fits_the_bitmask_and_limits() {
        assert!(PAIRS <= 16, "written is a u16 bitmask");
        assert!(QUERIES_PER_SLOT <= wgpu::QUERY_SET_MAX_QUERIES);
        // The Metal whole-slot resolve lands contiguously in the slot block.
        assert!(QUERIES_PER_SLOT as u64 * wgpu::QUERY_SIZE as u64 <= SLOT_BYTES);
        assert_eq!(GpuPass::Probe as u32, FIXED_PAIRS, "probe pairs follow the fixed passes");
    }

    #[test]
    fn total_sums_per_submit_spans_not_the_gap_between_them() {
        // Main encoder: live composite 100..400, final 450..500 → span 400.
        // Preview submit much later (after present): 10_000..10_050 → 50.
        let ticks = |pair: u32| match pair {
            p if p == GpuPass::CompositeLive as u32 => Some((100, 400)),
            p if p == GpuPass::Final as u32 => Some((450, 500)),
            p if p == GpuPass::Preview as u32 => Some((10_000, 10_050)),
            _ => None,
        };
        let written = (1 << GpuPass::CompositeLive as u32)
            | (1 << GpuPass::Final as u32)
            | (1 << GpuPass::Preview as u32);
        let f = collect_frame(written, 0, 0, ticks);
        assert_eq!(f.ticks[GpuPass::CompositeLive as usize], 300);
        assert_eq!(f.ticks[GpuPass::Final as usize], 50);
        assert_eq!(f.total, 400 + 50);
        assert!(f.ran & (1 << GpuPass::WarpBake as u8) == 0);
        assert!(f.ran & (1 << GpuPass::Probe as u8) == 0);
    }

    #[test]
    fn overlapping_main_passes_are_charged_from_the_previous_end() {
        // TBDR: final's start-of-vertex fires at 120, during the composite's
        // fragments (100..400); its own work is 400..450.
        let ticks = |pair: u32| match pair {
            p if p == GpuPass::CompositeLive as u32 => Some((100, 400)),
            p if p == GpuPass::Final as u32 => Some((120, 450)),
            _ => None,
        };
        let written = (1 << GpuPass::CompositeLive as u32) | (1 << GpuPass::Final as u32);
        let f = collect_frame(written, 0, 0, ticks);
        assert_eq!(f.ticks[GpuPass::CompositeLive as usize], 300);
        assert_eq!(f.ticks[GpuPass::Final as usize], 50);
        assert_eq!(f.total, 350, "passes tile the span");
    }

    #[test]
    fn preview_overlapping_the_main_submit_is_charged_from_its_end() {
        // The app's over-read: the preview blit's begin stamp fires at 150,
        // during the live composite (100..400); its own work is 400..420.
        let ticks = |pair: u32| match pair {
            p if p == GpuPass::CompositeLive as u32 => Some((100, 400)),
            p if p == GpuPass::Preview as u32 => Some((150, 420)),
            _ => None,
        };
        let written = (1 << GpuPass::CompositeLive as u32) | (1 << GpuPass::Preview as u32);
        let f = collect_frame(written, 0, 0, ticks);
        assert_eq!(f.ticks[GpuPass::Preview as usize], 20);
        assert_eq!(f.total, 320, "total is the union, not span + raw preview");
        // A preview with no main passes in the frame (occluded, nothing
        // rendered) keeps its raw span.
        let f = collect_frame(1 << GpuPass::Preview as u32, 0, 0, ticks);
        assert_eq!((f.ticks[GpuPass::Preview as usize], f.total), (270, 270));
    }

    #[test]
    fn a_frame_with_only_unsampled_passes_has_nothing_ran() {
        let f = collect_frame(1 << GpuPass::CompositeLive as u32, 0, 0, |_| None);
        assert_eq!((f.ran, f.total), (0, 0));
    }

    #[test]
    fn probe_extrapolates_timed_mean_over_all_probe_frames() {
        let ticks = |pair: u32| (pair >= FIXED_PAIRS).then_some((1, 101)); // 100 each
        let written = ((1u32 << PROBE_PAIRS) - 1) << FIXED_PAIRS;
        let f = collect_frame(written as u16, PROBE_PAIRS as u8, 20, ticks);
        assert_eq!(f.ticks[GpuPass::Probe as usize], 2_000);
        assert_eq!(f.total, 2_000);
    }

    #[test]
    fn garbage_samples_are_dropped() {
        assert!(!sane(0, 5, 1.0), "never-written query reads 0");
        assert!(!sane(10, 5, 1.0), "end before begin");
        assert!(!sane(1, 2_000_000_000, 1.0), "over a second");
        assert!(!sane(u64::MAX, u64::MAX, 1.0), "Metal counter error value");
        assert!(sane(1, 1_000, 1.0));
    }
}

