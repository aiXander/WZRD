//! §5.6 shader pre-flight probe — a load *predictor* for pipelines entering
//! the design leg.
//!
//! The design leg shares the process and GPU with the live leg, so a
//! pathological AI-written shader entering *design* still stalls the *live*
//! output. Before a new pipeline is allowed into the design plan (whatever
//! the entry path — `effect.upsert`, watcher reloads, `scene.load`), the
//! engine renders it ~60 frames to a scratch offscreen target at **half**
//! pack resolution, measures p95 frame time, and scales it up to a predicted
//! full-res p95 (fragment cost scales with pixel count).
//!
//! Two amendments baked in (2026-07-12 design review):
//!
//! - **Overhead calibration.** At reduced res a probe frame is dominated by
//!   fixed per-frame cost (encode, submit, scheduling); naively multiplying
//!   that by the pixel ratio predicts red for shaders that are fine. The
//!   first session probes a trivial shader to measure the fixed floor, then
//!   `predicted = overhead + (measured − overhead) × full_px/probe_px`.
//!   Half res (4× ratio) keeps the correction small — quarter res was
//!   rejected because a 16× ratio amplifies calibration error.
//! - **Pessimistic driver values.** A shader whose cost scales with an
//!   audio-driven param probes cheap in silence and blows the budget on the
//!   first drop. Probe uniforms pin `audio.*` to 1.0 and take scalar params
//!   at their descriptor `max` where one exists (current value otherwise) —
//!   the probe answers "worst case at this venue", not "cost right now".
//!
//! Verdict: three bands against two operator thresholds A < B (predicted
//! full-res p95, ms). Green (< A) passes clean; yellow (A..=B) still swaps
//! into design but is flagged; red (> B) is refused entry entirely — the
//! only hard gate. Thresholds are venue state: they live in the §5.3
//! session sidecar, not scene.json, set via `probe.setThresholds`.
//!
//! Probe frames are **sequenced between live frames** (`ProbeSession::step`
//! consumes a few ms of budget per render-loop iteration), so a probe burst
//! costs the live output a few ms/frame, never a stall. Known residual
//! risk: an in-process probe cannot contain a shader that *hangs* the GPU
//! device — that class is the §5.11 recovery contract's job.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::drivers::{DriverSpec, ScalarValue, SliderBank};
use crate::gpu::{FrameStateGpu, GpuContext, LayerParamsGpu, COMPOSITE_FORMAT};

/// Pipeline-cache key of the trivial calibration shader. Never evicted by
/// the §5.6 pipeline GC (it's in the retain set alongside the built-ins).
pub const PROBE_NULL_KEY: &str = "__wzrd_probe_null";

/// The calibration effect body — deliberately near-zero fragment cost so a
/// probe run of it measures the fixed per-frame floor.
pub const PROBE_NULL_WGSL: &str =
    "fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> { return vec4<f32>(0.0); }";

/// Frames rendered per probed pipeline / per calibration run. The first
/// `PROBE_WARMUP` are discarded (pipeline warmup, first-use residency).
const PROBE_FRAMES: u32 = 60;
const PROBE_WARMUP: u32 = 8;
const CALIBRATION_FRAMES: u32 = 40;

/// Per-iteration probe budget — how much render-loop time a `step` call may
/// consume before yielding back to the live frame.
const STEP_BUDGET: Duration = Duration::from_micros(3500);

// ---------- thresholds ----------

/// Operator thresholds A < B (ms of predicted full-res p95). Atomics so the
/// WS thread writes inline (`probe.setThresholds`) and the render thread
/// reads lock-free, mirroring the §5.4 `Masters` pattern.
pub struct ProbeThresholds {
    a_ms: AtomicU32,
    b_ms: AtomicU32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProbeThresholdsSnapshot {
    pub a_ms: f32,
    pub b_ms: f32,
}

impl Default for ProbeThresholdsSnapshot {
    fn default() -> Self {
        // Budget at 60 Hz is 16.6 ms; defaults leave live headroom.
        Self { a_ms: 8.0, b_ms: 14.0 }
    }
}

impl ProbeThresholds {
    pub fn new() -> Arc<Self> {
        let d = ProbeThresholdsSnapshot::default();
        Arc::new(Self {
            a_ms: AtomicU32::new(d.a_ms.to_bits()),
            b_ms: AtomicU32::new(d.b_ms.to_bits()),
        })
    }

    /// Validate + store. Fails loudly on non-finite / non-positive / A ≥ B.
    pub fn set(&self, a_ms: f32, b_ms: f32) -> Result<()> {
        if !a_ms.is_finite() || !b_ms.is_finite() {
            bail!("thresholds must be finite");
        }
        if a_ms <= 0.0 || b_ms <= 0.0 || b_ms > 1000.0 {
            bail!("thresholds must be in (0, 1000] ms");
        }
        if a_ms >= b_ms {
            bail!("threshold A ({a_ms} ms) must be below B ({b_ms} ms)");
        }
        self.a_ms.store(a_ms.to_bits(), Ordering::Relaxed);
        self.b_ms.store(b_ms.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    pub fn a_ms(&self) -> f32 {
        f32::from_bits(self.a_ms.load(Ordering::Relaxed))
    }
    pub fn b_ms(&self) -> f32 {
        f32::from_bits(self.b_ms.load(Ordering::Relaxed))
    }

    pub fn snapshot(&self) -> ProbeThresholdsSnapshot {
        ProbeThresholdsSnapshot {
            a_ms: self.a_ms(),
            b_ms: self.b_ms(),
        }
    }

    /// Restore from the session sidecar (validated through `set`; a
    /// hand-edited file can't smuggle A ≥ B in).
    pub fn restore(&self, snap: &ProbeThresholdsSnapshot) {
        if let Err(e) = self.set(snap.a_ms, snap.b_ms) {
            log::warn!("ignoring probe thresholds from session sidecar: {e}");
        }
    }
}

// ---------- verdict ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    Green,
    Yellow,
    Red,
}

impl Band {
    pub fn as_str(&self) -> &'static str {
        match self {
            Band::Green => "green",
            Band::Yellow => "yellow",
            Band::Red => "red",
        }
    }

    fn classify(predicted_ms: f32, a_ms: f32, b_ms: f32) -> Self {
        if predicted_ms < a_ms {
            Band::Green
        } else if predicted_ms <= b_ms {
            Band::Yellow
        } else {
            Band::Red
        }
    }
}

/// Per-pipeline verdict, and the payload shape that rides `hot_reload`
/// telemetry / the RPC reply so the authoring agent self-corrects on
/// performance, not just compile errors.
#[derive(Debug, Clone, Serialize)]
pub struct KeyVerdict {
    pub key: String,
    /// Effect name (for humans/logs; keys are content hashes).
    pub label: String,
    pub predicted_p95_ms: f32,
    pub band: String,
    /// 160-px JPEG of the probe target's last frame, base64.
    pub thumbnail_b64: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionResult {
    /// Fixed-floor measurement from this session's calibration item (if it
    /// ran one). The caller stores it for future sessions.
    pub measured_overhead_ms: Option<f32>,
    pub verdicts: Vec<KeyVerdict>,
    /// Worst band across all probed pipelines — red anywhere refuses the
    /// whole apply.
    pub worst_band: Band,
    /// Highest predicted p95 across pipelines (the headline number).
    pub worst_predicted_ms: f32,
}

// ---------- session ----------

/// What the caller (Core) supplies per new pipeline: the cache key, a human
/// label, and a fully-built pessimistic uniform payload.
pub struct ProbeItemSpec {
    pub key: String,
    pub label: String,
    pub layer_params: LayerParamsGpu,
}

struct ProbeItem {
    key: String,
    label: String,
    calibration: bool,
    frames_total: u32,
    frames_done: u32,
    samples: Vec<f32>,
    bind_group: wgpu::BindGroup,
    /// Kept alive for the bind group.
    _layer_buffer: wgpu::Buffer,
    p95_ms: Option<f32>,
    thumbnail_b64: Option<String>,
}

/// An in-flight probe run: renders items' frames in small per-iteration
/// slices so live frames keep flowing (see `STEP_BUDGET`).
pub struct ProbeSession {
    items: Vec<ProbeItem>,
    current: usize,
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    probe_w: u32,
    probe_h: u32,
    /// full_px / probe_px — the fragment-cost scale factor.
    ratio: f32,
    frame_buf: wgpu::Buffer,
}

impl ProbeSession {
    /// `calibrate` prepends a trivial-shader run to measure the fixed floor
    /// (the caller passes true when it has no stored overhead yet). The
    /// `PROBE_NULL_KEY` pipeline must already be in the cache when set.
    pub fn new(
        gpu: &GpuContext,
        full_w: u32,
        full_h: u32,
        specs: Vec<ProbeItemSpec>,
        calibrate: bool,
    ) -> Self {
        let probe_w = (full_w / 2).max(1);
        let probe_h = (full_h / 2).max(1);
        let ratio = (full_w as f32 * full_h as f32) / (probe_w as f32 * probe_h as f32);

        let target_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("probe scratch target"),
            size: wgpu::Extent3d {
                width: probe_w,
                height: probe_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Probe frames evaluate against their own FrameState so the live
        // legs' shared uniform never sees the pessimistic values.
        let frame_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe frame state"),
            contents: bytemuck::bytes_of(&pessimistic_frame_state(0, probe_w, probe_h)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut items = Vec::new();
        if calibrate {
            items.push(Self::build_item(
                gpu,
                &frame_buf,
                ProbeItemSpec {
                    key: PROBE_NULL_KEY.to_string(),
                    label: "calibration".to_string(),
                    layer_params: LayerParamsGpu::build(
                        0,
                        0,
                        &crate::gpu::LayerIdentity {
                            layer_seed: 0.0,
                            layer_index: 0,
                            layer_count: 1,
                            centroid_uv: [0.5, 0.5],
                            bbox_uv: [0.0, 0.0, 1.0, 1.0],
                        },
                        &[],
                        &[],
                    ),
                },
                true,
            ));
        }
        for spec in specs {
            items.push(Self::build_item(gpu, &frame_buf, spec, false));
        }

        Self {
            items,
            current: 0,
            target_texture,
            target_view,
            probe_w,
            probe_h,
            ratio,
            frame_buf,
        }
    }

    fn build_item(
        gpu: &GpuContext,
        frame_buf: &wgpu::Buffer,
        spec: ProbeItemSpec,
        calibration: bool,
    ) -> ProbeItem {
        let layer_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("probe layer params [{}]", spec.label)),
            contents: bytemuck::bytes_of(&spec.layer_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("probe bg [{}]", spec.label)),
            layout: &gpu.layer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu.mask_atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu.mask_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer_buffer.as_entire_binding(),
                },
            ],
        });
        ProbeItem {
            key: spec.key,
            label: spec.label,
            calibration,
            frames_total: if calibration {
                CALIBRATION_FRAMES
            } else {
                PROBE_FRAMES
            },
            frames_done: 0,
            samples: Vec::with_capacity(PROBE_FRAMES as usize),
            bind_group,
            _layer_buffer: layer_buffer,
            p95_ms: None,
            thumbnail_b64: None,
        }
    }

    /// Pipeline keys under test — kept alive by the §5.6 pipeline GC while
    /// the session is in flight.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(|i| i.key.as_str())
    }

    /// Run probe frames until the per-iteration budget is spent. Returns
    /// true once every item has finished (call `finalize` then).
    pub fn step(&mut self, gpu: &GpuContext) -> bool {
        let slice_start = Instant::now();
        while slice_start.elapsed() < STEP_BUDGET {
            let Some(item) = self.items.get_mut(self.current) else {
                return true;
            };
            if item.frames_done >= item.frames_total {
                item.p95_ms = Some(p95(&item.samples));
                if !item.calibration {
                    item.thumbnail_b64 = read_thumbnail(
                        gpu,
                        &self.target_texture,
                        self.probe_w,
                        self.probe_h,
                    );
                }
                self.current += 1;
                continue;
            }
            let Some(pipeline) = gpu.pipeline_cache.get(&item.key) else {
                // Should be unreachable (the caller compiled before probing);
                // treat as a zero-cost item rather than wedging the session.
                log::warn!("probe: pipeline {} vanished from cache — skipping", item.key);
                item.frames_done = item.frames_total;
                continue;
            };

            // One measured probe frame: write the (animated) pessimistic
            // frame state, render, then block until the GPU drains. The
            // blocking wait is the measurement — it's bounded by the very
            // frame cost we're probing, and `step` yields between frames.
            gpu.queue.write_buffer(
                &self.frame_buf,
                0,
                bytemuck::bytes_of(&pessimistic_frame_state(
                    item.frames_done,
                    self.probe_w,
                    self.probe_h,
                )),
            );
            let t0 = Instant::now();
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("probe frame encoder"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("probe pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &item.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));
            gpu.device.poll(wgpu::Maintain::Wait);
            let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
            if item.frames_done >= PROBE_WARMUP {
                item.samples.push(elapsed_ms);
            }
            item.frames_done += 1;
        }
        self.current >= self.items.len()
    }

    /// Compute verdicts. `stored_overhead_ms` is the caller's remembered
    /// calibration floor; a calibration item in this session supersedes it.
    pub fn finalize(self, stored_overhead_ms: Option<f32>, a_ms: f32, b_ms: f32) -> SessionResult {
        let measured_overhead_ms = self
            .items
            .iter()
            .find(|i| i.calibration)
            .and_then(|i| i.p95_ms);
        let overhead = measured_overhead_ms.or(stored_overhead_ms).unwrap_or(0.0);

        let mut verdicts = Vec::new();
        let mut worst_band = Band::Green;
        let mut worst_predicted = 0.0f32;
        for item in self.items.into_iter().filter(|i| !i.calibration) {
            let measured = item.p95_ms.unwrap_or(0.0);
            let predicted = overhead + (measured - overhead).max(0.0) * self.ratio;
            let band = Band::classify(predicted, a_ms, b_ms);
            worst_band = worst_band.max(band);
            worst_predicted = worst_predicted.max(predicted);
            log::info!(
                "probe [{}]: measured p95 {:.2} ms @ half res → predicted {:.2} ms full res ({})",
                item.label,
                measured,
                predicted,
                band.as_str()
            );
            verdicts.push(KeyVerdict {
                key: item.key,
                label: item.label,
                predicted_p95_ms: predicted,
                band: band.as_str().to_string(),
                thumbnail_b64: item.thumbnail_b64,
            });
        }
        SessionResult {
            measured_overhead_ms,
            verdicts,
            worst_band,
            worst_predicted_ms: worst_predicted,
        }
    }
}

/// Worst-case FrameState for probe frames: audio pinned to 1.0 (a shader
/// that scales work with signal probes at full blast), clock phases animated
/// across frames so time-dependent branches execute.
fn pessimistic_frame_state(frame_idx: u32, w: u32, h: u32) -> FrameStateGpu {
    let t = frame_idx as f32 / 60.0;
    FrameStateGpu {
        time: t,
        bar_phase: (t / 2.0).fract(),
        beat_phase: (t * 2.0).fract(),
        bpm: 120.0,
        audio_low: 1.0,
        audio_mid: 1.0,
        audio_high: 1.0,
        onset_low: 1.0,
        onset_mid: 1.0,
        onset_high: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
        resolution: [w as f32, h as f32, 0.0, 0.0],
    }
}

/// Pessimistic scalar for a probed param: descriptor `max` when declared,
/// otherwise the current value evaluated with audio pinned to 1.0.
pub fn pessimistic_scalar(
    value: &ScalarValue,
    meta_max: Option<f32>,
    override_value: Option<f32>,
    sliders: &SliderBank,
) -> f32 {
    if let Some(m) = meta_max {
        return m;
    }
    if let Some(v) = override_value {
        return v;
    }
    match value {
        ScalarValue::Const(v) => *v,
        ScalarValue::Driver(d) => match d {
            DriverSpec::Const(v) => *v,
            DriverSpec::ClockBars { .. }
            | DriverSpec::ClockBeats { .. }
            | DriverSpec::ClockPhase { .. } => 1.0,
            DriverSpec::ClockTime => 3600.0,
            DriverSpec::AudioBand(_) => 1.0,
            DriverSpec::AudioOnset { .. } => 1.0,
            DriverSpec::UiSlider { name, default } => sliders.get(name).unwrap_or(*default),
        },
    }
}

fn p95(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f32) * 0.95).clamp(0.0, (v.len() - 1) as f32) as usize;
    v[idx]
}

/// Blocking readback of the probe target's last frame → 160-px JPEG. Runs
/// once per probed pipeline at session end; the wait is a handful of ms on
/// a texture this small.
fn read_thumbnail(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    w: u32,
    h: u32,
) -> Option<String> {
    let bytes_per_row = ((w * 8 + 255) / 256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe thumbnail readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe thumbnail encoder"),
        });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(enc.finish()));
    let (tx, rx) = crossbeam_channel::bounded(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res.is_ok());
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    if !rx.try_recv().unwrap_or(false) {
        return None;
    }
    let jpeg = {
        let data = buffer.slice(..).get_mapped_range();
        crate::telemetry::encode_jpeg_thumbnail(&data, w, h, bytes_per_row, 160).ok()
    };
    buffer.unmap();
    use base64::Engine;
    jpeg.map(|(bytes, _, _)| base64::engine::general_purpose::STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::SliderBank;

    #[test]
    fn thresholds_validate_and_clamp() {
        let t = ProbeThresholds::new();
        assert_eq!(t.a_ms(), 8.0);
        assert_eq!(t.b_ms(), 14.0);
        assert!(t.set(4.0, 10.0).is_ok());
        assert_eq!(t.a_ms(), 4.0);
        assert!(t.set(10.0, 4.0).is_err()); // A must be < B
        assert!(t.set(f32::NAN, 10.0).is_err());
        assert!(t.set(-1.0, 10.0).is_err());
        // Failed sets leave the previous values intact.
        assert_eq!(t.a_ms(), 4.0);
        assert_eq!(t.b_ms(), 10.0);
    }

    #[test]
    fn band_classification() {
        assert_eq!(Band::classify(3.0, 8.0, 14.0), Band::Green);
        assert_eq!(Band::classify(8.0, 8.0, 14.0), Band::Yellow);
        assert_eq!(Band::classify(14.0, 8.0, 14.0), Band::Yellow);
        assert_eq!(Band::classify(14.1, 8.0, 14.0), Band::Red);
        assert!(Band::Red > Band::Yellow && Band::Yellow > Band::Green);
    }

    #[test]
    fn pessimistic_scalars_pin_worst_case() {
        let sliders = SliderBank::new();
        sliders.set("glow", 0.4);
        // Descriptor max wins over everything.
        assert_eq!(
            pessimistic_scalar(&ScalarValue::Const(0.1), Some(2.0), Some(0.5), &sliders),
            2.0
        );
        // Override next.
        assert_eq!(
            pessimistic_scalar(&ScalarValue::Const(0.1), None, Some(0.5), &sliders),
            0.5
        );
        // Audio drivers pin to 1.0.
        assert_eq!(
            pessimistic_scalar(
                &ScalarValue::Driver(DriverSpec::AudioBand(crate::osc::AudioBand::Low)),
                None,
                None,
                &sliders
            ),
            1.0
        );
        // Sliders read their live value.
        assert_eq!(
            pessimistic_scalar(
                &ScalarValue::Driver(DriverSpec::UiSlider {
                    name: "glow".into(),
                    default: 0.0
                }),
                None,
                None,
                &sliders
            ),
            0.4
        );
    }

    #[test]
    fn p95_of_samples() {
        let v: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert_eq!(p95(&v), 96.0);
        assert_eq!(p95(&[]), 0.0);
    }
}
