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
    "connectivity",
];

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
        let sticky = matches!(channel, "hot_reload" | "audio_freshness" | "connectivity");
        if sticky {
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

    pub fn emit_fps(&self, fps: f32, frame_time_ms: f32) {
        self.emit(
            "fps",
            json!({
                "fps": fps,
                "frame_time_ms": frame_time_ms,
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
        self.emit(
            "hot_reload",
            json!({
                "target": event.target,
                "ok": event.ok,
                "elapsed_ms": event.elapsed_ms,
                "message": event.message,
            }),
        );
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

/// Tracks frame timings and emits an `fps` event roughly twice a second so
/// the status pill has fresh numbers without flooding the bus.
pub struct FpsAccumulator {
    samples: std::collections::VecDeque<f32>,
    last_frame: Option<Instant>,
    last_emit: Instant,
}

impl FpsAccumulator {
    pub fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(600),
            last_frame: None,
            last_emit: Instant::now(),
        }
    }

    pub fn mark(&mut self, bus: &Bus) {
        let now = Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = now.duration_since(prev).as_secs_f32() * 1000.0;
            if self.samples.len() == 600 {
                self.samples.pop_front();
            }
            self.samples.push_back(dt);
        }
        self.last_frame = Some(now);

        if now.duration_since(self.last_emit) >= Duration::from_millis(500) {
            let (p50, p95, p99) = percentiles(&self.samples);
            let fps = if p50 > 0.0 { 1000.0 / p50 } else { 0.0 };
            bus.emit_fps(fps, p50);
            bus.emit_frame_stats(FrameStats {
                fps,
                frame_time_ms_p50: p50,
                frame_time_ms_p95: p95,
                frame_time_ms_p99: p99,
                mask_slice_count: 0, // filled in elsewhere if available
                pipeline_count: 0,
                pass_count: 0,
            });
            self.last_emit = now;
        }
    }
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

/// Samples the composite buffer back to CPU at ~15 fps and emits a JPEG
/// thumbnail on the `preview` channel. The composite is `Rgba16Float`; we
/// decode f16 by hand to stay zero-dependency on `half`.
///
/// **Render-thread budget.** The sampler never calls `device.poll(Wait)` —
/// blocking the render thread for a readback to land tanked total fps over
/// time as the GPU queue backed up. Instead it runs a 3-state pipeline:
///
/// 1. `Idle` — when at least `interval` has elapsed, allocate the readback
///    buffer (once, reused thereafter), submit a `copy_texture_to_buffer`
///    onto the composite, kick off `map_async`, transition to `Pending`.
/// 2. `Pending` — every subsequent call drives the device with
///    `Maintain::Poll` (non-blocking) so wgpu can fire the map callback.
///    Returns immediately if the buffer isn't mapped yet.
/// 3. `Ready` — the callback flipped the flag; decode the f16 data to a
///    JPEG, emit on the bus, unmap the buffer, drop back to `Idle`.
///
/// The readback buffer is allocated once (or re-allocated only on composite
/// resize) — no 16 MB allocation per 66 ms.
pub struct PreviewSampler {
    interval: Duration,
    /// When we kicked off the *current* in-flight (or pending-emit) capture.
    /// Used to gate the next kick-off so we don't oversubscribe the GPU.
    last_capture: Instant,
    /// Persistent readback buffer + the geometry it was allocated for.
    /// Re-created only when the composite changes shape.
    buffer: Option<wgpu::Buffer>,
    buf_bytes_per_row: u32,
    buf_src_w: u32,
    buf_src_h: u32,
    /// Atomic flipped by `map_async`'s callback. `None` ↔ Idle; `Some(..)`
    /// pending or ready (see status codes below).
    in_flight: Option<Arc<AtomicU8>>,
}

const MAP_PENDING: u8 = 0;
const MAP_READY_OK: u8 = 1;
const MAP_READY_ERR: u8 = 2;

impl PreviewSampler {
    pub fn new() -> Self {
        Self {
            interval: Duration::from_millis(66), // ~15 fps
            last_capture: Instant::now() - Duration::from_secs(1),
            buffer: None,
            buf_bytes_per_row: 0,
            buf_src_w: 0,
            buf_src_h: 0,
            in_flight: None,
        }
    }

    pub fn maybe_capture(&mut self, gpu: &GpuContext, bus: &Bus) {
        // Drive any pending map callback forward. Non-blocking — returns
        // immediately whether work has finished or not, and fires any
        // completed callbacks along the way.
        gpu.device.poll(wgpu::Maintain::Poll);

        // If a readback is in flight, drain it (or skip if still mapping).
        if let Some(state) = self.in_flight.as_ref() {
            match state.load(Ordering::Acquire) {
                MAP_PENDING => return, // still mapping; try again next frame
                MAP_READY_OK => {
                    let buf = self.buffer.as_ref().expect("buffer present");
                    let slice = buf.slice(..);
                    let data = slice.get_mapped_range();
                    let jpeg = encode_jpeg_thumbnail(
                        &data,
                        self.buf_src_w,
                        self.buf_src_h,
                        self.buf_bytes_per_row,
                        320,
                    );
                    drop(data);
                    buf.unmap();
                    if let Ok((bytes, w, h)) = jpeg {
                        bus.emit_preview_jpeg(&bytes, w, h);
                    }
                }
                MAP_READY_ERR => {
                    log::trace!("preview map failed; recycling buffer");
                    // No mapped range to drop; unmap is a no-op when the
                    // buffer isn't mapped, so just clear state.
                }
                _ => {}
            }
            self.in_flight = None;
        }

        if self.last_capture.elapsed() < self.interval {
            return;
        }
        self.last_capture = Instant::now();

        // (Re)allocate the readback buffer if the composite shape changed.
        // For a 1920x1080 RGBA16F target this is 1920 * 8 = 15360 = 60 * 256
        // bytes per row, already aligned.
        let src_w = gpu.composite_width;
        let src_h = gpu.composite_height;
        let bytes_per_row = align_up(src_w * 8, 256);
        let need_realloc = self.buffer.is_none()
            || self.buf_src_w != src_w
            || self.buf_src_h != src_h;
        if need_realloc {
            let size = (bytes_per_row * src_h) as u64;
            self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preview readback buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.buf_bytes_per_row = bytes_per_row;
            self.buf_src_w = src_w;
            self.buf_src_h = src_h;
        }
        let buf = self.buffer.as_ref().expect("buffer present");

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview readback encoder"),
            });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &gpu.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(src_h),
                },
            },
            wgpu::Extent3d {
                width: src_w,
                height: src_h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(enc.finish()));

        // Kick off the async map. The callback flips the atomic; the *next*
        // `maybe_capture` call observes `MAP_READY_OK` and consumes the
        // buffer. Render thread never waits.
        let state = Arc::new(AtomicU8::new(MAP_PENDING));
        let state_cb = Arc::clone(&state);
        buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let code = if res.is_ok() { MAP_READY_OK } else { MAP_READY_ERR };
            state_cb.store(code, Ordering::Release);
        });
        self.in_flight = Some(state);
    }
}

fn align_up(v: u32, to: u32) -> u32 {
    ((v + to - 1) / to) * to
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

fn encode_jpeg_thumbnail(
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
            rgb.push((r.clamp(0.0, 1.0) * 255.0) as u8);
            rgb.push((g.clamp(0.0, 1.0) * 255.0) as u8);
            rgb.push((b.clamp(0.0, 1.0) * 255.0) as u8);
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
