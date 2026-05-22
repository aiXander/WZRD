//! OSC receiver — pulls per-frame audio features from the standalone
//! Realtime Audio Feature Server (Python, separate process). See
//! `Realtime_PyAudio_FFT/README.md` for the wire contract. We bind a UDP
//! socket on the configured port, decode rosc packets on a dedicated
//! thread, and write into `AudioFeatures` via relaxed atomic stores. The
//! render thread never blocks.
//!
//! Lifecycle: socket is bound at startup whether or not the server is
//! running. Packets flow when the server is up; atomics freeze when it's
//! down. Restart in either direction is invisible to the rest of the
//! engine — the only side-effect is a single watchdog log line per
//! fresh ↔ stale transition (see `audio_refactor_plan.md` §3.1).

use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Shared atomic snapshot of the latest audio features ingested over OSC.
/// f32 fields stored as `AtomicU32` bit patterns so the render thread can
/// load them without locks.
pub struct AudioFeatures {
    /// From `/audio/meta`; diagnostic only (not used in render math today).
    sample_rate: AtomicU32,
    /// `/audio/lmh` args — already auto-scaled by the server into ~[0, 1].
    band_low: AtomicU32,
    band_mid: AtomicU32,
    band_high: AtomicU32,
    /// Engine-local millisecond timestamps stamped on `/audio/onset/<band>`.
    onset_low_ms: AtomicU64,
    onset_mid_ms: AtomicU64,
    onset_high_ms: AtomicU64,
    /// Engine-local ms of the most-recently-dispatched OSC packet. Powers
    /// the fresh ↔ stale watchdog in the recv loop. `0` until the first
    /// packet arrives.
    last_packet_ms: AtomicU64,
    start: Instant,
}

impl AudioFeatures {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            sample_rate: AtomicU32::new(0),
            band_low: AtomicU32::new(0),
            band_mid: AtomicU32::new(0),
            band_high: AtomicU32::new(0),
            onset_low_ms: AtomicU64::new(0),
            onset_mid_ms: AtomicU64::new(0),
            onset_high_ms: AtomicU64::new(0),
            last_packet_ms: AtomicU64::new(0),
            start: Instant::now(),
        })
    }

    pub fn band(&self, band: AudioBand) -> f32 {
        let raw = match band {
            AudioBand::Low => self.band_low.load(Ordering::Relaxed),
            AudioBand::Mid => self.band_mid.load(Ordering::Relaxed),
            AudioBand::High => self.band_high.load(Ordering::Relaxed),
        };
        f32::from_bits(raw)
    }

    /// Returns a decaying onset envelope in [0,1]. 1.0 right after the
    /// trigger fires, exponentially decays with the given time constant.
    /// Matches the prior cpal-driven shape so shaders need no changes.
    pub fn onset_envelope(&self, band: AudioBand, decay_seconds: f32) -> f32 {
        let stamp = match band {
            AudioBand::Low => self.onset_low_ms.load(Ordering::Relaxed),
            AudioBand::Mid => self.onset_mid_ms.load(Ordering::Relaxed),
            AudioBand::High => self.onset_high_ms.load(Ordering::Relaxed),
        };
        if stamp == 0 {
            return 0.0;
        }
        let now_ms = self.now_ms();
        let dt_ms = now_ms.saturating_sub(stamp);
        let dt_s = (dt_ms as f32) * 1e-3;
        (-dt_s / decay_seconds.max(1e-3)).exp().clamp(0.0, 1.0)
    }

    /// True if a packet arrived in the last `stale_after_ms` ms.
    pub fn is_fresh(&self, stale_after_ms: u64) -> bool {
        let stamp = self.last_packet_ms.load(Ordering::Relaxed);
        if stamp == 0 {
            return false;
        }
        self.now_ms().saturating_sub(stamp) <= stale_after_ms
    }

    /// Engine-local timestamp of the most-recently-dispatched OSC packet
    /// (millis since `AudioFeatures::new`). `0` until the first packet
    /// arrives. Used by the Phase 4 audio_freshness telemetry channel.
    pub fn last_packet_ms(&self) -> u64 {
        self.last_packet_ms.load(Ordering::Relaxed)
    }

    /// `/audio/meta`-reported sample rate. `0` until the audio server sends
    /// one. Diagnostic-only; not used in render math.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed)
    }

    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    fn store_f32(&self, slot: &AtomicU32, value: f32) {
        slot.store(value.to_bits(), Ordering::Relaxed);
    }

    fn stamp_onset_ms(&self, slot: &AtomicU64) {
        // No refractory clamp here — the server already enforces per-band
        // refractory windows. Trust the wire.
        let now_ms = self.now_ms().max(1);
        slot.store(now_ms, Ordering::Relaxed);
    }

    fn mark_packet(&self) {
        let now_ms = self.now_ms().max(1);
        self.last_packet_ms.store(now_ms, Ordering::Relaxed);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AudioBand {
    Low,
    Mid,
    High,
}

impl AudioBand {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "mid" => Some(Self::Mid),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

/// Owns the recv thread. Dropped when the engine shuts down. The thread
/// itself runs the OS event loop on UDP recv and never exits during normal
/// operation — we rely on process exit to clean it up.
pub struct OscListener {
    #[allow(dead_code)]
    join: JoinHandle<()>,
}

/// Bind the UDP socket + spawn the recv thread. Failure is informational:
/// the render path keeps running with zeroed features so a busy port or a
/// permission denial doesn't take the projector down.
pub fn try_spawn(state: Arc<AudioFeatures>, addr: SocketAddr) -> Option<OscListener> {
    match spawn(state, addr) {
        Ok(listener) => {
            log::info!("OSC listening on {addr} (waiting for packets…)");
            Some(listener)
        }
        Err(err) => {
            log::warn!("OSC disabled: {err:#}");
            None
        }
    }
}

fn spawn(state: Arc<AudioFeatures>, addr: SocketAddr) -> Result<OscListener> {
    let socket = UdpSocket::bind(addr).with_context(|| format!("binding UDP {addr}"))?;
    socket
        .set_read_timeout(Some(Duration::from_millis(250)))
        .context("setting socket read timeout")?;

    let join = thread::Builder::new()
        .name("osc-recv".into())
        .spawn(move || recv_loop(state, socket))
        .context("spawning osc-recv thread")?;

    Ok(OscListener { join })
}

fn recv_loop(state: Arc<AudioFeatures>, socket: UdpSocket) {
    let mut buf = [0u8; 8192]; // rosc decode_udp expects a contiguous packet
    let mut was_fresh = false;
    loop {
        match socket.recv_from(&mut buf) {
            Ok((n, _peer)) => match rosc::decoder::decode_udp(&buf[..n]) {
                Ok((_, rosc::OscPacket::Message(msg))) => dispatch(&state, &msg),
                Ok((_, rosc::OscPacket::Bundle(bundle))) => dispatch_bundle(&state, &bundle),
                Err(e) => log::trace!("osc decode: {e}"),
            },
            Err(e)
                if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut =>
            {
                // Tick: no packet this interval, fall through to watchdog edge check.
            }
            Err(e) => {
                log::warn!("osc recv: {e}");
                thread::sleep(Duration::from_millis(50));
            }
        }

        // Watchdog: one log line per fresh ↔ stale transition.
        let fresh = state.is_fresh(2_000);
        if fresh != was_fresh {
            if fresh {
                log::info!("OSC: connected (packets arriving)");
            } else if state.last_packet_ms.load(Ordering::Relaxed) != 0 {
                log::warn!(
                    "OSC: stale (no packets for >2s) — is the audio server running?"
                );
            }
            was_fresh = fresh;
        }
    }
}

fn dispatch_bundle(state: &AudioFeatures, bundle: &rosc::OscBundle) {
    for packet in &bundle.content {
        match packet {
            rosc::OscPacket::Message(m) => dispatch(state, m),
            rosc::OscPacket::Bundle(b) => dispatch_bundle(state, b),
        }
    }
}

fn dispatch(state: &AudioFeatures, msg: &rosc::OscMessage) {
    state.mark_packet();
    match msg.addr.as_str() {
        "/audio/lmh" => {
            if let (Some(low), Some(mid), Some(high)) = (
                msg.args.first().and_then(arg_f32),
                msg.args.get(1).and_then(arg_f32),
                msg.args.get(2).and_then(arg_f32),
            ) {
                state.store_f32(&state.band_low, low);
                state.store_f32(&state.band_mid, mid);
                state.store_f32(&state.band_high, high);
            } else {
                log::trace!("/audio/lmh: expected 3 floats, got {:?}", msg.args);
            }
        }
        "/audio/onset/low" => state.stamp_onset_ms(&state.onset_low_ms),
        "/audio/onset/mid" => state.stamp_onset_ms(&state.onset_mid_ms),
        "/audio/onset/high" => state.stamp_onset_ms(&state.onset_high_ms),
        "/audio/meta" => {
            if let Some(sr) = msg.args.first().and_then(arg_i32) {
                let prev = state.sample_rate.swap(sr as u32, Ordering::Relaxed);
                if prev != sr as u32 {
                    log::info!("OSC /audio/meta: sample_rate = {sr} Hz");
                }
            }
        }
        // Decoded so we don't log them as "unhandled"; values discarded in v1.
        "/audio/bpm" | "/audio/fft" => {}
        other => log::trace!("unhandled OSC addr {other}"),
    }
}

fn arg_f32(t: &rosc::OscType) -> Option<f32> {
    match t {
        rosc::OscType::Float(v) => Some(*v),
        rosc::OscType::Double(v) => Some(*v as f32),
        rosc::OscType::Int(v) => Some(*v as f32),
        _ => None,
    }
}

fn arg_i32(t: &rosc::OscType) -> Option<i32> {
    match t {
        rosc::OscType::Int(v) => Some(*v),
        rosc::OscType::Long(v) => Some(*v as i32),
        rosc::OscType::Float(v) => Some(*v as i32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(addr: &str, args: Vec<rosc::OscType>) -> rosc::OscMessage {
        rosc::OscMessage {
            addr: addr.to_string(),
            args,
        }
    }

    #[test]
    fn lmh_updates_band_atomics() {
        let state = AudioFeatures::new();
        dispatch(
            &state,
            &msg(
                "/audio/lmh",
                vec![
                    rosc::OscType::Float(0.25),
                    rosc::OscType::Float(0.50),
                    rosc::OscType::Float(0.75),
                ],
            ),
        );
        assert!((state.band(AudioBand::Low) - 0.25).abs() < 1e-6);
        assert!((state.band(AudioBand::Mid) - 0.50).abs() < 1e-6);
        assert!((state.band(AudioBand::High) - 0.75).abs() < 1e-6);
        assert!(state.last_packet_ms.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn onset_stamps_and_envelope_decays() {
        let state = AudioFeatures::new();
        // Fresh state — envelope is zero before any onset arrives.
        assert_eq!(state.onset_envelope(AudioBand::Low, 0.1), 0.0);
        dispatch(&state, &msg("/audio/onset/low", vec![rosc::OscType::Int(1)]));
        // Right after the trigger, envelope should be ~1.0 (some sub-ms drift OK).
        let v = state.onset_envelope(AudioBand::Low, 0.5);
        assert!(v > 0.95, "expected near 1.0 right after trigger, got {v}");
    }

    #[test]
    fn meta_records_sample_rate() {
        let state = AudioFeatures::new();
        dispatch(&state, &msg("/audio/meta", vec![rosc::OscType::Int(48_000)]));
        assert_eq!(state.sample_rate.load(Ordering::Relaxed), 48_000);
    }

    #[test]
    fn bpm_and_fft_dropped_silently() {
        let state = AudioFeatures::new();
        dispatch(&state, &msg("/audio/bpm", vec![rosc::OscType::Float(128.0)]));
        dispatch(&state, &msg("/audio/fft", vec![rosc::OscType::Float(0.1)]));
        // Just confirms the packet was marked (no panic, no unhandled-log noise).
        assert!(state.last_packet_ms.load(Ordering::Relaxed) > 0);
    }
}
