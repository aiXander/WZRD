//! Driver bus (§3.7) — anything that produces a per-frame value lives here.
//!
//! Drivers are values, not callbacks: the runtime ticks them once per frame
//! and substitutes the result into the consuming effect's params. v1 ships
//! only scalar-valued (`f32`) drivers — enough to cover the audio-reactive
//! tree scene from §1.2. `Event`-typed drivers (raw onsets, MIDI noteOn) are
//! consumed as decaying envelopes by `audio.onset`; full discrete-event
//! plumbing waits for Phase 6+ cue editing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::osc::{AudioBand, AudioFeatures};

/// Live values for `ui.slider` drivers, keyed by slider name. Written by the
/// IPC layer (`param.set`, handled inline on the WS thread — no render-thread
/// hop, so knob latency is one frame at most) and read once per frame per
/// bound slider by the render thread.
#[derive(Default)]
pub struct SliderBank {
    values: RwLock<HashMap<String, f32>>,
}

impl SliderBank {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set(&self, name: &str, value: f32) {
        self.values
            .write()
            .expect("slider bank lock")
            .insert(name.to_string(), value);
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        self.values
            .read()
            .expect("slider bank lock")
            .get(name)
            .copied()
    }

    pub fn snapshot(&self) -> Vec<(String, f32)> {
        self.values
            .read()
            .expect("slider bank lock")
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// §5.6 full-control-switch: replace this bank's contents with a copy of
    /// `other` (promote copies design→live; pull copies live→design).
    pub fn copy_from(&self, other: &SliderBank) {
        let src = other.values.read().expect("slider bank lock").clone();
        *self.values.write().expect("slider bank lock") = src;
    }
}

/// §5.4 masters — engine-level, operator-owned globals. Deliberately outside
/// `scene.json` (the AI's editing surface): a scene rewrite can never touch
/// them. Written inline on the WS thread by `master.set` (same pattern as
/// [`SliderBank`]), read once per frame by the render thread, persisted via
/// the §5.3 session sidecar. Values are f32 bits in atomics so no lock sits
/// on the frame path.
pub struct Masters {
    brightness: AtomicU32,
    speed: AtomicU32,
    saturation: AtomicU32,
    audio_listen: AtomicU32,
}

/// Plain-value view of [`Masters`] — the shape used by the `masters`
/// telemetry channel and the session sidecar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MastersSnapshot {
    pub brightness: f32,
    pub speed: f32,
    pub saturation: f32,
    pub audio_listen: f32,
}

impl Default for MastersSnapshot {
    fn default() -> Self {
        Self {
            brightness: 1.0,
            speed: 1.0,
            saturation: 1.0,
            audio_listen: 1.0,
        }
    }
}

impl Masters {
    pub fn new() -> Arc<Self> {
        let one = 1.0f32.to_bits();
        Arc::new(Self {
            brightness: AtomicU32::new(one),
            speed: AtomicU32::new(one),
            saturation: AtomicU32::new(one),
            audio_listen: AtomicU32::new(one),
        })
    }

    fn cell(&self, name: &str) -> Option<(&AtomicU32, f32, f32)> {
        match name {
            "brightness" => Some((&self.brightness, 0.0, 2.0)),
            "speed" => Some((&self.speed, 0.0, 8.0)),
            "saturation" => Some((&self.saturation, 0.0, 2.0)),
            "audioListen" | "audio_listen" => Some((&self.audio_listen, 0.0, 1.0)),
            _ => None,
        }
    }

    /// Set a master by name. Clamps into the master's legal range and
    /// returns the value actually stored; unknown names fail loudly.
    pub fn set(&self, name: &str, value: f32) -> Result<f32> {
        let (cell, lo, hi) = self.cell(name).ok_or_else(|| {
            anyhow!("unknown master {name:?} (brightness | speed | saturation | audioListen)")
        })?;
        let v = value.clamp(lo, hi);
        cell.store(v.to_bits(), Ordering::Relaxed);
        Ok(v)
    }

    pub fn brightness(&self) -> f32 {
        f32::from_bits(self.brightness.load(Ordering::Relaxed))
    }
    pub fn speed(&self) -> f32 {
        f32::from_bits(self.speed.load(Ordering::Relaxed))
    }
    pub fn saturation(&self) -> f32 {
        f32::from_bits(self.saturation.load(Ordering::Relaxed))
    }
    pub fn audio_listen(&self) -> f32 {
        f32::from_bits(self.audio_listen.load(Ordering::Relaxed))
    }

    pub fn snapshot(&self) -> MastersSnapshot {
        MastersSnapshot {
            brightness: self.brightness(),
            speed: self.speed(),
            saturation: self.saturation(),
            audio_listen: self.audio_listen(),
        }
    }

    /// Restore from a sidecar snapshot (clamped through the same ranges as
    /// `set` so a hand-edited session.json can't smuggle wild values in).
    pub fn restore(&self, snap: &MastersSnapshot) {
        let _ = self.set("brightness", snap.brightness);
        let _ = self.set("speed", snap.speed);
        let _ = self.set("saturation", snap.saturation);
        let _ = self.set("audioListen", snap.audio_listen);
    }

    /// §5.6 full-control-switch: adopt `other`'s values wholesale.
    pub fn copy_from(&self, other: &Masters) {
        self.restore(&other.snapshot());
    }
}

/// §5.4 crossfade-time master — the default promote fade, in **seconds**.
/// Unlike the per-leg [`Masters`] (brightness/speed/…), a promote is a single
/// engine-wide operator action, so there is exactly one crossfade value for
/// the whole engine: it is *not* duplicated per leg and is never copied on
/// promote/pull. The UI drives it on a logarithmic 0..30 s slider (hand-set
/// fade times over the old CUT/0.5s/2s/8s presets); stored linear here as f32
/// bits in an atomic, same lock-free discipline as `Masters`. Persisted in
/// the §5.3 session sidecar and surfaced on the `masters` telemetry channel.
pub struct Crossfade {
    seconds: AtomicU32,
}

impl Crossfade {
    /// Default matches the historical hard-coded promote fade (500 ms), so an
    /// untouched master reproduces the pre-master behaviour exactly.
    pub const DEFAULT_SECONDS: f32 = 0.5;
    pub const MAX_SECONDS: f32 = 30.0;

    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            seconds: AtomicU32::new(Self::DEFAULT_SECONDS.to_bits()),
        })
    }

    /// Set from seconds; clamps into `0..=30` s (a non-finite value falls back
    /// to the default) and returns the value actually stored.
    pub fn set(&self, seconds: f32) -> f32 {
        let v = if seconds.is_finite() {
            seconds.clamp(0.0, Self::MAX_SECONDS)
        } else {
            Self::DEFAULT_SECONDS
        };
        self.seconds.store(v.to_bits(), Ordering::Relaxed);
        v
    }

    pub fn seconds(&self) -> f32 {
        f32::from_bits(self.seconds.load(Ordering::Relaxed))
    }

    /// The promote path speaks milliseconds.
    pub fn ms(&self) -> f32 {
        self.seconds() * 1000.0
    }
}

/// §5.5 per-binding param override table. `param.set {binding, param, value}`
/// pins any *scalar* param — const or driver-bound — to a live value without
/// a plan rebuild; the compositor consults this table each tick before
/// evaluating the underlying [`ScalarValue`]. Keyed by (binding id, param
/// name), which is also the carry-forward rule: when the AI regenerates an
/// effect, an override survives exactly as long as a scalar param with the
/// same name still exists on the binding (spec §4: hand-tuning carries
/// forward; a vanished param's override just sits inert).
#[derive(Default)]
pub struct ParamOverrides {
    // Nested (binding → param → value) so the per-frame lookup borrows
    // &str keys instead of allocating a (String, String) tuple per param.
    values: RwLock<HashMap<String, HashMap<String, f32>>>,
}

impl ParamOverrides {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set(&self, binding: &str, param: &str, value: f32) {
        self.values
            .write()
            .expect("param overrides lock")
            .entry(binding.to_string())
            .or_default()
            .insert(param.to_string(), value);
    }

    /// Returns true if an override existed.
    pub fn clear(&self, binding: &str, param: &str) -> bool {
        let mut map = self.values.write().expect("param overrides lock");
        let Some(inner) = map.get_mut(binding) else {
            return false;
        };
        let existed = inner.remove(param).is_some();
        if inner.is_empty() {
            map.remove(binding);
        }
        existed
    }

    pub fn get(&self, binding: &str, param: &str) -> Option<f32> {
        self.values
            .read()
            .expect("param overrides lock")
            .get(binding)
            .and_then(|m| m.get(param))
            .copied()
    }

    pub fn snapshot(&self) -> Vec<(String, String, f32)> {
        self.values
            .read()
            .expect("param overrides lock")
            .iter()
            .flat_map(|(b, m)| {
                m.iter()
                    .map(move |(p, v)| (b.clone(), p.clone(), *v))
            })
            .collect()
    }

    /// §5.6 full-control-switch: replace this table with a copy of `other`.
    pub fn copy_from(&self, other: &ParamOverrides) {
        let src = other.values.read().expect("param overrides lock").clone();
        *self.values.write().expect("param overrides lock") = src;
    }
}

/// Scalar (f32) parameter value — either a literal or a driver.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Const(f32),
    Driver(DriverSpec),
}

impl ScalarValue {
    pub fn parse(v: &Value) -> Result<Self> {
        match v {
            Value::Null => Ok(Self::Const(0.0)),
            Value::Bool(b) => Ok(Self::Const(if *b { 1.0 } else { 0.0 })),
            Value::Number(n) => Ok(Self::Const(
                n.as_f64()
                    .ok_or_else(|| anyhow!("scalar must be finite, got {n}"))?
                    as f32,
            )),
            Value::Object(_) => Ok(Self::Driver(DriverSpec::parse(v)?)),
            other => bail!("expected scalar or driver object, got {other}"),
        }
    }

    pub fn eval(&self, frame: &FrameContext) -> f32 {
        match self {
            ScalarValue::Const(v) => *v,
            ScalarValue::Driver(d) => d.eval(frame),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            ScalarValue::Const(v) => format!("const({v})"),
            ScalarValue::Driver(d) => d.describe(),
        }
    }
}

/// All built-in drivers. Each evaluates to a single f32 in (loosely) [0,1] or
/// a small positive range. Effect shaders treat these as plain floats; they
/// don't need to know whether the value came from a literal, the clock, or
/// the audio thread.
#[derive(Debug, Clone)]
pub enum DriverSpec {
    Const(f32),
    /// Phase ramp 0..1 across `n` bars.
    ClockBars { n: f32 },
    /// Phase ramp 0..1 across `n` beats.
    ClockBeats { n: f32 },
    /// Hz-based phase ramp 0..1 across `1/rate` seconds.
    ClockPhase { rate: f32 },
    /// Wallclock seconds since engine start (unbounded; for raw `time`).
    ClockTime,
    AudioBand(AudioBand),
    AudioOnset {
        band: AudioBand,
        decay: f32,
    },
    /// UI slider — reads the live value from the [`SliderBank`] (fed by the
    /// `param.set` RPC); falls back to `default` until first touched.
    UiSlider { name: String, default: f32 },
}

impl DriverSpec {
    pub fn parse(v: &Value) -> Result<Self> {
        let driver = v
            .get("driver")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("driver object missing `driver` field: {v}"))?;
        match driver {
            "const" => {
                let value = v.get("value").and_then(Value::as_f64).unwrap_or(0.0);
                Ok(Self::Const(value as f32))
            }
            "clock.bars" => {
                let n = v.get("n").and_then(Value::as_f64).unwrap_or(1.0).max(1e-3) as f32;
                Ok(Self::ClockBars { n })
            }
            "clock.beats" => {
                let n = v.get("n").and_then(Value::as_f64).unwrap_or(1.0).max(1e-3) as f32;
                Ok(Self::ClockBeats { n })
            }
            "clock.phase" => {
                let rate = v.get("rate").and_then(Value::as_f64).unwrap_or(0.1) as f32;
                Ok(Self::ClockPhase { rate })
            }
            "clock.time" => Ok(Self::ClockTime),
            "audio.rms" => bail!(
                "audio.rms removed in OSC refactor — bind audio.band(\"low|mid|high\") instead"
            ),
            "audio.band" => {
                let band = parse_band(v)?;
                Ok(Self::AudioBand(band))
            }
            "audio.onset" => {
                let band = parse_band(v)?;
                let decay = v.get("decay").and_then(Value::as_f64).unwrap_or(0.15) as f32;
                Ok(Self::AudioOnset { band, decay })
            }
            "ui.slider" => {
                let name = v
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("ui.slider requires `name`"))?
                    .to_string();
                let default = v.get("default").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                Ok(Self::UiSlider { name, default })
            }
            other => bail!("unknown driver {other:?}"),
        }
    }

    pub fn eval(&self, frame: &FrameContext) -> f32 {
        match self {
            DriverSpec::Const(v) => *v,
            DriverSpec::ClockBars { n } => phase(frame.bar_time(), *n),
            DriverSpec::ClockBeats { n } => phase(frame.beat_time(), *n),
            DriverSpec::ClockPhase { rate } => phase(frame.elapsed_sec, 1.0 / rate.max(1e-6)),
            DriverSpec::ClockTime => frame.elapsed_sec,
            DriverSpec::AudioBand(b) => frame.band(*b),
            DriverSpec::AudioOnset { band, decay } => frame.onset(*band, *decay),
            DriverSpec::UiSlider { name, default } => {
                frame.sliders.get(name).unwrap_or(*default)
            }
        }
    }

    /// Short human-readable source label for telemetry ("clock.bars(8)",
    /// "audio.band(low)", "ui.slider(glow)"). Shown in the UI driver rack.
    pub fn describe(&self) -> String {
        match self {
            DriverSpec::Const(v) => format!("const({v})"),
            DriverSpec::ClockBars { n } => format!("clock.bars({n})"),
            DriverSpec::ClockBeats { n } => format!("clock.beats({n})"),
            DriverSpec::ClockPhase { rate } => format!("clock.phase({rate})"),
            DriverSpec::ClockTime => "clock.time".to_string(),
            DriverSpec::AudioBand(b) => format!("audio.band({})", band_name(*b)),
            DriverSpec::AudioOnset { band, .. } => {
                format!("audio.onset({})", band_name(*band))
            }
            DriverSpec::UiSlider { name, .. } => format!("ui.slider({name})"),
        }
    }
}

/// Re-pick cadence for §5.2 `pick` selectors. Deliberately restricted to
/// transport-derived clocks: the pick cycle must be a pure function of
/// transport time so runs are deterministic and the §5.6 design-leg preview
/// picks the same layer its promote will. Audio-driven rates would make the
/// pick history depend on live signal state — rejected at parse.
#[derive(Debug, Clone)]
pub enum PickRate {
    Bars { n: f32 },
    Beats { n: f32 },
    /// From `clock.phase { rate }` — period is `1/rate` seconds.
    Seconds { period: f32 },
}

impl PickRate {
    pub fn parse(v: &Value) -> Result<Self> {
        let driver = v
            .get("driver")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("pick rate object missing `driver` field: {v}"))?;
        match driver {
            "clock.bars" => {
                let n = v.get("n").and_then(Value::as_f64).unwrap_or(1.0).max(1e-3) as f32;
                Ok(Self::Bars { n })
            }
            "clock.beats" => {
                let n = v.get("n").and_then(Value::as_f64).unwrap_or(1.0).max(1e-3) as f32;
                Ok(Self::Beats { n })
            }
            "clock.phase" => {
                let rate = v.get("rate").and_then(Value::as_f64).unwrap_or(0.1) as f32;
                Ok(Self::Seconds {
                    period: 1.0 / rate.max(1e-6),
                })
            }
            other => bail!(
                "pick rate must be a clock.bars / clock.beats / clock.phase driver \
                 (got {other:?}) — picks are transport-locked so runs stay deterministic"
            ),
        }
    }

    /// Monotonic cycle counter: how many whole periods have elapsed. The
    /// pick re-rolls when this changes.
    pub fn cycle(&self, frame: &FrameContext) -> u64 {
        let (t, period) = match self {
            Self::Bars { n } => (frame.bar_time(), *n),
            Self::Beats { n } => (frame.beat_time(), *n),
            Self::Seconds { period } => (frame.elapsed_sec, *period),
        };
        (t / period.max(1e-6)).floor().max(0.0) as u64
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Bars { n } => format!("clock.bars({n})"),
            Self::Beats { n } => format!("clock.beats({n})"),
            Self::Seconds { period } => format!("clock.phase({})", 1.0 / period.max(1e-6)),
        }
    }
}

fn band_name(b: AudioBand) -> &'static str {
    match b {
        AudioBand::Low => "low",
        AudioBand::Mid => "mid",
        AudioBand::High => "high",
    }
}

fn parse_band(v: &Value) -> Result<AudioBand> {
    let band = v
        .get("band")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("audio driver requires `band` (\"low\" | \"mid\" | \"high\")"))?;
    AudioBand::parse(band).ok_or_else(|| anyhow!("unknown band {band:?}"))
}

fn phase(t: f32, period: f32) -> f32 {
    let p = period.max(1e-6);
    let mut x = (t / p).fract();
    if x < 0.0 {
        x += 1.0;
    }
    x
}

/// Sampled-once-per-frame view of clock + audio state. Passed by reference
/// to every driver `eval()` and the FrameState uniform builder.
pub struct FrameContext<'a> {
    pub elapsed_sec: f32,
    pub bpm: f32,
    pub audio: &'a AudioFeatures,
    pub sliders: &'a SliderBank,
    /// §5.4 audio-listen master — scales every `audio.*` read toward 0. Use
    /// [`FrameContext::band`]/[`FrameContext::onset`] instead of touching
    /// `audio` directly so the master applies uniformly (drivers, FrameState
    /// uniform, telemetry all see the same scaled value).
    pub audio_listen: f32,
}

impl<'a> FrameContext<'a> {
    pub fn beat_time(&self) -> f32 {
        self.elapsed_sec * self.bpm / 60.0
    }
    pub fn bar_time(&self) -> f32 {
        // 4 beats per bar, the assumption hard-coded in Phase 3. Time-signature
        // overrides land alongside the cue editor in Phase 6+.
        self.beat_time() / 4.0
    }
    pub fn band(&self, b: AudioBand) -> f32 {
        self.audio.band(b) * self.audio_listen
    }
    pub fn onset(&self, b: AudioBand, decay: f32) -> f32 {
        self.audio.onset_envelope(b, decay) * self.audio_listen
    }
}

/// Musical clock owned by the render thread. Since §5.4 it *integrates* time
/// per frame instead of reading a wall clock: `time += dt · speed`, where
/// `speed` is the operator's speed master. A speed change therefore bends
/// time (phases keep continuity) instead of jumping the absolute clock —
/// the difference between a smooth half-time drop and every phase-driven
/// effect snapping to a new position.
pub struct Transport {
    bpm: f32,
    time_sec: f64,
    last_step: Instant,
}

impl Transport {
    pub fn new(bpm: f32) -> Self {
        Self {
            bpm: bpm.max(1.0),
            time_sec: 0.0,
            last_step: Instant::now(),
        }
    }

    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.max(1.0);
    }

    /// Advance the clock by the wall time since the previous step, scaled by
    /// `speed`. Called exactly once per frame by the render thread before
    /// the plan tick.
    pub fn step(&mut self, speed: f32) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_step).as_secs_f64();
        self.last_step = now;
        self.step_by(dt, speed);
    }

    fn step_by(&mut self, dt: f64, speed: f32) {
        self.time_sec += dt * speed.max(0.0) as f64;
    }

    pub fn elapsed_sec(&self) -> f32 {
        self.time_sec as f32
    }

    pub fn bpm(&self) -> f32 {
        self.bpm
    }

    /// §5.6 full-control-switch: adopt `other`'s musical position + tempo.
    /// On promote the live leg adopts the design clock wholesale, so the
    /// promoted content continues *exactly* as it looked in the design
    /// preview (phases, picks and all — the old live content is gone, so
    /// its phase continuity no longer matters).
    pub fn sync_from(&mut self, other: &Transport) {
        self.bpm = other.bpm;
        self.time_sec = other.time_sec;
        self.last_step = Instant::now();
    }

    pub fn frame_context<'a>(
        &self,
        audio: &'a AudioFeatures,
        sliders: &'a SliderBank,
        audio_listen: f32,
    ) -> FrameContext<'a> {
        FrameContext {
            elapsed_sec: self.elapsed_sec(),
            bpm: self.bpm,
            audio,
            sliders,
            audio_listen,
        }
    }
}

/// Parse a colour parameter — only literals for now. Driver-controlled
/// colours are a follow-up if it ever feels missed; most "make this colour
/// pulse" cases are already covered by feeding a scalar driver into the
/// effect.
pub fn parse_color_value(v: &Value) -> Result<[f32; 4]> {
    match v {
        Value::String(s) => parse_hex_color(s),
        Value::Array(arr) => {
            let nums: Vec<f32> = arr
                .iter()
                .map(|n| {
                    n.as_f64()
                        .map(|x| x as f32)
                        .ok_or_else(|| anyhow!("colour array entry {n} is not a number"))
                })
                .collect::<Result<_>>()?;
            match nums.as_slice() {
                [r, g, b] => Ok([*r, *g, *b, 1.0]),
                [r, g, b, a] => Ok([*r, *g, *b, *a]),
                _ => bail!("colour arrays must be length 3 or 4, got {}", nums.len()),
            }
        }
        other => bail!("unsupported colour value: {other}"),
    }
}

fn parse_hex_color(s: &str) -> Result<[f32; 4]> {
    let s = s.trim_start_matches('#');
    let bytes = match s.len() {
        3 => {
            let r = u8::from_str_radix(&s[0..1].repeat(2), 16)?;
            let g = u8::from_str_radix(&s[1..2].repeat(2), 16)?;
            let b = u8::from_str_radix(&s[2..3].repeat(2), 16)?;
            [r, g, b, 255]
        }
        4 => {
            let r = u8::from_str_radix(&s[0..1].repeat(2), 16)?;
            let g = u8::from_str_radix(&s[1..2].repeat(2), 16)?;
            let b = u8::from_str_radix(&s[2..3].repeat(2), 16)?;
            let a = u8::from_str_radix(&s[3..4].repeat(2), 16)?;
            [r, g, b, a]
        }
        6 => {
            let r = u8::from_str_radix(&s[0..2], 16)?;
            let g = u8::from_str_radix(&s[2..4], 16)?;
            let b = u8::from_str_radix(&s[4..6], 16)?;
            [r, g, b, 255]
        }
        8 => {
            let r = u8::from_str_radix(&s[0..2], 16)?;
            let g = u8::from_str_radix(&s[2..4], 16)?;
            let b = u8::from_str_radix(&s[4..6], 16)?;
            let a = u8::from_str_radix(&s[6..8], 16)?;
            [r, g, b, a]
        }
        n => bail!("hex colour must be 3/4/6/8 chars long, got {n}"),
    };
    Ok([
        bytes[0] as f32 / 255.0,
        bytes[1] as f32 / 255.0,
        bytes[2] as f32 / 255.0,
        bytes[3] as f32 / 255.0,
    ])
}

/// Convenience: build a black `Arc<AudioFeatures>` for tests + headless runs.
#[allow(dead_code)]
pub fn null_audio() -> Arc<AudioFeatures> {
    AudioFeatures::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masters_clamp_and_reject_unknown() {
        let m = Masters::new();
        assert_eq!(m.set("brightness", 5.0).unwrap(), 2.0);
        assert_eq!(m.set("audioListen", -1.0).unwrap(), 0.0);
        assert_eq!(m.set("audio_listen", 0.5).unwrap(), 0.5);
        assert!(m.set("volume", 1.0).is_err());
        assert_eq!(m.brightness(), 2.0);
        assert_eq!(m.audio_listen(), 0.5);
    }

    #[test]
    fn crossfade_clamps_and_converts() {
        let c = Crossfade::new();
        assert_eq!(c.seconds(), Crossfade::DEFAULT_SECONDS);
        assert_eq!(c.ms(), 500.0);
        assert_eq!(c.set(-3.0), 0.0); // below range → CUT
        assert_eq!(c.set(100.0), Crossfade::MAX_SECONDS); // above range → 30 s
        assert_eq!(c.set(f32::NAN), Crossfade::DEFAULT_SECONDS); // non-finite → default
        assert_eq!(c.set(8.0), 8.0);
        assert_eq!(c.ms(), 8000.0);
    }

    #[test]
    fn overrides_set_get_clear() {
        let o = ParamOverrides::new();
        assert_eq!(o.get("b1", "amp"), None);
        o.set("b1", "amp", 0.7);
        assert_eq!(o.get("b1", "amp"), Some(0.7));
        assert_eq!(o.get("b1", "freq"), None);
        assert!(o.clear("b1", "amp"));
        assert!(!o.clear("b1", "amp"));
        assert_eq!(o.get("b1", "amp"), None);
        assert!(o.snapshot().is_empty());
    }

    #[test]
    fn transport_speed_bends_time() {
        let mut t = Transport::new(120.0);
        t.step_by(1.0, 1.0);
        t.step_by(1.0, 0.5);
        t.step_by(1.0, 0.0);
        assert!((t.elapsed_sec() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn audio_listen_scales_audio_drivers_only() {
        let audio = null_audio();
        let sliders = SliderBank::new();
        let t = Transport::new(120.0);
        let ctx = t.frame_context(&audio, &sliders, 0.0);
        // Bands are 0 on null audio anyway; the point is the scaling path
        // compiles and clock drivers ignore the listen master.
        assert_eq!(ctx.band(AudioBand::Low), 0.0);
        let clock = DriverSpec::ClockPhase { rate: 1.0 };
        let _ = clock.eval(&ctx);
    }
}
