//! Driver bus (§3.7) — anything that produces a per-frame value lives here.
//!
//! Drivers are values, not callbacks: the runtime ticks them once per frame
//! and substitutes the result into the consuming effect's params. v1 ships
//! only scalar-valued (`f32`) drivers — enough to cover the audio-reactive
//! tree scene from §1.2. `Event`-typed drivers (raw onsets, MIDI noteOn) are
//! consumed as decaying envelopes by `audio.onset`; full discrete-event
//! plumbing waits for Phase 6+ cue editing.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::osc::{AudioBand, AudioFeatures};

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
    /// UI slider — stubbed in Phase 3 (no webview yet), returns `default`.
    UiSlider {
        #[allow(dead_code)]
        name: String,
        default: f32,
    },
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
            DriverSpec::AudioBand(b) => frame.audio.band(*b),
            DriverSpec::AudioOnset { band, decay } => frame.audio.onset_envelope(*band, *decay),
            DriverSpec::UiSlider { default, .. } => *default,
        }
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
}

/// Cheap wall-clock owned by the main thread. Stepped each frame; provides
/// the elapsed time and (for Phase 3) the BPM read from the scene file.
pub struct Transport {
    start: Instant,
    bpm: f32,
}

impl Transport {
    pub fn new(bpm: f32) -> Self {
        Self {
            start: Instant::now(),
            bpm: bpm.max(1.0),
        }
    }

    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm.max(1.0);
    }

    pub fn elapsed_sec(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    pub fn frame_context<'a>(&self, audio: &'a AudioFeatures) -> FrameContext<'a> {
        FrameContext {
            elapsed_sec: self.elapsed_sec(),
            bpm: self.bpm,
            audio,
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
