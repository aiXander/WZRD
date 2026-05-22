//! Built-in effects. Phase 2 ships exactly one: `tint`. The shape of this
//! module is the place new built-ins land (§3.6) and the place the project-
//! local + inline-WGSL discovery (D15) plugs in during Phase 3.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

/// Resolved effect instance for a single binding, ready to bind to a pass.
#[derive(Debug, Clone)]
pub struct EffectInstance {
    /// Effect kind — surfaced in error messages and the eventual telemetry
    /// stream. Phase 2 has no consumer beyond logs / tests.
    #[allow(dead_code)]
    pub name: String,
    pub params: EffectParams,
}

#[derive(Debug, Clone)]
pub enum EffectParams {
    /// `tint` — paint the layer's mask in a single colour. Acts as both the
    /// debug baseline and Phase 2's only built-in.
    Tint { color: [f32; 4] },
}

impl EffectInstance {
    pub fn from_spec(name: &str, params: &Value) -> Result<Self> {
        match name {
            "tint" => {
                let color = params
                    .get("color")
                    .ok_or_else(|| anyhow!("`tint` effect requires a `color` param"))?;
                let rgba = parse_color(color)
                    .with_context(|| format!("parsing tint.color = {color}"))?;
                Ok(EffectInstance {
                    name: name.to_string(),
                    params: EffectParams::Tint { color: rgba },
                })
            }
            other => bail!(
                "unknown effect {other:?}. Phase 2 ships only `tint`; project-local \
                 + inline-WGSL effects land in Phase 3."
            ),
        }
    }

    #[cfg(test)]
    pub fn tint_color(&self) -> [f32; 4] {
        match &self.params {
            EffectParams::Tint { color } => *color,
        }
    }
}

/// Accepts `"#rgb"`, `"#rgba"`, `"#rrggbb"`, `"#rrggbbaa"`, or a 3/4-element
/// JSON array of floats in [0,1]. Tight subset — extend as more effects need
/// other input types.
fn parse_color(v: &Value) -> Result<[f32; 4]> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_named_tint() {
        let inst = EffectInstance::from_spec("tint", &json!({"color": "#ff8000"})).unwrap();
        let c = inst.tint_color();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!((c[2] - 0.0).abs() < 1e-6);
        assert!((c[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_effect() {
        let err = EffectInstance::from_spec("xshimmer", &json!({})).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("unknown effect"), "{msg}");
    }
}
