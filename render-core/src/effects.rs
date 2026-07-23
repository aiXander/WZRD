//! Effect catalog (§3.6) — built-ins, inline WGSL, project-local WGSL.
//!
//! v1 ships a handful of built-ins as reference implementations. The
//! architectural commitment (D15) is that authors extend the engine by
//! *writing new shader code*, not by picking from a menu — inline WGSL goes
//! straight in `scene.json`, project-local effects live under
//! `<scene_dir>/effects/<name>/{shader.wgsl, descriptor.json}` and hot-reload
//! on save.
//!
//! Every effect — built-in or user-authored — speaks the same shader-side
//! contract: a `fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32>` function
//! and access to a fixed `FrameState` + `LayerParams` binding (see
//! `shaders/effect_template.wgsl`). Built-ins compile into one shared
//! pipeline with an `effect_id` switch; user effects each get their own.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::drivers::{parse_color_value, ScalarValue};

/// Maximum scalar / colour slots the per-binding uniform exposes. Anything
/// past this is just bytes the GPU never reads.
pub const MAX_SCALAR_PARAMS: usize = 8;
pub const MAX_COLOR_PARAMS: usize = 4;

/// Effect-id constants for built-ins — must match the switch in
/// `shaders/builtin.wgsl`.
pub mod builtin_id {
    pub const TINT: u32 = 0;
    pub const HUE_CYCLE: u32 = 1;
    pub const FLASH: u32 = 2;
    pub const WOBBLE: u32 = 3;
}

/// §5.5 UI metadata on a scalar input — lets the UI/agent render a control
/// without guessing ranges. All optional; `None` means "the author didn't
/// say" and consumers pick their own default.
#[derive(Debug, Clone, Default)]
pub struct ScalarMeta {
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub step: Option<f32>,
    pub unit: Option<String>,
    /// Widget hint: "slider" | "knob" | "toggle". Advisory only.
    pub widget: Option<String>,
}

impl ScalarMeta {
    fn range(min: f32, max: f32, step: f32) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            step: Some(step),
            unit: None,
            widget: Some("slider".into()),
        }
    }
}

/// Declared input on an effect. Drives both scene-time param parsing and
/// the params_f / params_c slot layout the shader sees.
#[derive(Debug, Clone)]
pub enum InputSlot {
    Scalar {
        name: String,
        default: f32,
        meta: ScalarMeta,
    },
    Color {
        name: String,
        default: [f32; 4],
        /// Widget hint ("palette" | "color"). Advisory only.
        widget: Option<String>,
    },
}

impl InputSlot {
    pub fn name(&self) -> &str {
        match self {
            InputSlot::Scalar { name, .. } | InputSlot::Color { name, .. } => name,
        }
    }

    /// JSON shape served by `effect.describe` — one entry per input, typed,
    /// with whatever UI metadata the descriptor declared.
    pub fn describe(&self) -> Value {
        match self {
            InputSlot::Scalar {
                name,
                default,
                meta,
            } => serde_json::json!({
                "name": name,
                "type": "float",
                "default": default,
                "min": meta.min,
                "max": meta.max,
                "step": meta.step,
                "unit": meta.unit,
                "widget": meta.widget,
            }),
            InputSlot::Color {
                name,
                default,
                widget,
            } => serde_json::json!({
                "name": name,
                "type": "color",
                "default": default,
                "widget": widget,
            }),
        }
    }
}

/// Static description of an effect — its inputs in declaration order, its
/// shader path/source, its category (built-in vs user). Authors write one
/// of these per project-local effect; built-ins ship hard-coded copies.
#[derive(Debug, Clone)]
pub struct EffectDef {
    pub name: String,
    pub kind: EffectKind,
    pub inputs: Vec<InputSlot>,
}

#[derive(Debug, Clone)]
pub enum EffectKind {
    /// One of the shipped reference effects (`tint`, `hueCycle`, …). All
    /// built-ins share a single shader + pipeline; the effect_id uniform
    /// switches between them.
    BuiltIn { effect_id: u32 },
    /// User WGSL — body of `fn effect(...)`. Pipeline key uniquely identifies
    /// the pipeline cache slot; it is **content-derived** (path + source
    /// hash for project-local effects, source hash for inline ones) so an
    /// edited shader gets a *new* cache slot — the §5.6 live leg keeps
    /// drawing the old pipeline until promote, and cache cleanup is a GC
    /// pass over the keys both legs still reference (never an eviction of a
    /// referenced key).
    User {
        pipeline_key: String,
        /// WGSL source as it was last successfully loaded. Used at pipeline
        /// rebuild time; not re-read on every frame.
        wgsl: String,
        /// Original file path, when applicable. Kept for diagnostics / future
        /// per-file watcher targeting; the registry already handles reloads.
        #[allow(dead_code)]
        source_path: Option<PathBuf>,
    },
}

/// Resolved per-binding param payload. `scalars` and `colors` are parallel
/// to the effect's scalar/colour slots in declaration order — i.e. the
/// N-th scalar input lands in `params_f[N]`, etc.
#[derive(Debug, Clone)]
pub struct EffectBinding {
    #[allow(dead_code)]
    pub def: EffectDef,
    pub scalars: Vec<ScalarValue>,
    pub colors: Vec<[f32; 4]>,
}

impl EffectBinding {
    /// Build a binding from a scene.json `params` object + a resolved effect
    /// definition. Missing params fall back to the input's default; unknown
    /// keys fail loudly.
    pub fn from_params(def: EffectDef, params: &Value) -> Result<Self> {
        let params_obj = match params {
            Value::Null => None,
            Value::Object(m) => Some(m),
            other => bail!("`params` must be an object, got {other}"),
        };

        // Catch typos before silently defaulting.
        if let Some(map) = params_obj {
            let known: std::collections::HashSet<&str> =
                def.inputs.iter().map(|i| i.name()).collect();
            for k in map.keys() {
                if !known.contains(k.as_str()) {
                    bail!(
                        "effect {:?} has no param {:?}; declared inputs: {:?}",
                        def.name,
                        k,
                        def.inputs.iter().map(|i| i.name()).collect::<Vec<_>>()
                    );
                }
            }
        }

        let mut scalars = Vec::new();
        let mut colors = Vec::new();
        for input in &def.inputs {
            match input {
                InputSlot::Scalar { name, default, .. } => {
                    let v = params_obj.and_then(|m| m.get(name));
                    let resolved = match v {
                        Some(value) => ScalarValue::parse(value)
                            .with_context(|| format!("parsing scalar param {name:?}"))?,
                        None => ScalarValue::Const(*default),
                    };
                    scalars.push(resolved);
                }
                InputSlot::Color { name, default, .. } => {
                    let v = params_obj.and_then(|m| m.get(name));
                    let resolved = match v {
                        Some(value) => parse_color_value(value)
                            .with_context(|| format!("parsing colour param {name:?}"))?,
                        None => *default,
                    };
                    colors.push(resolved);
                }
            }
        }
        if scalars.len() > MAX_SCALAR_PARAMS {
            bail!(
                "effect {:?} declares {} scalar inputs but the shader exposes {} slots",
                def.name,
                scalars.len(),
                MAX_SCALAR_PARAMS
            );
        }
        if colors.len() > MAX_COLOR_PARAMS {
            bail!(
                "effect {:?} declares {} colour inputs but the shader exposes {} slots",
                def.name,
                colors.len(),
                MAX_COLOR_PARAMS
            );
        }
        Ok(Self {
            def,
            scalars,
            colors,
        })
    }
}

/// Project-local descriptor (`effects/<name>/descriptor.json`) — minimal
/// Phase 3 shape. We don't model `category` or `spatial` yet; the compositor
/// treats every effect as a single-pass color-only `effect()` function.
#[derive(Debug, Clone, Deserialize)]
struct DescriptorFile {
    #[allow(dead_code)]
    name: Option<String>,
    #[serde(default)]
    inputs: Vec<DescriptorInput>,
}

#[derive(Debug, Clone, Deserialize)]
struct DescriptorInput {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    default: Option<Value>,
    // §5.5 UI metadata — all optional so pre-existing descriptors keep
    // parsing unchanged.
    #[serde(default)]
    min: Option<f32>,
    #[serde(default)]
    max: Option<f32>,
    #[serde(default)]
    step: Option<f32>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    widget: Option<String>,
}

impl DescriptorInput {
    fn into_slot(self) -> Result<InputSlot> {
        match self.ty.as_str() {
            "float" | "f32" | "number" => {
                let default = self
                    .default
                    .as_ref()
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                Ok(InputSlot::Scalar {
                    name: self.name,
                    default,
                    meta: ScalarMeta {
                        min: self.min,
                        max: self.max,
                        step: self.step,
                        unit: self.unit,
                        widget: self.widget,
                    },
                })
            }
            "color" | "rgba" => {
                let default = match &self.default {
                    Some(v) => parse_color_value(v)
                        .with_context(|| format!("colour default for input {:?}", self.name))?,
                    None => [0.0, 0.0, 0.0, 1.0],
                };
                Ok(InputSlot::Color {
                    name: self.name,
                    default,
                    widget: self.widget,
                })
            }
            other => bail!("unsupported input type {other:?} for {:?}", self.name),
        }
    }
}

/// Repository of resolved effect definitions, keyed by the name a scene
/// uses to reference them.
///
/// Built-ins are inserted on construction. Project-local effects are scanned
/// from `<effects_dir>` and re-scanned by the file watcher; an effect with
/// the same name as a built-in shadows the built-in (intentional — lets a
/// project override `hueCycle` without renaming).
pub struct EffectRegistry {
    pub effects: HashMap<String, EffectDef>,
    pub effects_dir: Option<PathBuf>,
    /// Last-seen mtime for the (shader + descriptor) pair of each loaded
    /// user effect, keyed by shader path. Lets [`rescan_disk`] skip
    /// unchanged effects on a directory rescan instead of recompiling every
    /// user pipeline on every save (architecture review v1 #7).
    file_mtimes: HashMap<PathBuf, SystemTime>,
}

impl EffectRegistry {
    pub fn new(effects_dir: Option<PathBuf>) -> Self {
        let mut r = Self {
            effects: HashMap::new(),
            effects_dir,
            file_mtimes: HashMap::new(),
        };
        for d in built_in_defs() {
            r.effects.insert(d.name.clone(), d);
        }
        let _ = r.rescan_disk();
        r
    }

    /// Re-walk the project-local effects directory. Silent on missing dir.
    /// Surfaces per-effect errors via `log::warn` and skips the offender,
    /// keeping the previous valid definition (and pipeline) in place.
    ///
    /// Returns the pipeline cache keys whose source changed (new effect, or
    /// shader/descriptor edited). Callers use this to selectively invalidate
    /// pipelines instead of nuking the whole user-pipeline cache on every
    /// editor save.
    pub fn rescan_disk(&mut self) -> Vec<String> {
        let mut changed: Vec<String> = Vec::new();
        let Some(dir) = self.effects_dir.clone() else {
            return changed;
        };
        if !dir.exists() {
            return changed;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(err) => {
                log::warn!("could not read effects dir {}: {err}", dir.display());
                return changed;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let shader_path = path.join("shader.wgsl");
            let descriptor_path = path.join("descriptor.json");
            if !shader_path.exists() {
                continue;
            }

            // Editor saves often fire 2–3 notify events per atomic write
            // (notify emits create + modify + chmod); without this check the
            // user's whole pipeline cache thrashes on each save.
            let mtime = combined_mtime(&shader_path, &descriptor_path);
            if let (Some(new), Some(prev)) = (mtime, self.file_mtimes.get(&shader_path)) {
                if new == *prev {
                    continue;
                }
            }

            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string());
            let Some(name) = name else { continue };

            match load_user_effect(&name, &shader_path, &descriptor_path) {
                Ok(def) => {
                    log::info!(
                        "(re)loaded effect {:?} from {}",
                        name,
                        shader_path.display()
                    );
                    if let EffectKind::User { pipeline_key, .. } = &def.kind {
                        changed.push(pipeline_key.clone());
                    }
                    self.effects.insert(name, def);
                    if let Some(m) = mtime {
                        self.file_mtimes.insert(shader_path, m);
                    }
                }
                Err(err) => {
                    log::warn!(
                        "skipping effect {:?} ({}): {err:#}",
                        name,
                        shader_path.display()
                    );
                }
            }
        }
        changed
    }

    pub fn resolve_named(&self, name: &str) -> Result<EffectDef> {
        self.effects
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("unknown effect {:?}", name))
    }

    /// §5.5 `effect.describe` payload — one named effect, or the full
    /// catalog sorted by name when `name` is `None`. The single-effect form
    /// includes the WGSL source for user effects (§5.10 — the MCP `effects`
    /// full-depth read serves it to the authoring agent); the catalog form
    /// stays names + inputs only to keep it cheap.
    pub fn describe(&self, name: Option<&str>) -> Result<Value> {
        match name {
            Some(n) => self.resolve_named(n).map(|d| {
                let mut v = describe_def(&d);
                if let EffectKind::User { wgsl, .. } = &d.kind {
                    v["wgsl"] = serde_json::json!(wgsl);
                }
                v
            }),
            None => {
                let mut defs: Vec<&EffectDef> = self.effects.values().collect();
                defs.sort_by(|a, b| a.name.cmp(&b.name));
                Ok(serde_json::json!({
                    "effects": defs.iter().map(|d| describe_def(d)).collect::<Vec<_>>(),
                }))
            }
        }
    }

    /// Build an effect def for an inline `{ inline: true, wgsl: ..., inputs: [...] }`
    /// spec. Doesn't touch the registry — inline effects are one-shot.
    pub fn resolve_inline(&self, spec: &InlineEffectSpec) -> Result<EffectDef> {
        let pipeline_key = format!("inline:{}", short_hash(&spec.wgsl));
        let inputs = spec
            .inputs
            .iter()
            .cloned()
            .map(|i| i.into_slot())
            .collect::<Result<Vec<_>>>()?;
        Ok(EffectDef {
            name: spec.name.clone().unwrap_or_else(|| pipeline_key.clone()),
            kind: EffectKind::User {
                pipeline_key,
                wgsl: spec.wgsl.clone(),
                source_path: None,
            },
            inputs,
        })
    }
}

/// Inline-effect declaration straight off scene.json.
#[derive(Debug, Clone, Deserialize)]
pub struct InlineEffectSpec {
    /// If `false`, treat as built-in lookup by `name`. Required by D15 for
    /// disambiguation; in practice everything we accept here will be true.
    #[serde(default)]
    #[allow(dead_code)]
    pub inline: bool,
    pub wgsl: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    inputs: Vec<DescriptorInput>,
}

fn describe_def(def: &EffectDef) -> Value {
    serde_json::json!({
        "name": def.name,
        "kind": match def.kind {
            EffectKind::BuiltIn { .. } => "builtin",
            EffectKind::User { .. } => "user",
        },
        "inputs": def.inputs.iter().map(InputSlot::describe).collect::<Vec<_>>(),
    })
}

fn combined_mtime(shader: &Path, descriptor: &Path) -> Option<SystemTime> {
    let m1 = fs::metadata(shader).and_then(|m| m.modified()).ok()?;
    let m2 = fs::metadata(descriptor).and_then(|m| m.modified()).ok();
    Some(match m2 {
        Some(m2) if m2 > m1 => m2,
        _ => m1,
    })
}

fn load_user_effect(name: &str, shader_path: &Path, descriptor_path: &Path) -> Result<EffectDef> {
    let wgsl = fs::read_to_string(shader_path)
        .with_context(|| format!("reading {}", shader_path.display()))?;
    let inputs = if descriptor_path.exists() {
        let raw = fs::read_to_string(descriptor_path)
            .with_context(|| format!("reading {}", descriptor_path.display()))?;
        let parsed: DescriptorFile = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", descriptor_path.display()))?;
        parsed
            .inputs
            .into_iter()
            .map(|i| i.into_slot())
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    Ok(EffectDef {
        name: name.to_string(),
        kind: EffectKind::User {
            // Content-derived (§5.6): an edit yields a fresh key, so the
            // old pipeline survives in the cache for the live leg.
            pipeline_key: format!("file:{}#{}", shader_path.display(), short_hash(&wgsl)),
            wgsl,
            source_path: Some(shader_path.to_path_buf()),
        },
        inputs,
    })
}

fn built_in_defs() -> Vec<EffectDef> {
    use builtin_id::*;
    // §5.5: built-ins declare UI metadata like any user descriptor would, so
    // `effect.describe` renders controls without guessing ranges.
    let color = |name: &str, default: [f32; 4]| InputSlot::Color {
        name: name.into(),
        default,
        widget: None,
    };
    vec![
        EffectDef {
            name: "tint".into(),
            kind: EffectKind::BuiltIn { effect_id: TINT },
            inputs: vec![color("color", [1.0, 1.0, 1.0, 1.0])],
        },
        EffectDef {
            name: "hueCycle".into(),
            kind: EffectKind::BuiltIn { effect_id: HUE_CYCLE },
            // params_f[0] = phase (driver-supplied, typically clock.bars / clock.phase),
            // params_c[0..3] = palette stops (4-stop palette).
            inputs: vec![
                InputSlot::Scalar {
                    name: "phase".into(),
                    default: 0.0,
                    meta: ScalarMeta::range(0.0, 1.0, 0.001),
                },
                color("color0", [0.1, 0.7, 0.3, 1.0]),
                color("color1", [0.3, 0.9, 0.4, 1.0]),
                color("color2", [0.9, 0.7, 0.2, 1.0]),
                color("color3", [0.6, 0.2, 0.5, 1.0]),
            ],
        },
        EffectDef {
            name: "flash".into(),
            kind: EffectKind::BuiltIn { effect_id: FLASH },
            // params_f[0] = envelope (typically audio.onset),
            // params_f[1] = base intensity (low-level always-on tint),
            // params_c[0] = flash color.
            inputs: vec![
                InputSlot::Scalar {
                    name: "envelope".into(),
                    default: 0.0,
                    meta: ScalarMeta::range(0.0, 1.0, 0.005),
                },
                InputSlot::Scalar {
                    name: "base".into(),
                    default: 0.0,
                    meta: ScalarMeta::range(0.0, 1.0, 0.005),
                },
                color("color", [1.0, 1.0, 1.0, 1.0]),
            ],
        },
        EffectDef {
            name: "wobble".into(),
            kind: EffectKind::BuiltIn { effect_id: WOBBLE },
            // params_f[0] = amplitude (uv displacement, in normalized units),
            // params_f[1] = frequency,
            // params_f[2] = time (driver),
            // params_c[0] = ink color.
            inputs: vec![
                InputSlot::Scalar {
                    name: "amp".into(),
                    default: 0.02,
                    meta: ScalarMeta::range(0.0, 0.2, 0.001),
                },
                InputSlot::Scalar {
                    name: "freq".into(),
                    default: 8.0,
                    meta: ScalarMeta::range(0.0, 64.0, 0.1),
                },
                InputSlot::Scalar {
                    name: "time".into(),
                    default: 0.0,
                    meta: ScalarMeta {
                        unit: Some("s".into()),
                        ..Default::default()
                    },
                },
                color("color", [0.8, 0.4, 0.9, 1.0]),
            ],
        },
    ]
}

/// 64-bit FNV-1a — good enough to key the pipeline cache off WGSL source.
fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn binds_tint_color() {
        let r = EffectRegistry::new(None);
        let def = r.resolve_named("tint").unwrap();
        let binding =
            EffectBinding::from_params(def, &json!({"color": "#ff8000"})).unwrap();
        assert_eq!(binding.scalars.len(), 0);
        assert_eq!(binding.colors.len(), 1);
        let c = binding.colors[0];
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!((c[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_effect_name() {
        let r = EffectRegistry::new(None);
        let err = r.resolve_named("xshimmer").unwrap_err();
        assert!(format!("{err:#}").contains("unknown effect"));
    }

    #[test]
    fn rejects_unknown_param_key() {
        let r = EffectRegistry::new(None);
        let def = r.resolve_named("tint").unwrap();
        let err =
            EffectBinding::from_params(def, &json!({"colour": "#fff"})).unwrap_err();
        assert!(format!("{err:#}").contains("has no param"));
    }

    #[test]
    fn describe_includes_ui_metadata() {
        let r = EffectRegistry::new(None);
        let v = r.describe(Some("wobble")).unwrap();
        let inputs = v["inputs"].as_array().unwrap();
        let amp = inputs.iter().find(|i| i["name"] == "amp").unwrap();
        assert_eq!(amp["type"], "float");
        assert!((amp["max"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        let all = r.describe(None).unwrap();
        assert!(all["effects"].as_array().unwrap().len() >= 4);
        assert!(r.describe(Some("nope")).is_err());
    }

    #[test]
    fn parses_driver_scalar() {
        let r = EffectRegistry::new(None);
        let def = r.resolve_named("hueCycle").unwrap();
        let binding = EffectBinding::from_params(
            def,
            &json!({ "phase": { "driver": "clock.bars", "n": 8 } }),
        )
        .unwrap();
        assert!(matches!(binding.scalars[0], ScalarValue::Driver(_)));
    }
}
