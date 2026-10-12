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
    pub const VIDEO_CLIP: u32 = 4;
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
    /// Gig plan C5 — a texture input (HAP clip or still) read in WGSL via
    /// `sample_image(uv)`. Takes no uniform slot; at most one per effect.
    Image { name: String },
}

impl InputSlot {
    pub fn name(&self) -> &str {
        match self {
            InputSlot::Scalar { name, .. }
            | InputSlot::Color { name, .. }
            | InputSlot::Image { name } => name,
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
            InputSlot::Image { name } => serde_json::json!({
                "name": name,
                "type": "image",
                "default": null,
            }),
        }
    }
}

/// C5 placement: where the image lands on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImagePlacement {
    /// Scaled into each resolved layer's padded `source_box` (`bbox`
    /// fallback) — one clip reusable across regions. The C5 default.
    Region,
    /// Full-canvas image, cut by the mask.
    Canvas,
}

/// The parsed `image` param of a binding (gig plan C5):
/// `{ "clip": "<path rel. to scene dir>", "placement"?, "rate"? }`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageParam {
    pub clip: String,
    pub placement: ImagePlacement,
    /// Playhead rate (stateless: `(leg time × rate) mod duration`). Ignored
    /// for stills.
    pub rate: f32,
}

impl ImageParam {
    pub fn parse(v: &Value) -> Result<Self> {
        let Value::Object(m) = v else {
            bail!(
                "image param must be an object {{\"clip\": \"generated/x.mov\", \"placement\"?: \
                 \"region\"|\"canvas\", \"rate\"?: number}}, got {v}"
            );
        };
        for k in m.keys() {
            if !matches!(k.as_str(), "clip" | "placement" | "rate") {
                bail!("image param has no key {k:?}; expected clip / placement / rate");
            }
        }
        let clip = match m.get("clip") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => bail!("image param needs a non-empty string \"clip\" (path relative to the scene dir)"),
        };
        let placement = match m.get("placement") {
            None | Some(Value::Null) => ImagePlacement::Region,
            Some(Value::String(s)) if s == "region" => ImagePlacement::Region,
            Some(Value::String(s)) if s == "canvas" => ImagePlacement::Canvas,
            Some(other) => bail!("image placement must be \"region\" or \"canvas\", got {other}"),
        };
        let rate = match m.get("rate") {
            None | Some(Value::Null) => 1.0,
            Some(r) => match r.as_f64() {
                Some(x) if x.is_finite() => x as f32,
                _ => bail!("image rate must be a finite number, got {r}"),
            },
        };
        Ok(Self {
            clip,
            placement,
            rate,
        })
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
    /// Where the effect may emit light — decides whether the compositor
    /// scissors its passes to the layer's mask support (§2.4).
    pub bounds: Bounds,
}

/// An effect's spatial reach (§2.4 "output bounds"). The compositor clips a
/// [`Bounds::Region`] pass to its layer's non-zero mask box (padded for the
/// linear filter) — invisible for any effect whose output is scaled by the
/// `mask` argument, since outside that box `mask` is exactly 0, and a large
/// fragment saving for small regions. [`Bounds::Canvas`] passes shade the
/// whole canvas, as every pass did before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bounds {
    Region,
    Canvas,
}

impl Bounds {
    pub fn as_str(self) -> &'static str {
        match self {
            Bounds::Region => "region",
            Bounds::Canvas => "canvas",
        }
    }

    /// Descriptor / inline `"bounds"` override, else inferred from the
    /// source by [`Bounds::infer`].
    fn resolve(declared: Option<&str>, wgsl: &str) -> Result<Self> {
        match declared {
            None => Ok(Self::infer(wgsl)),
            Some("region") => Ok(Bounds::Region),
            Some("canvas") => Ok(Bounds::Canvas),
            Some(other) => bail!("\"bounds\" must be \"region\" or \"canvas\", got {other:?}"),
        }
    }

    /// The conservative inference rule for an undeclared user effect:
    /// `Canvas` when the body reads the mask anywhere but through its
    /// `mask` argument (`sample_mask(…)` / `mask_atlas` — e.g. a
    /// displaced-edge wobble lights pixels whose own mask is 0), or never
    /// uses the `mask` argument at all (a deliberate full-canvas effect);
    /// `Region` otherwise. A body that fails to parse is `Canvas` (its
    /// compile fails on its own).
    pub fn infer(wgsl: &str) -> Self {
        let code = strip_wgsl_comments(wgsl);
        if code.contains("sample_mask") || code.contains("mask_atlas") || !mask_arg_used(wgsl) {
            Bounds::Canvas
        } else {
            Bounds::Region
        }
    }
}

/// Does `fn effect`'s second argument feed anything? naga's analyzer
/// counts every expression/statement use of the argument expression.
fn mask_arg_used(body: &str) -> bool {
    let composed = crate::gpu::compose_shader(body);
    let Ok(module) = naga::front::wgsl::parse_str(&composed) else {
        return false;
    };
    let Ok(info) = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module) else {
        return false;
    };
    let Some((handle, func)) = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("effect"))
    else {
        return false;
    };
    let fi = &info[handle];
    let used = func
        .expressions
        .iter()
        .any(|(h, e)| matches!(e, naga::Expression::FunctionArgument(1)) && fi[h].ref_count > 0);
    used
}

/// `//` and `/* */` comments blanked, so a comment that merely mentions
/// `sample_mask` doesn't count as a call.
fn strip_wgsl_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("//") {
            rest = r.find('\n').map_or("", |i| &r[i..]);
        } else if let Some(r) = rest.strip_prefix("/*") {
            rest = r.find("*/").map_or("", |i| &r[i + 2..]);
            out.push(' ');
        } else {
            let c = rest.chars().next().unwrap_or(' ');
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
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
    /// The C5 `image` param when the effect declares an image input and the
    /// binding sets it. `None` = the input samples transparent black.
    pub image: Option<ImageParam>,
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
        let mut image = None;
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
                InputSlot::Image { name } => {
                    if let Some(value) = params_obj.and_then(|m| m.get(name)) {
                        if !value.is_null() {
                            image = Some(
                                ImageParam::parse(value)
                                    .with_context(|| format!("parsing image param {name:?}"))?,
                            );
                        }
                    }
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
            image,
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
    /// `"region"` | `"canvas"` — see [`Bounds`]; absent = inferred.
    #[serde(default)]
    bounds: Option<String>,
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
            "image" => Ok(InputSlot::Image { name: self.name }),
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
        let inputs = parse_inputs(spec.inputs.clone())?;
        let bounds = Bounds::resolve(spec.bounds.as_deref(), &spec.wgsl)?;
        Ok(EffectDef {
            name: spec.name.clone().unwrap_or_else(|| pipeline_key.clone()),
            kind: EffectKind::User {
                pipeline_key,
                wgsl: spec.wgsl.clone(),
                source_path: None,
            },
            inputs,
            bounds,
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
    /// `"region"` | `"canvas"` — see [`Bounds`]; absent = inferred.
    #[serde(default)]
    pub bounds: Option<String>,
}

fn describe_def(def: &EffectDef) -> Value {
    serde_json::json!({
        "name": def.name,
        "kind": match def.kind {
            EffectKind::BuiltIn { .. } => "builtin",
            EffectKind::User { .. } => "user",
        },
        "bounds": def.bounds.as_str(),
        "inputs": def.inputs.iter().map(InputSlot::describe).collect::<Vec<_>>(),
    })
}

/// Descriptor inputs → slots, enforcing the v1 limit of one `image` input
/// per effect (the prelude exposes a single image texture binding).
fn parse_inputs(inputs: Vec<DescriptorInput>) -> Result<Vec<InputSlot>> {
    let slots = inputs
        .into_iter()
        .map(|i| i.into_slot())
        .collect::<Result<Vec<_>>>()?;
    let images = slots
        .iter()
        .filter(|s| matches!(s, InputSlot::Image { .. }))
        .count();
    if images > 1 {
        bail!("an effect may declare at most one `image` input (found {images})");
    }
    Ok(slots)
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
    let (inputs, declared_bounds) = if descriptor_path.exists() {
        let raw = fs::read_to_string(descriptor_path)
            .with_context(|| format!("reading {}", descriptor_path.display()))?;
        let parsed: DescriptorFile = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", descriptor_path.display()))?;
        (parse_inputs(parsed.inputs)?, parsed.bounds)
    } else {
        (Vec::new(), None)
    };
    let bounds = Bounds::resolve(declared_bounds.as_deref(), &wgsl)
        .with_context(|| format!("{}", descriptor_path.display()))?;
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
        bounds,
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
            bounds: Bounds::Region,
            inputs: vec![color("color", [1.0, 1.0, 1.0, 1.0])],
        },
        EffectDef {
            name: "hueCycle".into(),
            kind: EffectKind::BuiltIn { effect_id: HUE_CYCLE },
            bounds: Bounds::Region,
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
            bounds: Bounds::Region,
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
            // Reads the mask at a displaced uv: its edge leaves the region.
            bounds: Bounds::Canvas,
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
        EffectDef {
            name: "videoClip".into(),
            kind: EffectKind::BuiltIn { effect_id: VIDEO_CLIP },
            bounds: Bounds::Region,
            // C5 reference clip player: the `image` input (HAP clip or still)
            // cut by the mask, params_f[0] = gain.
            inputs: vec![
                InputSlot::Image {
                    name: "image".into(),
                },
                InputSlot::Scalar {
                    name: "gain".into(),
                    default: 1.0,
                    meta: ScalarMeta::range(0.0, 2.0, 0.01),
                },
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
    fn parses_image_param() {
        let r = EffectRegistry::new(None);
        let def = r.resolve_named("videoClip").unwrap();
        let b = EffectBinding::from_params(
            def.clone(),
            &json!({ "image": { "clip": "generated/a.mov" }, "gain": 0.5 }),
        )
        .unwrap();
        let img = b.image.unwrap();
        assert_eq!(img.clip, "generated/a.mov");
        assert_eq!(img.placement, ImagePlacement::Region); // C5 default
        assert_eq!(img.rate, 1.0);
        assert_eq!(b.scalars.len(), 1); // the image takes no uniform slot

        let b = EffectBinding::from_params(
            def.clone(),
            &json!({ "image": { "clip": "x.png", "placement": "canvas", "rate": -0.5 } }),
        )
        .unwrap();
        let img = b.image.unwrap();
        assert_eq!((img.placement, img.rate), (ImagePlacement::Canvas, -0.5));

        assert!(EffectBinding::from_params(def.clone(), &json!({})).unwrap().image.is_none());
        for bad in [
            json!({ "image": "x.mov" }),
            json!({ "image": { "clip": "" } }),
            json!({ "image": { "clip": "x.mov", "placment": "canvas" } }),
            json!({ "image": { "clip": "x.mov", "placement": "fill" } }),
        ] {
            assert!(EffectBinding::from_params(def.clone(), &bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn descriptor_allows_one_image_input() {
        let one: Vec<DescriptorInput> =
            serde_json::from_value(json!([{ "name": "image", "type": "image" }])).unwrap();
        let slots = parse_inputs(one).unwrap();
        assert!(matches!(&slots[0], InputSlot::Image { name } if name == "image"));
        assert_eq!(slots[0].describe()["type"], "image");
        let two: Vec<DescriptorInput> = serde_json::from_value(json!([
            { "name": "a", "type": "image" },
            { "name": "b", "type": "image" }
        ]))
        .unwrap();
        assert!(parse_inputs(two).is_err());
    }

    /// §2.4 output bounds inference: only a body that scales by its own
    /// `mask` argument (and reads the mask nowhere else) is clipped.
    #[test]
    fn bounds_inference_is_conservative() {
        let fx = |body: &str| format!("fn effect(uv: vec2<f32>, m: f32) -> vec4<f32> {{ {body} }}");
        assert_eq!(Bounds::infer(&fx("return vec4<f32>(uv, 0.0, 1.0) * m;")), Bounds::Region);
        // Mentioning sample_mask in a comment doesn't count.
        assert_eq!(Bounds::infer(&fx("// sample_mask(uv)\nreturn vec4<f32>(m);")), Bounds::Region);
        // Displaced mask read, direct atlas read, mask ignored, unparseable.
        assert_eq!(Bounds::infer(&fx("return vec4<f32>(sample_mask(uv + 0.1) * m);")), Bounds::Canvas);
        assert_eq!(
            Bounds::infer(&fx("return textureSampleLevel(mask_atlas, mask_sampler, uv, 0, 0.0) * m;")),
            Bounds::Canvas
        );
        assert_eq!(Bounds::infer(&fx("return vec4<f32>(uv, 0.0, 1.0);")), Bounds::Canvas);
        assert_eq!(Bounds::infer("fn effect("), Bounds::Canvas);
        // Declared wins either way; junk is an error.
        assert_eq!(Bounds::resolve(Some("canvas"), &fx("return vec4<f32>(m);")).unwrap(), Bounds::Canvas);
        assert_eq!(Bounds::resolve(Some("region"), &fx("return vec4<f32>(1.0);")).unwrap(), Bounds::Region);
        assert!(Bounds::resolve(Some("bbox"), "").is_err());
        // Built-ins: only wobble (displaced mask) shades the canvas.
        let r = EffectRegistry::new(None);
        for (name, b) in [("tint", Bounds::Region), ("videoClip", Bounds::Region), ("wobble", Bounds::Canvas)] {
            assert_eq!(r.resolve_named(name).unwrap().bounds, b, "{name}");
        }
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
