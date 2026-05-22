//! `scene.json` parser — the canonical control contract (D13).
//!
//! The schema mirrors §3.4 of the architecture doc. v1's parser is strict:
//! unknown effect names fail loudly, malformed selectors fail loudly. The TS
//! DSL is *not* parsed here — humans transpile it before calling us.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use crate::pack::LoadedPack;

pub const SCENE_VERSION: u32 = 1;

/// Canonical scene as parsed from disk.
#[derive(Debug, Clone, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    /// Path to the layer pack directory. Resolved relative to the scene file.
    pub pack: String,
    /// Transport state. Used by `clock.*` drivers in Phase 3 (currently parsed
    /// but not consumed).
    #[serde(default)]
    #[allow(dead_code)]
    pub transport: TransportSpec,
    #[serde(default)]
    pub bindings: Vec<BindingSpec>,
    /// Post-fx stack. Parsed for forward compatibility; ignored in Phase 2.
    #[serde(default)]
    #[allow(dead_code)]
    pub post: Vec<BindingSpec>,
    /// 3×3 homography stored row-major; `None` = identity.
    #[serde(default, rename = "projectorCalibration")]
    pub projector_calibration: Option<[[f32; 3]; 3]>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TransportSpec {
    #[serde(default = "default_bpm")]
    #[allow(dead_code)]
    pub bpm: f32,
}

fn default_bpm() -> f32 {
    120.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct BindingSpec {
    pub id: String,
    pub select: SelectorSpec,
    pub effect: EffectRef,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SelectorSpec {
    All { all: bool },
    Id { id: String },
    Tag { tag: String },
    Group { group: String },
}

/// Effect reference. The string form points at a built-in or
/// project-local effect by name. The object form is an inline WGSL spec
/// (D15) — fields are parsed lazily by `effects::InlineEffectSpec` so we
/// don't pin its surface here.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EffectRef {
    Named(String),
    Inline(serde_json::Value),
}

impl SceneFile {
    /// Parse a scene JSON string. Doesn't touch disk — see [`load`].
    pub fn parse(raw: &str) -> Result<Self> {
        let scene: SceneFile = serde_json::from_str(raw).context("parsing scene.json")?;
        if scene.version != SCENE_VERSION {
            bail!(
                "scene.json version {} unsupported (this build expects {})",
                scene.version,
                SCENE_VERSION
            );
        }
        // Stable binding ids are non-negotiable — they're the diff key for
        // hot-reload (§4.2, open uncertainty #9).
        let mut seen = std::collections::HashSet::new();
        for b in &scene.bindings {
            if b.id.is_empty() {
                bail!("binding with empty id — every binding needs a stable id");
            }
            if !seen.insert(b.id.clone()) {
                bail!("duplicate binding id {:?}", b.id);
            }
        }
        Ok(scene)
    }

    pub fn load(scene_path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(scene_path)
            .with_context(|| format!("reading {}", scene_path.display()))?;
        Self::parse(&raw)
    }

    /// Absolute path to the layer pack directory, anchored on the scene file.
    pub fn pack_dir(&self, scene_path: &Path) -> PathBuf {
        let candidate = Path::new(&self.pack);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            scene_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(candidate)
        }
    }
}

/// Resolve a selector against a loaded pack into an ordered set of slice
/// indices. Order follows pack manifest order so blend behaviour is stable.
pub fn resolve_selector(selector: &SelectorSpec, pack: &LoadedPack) -> Result<Vec<u32>> {
    let mut slices: Vec<u32> = match selector {
        SelectorSpec::All { all } => {
            if !*all {
                bail!("selector with `all: false` is meaningless");
            }
            (0..pack.layer_count).collect()
        }
        SelectorSpec::Id { id } => vec![*pack
            .id_to_slice
            .get(id)
            .ok_or_else(|| anyhow!("selector references unknown layer id {:?}", id))?],
        SelectorSpec::Tag { tag } => pack
            .tag_to_slices
            .get(tag)
            .cloned()
            .ok_or_else(|| anyhow!("selector references unknown tag {:?}", tag))?,
        SelectorSpec::Group { group } => pack
            .group_to_slices
            .get(group)
            .cloned()
            .ok_or_else(|| anyhow!("selector references unknown group {:?}", group))?,
    };
    slices.sort_unstable();
    slices.dedup();
    Ok(slices)
}
