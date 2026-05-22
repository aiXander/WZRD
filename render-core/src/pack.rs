//! Layer-pack loader.
//!
//! Reads a pack directory written by `wzrd.layerpack` (Python) — the
//! offline ↔ runtime data contract (§4.1, D3). Layer masks land in a single
//! `Texture2DArray<R8>` (D4), keyed by stable semantic `id` so scene bindings
//! survive re-shoots and re-segmentations (D7).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

/// Hard cap from D4 — `Texture2DArray<R8>` slice budget.
pub const MAX_LAYERS: usize = 256;

/// Schema version this loader understands. Refuses unknown majors.
pub const PACK_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct PackManifest {
    pub version: u32,
    pub projector_resolution: [u32; 2],
    /// Path to the original capture (relative to pack dir). UI uses it for
    /// preview overlays; Phase 2 just round-trips it.
    #[serde(default)]
    #[allow(dead_code)]
    pub source_capture: Option<String>,
    /// Path to the darkened/aligned surface. UI/preview only in Phase 2.
    #[serde(default)]
    #[allow(dead_code)]
    pub surface: Option<String>,
    pub layers: Vec<LayerEntry>,
    #[serde(default)]
    pub groups: Vec<GroupEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct LayerEntry {
    pub id: String,
    pub mask: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub centroid: Option<[f32; 2]>,
    #[serde(default)]
    pub area_px: Option<u64>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub z: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroupEntry {
    pub id: String,
    pub members: Vec<String>,
}

/// In-memory layer pack ready for upload to the GPU.
pub struct LoadedPack {
    /// Where the pack was loaded from — kept for hot-reload diagnostics.
    #[allow(dead_code)]
    pub pack_dir: PathBuf,
    pub manifest: PackManifest,
    /// One contiguous `width*height*N` R8 buffer, slice-major.
    pub mask_atlas: Vec<u8>,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub layer_count: u32,
    /// Map from semantic id → slice index.
    pub id_to_slice: HashMap<String, u32>,
    /// Map from tag → list of slice indices that carry that tag.
    pub tag_to_slices: HashMap<String, Vec<u32>>,
    /// Map from group id → list of slice indices.
    pub group_to_slices: HashMap<String, Vec<u32>>,
}

impl LoadedPack {
    pub fn load(pack_dir: &Path) -> Result<Self> {
        let manifest_path = pack_dir.join("scene.json");
        let raw = fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading layer-pack manifest {}", manifest_path.display()))?;
        let manifest: PackManifest = serde_json::from_str(&raw)
            .with_context(|| format!("parsing layer-pack manifest {}", manifest_path.display()))?;

        if manifest.version != PACK_VERSION {
            bail!(
                "layer-pack version {} unsupported (this build expects {})",
                manifest.version,
                PACK_VERSION
            );
        }
        if manifest.layers.is_empty() {
            bail!("layer pack at {} has no layers", pack_dir.display());
        }
        if manifest.layers.len() > MAX_LAYERS {
            bail!(
                "layer pack has {} layers but the Texture2DArray cap is {} (D4)",
                manifest.layers.len(),
                MAX_LAYERS
            );
        }
        let [pw, ph] = manifest.projector_resolution;
        if pw == 0 || ph == 0 {
            bail!("projector_resolution must be non-zero, got {pw}x{ph}");
        }

        // Check duplicate ids up-front for a clean error rather than HashMap
        // silently overwriting.
        let mut seen: HashSet<&str> = HashSet::new();
        for layer in &manifest.layers {
            if !seen.insert(layer.id.as_str()) {
                bail!("duplicate layer id {:?} in pack manifest", layer.id);
            }
        }

        let layer_count = manifest.layers.len() as u32;
        let slice_pixels = (pw as usize) * (ph as usize);
        let mut atlas = vec![0u8; slice_pixels * manifest.layers.len()];
        let mut id_to_slice: HashMap<String, u32> = HashMap::new();
        let mut tag_to_slices: HashMap<String, Vec<u32>> = HashMap::new();

        for (idx, layer) in manifest.layers.iter().enumerate() {
            let mask_path = pack_dir.join(&layer.mask);
            let img = image::open(&mask_path)
                .with_context(|| format!("opening mask {}", mask_path.display()))?
                .into_luma8();
            if img.width() != pw || img.height() != ph {
                bail!(
                    "mask {} is {}x{} but projector_resolution is {}x{} — \
                     all masks must share the projector resolution (D4)",
                    mask_path.display(),
                    img.width(),
                    img.height(),
                    pw,
                    ph
                );
            }
            let offset = idx * slice_pixels;
            atlas[offset..offset + slice_pixels].copy_from_slice(img.as_raw());

            id_to_slice.insert(layer.id.clone(), idx as u32);
            for tag in &layer.tags {
                tag_to_slices.entry(tag.clone()).or_default().push(idx as u32);
            }
        }

        let mut group_to_slices: HashMap<String, Vec<u32>> = HashMap::new();
        for group in &manifest.groups {
            let mut slices = Vec::with_capacity(group.members.len());
            for member in &group.members {
                let slice = id_to_slice.get(member).copied().ok_or_else(|| {
                    anyhow!(
                        "group {:?} references unknown layer id {:?}",
                        group.id,
                        member
                    )
                })?;
                slices.push(slice);
            }
            group_to_slices.insert(group.id.clone(), slices);
        }

        Ok(Self {
            pack_dir: pack_dir.to_path_buf(),
            manifest,
            mask_atlas: atlas,
            atlas_width: pw,
            atlas_height: ph,
            layer_count,
            id_to_slice,
            tag_to_slices,
            group_to_slices,
        })
    }
}
