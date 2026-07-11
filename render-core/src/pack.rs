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

/// Per-layer spatial identity in uv space (§5.2). Sourced from the manifest
/// (`wzrd.layerpack` writes pixel-space `bbox`/`centroid`) or computed from
/// the mask bytes at load when the manifest omits them — computed here
/// because `mask_atlas` is dropped after GPU upload.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeom {
    /// Weighted mask centroid, uv space.
    pub centroid_uv: [f32; 2],
    /// (min_x, min_y, max_x, max_y), uv space, max-exclusive like the
    /// manifest's pixel bbox.
    pub bbox_uv: [f32; 4],
}

/// In-memory layer pack ready for upload to the GPU.
pub struct LoadedPack {
    /// Where the pack was loaded from — kept for hot-reload diagnostics.
    #[allow(dead_code)]
    pub pack_dir: PathBuf,
    pub manifest: PackManifest,
    /// One contiguous `width*height*N` R8 buffer, slice-major. Owns the
    /// mask bytes only until [`crate::gpu::GpuContext::new`] uploads them to
    /// the `Texture2DArray<R8>` — at that point the caller is expected to
    /// drop the Vec (the GPU has the canonical copy). See the post-init
    /// block in `main.rs::resumed`.
    pub mask_atlas: Vec<u8>,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub layer_count: u32,
    /// Slice-indexed spatial identity (parallel to `manifest.layers`).
    pub geoms: Vec<LayerGeom>,
    /// Map from semantic id → slice index.
    pub id_to_slice: HashMap<String, u32>,
    /// Map from tag → list of slice indices that carry that tag.
    pub tag_to_slices: HashMap<String, Vec<u32>>,
    /// Map from group id → list of slice indices.
    pub group_to_slices: HashMap<String, Vec<u32>>,
}

impl LoadedPack {
    pub fn load(pack_dir: &Path) -> Result<Self> {
        // `pack.json` is the canonical manifest name; we also accept the
        // legacy `scene.json` (Phase 1/2 wrote it under that name and shared
        // the filename with the runtime control file, which was a footgun —
        // see architecture review v1 issue #5).
        let manifest_path = pack_dir.join("pack.json");
        let raw = if manifest_path.exists() {
            fs::read_to_string(&manifest_path)
                .with_context(|| format!("reading layer-pack manifest {}", manifest_path.display()))?
        } else {
            let legacy = pack_dir.join("scene.json");
            if legacy.exists() {
                log::warn!(
                    "pack at {} uses legacy `scene.json` manifest name; \
                     rename to `pack.json` (this fallback will be removed in a future build)",
                    pack_dir.display()
                );
                fs::read_to_string(&legacy)
                    .with_context(|| format!("reading legacy layer-pack manifest {}", legacy.display()))?
            } else {
                bail!(
                    "no layer-pack manifest found in {} (expected pack.json)",
                    pack_dir.display()
                );
            }
        };
        let manifest: PackManifest = serde_json::from_str(&raw)
            .with_context(|| format!("parsing layer-pack manifest under {}", pack_dir.display()))?;

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
        let mut geoms: Vec<LayerGeom> = Vec::with_capacity(manifest.layers.len());
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

            geoms.push(match (layer.bbox, layer.centroid) {
                (Some(b), Some(c)) => LayerGeom {
                    centroid_uv: [c[0] / pw as f32, c[1] / ph as f32],
                    bbox_uv: [
                        b[0] as f32 / pw as f32,
                        b[1] as f32 / ph as f32,
                        b[2] as f32 / pw as f32,
                        b[3] as f32 / ph as f32,
                    ],
                },
                _ => geom_from_mask(&atlas[offset..offset + slice_pixels], pw, ph),
            });

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
            geoms,
            id_to_slice,
            tag_to_slices,
            group_to_slices,
        })
    }
}

/// Mirror of `wzrd.layerpack._bbox_and_centroid`: bbox on the thresholded
/// binary mask (>=128), centroid weighted by pixel value (stabler for soft
/// masks). Empty mask falls back to full-frame bbox + center, like Python.
fn geom_from_mask(mask: &[u8], width: u32, height: u32) -> LayerGeom {
    let (w, h) = (width as usize, height as usize);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let (mut wsum, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        for (x, &v) in row.iter().enumerate() {
            if v >= 128 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
            if v > 0 {
                let wgt = v as f64;
                wsum += wgt;
                cx += x as f64 * wgt;
                cy += y as f64 * wgt;
            }
        }
    }
    if x0 == usize::MAX || wsum == 0.0 {
        return LayerGeom {
            centroid_uv: [0.5, 0.5],
            bbox_uv: [0.0, 0.0, 1.0, 1.0],
        };
    }
    LayerGeom {
        centroid_uv: [
            (cx / wsum / width as f64) as f32,
            (cy / wsum / height as f64) as f32,
        ],
        bbox_uv: [
            x0 as f32 / width as f32,
            y0 as f32 / height as f32,
            x1 as f32 / width as f32,
            y1 as f32 / height as f32,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geom_from_mask_finds_bbox_and_centroid() {
        // 4x4, a single full-intensity pixel at (2, 1).
        let mut mask = vec![0u8; 16];
        mask[1 * 4 + 2] = 255;
        let g = geom_from_mask(&mask, 4, 4);
        assert_eq!(g.bbox_uv, [0.5, 0.25, 0.75, 0.5]);
        assert!((g.centroid_uv[0] - 0.5).abs() < 1e-6);
        assert!((g.centroid_uv[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn geom_from_mask_empty_falls_back_to_full_frame() {
        let g = geom_from_mask(&[0u8; 16], 4, 4);
        assert_eq!(g.centroid_uv, [0.5, 0.5]);
        assert_eq!(g.bbox_uv, [0.0, 0.0, 1.0, 1.0]);
    }
}
