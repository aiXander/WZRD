//! Layer-pack loader.
//!
//! Reads a pack directory written by `wzrd.layerpack` (Python) — the
//! offline ↔ runtime data contract (§4.1, D3). Layer masks land in a single
//! `Texture2DArray<R8>` (D4), keyed by stable semantic `id` so scene bindings
//! survive re-shoots and re-segmentations (D7).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

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

/// §5.13 identity sidecar (`<pack_dir>/identity.json`) — human-authored,
/// pack-adjacent metadata overlaid at load time. Keeps `pack.json`
/// machine-authored and "pack ids stable, period": groups and labels are a
/// property of the *surface* (not a performance, so not scene.json; not the
/// pack, so re-segmentation never eats them). Engine-written via
/// `identity.setGroups`; the same file later carries the §2.2 re-import
/// identity table.
pub const IDENTITY_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IdentityFile {
    pub version: u32,
    /// Group id → member layer ids. A group with the same id as a pack
    /// group *replaces* it in the merged view.
    pub groups: BTreeMap<String, Vec<String>>,
    /// Layer id → human label. Overrides the pack manifest's `label`.
    pub labels: BTreeMap<String, String>,
}

impl Default for IdentityFile {
    fn default() -> Self {
        Self {
            version: IDENTITY_VERSION,
            groups: BTreeMap::new(),
            labels: BTreeMap::new(),
        }
    }
}

pub fn identity_path(pack_dir: &Path) -> PathBuf {
    pack_dir.join("identity.json")
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
    /// Map from group id → list of slice indices. Built from the **merged**
    /// group view (§5.13: identity sidecar over pack manifest).
    pub group_to_slices: HashMap<String, Vec<u32>>,
    /// §5.13 identity sidecar contents (empty when no `identity.json`).
    pub identity: IdentityFile,
    /// Merged group view: pack manifest groups with identity groups laid
    /// over them (same id replaces, new ids append). What `pack.info` and
    /// selector resolution see.
    pub merged_groups: Vec<GroupEntry>,
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

        // Pack-manifest groups are machine-authored — strict.
        for group in &manifest.groups {
            for member in &group.members {
                if !id_to_slice.contains_key(member) {
                    bail!(
                        "group {:?} references unknown layer id {:?}",
                        group.id,
                        member
                    );
                }
            }
        }

        // §5.13 identity sidecar — human-authored, loaded leniently: a stale
        // entry (layer id gone after a re-segmentation) warns and is skipped
        // rather than refusing to boot.
        let identity = match load_identity(&identity_path(pack_dir)) {
            Ok(i) => i,
            Err(err) => {
                log::warn!(
                    "ignoring identity sidecar {}: {err:#}",
                    identity_path(pack_dir).display()
                );
                IdentityFile::default()
            }
        };

        let mut pack = Self {
            pack_dir: pack_dir.to_path_buf(),
            manifest,
            mask_atlas: atlas,
            atlas_width: pw,
            atlas_height: ph,
            layer_count,
            geoms,
            id_to_slice,
            tag_to_slices,
            group_to_slices: HashMap::new(),
            identity,
            merged_groups: Vec::new(),
        };
        pack.recompute_identity_merge();
        Ok(pack)
    }

    /// Rebuild `merged_groups` + `group_to_slices` from the pack manifest
    /// with the identity sidecar laid over it. Lenient: identity entries
    /// referencing unknown layer ids warn and drop (stale sidecar after a
    /// re-shoot must not take the engine down) — the strict validation for
    /// *new* writes lives in [`LoadedPack::apply_identity_delta`].
    pub fn recompute_identity_merge(&mut self) {
        let mut merged: Vec<GroupEntry> = self.manifest.groups.clone();
        for (gid, members) in &self.identity.groups {
            let valid: Vec<String> = members
                .iter()
                .filter(|m| {
                    let known = self.id_to_slice.contains_key(m.as_str());
                    if !known {
                        log::warn!(
                            "identity.json group {:?} references unknown layer id {:?} — skipped",
                            gid,
                            m
                        );
                    }
                    known
                })
                .cloned()
                .collect();
            match merged.iter_mut().find(|g| &g.id == gid) {
                Some(g) => g.members = valid,
                None => merged.push(GroupEntry {
                    id: gid.clone(),
                    members: valid,
                }),
            }
        }
        for id in self.identity.labels.keys() {
            if !self.id_to_slice.contains_key(id.as_str()) {
                log::warn!(
                    "identity.json labels unknown layer id {:?} — skipped",
                    id
                );
            }
        }

        let mut group_to_slices: HashMap<String, Vec<u32>> = HashMap::new();
        for group in &merged {
            let slices = group
                .members
                .iter()
                .filter_map(|m| self.id_to_slice.get(m).copied())
                .collect();
            group_to_slices.insert(group.id.clone(), slices);
        }
        self.merged_groups = merged;
        self.group_to_slices = group_to_slices;
    }

    /// Effective human label for the layer at `slice`: identity sidecar
    /// override first, pack manifest `label` second.
    pub fn merged_label(&self, slice: usize) -> Option<String> {
        let layer = self.manifest.layers.get(slice)?;
        self.identity
            .labels
            .get(&layer.id)
            .cloned()
            .or_else(|| layer.label.clone())
    }

    /// §5.13 `identity.setGroups` — merge a delta into the identity sidecar.
    /// Per-key semantics: `Some(members)` sets that group / label,
    /// `None` (JSON `null`) removes it; empty member lists also remove the
    /// group. Strict: unknown layer ids are prescriptive errors (this is the
    /// agent/UI self-correction surface). Returns a one-line change summary.
    pub fn apply_identity_delta(
        &mut self,
        groups: Option<BTreeMap<String, Option<Vec<String>>>>,
        labels: Option<BTreeMap<String, Option<String>>>,
    ) -> Result<String> {
        let known_ids = || -> Vec<&str> {
            self.manifest.layers.iter().map(|l| l.id.as_str()).collect()
        };
        let mut summary: Vec<String> = Vec::new();
        if let Some(groups) = groups {
            for (gid, members) in groups {
                match members {
                    Some(members) if !members.is_empty() => {
                        for m in &members {
                            if !self.id_to_slice.contains_key(m.as_str()) {
                                bail!(
                                    "unknown layer id {:?} in group {:?}; layer ids: {:?}",
                                    m,
                                    gid,
                                    known_ids()
                                );
                            }
                        }
                        summary.push(format!("group {gid}({})", members.len()));
                        self.identity.groups.insert(gid, members);
                    }
                    _ => {
                        if self.identity.groups.remove(&gid).is_some() {
                            summary.push(format!("-group {gid}"));
                        }
                    }
                }
            }
        }
        if let Some(labels) = labels {
            for (id, label) in labels {
                if !self.id_to_slice.contains_key(id.as_str()) {
                    bail!(
                        "unknown layer id {:?} in labels; layer ids: {:?}",
                        id,
                        known_ids()
                    );
                }
                match label {
                    Some(label) if !label.trim().is_empty() => {
                        summary.push(format!("label {id}={label:?}"));
                        self.identity.labels.insert(id, label);
                    }
                    _ => {
                        if self.identity.labels.remove(&id).is_some() {
                            summary.push(format!("-label {id}"));
                        }
                    }
                }
            }
        }
        self.recompute_identity_merge();
        Ok(if summary.is_empty() {
            "no-op".to_string()
        } else {
            summary.join(", ")
        })
    }

    /// Write the identity sidecar atomically (temp + rename, mirroring the
    /// session sidecar). The engine is the sole writer.
    pub fn save_identity(&self) -> Result<PathBuf> {
        let path = identity_path(&self.pack_dir);
        let raw = serde_json::to_vec_pretty(&self.identity).context("serializing identity")?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &raw).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path)
            .with_context(|| format!("renaming into {}", path.display()))?;
        Ok(path)
    }
}

fn load_identity(path: &Path) -> Result<IdentityFile> {
    if !path.exists() {
        return Ok(IdentityFile::default());
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let file: IdentityFile =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if file.version != IDENTITY_VERSION {
        bail!(
            "identity.json version {} unsupported (this build expects {})",
            file.version,
            IDENTITY_VERSION
        );
    }
    Ok(file)
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

    /// Minimal in-memory pack for identity-merge tests (no masks touched).
    fn test_pack() -> LoadedPack {
        let manifest = PackManifest {
            version: PACK_VERSION,
            projector_resolution: [4, 4],
            source_capture: None,
            surface: None,
            layers: vec![
                LayerEntry {
                    id: "a".into(),
                    mask: "a.png".into(),
                    label: Some("packlabel-a".into()),
                    tags: vec![],
                    bbox: None,
                    centroid: None,
                    area_px: None,
                    parent: None,
                    z: 0,
                },
                LayerEntry {
                    id: "b".into(),
                    mask: "b.png".into(),
                    label: None,
                    tags: vec![],
                    bbox: None,
                    centroid: None,
                    area_px: None,
                    parent: None,
                    z: 0,
                },
            ],
            groups: vec![GroupEntry {
                id: "packgroup".into(),
                members: vec!["a".into()],
            }],
        };
        let mut id_to_slice = HashMap::new();
        id_to_slice.insert("a".to_string(), 0u32);
        id_to_slice.insert("b".to_string(), 1u32);
        let mut pack = LoadedPack {
            pack_dir: std::env::temp_dir().join(format!("wzrd-idpack-{}", std::process::id())),
            manifest,
            mask_atlas: Vec::new(),
            atlas_width: 4,
            atlas_height: 4,
            layer_count: 2,
            geoms: Vec::new(),
            id_to_slice,
            tag_to_slices: HashMap::new(),
            group_to_slices: HashMap::new(),
            identity: IdentityFile::default(),
            merged_groups: Vec::new(),
        };
        pack.recompute_identity_merge();
        pack
    }

    #[test]
    fn identity_delta_merges_groups_and_labels() {
        let mut pack = test_pack();
        // Pack group visible before any identity overlay.
        assert_eq!(pack.group_to_slices["packgroup"], vec![0]);

        let mut groups = BTreeMap::new();
        groups.insert("canopy".to_string(), Some(vec!["a".into(), "b".into()]));
        // Identity group with the pack group's id replaces its membership.
        groups.insert("packgroup".to_string(), Some(vec!["b".into()]));
        let mut labels = BTreeMap::new();
        labels.insert("b".to_string(), Some("trunk".to_string()));
        pack.apply_identity_delta(Some(groups), Some(labels)).unwrap();

        assert_eq!(pack.group_to_slices["canopy"], vec![0, 1]);
        assert_eq!(pack.group_to_slices["packgroup"], vec![1]);
        assert_eq!(pack.merged_label(1), Some("trunk".to_string()));
        // Pack label survives where identity has no override.
        assert_eq!(pack.merged_label(0), Some("packlabel-a".to_string()));

        // Null removes the identity group; the pack group's own membership
        // returns.
        let mut rm = BTreeMap::new();
        rm.insert("packgroup".to_string(), None);
        rm.insert("canopy".to_string(), None);
        pack.apply_identity_delta(Some(rm), None).unwrap();
        assert_eq!(pack.group_to_slices["packgroup"], vec![0]);
        assert!(!pack.group_to_slices.contains_key("canopy"));
    }

    #[test]
    fn identity_delta_rejects_unknown_layer_ids() {
        let mut pack = test_pack();
        let mut groups = BTreeMap::new();
        groups.insert("g".to_string(), Some(vec!["nope".into()]));
        let err = pack.apply_identity_delta(Some(groups), None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("unknown layer id"), "{msg}");
        assert!(msg.contains("\"a\""), "prescriptive: lists known ids — {msg}");
    }

    #[test]
    fn identity_round_trips_through_disk() {
        let mut pack = test_pack();
        std::fs::create_dir_all(&pack.pack_dir).unwrap();
        let mut labels = BTreeMap::new();
        labels.insert("a".to_string(), Some("stam".to_string()));
        pack.apply_identity_delta(None, Some(labels)).unwrap();
        let path = pack.save_identity().unwrap();
        let loaded = load_identity(&path).unwrap();
        assert_eq!(loaded.labels.get("a"), Some(&"stam".to_string()));
        std::fs::remove_dir_all(&pack.pack_dir).ok();
    }
}
