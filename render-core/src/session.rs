//! §5.3 session sidecar — operator-owned state next to the scene.
//!
//! `scene.json` is what the surface *does* (AI + human authored);
//! `session.json` is how *this venue, this night* is set (operator only).
//! The engine is the sole writer: the file is written on explicit
//! `session.save`, debounced after master/knob changes, and on
//! SIGTERM/SIGINT (the §5.11 power-blink snapshot). `scene.load` and the
//! file watcher never touch it, and no RPC on the authoring surface writes
//! it — so an AI scene rewrite can never drop projector calibration or the
//! operator's hand-tuned knobs.
//!
//! Contents: projector calibration (moved out of scene.json — calibration is
//! per-venue while a scene is per-artwork), the §5.4 masters, `ui.slider`
//! values, and the §5.5 per-binding param overrides.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::drivers::MastersSnapshot;

pub const SESSION_VERSION: u32 = 1;

/// On-disk shape of `session.json`. BTreeMaps keep the serialized output
/// stable so successive engine writes diff cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SessionFile {
    pub version: u32,
    /// 3×3 homography, row-major; `None` = identity / not yet calibrated.
    pub projector_calibration: Option<[[f32; 3]; 3]>,
    pub masters: Option<MastersSnapshot>,
    /// `ui.slider` values by slider name (the SliderBank snapshot).
    pub params: BTreeMap<String, f32>,
    /// §5.5 per-binding scalar overrides: binding id → param name → value.
    pub overrides: BTreeMap<String, BTreeMap<String, f32>>,
}

impl Default for SessionFile {
    fn default() -> Self {
        Self {
            version: SESSION_VERSION,
            projector_calibration: None,
            masters: None,
            params: BTreeMap::new(),
            overrides: BTreeMap::new(),
        }
    }
}

/// The sidecar lives next to the scene file. Deliberately *per directory*,
/// not per scene: calibration and masters describe the venue setup, and all
/// scenes played from one project directory share that physical reality.
pub fn session_path(scene_path: &Path) -> PathBuf {
    scene_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("session.json")
}

/// Load the sidecar if present. `Ok(None)` = no file (fresh venue).
pub fn load(path: &Path) -> Result<Option<SessionFile>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let file: SessionFile =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    if file.version != SESSION_VERSION {
        bail!(
            "session.json version {} unsupported (this build expects {})",
            file.version,
            SESSION_VERSION
        );
    }
    Ok(Some(file))
}

/// Write the sidecar atomically (temp file + rename) so a power blink
/// mid-write can't leave a torn file — the previous snapshot survives.
pub fn save(path: &Path, file: &SessionFile) -> Result<()> {
    let raw = serde_json::to_vec_pretty(file).context("serializing session")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &raw).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}

/// Millis since the UNIX epoch — the currency of the shared dirty stamp.
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Mark operator state dirty. The render thread debounces on this stamp in
/// `Core::poll_inbound` and persists ~1.5 s after the last touch, so a
/// slider drag becomes one write, not hundreds.
pub fn touch(dirty: &AtomicU64) {
    dirty.store(now_ms().max(1), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("wzrd-session-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.json");

        let mut file = SessionFile::default();
        file.projector_calibration = Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        file.masters = Some(MastersSnapshot {
            brightness: 0.8,
            speed: 1.0,
            saturation: 1.0,
            audio_listen: 0.5,
        });
        file.params.insert("flash_base".into(), 0.35);
        file.overrides
            .entry("wobble_demo".into())
            .or_default()
            .insert("amp".into(), 0.02);

        save(&path, &file).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded.version, SESSION_VERSION);
        assert!(loaded.projector_calibration.is_some());
        assert_eq!(loaded.params.get("flash_base"), Some(&0.35));
        assert_eq!(
            loaded.overrides.get("wobble_demo").and_then(|m| m.get("amp")),
            Some(&0.02)
        );
        assert!((loaded.masters.unwrap().audio_listen - 0.5).abs() < 1e-6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_none() {
        assert!(load(Path::new("/nonexistent/wzrd/session.json"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn rejects_unknown_version() {
        let dir = std::env::temp_dir().join(format!("wzrd-session-ver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.json");
        std::fs::write(&path, r#"{ "version": 99 }"#).unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
