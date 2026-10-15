//! "Save as…" (studio plan D2, decision S21): fork the open show into a
//! sibling folder and open the fork. Saving is otherwise autosave — there
//! is no dirty/revert model — so this is the only explicit save.
//!
//! What a fork carries: every JSON file at the show's root (scene files,
//! `session.json`, `alignment.json`, `decks.json`, `ui_layout.json`,
//! `project.json`), `effects/`, `packs/`, `looks/` and the engine's design
//! draft (`.wzrd/design.scene.json`). What it doesn't: media (`generated/`,
//! `surfaces/`, …) — clips stay where they are and are **referenced**: a
//! relative `"clip"` value anywhere in the copied JSON (C5 bindings in the
//! scene, looks and the draft) is rewritten to the absolute path in the
//! original show, so both shows play the same files. `decks.json` slots are
//! absolute already (L6).
//!
//! The UI settles pending scene writes and saves the engine session first
//! (`state/actions.ts::saveProjectAs`), so the fork has what's on screen.
//! The copy goes to a temp folder that is renamed into place, so a failure
//! never leaves a half fork.

use std::path::{Path, PathBuf};

use serde_json::Value;
use tauri::{AppHandle, State};

use crate::{projects, AppState};

/// Folders copied whole (relative `clip`s inside their JSON rewritten).
const DIRS: &[&str] = &["effects", "packs", "looks"];
/// The engine's design-leg draft (render-engine.md §2.6).
const DRAFT: &str = ".wzrd/design.scene.json";

/// A folder name: non-empty, no path separators, not hidden.
fn check_name(name: &str) -> Result<&str, String> {
    let n = name.trim();
    if n.is_empty() || n.starts_with('.') || n.contains(['/', '\\', ':']) {
        return Err(format!("“{n}” isn't a usable folder name"));
    }
    Ok(n)
}

/// Rewrite every relative string under a `"clip"` key to `src_dir/<it>`.
fn absolutize_clips(v: &mut Value, src_dir: &Path) -> usize {
    match v {
        Value::Object(m) => {
            let mut n = 0;
            for (k, x) in m.iter_mut() {
                if k == "clip" {
                    if let Value::String(s) = x {
                        if !s.is_empty() && !Path::new(s.as_str()).is_absolute() {
                            *s = src_dir.join(s.as_str()).to_string_lossy().into_owned();
                            n += 1;
                        }
                        continue;
                    }
                }
                n += absolutize_clips(x, src_dir);
            }
            n
        }
        Value::Array(a) => a.iter_mut().map(|x| absolutize_clips(x, src_dir)).sum(),
        _ => 0,
    }
}

/// Copy one file; JSON goes through the clip rewrite (a file that doesn't
/// parse is copied byte for byte — it isn't ours to judge).
fn copy_file(from: &Path, to: &Path, src_dir: &Path) -> Result<(), String> {
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let is_json = from.extension().and_then(|e| e.to_str()) == Some("json");
    if is_json {
        if let Ok(mut v) = std::fs::read_to_string(from)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
        {
            if absolutize_clips(&mut v, src_dir) > 0 {
                let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
                return std::fs::write(to, text + "\n").map_err(|e| format!("{}: {e}", to.display()));
            }
        }
    }
    std::fs::copy(from, to).map(|_| ()).map_err(|e| format!("{}: {e}", from.display()))
}

fn copy_tree(from: &Path, to: &Path, src_dir: &Path) -> Result<(), String> {
    for entry in std::fs::read_dir(from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
        let p = entry.path();
        let name = entry.file_name();
        if p.is_dir() {
            copy_tree(&p, &to.join(&name), src_dir)?;
        } else if p.is_file() {
            copy_file(&p, &to.join(&name), src_dir)?;
        }
    }
    Ok(())
}

/// Fork `src_dir` into `dest` (must not exist). Returns the files copied.
pub fn fork(src_dir: &Path, dest: &Path) -> Result<usize, String> {
    if dest.exists() {
        return Err(format!("{} exists already — pick another name", dest.display()));
    }
    let parent = dest.parent().ok_or("no parent folder")?;
    let tmp = parent.join(format!(".{}.forking-{}", dest.file_name().unwrap_or_default().to_string_lossy(), std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let run = || -> Result<usize, String> {
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let mut n = 0;
        for entry in std::fs::read_dir(src_dir).map_err(|e| format!("{}: {e}", src_dir.display()))?.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("json") {
                copy_file(&p, &tmp.join(entry.file_name()), src_dir)?;
                n += 1;
            }
        }
        for d in DIRS {
            let from = src_dir.join(d);
            if from.is_dir() {
                copy_tree(&from, &tmp.join(d), src_dir)?;
                n += 1;
            }
        }
        let draft = src_dir.join(DRAFT);
        if draft.is_file() {
            copy_file(&draft, &tmp.join(DRAFT), src_dir)?;
            n += 1;
        }
        std::fs::rename(&tmp, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
        Ok(n)
    };
    run().inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&tmp);
    })
}

/// Fork the open show into `<its parent>/<name>` and open the fork (same
/// scene file name). The original is never written.
#[tauri::command]
pub fn project_save_as(app: AppHandle, state: State<AppState>, name: String) -> Result<Value, String> {
    let name = check_name(&name)?;
    let session = state.session();
    let src_dir = session.scene_dir();
    let scene_file = session.scene_path.file_name().ok_or("no scene file")?.to_owned();
    let dest: PathBuf = src_dir.parent().ok_or("the show has no parent folder")?.join(name);
    drop(session);
    let n = fork(&src_dir, &dest)?;
    log::info!("save as: {} → {} ({n} entries)", src_dir.display(), dest.display());
    projects::open(&app, &state, dest.join(scene_file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fork_copies_the_show_and_references_clips() {
        let root = std::env::temp_dir().join(format!("wzrd-fork-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let src = root.join("show");
        std::fs::create_dir_all(src.join("generated")).unwrap();
        std::fs::create_dir_all(src.join("effects/glow")).unwrap();
        std::fs::create_dir_all(src.join("looks")).unwrap();
        std::fs::create_dir_all(src.join(".wzrd")).unwrap();
        std::fs::write(src.join("generated/a.mov"), b"clip").unwrap();
        std::fs::write(src.join("effects/glow/shader.wgsl"), b"// fx").unwrap();
        let scene = json!({"bindings": [
            {"id": "c", "params": {"image": {"clip": "generated/a.mov"}}},
            {"id": "d", "params": {"image": {"clip": "/abs/b.mov"}}}]});
        std::fs::write(src.join("show.scene.json"), scene.to_string()).unwrap();
        std::fs::write(src.join("looks/drop.json"), scene.to_string()).unwrap();
        std::fs::write(src.join("decks.json"), r#"{"schema":1}"#).unwrap();
        std::fs::write(src.join(".wzrd/design.scene.json"), scene.to_string()).unwrap();
        std::fs::write(src.join(".wzrd/scene_agent_latest.json"), "{}").unwrap();

        let dest = root.join("fork");
        fork(&src, &dest).unwrap();
        let abs = src.join("generated/a.mov").to_string_lossy().into_owned();
        for f in ["show.scene.json", "looks/drop.json", ".wzrd/design.scene.json"] {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(dest.join(f)).unwrap()).unwrap();
            assert_eq!(v["bindings"][0]["params"]["image"]["clip"], json!(abs), "{f}");
            assert_eq!(v["bindings"][1]["params"]["image"]["clip"], json!("/abs/b.mov"), "{f}");
        }
        assert!(dest.join("effects/glow/shader.wgsl").is_file());
        assert!(dest.join("decks.json").is_file());
        assert!(!dest.join("generated").exists(), "media are referenced, not copied");
        assert!(!dest.join(".wzrd/scene_agent_latest.json").exists());
        // The original is untouched.
        let v: Value = serde_json::from_str(&std::fs::read_to_string(src.join("show.scene.json")).unwrap()).unwrap();
        assert_eq!(v["bindings"][0]["params"]["image"]["clip"], json!("generated/a.mov"));
        // A second fork onto the same name refuses; no temp folder is left.
        assert!(fork(&src, &dest).is_err());
        let leftovers = std::fs::read_dir(&root).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with('.')).count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(&root);
        assert!(check_name("../x").is_err() && check_name(" ").is_err() && check_name(".hid").is_err());
    }
}
