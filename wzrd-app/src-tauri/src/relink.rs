//! "Relink missing" (studio plan D3): find clips whose file moved and point
//! the show at their new place. On demand only — never a background scan.
//!
//! What it looks at: every slot in `decks.json`, every `"clip"` in the
//! design scene the UI passes in (the scene on screen, not a stale file)
//! and in `looks/*.json`. A path that isn't on disk is "missing"; its
//! render id + fingerprint come from the slot, else from the library's
//! `files` row for the old path (the hub keeps rows of files that vanished).
//!
//! Search, in order, per missing file:
//! 1. another indexed file with the same render id and the same extension
//!    (the HAP stays a HAP), on disk;
//! 2. an indexed file with the same fingerprint, on disk;
//! 3. a walk of the library roots + the projects root: `stat` every file,
//!    fingerprint only those whose size matches a missing one.
//! A file the library never saw at its old place (no id, no fingerprint —
//! e.g. a clip typed into a scene) falls back to its **file name**: an
//! indexed file with that name, else the first one the walk meets.
//!
//! Writes: `decks.json` (through `decks::write_in`, the same lock as the
//! app's edits) and changed look files. The scene is **not** written here:
//! the reply's `scene` map (`{old clip string: new path}`) goes back to the
//! UI, which applies it through `sceneCommit` like every scene edit.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};
use tauri::{AppHandle, State};

use crate::{decks, library, projects, AppState};

/// Directories the walk never enters (besides dot-folders).
const SKIP_DIRS: &[&str] = &["node_modules", "target", "__pycache__"];

#[derive(Default)]
struct Missing {
    id: Option<String>,
    fp: Option<String>,
}

/// Visit every string under a `"clip"` key; `f` returns a replacement.
fn map_clips(v: &mut Value, f: &mut dyn FnMut(&str) -> Option<String>) -> usize {
    match v {
        Value::Object(m) => {
            let mut n = 0;
            for (k, x) in m.iter_mut() {
                if let (true, Value::String(s)) = (k == "clip", &mut *x) {
                    if let Some(new) = f(s) {
                        *s = new;
                        n += 1;
                    }
                } else {
                    n += map_clips(x, f);
                }
            }
            n
        }
        Value::Array(a) => a.iter_mut().map(|x| map_clips(x, f)).sum(),
        _ => 0,
    }
}

fn resolve(scene_dir: &Path, clip: &str) -> PathBuf {
    let p = Path::new(clip);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        scene_dir.join(p)
    }
}

fn ext(p: &str) -> String {
    Path::new(p)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn look_files(scene_dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(scene_dir.join("looks"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect()
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok())
}

/// Walk `root` for files whose size is wanted; fingerprint those and match.
fn walk(
    root: &Path,
    by_size: &HashMap<u64, Vec<(String, String)>>,
    by_name: &HashMap<String, Vec<String>>,
    found: &mut BTreeMap<String, String>,
    want: usize,
) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if found.len() >= want {
            return;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_ref()) {
                    stack.push(e.path());
                }
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            for old in by_name.get(name.as_ref()).into_iter().flatten() {
                if !found.contains_key(old) {
                    found.insert(old.clone(), e.path().to_string_lossy().into_owned());
                }
            }
            let Some(cands) = e.metadata().ok().and_then(|m| by_size.get(&m.len())) else { continue };
            if cands.iter().all(|(old, _)| found.contains_key(old)) {
                continue;
            }
            let Ok(fp) = decks::fingerprint(&e.path()) else { continue };
            for (old, want_fp) in cands {
                if *want_fp == fp && !found.contains_key(old) {
                    found.insert(old.clone(), e.path().to_string_lossy().into_owned());
                }
            }
        }
    }
}

pub fn run(scene_dir: &Path, roots: &[PathBuf], scene: Option<&str>) -> Result<Value, String> {
    let t0 = Instant::now();
    // 1. What's missing (absolute old path → what we know).
    let mut want: BTreeMap<String, Missing> = BTreeMap::new();
    let mut doc = decks::read_in(scene_dir)?["doc"].take();
    for deck in doc["decks"].as_array().into_iter().flatten() {
        for slot in deck["rows"].as_object().into_iter().flat_map(|r| r.values()).flat_map(|v| v.as_array()).flatten() {
            let Some(p) = slot["path"].as_str() else { continue };
            if !Path::new(p).exists() {
                let m = want.entry(p.to_string()).or_default();
                m.id = m.id.take().or_else(|| slot["id"].as_str().map(str::to_string));
                m.fp = m.fp.take().or_else(|| slot["fp"].as_str().map(str::to_string));
            }
        }
    }
    let mut scene_v = scene.and_then(|t| serde_json::from_str::<Value>(t).ok());
    let mut collect = |v: &mut Value| {
        map_clips(v, &mut |c| {
            let abs = resolve(scene_dir, c);
            if !abs.exists() {
                want.entry(abs.to_string_lossy().into_owned()).or_default();
            }
            None
        });
    };
    if let Some(v) = scene_v.as_mut() {
        collect(v);
    }
    let looks: Vec<(PathBuf, Value)> = look_files(scene_dir).into_iter().filter_map(|p| read_json(&p).map(|v| (p, v))).collect();
    for (_, v) in &looks {
        collect(&mut v.clone());
    }
    if want.is_empty() {
        return Ok(json!({ "missing": 0, "found": 0, "not_found": [], "scene": {}, "decks": 0, "looks": 0 }));
    }

    // 2. Library lookups (render id, then fingerprint).
    let mut found: BTreeMap<String, String> = BTreeMap::new();
    if let Some(conn) = library::open_db()? {
        for (old, m) in want.iter_mut() {
            if m.id.is_none() || m.fp.is_none() {
                if let Ok((id, fp)) = conn.query_row("SELECT render_id, fp FROM files WHERE path = ?1", [old.as_str()], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                }) {
                    m.id = m.id.take().or(Some(id));
                    m.fp = m.fp.take().or(Some(fp));
                }
            }
            let pick = |sql: &str, key: &str| -> Option<String> {
                let mut st = conn.prepare_cached(sql).ok()?;
                let paths: Vec<String> = st.query_map([key], |r| r.get::<_, String>(0)).ok()?.flatten().collect();
                paths.into_iter().find(|p| p != old && ext(p) == ext(old) && Path::new(p).is_file())
            };
            let hit = m
                .id
                .as_deref()
                .and_then(|id| pick("SELECT path FROM files WHERE render_id = ?1", id))
                .or_else(|| m.fp.as_deref().and_then(|fp| pick("SELECT path FROM files WHERE fp = ?1", fp)));
            let hit = hit.or_else(|| {
                if m.fp.is_some() {
                    return None; // the walk matches it exactly
                }
                let name = Path::new(old).file_name()?.to_string_lossy().into_owned();
                pick("SELECT path FROM files WHERE name = ?1", &name)
            });
            if let Some(p) = hit {
                found.insert(old.clone(), p);
            }
        }
    }

    // 3. Walk the roots by size → fingerprint.
    let mut by_size: HashMap<u64, Vec<(String, String)>> = HashMap::new();
    let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
    for (old, m) in &want {
        if found.contains_key(old) {
            continue;
        }
        match &m.fp {
            Some(fp) => {
                if let Some(size) = fp.split('-').next().and_then(|s| s.parse::<u64>().ok()) {
                    by_size.entry(size).or_default().push((old.clone(), fp.clone()));
                }
            }
            None => {
                if let Some(n) = Path::new(old).file_name() {
                    by_name.entry(n.to_string_lossy().into_owned()).or_default().push(old.clone());
                }
            }
        }
    }
    if !by_size.is_empty() || !by_name.is_empty() {
        for root in roots {
            walk(root, &by_size, &by_name, &mut found, want.len());
        }
    }

    // 4. Rewrite decks + looks; hand the scene map back.
    let mut deck_slots = 0;
    for deck in doc["decks"].as_array_mut().into_iter().flatten() {
        for slot in deck["rows"].as_object_mut().into_iter().flat_map(|r| r.values_mut()).filter_map(Value::as_array_mut).flatten() {
            let Some(new) = slot["path"].as_str().and_then(|p| found.get(p)).cloned() else { continue };
            if slot["fp"].is_null() {
                slot["fp"] = decks::fingerprint(Path::new(&new)).map(Value::from).unwrap_or(Value::Null);
            }
            slot["path"] = Value::from(new);
            deck_slots += 1;
        }
    }
    if deck_slots > 0 {
        decks::write_in(scene_dir, &doc, None)?;
    }
    let remap = |c: &str| found.get(resolve(scene_dir, c).to_string_lossy().as_ref()).cloned();
    let mut look_n = 0;
    for (path, mut v) in looks {
        if map_clips(&mut v, &mut |c| remap(c)) > 0 {
            let tmp = path.with_extension("json.relink.tmp");
            let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
            std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, &path)).map_err(|e| format!("{}: {e}", path.display()))?;
            look_n += 1;
        }
    }
    let mut scene_map = serde_json::Map::new();
    if let Some(v) = scene_v.as_mut() {
        map_clips(v, &mut |c| {
            if let Some(new) = remap(c) {
                scene_map.insert(c.to_string(), Value::from(new));
            }
            None
        });
    }
    let not_found: Vec<&String> = want.keys().filter(|k| !found.contains_key(*k)).collect();
    // Folders the files turned up in — the UI asks the hub to index them,
    // so the relinked slots get their thumbnails and labels back.
    let mut dirs: Vec<String> = found
        .values()
        .filter_map(|p| Path::new(p).parent().map(|d| d.to_string_lossy().into_owned()))
        .collect();
    dirs.sort();
    dirs.dedup();
    log::info!(
        "relink: {} missing, {} found ({} deck slots, {} looks, {} scene clips) in {} ms",
        want.len(),
        found.len(),
        deck_slots,
        look_n,
        scene_map.len(),
        t0.elapsed().as_millis()
    );
    Ok(json!({
        "missing": want.len(),
        "found": found.len(),
        "not_found": not_found,
        "scene": scene_map,
        "decks": deck_slots,
        "looks": look_n,
        "dirs": dirs,
        "ms": t0.elapsed().as_millis() as u64,
    }))
}

/// The walk's roots: the library home + every root, and the projects root.
pub fn search_roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = library::roots()
        .ok()
        .and_then(|r| r.as_array().cloned())
        .into_iter()
        .flatten()
        .filter_map(|r| r["path"].as_str().map(PathBuf::from))
        .collect();
    v.push(projects::projects_root(app));
    v.sort();
    v.dedup();
    // A root inside another is walked by its parent already.
    let all = v.clone();
    v.retain(|p| !all.iter().any(|q| q != p && p.starts_with(q)));
    v
}

#[tauri::command]
pub async fn relink_missing(app: AppHandle, state: State<'_, AppState>, scene: Option<String>) -> Result<Value, String> {
    let dir = state.session().scene_dir();
    let roots = search_roots(&app);
    tauri::async_runtime::spawn_blocking(move || run(&dir, &roots, scene.as_deref()))
        .await
        .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relinks_a_moved_folder_by_size_and_fingerprint() {
        let root = std::env::temp_dir().join(format!("wzrd-relink-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let show = root.join("show");
        let clips = root.join("lib/clips");
        std::fs::create_dir_all(&clips).unwrap();
        std::fs::create_dir_all(show.join("looks")).unwrap();
        std::fs::write(clips.join("a.mov"), b"aaaa-clip").unwrap();
        std::fs::write(clips.join("b.mov"), b"bb-clip").unwrap();
        let a = clips.join("a.mov").to_string_lossy().into_owned();
        let b = clips.join("b.mov").to_string_lossy().into_owned();
        let fa = decks::fingerprint(&clips.join("a.mov")).unwrap();
        let doc = json!({"schema": 1, "active": "Main", "decks": [{"name": "Main", "rows": {
            "_unassigned": [{"path": a, "id": null, "fp": fa}, null]}}]});
        std::fs::write(show.join("decks.json"), doc.to_string()).unwrap();
        let fb = decks::fingerprint(&clips.join("b.mov")).unwrap();
        // b is known only by fingerprint through a look (no library row in
        // this test) — so put it in a deck slot of a second deck too.
        let mut doc2 = doc.clone();
        doc2["decks"].as_array_mut().unwrap().push(json!({"name": "B", "rows": {"_unassigned": [{"path": b, "id": null, "fp": fb}]}}));
        std::fs::write(show.join("decks.json"), doc2.to_string()).unwrap();
        let look = json!({"bindings": [{"params": {"image": {"clip": a}}}]});
        std::fs::write(show.join("looks/l.json"), look.to_string()).unwrap();
        std::fs::write(clips.join("c.mov"), b"c-clip, never indexed").unwrap();
        let c = clips.join("c.mov").to_string_lossy().into_owned();
        let scene = json!({"bindings": [{"params": {"image": {"clip": b}}}, {"params": {"image": {"clip": "gone/x.mov"}}},
                                        {"params": {"image": {"clip": c}}}]});

        // Move the folder ("in Finder").
        std::fs::rename(&clips, root.join("lib/moved")).unwrap();
        let r = run(&show, &[root.join("lib")], Some(&scene.to_string())).unwrap();
        let na = root.join("lib/moved/a.mov").to_string_lossy().into_owned();
        let nb = root.join("lib/moved/b.mov").to_string_lossy().into_owned();
        assert_eq!(r["found"], json!(3), "{r}");
        assert_eq!(r["missing"], json!(4));
        assert_eq!(r["scene"][&c], json!(root.join("lib/moved/c.mov").to_string_lossy()), "by name");
        assert_eq!(r["decks"], json!(2));
        assert_eq!(r["looks"], json!(1));
        assert_eq!(r["scene"][&b], json!(nb));
        let d: Value = serde_json::from_str(&std::fs::read_to_string(show.join("decks.json")).unwrap()).unwrap();
        assert_eq!(d["decks"][0]["rows"]["_unassigned"][0]["path"], json!(na));
        let l: Value = serde_json::from_str(&std::fs::read_to_string(show.join("looks/l.json")).unwrap()).unwrap();
        assert_eq!(l["bindings"][0]["params"]["image"]["clip"], json!(na));
        // Nothing left to do on a second run (the unknown scene clip stays reported).
        let r2 = run(&show, &[root.join("lib")], Some(&scene.to_string().replace(&b, &nb).replace(&c, "/nowhere/c2.mov"))).unwrap();
        assert_eq!(r2["missing"], json!(2)); // gone/x.mov + c2.mov: no file by that name anywhere
        let _ = std::fs::remove_dir_all(&root);
    }
}
