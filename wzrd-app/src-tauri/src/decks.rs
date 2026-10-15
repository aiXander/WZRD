//! Clip decks (studio plan D1, contract L6): `<project>/decks.json`, which
//! replaces the gig's `clipgrid.json`.
//!
//! ```json
//! {"schema": 1, "active": "Intro",
//!  "decks": [{"name": "Intro",
//!             "rows": {"<region id>": [{"path": "/abs/clip.mov", "id": "r_…", "fp": "…"}, null],
//!                      "_unassigned": []}}]}
//! ```
//!
//! Slots reference library files **in place, by absolute path** — nothing
//! is copied; `id` (render id) and `fp` (content fingerprint) are what
//! Relink (D3) matches on when a path goes missing. Two writers: the app
//! (`decks_write`, compare-and-swap on the file's mtime so a concurrent
//! agent `add_to_deck` is never overwritten — the webview re-reads and
//! re-applies its edit) and the hub's `add_to_deck` MCP tool. Both write
//! atomically (temp + rename).
//!
//! Migration, once, when a show has no `decks.json`: `clipgrid.json`'s
//! filing + every `generated/*.mov` (row = the filing, else the legacy
//! sidecar's `region`) → one deck "Main" with absolute paths.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde_json::{json, Map, Value};
use tauri::State;

use crate::{library, AppState};

const DECKS_FILE: &str = "decks.json";
pub const UNASSIGNED: &str = "_unassigned";
const CHUNK: u64 = 1 << 20;

/// `<size>-<sha1 of the first + last MiB, 20 hex>` — byte-for-byte the
/// hub's `identity.fingerprint` (the second read starts at
/// `max(1 MiB, size − 1 MiB)`, so a file under 2 MiB is hashed whole).
pub fn fingerprint(path: &Path) -> std::io::Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let size = f.metadata()?.len();
    let mut h = sha1_smol::Sha1::new();
    let mut buf = Vec::with_capacity(CHUNK as usize);
    (&mut f).take(CHUNK).read_to_end(&mut buf)?;
    h.update(&buf);
    if size > CHUNK {
        f.seek(SeekFrom::Start(CHUNK.max(size - CHUNK)))?;
        buf.clear();
        f.take(CHUNK).read_to_end(&mut buf)?;
        h.update(&buf);
    }
    Ok(format!("{size}-{}", &h.digest().to_string()[..20]))
}

fn mtime_ms(p: &Path) -> Option<u64> {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// Serialises this app's writes: the CAS check + write must be one step
/// (two quick edits — a double-click is "show deck" + "rename" — race
/// otherwise), and each write gets its own temp file.
static WRITE: std::sync::Mutex<()> = std::sync::Mutex::new(());
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn write_atomic(target: &Path, doc: &Value) -> Result<(), String> {
    let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = target.with_extension(format!("json.{}-{seq}.tmp", std::process::id()));
    let text = serde_json::to_string_pretty(doc).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, text).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, target).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("replacing {}: {e}", target.display())
    })
}

/// A slot for `path`: its render id + fingerprint from library.db when
/// indexed, else the fingerprint computed here (id unknown).
pub fn slot_for(path: &Path) -> Value {
    let p = path.to_string_lossy().into_owned();
    if let Ok(Some(row)) = library::file_row(&p) {
        return json!({ "path": p, "id": row.0, "fp": row.1 });
    }
    json!({ "path": p, "id": Value::Null, "fp": fingerprint(path).ok() })
}

fn empty_doc() -> Value {
    json!({ "schema": 1, "active": "Main", "decks": [{ "name": "Main", "rows": { UNASSIGNED: [] } }] })
}

/// The one-time migration: `clipgrid.json` + `generated/*.mov` → "Main".
fn migrate(scene_dir: &Path) -> Value {
    let gen = scene_dir.join("generated");
    let filing = read_json(&scene_dir.join("clipgrid.json"))
        .and_then(|v| v.get("rows").cloned())
        .unwrap_or(Value::Null);
    let mut clips: Vec<(u64, PathBuf)> = std::fs::read_dir(&gen)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("mov"))
        .map(|p| (mtime_ms(&p).unwrap_or(0), p))
        .collect();
    clips.sort_by(|a, b| b.0.cmp(&a.0)); // newest first, as the old grid
    let mut rows: Map<String, Value> = Map::new();
    rows.insert(UNASSIGNED.into(), json!([]));
    for (_, p) in clips {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        let rel = format!("generated/{name}");
        let region = match filing.get(&rel) {
            Some(Value::String(r)) => Some(r.clone()),
            Some(Value::Null) => None,
            _ => read_json(&gen.join(format!("{name}.meta.json")))
                .and_then(|s| s.get("region")?.as_str().map(str::to_string)),
        };
        let row = region.unwrap_or_else(|| UNASSIGNED.into());
        let slots = rows.entry(row).or_insert_with(|| json!([]));
        if let Some(a) = slots.as_array_mut() {
            a.push(slot_for(&p));
        }
    }
    json!({ "schema": 1, "active": "Main", "decks": [{ "name": "Main", "rows": rows }] })
}

/// Paths of every slot in `doc` whose file doesn't exist.
fn missing_paths(doc: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for deck in doc["decks"].as_array().into_iter().flatten() {
        for slots in deck["rows"].as_object().into_iter().flat_map(|m| m.values()) {
            for s in slots.as_array().into_iter().flatten() {
                if let Some(p) = s.get("path").and_then(Value::as_str) {
                    if !Path::new(p).is_file() && !out.iter().any(|o| o == p) {
                        out.push(p.to_string());
                    }
                }
            }
        }
    }
    out
}

/// The show's decks (migrating once if needed) + the file's mtime (the CAS
/// token for `decks_write`) + which slot paths are missing on disk.
pub fn read_in(scene_dir: &Path) -> Result<Value, String> {
    let path = scene_dir.join(DECKS_FILE);
    let doc = match read_json(&path) {
        Some(d) if d.get("decks").is_some_and(Value::is_array) => d,
        Some(_) => return Err(format!("{} is not a decks file (schema 1)", path.display())),
        None => {
            let d = if scene_dir.join("generated").is_dir() || scene_dir.join("clipgrid.json").is_file() {
                migrate(scene_dir)
            } else {
                empty_doc()
            };
            write_atomic(&path, &d)?;
            log::info!("decks: created {} (migrated the clip grid)", path.display());
            d
        }
    };
    Ok(json!({
        "doc": doc,
        "mtime_ms": mtime_ms(&path),
        "missing": missing_paths(&doc),
    }))
}

/// Write the whole doc if the file still has `base_mtime_ms` (else
/// `Err("changed")`: someone — an agent's `add_to_deck` — wrote since the
/// caller read; re-read and re-apply).
pub fn write_in(scene_dir: &Path, doc: &Value, base_mtime_ms: Option<u64>) -> Result<Value, String> {
    validate(doc)?;
    let path = scene_dir.join(DECKS_FILE);
    let _one_writer = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    if base_mtime_ms.is_some() && mtime_ms(&path) != base_mtime_ms {
        return Err("changed".into());
    }
    write_atomic(&path, doc)?;
    Ok(json!({ "mtime_ms": mtime_ms(&path), "missing": missing_paths(doc) }))
}

fn validate(doc: &Value) -> Result<(), String> {
    let decks = doc["decks"].as_array().ok_or("decks: `decks` must be an array")?;
    if decks.is_empty() {
        return Err("decks: a show keeps at least one deck".into());
    }
    let mut names = std::collections::HashSet::new();
    for d in decks {
        let n = d["name"].as_str().filter(|n| !n.trim().is_empty()).ok_or("decks: every deck needs a name")?;
        if !names.insert(n) {
            return Err(format!("decks: two decks are called {n:?}"));
        }
        let rows = d["rows"].as_object().ok_or("decks: `rows` must be an object")?;
        for slots in rows.values() {
            for s in slots.as_array().ok_or("decks: a row must be an array")? {
                if !s.is_null() && !s.get("path").and_then(Value::as_str).is_some_and(|p| Path::new(p).is_absolute()) {
                    return Err("decks: slot paths must be absolute".into());
                }
            }
        }
    }
    let active = doc["active"].as_str().unwrap_or("");
    if !names.contains(active) {
        return Err(format!("decks: active deck {active:?} doesn't exist"));
    }
    Ok(())
}

#[tauri::command]
pub async fn decks_read(state: State<'_, AppState>) -> Result<Value, String> {
    let dir = state.session().scene_dir();
    tauri::async_runtime::spawn_blocking(move || read_in(&dir))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn decks_write(state: State<'_, AppState>, doc: Value, base_mtime_ms: Option<u64>) -> Result<Value, String> {
    let dir = state.session().scene_dir();
    tauri::async_runtime::spawn_blocking(move || write_in(&dir, &doc, base_mtime_ms))
        .await
        .map_err(|e| e.to_string())?
}

/// A deck slot for each path (render id + fingerprint) — what the webview
/// stores when it adds library files to a deck.
#[tauri::command]
pub async fn deck_slots(paths: Vec<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        Value::Array(paths.iter().map(|p| slot_for(Path::new(p))).collect())
    })
    .await
    .map_err(|e| e.to_string())
}

/// D1's incoming strip: the newest playable renders made for `project`.
#[tauri::command]
pub async fn library_incoming(project: String, limit: Option<u32>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        library::recent_for_project(&project, limit.unwrap_or(24)).map(Value::Array)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Devport twins (browser bridge).
pub fn devport_dispatch(cmd: &str, a: &Value, state: &AppState) -> Option<Result<Value, String>> {
    let dir = || state.session().scene_dir();
    Some(match cmd {
        "decks_read" => read_in(&dir()),
        "decks_write" => write_in(&dir(), &a["doc"], a.get("baseMtimeMs").and_then(Value::as_u64)),
        "deck_slots" => Ok(Value::Array(
            a["paths"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|p| slot_for(Path::new(p)))
                .collect(),
        )),
        "library_incoming" => library::recent_for_project(
            a["project"].as_str().unwrap_or_default(),
            a.get("limit").and_then(Value::as_u64).unwrap_or(24) as u32,
        )
        .map(Value::Array),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_matches_the_hub_formula() {
        let dir = std::env::temp_dir().join(format!("wzrd-fp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let small = dir.join("a.bin");
        std::fs::write(&small, b"hello").unwrap();
        // sha1("hello") = aaf4c61d…
        assert_eq!(fingerprint(&small).unwrap(), "5-aaf4c61ddcc5e8a2dabe");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validate_rejects_relative_paths_and_dup_names() {
        let ok = json!({"schema":1,"active":"A","decks":[{"name":"A","rows":{"_unassigned":[null,{"path":"/x.mov"}]}}]});
        assert!(validate(&ok).is_ok());
        let rel = json!({"schema":1,"active":"A","decks":[{"name":"A","rows":{"r":[{"path":"x.mov"}]}}]});
        assert!(validate(&rel).is_err());
        let dup = json!({"schema":1,"active":"A","decks":[{"name":"A","rows":{}},{"name":"A","rows":{}}]});
        assert!(validate(&dup).is_err());
    }
}
