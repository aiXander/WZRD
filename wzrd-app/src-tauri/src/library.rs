//! Read-only view of the media library (studio plan L1/L3, package A1).
//!
//! The hub is `library.db`'s single writer; the app opens it **read-only**
//! so Studio can browse (and play) while the hub is down. Every write —
//! index, label, move, delete — goes through the hub's `/api` (`hub.rs`).
//!
//! A folder listing merges the filesystem with the database: every media
//! file on disk is an item; the ones the hub has indexed carry their
//! render row (score, tags, tool, thumbnail…). Unindexed files show bare
//! until Studio's "index on open" (S3) catches up — the webview asks the
//! hub to index the folder and re-lists on its 2 s poll.
//!
//! Media files reach the webview as URLs, not bytes: Tauri's asset protocol
//! (scope: the library home, every root, the projects root — `allow_scope`)
//! and, for the browser bridge, the devport's `GET /file` (`file_allowed`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use tauri::{AppHandle, Manager};

use crate::hub;

/// Extensions Studio lists as items (anything else in a folder is ignored).
const MEDIA_EXT: &[&str] = &["mp4", "mov", "m4v", "webm", "png", "jpg", "jpeg", "webp", "gif"];

pub fn is_media(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .is_some_and(|e| MEDIA_EXT.contains(&e.as_str()))
}

/// Open `library.db` read-only. `None` when the hub has never created it.
fn open_db() -> Result<Option<Connection>, String> {
    let path = hub::db_path();
    if !path.is_file() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(
        &path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("opening {}: {e}", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(2)).ok();
    Ok(Some(conn))
}

fn json_col(s: Option<String>) -> Value {
    s.and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

/// One file + its render — the same keys as the hub's `library._item`
/// (`/api/library/list`), so the webview has one `LibraryItem` type whether
/// the listing came from the hub or from this read-only reader.
const ITEM_SELECT: &str = "
    SELECT f.path, f.name, f.dir, f.render_id, f.variant, f.fp, f.size, f.mtime, f.kind, f.codec,
           f.w, f.h, f.frames, f.fps, f.duration_s, f.missing,
           r.kind, r.tool, r.created, r.project, r.region, json_extract(r.record, '$.est_usd'),
           r.score, r.tags, r.agent_tags, r.notes, r.thumb
    FROM files f LEFT JOIN renders r ON r.id = f.render_id";

fn item_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let kind: Option<String> = row.get(8)?;
    let codec: Option<String> = row.get(9)?;
    let thumb: Option<String> = row.get(26)?;
    Ok(json!({
        "path": row.get::<_, String>(0)?,
        "name": row.get::<_, String>(1)?,
        "dir": row.get::<_, String>(2)?,
        "render_id": row.get::<_, String>(3)?,
        "variant": row.get::<_, Option<String>>(4)?,
        "fp": row.get::<_, Option<String>>(5)?,
        "size": row.get::<_, Option<i64>>(6)?,
        "mtime": row.get::<_, Option<f64>>(7)?,
        "playable": codec.as_deref() == Some("hap") || kind.as_deref() == Some("image"),
        "kind": kind,
        "codec": codec,
        "w": row.get::<_, Option<i64>>(10)?,
        "h": row.get::<_, Option<i64>>(11)?,
        "frames": row.get::<_, Option<i64>>(12)?,
        "fps": row.get::<_, Option<f64>>(13)?,
        "duration_s": row.get::<_, Option<f64>>(14)?,
        "missing": row.get::<_, Option<i64>>(15)?.unwrap_or(0) != 0,
        "wzrd": row.get::<_, Option<String>>(16)?.as_deref() == Some("wzrd"),
        "tool": row.get::<_, Option<String>>(17)?,
        "created": row.get::<_, Option<String>>(18)?,
        "project": row.get::<_, Option<String>>(19)?,
        "region": row.get::<_, Option<String>>(20)?,
        "est_usd": row.get::<_, Option<f64>>(21)?,
        "score": row.get::<_, Option<i64>>(22)?,
        "tags": json_col(row.get(23)?),
        "agent_tags": json_col(row.get(24)?),
        "notes": row.get::<_, Option<String>>(25)?,
        "thumb": thumb.filter(|t| Path::new(t).is_file()),
    }))
}

/// One folder, shaped like the hub's `/api/library/list` (`{dir, root,
/// dirs: [{name, path}], items, pending}`): indexed, present rows newest
/// first; media on disk the database doesn't know (or that changed since)
/// listed by path in `pending`. Never indexes — that's the hub's job.
pub fn list_dir(dir: &str) -> Result<Value, String> {
    let dir_path = PathBuf::from(dir);
    if !dir_path.is_absolute() {
        return Err(format!("library_dir: {dir:?} is not absolute"));
    }
    let mut dirs = Vec::new();
    let mut on_disk: HashMap<String, u64> = HashMap::new(); // path → size
    let entries = std::fs::read_dir(&dir_path).map_err(|e| format!("reading {dir}: {e}"))?;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        if meta.is_dir() {
            dirs.push(json!({ "name": name, "path": e.path().to_string_lossy() }));
        } else if is_media(&e.path()) {
            on_disk.insert(e.path().to_string_lossy().into_owned(), meta.len());
        }
    }
    dirs.sort_by_key(|d| d["name"].as_str().unwrap_or("").to_lowercase());

    let mut items = Vec::new();
    if let Some(conn) = open_db()? {
        let sql = format!("{ITEM_SELECT} WHERE f.dir = ?1 AND f.missing = 0 ORDER BY f.mtime DESC");
        let mut stmt = conn.prepare_cached(&sql).map_err(|e| e.to_string())?;
        let rows = stmt.query_map([dir], item_from_row).map_err(|e| e.to_string())?;
        for r in rows.flatten() {
            let path = r["path"].as_str().unwrap_or("").to_string();
            match on_disk.remove(&path) {
                Some(size) if r["size"].as_u64() == Some(size) => items.push(r),
                Some(_) => {
                    on_disk.insert(path, 0); // changed since indexed → pending
                }
                None => {} // gone from disk; the hub marks it missing on its next listing
            }
        }
    }
    let mut pending: Vec<String> = on_disk.into_keys().collect();
    pending.sort();
    let root = roots()
        .ok()
        .and_then(|rs| {
            rs.as_array()?
                .iter()
                .filter_map(|r| r.get("path")?.as_str())
                .filter(|r| dir_path.starts_with(r))
                .max_by_key(|r| r.len())
                .map(str::to_string)
        });
    Ok(json!({ "dir": dir, "root": root, "dirs": dirs, "items": items, "pending": pending }))
}

/// The library home plus every extra root (L1 `roots` table). Works with no
/// database yet (home only).
pub fn roots() -> Result<Value, String> {
    let home = hub::media_root().to_string_lossy().into_owned();
    let mut out = vec![json!({ "path": home, "home": true })];
    if let Some(conn) = open_db()? {
        let mut stmt = conn
            .prepare("SELECT path, home, added FROM roots ORDER BY added")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<String>>(2)?))
            })
            .map_err(|e| e.to_string())?;
        for (path, is_home, added) in rows.flatten() {
            if is_home != 0 || path == home {
                continue;
            }
            out.push(json!({
                "path": path,
                "home": false,
                "added": added,
                "exists": Path::new(&path).is_dir(),
            }));
        }
    }
    Ok(Value::Array(out))
}

/// One render, shaped like the hub's `/api/library/item`: the render row,
/// its L2 `record`, and every file that carries its id.
pub fn record(id: &str) -> Result<Value, String> {
    let Some(conn) = open_db()? else {
        return Err("no library database yet".into());
    };
    let mut stmt = conn
        .prepare_cached(
            "SELECT id, kind, tool, created, project, region, job, parent, record, score, tags,
                    agent_tags, notes, signals, purged, thumb FROM renders WHERE id = ?1",
        )
        .map_err(|e| e.to_string())?;
    let mut render = stmt
        .query_row([id], |r| {
            let thumb: Option<String> = r.get(15)?;
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "kind": r.get::<_, Option<String>>(1)?,
                "tool": r.get::<_, Option<String>>(2)?,
                "created": r.get::<_, Option<String>>(3)?,
                "project": r.get::<_, Option<String>>(4)?,
                "region": r.get::<_, Option<String>>(5)?,
                "job": r.get::<_, Option<String>>(6)?,
                "parent": r.get::<_, Option<String>>(7)?,
                "record": json_col(r.get(8)?),
                "score": r.get::<_, Option<i64>>(9)?,
                "tags": json_col(r.get(10)?),
                "agent_tags": json_col(r.get(11)?),
                "notes": r.get::<_, Option<String>>(12)?,
                "signals": json_col(r.get(13)?),
                "purged": r.get::<_, Option<i64>>(14)?.unwrap_or(0) != 0,
                "thumb": thumb.filter(|t| Path::new(t).is_file()),
            }))
        })
        .map_err(|e| format!("render {id}: {e}"))?;
    let mut fstmt = conn
        .prepare_cached(&format!("{ITEM_SELECT} WHERE f.render_id = ?1"))
        .map_err(|e| e.to_string())?;
    let files: Vec<Value> = fstmt
        .query_map([id], item_from_row)
        .map_err(|e| e.to_string())?
        .flatten()
        .collect();
    render["files"] = json!(files);
    Ok(render)
}

/// Every free tag in use (Xander's + the agent's) → use count, most used
/// first — the hub's `/api/library/tags` shape, for autocomplete.
pub fn tags() -> Result<Value, String> {
    let Some(conn) = open_db()? else { return Ok(json!({})) };
    let mut stmt = conn
        .prepare("SELECT tags, agent_tags FROM renders WHERE tags != '[]' OR agent_tags != '[]'")
        .map_err(|e| e.to_string())?;
    let mut counts: HashMap<String, u64> = HashMap::new();
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    for (a, b) in rows.flatten() {
        for t in [a, b] {
            if let Ok(Value::Array(v)) = serde_json::from_str::<Value>(&t) {
                for tag in v.into_iter().filter_map(|x| x.as_str().map(str::to_string)) {
                    *counts.entry(tag).or_default() += 1;
                }
            }
        }
    }
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(Value::Object(v.into_iter().map(|(k, n)| (k, json!(n))).collect()))
}

/// Files by render id or fingerprint — D3's Relink search, steps 1 and 2.
#[allow(dead_code)] // D3 Relink
pub fn find_files(id: Option<&str>, fp: Option<&str>) -> Result<Vec<Value>, String> {
    let Some(conn) = open_db()? else { return Ok(Vec::new()) };
    let (sql, key) = match (id, fp) {
        (Some(id), _) => (format!("{ITEM_SELECT} WHERE f.render_id = ?1 AND f.missing = 0"), id),
        (None, Some(fp)) => (format!("{ITEM_SELECT} WHERE f.fp = ?1 AND f.missing = 0"), fp),
        (None, None) => return Ok(Vec::new()),
    };
    let mut stmt = conn.prepare_cached(&sql).map_err(|e| e.to_string())?;
    let rows: Vec<Value> = stmt
        .query_map([key], item_from_row)
        .map_err(|e| e.to_string())?
        .flatten()
        .collect();
    Ok(rows)
}

/// The newest playable renders made for `project` (D1's incoming strip).
#[allow(dead_code)] // D1 incoming strip
pub fn recent_for_project(project: &str, limit: u32) -> Result<Vec<Value>, String> {
    let Some(conn) = open_db()? else { return Ok(Vec::new()) };
    let sql = format!(
        "{ITEM_SELECT} WHERE r.project = ?1 AND f.variant = 'hap' AND f.missing = 0
         ORDER BY f.mtime DESC LIMIT ?2"
    );
    let mut stmt = conn.prepare_cached(&sql).map_err(|e| e.to_string())?;
    let rows: Vec<Value> = stmt
        .query_map(rusqlite::params![project, limit], item_from_row)
        .map_err(|e| e.to_string())?
        .flatten()
        .collect();
    Ok(rows)
}

// ---------------------------------------------------------------------------
// File access scope (asset protocol + devport `/file`)
// ---------------------------------------------------------------------------

/// Directories the webview may load media from: the library home (incl. the
/// thumbnail cache), every extra root, and the projects root.
pub fn allowed_dirs(projects_root: Option<PathBuf>) -> Vec<PathBuf> {
    let mut dirs = vec![hub::media_root()];
    if let Ok(Value::Array(rs)) = roots() {
        dirs.extend(
            rs.iter()
                .filter_map(|r| r.get("path").and_then(Value::as_str))
                .map(PathBuf::from),
        );
    }
    dirs.extend(projects_root);
    dirs.sort();
    dirs.dedup();
    dirs
}

/// Grant the asset protocol every allowed dir (idempotent; re-run after a
/// root is added).
pub fn allow_scope(app: &AppHandle) {
    let projects_root = crate::projects::projects_root(app);
    let scope = app.asset_protocol_scope();
    for d in allowed_dirs(Some(projects_root)) {
        if let Err(e) = scope.allow_directory(&d, true) {
            log::warn!("asset scope {}: {e}", d.display());
        }
    }
}

/// Whether `path` (absolute, no `..`) lies inside an allowed dir.
pub fn file_allowed(app: &AppHandle, path: &Path) -> bool {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return false;
    }
    let projects_root = crate::projects::projects_root(app);
    allowed_dirs(Some(projects_root)).iter().any(|d| path.starts_with(d))
}

pub fn content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("mp4" | "m4v") => "video/mp4",
        Some("mov") => "video/quicktime",
        Some("webm") => "video/webm",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// Tauri commands (async: they touch disk + SQLite, never on the UI thread)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn library_dir(dir: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || list_dir(&dir))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn library_roots(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        allow_scope(&app);
        roots()
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn library_record(id: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || record(&id))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn library_tags() -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(tags)
        .await
        .map_err(|e| e.to_string())?
}

/// Mirror of `library_dir`'s args for the devport (sync, main thread).
pub fn devport_dispatch(cmd: &str, a: &Value, app: &AppHandle) -> Option<Result<Value, String>> {
    let s = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
    Some(match cmd {
        "library_dir" => match s("dir") {
            Some(d) => list_dir(&d),
            None => Err("library_dir: missing dir".into()),
        },
        "library_roots" => {
            allow_scope(app);
            roots()
        }
        "library_record" => match s("id") {
            Some(id) => record(&id),
            None => Err("library_record: missing id".into()),
        },
        "library_tags" => tags(),
        _ => return None,
    })
}
