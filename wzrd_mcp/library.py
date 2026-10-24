"""The media library: `~/WZRD/media` + extra roots, and `library.db` (Studio L1, L3).

Every file WZRD makes lands in the library home (`media_root()`, override
`WZRD_MEDIA_ROOT`): in the Studio *target folder* (L5) when one is set, else
`inbox/`. Extra roots (`~/Visuals`, a project's `generated/`) are indexed in
place and never moved. Identity travels in the file (`identity.py`); this
module keeps what's known about each file and render in SQLite:

- `renders` — one row per render id: the L2 creation record plus the labels
  (`score`, `tags`, `agent_tags`, `notes`), implicit `signals` (S6) and
  `purged`. A foreign file (no tag) gets `fp_<fingerprint>` as its id.
- `files` — one row per path, "last seen here": render id, fingerprint, probe,
  `missing`. A path is never an identity.
- `jobs` — render-tagged tool calls (`jobs.py`), and the daily spend.
- `roots`, `selection` (L5), `settings` (the daily cap). The home and the
  projects root (`project.projects_root()`, show-setup G9) are always roots.
- `trash` — soft-deleted files (show-setup T1): `delete` moves a file into a
  hidden trash folder; `trash_flush` unlinks it after `TRASH_TTL_S` (a hub
  thread), `trash_restore` puts it back.

The hub is the single writer; the app opens the file read-only. Other
writers (a `--stdio` server) share it through WAL + a busy timeout.
Migrations run on open; the version is `PRAGMA user_version`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from . import identity
from ._log import _emit

SCHEMA_VERSION = 3
INBOX = "inbox"
WZRD_DIR = ".wzrd"
TRASH_DIR = "trash"  # <media_root>/.wzrd/trash/<id>/<name>
TRASH_BESIDE = ".wzrd-trash"  # <parent>/.wzrd-trash/<id>/<name> on another device
TRASH_TTL_S = 1800.0  # a deleted file is flushed for good after 30 min (show-setup G6)

# ---------------------------------------------------------------------------
# Paths (L1)
# ---------------------------------------------------------------------------


def media_root() -> Path:
    env = os.getenv("WZRD_MEDIA_ROOT")
    root = Path(env).expanduser() if env else Path.home() / "WZRD" / "media"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def wzrd_dir() -> Path:
    d = media_root() / WZRD_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return wzrd_dir() / "library.db"


def thumbs_dir() -> Path:
    d = wzrd_dir() / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def intermediates_dir() -> Path:
    """Scratch outputs of a render that aren't library items (TextureFlow's previews)."""
    d = wzrd_dir() / "intermediates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def inbox() -> Path:
    d = media_root() / INBOX
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(render_id: str) -> Path:
    return thumbs_dir() / f"{re.sub(r'[^A-Za-z0-9_-]', '_', render_id)}.jpg"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_MIGRATIONS: list[str] = [
    # v1 — Studio H1
    """
    CREATE TABLE renders (
        id          TEXT PRIMARY KEY,      -- r_<ulid> (tagged render) | fp_<fingerprint> (foreign file)
        kind        TEXT NOT NULL,         -- 'wzrd' | 'foreign'
        tool        TEXT,
        created     TEXT,                  -- ISO 8601 UTC
        project     TEXT,
        region      TEXT,
        job         TEXT,
        parent      TEXT,                  -- render id
        record      TEXT NOT NULL,         -- the L2 record (JSON); '{}' for a foreign file
        score       INTEGER,               -- 0-9, NULL = unrated (Xander's only)
        tags        TEXT NOT NULL DEFAULT '[]',
        agent_tags  TEXT NOT NULL DEFAULT '[]',
        notes       TEXT,
        signals     TEXT NOT NULL DEFAULT '{}',
        purged      INTEGER NOT NULL DEFAULT 0,
        thumb       TEXT,                  -- absolute path in .wzrd/thumbs, NULL = none yet
        updated     REAL NOT NULL          -- epoch seconds of the last change
    );
    CREATE INDEX renders_tool ON renders(tool);
    CREATE INDEX renders_project ON renders(project);
    CREATE INDEX renders_score ON renders(score);
    CREATE TABLE files (
        path        TEXT PRIMARY KEY,
        dir         TEXT NOT NULL,
        name        TEXT NOT NULL,
        render_id   TEXT NOT NULL,
        variant     TEXT NOT NULL DEFAULT 'master',   -- 'master' | 'hap'
        fp          TEXT NOT NULL,
        size        INTEGER NOT NULL,
        mtime       REAL NOT NULL,
        kind        TEXT,                  -- 'image' | 'video'
        codec       TEXT,                  -- ffprobe codec_name ('hap', 'h264', 'dxv', 'png', …)
        w INTEGER, h INTEGER, frames INTEGER, fps REAL, duration_s REAL,
        last_seen   REAL NOT NULL,
        missing     INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX files_dir ON files(dir);
    CREATE INDEX files_render ON files(render_id);
    CREATE INDEX files_fp ON files(fp);
    CREATE TABLE jobs (
        id          TEXT PRIMARY KEY,      -- j_<ulid>
        tool        TEXT NOT NULL,
        args        TEXT NOT NULL,         -- JSON, the caller's non-null args
        by          TEXT NOT NULL,         -- 'agent' | 'app'
        session     TEXT,
        pid         INTEGER NOT NULL,      -- the process running it (dead + no `ended` = lost)
        started     REAL NOT NULL,         -- epoch seconds
        ended       REAL,
        ok          INTEGER,               -- 1 / 0, NULL while running
        error       TEXT,
        outputs     TEXT NOT NULL DEFAULT '[]',   -- JSON list of absolute paths
        est_usd     REAL NOT NULL DEFAULT 0,      -- pre-estimate while running, then the recorded cost
        duration_s  REAL,
        cancelled   INTEGER NOT NULL DEFAULT 0,
        dest        TEXT,                  -- folder the outputs land in
        size_key    TEXT,                  -- the size args, for the ETA median
        remote      TEXT,                  -- remote call id (Modal FunctionCall / FAL request)
        project     TEXT
    );
    CREATE INDEX jobs_started ON jobs(started);
    CREATE TABLE roots (
        path        TEXT PRIMARY KEY,
        home        INTEGER NOT NULL DEFAULT 0,
        added       TEXT NOT NULL
    );
    CREATE TABLE selection (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        data        TEXT NOT NULL,         -- L5 JSON {styles, items, target, project}
        updated     REAL NOT NULL
    );
    CREATE TABLE settings (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL          -- JSON
    );
    """,
    # v2 — show setup T1: soft delete
    """
    CREATE TABLE trash (
        id          TEXT PRIMARY KEY,      -- t_<ulid>
        path        TEXT NOT NULL,         -- where it lived (and returns to)
        trash_path  TEXT NOT NULL,         -- where it sits now
        render_id   TEXT,
        size        INTEGER,
        deleted     REAL NOT NULL,         -- epoch seconds
        flushed     REAL                   -- epoch seconds, NULL while restorable
    );
    CREATE INDEX trash_deleted ON trash(deleted);
    """,
    # v3 — TextureFlow render queue: the job's backend, its place while queued, live progress
    """
    ALTER TABLE jobs ADD COLUMN backend TEXT;      -- texture_flow: 'modal' | 'local'
    ALTER TABLE jobs ADD COLUMN lane_rank REAL;    -- set while queued (hub lane / desktop queue), NULL once running
    ALTER TABLE jobs ADD COLUMN progress TEXT;     -- JSON, the last progress block (tf_queue)
    """,
]

_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[Path] = None
_lock = threading.RLock()


def _open() -> sqlite3.Connection:
    global _conn, _conn_path
    path = db_path()
    if _conn is not None and _conn_path == path:
        return _conn
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, script in enumerate(_MIGRATIONS[version:], start=version + 1):
        try:
            conn.executescript(f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version={i};\nCOMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    home = str(media_root())
    conn.execute("INSERT OR IGNORE INTO roots(path, home, added) VALUES (?, 1, ?)", (home, _now_iso()))
    conn.execute("INSERT OR IGNORE INTO roots(path, home, added) VALUES (?, 0, ?)",
                 (_projects_root(), _now_iso()))
    _conn, _conn_path = conn, path
    return conn


def _projects_root() -> str:
    from . import project

    return str(project.projects_root())


@contextmanager
def tx():
    """One write transaction (serialised in-process, WAL across processes)."""
    with _lock:
        conn = _open()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with _lock:
        return _open().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    rows = query(sql, params)
    return rows[0] if rows else None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _loads(text: Optional[str], default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Settings, roots, selection (L5)
# ---------------------------------------------------------------------------
def get_setting(key: str, default: Any = None) -> Any:
    row = query_one("SELECT value FROM settings WHERE key=?", (key,))
    return _loads(row["value"], default) if row else default


def set_setting(key: str, value: Any) -> None:
    with tx() as c:
        c.execute("INSERT INTO settings(key, value) VALUES (?, ?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, _dumps(value)))


def ensure_projects_root() -> None:
    """The projects root is always a root (G9) — re-added if it went missing
    (e.g. `WZRD_PROJECTS_ROOT` changed since the database was opened)."""
    with tx() as c:
        c.execute("INSERT OR IGNORE INTO roots(path, home, added) VALUES (?, 0, ?)",
                  (_projects_root(), _now_iso()))


def roots() -> list[dict]:
    if not query_one("SELECT 1 FROM roots WHERE path=?", (_projects_root(),)):
        ensure_projects_root()
    return [{"path": r["path"], "home": bool(r["home"]), "added": r["added"],
             "exists": Path(r["path"]).is_dir()}
            for r in query("SELECT * FROM roots ORDER BY home DESC, path")]


def add_root(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"{p} is not a folder")
    with tx() as c:
        c.execute("INSERT OR IGNORE INTO roots(path, home, added) VALUES (?, 0, ?)", (str(p), _now_iso()))
    return {"path": str(p), "roots": roots()}


def remove_root(path: str) -> dict:
    """Forget a root (its rows stay — the records are kept)."""
    p = str(Path(path).expanduser().resolve())
    if p == _projects_root():
        raise ValueError("the projects root is always a library root (every show's folders)")
    with tx() as c:
        c.execute("DELETE FROM roots WHERE path=? AND home=0", (p,))
    return {"roots": roots()}


def root_of(path: str | Path) -> Optional[str]:
    p = Path(path).resolve()
    for r in sorted(roots(), key=lambda r: -len(r["path"])):
        if p == Path(r["path"]) or p.is_relative_to(r["path"]):
            return r["path"]
    return None


EMPTY_SELECTION = {"styles": [], "items": [], "target": None, "project": None}


def _exists(p: Any) -> bool:
    return isinstance(p, str) and (p.startswith(("http://", "https://")) or Path(p).exists())


def selection() -> dict:
    """The selection, minus tray/items entries whose file is gone (deleted, or a
    temp folder cleaned up): a dead style path would otherwise refuse every render."""
    row = query_one("SELECT data FROM selection WHERE id=1")
    data = {**EMPTY_SELECTION, **(_loads(row["data"], {}) if row else {})}
    for k in ("styles", "items"):
        data[k] = [p for p in data[k] or [] if _exists(p)]
    return data


def set_selection(patch: dict) -> dict:
    """Merge *patch* (any of styles/items/target/project) into the selection."""
    unknown = set(patch) - set(EMPTY_SELECTION)
    if unknown:
        raise ValueError(f"unknown selection keys {sorted(unknown)}")
    data = {**selection(), **patch}
    with tx() as c:
        c.execute("INSERT INTO selection(id, data, updated) VALUES (1, ?, ?) "
                  "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                  (_dumps(data), time.time()))
    return data


# ---------------------------------------------------------------------------
# Where new renders land (S1, S14)
# ---------------------------------------------------------------------------
#: A generator call's `dest=` (None/"library" | "project" | an absolute folder).
current_dest: ContextVar[Optional[str]] = ContextVar("wzrd_dest", default=None)


def target_dir() -> Path:
    """The Studio target folder (L5) if it exists, else the library's inbox/."""
    t = selection().get("target")
    if t and Path(t).is_dir():
        return Path(t)
    return inbox()


_PROJECT_SUB = re.compile(r"^project/([a-z0-9][a-z0-9_-]{0,47})/?$")


def check_dest(dest: Optional[str]) -> None:
    if dest in (None, "", "library", "project"):
        return
    if dest.startswith("project/"):
        if not _PROJECT_SUB.match(dest):
            raise ValueError(f'dest "project/<folder>" takes one lowercase folder name, e.g. '
                             f'"project/segmentation" (got {dest!r})')
        return
    if not Path(dest).expanduser().is_absolute():
        raise ValueError(f'dest must be "library", "project", "project/<folder>" or an absolute '
                         f'folder path (got {dest!r})')


def dest_dir() -> Path:
    """Where this call's generator outputs go: `dest="project"` → the project's
    generated/; `"project/<sub>"` → `<project>/<sub>/`; an absolute folder →
    there; else the target folder / inbox/."""
    dest = current_dest.get()
    sub = _PROJECT_SUB.match(dest) if dest else None
    if dest == "project" or sub:
        from . import project

        return project.kind_dir(sub.group(1) if sub else project.GENERATED)
    if dest and dest != "library":
        d = Path(dest).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    return target_dir()


def output_path(tool: str, ext: str, name: str = "") -> str:
    """Reserve a fresh file path for a generator output (see `dest_dir`)."""
    from . import project

    if ext and not ext.startswith("."):
        ext = f".{ext}"
    stem = project.slugify(name) if name else f"{tool}_{project.stamp()}"
    return str(project.unique(dest_dir(), stem, ext))


# ---------------------------------------------------------------------------
# Rows → API dicts
# ---------------------------------------------------------------------------
def _render_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"], "kind": r["kind"], "tool": r["tool"], "created": r["created"],
        "project": r["project"], "region": r["region"], "job": r["job"], "parent": r["parent"],
        "score": r["score"], "tags": _loads(r["tags"], []), "agent_tags": _loads(r["agent_tags"], []),
        "notes": r["notes"], "signals": _loads(r["signals"], {}), "purged": bool(r["purged"]),
        "thumb": r["thumb"] if r["thumb"] and Path(r["thumb"]).is_file() else None,
    }


_ITEM_SQL = """
SELECT f.*, r.kind AS rkind, r.tool, r.created, r.project, r.region, r.score, r.tags, r.agent_tags,
       r.notes, r.thumb, r.record
FROM files f LEFT JOIN renders r ON r.id = f.render_id
"""


def _tf_brief(record: dict) -> Optional[dict]:
    """`record.textureflow` in brief, for the grid's draft badge (render-queue plan, "U asks")."""
    tf = record.get("textureflow")
    if not isinstance(tf, dict):
        return None
    draft = tf.get("draft") if isinstance(tf.get("draft"), dict) else {}
    return {"stage": tf.get("stage"), "request_id": tf.get("request_id"), "backend": tf.get("backend"),
            "expires_at": draft.get("expires_at")}


def _item(row: sqlite3.Row) -> dict:
    record = _loads(row["record"], {})
    return {
        "path": row["path"], "name": row["name"], "dir": row["dir"],
        "render_id": row["render_id"], "variant": row["variant"], "fp": row["fp"],
        "size": row["size"], "mtime": row["mtime"], "kind": row["kind"], "codec": row["codec"],
        "w": row["w"], "h": row["h"], "frames": row["frames"], "fps": row["fps"],
        "duration_s": row["duration_s"], "missing": bool(row["missing"]),
        "playable": row["codec"] == "hap" or row["kind"] == "image",
        "wzrd": row["rkind"] == "wzrd", "tool": row["tool"], "created": row["created"],
        "project": row["project"], "region": row["region"],
        "est_usd": record.get("est_usd"),
        "tf": _tf_brief(record),
        "score": row["score"], "tags": _loads(row["tags"], []),
        "agent_tags": _loads(row["agent_tags"], []), "notes": row["notes"],
        "thumb": row["thumb"] if row["thumb"] and Path(row["thumb"]).is_file() else None,
    }


# ---------------------------------------------------------------------------
# Indexing (S3: on open, plus "Index this folder")
# ---------------------------------------------------------------------------
def _legacy_record(path: Path) -> Optional[dict]:
    """A pre-Studio `<file>.meta.json` sidecar (gig C2), absorbed on first index."""
    side = Path(f"{path}.meta.json")
    try:
        meta = json.loads(side.read_text())
    except (OSError, ValueError):
        return None
    return {
        "schema": identity.TAG_SCHEMA, "variant": "hap" if path.suffix.lower() == ".mov" else "master",
        "tool": meta.get("tool"), "args": meta.get("args") or {},
        "inputs": [{"name": Path(i).name, "path": i, "fp": None} for i in meta.get("inputs") or []],
        "parent": None, "parent_path": meta.get("parent"), "created": meta.get("created"),
        "est_usd": meta.get("est_usd"), "duration_s": None, "job": None,
        "project": path.parent.parent.name if path.parent.name == "generated" else None,
        "canvas": None, "region": meta.get("region"), "source_box": meta.get("source_box"),
        "legacy_sidecar": str(side),
    }


def upsert_render(c: sqlite3.Connection, record: dict, kind: str = "wzrd") -> None:
    """Insert a render row from its L2 record; an existing row keeps its labels.
    A HAP variant's record never replaces its master's (a DXV's HAP leaves the
    DXV's foreign row as it is)."""
    if record.get("variant") == "hap" and c.execute("SELECT 1 FROM renders WHERE id=?", (record["id"],)).fetchone():
        c.execute("UPDATE renders SET purged=0 WHERE id=?", (record["id"],))
        return
    if record.get("variant") == "hap" and str(record["id"]).startswith("fp_"):
        kind = "foreign"
    c.execute(
        """INSERT INTO renders(id, kind, tool, created, project, region, job, parent, record, updated)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             record = CASE WHEN renders.kind = 'foreign' OR renders.record = '{}' THEN excluded.record
                           ELSE renders.record END,
             kind = CASE WHEN excluded.kind = 'wzrd' THEN 'wzrd' ELSE renders.kind END,
             tool = COALESCE(renders.tool, excluded.tool),
             created = COALESCE(renders.created, excluded.created),
             project = COALESCE(renders.project, excluded.project),
             region = COALESCE(renders.region, excluded.region),
             job = COALESCE(renders.job, excluded.job),
             parent = COALESCE(renders.parent, excluded.parent),
             purged = 0""",
        (record["id"], kind, record.get("tool"), record.get("created"), record.get("project"),
         record.get("region"), record.get("job"), record.get("parent"),
         _dumps(record) if kind == "wzrd" else "{}", time.time()),
    )


def index_file(path: str | Path, force: bool = False, record: Optional[dict] = None) -> Optional[dict]:
    """Bring one file's rows up to date and return its item. Cheap when unchanged.

    Identity, in order: the embedded tag (or *record*, given at creation) →
    a known fingerprint (a moved file without a tag) → a legacy sidecar →
    a foreign file (`fp_<fingerprint>`). A row whose path is gone but whose
    fingerprint shows up here is that file, moved: the row follows it.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file() or identity.media_kind(p) is None:
        return None
    st = p.stat()
    if not force and record is None:
        row = query_one("SELECT size, mtime, missing FROM files WHERE path=?", (str(p),))
        if row and row["size"] == st.st_size and abs(row["mtime"] - st.st_mtime) < 1e-3 and not row["missing"]:
            return item(str(p))

    fp = identity.fingerprint(p)
    probe = identity.probe_file(p)
    tag = record or identity.read_tag(p, probe)
    kind = "wzrd"
    if tag:
        rid, variant = tag["id"], tag.get("variant") or "master"
    else:
        known = query_one("SELECT render_id, variant FROM files WHERE fp=? ORDER BY missing, last_seen DESC", (fp,))
        legacy = None if known else _legacy_record(p)
        if known:
            rid, variant = known["render_id"], known["variant"]
        elif legacy:
            rid, variant = identity.new_id(), legacy["variant"]
            tag = {**legacy, "id": rid}
        else:
            rid, variant, kind = f"fp_{fp}", "master", "foreign"

    with tx() as c:
        if tag:
            upsert_render(c, tag)
        elif kind == "foreign" or not c.execute("SELECT 1 FROM renders WHERE id=?", (rid,)).fetchone():
            upsert_render(c, {"id": rid}, kind="foreign")
        else:  # a known render's file showed up again
            c.execute("UPDATE renders SET purged=0 WHERE id=?", (rid,))
        # A row of the same file left behind at an old path → it moved here.
        for old in c.execute("SELECT path FROM files WHERE fp=? AND path<>?", (fp, str(p))).fetchall():
            if not Path(old["path"]).exists():
                c.execute("DELETE FROM files WHERE path=?", (old["path"],))
        c.execute(
            """INSERT INTO files(path, dir, name, render_id, variant, fp, size, mtime, kind, codec,
                                 w, h, frames, fps, duration_s, last_seen, missing)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
               ON CONFLICT(path) DO UPDATE SET render_id=excluded.render_id, variant=excluded.variant,
                 fp=excluded.fp, size=excluded.size, mtime=excluded.mtime, kind=excluded.kind,
                 codec=excluded.codec, w=excluded.w, h=excluded.h, frames=excluded.frames,
                 fps=excluded.fps, duration_s=excluded.duration_s, last_seen=excluded.last_seen,
                 missing=0""",
            (str(p), str(p.parent), p.name, rid, variant, fp, st.st_size, st.st_mtime,
             probe.get("kind"), probe.get("codec"), probe.get("w"), probe.get("h"), probe.get("frames"),
             probe.get("fps"), probe.get("duration_s"), time.time()),
        )
    ensure_thumb(rid, p, probe.get("duration_s"))
    return item(str(p))


def ensure_thumb(render_id: str, path: Path, duration_s: Optional[float] = None) -> Optional[str]:
    """The render's cached thumbnail, made from *path* if it has none yet."""
    row = query_one("SELECT thumb FROM renders WHERE id=?", (render_id,))
    if row and row["thumb"] and Path(row["thumb"]).is_file():
        return row["thumb"]
    dest = thumb_path(render_id)
    if not dest.is_file() and not identity.make_thumb(path, dest, duration_s):
        return None
    with tx() as c:
        c.execute("UPDATE renders SET thumb=? WHERE id=?", (str(dest), render_id))
    return str(dest)


def item(path: str) -> Optional[dict]:
    row = query_one(_ITEM_SQL + " WHERE f.path=?", (path,))
    return _item(row) if row else None


# Background indexer: new or changed files found by a listing (or a deep
# "Index this folder") are probed off the request path.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wzrd-index")
_pending: set[str] = set()
_pending_lock = threading.Lock()


def _index_bg(path: str) -> None:
    try:
        index_file(path)
    except Exception as exc:  # noqa: BLE001
        _emit(f"[library] index {path} failed: {exc}")
    finally:
        with _pending_lock:
            _pending.discard(path)


def queue_index(paths: Iterable[str]) -> int:
    n = 0
    with _pending_lock:
        for p in paths:
            if p not in _pending:
                _pending.add(p)
                _pool.submit(_index_bg, p)
                n += 1
    return n


def pending_count() -> int:
    with _pending_lock:
        return len(_pending)


def _scan(folder: Path) -> tuple[list[os.DirEntry], list[os.DirEntry]]:
    dirs, files = [], []
    for e in os.scandir(folder):
        if e.name.startswith("."):
            continue
        if e.is_dir():
            dirs.append(e)
        elif e.is_file() and identity.media_kind(e.name):
            files.append(e)
    return dirs, files


def list_folder(folder: str) -> dict:
    """One folder of the library: subfolders + items. Indexes on open (S3):
    unknown or changed files are queued and show as `pending` until the next
    listing; files that vanished get `missing = 1` (their records stay)."""
    d = Path(folder).expanduser().resolve()
    if not d.is_dir():
        raise ValueError(f"{d} is not a folder")
    dirs, files = _scan(d)
    known = {r["path"]: r for r in query("SELECT path, size, mtime, missing FROM files WHERE dir=?", (str(d),))}
    stale = []
    for e in files:
        st = e.stat()
        r = known.get(e.path)
        if not r or r["missing"] or r["size"] != st.st_size or abs(r["mtime"] - st.st_mtime) > 1e-3:
            stale.append(e.path)
    queue_index(stale)
    present = {e.path for e in files}
    gone = [p for p, r in known.items() if p not in present and not r["missing"]]
    if gone:
        with tx() as c:
            c.executemany("UPDATE files SET missing=1 WHERE path=?", [(p,) for p in gone])
    rows = query(_ITEM_SQL + " WHERE f.dir=? AND f.missing=0 ORDER BY f.mtime DESC", (str(d),))
    stale_set = set(stale)
    items = [_item(r) for r in rows if r["path"] not in stale_set]
    return {
        "dir": str(d), "root": root_of(d),
        "dirs": [{"name": e.name, "path": e.path} for e in sorted(dirs, key=lambda e: e.name.lower())],
        "items": items,
        "pending": sorted(stale_set),
    }


def index_tree(folder: str, deep: bool = True) -> dict:
    """"Index this folder": queue every media file below *folder* (or just in it)."""
    d = Path(folder).expanduser().resolve()
    if not d.is_dir():
        raise ValueError(f"{d} is not a folder")
    paths: list[str] = []
    walker = os.walk(d) if deep else [(str(d), [], [e.name for e in os.scandir(d) if e.is_file()])]
    for top, subdirs, names in walker:
        subdirs[:] = [s for s in subdirs if not s.startswith(".")]
        paths += [os.path.join(top, n) for n in names
                  if not n.startswith(".") and identity.media_kind(n)]
    return {"dir": str(d), "files": len(paths), "queued": queue_index(paths)}


# ---------------------------------------------------------------------------
# Records + labels
# ---------------------------------------------------------------------------
def resolve_id(ref: str) -> str:
    """A render id from an id or a file path (indexing the file if needed)."""
    if ref.startswith(("r_", "fp_")):
        if not query_one("SELECT 1 FROM renders WHERE id=?", (ref,)):
            raise ValueError(f"no render {ref!r}")
        return ref
    it = index_file(ref)
    if not it:
        raise ValueError(f"{ref!r} is neither a render id nor a media file")
    return it["render_id"]


def render(ref: str) -> dict:
    """One render in full: record, labels and every file it has (incl. missing)."""
    rid = resolve_id(ref)
    r = query_one("SELECT * FROM renders WHERE id=?", (rid,))
    files = [_item(f) for f in query(_ITEM_SQL + " WHERE f.render_id=? ORDER BY f.missing, f.variant", (rid,))]
    return {**_render_dict(r), "record": _loads(r["record"], {}), "files": files}


def label(refs: list[str], *, score: Any = ..., tags: Optional[list[str]] = None,
          add_tags: Optional[list[str]] = None, remove_tags: Optional[list[str]] = None,
          notes: Any = ..., agent: bool = False) -> list[dict]:
    """Set labels on renders. The app's tags go to `tags`, an agent's to
    `agent_tags`; an agent never sets the score (S18). `...` = leave as is."""
    if agent and score is not ...:
        raise ValueError("an agent never sets the score")
    if score is not ... and score is not None and not (isinstance(score, int) and 0 <= score <= 9):
        raise ValueError("score must be an integer 0-9 (or null to clear)")
    col = "agent_tags" if agent else "tags"
    out = []
    ids = [resolve_id(r) for r in refs]
    with tx() as c:
        for rid in ids:
            row = c.execute(f"SELECT {col}, notes FROM renders WHERE id=?", (rid,)).fetchone()
            current = _loads(row[col], [])
            if tags is not None:
                current = []
                add = tags
            else:
                add = add_tags or []
            for t in add:
                t = t.strip()
                if t and t not in current:
                    current.append(t)
            current = [t for t in current if t not in set(remove_tags or [])]
            sets, params = [f"{col}=?", "updated=?"], [_dumps(current), time.time()]
            if score is not ...:
                sets.append("score=?")
                params.append(score)
            if notes is not ...:
                sets.append("notes=?")
                params.append(notes)
            c.execute(f"UPDATE renders SET {', '.join(sets)} WHERE id=?", (*params, rid))
    for rid in ids:
        r = query_one("SELECT * FROM renders WHERE id=?", (rid,))
        out.append(_render_dict(r))
    return out


def add_signal(ref: str, kind: str, seconds: Optional[float] = None) -> dict:
    """S6 implicit signals: `deck_add` (count), `played` (count + seconds live)."""
    if kind not in ("deck_add", "played"):
        raise ValueError("signal kind must be 'deck_add' or 'played'")
    rid = resolve_id(ref)
    with tx() as c:
        sig = _loads(c.execute("SELECT signals FROM renders WHERE id=?", (rid,)).fetchone()[0], {})
        sig[kind] = int(sig.get(kind, 0)) + 1
        if kind == "played" and seconds:
            sig["live_s"] = round(float(sig.get("live_s", 0)) + float(seconds), 1)
        c.execute("UPDATE renders SET signals=?, updated=? WHERE id=?", (_dumps(sig), time.time(), rid))
    return {"id": rid, "signals": sig}


def all_tags() -> dict:
    """Every tag in use with its count, for autocomplete."""
    counts: dict[str, int] = {}
    for r in query("SELECT tags, agent_tags FROM renders WHERE tags<>'[]' OR agent_tags<>'[]'"):
        for t in _loads(r["tags"], []) + _loads(r["agent_tags"], []):
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Move / rename / delete (S7: delete = media gone, record kept; never refused)
# ---------------------------------------------------------------------------
def _siblings(path: Path) -> list[Path]:
    """The other files of the same render in the same folder with the same stem
    (a render's mp4 + its HAP .mov travel together)."""
    row = query_one("SELECT render_id FROM files WHERE path=?", (str(path),))
    if not row:
        return []
    return [Path(r["path"]) for r in query(
        "SELECT path FROM files WHERE render_id=? AND dir=? AND path<>? AND missing=0",
        (row["render_id"], str(path.parent), str(path)))
        if Path(r["path"]).stem == path.stem and Path(r["path"]).is_file()]


def _relocate(src: Path, dst: Path) -> None:
    if dst.exists():
        raise ValueError(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    with tx() as c:
        c.execute("DELETE FROM files WHERE path=?", (str(dst),))
        c.execute("UPDATE files SET path=?, dir=?, name=?, last_seen=? WHERE path=?",
                  (str(dst), str(dst.parent), dst.name, time.time(), str(src)))
    index_file(dst)


def _expand(paths: list[str], siblings: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        src = Path(p).expanduser().resolve()
        if not src.is_file():
            raise ValueError(f"{src} is not a file")
        index_file(src)
        for f in [src, *(_siblings(src) if siblings else [])]:
            if f not in out:
                out.append(f)
    return out


def move(paths: list[str], dest_dir: str, siblings: bool = True) -> list[dict]:
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for src in _expand(paths, siblings):
        dst = dest / src.name
        if dst == src:
            continue
        _relocate(src, dst)
        moved.append({"from": str(src), "to": str(dst)})
    return moved


def rename(path: str, name: str, siblings: bool = True) -> list[dict]:
    """New file name (stem; the extension is kept). Siblings get the same stem."""
    src = Path(path).expanduser().resolve()
    stem = Path(name).stem if Path(name).suffix.lower() == src.suffix.lower() else name
    if not stem or "/" in stem or stem.startswith("."):
        raise ValueError(f"bad name {name!r}")
    out = []
    for f in _expand([str(src)], siblings):
        dst = f.with_name(stem + f.suffix)
        if dst != f:
            _relocate(f, dst)
            out.append({"from": str(f), "to": str(dst)})
    return out


def _trash_home(src: Path, tid: str) -> Path:
    """Where a deleted file waits (T1): the media root's `.wzrd/trash/` when on
    the same device (a rename), else a hidden `.wzrd-trash/` beside it."""
    try:
        same = src.stat().st_dev == wzrd_dir().stat().st_dev
    except OSError:
        same = False
    base = wzrd_dir() / TRASH_DIR if same else src.parent / TRASH_BESIDE
    return base / tid / src.name


def delete(paths: list[str], siblings: bool = True) -> dict:
    """Soft-delete media files (T1): each moves to the trash, restorable for
    `TRASH_TTL_S`, then flushed. Records stay; a render is `purged` once a
    flush leaves it no file. Never refused — a show slot using it warns on its own."""
    deleted, trashed, ids = [], [], set()
    for f in _expand(paths, siblings):
        row = query_one("SELECT render_id FROM files WHERE path=?", (str(f),))
        tid = identity.new_id("t_")
        dst = _trash_home(f, tid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = f.stat().st_size
        shutil.move(str(f), str(dst))
        deleted.append(str(f))
        trashed.append({"id": tid, "path": str(f)})
        with tx() as c:
            c.execute("UPDATE files SET missing=1, last_seen=? WHERE path=?", (time.time(), str(f)))
            c.execute("INSERT INTO trash(id, path, trash_path, render_id, size, deleted) VALUES (?,?,?,?,?,?)",
                      (tid, str(f), str(dst), row["render_id"] if row else None, size, time.time()))
        if row:
            ids.add(row["render_id"])
    return {"deleted": deleted, "records_kept": len(ids), "trashed": trashed}


# ---------------------------------------------------------------------------
# Trash (T1)
# ---------------------------------------------------------------------------
def trash_list() -> dict:
    """Restorable deleted files, newest first."""
    rows = query("SELECT t.*, r.thumb FROM trash t LEFT JOIN renders r ON r.id = t.render_id "
                 "WHERE t.flushed IS NULL ORDER BY t.deleted DESC")
    return {"ttl_s": TRASH_TTL_S, "items": [{
        "id": r["id"], "path": r["path"], "name": Path(r["path"]).name, "dir": str(Path(r["path"]).parent),
        "render_id": r["render_id"], "size": r["size"], "deleted": r["deleted"],
        "expires": r["deleted"] + TRASH_TTL_S, "kind": identity.media_kind(r["path"]),
        "thumb": r["thumb"] if r["thumb"] and Path(r["thumb"]).is_file() else None,
    } for r in rows]}


def _free_path(path: Path) -> Path:
    """*path*, or `<stem>_restored[_n]<ext>` beside it when taken."""
    if not path.exists():
        return path
    candidate, n = path.with_name(f"{path.stem}_restored{path.suffix}"), 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_restored_{n}{path.suffix}")
        n += 1
    return candidate


def _drop_trash_dir(trash_path: Path) -> None:
    try:
        trash_path.parent.rmdir()  # the <id>/ folder
        if trash_path.parent.parent.name == TRASH_BESIDE:
            trash_path.parent.parent.rmdir()  # an emptied .wzrd-trash/ beside the file
    except OSError:
        pass


def trash_restore(ids: list[str]) -> dict:
    """Put trashed files back where they lived (a taken path gets `_restored`)
    and re-index them; their render is live again."""
    restored, errors = [], []
    for tid in ids:
        row = query_one("SELECT * FROM trash WHERE id=?", (tid,))
        if not row:
            errors.append(f"{tid}: not in the trash")
            continue
        if row["flushed"] is not None:
            errors.append(f"{tid}: already flushed")
            continue
        src = Path(row["trash_path"])
        if not src.is_file():
            errors.append(f"{tid}: {src} is gone")
            continue
        dst = _free_path(Path(row["path"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        _drop_trash_dir(src)
        with tx() as c:
            c.execute("DELETE FROM trash WHERE id=?", (tid,))
            if row["render_id"]:
                c.execute("UPDATE renders SET purged=0, updated=? WHERE id=?", (time.time(), row["render_id"]))
        index_file(dst, force=True)
        restored.append({"id": tid, "path": str(dst)})
    return {"restored": restored, "errors": errors}


def trash_flush(ids: Optional[list[str]] = None, older_than_s: Optional[float] = None) -> dict:
    """Unlink trashed files for good: *ids*, else every entry (only those older
    than *older_than_s* when given). A render with no file left is `purged`."""
    if ids is not None and not ids:
        return {"flushed": 0}
    sql, params = "SELECT * FROM trash WHERE flushed IS NULL", []
    if ids is not None:
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += list(ids)
    if older_than_s is not None:
        sql += " AND deleted <= ?"
        params.append(time.time() - older_than_s)
    rows = query(sql, params)
    rids = set()
    for r in rows:
        p = Path(r["trash_path"])
        p.unlink(missing_ok=True)
        _drop_trash_dir(p)
        with tx() as c:
            c.execute("UPDATE trash SET flushed=? WHERE id=?", (time.time(), r["id"]))
        if r["render_id"]:
            rids.add(r["render_id"])
    with tx() as c:
        for rid in rids:
            left = c.execute("SELECT COUNT(*) FROM files WHERE render_id=? AND missing=0", (rid,)).fetchone()[0]
            if not left:
                c.execute("UPDATE renders SET purged=1, updated=? WHERE id=?", (time.time(), rid))
    return {"flushed": len(rows)}


def purge(folder: str, max_score: int, recursive: bool = False, dry_run: bool = False,
          min_score: int = 0) -> dict:
    """S5 bulk delete: media in *folder* scored *min_score* ≤ score ≤ *max_score*
    (Studio's score range filter; unrated files are never touched)."""
    d = str(Path(folder).expanduser().resolve())
    where = "(f.dir=? OR f.dir LIKE ?)" if recursive else "f.dir=?"
    params: list[Any] = [d, d.rstrip("/") + "/%"] if recursive else [d]
    rows = query(f"SELECT f.path FROM files f JOIN renders r ON r.id=f.render_id "
                 f"WHERE {where} AND f.missing=0 AND r.score IS NOT NULL AND r.score>=? AND r.score<=?",
                 (*params, int(min_score), int(max_score)))
    paths = [r["path"] for r in rows if Path(r["path"]).is_file()]
    if dry_run:
        return {"would_delete": paths}
    return delete(paths, siblings=False) if paths else {"deleted": [], "records_kept": 0, "trashed": []}
