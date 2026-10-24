"""TextureFlow's arg spec, as the endpoint publishes it — what `texture_flow` is built from.

The endpoint owns its inputs: textureflow-lab's `tflab/spec.yaml` (types, defaults, ranges, choices, an
agent-facing description per arg, a model-level `summary`), served by the deployed app's CPU function
`spec()` → `{app, git_sha, spec}`. WZRD reads it like an API client reads an OpenAPI schema:

- `current()` — the spec this process uses, loaded once and **never from the network**: the disk cache
  `<media>/.wzrd/textureflow_spec/<app>.json` (the last `spec()` reply of `$TEXTUREFLOW_APP`), else the
  committed snapshot `textureflow_spec.json` beside this file (offline, first boot, and apps without a
  `spec()`: the fallback `wzrd-textureflow`, or `textureflow` deployed before the lab shipped it).
- `refresh_in_background()` — hub start: fetch `spec()` on a daemon thread and write the cache. A changed
  spec takes effect on the next hub start (the tool's schema is fixed at registration).
- `parameters()` / `args_doc()` — the generated part of `texture_flow`'s signature and docstring
  (`tools.py` merges in WZRD's own args); defaults / ranges / choices are rendered from the structured
  fields, so they can't drift from what the server enforces.
- `unknown()` / `check_recipes()` — args the spec doesn't know (a renamed/removed arg in a saved recipe).

Refresh the snapshot after a lab spec change: `uv run python -m wzrd_mcp.tf_spec --snapshot` (from the
deployed app) or `--snapshot <lab>/tflab/spec.yaml`. Reference: docs/reference/textureflow.md.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ._log import _emit

SNAPSHOT_PATH = Path(__file__).with_name("textureflow_spec.json")
_TYPES = {"integer": int, "float": float, "boolean": bool, "string": str}


@dataclass(frozen=True)
class Loaded:
    reply: dict  # {app, git_sha, spec}
    source: str  # "cache <path>" | "snapshot"

    @property
    def spec(self) -> dict:
        return self.reply["spec"]

    @property
    def params(self) -> dict:
        return self.spec["parameters"]


def target() -> tuple[str, str]:
    """(app, Modal env) the Modal backend renders on."""
    return os.getenv("TEXTUREFLOW_APP", "textureflow"), os.getenv("WZRD_MODAL_ENV", "wzrd")


def cache_path(app: str) -> Path:
    from . import library

    return library.wzrd_dir() / "textureflow_spec" / f"{app}.json"


def _valid(reply: Any) -> bool:
    try:
        params = reply["spec"]["parameters"]
        return bool(params) and all(isinstance(p, dict) and ("type" in p or "anyOf" in p) for p in params.values())
    except (KeyError, TypeError):
        return False


def _load() -> Loaded:
    app, _ = target()
    try:
        path = cache_path(app)
        if path.is_file():
            reply = json.loads(path.read_text())
            if _valid(reply):
                return Loaded(reply, f"cache {path}")
            _emit(f"⚠ TextureFlow spec cache {path} is malformed — using the snapshot")
    except (OSError, ValueError) as exc:
        _emit(f"⚠ TextureFlow spec cache unreadable ({exc}) — using the snapshot")
    return Loaded(json.loads(SNAPSHOT_PATH.read_text()), "snapshot")


_current: Optional[Loaded] = None
_latest: Optional[dict] = None  # the spec() reply this process fetched (refresh_in_background)


def current() -> Loaded:
    global _current
    if _current is None:
        _current = _load()
    return _current


def capabilities() -> set[str]:
    """What the deployed app says it can do beyond the contract (its `spec()` reply's `capabilities`, e.g.
    `progress_dict`: `render(..., progress_key=)` writes progress to a Modal Dict). Only a reply fetched from
    the app counts — this process's refresh, else the cache — never the committed snapshot, which would claim
    them for the fallback app or an older deploy."""
    reply = _latest
    if reply is None:
        cur = current()
        reply = cur.reply if cur.source.startswith("cache") else {}
    caps = reply.get("capabilities") or (reply.get("spec") or {}).get("capabilities") or []
    return {str(c) for c in caps} if isinstance(caps, list) else set()


def public() -> dict:
    """What `GET /api/textureflow/spec` returns: the loaded spec + where it came from."""
    cur = current()
    return {"app": cur.reply.get("app"), "git_sha": cur.reply.get("git_sha"), "source": cur.source,
            "target": target()[0], **cur.spec}


# ---------------------------------------------------------------------------
# Signature + docstring
# ---------------------------------------------------------------------------
def _py_type(p: dict) -> Any:
    if "anyOf" in p:
        return str  # one media ref: a local path or URL
    if p.get("type") == "array":
        return list[str]
    return _TYPES.get(p.get("type"), Any)


def parameters(spec: dict, skip: set[str] = frozenset()) -> list[inspect.Parameter]:
    """One `Optional[T] = None` parameter per spec arg (None = not sent: the server's default applies)."""
    out = []
    for name, p in spec["parameters"].items():
        if name in skip:
            _emit(f"⚠ TextureFlow spec arg {name!r} collides with a WZRD-side arg — not exposed")
            continue
        out.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None,
                                     annotation=Optional[_py_type(p)]))
    return out


def _fmt(v: Any, quoted: bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return f'"{v}"' if quoted and isinstance(v, str) else str(v)


def describe(p: dict, note: str = "") -> str:
    """The arg's description + its structured facts, one line."""
    bits = [" ".join(str(p.get("description", "")).split())]
    if note:
        bits.append(note)
    elif p.get("required") and "default" not in p:
        bits.append("Required.")
    if "choices" in p:
        notes = p.get("choice_notes") or {}
        bits.append("One of: " + ", ".join(_fmt(c, True) + (f" ({notes[c]})" if c in notes else "")
                                           for c in p["choices"]) + ".")
    if "minimum" in p or "maximum" in p:
        step = f", step {p['step']}" if p.get("step") else ""
        bits.append(f"Range {p.get('minimum', '…')}-{p.get('maximum', '…')}{step}.")
    if "min_length" in p or "max_length" in p:
        bits.append(f"{p.get('min_length', 0)}-{p.get('max_length', '…')} items.")
    if "default" in p:
        bits.append(f"Default {_fmt(p['default'], 'choices' in p)}.")
    return " ".join(b for b in bits if b)


def args_doc(spec: dict, notes: dict[str, str], skip: set[str] = frozenset()) -> list[str]:
    """`name: description` lines for a Google-style Args section."""
    return [f"{name}: {describe(p, notes.get(name, ''))}" for name, p in spec["parameters"].items()
            if name not in skip]


# ---------------------------------------------------------------------------
# Unknown args (recipes saved under an older spec)
# ---------------------------------------------------------------------------
def unknown(args: dict) -> list[str]:
    return [k for k in args if k not in current().params]


def check_recipes() -> None:
    """Log one warning per saved-recipe arg the spec doesn't know (texture_flow drops them)."""
    from .recipe_tools import _all_recipes

    for r in _all_recipes():
        bad = unknown(r.get("args") or {})
        if bad:
            _emit(f"⚠ recipe {r.get('name')!r} ({r['_dir']}) has args the TextureFlow spec doesn't know: "
                  f"{bad} — texture_flow(recipe=…) drops them; re-save the recipe or rename the args")


# ---------------------------------------------------------------------------
# Refresh from the deployed app
# ---------------------------------------------------------------------------
def fetch(app: str, env: str) -> dict:
    import modal

    return modal.Function.from_name(app, "spec", environment_name=env).remote()


def refresh_in_background() -> threading.Thread:
    """Fetch `spec()` of $TEXTUREFLOW_APP and cache it; never blocks the caller."""

    def run() -> None:
        app, env = target()
        try:
            reply = fetch(app, env)
        except Exception as exc:  # noqa: BLE001 — offline, no spec() on this app, auth: all non-fatal
            kind = "has no spec() (fallback or pre-spec deploy)" if "not found" in str(exc).lower() else "unreachable"
            _emit(f"TextureFlow spec: {env}/{app} {kind} — keeping {current().source} "
                  f"(v{current().spec.get('version')}): {type(exc).__name__}")
            return
        if not _valid(reply):
            _emit(f"⚠ TextureFlow spec: {env}/{app} spec() returned a malformed reply — ignored")
            return
        global _latest
        _latest = reply
        path = cache_path(app)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reply, indent=1))
        tmp.replace(path)
        cur = current()
        if reply["spec"] != cur.spec:
            _emit(f"TextureFlow spec: {env}/{app} serves v{reply['spec'].get('version')} "
                  f"(running v{cur.spec.get('version')} from {cur.source}) — cached; restart the hub to use it")
        else:
            _emit(f"TextureFlow spec: {env}/{app} v{reply['spec'].get('version')} "
                  f"({(reply.get('git_sha') or '?')[:7]}) — up to date, cached")

    t = threading.Thread(target=run, name="textureflow-spec-refresh", daemon=True)
    t.start()
    return t


def _snapshot(source: Optional[str]) -> None:
    """Rewrite the vendored snapshot from the deployed app or a lab `spec.yaml`."""
    if source:
        import yaml

        reply = {"app": "textureflow", "git_sha": None, "spec": yaml.safe_load(Path(source).read_text())}
    else:
        reply = fetch(*target())
    if not _valid(reply):
        raise SystemExit(f"not a valid spec: {source or target()}")
    SNAPSHOT_PATH.write_text(json.dumps(reply, indent=1) + "\n")
    print(f"wrote {SNAPSHOT_PATH} (v{reply['spec'].get('version')}, {len(reply['spec']['parameters'])} args)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", nargs="?", const="", metavar="SPEC_YAML",
                    help="rewrite textureflow_spec.json from a lab spec.yaml (default: the deployed app's spec())")
    a = ap.parse_args()
    if a.snapshot is None:
        print(json.dumps(public(), indent=1))
    else:
        _snapshot(a.snapshot or None)
