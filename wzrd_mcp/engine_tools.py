"""§5.10 authoring MCP — the engine slice of the WZRD RPC surface.

Exposes ONLY the authoring facets of the realtime engine to a local Claude
Code session: scoped reads (`get_scene_context`), facet-bound write verbs
(bindings / effects / layers), and two utilities. Live operator controls
(masters, promote/pull, preview source, probe thresholds) are deliberately
NOT tools — they are the human's UI knobs, and the human's `promote` is the
gate to the projector. Every write lands on the **design** leg (the engine's
blanket leg rule); no tool takes a `leg` parameter.

Transport: persistent JSON-RPC over WebSocket to the engine at
``ws://127.0.0.1:9123`` (override with ``WZRD_ENGINE_WS``). Localhost-only by
nature — these tools are disabled by default in ``server.py`` and enabled in
the local ``tools_config.json``; the Modal image never installs
``websockets``, so a cloud deployment cannot carry them.

Context economy: every read is self-contained (status header + recent
changes ride along), history is served as data (rev counter + change ring),
and the edit loop never re-emits whole scene JSON in either direction.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import Any, Literal, Optional

from fastmcp.exceptions import ToolError

from ._log import logged_tool
from .server import mcp, get_timeout

try:
    import websockets
except ImportError:  # pragma: no cover — the Modal image path
    websockets = None

_ENGINE_WS = os.environ.get("WZRD_ENGINE_WS", "ws://127.0.0.1:9123")

# Channels the persistent connection always listens on. Deliberately
# excludes `preview` (§2.6 demand gate: a standing preview subscription
# would keep the design leg rendering + JPEG-encoding all session) and the
# noisy `drivers`/`audio` channels (one-shot subscribed when the drivers
# facet is read).
_STANDING_CHANNELS = ["hot_reload", "connectivity", "fps", "deck", "masters", "changes"]

_FACETS = ("layers", "bindings", "effects", "drivers")


def _engine_tool(name: str):
    """Register an engine tool — no-op passthrough when websockets is
    missing (cloud deployment), so importing this module never breaks the
    server."""

    def deco(fn):
        if websockets is None:
            return fn
        return mcp.tool(timeout=get_timeout(name))(logged_tool(fn))

    return deco


# ---------------------------------------------------------------------------
# Engine client — one persistent connection, background reader, sticky cache
# ---------------------------------------------------------------------------


class EngineClient:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._conn = None
        self._reader: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        #: channel -> last payload (sticky channels replay on subscribe)
        self.sticky: dict[str, Any] = {}
        #: local mirror of the engine's change ring (push-fed)
        self.changes: deque = deque(maxlen=10)
        #: epoch/rev as last reported by hello / change events / replies
        self.epoch: Optional[int] = None
        self.rev: Optional[int] = None
        #: time.time() of the first failed connect since last success
        self.down_since: Optional[float] = None
        self._frame_waiters: dict[str, list[asyncio.Future]] = {}

    # -- connection -------------------------------------------------------

    async def ensure_connected(self) -> None:
        if self._conn is not None:
            return
        async with self._lock:
            if self._conn is not None:
                return
            try:
                conn = await websockets.connect(
                    self.uri, open_timeout=3, max_size=32 * 1024 * 1024
                )
            except Exception as exc:
                if self.down_since is None:
                    self.down_since = time.time()
                since = time.strftime("%H:%M:%S", time.localtime(self.down_since))
                raise ToolError(
                    f"engine unreachable since {since} at {self.uri} — start the WZRD "
                    "engine first (wzrd-app Tauri shell, or headless: "
                    "`render-core --scene <scene.json> --ws-addr 127.0.0.1:9123`). "
                    f"({exc})"
                )
            self._conn = conn
            self.down_since = None
            self._reader = asyncio.create_task(self._read_loop(conn))
            # Declare actor identity once per session (never per call), then
            # attach the standing telemetry subscription.
            hello = await self._request("hello", {"actor": "agent"})
            self.epoch = hello.get("epoch")
            self.rev = hello.get("rev")
            await self._request("telemetry.subscribe", {"channels": _STANDING_CHANNELS})

    async def _read_loop(self, conn) -> None:
        try:
            async for raw in conn:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if msg.get("method") == "telemetry.event":
                    params = msg.get("params") or {}
                    ch = params.get("channel")
                    payload = params.get("payload")
                    if not ch:
                        continue
                    self.sticky[ch] = payload
                    if ch == "changes" and isinstance(payload, dict):
                        rev = payload.get("rev")
                        if not self.changes or self.changes[-1].get("rev") != rev:
                            self.changes.append(payload)
                        if isinstance(rev, int):
                            self.rev = max(self.rev or 0, rev)
                        if payload.get("epoch") is not None:
                            self.epoch = payload["epoch"]
                    for fut in self._frame_waiters.pop(ch, []):
                        if not fut.done():
                            fut.set_result(payload)
                elif msg.get("id") in self._pending:
                    fut = self._pending.pop(msg["id"])
                    if not fut.done():
                        fut.set_result(msg)
        except Exception:
            pass
        finally:
            self._conn = None
            if self.down_since is None:
                self.down_since = time.time()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("engine connection dropped"))
            self._pending.clear()
            for waiters in self._frame_waiters.values():
                for fut in waiters:
                    if not fut.done():
                        fut.set_exception(ConnectionError("engine connection dropped"))
            self._frame_waiters.clear()

    # -- request/response -------------------------------------------------

    async def _request(self, method: str, params: dict | None, timeout: float = 10.0):
        conn = self._conn
        if conn is None:
            raise ToolError("engine connection dropped mid-call — retry the tool")
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await conn.send(
            json.dumps(
                {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            )
        )
        try:
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise ToolError(
                f"{method} timed out after {timeout:.0f}s (transport safety net — the "
                f"probe itself takes ~1-2s). Last known status:\n{self.status_header()}"
            )
        except ConnectionError as exc:
            raise ToolError(f"{method}: {exc}")
        if "error" in msg:
            raise ToolError(f"{method}: {msg['error'].get('message', msg['error'])}")
        result = msg.get("result")
        if isinstance(result, dict):
            if isinstance(result.get("rev"), int):
                self.rev = result["rev"]
            if result.get("epoch") is not None:
                self.epoch = result["epoch"]
        return result

    async def request(self, method: str, params: dict | None = None, timeout: float = 10.0):
        await self.ensure_connected()
        return await self._request(method, params, timeout)

    async def wait_frame(self, channel: str, timeout: float = 6.0):
        """One frame off a telemetry channel (used with a temporary
        subscription — preview / drivers / audio)."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._frame_waiters.setdefault(channel, []).append(fut)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise ToolError(f"no `{channel}` frame arrived within {timeout:.0f}s")

    # -- status header ----------------------------------------------------

    def status_header(self) -> str:
        """The tiny self-contained header riding every read — makes any
        earlier tool output safe to compact away."""
        if self._conn is None:
            since = (
                time.strftime("%H:%M:%S", time.localtime(self.down_since))
                if self.down_since
                else "boot"
            )
            return (
                f"engine: UNREACHABLE since {since} ({self.uri}) — start the WZRD "
                "engine (Tauri shell or `render-core --ws-addr 127.0.0.1:9123`)"
            )
        parts = [f"engine: connected {self.uri} | epoch {self.epoch} rev {self.rev}"]
        fps = self.sticky.get("fps") or {}
        deck = self.sticky.get("deck") or {}
        if fps:
            mode = "presenting" if fps.get("presenting", True) else "offscreen"
            parts.append(f"fps {fps.get('fps', 0):.0f} ({mode})")
        if deck:
            parts.append(
                f"deck: promote={deck.get('promote')} preview={deck.get('preview_source')}"
            )
        hot = self.sticky.get("hot_reload") or {}
        if hot:
            probe = hot.get("probe") or {}
            verdict = (
                f" probe={probe.get('band')} {probe.get('predicted_p95_ms', 0):.1f}ms"
                if probe
                else ""
            )
            ok = "ok" if hot.get("ok") else f"FAIL ({hot.get('message')})"
            parts.append(f"last apply [{hot.get('target')}]: {ok}{verdict}")
        masters = self.sticky.get("masters") or {}
        d = masters.get("design") or {}
        if d:
            parts.append(
                "design masters: "
                f"bri {d.get('brightness', 1):.2f} spd {d.get('speed', 1):.2f} "
                f"sat {d.get('saturation', 1):.2f} audio {d.get('audioListen', 1):.2f}"
            )
        return " | ".join(parts)

    def recent_changes_text(self, entries=None) -> str:
        entries = entries if entries is not None else list(self.changes)
        if not entries:
            return "recent changes: (none this session)"
        lines = ["recent changes (rev · actor · facet · summary):"]
        for e in entries[-10:]:
            ts = time.strftime("%H:%M:%S", time.localtime((e.get("ts_ms") or 0) / 1000))
            lines.append(
                f"  r{e.get('rev')} {ts} {e.get('actor')} {e.get('facet')}: {e.get('summary')}"
            )
        return "\n".join(lines)


_client = EngineClient(_ENGINE_WS)


# ---------------------------------------------------------------------------
# Rendering helpers (digest = terse aligned lines, not pretty JSON)
# ---------------------------------------------------------------------------


def _fmt_param(value: Any) -> str:
    if isinstance(value, dict) and "driver" in value:
        extras = ",".join(f"{k}={v}" for k, v in value.items() if k != "driver")
        return f"{value['driver']}({extras})" if extras else str(value["driver"])
    if isinstance(value, list):
        return "[" + ",".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _resolve_selector(select: dict, pack: dict) -> list[str]:
    layers = pack.get("layers", [])
    if select.get("all"):
        return [l["id"] for l in layers]
    if select.get("id"):
        return [select["id"]] if any(l["id"] == select["id"] for l in layers) else []
    if select.get("tag"):
        return [l["id"] for l in layers if select["tag"] in (l.get("tags") or [])]
    if select.get("group"):
        for g in pack.get("groups", []):
            if g["id"] == select["group"]:
                return list(g["members"])
        return []
    return []


def _layer_groups(pack: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for g in pack.get("groups", []):
        for m in g.get("members", []):
            out.setdefault(m, []).append(g["id"])
    return out


def _render_layers(pack: dict, depth: str, ids: Optional[list[str]]) -> str:
    groups_of = _layer_groups(pack)
    lines = [
        f"layers ({len(pack.get('layers', []))} @ {pack.get('width')}x{pack.get('height')}, "
        f"pack {pack.get('pack_dir')}):"
    ]
    for l in pack.get("layers", []):
        if ids and l["id"] not in ids and (l.get("label") or "") not in ids:
            continue
        c = l.get("centroid")
        centroid = f"({c[0]:.0f},{c[1]:.0f})" if c else "?"
        label = f" label={l['label']!r}" if l.get("label") else ""
        grps = ",".join(groups_of.get(l["id"], [])) or "-"
        line = (
            f"  {l['id']}{label} groups=[{grps}] z={l.get('z', 0)} centroid={centroid}"
        )
        if depth == "full":
            tags = ",".join(l.get("tags") or []) or "-"
            line += f" tags=[{tags}] bbox={l.get('bbox')} mask={l.get('mask_path')}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("  (no layers matched)")
    return "\n".join(lines)


def _render_bindings(
    scene: dict, pack: dict, overrides: dict, depth: str, ids: Optional[list[str]]
) -> str:
    bindings = scene.get("bindings", [])
    bpm = (scene.get("transport") or {}).get("bpm", 120)
    lines = [
        f"scene: transport.bpm={bpm} | bindings={len(bindings)} | post={len(scene.get('post', []))}"
    ]
    for b in bindings:
        if ids and b.get("id") not in ids:
            continue
        if depth == "full":
            lines.append(f"  {b.get('id')}: {json.dumps(b, indent=2)}")
            continue
        select = b.get("select") or {}
        sel_desc = ",".join(f"{k}={v}" for k, v in select.items() if k != "pick")
        if select.get("pick"):
            sel_desc += f" pick={select['pick'].get('mode')}"
        resolved = _resolve_selector(select, pack)
        shown = ",".join(resolved[:6]) + ("…" if len(resolved) > 6 else "")
        effect = b.get("effect")
        effect_name = effect if isinstance(effect, str) else "(inline wgsl)"
        params = b.get("params") or {}
        pdesc = ", ".join(f"{k}={_fmt_param(v)}" for k, v in params.items())
        ov = overrides.get(b.get("id"), {})
        ovdesc = (
            " [overridden: " + ", ".join(f"{k}={v:g}" for k, v in ov.items()) + "]"
            if ov
            else ""
        )
        lines.append(
            f"  {b.get('id')}: {sel_desc} → [{shown or 'NO MATCH'}] → "
            f"{effect_name}({pdesc}){ovdesc}"
        )
    if len(bindings) == 0:
        lines.append("  (no bindings — scene is empty; use upsert_binding or set_scene)")
    return "\n".join(lines)


def _render_effect_digest(catalog: list[dict]) -> str:
    lines = [f"effects ({len(catalog)}):"]
    for e in sorted(catalog, key=lambda x: x.get("name", "")):
        inputs = ", ".join(i.get("name", "?") for i in e.get("inputs", []))
        lines.append(f"  {e.get('name')} ({e.get('kind')}): {inputs or 'no inputs'}")
    return "\n".join(lines)


def _render_effect_full(desc: dict) -> str:
    lines = [f"effect {desc.get('name')} ({desc.get('kind')}):"]
    for i in desc.get("inputs", []):
        meta = " ".join(
            f"{k}={i[k]}" for k in ("min", "max", "step", "unit", "widget") if i.get(k) is not None
        )
        lines.append(
            f"  {i.get('name')}: {i.get('type')} default={_fmt_param(i.get('default'))} {meta}".rstrip()
        )
    if desc.get("wgsl"):
        lines.append("  --- shader.wgsl ---")
        lines.append(desc["wgsl"].rstrip())
    return "\n".join(lines)


def _image_block(b64: str):
    """Best-effort MCP image content block from base64 JPEG."""
    try:
        from mcp.types import ImageContent

        return ImageContent(type="image", data=b64, mimeType="image/jpeg")
    except Exception:
        return None


def _apply_reply_text(result: dict, what: str) -> str:
    rev = result.get("rev")
    probe = result.get("probe") or {}
    if probe:
        verdict = (
            f"probe: {probe.get('band')} (predicted full-res p95 "
            f"{probe.get('predicted_p95_ms', 0):.1f} ms)"
        )
    else:
        verdict = "probe: not run (no new pipelines — cached shaders reused)"
    return f"r{rev}: {what} applied to the design leg. {verdict}"


def _apply_reply(result: dict, what: str) -> list:
    out: list = [_apply_reply_text(result, what)]
    thumb = (result.get("probe") or {}).get("thumbnail_b64")
    if thumb:
        img = _image_block(thumb)
        if img is not None:
            out.append(img)
    return out


# ---------------------------------------------------------------------------
# The read tool
# ---------------------------------------------------------------------------


@_engine_tool("get_scene_context")
async def get_scene_context(
    scope: Optional[list[Literal["layers", "bindings", "effects", "drivers"]]] = None,
    depth: Literal["digest", "full"] = "digest",
    ids: Optional[list[str]] = None,
    since_rev: Optional[int] = None,
) -> str:
    """Read the current engine/scene state. CALL THIS FIRST on every operator
    command — it reflects the human's UI edits the instant they hit the
    engine (reads engine memory, never disk).

    Omit everything for the orient call: a full-breadth digest (every pack
    layer with its human labels + groups, every binding one-liner with
    resolved selector targets, effect names, live driver values) plus a
    status header and recent changes — a few hundred tokens. Then scope
    further reads to the facets the task touches.

    depth="full" REQUIRES an explicit scope (breadth is cheap, depth must be
    aimed). Use ids to drill down: scope=["effects"], depth="full",
    ids=["bloom"] returns one descriptor + its WGSL and nothing else.

    since_rev: elide facet sections with no changes after that rev (use the
    rev a previous call returned). On engine restart or ring wrap you get a
    full digest with an explicit note — never a silently-partial diff.
    """
    scopes = list(scope) if scope else list(_FACETS)
    for s in scopes:
        if s not in _FACETS:
            raise ToolError(f"unknown facet {s!r}; valid: {', '.join(_FACETS)}")
    if depth == "full" and not scope:
        raise ToolError(
            "depth='full' requires an explicit scope — full-everything is too big. "
            "Pick facets, e.g. scope=['effects'], and narrow with ids=[…]."
        )

    await _client.ensure_connected()

    # Change ring (authoritative backfill; also drives since_rev elision).
    ring = await _client.request(
        "changes.list",
        {"since_rev": since_rev, "epoch": _client.epoch if since_rev is not None else None},
    )
    entries = ring.get("entries", [])
    note = ring.get("note")
    _client.rev = ring.get("rev", _client.rev)

    if since_rev is not None and not note:
        touched = {e.get("facet") for e in entries}
        # layers changes also invalidate binding target resolution.
        if "layers" in touched:
            touched.add("bindings")
        scopes = [s for s in scopes if s in touched]
        if not scopes:
            return "\n".join(
                [
                    _client.status_header(),
                    f"no changes since rev {since_rev} — your view is current.",
                ]
            )

    sections: list[str] = [_client.status_header()]
    if note:
        sections.append(f"NOTE: {note}")
    sections.append(_client.recent_changes_text(entries if entries else None))

    pack = None
    if "layers" in scopes or "bindings" in scopes:
        pack = await _client.request("pack.info")

    if "layers" in scopes:
        sections.append(_render_layers(pack, depth, ids))

    if "bindings" in scopes:
        state = await _client.request("scene.getState", {})
        try:
            scene = json.loads(state["json"])
        except (KeyError, ValueError) as exc:
            raise ToolError(f"scene state unreadable: {exc}")
        params = await _client.request("param.list", {})
        sections.append(
            _render_bindings(scene, pack, params.get("overrides", {}), depth, ids)
        )

    if "effects" in scopes:
        if depth == "full":
            names = ids
            if not names:
                catalog = (await _client.request("effect.describe", {})).get("effects", [])
                names = [e["name"] for e in catalog]
            for n in names:
                desc = await _client.request("effect.describe", {"name": n})
                sections.append(_render_effect_full(desc))
        else:
            catalog = (await _client.request("effect.describe", {})).get("effects", [])
            if ids:
                catalog = [e for e in catalog if e.get("name") in ids]
            sections.append(_render_effect_digest(catalog))

    if "drivers" in scopes:
        # One-shot subscription — drivers/audio are 10-30 Hz channels; a
        # standing subscription would spam this client for nothing.
        await _client.request(
            "telemetry.subscribe", {"channels": _STANDING_CHANNELS + ["drivers", "audio"]}
        )
        try:
            rows = await _client.wait_frame("drivers", timeout=3.0)
            audio = None
            if depth == "full":
                audio = await _client.wait_frame("audio", timeout=3.0)
        finally:
            await _client.request(
                "telemetry.subscribe", {"channels": _STANDING_CHANNELS}
            )
        lines = ["drivers (design leg, live values):"]
        for r in (rows or {}).get("drivers", []):
            ov = " [override]" if r.get("overridden") else ""
            lines.append(
                f"  {r.get('binding_id')}.{r.get('param_name')} = "
                f"{r.get('value', 0):.3f} ({r.get('source')}){ov}"
            )
        if len(lines) == 1:
            lines.append("  (no driver-bound params in the design scene)")
        if depth == "full" and audio:
            lines.append(
                f"  audio bands L/M/H: {audio.get('band_low', 0):.2f}/"
                f"{audio.get('band_mid', 0):.2f}/{audio.get('band_high', 0):.2f} | "
                f"onsets: {audio.get('onset_low', 0):.2f}/{audio.get('onset_mid', 0):.2f}/"
                f"{audio.get('onset_high', 0):.2f}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Write verbs — each binds to exactly one facet
# ---------------------------------------------------------------------------


@_engine_tool("upsert_binding")
async def upsert_binding(id: str, binding: dict) -> list:
    """Create or replace ONE binding in the design scene (facet: bindings).
    The rest of the scene never enters context: this does an engine-side
    read → splice → compare-and-swap write, and rejects with a re-read
    instruction if the design moved since (retry by calling again).

    `binding` is the full binding object: {"select": {...}, "effect": "name",
    "params": {...}} — the `id` field is set from the id argument. Check the
    probe verdict in the reply before reporting success; a thumbnail of the
    design composite rides along when shaders were rebuilt.
    """
    state = await _client.request("scene.getState", {})
    try:
        scene = json.loads(state["json"])
    except (KeyError, ValueError) as exc:
        raise ToolError(f"scene state unreadable: {exc}")
    base_rev = state.get("rev")
    b = dict(binding)
    b["id"] = id
    bindings = scene.setdefault("bindings", [])
    replaced = False
    for i, existing in enumerate(bindings):
        if existing.get("id") == id:
            bindings[i] = b
            replaced = True
            break
    if not replaced:
        bindings.append(b)
    result = await _client.request(
        "scene.load",
        {"json": json.dumps(scene, indent=2), "base_rev": base_rev},
        timeout=15.0,
    )
    verb = "replaced" if replaced else "added"
    return _apply_reply(result, f"binding '{id}' {verb}")


@_engine_tool("remove_binding")
async def remove_binding(id: str) -> list:
    """Remove one binding from the design scene by id (facet: bindings)."""
    state = await _client.request("scene.getState", {})
    scene = json.loads(state["json"])
    base_rev = state.get("rev")
    bindings = scene.get("bindings", [])
    kept = [b for b in bindings if b.get("id") != id]
    if len(kept) == len(bindings):
        available = [b.get("id") for b in bindings]
        raise ToolError(f"no binding with id {id!r}; bindings: {available}")
    scene["bindings"] = kept
    result = await _client.request(
        "scene.load",
        {"json": json.dumps(scene, indent=2), "base_rev": base_rev},
        timeout=15.0,
    )
    return _apply_reply(result, f"binding '{id}' removed")


@_engine_tool("upsert_effect")
async def upsert_effect(
    name: str, wgsl: str, descriptor: Optional[dict] = None
) -> list:
    """Write or update a project-local WGSL effect (facet: effects). The
    engine naga-validates FIRST (a broken shader returns line-mapped
    diagnostics and touches nothing), then re-applies the design scene so
    the new pipeline is pre-flight probed — the reply carries the verdict
    (green/yellow/red) plus a thumbnail of the design composite: you SEE
    what you made. Red = refused (predicted frame cost too high) — simplify
    the shader and retry.

    `wgsl` is the body only: implement `fn effect(uv: vec2<f32>, mask: f32)
    -> vec4<f32>` (premultiplied RGBA, additive). Available: `state.*`
    (time/bar phase/audio), `f_param(N)`/`c_param(N)` slots, `sample_mask(uv)`,
    `layer_seed()`, `layer_index()`, `layer_count()`, `layer_centroid()`,
    `layer_bbox()`. `descriptor` declares typed inputs:
    {"inputs": [{"name": "rate", "type": "float", "default": 1.0, "min": 0,
    "max": 8}, {"name": "tint", "type": "color", "default": "#ffffff"}]}.

    NOTE: the probe only runs if the design scene binds this effect — bind
    it first (upsert_binding) to get a verdict + thumbnail on iteration.
    """
    params: dict[str, Any] = {"name": name, "wgsl": wgsl}
    if descriptor is not None:
        params["descriptor"] = descriptor
    result = await _client.request("effect.upsert", params, timeout=20.0)
    return _apply_reply(result, f"effect '{name}' upserted")


@_engine_tool("remove_effect")
async def remove_effect(name: str) -> str:
    """Delete a project-local effect's files from disk (facet: effects).
    The in-memory definition survives until restart, so bindings that still
    reference it keep drawing — remove those bindings too."""
    result = await _client.request("effect.remove", {"name": name}, timeout=15.0)
    return f"r{result.get('rev')}: effect '{name}' removed from disk."


@_engine_tool("set_groups")
async def set_groups(groups: dict[str, Optional[list[str]]]) -> str:
    """Create/update/remove named layer groups in the pack's identity
    sidecar (facet: layers). Per-key: a list of layer ids sets that group's
    membership; null (or []) removes the group. Groups are surface-level
    identity — scene selectors ({"group": "canopy"}) resolve through them,
    and bindings targeting a changed group re-resolve immediately."""
    result = await _client.request(
        "identity.setGroups", {"groups": groups}, timeout=15.0
    )
    pack = result.get("pack") or {}
    gdesc = ", ".join(
        f"{g['id']}[{len(g['members'])}]" for g in pack.get("groups", [])
    ) or "(none)"
    return f"r{result.get('rev')}: groups updated. Current groups: {gdesc}"


@_engine_tool("set_labels")
async def set_labels(labels: dict[str, Optional[str]]) -> str:
    """Set human labels on pack layers in the identity sidecar (facet:
    layers). Per-key: a string sets the label, null clears it. Labels are
    how the operator's surface-language ("the trunk") resolves to layer ids
    — label layers early so later commands can reference them by name."""
    result = await _client.request(
        "identity.setGroups", {"labels": labels}, timeout=15.0
    )
    pack = result.get("pack") or {}
    ldesc = ", ".join(
        f"{l['id']}={l['label']!r}" for l in pack.get("layers", []) if l.get("label")
    ) or "(none)"
    return f"r{result.get('rev')}: labels updated. Labelled layers: {ldesc}"


@_engine_tool("set_scene")
async def set_scene(scene_json: str) -> list:
    """Replace the ENTIRE design scene (escape hatch — prefer the granular
    upsert_binding/remove_binding verbs, which never re-emit the whole scene
    into context). Use only for initial authoring from empty or a structural
    rewrite. The scene must carry {"version": 1, "pack": "<relative pack
    path>", "bindings": [...]}. Reply includes the probe verdict when new
    shaders are involved."""
    try:
        json.loads(scene_json)
    except ValueError as exc:
        raise ToolError(f"scene_json is not valid JSON: {exc}")
    result = await _client.request("scene.load", {"json": scene_json}, timeout=20.0)
    return _apply_reply(result, "full scene")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@_engine_tool("validate_wgsl")
async def validate_wgsl(source: str) -> str:
    """Dry-run naga validation of an effect WGSL body (no write, no probe).
    Diagnostics are remapped to your source's line numbers. Cheap — use it
    to iterate on syntax before upsert_effect."""
    result = await _client.request("wgsl.validate", {"source": source})
    if result.get("ok"):
        return "WGSL valid (naga). Note: validity ≠ performance — the probe at upsert time has the last word."
    lines = [
        f"line {d.get('line')}:{d.get('column')}: {d.get('severity')}: {d.get('message')}"
        for d in result.get("diagnostics", [])
    ]
    return "WGSL invalid:\n" + "\n".join(lines)


@_engine_tool("get_preview")
async def get_preview() -> list:
    """One JPEG frame of the DESIGN composite — what your current design leg
    looks like right now. Use after edits that reuse cached shaders (no
    probe ran, so no thumbnail came back). Strictly one-shot: subscribes,
    takes one frame, unsubscribes (a standing subscription would keep the
    design leg rendering all session)."""
    await _client.ensure_connected()
    await _client.request(
        "telemetry.subscribe", {"channels": _STANDING_CHANNELS + ["preview"]}
    )
    try:
        frame = await _client.wait_frame("preview", timeout=6.0)
    finally:
        await _client.request("telemetry.subscribe", {"channels": _STANDING_CHANNELS})
    img = _image_block(frame.get("data_b64", ""))
    caption = (
        f"design composite {frame.get('width')}x{frame.get('height')} "
        f"({_client.status_header()})"
    )
    return [caption, img] if img is not None else [caption + " — image block unavailable"]
