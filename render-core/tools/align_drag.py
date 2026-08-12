#!/usr/bin/env python3
"""Drive the §5.14 alignment layer over the engine's WebSocket, with no UI.

This is the Phase-B proof that the alignment surface is UI-independent: every
verb the Align tab calls is reachable from here, so a projector can be aligned
from a laptop across the room, a MIDI box, or — the reason this file is kept
rather than thrown away — a camera loop.

    # engine must be running with a control surface:
    #   cargo run -- --scene examples/phase3_smoke.scene.json --windowed --ws-addr 127.0.0.1:9123
    # (the Tauri shell binds the same port, so it works against that too)

    uv run --with websockets render-core/tools/align_drag.py --demo sweep
    uv run --with websockets render-core/tools/align_drag.py --pattern grid
    uv run --with websockets render-core/tools/align_drag.py --get
    uv run --with websockets render-core/tools/align_drag.py --reset

The camera loop this grows into is the same shape: capture → detect → solve →
`alignment.set`. Nothing in the engine needs to change for it; `set` already
takes the corners and handles a solver would produce, and the LUT the engine
bakes is already the runtime representation a dense field would upload into.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys

try:
    import websockets
except ImportError:  # pragma: no cover - operator-facing hint
    sys.exit("needs `websockets`: uv run --with websockets render-core/tools/align_drag.py")

DEFAULT_URL = "ws://127.0.0.1:9123"


class Engine:
    """Minimal JSON-RPC 2.0 client. Sequential request/reply is plenty here."""

    def __init__(self, ws):
        self.ws = ws
        self.next_id = 0

    async def call(self, method: str, params: dict | None = None):
        self.next_id += 1
        req = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            req["params"] = params
        await self.ws.send(json.dumps(req))
        # Telemetry notifications share the socket; skip anything without our id.
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") != self.next_id:
                continue
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error']['message']}")
            return msg["result"]


async def sweep(engine: Engine, seconds: float, hz: float) -> None:
    """Orbit the top-left corner so the warp is unmistakable on the wall.

    Deliberately asymmetric (one corner only) — a symmetric wobble would look
    identical under a Y-flip bug, which is exactly the failure this is meant to
    make obvious.
    """
    base = await engine.call("alignment.get")
    corners = [list(c) for c in base["corners"]]
    origin = list(corners[0])
    steps = int(seconds * hz)
    print(f"sweeping TL corner around {origin} for {seconds:g}s at {hz:g} Hz")
    for i in range(steps):
        t = 2 * math.pi * i / hz / 2.0
        corners[0] = [origin[0] + 0.06 * math.cos(t), origin[1] + 0.04 * math.sin(t)]
        await engine.call("alignment.set", {"corners": corners})
        await asyncio.sleep(1.0 / hz)
    corners[0] = origin
    await engine.call("alignment.set", {"corners": corners})
    print("restored")


async def bulge(engine: Engine) -> None:
    """Add one local handle and push it around — the RBF residual, visibly.

    The handle is created with no `anchor`, so the engine anchors it at the
    current field: dropping it does nothing to the image, and only the drag
    that follows bends anything.
    """
    doc = await engine.call("alignment.set", {"points": [{"dest": [0.5, 0.5], "radius": 0.3}]})
    handle = doc["points"][-1]
    print(f"added handle {handle['id']} anchored at {handle['anchor']}")
    for i in range(60):
        t = 2 * math.pi * i / 30.0
        handle["dest"] = [0.5 + 0.08 * math.cos(t), 0.5 + 0.08 * math.sin(t)]
        await engine.call("alignment.set", {"points": [handle]})
        await asyncio.sleep(1 / 30)
    await engine.call("alignment.set", {"points": []})
    print("handle removed")


async def verify_isolation(engine: Engine, path: str) -> int:
    """§6 invariant: alignment survives everything else.

    Scene loads, promotes and pulls are about what the surface *does*;
    alignment is about where the light physically lands. The two must not be
    able to touch each other, so this drives a real scene mutation through the
    engine and asserts `alignment.json` came out byte-identical.
    """
    import hashlib
    import pathlib

    p = pathlib.Path(path)

    # Dirty the alignment first, then let the engine's debounce write it, so
    # the file under test is one the engine actually authored this run.
    await engine.call("alignment.set", {"corners": [[0.03, 0.01], [0.97, 0.0], [1.0, 0.98], [0.0, 1.0]]})
    await asyncio.sleep(2.0)
    if not p.exists():
        print(f"FAIL: {p} was never written", file=sys.stderr)
        return 1
    before = hashlib.sha256(p.read_bytes()).hexdigest()

    scene = (await engine.call("scene.getState"))["json"]
    await engine.call("scene.load", {"json": scene})
    await engine.call("pull")
    await asyncio.sleep(2.5)

    after = hashlib.sha256(p.read_bytes()).hexdigest()
    if before != after:
        print(f"FAIL: scene.load/pull mutated {p}\n  {before}\n  {after}", file=sys.stderr)
        return 1
    # The document is stored as f32, so compare with a tolerance rather than
    # for equality — 0.03 comes back as 0.029999999.
    doc = await engine.call("alignment.get")
    got = doc["corners"][0]
    if abs(got[0] - 0.03) > 1e-6 or abs(got[1] - 0.01) > 1e-6:
        print(f"FAIL: corners changed in memory: {doc['corners']}", file=sys.stderr)
        return 1
    print(f"OK: alignment.json byte-identical across scene.load + pull ({before[:12]}…)")
    return 0


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--get", action="store_true", help="print the alignment document")
    ap.add_argument("--reset", action="store_true", help="identity corners, no handles")
    ap.add_argument("--enable", choices=["on", "off"], help="toggle the warp")
    ap.add_argument("--pattern", choices=["none", "grid", "border", "corners"])
    ap.add_argument("--demo", choices=["sweep", "bulge"])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument(
        "--verify-isolation",
        metavar="ALIGNMENT_JSON",
        help="assert scene.load + pull leave the given alignment.json byte-identical (§6)",
    )
    args = ap.parse_args()

    async with websockets.connect(args.url) as ws:
        engine = Engine(ws)
        # Declare ourselves so design-facet changes are attributed correctly.
        # Alignment isn't a design facet, but the handshake also returns the
        # engine's epoch, which is a cheap liveness check.
        await engine.call("hello", {"actor": "agent"})

        if args.verify_isolation:
            sys.exit(await verify_isolation(engine, args.verify_isolation))
        if args.reset:
            print(json.dumps(await engine.call("alignment.reset"), indent=2))
        if args.enable:
            await engine.call("alignment.set", {"enabled": args.enable == "on"})
            print(f"warp {args.enable}")
        if args.pattern:
            await engine.call("alignment.setTestPattern", {"pattern": args.pattern})
            print(f"test pattern: {args.pattern}")
        if args.demo == "sweep":
            await sweep(engine, args.seconds, args.hz)
        elif args.demo == "bulge":
            await bulge(engine)
        if args.get or not (args.reset or args.enable or args.pattern or args.demo):
            print(json.dumps(await engine.call("alignment.get"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
