"""Collect `perf_spike` events from a running engine for N seconds.

    uv run -q --with websockets python spikes.py <seconds> [ws://127.0.0.1:9123]

Start it in the background *before* the action you're testing (minimize,
room switching, TAKE), then read its one-line summary.
"""
import asyncio, json, sys, time
import websockets


async def main(secs, url):
    async with websockets.connect(url, max_size=None) as w:
        await w.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "telemetry.subscribe",
                                 "params": {"channels": ["perf_spike"]}}))
        t, out = time.monotonic(), []
        while time.monotonic() - t < secs:
            try:
                m = json.loads(await asyncio.wait_for(w.recv(), timeout=0.25))
            except asyncio.TimeoutError:
                continue
            if m.get("method") == "telemetry.event" and m["params"]["channel"] == "perf_spike":
                out.append(m["params"]["payload"])
        worst = max((o.get("frame_ms", 0) for o in out), default=0)
        print(f"spikes: {len(out)}  worst frame_ms: {worst:.1f}")

asyncio.run(main(float(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9123"))
