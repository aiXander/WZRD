"""Gate clip continuity on the real engine: edits and TAKEs never blank or jump a clip.

Boots `--bench` (the fx_gate seven-clip layout, `videoClip` on every layer),
then during an 8 s 60 fps screen recording of the engine window hammers the
paths that used to reopen clips: a design `scene.load` every 0.4 s
alternating every clip's `rate` 1.0 ↔ 1.25 (a same-leg rebuild), and a cut
TAKE (`promote {fade_ms: 0, quantize: "now"}`) after every second edit (live
adopts design; design rebuilds; the idle design leg wakes through the clip
gate). See docs/reference/render-engine.md §2.1 "Clip continuity".

Analysis: a **transient** is a frame unlike both neighbours while the
neighbours resemble each other — a black frame, or a flash of a wrong clip
frame, inserted into continuous playback. Score per frame:
min(d(i-1,i), d(i,i+1)) / d(i-1,i+1), with d = mean abs luma difference;
normal motion scores ≈ 0.5, a 30 fps clip's duplicate frames 0, a loop seam
or real cut is a single step (low score). Pass = no frame scores > 3.

Same requirements as fx_gate.py (window visible + in front, release build,
ffmpeg, Screen Recording permission); overwrites the bench's scene.json.

    uv run python scripts/continuity_gate.py --bench projects/fx-bench
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fx_gate import Engine, attach, bench_scene, content_rect, projector_res, record  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SECS = 8.0
THRESHOLD = 3.0


def with_rate(scene: dict, rate: float) -> dict:
    s = json.loads(json.dumps(scene))
    for b in s["bindings"]:
        b["params"]["image"]["rate"] = rate
    return s


def transients(frames: np.ndarray) -> list[tuple[int, float]]:
    luma = frames.astype(np.float32).mean(axis=3)
    d = lambda a, b: float(np.abs(luma[a] - luma[b]).mean())  # noqa: E731
    out = []
    for i in range(1, len(luma) - 1):
        score = min(d(i - 1, i), d(i, i + 1)) / (d(i - 1, i + 1) + 0.5)
        if score > THRESHOLD:
            out.append((i, round(score, 1)))
    return out


async def run(bench: Path, out: Path) -> dict:
    from wzrd.check_frames import _video_frames

    base = bench_scene(bench, "videoClip")
    e = Engine(bench, base, out / "continuity.log")
    edits = takes = 0
    try:
        r = await attach(2)
        rect = content_rect(e.front_window(), projector_res(bench))
        await asyncio.sleep(2)
        mp4 = out / "rec_continuity.mp4"
        rec = threading.Thread(target=record, args=(rect, SECS, mp4))
        rec.start()
        await asyncio.sleep(0.5)
        while rec.is_alive():
            st = await r.call("scene.getState", {"leg": "design"})
            rate = 1.25 if edits % 2 == 0 else 1.0
            await r.call("scene.load", {"json": json.dumps(with_rate(base, rate)), "base_rev": st.get("rev")})
            edits += 1
            if edits % 2 == 0:
                await r.call("promote", {"fade_ms": 0, "quantize": "now"})
                takes += 1
            await asyncio.sleep(0.4)
        rec.join()
    finally:
        e.stop()
    frames, fps = _video_frames(mp4)
    bad = transients(frames)
    return {"status": "pass" if not bad else "fail", "transients": bad, "edits": edits,
            "takes": takes, "rec": {"frames": len(frames), "fps": round(fps, 2)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "test_results" / "continuity_gate")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(run(a.bench.resolve(), a.out))
    (a.out / "continuity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
