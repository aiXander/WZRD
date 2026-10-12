"""Interleaved A/B of two render-core binaries on bench scenarios.

    python3 ab.py <outdir> <rounds> <binA> <binB> <scenario> [<scenario> ...]

Runs A and B alternately (ABBA order per round) so both see the same machine
load, then prints the median of each metric per scenario. Use this — not a
single perf_gate run — to judge a change on a shared or loaded machine.
Binaries are absolute paths; scenarios are names under render-core/perf/scenarios/.
"""
import json, os, statistics as st, subprocess, sys
from pathlib import Path

out, rounds, A, B, scens = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
os.makedirs(out, exist_ok=True)
rc = Path(__file__).resolve().parents[4] / "render-core"
env = {**os.environ, "RUST_LOG": "info,wgpu_core=warn,wgpu_hal=warn,naga=warn"}


def pick(d):
    p = d["perf"]; g = p.get("gpu") or {}; gp = g.get("passes") or {}
    lat = (p.get("latency") or {}).get("knob_submit") or {}
    clips = p.get("clips") or {}
    return {"work_p95": p["frame"]["work"]["p95"],
            "gpu_total_p50": (g.get("total") or {}).get("p50"), "gpu_total_p95": (g.get("total") or {}).get("p95"),
            "live_p50": (gp.get("composite.live") or {}).get("p50"), "design_p50": (gp.get("composite.design") or {}).get("p50"),
            "decode_p95": clips.get("decode_p95"), "decodes": clips.get("decodes"), "stale": clips.get("stale"),
            "knob_p95": lat.get("p95"), "fps": p.get("fps")}


res = {}
for sc in scens:
    for i in range(rounds):
        for tag, b in ((("A", A), ("B", B)) if i % 2 == 0 else (("B", B), ("A", A))):
            f = f"{out}/{sc}-{tag}-{i}.json"
            try:
                subprocess.run([b, "--bench", f"perf/scenarios/{sc}.json", "--bench-out", f],
                               cwd=rc, env=env, timeout=240, capture_output=True)
                res.setdefault((sc, tag), []).append(pick(json.load(open(f))))
            except Exception as e:  # a hung/crashed run is a result too
                print("ERR", sc, tag, e, flush=True)
    for key in ["work_p95", "gpu_total_p50", "gpu_total_p95", "live_p50", "design_p50",
                "decode_p95", "decodes", "stale", "knob_p95", "fps"]:
        a = [r[key] for r in res.get((sc, "A"), []) if r.get(key) is not None]
        b = [r[key] for r in res.get((sc, "B"), []) if r.get(key) is not None]
        if a and b:
            ma, mb = st.median(a), st.median(b)
            print(f"{sc:15} {key:14} A {ma:8.2f}  B {mb:8.2f}  Δ {((mb - ma) / ma * 100 if ma else 0):+6.0f}%", flush=True)
