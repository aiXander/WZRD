# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Perf gate (perf-monitor PM3): run the windowless bench suite, compare to this machine's baseline.

    uv run scripts/perf_gate.py                       # build, run the suite 3x, print the table, exit 1 on FAIL
    uv run scripts/perf_gate.py smoke clips-7         # only these scenarios (or a path to a scratch scenario
                                                      # file — judged against the baseline of its file stem)
    uv run scripts/perf_gate.py --no-capacity         # skip the informational capacity-clips search
    uv run scripts/perf_gate.py --update-baseline --reason "clip effects add a pass"   # record a deliberate cost

Each scenario (`render-core/perf/scenarios/*.json`) runs `render-core --bench` (no window, offscreen
final pass, uncapped unless the scenario is paced) `--runs` times; per metric the best run (CPU stage /
GPU pass costs — noise only adds time) or the median (latency, counts) is compared with `render-core/perf/baselines/<machine-key>.json` (machine key = GPU adapter + hostname). Verdicts:

- FAIL (regression): a GPU pass p95 rises > 15 % AND > 0.5 ms (CPU stage: > 15 % AND > 1 ms — shared-machine
  noise), or a latency p95 rises > 2 ms, or stale clip frames rise (tracked scenarios).
- FAIL (absolute budget, regardless of baseline): see BUDGETS below.
- A failing scenario is re-run once before it fails. > 10 % spread across the runs (and > 0.3 ms) prints
  NOISY (several agents and cargo builds share this machine — GPU numbers are far steadier than CPU time).
- No baseline for this machine → budgets only.

Clip scenarios need 1080p HAP clips: generated once into `render-core/perf/.media/` (gitignored; ffmpeg
with the `hap` encoder). Reports + logs land in `render-core/perf/.work/`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RC = REPO / "render-core"
PERF = RC / "perf"
SCENARIOS = PERF / "scenarios"
BASELINES = PERF / "baselines"
MEDIA = PERF / ".media"
WORK = PERF / ".work"
ENGINE = RC / "target" / "release" / "render-core"
PACK = REPO / "test_results" / "layerpack" / "pack"

# --- Budgets (absolute; FAIL regardless of baseline). Proposals pending Xander's confirmation — ----------
# --- perf-monitor plan "Open questions". One place, on purpose. ------------------------------------------
BUDGET_GPU_TOTAL_P95_MS = 8.0    # uncapped bench, 1080p output: half a 60 Hz frame. Never judged at 60 Hz
                                 # (the Apple GPU downclocks when lightly loaded — paced scenarios skip it).
BUDGET_CPU_WORK_P95_MS = 4.0     # render-thread work (Σ stages − acquire − pace).
BUDGET_FRAME_MAX_MS = 33.0       # no frame's render-thread work > 2 vsyncs in a steady scenario. Work, not
                                 # interval: uncapped, the interval's `acquire` is GPU backpressure, and on
                                 # this shared desktop the GPU stalls ~45 ms now and then for other apps'
                                 # compositing (seen in ~1 run in 5) — GPU cost is judged by gpu.total.
BUDGET_STALE_FRAMES = 0          # clip frames wanted but not decoded in time ("stale": "zero" scenarios).
BUDGET_KNOB_SUBMIT_P95_MS = 36.0 # knob→submit. Not 17: legs tick *before* the vsync wait in acquire, so a
                                 # knob write waits for the next tick (≤ 1 frame) and then a whole vsync
                                 # (1 frame): ceiling 2 × 16.7 = 33.4 ms + sub-ms jitter. The bench measures
                                 # 33.3–34.7 ms p95 (6 runs, 2026-10-12), so 34 flapped; 36 = the ceiling +
                                 # jitter margin. Fixing the tick order is its own change (reference §4 #9);
                                 # the +2 ms regression rule catches growth below the budget.
BUDGET_RESOURCE_GROWTH = 0       # wgpu live objects (excl. command buffers) gained over edit-churn's window.
BUDGET_MONITOR_P95_MS = 0.1      # the frame profiler's own cost per frame.

# --- Regression + noise rules (plan §3.3) -------------------------------------------------------------------
REG_REL = 0.15        # stage/pass p95 must rise > 15 % …
REG_ABS_MS = 0.5      # … and > 0.5 ms (GPU passes), both, so tiny numbers never flap. The plan said 0.3;
                      # light scenes (clips-7, ~1 ms GPU) sit in two clock states 0.4 ms apart.
REG_ABS_CPU_MS = 1.0  # CPU stages need > 1 ms: render-thread p95s on a machine shared with other agents,
                      # Chrome and cargo swing 0.3–0.9 ms run to run even on the best of 3 (2026-10-12).
REG_LATENCY_MS = 2.0  # latency p95 rise that fails.
NOISE_SPREAD = 0.10   # (max − min) / median across runs above this prints NOISY …
NOISE_ABS_MS = 0.3    # … when the range is also big enough to flip a verdict (sub-ms CPU p95s jitter
                      # by 20–40 % on a shared machine without meaning anything).
STALE_TRACK_SLACK = (1.5, 10)  # tracked stale counts fail when > baseline × 1.5 AND > baseline + 10.
CAPACITY_BUDGET_MS = 1000.0 / 60.0

CLIP_FILES = 4
CLIP_ENCODE = "testsrc2=size=1920x1080:rate=30:duration=3,hue=h={hue},noise=alls=12:allf=t"


# ------------------------------------------------------------------------------------------------- helpers

def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def get(d, path: str):
    for k in path.split("/"):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v, unit: str = "ms") -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if unit == "n":
        return f"{v:.0f}"
    return f"{v:.2f}"


# ------------------------------------------------------------------------------------------------- prepare

def build() -> None:
    print("· cargo build --release (render-core)", flush=True)
    sh(["cargo", "build", "--release", "--quiet"], cwd=RC)


def ensure_pack() -> None:
    if (PACK / "pack.json").exists():
        return
    print("· building the example layer pack (test.py layerpack)", flush=True)
    sh(["uv", "run", "python", "test.py", "layerpack"], cwd=REPO)


def ensure_clips() -> None:
    """1080p 30 fps HAP clips with grain (≈1 MB/frame, so the decoder does real work), 3 s each."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    for i in range(CLIP_FILES):
        out = MEDIA / f"clip_{i}.mov"
        if out.exists() and out.stat().st_size > 0:
            continue
        print(f"· encoding {out.relative_to(REPO)}", flush=True)
        tmp = out.with_suffix(".tmp.mov")
        sh(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", CLIP_ENCODE.format(hue=i * 90),
            "-c:v", "hap", "-format", "hap", str(tmp)])
        tmp.rename(out)


# ------------------------------------------------------------------------------------------------- running

def run_bench(scenario: Path, tag: str, extra: list[str] | None = None) -> dict:
    out = WORK / "reports" / f"{tag}.json"
    log = WORK / "logs" / f"{tag}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.run([str(ENGINE), "--bench", str(scenario), "--bench-out", str(out), *(extra or [])],
                              stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "RUST_LOG": "info"})
    if proc.returncode != 0 or not out.exists():
        tail = log.read_text().splitlines()[-5:]
        raise RuntimeError(f"bench {tag} failed (exit {proc.returncode}); log {log}:\n  " + "\n  ".join(tail))
    return json.loads(out.read_text())


def metrics(report: dict) -> dict[str, float | None]:
    """Flatten one report into the gated metrics (ms unless named otherwise)."""
    p = report["perf"]
    m: dict[str, float | None] = {
        "gpu.total": get(p, "gpu/total/p95"),
        "cpu.work": get(p, "frame/work/p95"),
        "frame.max": get(p, "frame/work/max"),
        "monitor": get(p, "frame/monitor/p95"),
        "knob→submit": get(p, "latency/knob_submit/p95"),
        "stale": get(p, "clips/stale"),
        "decode": get(p, "clips/decode_p95"),
        "fps": p.get("fps"),
    }
    for stage, row in (p.get("cpu") or {}).items():
        m[f"cpu.{stage}"] = row.get("p95")
    for pas, row in (get(p, "gpu/passes") or {}).items():
        m[f"gpu.{pas}"] = row.get("p95")
    start, end = get(report, "resources/start"), get(report, "resources/end")
    if isinstance(start, dict) and isinstance(end, dict):
        m["resources.growth"] = sum(max(0, end.get(k, 0) - v) for k, v in start.items() if k != "command_buffers")
    errors = sum(a.get("errors", 0) for a in (report.get("actions") or {}).values())
    m["actions.errors"] = errors
    for method, a in (report.get("actions") or {}).items():
        m[f"rpc.{method}"] = a.get("p95_ms")
    return m


def best_cost(metric: str) -> bool:
    """CPU stage / GPU pass costs: noise from other processes only ever *adds* time, so the best run is
    the steadiest estimate of what the code costs (measured 2026-10-12: smoke cpu.work p95 0.15–0.39 ms
    run to run with other agents building; a median of 3 still flapped the +15 %/+0.3 ms rule)."""
    return metric.startswith(("cpu.", "gpu."))


def median_metrics(reports: list[dict]) -> tuple[dict, dict]:
    """Per metric: the best run for CPU/GPU costs, else the median; spread = (range / value, range)."""
    keys = {k for r in reports for k in metrics(r)}
    med, spread = {}, {}
    per = [metrics(r) for r in reports]
    for k in keys:
        vals = [m[k] for m in per if m.get(k) is not None]
        if not vals:
            med[k] = None
            continue
        med[k] = min(vals) if best_cost(k) else statistics.median(vals)
        rng = max(vals) - min(vals)
        spread[k] = (rng / med[k] if med[k] else 0.0, rng)
    return med, spread


# ------------------------------------------------------------------------------------------------- verdicts

class Row:
    def __init__(self, scenario, metric, base, now, verdict, note="", unit="ms", shown=True):
        self.scenario, self.metric, self.base, self.now = scenario, metric, base, now
        self.verdict, self.note, self.unit, self.shown = verdict, note, unit, shown

    @property
    def delta(self) -> str:
        if not isinstance(self.base, (int, float)) or not isinstance(self.now, (int, float)):
            return ""
        d = self.now - self.base
        if self.unit == "n":
            return f"{d:+.0f}"
        rel = f" ({d / self.base:+.0%})" if self.base else ""
        return f"{d:+.2f}{rel}"


def regressed(base, now, floor: float = REG_ABS_MS) -> bool:
    return base is not None and now is not None and now - base > floor and now > base * (1 + REG_REL)


def judge(name: str, sc: dict, med: dict, spread: dict, base: dict | None) -> list[Row]:
    gate = sc.get("gate") or {}
    paced = bool(sc.get("paced"))
    b = base or {}
    rows: list[Row] = []

    def headline(metric, label, budget=None, kind="cost", unit="ms", show=True):
        now, was = med.get(metric), b.get(metric)
        if now is None:
            return
        verdict, notes = "PASS" if base is not None else "NEW", []
        if budget is not None and now > budget:
            verdict = "FAIL"
            notes.append(f"budget {fmt(budget, unit)}")
        if base is not None and was is not None:
            floor = REG_ABS_CPU_MS if metric.startswith("cpu.") else REG_ABS_MS
            if kind == "cost" and regressed(was, now, floor):
                verdict = "FAIL"
                notes.append(f"> +{REG_REL:.0%} and > +{floor} ms")
            elif kind == "latency" and now - was > REG_LATENCY_MS:
                verdict = "FAIL"
                notes.append(f"> +{REG_LATENCY_MS:.0f} ms")
            elif kind == "count" and now > max(was * STALE_TRACK_SLACK[0], was + STALE_TRACK_SLACK[1]):
                verdict = "FAIL"
                notes.append("stale rose")
        rel, rng = spread.get(metric, (0.0, 0.0))
        if kind in ("cost", "latency") and rel > NOISE_SPREAD and rng > NOISE_ABS_MS:
            notes.append(f"NOISY (runs spread {rel:.0%})")
        rows.append(Row(name, label, was, now, verdict, ", ".join(notes), unit, show or verdict == "FAIL"))

    if gate.get("gpu", True) and not paced:
        headline("gpu.total", "gpu.total p95", BUDGET_GPU_TOTAL_P95_MS)
    if gate.get("cpu", True):
        headline("cpu.work", "cpu.work p95", BUDGET_CPU_WORK_P95_MS)
    if gate.get("steady"):
        headline("frame.max", "worst frame work", BUDGET_FRAME_MAX_MS, kind="info")
    if gate.get("latency"):
        headline("knob→submit", "knob→submit p95", BUDGET_KNOB_SUBMIT_P95_MS, kind="latency")
    if gate.get("stale"):
        zero = gate["stale"] in (True, "zero")
        headline("stale", "stale clip frames", BUDGET_STALE_FRAMES if zero else None,
                 kind="info" if zero else "count", unit="n")
        # Decode p95 swings with disk cache + core contention (±100 % run to run); the stale count is
        # what the audience sees, so decode is shown, never judged.
        headline("decode", "clip decode p95", kind="info")
    if gate.get("resources_flat"):
        headline("resources.growth", "wgpu objects gained", BUDGET_RESOURCE_GROWTH, kind="info", unit="n")
        for k in sorted(med):
            if k.startswith("rpc."):
                headline(k, f"{k[4:]} round-trip p95", kind="info")
    if gate.get("stale") == "track":
        for k in sorted(med):
            if k.startswith("rpc."):
                headline(k, f"{k[4:]} round-trip p95", kind="info")
    headline("monitor", "perf monitor p95", BUDGET_MONITOR_P95_MS, kind="info", show=False)
    if med.get("actions.errors"):
        rows.append(Row(name, "rpc errors", b.get("actions.errors"), med["actions.errors"], "FAIL",
                        "timeline actions failed — see the log", unit="n"))
    # Every CPU stage / GPU pass is regression-checked; only failures are shown.
    if base is not None:
        for k in sorted(med):
            if not (k.startswith("cpu.") or k.startswith("gpu.")) or k in ("cpu.work", "gpu.total"):
                continue
            if (k.startswith("gpu.") and (paced or not gate.get("gpu", True))) or \
               (k.startswith("cpu.") and (not gate.get("cpu", True) or k in ("cpu.acquire", "cpu.pace"))):
                continue
            floor = REG_ABS_CPU_MS if k.startswith("cpu.") else REG_ABS_MS
            if regressed(b.get(k), med[k], floor):
                rows.append(Row(name, f"{k} p95", b[k], med[k], "FAIL", f"> +{REG_REL:.0%} and > +{floor} ms"))
    return rows


# ------------------------------------------------------------------------------------------------- capacity

def capacity(sc_path: Path, sc: dict, runs_tag: str) -> tuple[int, str]:
    """Largest N clip bindings with headroom ≥ 0 and no stale frames (doubling, then bisect)."""
    template = json.loads((sc_path.parent / sc["scene"]).read_text())
    scene_dir = (sc_path.parent / sc["scene"]).resolve().parent
    proto = template["bindings"][0]
    layers = ["main_facade", "left_tower", "right_wing", "base_wall"]
    cap_dir = WORK / "capacity"
    cap_dir.mkdir(parents=True, exist_ok=True)
    limit = (sc.get("gate") or {}).get("capacity", {}).get("max", 64)
    tried: dict[int, tuple[bool, str]] = {}

    def ok(n: int) -> bool:
        if n in tried:
            return tried[n][0]
        binds = []
        for i in range(n):
            bnd = json.loads(json.dumps(proto))
            bnd["id"] = f"clip_{i:02d}"
            bnd["select"] = {"id": layers[i % len(layers)]}
            bnd["params"]["image"]["clip"] = str((MEDIA / f"clip_{i % CLIP_FILES}.mov").resolve())
            binds.append(bnd)
        scene = {**template, "pack": str((scene_dir / template["pack"]).resolve()), "bindings": binds}
        (cap_dir / f"clips-{n}.scene.json").write_text(json.dumps(scene, indent=1))
        scen = {k: v for k, v in sc.items() if k != "gate"}
        scen["scene"] = f"clips-{n}.scene.json"
        path = cap_dir / f"capacity-{n}.json"
        path.write_text(json.dumps(scen, indent=1))
        r = run_bench(path, f"{runs_tag}-capacity-{n}")
        m = metrics(r)
        busiest = max(m.get("gpu.total") or 0, m.get("cpu.work") or 0)
        good = busiest <= CAPACITY_BUDGET_MS and not m.get("stale")
        tried[n] = (good, f"{n} clips: gpu {fmt(m.get('gpu.total'))} / cpu {fmt(m.get('cpu.work'))} ms, stale {m.get('stale')}")
        return good

    n, last_ok = 1, 0
    while n <= limit and ok(n):
        last_ok, n = n, n * 2
    hi = min(n, limit + 1)
    lo = last_ok
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    first_bad = tried.get(hi, (None, ""))[1]
    return lo, first_bad


# ------------------------------------------------------------------------------------------------- baseline

def machine_key(report: dict) -> str:
    adapter = get(report, "machine/adapter") or "unknown-gpu"
    return f"{slug(adapter)}__{slug(socket.gethostname().split('.')[0])}"


def load_baseline(key: str) -> dict | None:
    path = BASELINES / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_baseline(key: str, results: dict, first: dict, reason: str, old: dict | None) -> Path:
    BASELINES.mkdir(parents=True, exist_ok=True)
    path = BASELINES / f"{key}.json"
    sha = git("rev-parse", "--short=12", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    doc = {
        "schema": 1,
        "machine": {**(first.get("machine") or {}), "hostname": socket.gethostname().split(".")[0]},
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "git": {"sha": sha, "dirty": dirty},
        "reason": reason,
        "scenarios": {},
    }
    if old:  # keep scenarios this run didn't touch
        doc["scenarios"].update(old.get("scenarios", {}))
    for name, (med, _spread, _sc) in results.items():
        doc["scenarios"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(med.items())}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    deltas = {}
    for name, (med, _s, _sc) in results.items():
        was = (old or {}).get("scenarios", {}).get(name, {})
        for k in ("gpu.total", "cpu.work", "knob→submit", "stale", "decode", "resources.growth"):
            if med.get(k) is not None and was.get(k) != med.get(k):
                deltas.setdefault(name, {})[k] = [was.get(k), round(med[k], 4)]
    with open(BASELINES / "history.jsonl", "a") as fh:
        fh.write(json.dumps({"date": dt.date.today().isoformat(), "machine": key, "sha": sha, "dirty": dirty,
                             "reason": reason, "deltas": deltas}, ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------------------------------------------------- main

def print_table(rows: list[Row], header: str, footer: str) -> None:
    shown = [r for r in rows if r.shown]
    cols = ["scenario", "metric", "baseline", "now", "Δ", "verdict"]
    data = [[r.scenario, r.metric, fmt(r.base, r.unit), fmt(r.now, r.unit), r.delta,
             r.verdict + (f" — {r.note}" if r.note else "")] for r in shown]
    widths = [max(len(c), *(len(d[i]) for d in data)) if data else len(c) for i, c in enumerate(cols)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    print("\n" + header)
    print(line(cols))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    prev = None
    for d in data:
        print(line([d[0] if d[0] != prev else ""] + d[1:]))
        prev = d[0]
    print(footer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--runs", type=int, default=3,
                    help="runs per scenario (baseline and check must use the same count: costs take the best run)")
    ap.add_argument("--no-build", action="store_true", help="skip cargo build --release")
    ap.add_argument("--no-capacity", action="store_true", help="skip the informational capacity-clips search")
    ap.add_argument("--update-baseline", action="store_true", help="write this run as the machine's baseline")
    ap.add_argument("--reason", default="", help="why the baseline moves (required with --update-baseline)")
    ap.add_argument("--json", type=Path, help="also write the verdict rows here")
    args = ap.parse_args()
    if args.update_baseline and not args.reason.strip():
        ap.error("--update-baseline needs --reason \"…\" (it lands in perf/baselines/history.jsonl)")

    t0 = time.monotonic()
    if not args.no_build:
        build()
    ensure_pack()
    files = sorted(SCENARIOS.glob("*.json"))
    if args.scenarios:
        # A path to a scenario file runs it under its file stem — a scratch copy named `smoke.json` is judged
        # against smoke's baseline (how the gate is proven red on a deliberately slow shader).
        paths = [Path(a).resolve() for a in args.scenarios if a.endswith(".json") and Path(a).exists()]
        wanted = {a for a in args.scenarios if not a.endswith(".json")}
        files = [f for f in files if f.stem in wanted] + paths
        missing = wanted - {f.stem for f in files}
        if missing:
            ap.error(f"unknown scenario(s): {', '.join(sorted(missing))}")
    scen = {f.stem: (f, json.loads(f.read_text())) for f in files}
    if any((sc.get("gate") or {}).get("media") == "clips" for _, sc in scen.values()):
        ensure_clips()
    stamp = dt.datetime.now().strftime("%H%M%S")

    results: dict[str, tuple[dict, dict, dict]] = {}
    first_report: dict = {}
    key, base_doc = None, None
    rows: list[Row] = []
    info: list[str] = []
    for name, (path, sc) in scen.items():
        gate = sc.get("gate") or {}
        if "capacity" in gate:
            if args.no_capacity:
                continue
            print(f"· {name} (search)", flush=True)
            n, why = capacity(path, sc, f"{stamp}-{name}")
            limit = gate["capacity"].get("max", 64)
            held = f"≥ {n} (the search cap)" if n >= limit else str(n)
            info.append(f"capacity-clips (info): this machine holds {held} 1080p HAP clip bindings with headroom ≥ 0 "
                        f"and no stale frames ({CLIP_FILES} files reused, so mostly page-cache I/O)"
                        + (f" — first miss: {why}" if why and n < limit else ""))
            continue
        runs = int(gate.get("runs", args.runs))
        for attempt in (1, 2):
            print(f"· {name} ×{runs}" + (" (re-run)" if attempt == 2 else ""), flush=True)
            reports = [run_bench(path, f"{stamp}-{name}-{attempt}{i}") for i in range(runs)]
            if not first_report:
                first_report = reports[0]
                key = machine_key(first_report)
                base_doc = load_baseline(key)
            base = (base_doc or {}).get("scenarios", {}).get(name)
            med, spread = median_metrics(reports)
            srows = judge(name, sc, med, spread, base)
            if attempt == 1 and any(r.verdict == "FAIL" for r in srows) and not args.update_baseline:
                failed = ", ".join(f"{r.metric} {fmt(r.base, r.unit)}→{fmt(r.now, r.unit)}" for r in srows if r.verdict == "FAIL")
                print(f"  ! {failed} — re-running once (noise rule)", flush=True)
                continue
            break
        results[name] = (med, spread, sc)
        rows.extend(srows)

    wall = time.monotonic() - t0
    fails = [r for r in rows if r.verdict == "FAIL"]
    sha = git("rev-parse", "--short=12", "HEAD")
    dirty = " (dirty)" if git("status", "--porcelain", "--untracked-files=no") else ""
    adapter = get(first_report, "machine/adapter") or "?"
    base_note = (f"baseline {key}.json @ {base_doc.get('git', {}).get('sha', '?')}" if base_doc
                 else f"no baseline for {key} — absolute budgets only")
    header = (f"**perf gate** — {adapter}, {sha}{dirty}, {base_note}; {args.runs} runs (costs: best run; latency/counts: median), "
              f"uncapped unless paced, GPU timed 1 frame in 4 (isolated on the GPU)")
    footer_lines = info + [f"{'FAIL' if fails else 'PASS'}: {len(fails)} failing row(s); gate wall time {wall:.0f} s"]
    print_table(rows, header, "\n".join(footer_lines))
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in rows], indent=1, default=str))
    if args.update_baseline and key:
        path = write_baseline(key, results, first_report, args.reason.strip(), base_doc)
        print(f"baseline written: {path.relative_to(REPO)} (+ history.jsonl)")
        return 0
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
