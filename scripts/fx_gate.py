"""Gate clip effects (gig F1) on the real engine: probe, fps, flash rate, preview.

For each effect in `library/effects/` (or the names given):

- **probe** — one engine; `effect.upsert` (naga) → `scene.load` binding it on
  the design leg → `effect.upsert` again, whose reply carries the pre-flight
  probe verdict + thumbnail (docs/reference/render-engine.md §2.6/§2.7).
- **run** — per effect a cold boot of `--bench` with the effect on one clip per
  layer (≤ 7 layers, clips round-robin from its `generated/*.mov`): `perf`
  frame-interval p95 over 10 s (live leg only, then with the design leg awake
  via a `preview` subscription), then an 8 s 60 fps screen recording of the
  engine window → `wzrd.check_frames` flash rate.
- **preview** — per effect a cold boot of a one-layer canvas scene playing
  `--preview-clip`, window screenshot → `library/effects/<name>/preview.jpg`.

The engine window must be VISIBLE and IN FRONT for run/preview: macOS throttles
a covered window's swapchain to ~30 fps, and the recording is a screen capture
(the `preview` channel's ~15 fps aliases strobes — docs/reference/effects-library.md).
Nobody should use the Mac while it runs. Needs a release build
(`cargo build --release` in render-core/), ffmpeg, swiftc, and the terminal's
Screen Recording permission.

    uv run python scripts/fx_gate.py probe   --bench projects/<p>
    uv run python scripts/fx_gate.py run     --bench projects/<p> [kaleido mirror ...]
    uv run python scripts/fx_gate.py preview --preview-clip projects/<p>/generated/x.mov

Results: `<out>/{probe,run}.json`, recordings `<out>/rec_<fx>.mp4` (default out:
test_results/fx_gate/). The bench project's scene.json/session.json and
effects/ are overwritten — point it at a scratch copy, never a show project.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "library" / "effects"
ENGINE = REPO / "render-core" / "target" / "release" / "render-core"
# The engine's own default filter (render-core/src/main.rs). Never plain `info`: wgpu-core logs "Device::maintain:
# waiting for submission index N" at INFO on every maintain (~180 lines/s), all of it written to stdout and fanned
# into the `log` telemetry channel — noise that perturbs the very timings a gate measures.
ENGINE_RUST_LOG = "info,wgpu_core=warn,wgpu_hal=warn,naga=warn"
PORT = 9177
MAX_CLIPS = 7  # the F1 gate: seven clips, each with the effect


def winbounds_bin() -> Path:
    src = REPO / "scripts" / "winbounds.swift"
    out = Path(tempfile.gettempdir()) / "wzrd_winbounds"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["swiftc", "-O", str(src), "-o", str(out)], check=True)
    return out


def bench_scene(project: Path, fx: str) -> dict:
    pack_json = next((project / "packs").glob("*/pack.json"))
    pack, pack_dir = json.loads(pack_json.read_text()), pack_json.parent.name
    clips = sorted((project / "generated").glob("*.mov"))
    if not clips:
        raise SystemExit(f"{project}/generated has no .mov clips")
    binds = []
    for i, layer in enumerate(pack["layers"][:MAX_CLIPS]):
        clip = clips[i % len(clips)].relative_to(project).as_posix()
        binds.append({"id": f"clip_{layer['id']}", "select": {"id": layer["id"]}, "effect": fx,
                      "params": {"image": {"clip": clip}}})
    return {"version": 1, "pack": f"packs/{pack_dir}", "transport": {"bpm": 120}, "bindings": binds, "post": []}


def projector_res(project: Path) -> tuple[int, int]:
    pack = json.loads(next((project / "packs").glob("*/pack.json")).read_text())
    w, h = pack["projector_resolution"]
    return w, h


def make_preview_project(clip: Path, root: Path) -> Path:
    """One full-frame layer at 1024×768, the clip placed on the canvas."""
    from PIL import Image

    proj = root / "fx_preview"
    shutil.rmtree(proj, ignore_errors=True)
    (proj / "packs" / "full" / "masks").mkdir(parents=True)
    (proj / "generated").mkdir()
    (proj / "effects").mkdir()
    Image.new("L", (1024, 768), 255).save(proj / "packs" / "full" / "masks" / "000_full.png")
    (proj / "packs" / "full" / "pack.json").write_text(json.dumps({
        "version": 1, "projector_resolution": [1024, 768], "source_capture": None, "surface": None,
        "groups": [], "layers": [{"id": "full", "mask": "masks/000_full.png", "label": "full",
                                  "tags": ["full"], "bbox": [0, 0, 1024, 768], "source_box": [0, 0, 1024, 768],
                                  "centroid": [512, 384], "area_px": 786432, "parent": None, "z": 0}]}))
    os.symlink(clip.resolve(), proj / "generated" / "preview.mov")
    return proj


def preview_scene(fx: str) -> dict:
    return {"version": 1, "pack": "packs/full", "transport": {"bpm": 120}, "post": [],
            "bindings": [{"id": "clip", "select": {"id": "full"}, "effect": fx,
                          "params": {"image": {"clip": "generated/preview.mov", "placement": "canvas"}}}]}


class Engine:
    def __init__(self, project: Path, scene: dict, log: Path):
        (project / "scene.json").write_text(json.dumps(scene, indent=1))
        (project / "session.json").unlink(missing_ok=True)
        self.log = open(log, "w")
        self.p = subprocess.Popen(
            [str(ENGINE), "--scene", str(project / "scene.json"), "--windowed", "--no-osc",
             "--ws-addr", f"127.0.0.1:{PORT}"],
            stdout=self.log, stderr=subprocess.STDOUT, env={**os.environ, "RUST_LOG": ENGINE_RUST_LOG})

    def stop(self):
        self.p.send_signal(signal.SIGINT)
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.log.close()
        time.sleep(0.5)

    def front_window(self) -> tuple[int, int, int, int, int]:
        """Activate the engine and return its largest normal window (id, x, y, w, h) in points."""
        out = subprocess.run([str(winbounds_bin()), str(self.p.pid), "--front"],
                             capture_output=True, text=True).stdout
        wins = [tuple(int(float(v)) for v in line.split()) for line in out.splitlines()]
        wins = [w[:5] for w in wins if w[5] == 0]
        if not wins:
            raise RuntimeError("engine window not on screen")
        return max(wins, key=lambda w: w[3] * w[4])


def content_rect(win, res):
    """Drop the title bar: the content is the bottom w × (w · h/w_res) points."""
    wid, x, y, w, h = win
    ch = round(w * res[1] / res[0])
    return wid, x, y + h - ch, w, ch


class Rpc:
    def __init__(self, ws):
        self.ws, self.id = ws, 0

    async def call(self, method, params=None, timeout=30.0):
        self.id += 1
        rid = self.id
        await self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}))
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = json.loads(await asyncio.wait_for(self.ws.recv(), timeout))
            if m.get("id") == rid:
                return {"__error": m["error"]} if "error" in m else m.get("result")
        raise TimeoutError(method)

    async def collect(self, secs: float, channel: str) -> list[dict]:
        out, t0 = [], time.time()
        while time.time() - t0 < secs:
            try:
                m = json.loads(await asyncio.wait_for(self.ws.recv(), 0.25))
            except asyncio.TimeoutError:
                continue
            if m.get("method") == "telemetry.event" and m["params"]["channel"] == channel:
                out.append(m["params"]["payload"])
        return out


async def attach(warm_s: float) -> Rpc:
    for _ in range(100):
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{PORT}", max_size=64 << 20)
            break
        except OSError:
            await asyncio.sleep(0.2)
    else:
        raise RuntimeError("engine WS never came up")
    r = Rpc(ws)
    await r.call("hello", {"actor": "agent"})
    await asyncio.sleep(warm_s)
    return r


def perf_summary(rows: list[dict]) -> dict:
    """Worst 2 Hz aggregate in the window (§2.9 `perf` payload)."""
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "interval_p95": round(max(p["frame"]["interval"]["p95"] for p in rows), 2),
        "interval_p50": round(max(p["frame"]["interval"]["p50"] for p in rows), 2),
        "work_p95": round(max(p["frame"]["work"]["p95"] for p in rows), 2),
        "headroom_min": round(min(p["headroom"] for p in rows), 3),
        "dropped": rows[-1]["dropped"]["total"] - rows[0]["dropped"]["total"],
        "stale": sum(c.get("stale", 0) for p in rows for c in p.get("clips", [])),
    }


def record(rect, secs: float, path: Path):
    """60 fps avfoundation capture of the content rect (retina ×2), scaled to 480 wide."""
    _, x, y, w, h = rect
    s = 2
    vf = f"crop={w * s}:{h * s}:{x * s}:{y * s},scale=480:-2,format=yuv420p"
    enc = ["-c:v", "h264_videotoolbox", "-b:v", "12M"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "avfoundation", "-framerate", "60",
                    "-capture_cursor", "0", "-i", "1:none", "-t", str(secs), "-vf", vf, "-r", "60",
                    *enc, str(path)], check=True)


async def do_probe(bench: Path, fxs: list[str], out: Path):
    (bench / "effects").mkdir(exist_ok=True)
    e = Engine(bench, bench_scene(bench, "videoClip"), out / "probe.log")
    res = {}
    try:
        r = await attach(2)
        for fx in fxs:
            wgsl = (LIB / fx / "shader.wgsl").read_text()
            desc = json.loads((LIB / fx / "descriptor.json").read_text())
            up1 = await r.call("effect.upsert", {"name": fx, "wgsl": wgsl, "descriptor": desc})
            st = await r.call("scene.getState", {"leg": "design"})
            await r.call("scene.load", {"json": json.dumps(bench_scene(bench, fx)), "base_rev": st.get("rev")})
            # A whitespace edit = a new source hash = a new pipeline, so this
            # re-apply runs the probe and the upsert reply carries the verdict.
            up2 = await r.call("effect.upsert", {"name": fx, "wgsl": wgsl + "\n", "descriptor": desc})
            probe = (up2 or {}).get("probe") or {}
            for v in probe.get("verdicts", []):
                thumb = v.pop("thumbnail_b64", None)
                if thumb:
                    (out / f"probe_{fx}.jpg").write_bytes(base64.b64decode(thumb))
            res[fx] = {"naga_ok": bool(up1 and up1.get("ok")), "band": probe.get("band"),
                       "compiled": probe.get("compiled"),
                       "predicted_p95_ms": round(probe.get("predicted_p95_ms") or 0, 2)}
            print(fx, res[fx], flush=True)
            st = await r.call("scene.getState", {"leg": "design"})
            await r.call("scene.load", {"json": json.dumps(bench_scene(bench, "videoClip")), "base_rev": st.get("rev")})
    finally:
        e.stop()
    (out / "probe.json").write_text(json.dumps(res, indent=1))


async def do_run(bench: Path, fxs: list[str], out: Path):
    from wzrd.check_frames import check_flash, _video_frames

    (bench / "effects").mkdir(exist_ok=True)
    res_path = out / "run.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for fx in fxs:
        if fx != "videoClip":
            shutil.rmtree(bench / "effects" / fx, ignore_errors=True)
            shutil.copytree(LIB / fx, bench / "effects" / fx)
        e = Engine(bench, bench_scene(bench, fx), out / f"run_{fx}.log")
        try:
            r = await attach(1)
            win = e.front_window()
            await asyncio.sleep(4)
            await r.call("telemetry.subscribe", {"channels": ["perf"]})
            live = perf_summary((await r.collect(10, "perf"))[1:])
            # One subscribe call per connection REPLACES the channel set.
            await r.call("telemetry.subscribe", {"channels": ["perf", "preview"]})
            await r.collect(3, "perf")
            both = perf_summary((await r.collect(8, "perf"))[1:])
            await r.call("telemetry.subscribe", {"channels": []})
            mp4 = out / f"rec_{fx}.mp4"
            record(content_rect(e.front_window(), projector_res(bench)), 8.0, mp4)
            frames, fps = _video_frames(mp4)
            flash = check_flash(frames, fps)
            res[fx] = {"live": live, "live+design": both, "flash": flash,
                       "rec": {"frames": len(frames), "fps": round(fps, 2), "window": list(win)}}
            print(fx, json.dumps(res[fx]), flush=True)
        finally:
            e.stop()
        res_path.write_text(json.dumps(res, indent=1))


async def do_preview(clip: Path, fxs: list[str], out: Path):
    proj = make_preview_project(clip, out)
    for fx in fxs:
        shutil.copytree(LIB / fx, proj / "effects" / fx, dirs_exist_ok=True)
        e = Engine(proj, preview_scene(fx), out / f"preview_{fx}.log")
        try:
            await attach(1)
            e.front_window()
            await asyncio.sleep(3)
            wid = e.front_window()[0]
            png = out / f"preview_{fx}.png"
            subprocess.run(["screencapture", "-x", "-o", f"-l{wid}", str(png)], check=True)
            # -l grabs the window incl. its title bar: keep the bottom 4:3 content.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(png), "-vf",
                            "crop=iw:iw*3/4:0:ih-iw*3/4,scale=640:-2", "-q:v", "3",
                            str(LIB / fx / "preview.jpg")], check=True)
            print(fx, "preview.jpg written", flush=True)
        finally:
            e.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probe", "run", "preview"])
    ap.add_argument("effects", nargs="*", help="effect names (default: all of library/effects; "
                                               "`run` also accepts videoClip as a baseline)")
    ap.add_argument("--bench", type=Path, help="scratch project: packs/<p>/ + generated/*.mov")
    ap.add_argument("--preview-clip", type=Path, help="HAP clip for preview.jpg")
    ap.add_argument("--out", type=Path, default=REPO / "test_results" / "fx_gate")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fxs = a.effects or sorted(p.name for p in LIB.iterdir() if (p / "shader.wgsl").exists())
    if a.cmd == "preview":
        if not a.preview_clip:
            ap.error("preview needs --preview-clip")
        asyncio.run(do_preview(a.preview_clip, fxs, a.out))
        return
    if not a.bench:
        ap.error(f"{a.cmd} needs --bench")
    asyncio.run((do_probe if a.cmd == "probe" else do_run)(a.bench.resolve(), fxs, a.out))


if __name__ == "__main__":
    main()
