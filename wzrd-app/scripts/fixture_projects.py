"""Build two throwaway projects for testing the app's project switcher (A1)
and clip grid (A2) without the MCP: `<root>/fixture-a` and `fixture-b`.

Each is a render-core scene directory (`scene.json` + `packs/main/`, the
example pack from `python test.py layerpack`) with a few clips in
`generated/` following the gig plan's contracts: C3 clip files
(`<stem>.mp4`, HAP `<stem>.mov`, `<stem>.thumb.jpg`) and C2 provenance
sidecars (`<stem>.mp4.meta.json`, `<stem>.mov.meta.json`). One clip has no
region (the "unassigned" row), one has no thumbnail (placeholder). Every
library clip effect (`library/effects/`, gig plan C6) is copied into the
project's `effects/` — a copy, never a link, as a real project gets them.

    uv run python wzrd-app/scripts/fixture_projects.py              # (re)build both
    uv run python wzrd-app/scripts/fixture_projects.py --add-clip fixture-a --region left_tower
                                                                     # simulate an agent render landing

Needs ffmpeg with the `hap` encoder. Projects root: `WZRD_PROJECTS_ROOT`,
else `<repo>/projects` (gitignored).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACK_SRC = REPO / "test_results" / "layerpack" / "pack"
LIBRARY_FX = REPO / "library" / "effects"
H264 = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"

SCENES = {
    "fixture-a": {
        "bpm": 120,
        "bindings": [
            {"id": "facade_hue", "select": {"id": "main_facade"}, "effect": "hueCycle",
             "params": {"phase": {"driver": "clock.bars", "n": 4}}},
            {"id": "tower_tint", "select": {"id": "left_tower"}, "effect": "tint",
             "params": {"color": "#ff6a00"}},
        ],
        # (stem, region, hue shift, thumb?)
        "clips": [
            ("tf_a01", "main_facade", 0, True),
            ("tf_a02", "main_facade", 120, True),
            ("tf_a03", "left_tower", 240, True),
            ("tf_a04", None, 60, True),
            ("tf_a05", "base_wall", 300, False),
        ],
    },
    "fixture-b": {
        "bpm": 96,
        "bindings": [
            {"id": "wing_tint", "select": {"id": "right_wing"}, "effect": "tint",
             "params": {"color": "#2a7bff"}},
        ],
        "clips": [
            ("tf_b01", "right_wing", 180, True),
            ("tf_b02", "base_wall", 90, True),
        ],
    },
}


def projects_root() -> Path:
    return Path(os.environ.get("WZRD_PROJECTS_ROOT", REPO / "projects"))


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def layer_boxes(project: Path) -> dict[str, list[int]]:
    pack = json.loads((project / "packs" / "main" / "pack.json").read_text())
    return {l["id"]: l.get("source_box") or l["bbox"] for l in pack["layers"]}


def make_clip(project: Path, stem: str, region: str | None, hue: int, thumb: bool) -> None:
    gen = project / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    box = layer_boxes(project).get(region) if region else None
    # Rendered at the region's box size (D4: clips stay at rendered
    # resolution); HAP needs multiples of 4.
    w, h = ((box[2] - box[0]), (box[3] - box[1])) if box else (1024, 576)
    w, h = max(64, w // 4 * 4), max(64, h // 4 * 4)
    mp4, mov, jpg = gen / f"{stem}.mp4", gen / f"{stem}.mov", gen / f"{stem}.thumb.jpg"
    src = f"testsrc2=size={w}x{h}:rate=30:duration=4,hue=h={hue}"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-c:v", H264, "-b:v", "4M", "-pix_fmt", "yuv420p", str(mp4)])
    run(["ffmpeg", "-y", "-i", str(mp4), "-c:v", "hap", "-format", "hap", str(mov)])
    if thumb:
        run(["ffmpeg", "-y", "-ss", "2", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=320:-2", str(jpg)])
    else:
        jpg.unlink(missing_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    common = {"region": region, "source_box": box}
    (gen / f"{stem}.mp4.meta.json").write_text(json.dumps({
        "schema": 1, "tool": "texture_flow",
        "args": {"prompt": f"fixture clip {stem}: test pattern, hue {hue}", "seed": hue, "region": region},
        "inputs": [], "parent": None, "created": now, "est_usd": 0.12, **common,
    }, indent=2))
    (gen / f"{stem}.mov.meta.json").write_text(json.dumps({
        "schema": 1, "tool": "to_hap", "args": {"codec": "hap"},
        "inputs": [str(mp4)], "parent": str(mp4), "created": now, "est_usd": 0, **common,
    }, indent=2))
    print(f"  {stem}: {w}x{h} region={region} thumb={thumb}")


def build(name: str, spec: dict) -> None:
    project = projects_root() / name
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(PACK_SRC, project / "packs" / "main")
    shutil.copytree(LIBRARY_FX, project / "effects")
    scene = {"version": 1, "pack": "packs/main", "transport": {"bpm": spec["bpm"]},
             "bindings": spec["bindings"], "post": []}
    (project / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"{project}")
    for stem, region, hue, thumb in spec["clips"]:
        make_clip(project, stem, region, hue, thumb)
        time.sleep(1.1)  # distinct `created` seconds → stable newest-first order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add-clip", metavar="PROJECT", help="add one fresh clip to an existing fixture project")
    ap.add_argument("--region", default=None, help="region for --add-clip (omit = unassigned)")
    args = ap.parse_args()
    if not (PACK_SRC / "pack.json").exists():
        sys.exit(f"missing {PACK_SRC} — run `python test.py layerpack` first")
    if args.add_clip:
        project = projects_root() / args.add_clip
        stem = "tf_" + dt.datetime.now().strftime("%H%M%S")
        make_clip(project, stem, args.region, int(time.time()) % 360, True)
        return
    for name, spec in SCENES.items():
        build(name, spec)


if __name__ == "__main__":
    main()
