"""Build two throwaway projects for testing the app's project switcher (A1)
and clip grid / decks (A2, D1) without the MCP: `<root>/fixture-a` and `fixture-b`.

Each is a render-core scene directory (`scene.json` + `packs/main/`, the
example pack from `python test.py layerpack`) with a few clips in
`generated/` shaped like real hub renders (Studio contracts L2/L3): a
`<stem>.mp4` master and its HAP `<stem>.mov` beside it, both carrying the
embedded `wzrd` tag with one shared render id, and registered in the media
library's `library.db` (rows + a cached thumbnail; `generated/` is added as a
library root). No sidecars. One clip has no region (the "unassigned" row),
one has no thumbnail (placeholder). Every
library clip effect (`library/effects/`, gig plan C6) is copied into the
project's `effects/` — a copy, never a link, as a real project gets them —
plus the generative example effects (`render-core/examples/effects/`) the
scenes bind one per region.

    uv run python wzrd-app/scripts/fixture_projects.py              # (re)build both
    uv run python wzrd-app/scripts/fixture_projects.py --add-clip fixture-a --region left_tower
                                                                     # simulate an agent render landing
                                                                     # (in the library's target folder / inbox/)

Needs ffmpeg with the `hap` encoder. Projects root: `WZRD_PROJECTS_ROOT`,
else `<repo>/projects` (gitignored). Library: `WZRD_MEDIA_ROOT`, else
`~/WZRD/media` — writes go straight to its database, so run it while no hub
is mid-write (WAL makes a running hub safe).
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
sys.path.insert(0, str(REPO))
from wzrd_mcp import identity, library  # noqa: E402
PACK_SRC = REPO / "test_results" / "layerpack" / "pack"
LIBRARY_FX = REPO / "library" / "effects"
EXAMPLE_FX = REPO / "render-core" / "examples" / "effects"
H264 = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"

SCENES = {
    "fixture-a": {
        "bpm": 120,
        # One generative shader per region (render-core/examples/effects);
        # background stays dark — additive, unchanged areas project nothing.
        "bindings": [
            {"id": "facade_silk", "select": {"id": "main_facade"}, "effect": "silk",
             "params": {"scale": 2.2, "speed": 1, "bright": 1.2,
                        "colorA": "#3a5cff", "colorB": "#ff4fd8"}},
            {"id": "tower_embers", "select": {"id": "left_tower"}, "effect": "embers",
             "params": {"density": 9, "rise": 1, "bright": 1.3,
                        "hot": "#ffe2a0", "cool": "#ff3a0a"}},
            {"id": "wing_contours", "select": {"id": "right_wing"}, "effect": "contours",
             "params": {"scale": 2.5, "lines": 18, "speed": 1, "bright": 1.2,
                        "line": "#2ee6c8", "sweep": "#e8fff9"}},
            {"id": "wall_scanwave", "select": {"id": "base_wall"}, "effect": "scanwave",
             "params": {"count": 5, "amp": 0.5, "speed": 1, "bright": 1.1,
                        "colorA": "#7cff4f", "colorB": "#22d3ff"}},
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
        # Same shaders, re-cast, so the project switcher visibly changes.
        "bindings": [
            {"id": "facade_contours", "select": {"id": "main_facade"}, "effect": "contours",
             "params": {"scale": 1.6, "lines": 18, "speed": 0.6, "bright": 1.1,
                        "line": "#ffb347", "sweep": "#fff4e0"}},
            {"id": "towers_silk", "select": {"group": "towers"}, "effect": "silk",
             "params": {"scale": 3.0, "speed": 0.7, "bright": 1.0,
                        "colorA": "#00c2ff", "colorB": "#7a2cff"}},
            {"id": "wall_embers", "select": {"id": "base_wall"}, "effect": "embers",
             "params": {"density": 22, "rise": 0.6, "bright": 1.2,
                        "hot": "#d6f7ff", "cool": "#2a7bff"}},
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


def make_clip(project: Path, stem: str, region: str | None, hue: int, thumb: bool,
              folder: Path | None = None) -> None:
    gen = folder or project / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    box = layer_boxes(project).get(region) if region else None
    # Rendered at the region's box size (D4: clips stay at rendered
    # resolution); HAP needs multiples of 4.
    w, h = ((box[2] - box[0]), (box[3] - box[1])) if box else (1024, 576)
    w, h = max(64, w // 4 * 4), max(64, h // 4 * 4)
    mp4, mov = gen / f"{stem}.mp4", gen / f"{stem}.mov"
    src = f"testsrc2=size={w}x{h}:rate=30:duration=4,hue=h={hue}"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-c:v", H264, "-b:v", "4M", "-pix_fmt", "yuv420p", str(mp4)])
    run(["ffmpeg", "-y", "-i", str(mp4), "-c:v", "hap", "-format", "hap", str(mov)])
    record = {
        "schema": identity.TAG_SCHEMA, "id": identity.new_id(), "variant": "master",
        "tool": "texture_flow", "args": {"seed": hue, "n_seconds": 4, "width": w, "height": h},
        "inputs": [], "parent": None,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "est_usd": 0.12, "duration_s": 30.0, "job": None, "project": project.name,
        "canvas": None, "region": region, "source_box": box,
    }
    for path, rec in ((mp4, record), (mov, {**record, "variant": "hap"})):
        identity.write_tag(path, rec)
        library.index_file(path, record=rec)
    if not thumb:  # the placeholder case
        library.thumb_path(record["id"]).unlink(missing_ok=True)
        with library.tx() as c:
            c.execute("UPDATE renders SET thumb=NULL WHERE id=?", (record["id"],))
    print(f"  {stem}: {w}x{h} region={region} thumb={thumb} id={record['id']}")


def build(name: str, spec: dict) -> None:
    project = projects_root() / name
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(PACK_SRC, project / "packs" / "main")
    shutil.copytree(LIBRARY_FX, project / "effects")
    shutil.copytree(EXAMPLE_FX, project / "effects", dirs_exist_ok=True)
    scene = {"version": 1, "pack": "packs/main", "transport": {"bpm": spec["bpm"]},
             "bindings": spec["bindings"], "post": []}
    (project / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"{project}")
    (project / "generated").mkdir()
    library.add_root(str(project / "generated"))
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
        make_clip(project, stem, args.region, int(time.time()) % 360, True, folder=library.target_dir())
        return
    for name, spec in SCENES.items():
        build(name, spec)


if __name__ == "__main__":
    main()
