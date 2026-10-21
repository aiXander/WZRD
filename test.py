#!/usr/bin/env python3
"""Smoke-test script for WZRD submodules."""

import argparse
import os
import shutil
import sys
import time
import traceback

# ── Test asset input paths ──────────────
TEST_DAY_IMAGE = "test_imgs/day.jpg"
TEST_NIGHT_IMAGE = "test_imgs/night.jpg"

TEST_SEGMAP = "test_imgs/segmap.jpeg"
# The photo `segmap.jpeg` was segmented from (same framing, 2x its size) —
# the layer pack's surface. Not `day.jpg`: that's a different building.
TEST_SEGMAP_PHOTO = "test_imgs/aligned.jpg"

TEST_VIDEO = "test_imgs/test_video/video.mp4"
TEST_BACKGROUND = "test_imgs/test_video/background.jpg"

RESULTS_ROOT = "test_results"


# ── Per-module test functions ───────────────────────────────────────────────

def test_darken():
    from wzrd import darken
    _require(TEST_DAY_IMAGE)
    out_dir = _result_dir("darken")
    _copy_input(TEST_DAY_IMAGE, out_dir)
    result = darken.darken_image_file(TEST_DAY_IMAGE, output_path=os.path.join(out_dir, "output.jpg"))
    assert result['image'] is not None, "darken returned None image"


def test_subtract_video():
    from wzrd import subtract_video
    _require(TEST_VIDEO, TEST_BACKGROUND)
    out_dir = _result_dir("subtract_video")
    _copy_input(TEST_VIDEO, out_dir)
    _copy_input(TEST_BACKGROUND, out_dir)
    out = os.path.join(out_dir, "output.mp4")
    info = subtract_video.subtract_background_video(TEST_VIDEO, TEST_BACKGROUND, output_path=out)
    assert os.path.isfile(out), f"output video not created: {out}"


def test_islands():
    from wzrd import islands
    _require(TEST_SEGMAP)
    out_dir = _result_dir("islands")
    _copy_input(TEST_SEGMAP, out_dir)
    regions_dir = os.path.join(out_dir, "regions")
    regions, _ = islands.extract_color_regions(TEST_SEGMAP, regions_dir)
    assert isinstance(regions, list), "islands did not return a list"


def test_layerpack():
    from wzrd import islands, layerpack
    from PIL import Image
    import json
    _require(TEST_SEGMAP, TEST_SEGMAP_PHOTO)
    out_dir = _result_dir("layerpack")
    _copy_input(TEST_SEGMAP, out_dir)
    # Surface at the segmap's own size (1376x768, the Nano Banana canvas),
    # so masks land 1:1 on the photo they were drawn from.
    surface_path = os.path.join(out_dir, "surface_src.jpg")
    seg_size = Image.open(TEST_SEGMAP).size
    Image.open(TEST_SEGMAP_PHOTO).convert("RGB").resize(seg_size, Image.LANCZOS).save(surface_path, quality=92)
    # First produce some masks via the islands pipeline.
    regions_dir = os.path.join(out_dir, "regions")
    regions, _ = islands.extract_color_regions(TEST_SEGMAP, regions_dir)
    # The named[] mapping below is positional, so a change in how the test
    # segmap segments must fail loudly rather than silently misname regions.
    assert len(regions) == 4, f"expected 4 regions from the test segmap, got {len(regions)}"
    # Hand-author the tags file that names every region. These ids are what
    # the example scenes in `render-core/examples/` select on and what the
    # Tauri inspector shows an operator, so they are named after what the
    # region actually *is* in the test photo (a Moroccan kasbah wall) rather than
    # left to fall back to the islands mask stem
    # (`region_mask_003_color_fefd00` — unreadable in a binding list).
    #
    # `extract_color_regions` returns regions largest-area-first, and the
    # test segmap is fixed, so this positional mapping is stable:
    #   0 blue centre block · 1 green left tower · 2 yellow right wing
    #   3 red foreground base wall
    named = [
        ("main_facade", "Main facade", ["facade"], 2),
        ("left_tower", "Left tower", ["tower"], 3),
        ("right_wing", "Right wing", ["tower"], 4),
        ("base_wall", "Base wall", ["wall"], 5),
    ]
    tags = {
        "layers": {
            regions[i]["region_mask"]: {"id": lid, "label": label, "tags": tg, "z": z}
            for i, (lid, label, tg, z) in enumerate(named[: len(regions)])
        },
        "groups": [{"id": "towers", "members": ["left_tower", "right_wing"]}],
        "background": True,
    }
    tags_path = os.path.join(out_dir, "tags.json")
    with open(tags_path, "w") as f:
        json.dump(tags, f)
    pack_dir = os.path.join(out_dir, "pack")
    scene = layerpack.build_layerpack(
        masks_dir=regions_dir,
        output_dir=pack_dir,
        surface=surface_path,
        tags=tags_path,
        feather_px=1,
        include_background=True,
    )
    assert scene["version"] == 1
    assert scene["surface"] == "surface.png"
    assert os.path.isfile(os.path.join(pack_dir, "pack.json"))
    assert os.path.isfile(os.path.join(pack_dir, "surface.png"))
    assert any(layer["id"] == "main_facade" for layer in scene["layers"]), \
        "tags mapping did not apply"
    assert any(layer["id"] == "background" for layer in scene["layers"]), \
        "background slice missing"
    for layer in scene["layers"]:
        mask_path = os.path.join(pack_dir, layer["mask"])
        assert os.path.isfile(mask_path), f"mask missing: {mask_path}"
    assert any(g["id"] == "towers" for g in scene["groups"]), \
        "towers group not emitted"


def test_reproject():
    from wzrd import reproject
    _require(TEST_VIDEO)
    out_dir = _result_dir("reproject")
    _copy_input(TEST_VIDEO, out_dir)
    out = os.path.join(out_dir, "output.mp4")
    metadata = {"x": 0, "y": 0, "width": 320, "height": 240}
    info = reproject.reproject_video(TEST_VIDEO, metadata, output_path=out)
    assert os.path.isfile(out), f"output video not created: {out}"


def test_detect():
    from wzrd import detect
    _require(TEST_NIGHT_IMAGE)
    out_dir = _result_dir("detect")
    _copy_input(TEST_NIGHT_IMAGE, out_dir)
    cropped, info = detect.detect_projection_area(
        TEST_NIGHT_IMAGE,
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert cropped is not None, "detect returned None image"


def test_align():
    from wzrd import detect, align
    _require(TEST_DAY_IMAGE, TEST_NIGHT_IMAGE)
    out_dir = _result_dir("align")
    _copy_input(TEST_DAY_IMAGE, out_dir)
    _copy_input(TEST_NIGHT_IMAGE, out_dir)
    # First detect the projection crop from the night image
    cropped, _ = detect.detect_projection_area(
        TEST_NIGHT_IMAGE,
        output_path=os.path.join(out_dir, "detected_crop.jpg"),
    )
    # Then align the day image to match the detected crop
    warped, info = align.align_images_file(
        TEST_DAY_IMAGE, os.path.join(out_dir, "detected_crop.jpg"),
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert warped is not None, "align returned None image"


def test_prepare_surface_darken_only():
    from wzrd.prepare_surface import prepare_surface
    _require(TEST_DAY_IMAGE)
    out_dir = _result_dir("prepare_surface_darken_only")
    _copy_input(TEST_DAY_IMAGE, out_dir)
    result = prepare_surface(
        TEST_DAY_IMAGE,
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert result['image'] is not None, "prepare_surface returned None"
    assert os.path.isfile(os.path.join(out_dir, "output.jpg"))


def test_prepare_surface_full():
    from wzrd.prepare_surface import prepare_surface
    _require(TEST_DAY_IMAGE, TEST_NIGHT_IMAGE)
    out_dir = _result_dir("prepare_surface_full")
    _copy_input(TEST_DAY_IMAGE, out_dir)
    _copy_input(TEST_NIGHT_IMAGE, out_dir)
    result = prepare_surface(
        TEST_NIGHT_IMAGE,
        day_image_path=TEST_DAY_IMAGE,
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert result['image'] is not None, "prepare_surface returned None"
    assert os.path.isfile(os.path.join(out_dir, "output.jpg"))


def test_concat():
    """Two segments → one clip; boundary + loop frames dropped."""
    _require(TEST_VIDEO)
    import cv2
    from wzrd.concat import concat_videos
    out_dir = _result_dir("concat")
    n = int(cv2.VideoCapture(TEST_VIDEO).get(cv2.CAP_PROP_FRAME_COUNT))
    info = concat_videos([TEST_VIDEO] * 2, os.path.join(out_dir, "story.mp4"), seamless_loop=True)
    assert info["frames"] == 2 * n - 1 - 1, info
    assert os.path.isfile(os.path.join(out_dir, "story.mp4"))


def test_check_frames():
    """A 5 Hz full-frame strobe fails the flash check; a 1 Hz pulse passes it."""
    import numpy as np
    from wzrd.check_frames import check_frames
    from wzrd.utils import VideoWriter
    out_dir = _result_dir("check_frames")
    for hz, want in ((5, "fail"), (1, "pass")):
        path = os.path.join(out_dir, f"strobe_{hz}hz.mp4")
        with VideoWriter(path, 160, 90, 30) as w:
            for i in range(90):
                w.write(np.full((90, 160, 3), 230 if int(i * 2 * hz / 30) % 2 == 0 else 0, np.uint8))
        flash = check_frames(path)["checks"]["flash"]
        assert flash["status"] == want, (hz, flash)


def test_clips():
    """mp4 → HAP .mov the engine accepts (Hap1, sides padded to multiples of 4) + thumb + stills."""
    import subprocess
    import numpy as np
    from wzrd.clips import extract_frame, thumbnail, to_hap
    from wzrd.utils import VideoWriter
    out_dir = _result_dir("clips")
    src = os.path.join(out_dir, "clip.mp4")
    with VideoWriter(src, 202, 114, 24) as w:  # 202x114: not multiples of 4
        for i in range(24):
            w.write(np.full((114, 202, 3), i * 10, np.uint8))
    info = to_hap(src)
    assert info["size"] == [204, 116] and info["padded"], info
    tag = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_tag_string", "-of", "csv=p=0", info["path"]],
                         capture_output=True, text=True).stdout.strip()
    assert tag == "Hap1", tag
    assert os.path.isfile(thumbnail(src, os.path.join(out_dir, "clip.thumb.jpg")))
    for at in ("first", "last", 0.5):
        assert os.path.isfile(extract_frame(src, at, os.path.join(out_dir, f"f_{at}.png")))


def test_mcp():
    """The MCP server boots, serves skills three ways, and ships text-only results."""
    import asyncio
    from fastmcp import Client
    from wzrd_mcp import skill_catalog
    from wzrd_mcp.server import mcp

    async def run():
        async with Client(mcp) as c:
            tools = {t.name: t for t in await c.list_tools()}
            assert "load_skill" in tools and "view_media" in tools and "check_frames" in tools
            assert all(t.output_schema is None for t in tools.values()), "a tool advertises an output schema"
            for name in skill_catalog.catalog():
                assert name in mcp.instructions, f"skill {name} missing from the instructions index"
                r = await c.call_tool("load_skill", {"name": name})
                assert len(r.content) == 1 and r.structured_content is None
                assert (await c.read_resource(f"skill://{name}/SKILL.md"))[0].text.startswith("---")
            workflows = [s.name for s in skill_catalog.catalog().values() if s.kind == "workflow"]
            prompts = {p.name for p in await c.list_prompts()}
            assert prompts == {w.replace("-", "_") for w in workflows}, prompts
            r = await c.call_tool("set_project", {})
            assert len(r.content) == 1 and r.content[0].text.startswith("{")

    asyncio.run(run())
    _show_setup()


def _show_setup():
    """Show setup (docs/TODO/show-setup.md): assets + `asset:` expansion,
    rectify_surface, prompts, dest="project/<sub>", soft delete — against
    temp library/projects roots."""
    import asyncio
    import json
    import tempfile
    from pathlib import Path
    from fastmcp import Client
    from fastmcp.exceptions import ToolError
    from PIL import Image
    from wzrd_mcp import library, project, prompts
    from wzrd_mcp.server import mcp

    _require(TEST_NIGHT_IMAGE, TEST_DAY_IMAGE)
    tmp = Path(tempfile.mkdtemp(prefix="wzrd_setup_"))
    saved = {k: os.environ.get(k) for k in ("WZRD_MEDIA_ROOT", "WZRD_PROJECTS_ROOT")}
    os.environ["WZRD_MEDIA_ROOT"] = str(tmp / "media")
    os.environ["WZRD_PROJECTS_ROOT"] = str(tmp / "projects")
    show = "setup-t"

    async def call(c, tool, args):
        r = await c.call_tool(tool, {**args, "project": show} if tool != "set_project" else args)
        return json.loads(r.content[0].text)

    async def run():
        async with Client(mcp) as c:
            info = await call(c, "set_project", {"name": show, "canvas": "16:9"})
            assert set(project.LAYOUT) <= set(info["layout"]) and info["assets"] == {}
            photos = (await call(c, "import_photos", {"paths": [os.path.abspath(TEST_NIGHT_IMAGE),
                                                               os.path.abspath(TEST_DAY_IMAGE)],
                                                     "conform": False}))["photos"]
            await call(c, "pin_asset", {"name": "photo_lit", "path": photos[0]["path"]})
            assets = (await call(c, "pin_asset", {"name": "photo_ambient", "path": photos[1]["path"]}))["assets"]
            assert assets["photo_lit"]["exists"] and assets["photo_lit"]["id"]
            try:
                await c.call_tool("rectify_surface", {"lit_photo": "asset:nope", "project": show})
                raise AssertionError("unknown asset was accepted")
            except ToolError as exc:
                assert "photo_lit" in str(exc), exc
            res = await call(c, "rectify_surface", {"lit_photo": "asset:photo_lit",
                                                    "ambient_photo": "asset:photo_ambient"})
            for k in ("surface_lit", "surface", "surface_ambient"):
                assert Image.open(res[k]).size == (1376, 768), (k, Image.open(res[k]).size)
                item = library.item(res[k])
                assert item and item["wzrd"] and item["tool"] == "rectify_surface", item
            rec = library.render(res["surface"])["record"]
            assert rec["args"]["lit_photo"] == photos[0]["path"], rec["args"]  # expanded, not "asset:…"
            assert rec["corners"] == res["corners"] == rec["args"]["corners"] and rec["manual_corners"] is False
            assert res["align"] and not res["manual_corners"]
            nudged = [[x + 0.005, y] for x, y in res["corners"]]
            res2 = await call(c, "rectify_surface", {"lit_photo": "asset:photo_lit", "corners": nudged})
            assert res2["manual_corners"] and res2["surface_ambient"] is None and res2["run"] != res["run"]
            assert all(abs(a - b) < 1e-5 for p, q in zip(res2["corners"], nudged) for a, b in zip(p, q))
            # prompts: the shipped WZRD default, then a project fork listed first
            found = (await call(c, "list_prompts", {"kind": "segmentation"}))["prompts"]
            assert found[0]["scope"] == "wzrd" and prompts.PLACEHOLDER in found[0]["prompt"]
            await call(c, "save_prompt", {"kind": "segmentation", "name": "default", "prompt": "x {n_regions}"})
            found = (await call(c, "list_prompts", {"kind": "segmentation"}))["prompts"]
            assert [p["scope"] for p in found] == ["project", "wzrd"], found

    try:
        asyncio.run(run())
        # dest="project/<sub>" and the projects root as a library root
        library.check_dest("project/segmentation")
        token = project.call_project.set(show)
        library.current_dest.set("project/segmentation")
        assert library.dest_dir() == project.project_dir(show) / "segmentation"
        abs_dest = str(project.project_dir(show) / "segmentation")
        library.current_dest.set(abs_dest)
        assert str(library.dest_dir()) == abs_dest  # what a job row records as its dest
        library.current_dest.set(None)
        project.call_project.reset(token)
        assert str(project.projects_root()) in {r["path"] for r in library.roots()}
        # soft delete → restore → delete → flush
        surf = sorted((project.project_dir(show) / "surfaces").glob("*_surface.png"))[0]
        rid = library.item(str(surf))["render_id"]
        out = library.delete([str(surf)], siblings=False)
        assert not surf.exists() and out["trashed"][0]["path"] == str(surf)
        tid = out["trashed"][0]["id"]
        assert library.trash_list()["items"][0]["id"] == tid
        assert library.trash_restore([tid])["restored"][0]["path"] == str(surf) and surf.is_file()
        assert library.item(str(surf))["render_id"] == rid and not library.trash_list()["items"]
        library.label([str(surf)], score=1)
        tid = library.purge(str(surf.parent), max_score=1)["trashed"][0]["id"]  # delete_scored
        assert library.trash_flush([tid]) == {"flushed": 1} and not library.trash_list()["items"]
        assert not surf.exists() and library.render(rid)["purged"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test registry ───────────────────────────────────────────────────────────

TESTS = {
    "darken": test_darken,
    "detect": test_detect,
    "align": test_align,
    "prepare_surface_darken_only": test_prepare_surface_darken_only,
    "prepare_surface_full": test_prepare_surface_full,
    "subtract_video": test_subtract_video,
    "islands": test_islands,
    "layerpack": test_layerpack,
    "reproject": test_reproject,
    "concat": test_concat,
    "check_frames": test_check_frames,
    "clips": test_clips,
    "mcp": test_mcp,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

class _MissingFile(Exception):
    pass


def _require(*paths):
    for p in paths:
        if not os.path.isfile(p):
            raise _MissingFile(f"required input file not found: {p}")


def _result_dir(test_name):
    """Create and return test_results/<test_name>/, clearing any previous run."""
    d = os.path.join(RESULTS_ROOT, test_name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return d


def _copy_input(src, dst_dir):
    """Copy an input file into dst_dir, preserving the basename."""
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _run_test(name, fn):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS  ({elapsed:.2f}s)")
        return True
    except _MissingFile as e:
        elapsed = time.perf_counter() - t0
        print(f"  SKIP  {e}  ({elapsed:.2f}s)")
        return None  # skip
    except Exception:
        elapsed = time.perf_counter() - t0
        traceback.print_exc()
        print(f"  FAIL  ({elapsed:.2f}s)")
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WZRD smoke tests")
    parser.add_argument("modules", nargs="*", metavar="MODULE",
                        help=f"modules to test: {', '.join(TESTS)}")
    parser.add_argument("--all", action="store_true", help="test every module")
    args = parser.parse_args()

    if args.all:
        selected = list(TESTS)
    elif args.modules:
        bad = [m for m in args.modules if m not in TESTS]
        if bad:
            parser.error(f"unknown module(s): {', '.join(bad)}")
        selected = args.modules
    else:
        parser.print_help()
        sys.exit(0)

    passed, failed, skipped = 0, 0, 0
    for name in selected:
        result = _run_test(name, TESTS[name])
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
