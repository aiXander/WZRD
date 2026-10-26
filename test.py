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


def test_rectify():
    """Lit + ambient photo → the three surface images at the canvas."""
    from wzrd.prepare_surface import rectify_surface
    _require(TEST_DAY_IMAGE, TEST_NIGHT_IMAGE)
    res = rectify_surface(TEST_NIGHT_IMAGE, TEST_DAY_IMAGE, output_size=(1376, 768))
    for key in ("lit", "surface", "ambient"):
        assert res[key] is not None and res[key].size == (1376, 768), (key, res[key])
    assert len(res["corners"]) == 4 and res["align"] and not res["manual_corners"]


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
    from fastmcp.exceptions import ToolError
    from wzrd_mcp import hub, skill_catalog
    from wzrd_mcp.server import APP_ONLY, mcp

    async def run():
        async with Client(mcp) as c:
            tools = {t.name: t for t in await c.list_tools()}
            assert "load_skill" in tools and "view_media" in tools and "check_frames" in tools
            assert all(t.output_schema is None for t in tools.values()), "a tool advertises an output schema"
            missing = [n for n in tools if n not in mcp.instructions]
            assert not missing, f"tools missing from server._JOB_MAP: {missing}"
            for name in skill_catalog.catalog():
                assert name in mcp.instructions, f"skill {name} missing from server._JOB_MAP"
                r = await c.call_tool("load_skill", {"name": name})
                assert len(r.content) == 1 and r.structured_content is None
                assert (await c.read_resource(f"skill://{name}/SKILL.md"))[0].text.startswith("---")
            workflows = [s.name for s in skill_catalog.catalog().values() if s.kind == "workflow"]
            prompts = {p.name for p in await c.list_prompts()}
            assert prompts == {w.replace("-", "_") for w in workflows}, prompts
            # app-only tools (the show-setup steps): hidden and refused here, run by the app
            app_only = [t.name for t in await mcp.local_provider.list_tools() if APP_ONLY in (t.tags or set())]
            assert "set_project" in app_only and not set(app_only) & set(tools), app_only
            try:
                await c.call_tool("set_project", {})
                raise AssertionError("an app-only tool was callable over MCP")
            except ToolError:
                pass
            assert "existing_projects" in await hub.run_tool("set_project", {})
            r = await c.call_tool("show_info", {})
            assert len(r.content) == 1 and r.content[0].text.startswith("{")

    asyncio.run(run())
    _show_setup()


def test_textureflow():
    """texture_flow's backend switch: the default Modal path (Modal mocked) and
    backend="local" against a fake job server speaking the local wire contract
    v1 (health, multipart submit, long-poll, result + files, cancel, 400, 401,
    unreachable, timeout, server-side error), the
    spec the tool is generated from (tf_spec: cache vs snapshot, background
    refresh, recipe args the spec doesn't know), and draft → finish (spec
    2.2.0's draft / finish_from: request_id in the record + filename, the
    finish on the draft's backend, lineage)."""
    import asyncio
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    import modal
    from aiohttp import web
    from fastmcp import Client
    from fastmcp.exceptions import ToolError
    from wzrd_mcp import hub, jobs, library, tf_spec
    from wzrd_mcp.server import mcp
    from wzrd_mcp import tools  # after the server (import order)

    _require(TEST_DAY_IMAGE)
    tmp = Path(tempfile.mkdtemp(prefix="wzrd_tf_"))
    env_keys = ("WZRD_MEDIA_ROOT", "WZRD_PROJECTS_ROOT", "TEXTUREFLOW_BACKEND", "TEXTUREFLOW_APP",
                "WZRD_MODAL_ENV", "TEXTUREFLOW_LOCAL_URL", "TEXTUREFLOW_LOCAL_TOKEN", "WZRD_RECIPES_ROOT",
                "TEXTUREFLOW_LOCAL_SSH", "TEXTUREFLOW_MODAL_SLOTS")
    saved_env = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys[2:]:
        os.environ.pop(k, None)
    os.environ["WZRD_MEDIA_ROOT"] = str(tmp / "media")
    os.environ["WZRD_PROJECTS_ROOT"] = str(tmp / "projects")
    saved_modal = (modal.Cls.from_name, modal.Function.from_name, modal.Dict.from_name)
    mp4 = tmp / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=8", "-t", "1",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True)
    video = mp4.read_bytes()
    names = ("output_video", "first_pass_video", "control_signal_preview", "mapping_motion_preview")
    timings = {"prep_s": 0.1, "queue_s": 0.0, "render_s": 20.0, "total_s": 21.0, "cold_start": False,
               "boot_s": None, "restore_s": None}
    base = {"draft": dict(use_upscale=False, upscale_esrgan=False)}
    style = os.path.abspath(TEST_DAY_IMAGE)
    rendered = set(names) | {"seed", "params", "request_id", "timings", "clip", "job"}
    rids = iter(range(1, 1000))

    q7 = {"stage_s": {"styles": 0.5, "sample": 9.0, "vae_decode": 1.0, "rife": 2.0, "encode": 1.5},
          "size": [640, 368], "final_size": [1920, 1080], "n_frames": 48, "n_output_frames": 192, "steps": 10}

    def tf_info(args: dict, rid: str, **extra) -> dict:
        """`info` as the server sends it: spec ≥ 2.2.0 servers (args carry draft / finish_from)
        add the stage + the bundle; older ones send neither (WZRD must cope with both)."""
        extra = {**q7, **extra}
        if not ({"draft", "finish_from"} & set(args)):
            return extra
        src = args.get("finish_from") or rid
        info = {**extra, "stage": "finish" if args.get("finish_from") else "draft" if args.get("draft") else "full",
                "draft": {"id": src, "expires_at": "2099-01-01T00:00:00Z", "bytes": 1}}
        return {**info, "finished_from": src} if args.get("finish_from") else info

    # -- Modal, mocked -----------------------------------------------------
    modal_calls: list = []

    def no_modal(*a, **kw):
        raise AssertionError(f"Modal was called: {a} {kw}")

    modal_gate: dict = {"event": None}  # an asyncio.Event: Modal renders block until it is set
    modal_kw: list = []  # the extra spawn kwargs (progress_key) per spawn

    class FakeCall:
        object_id = "fc-test"

        def __init__(self, args, files):
            self.args, self.files = args, files
            self.get, self.cancel = self, self

        async def aio(self):  # get.aio() / cancel.aio()
            if modal_gate["event"] is not None:
                await modal_gate["event"].wait()
            rid = f"tf-modal-{next(rids)}"
            return {"request_id": rid, "params": {"seed": 1111, **self.args},
                    "info": tf_info(self.args, rid, app="textureflow"), "timings": timings,
                    "outputs": {n: {"filename": f"{n}.mp4", "bytes": video, "mime": "video/mp4"}
                                for n in names}}

    class FakeSpawn:
        async def aio(self, args, files, **kw):
            modal_calls.append(("render", args, files))
            modal_kw.append(kw)
            return FakeCall(args, files)

    dict_reads: list = []

    class FakeDict:  # Modal Dict `textureflow-progress`: a scripted sequence of progress values
        script = [None, {"stage": "started", "cold_start": False, "t": 0},
                  {"stage": "styles", "step": None, "total": None, "t": 1},
                  {"stage": "KSampler", "step": 2, "total": 10, "t": 2}]

        def __init__(self, *a, **kw):
            dict_reads.append(("from_name", a, kw))
            self.get, self.pop = self._Get(), self._Pop()

        class _Get:
            async def aio(self, key):
                dict_reads.append(("get", key))
                n = sum(1 for r in dict_reads if r[0] == "get")
                return FakeDict.script[min(n, len(FakeDict.script)) - 1]

        class _Pop:
            async def aio(self, key):
                dict_reads.append(("pop", key))

    class FakeCls:
        def __init__(self, *a, **kw):
            modal_calls.append(("cls", a, kw))

        def __call__(self):
            self.render = type("R", (), {"spawn": FakeSpawn()})()
            return self

    # -- the fake local job server (wire contract v1) ----------------------
    TOKEN, GPU = "s3cret", "NVIDIA GeForce RTX 3090"
    srv = {"jobs": {}, "submits": [], "cancels": [], "requests": 0, "worker": "ready", "legacy": False,
           "orders": [], "go": set()}

    @web.middleware
    async def auth(request, handler):
        srv["requests"] += 1
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    async def health(request):
        return web.json_response({"ok": True, "gpu": GPU, "busy": False, "queue": 0, "worker": srv["worker"],
                                  "commit": "abc123", "vram_free_gb": 23.1})

    async def submit(request):
        args, files, meta = None, {}, None
        reader = await request.multipart()
        async for part in reader:
            if part.name == "args":
                assert part.headers.get("Content-Type") == "application/json", part.headers
                args = json.loads(await part.text())
            elif part.name == "meta" and not srv["legacy"]:
                meta = json.loads(await part.text())
            elif part.name == "meta":  # a server before the queue routes
                return web.json_response({"error": "unexpected multipart part 'meta' (want 'args' and 'file')"},
                                         status=400)
            else:
                assert part.name == "file", part.name
                files[part.filename] = bytes(await part.read())
        if args.get("n_steps") == 99:
            return web.json_response({"error": "n_steps: 99 is outside 4-14"}, status=400)
        srv["n"] = srv.get("n", 0) + 1
        jid = f"lj{srv['n']}"
        script = {666: "hold", 13: "fail", 777: "wait"}.get(args.get("seed"), "ok")
        down = srv["worker"] == "down"  # accepted, but failed at once
        srv["jobs"][jid] = {"state": "error" if down else "queued", "script": script, "polls": 0,
                            "args": args, "changed": asyncio.Event(), "meta": meta or {},
                            "error": "render worker is down" if down else None}
        srv["submits"].append({"job_id": jid, "args": args, "files": files, "meta": meta})
        return web.json_response({"job_id": jid, "queue_position": qpos(jid)})

    def queued_ids():
        order = srv.get("order") or []
        q = [j for j, x in srv["jobs"].items() if x["state"] == "queued"]
        return [j for j in order if j in q] + [j for j in q if j not in order]

    def qpos(jid):
        q = queued_ids()
        return q.index(jid) if jid in q else None

    def status_doc(jid, job):
        running = job["state"] == "running"
        return {"job_id": jid, "state": job["state"], "stage": "KSampler" if running else None,
                "step": job["polls"] if running else None, "total": 3 if running else None,
                "queue_position": qpos(jid) if job["state"] == "queued" else 0 if running else None,
                "error": (job["error"] or "CUDA out of memory") if job["state"] == "error" else None}

    async def status(request):
        jid = request.match_info["jid"]
        job = srv["jobs"].get(jid)
        if not job:
            return web.json_response({"error": "unknown job"}, status=404)
        assert 0 <= float(request.query.get("wait", 0)) <= 60
        if job["state"] in ("queued", "running"):
            if job["script"] == "wait" and job["state"] == "queued" and jid not in srv["go"]:
                await asyncio.sleep(0.05)
                return web.json_response(status_doc(jid, job))
            if job["script"] == "hold" and job["state"] == "running":
                try:  # long-poll until something changes (cancel), capped short for the test
                    await asyncio.wait_for(job["changed"].wait(), 1.0)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0.02)
                job["polls"] += 1
                job["state"] = ("running" if job["polls"] < 3 or job["script"] == "hold"
                                else "error" if job["script"] == "fail" else "done")
        return web.json_response(status_doc(jid, job))

    async def result(request):
        jid = request.match_info["jid"]
        job = srv["jobs"][jid]
        if job["state"] != "done":
            return web.json_response({"error": "not done", "state": job["state"]}, status=409)
        return web.json_response({
            "request_id": f"req-{jid}", "params": {"seed": 4242, **job["args"]},
            "info": tf_info(job["args"], f"req-{jid}", app="textureflow-local", gpu=GPU),
            "timings": {**timings, "queue_s": 1.5, "cold_start": True},
            "outputs": {n: {"filename": f"{n}.mp4", "mime": "video/mp4",
                            "url": f"/v1/jobs/{jid}/files/{n}"} for n in names}})

    async def files(request):
        assert request.match_info["name"] in names
        return web.Response(body=video, content_type="video/mp4")

    async def cancel(request):
        jid = request.match_info["jid"]
        job = srv["jobs"][jid]
        srv["cancels"].append(jid)
        if job["state"] == "queued":
            job["state"] = "cancelled"
        elif job["state"] == "running":  # answers "running"; cancelled once the worker stops
            async def stop():
                await asyncio.sleep(0.1)
                job["state"] = "cancelled"
                job["changed"].set()
            asyncio.get_running_loop().create_task(stop())
        return web.json_response({"job_id": jid, "state": job["state"]})

    async def queue(request):
        if srv["legacy"]:
            return web.json_response({"error": "not found"}, status=404)
        doc = lambda j: {**status_doc(j, srv["jobs"][j]), "client": srv["jobs"][j]["meta"].get("client"),  # noqa: E731
                         "label": srv["jobs"][j]["meta"].get("label"), "created_at": 0, "started_at": None}
        running = [j for j, x in srv["jobs"].items() if x["state"] == "running"]
        return web.json_response({"running": doc(running[0]) if running else None,
                                  "queued": [doc(j) for j in queued_ids()], "worker": srv["worker"],
                                  "gpu_holder": None})

    async def queue_order(request):
        if srv["legacy"]:
            return web.json_response({"error": "not found"}, status=404)
        order = (await request.json())["order"]
        srv["orders"].append(order)
        srv["order"] = order
        return web.json_response({"order": queued_ids()})

    app = web.Application(middlewares=[auth])
    app.add_routes([web.get("/v1/queue", queue), web.post("/v1/queue/order", queue_order),
                    web.get("/v1/health", health), web.post("/v1/render", submit),
                    web.get("/v1/jobs/{jid}", status), web.get("/v1/jobs/{jid}/result", result),
                    web.get("/v1/jobs/{jid}/files/{name}", files),
                    web.post("/v1/jobs/{jid}/cancel", cancel)])

    async def call(c, tool, args):
        r = await c.call_tool(tool, args)
        return json.loads(r.content[0].text)

    async def refused(c, tool, args, *needles):
        try:
            await c.call_tool(tool, args)
        except ToolError as exc:
            for n in needles:
                assert n in str(exc), (n, str(exc))
            return str(exc)
        raise AssertionError(f"{tool}({args}) was accepted")

    async def run():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"

        async def silent(reader, writer):  # accepts, never answers: the 2 s health timeout
            await asyncio.sleep(30)
        mute = await asyncio.start_server(silent, "127.0.0.1", 0)
        mute_url = f"http://127.0.0.1:{mute.sockets[0].getsockname()[1]}"
        try:
            async with Client(mcp) as c:
                await hub.run_tool("set_project", {"name": "tf-t"})

                # 1. backend unset → today's Modal call, exactly.
                modal.Cls.from_name = FakeCls
                res_m = await call(c, "texture_flow", {"images": [style], **base["draft"], "project": "tf-t"})
                assert modal_calls[0] == ("cls", ("textureflow", "TextureFlow"), {"environment_name": "wzrd"})
                _, sent_m, files_m = modal_calls[1]
                assert set(sent_m) == {"images", "use_upscale", "upscale_esrgan"}, sent_m
                assert rendered | {"spend"} <= set(res_m), set(res_m)
                rec = library.render(res_m["output_video"])["record"]
                assert rec["est_usd"] > 0 and rec["args"]["model"] == "textureflow (TextureFlow)", rec
                assert "backend" not in rec["args"], rec["args"]
                # a pre-2.2.0 server: the id is kept and names the file, no bundle
                assert rec["textureflow"] == {
                    "request_id": res_m["request_id"], "backend": "modal", "stage": "full", "draft": None,
                    "params": res_m["params"], "stage_s": q7["stage_s"], "gpu": None, "first_pass_size": [640, 368],
                    "output_size": [1920, 1080], "n_frames": 48, "n_output_frames": 192, "steps": 10,
                    "timings": timings}, rec["textureflow"]  # Q7: the ETA's training data
                assert modal_kw[-1] == {}  # the snapshot claims no capabilities: no progress_key
                j = jobs.get(res_m["job"])
                assert j["backend"] == "modal" and j["pinned"] is False and j["lane_pos"] is None, j
                assert j["progress"]["phase"] == "saving" and j["progress"]["pct"] == 1.0, j["progress"]
                assert Path(res_m["output_video"]).name == f"texture_flow_{res_m['request_id']}.mp4"
                assert "draft" not in res_m
                assert library.query_one("SELECT remote FROM jobs WHERE id=?", (res_m["job"],))["remote"] == "modal|fc-test"
                assert srv["requests"] == 0
                assert jobs.estimate("texture_flow", base["draft"]) > 0
                assert jobs.estimate("texture_flow", {"backend": "local"}) == 0

                # 2. local: never Modal; missing config / bad backend are clear errors.
                modal.Cls.from_name = modal.Function.from_name = no_modal
                await refused(c, "texture_flow", {"images": [style], "backend": "local"}, "TEXTUREFLOW_LOCAL_URL")
                await refused(c, "texture_flow", {"images": [style], "backend": "gpu"}, "'modal'", "'local'")
                os.environ["TEXTUREFLOW_LOCAL_URL"] = url
                await refused(c, "texture_flow", {"images": [style], "backend": "local"}, "TEXTUREFLOW_LOCAL_TOKEN")
                os.environ["TEXTUREFLOW_LOCAL_TOKEN"] = "wrong"
                await refused(c, "texture_flow", {"images": [style], "backend": "local"}, "401", "TEXTUREFLOW_LOCAL_TOKEN")
                os.environ["TEXTUREFLOW_LOCAL_TOKEN"] = TOKEN

                # 3. local happy path: same args + files on the wire, same response, $0.
                res_l = await call(c, "texture_flow", {"images": [style], **base["draft"], "backend": "local",
                                                       "project": "tf-t"})
                sub = srv["submits"][-1]
                assert sub["args"] == sent_m and sub["files"] == files_m, (sub["args"], sent_m)
                assert set(res_l) == set(res_m) - {"spend"}, set(res_l) ^ set(res_m)
                assert res_l["seed"] == 4242 and res_l["timings"]["queue_s"] == 1.5
                assert all(Path(res_l[n]).read_bytes() == video for n in names[1:])  # previews: untagged copies
                assert Path(res_l["output_video"]).stat().st_size > 0 and Path(res_l["clip"]).is_file()
                rec = library.render(res_l["output_video"])["record"]
                assert rec["est_usd"] == 0 and rec["args"]["model"] == f"textureflow-local (TextureFlow, {GPU})", rec
                assert rec["args"]["backend"] == "local" and rec["args"]["seed"] == 4242
                row = library.query_one("SELECT remote, est_usd FROM jobs WHERE id=?", (res_l["job"],))
                assert row["remote"] == f"local|{sub['job_id']}" and row["est_usd"] == 0, dict(row)

                # 4. TEXTUREFLOW_BACKEND=local is the default; backend="modal" still overrides it.
                os.environ["TEXTUREFLOW_BACKEND"] = "local"
                n = len(srv["submits"])
                await call(c, "texture_flow", {"images": [style], **base["draft"]})
                assert len(srv["submits"]) == n + 1
                modal.Cls.from_name = FakeCls
                await call(c, "texture_flow", {"images": [style], **base["draft"], "backend": "modal"})
                assert len(srv["submits"]) == n + 1 and modal_calls[-1][0] == "render"
                modal.Cls.from_name = no_modal
                del os.environ["TEXTUREFLOW_BACKEND"]

                # 5. validation (400) passes the server's text through; nothing queued.
                n = len(srv["jobs"])
                msg = await refused(c, "texture_flow", {"images": [style], "n_steps": 99, "backend": "local"})
                assert msg == "n_steps: 99 is outside 4-14" and len(srv["jobs"]) == n, msg
                # a render that fails on the desktop
                await refused(c, "texture_flow", {"images": [style], "seed": 13, "backend": "local"},
                              "CUDA out of memory")

                # 6. cancelling (the app's ✕) stops the desktop render; the job reads cancelled.
                started = await call(c, "texture_flow", {"images": [style], "seed": 666, "backend": "local",
                                                         "wait": False})
                job = started["job"]
                for _ in range(200):
                    row = library.query_one("SELECT remote FROM jobs WHERE id=?", (job,))
                    if row["remote"] and srv["jobs"][row["remote"].split("|")[1]]["state"] == "running":
                        break
                    await asyncio.sleep(0.02)
                jid = row["remote"].split("|")[1]
                out = await jobs.cancel(job)
                assert out["status"] == "cancelled" and out["remote_cancel"] == "local", out
                for _ in range(50):
                    if srv["jobs"][jid]["state"] == "cancelled":
                        break
                    await asyncio.sleep(0.02)
                assert srv["cancels"] == [jid] and srv["jobs"][jid]["state"] == "cancelled"
                assert await jobs._cancel_remote(f"local|{jid}") == "local"  # a lost job's path

                # 7. worker booting = up (the job queues); down = accepted, failed on the first poll.
                srv["worker"] = "booting"
                await call(c, "texture_flow", {"images": [style], **base["draft"], "backend": "local"})
                srv["worker"] = "down"
                await refused(c, "texture_flow", {"images": [style], "backend": "local"}, "render worker is down")
                srv["worker"] = "ready"

                # 8. desktop off (refused) / not answering (timeout): fast, clear, nothing sent.
                n = srv["requests"]
                for dead in (f"http://127.0.0.1:{_free_port()}", mute_url):
                    os.environ["TEXTUREFLOW_LOCAL_URL"] = dead
                    t0 = time.perf_counter()
                    await refused(c, "texture_flow", {"images": [style], "backend": "local"},
                                  "unreachable", "desktop on", dead)
                    assert time.perf_counter() - t0 < 3.0, (dead, time.perf_counter() - t0)
                assert srv["requests"] == n

                # 9. the spec: the schema is the spec's args + WZRD's own; a recipe arg
                # the spec doesn't know is dropped with a warning, not sent.
                snap = json.loads(tf_spec.SNAPSHOT_PATH.read_text())["spec"]
                tool = next(t for t in await c.list_tools() if t.name == "texture_flow")
                own = {"region", "allow_over_cap", "hap", "recipe", "backend", "project", "dest", "wait"}
                assert set(tool.inputSchema["properties"]) == set(snap["parameters"]) | own, tool.inputSchema
                assert "Range 256-3840, step 2. Default 1920." in tool.inputSchema["properties"]["output_width"]["description"]
                rdir = tmp / "recipes" / "old-look"
                rdir.mkdir(parents=True)
                (rdir / "recipe.json").write_text(json.dumps(
                    {"schema": 1, "name": "old-look", "images": [style], "args": {"n_steps": 4, "cfg_scale": 3}}))
                os.environ["WZRD_RECIPES_ROOT"] = str(tmp / "recipes")
                modal.Cls.from_name = FakeCls
                res_r = await call(c, "texture_flow", {"recipe": "old-look", **base["draft"]})
                sent_r = modal_calls[-1][1]
                assert "cfg_scale" not in sent_r and sent_r["n_steps"] == 4, sent_r
                assert "cfg_scale" in res_r["warnings"][0], res_r.get("warnings")
                modal.Cls.from_name = no_modal

                # 10. draft → pick → finish against a spec-2.2.0 tool (the arg spec gains draft +
                # finish_from; the tool is re-generated from it, as a hub restart would).
                spec22 = json.loads(json.dumps(snap))
                spec22["parameters"]["draft"] = {"type": "boolean", "default": False, "description": "Draft."}
                spec22["parameters"]["finish_from"] = {"type": "string", "default": "", "description": "Finish."}
                tf_spec._current = tf_spec.Loaded({"app": "textureflow", "git_sha": "t", "spec": spec22}, "test")
                tools._tf_tool()
                tool = next(t for t in await c.list_tools() if t.name == "texture_flow")
                assert {"draft", "finish_from"} <= set(tool.inputSchema["properties"])
                assert "or finish_from is set" in tool.inputSchema["properties"]["images"]["description"]
                modal.Cls.from_name = FakeCls
                draft = await call(c, "texture_flow", {"images": [style], "draft": True, "project": "tf-t"})
                assert modal_calls[-1][1] == {"images": [Path(style).name], "draft": True}, modal_calls[-1][1]
                rid = draft["request_id"]
                assert draft["draft"] == {"id": rid, "expires_at": "2099-01-01T00:00:00Z"}, draft.get("draft")
                assert Path(draft["output_video"]).name == f"texture_flow_{rid}.mp4"
                d_rec = library.render(draft["output_video"])
                assert d_rec["record"]["textureflow"]["stage"] == "draft" and d_rec["record"]["args"]["draft"] is True
                assert jobs.estimate("texture_flow", {"images": [style], "draft": True}) == tools._TF_EST_DRAFT_USD
                assert jobs.estimate("texture_flow", {"finish_from": rid}) == tools._TF_EST_FINISH_USD
                # the finish: one string, no files; lineage + the draft's settings in its record
                fin = await call(c, "texture_flow", {"finish_from": rid, "upscale_esrgan": False})
                assert modal_calls[-1][1:] == ({"finish_from": rid, "upscale_esrgan": False}, {}), modal_calls[-1]
                f_rec = library.render(fin["output_video"])
                assert f_rec["parent"] == d_rec["id"], (f_rec["parent"], d_rec["id"])
                tf = f_rec["record"]["textureflow"]
                assert tf["stage"] == "finish" and tf["finished_from"] == rid and tf["backend"] == "modal", tf
                assert f_rec["record"]["args"]["images"] == [style] and "draft" not in f_rec["record"]["args"]
                assert fin["finished_from"] == rid
                await refused(c, "texture_flow", {"finish_from": rid, "images": [style]}, "finish_from takes no images")
                # a local draft finishes on the desktop, whatever the default backend; Modal is never asked
                modal.Cls.from_name = no_modal
                os.environ["TEXTUREFLOW_LOCAL_URL"] = url  # §8 left a dead one
                ldraft = await call(c, "texture_flow", {"images": [style], "draft": True, "backend": "local"})
                lrid = ldraft["request_id"]
                assert library.render(ldraft["output_video"])["record"]["textureflow"]["backend"] == "local"
                assert jobs.estimate("texture_flow", {"finish_from": lrid}) == 0
                lfin = await call(c, "texture_flow", {"finish_from": lrid})
                assert srv["submits"][-1]["args"] == {"finish_from": lrid} and srv["submits"][-1]["files"] == {}
                assert library.render(lfin["output_video"])["parent"] == library.render(ldraft["output_video"])["id"]
                await refused(c, "texture_flow", {"finish_from": lrid, "backend": "modal"}, "backend='local'")
                # the draft's mp4 deleted: the finish still links to it by render id
                Path(draft["output_video"]).unlink()
                library.index_file(draft["output_video"])
                modal.Cls.from_name = FakeCls
                fin2 = await call(c, "texture_flow", {"finish_from": rid})
                assert library.render(fin2["output_video"])["parent"] == d_rec["id"]
                modal.Cls.from_name = no_modal

                # 11. the render queue (docs/TODO/textureflow-render-queue.md), through the hub's /api routes:
                # the Modal lane (1 slot here), reorder, cancel a queued job, move to local, a pinned finish.
                import httpx
                from wzrd_mcp import textureflow_local, tf_queue  # noqa: F401
                api = httpx.AsyncClient(transport=httpx.ASGITransport(app=mcp.http_app(transport="streamable-http")),
                                        base_url="http://hub")

                async def get(path):
                    r = await api.get(path)
                    assert r.status_code == 200, (path, r.text)
                    return r.json()

                async def post(path, body, status=200):
                    r = await api.post(path, json=body)
                    assert r.status_code == status, (path, r.status_code, r.text)
                    return r.json()

                async def until(pred, what, timeout=15.0):
                    t_end = time.perf_counter() + timeout
                    while not pred():
                        assert time.perf_counter() < t_end, f"timed out waiting for {what}"
                        await asyncio.sleep(0.05)

                def st(j):
                    return jobs.get(j)["status"]

                def mp4_of(j):
                    return next(o for o in jobs.get(j)["outputs"] if o.endswith(".mp4"))

                modal.Cls.from_name = FakeCls  # Modal slots: default 1
                modal_gate["event"] = asyncio.Event()
                n_render = sum(1 for x in modal_calls if x[0] == "render")
                a, b, d = [(await call(c, "texture_flow", {"images": [style], **base["draft"], "seed": 100 + i,
                                                            "wait": False}))["job"] for i in range(3)]
                await until(lambda: st(a) == "running" and st(b) == st(d) == "queued", "the Modal lane")
                assert [jobs.get(x)["lane_pos"] for x in (b, d)] == [0, 1]
                jb = jobs.get(b)
                assert jb["progress"]["phase"] == "queued" and jb["progress"]["eta_s"] > 0 and jb["backend"] == "modal"
                assert jobs.get(d)["progress"]["eta_s"] > jb["progress"]["eta_s"]  # waits for one more ahead
                assert [x["job"] for x in (await get("/api/tf/queue"))["modal"]] == [a, b, d]
                be = await get("/api/tf/backends")
                assert be["modal"] == {"state": "ready", "slots": 1, "max_slots": 3, "running": 1, "queued": 2}, be["modal"]
                assert (await post("/api/jobs/reorder", {"backend": "modal", "order": [d, "j_nope"]}))["order"] == [d, b]
                assert jobs.get(d)["lane_pos"] == 0 and jobs.get(b)["lane_pos"] == 1
                assert (await post("/api/jobs/cancel", {"job": b}))["status"] == "cancelled"
                jobs.set_daily_cap(0.0)
                moved = await post("/api/jobs/move", {"job": d, "backend": "local"})  # $0: no cap check
                jobs.set_daily_cap(20.0)
                assert moved["backend"] == "local" and moved["est_usd"] == 0 and moved["args"]["backend"] == "local"
                await until(lambda: st(d) == "done", "the moved job")
                modal_gate["event"].set()
                await until(lambda: st(a) == "done", "the running Modal job")
                assert [x[1].get("seed") for x in modal_calls if x[0] == "render"][n_render:] == [100]  # b, d never spawned
                rec = library.render(mp4_of(d))["record"]
                assert rec["textureflow"]["backend"] == "local" and rec["args"]["backend"] == "local", rec["args"]
                # a finish is pinned to its draft's backend
                modal_gate["event"] = asyncio.Event()
                hold = (await call(c, "texture_flow", {"images": [style], **base["draft"], "wait": False}))["job"]
                fq = (await call(c, "texture_flow", {"finish_from": rid, "wait": False}))["job"]
                await until(lambda: st(fq) == "queued", "the queued finish")
                assert jobs.get(fq)["pinned"] is True
                err = await post("/api/jobs/move", {"job": fq, "backend": "local"}, 400)
                assert "can't move" in err["error"], err
                await post("/api/jobs/move", {"job": hold, "backend": "local"}, 400)  # running: not queued
                modal_gate["event"].set()
                await until(lambda: st(fq) == "done" and st(hold) == "done", "the finish")
                modal_gate["event"] = None

                # 11b. Modal slots are an operator setting (library.db, 1..3); lowering never cancels a running job.
                be = (await get("/api/tf/backends"))["modal"]
                assert (be["slots"], be["max_slots"]) == (1, 3), be
                for bad in (0, 4, 1.5, "2"):
                    await post("/api/tf/modal", {"slots": bad}, 400)
                modal_gate["event"] = asyncio.Event()
                s3 = [(await call(c, "texture_flow", {"images": [style], **base["draft"], "wait": False}))["job"]
                      for _ in range(3)]
                await until(lambda: [st(x) for x in s3] == ["running", "queued", "queued"], "1 slot")
                assert (await post("/api/tf/modal", {"slots": 2}))["slots"] == 2
                await until(lambda: [st(x) for x in s3] == ["running", "running", "queued"], "2 slots")
                assert library.get_setting("tf_modal_slots") == 2
                await post("/api/tf/modal", {"slots": 1})
                await asyncio.sleep(0.3)
                assert [st(x) for x in s3] == ["running", "running", "queued"]  # both keep running
                modal_gate["event"].set()
                await until(lambda: all(st(x) == "done" for x in s3), "the lane drains")
                modal_gate["event"] = None

                # 12. the local lane = the desktop server's queue: meta, lane_pos incl. another client's job,
                # reorder forwarded (foreign jobs keep their place), move local → Modal cancels the server job.
                srv["jobs"]["lab1"] = {"state": "queued", "script": "wait", "polls": 0, "args": {}, "error": None,
                                       "changed": asyncio.Event(), "meta": {"client": "lab", "label": "explore02"}}
                l1, l2 = [(await call(c, "texture_flow", {"images": [style], **base["draft"], "seed": 777,
                                                          "backend": "local", "wait": False}))["job"] for _ in range(2)]
                await until(lambda: jobs.get(l2)["lane_pos"] == 2 and jobs.get(l1)["lane_pos"] == 1, "server positions")
                assert st(l1) == "queued" and jobs.get(l1)["progress"]["phase"] == "queued"
                assert srv["submits"][-1]["meta"] == {"client": "wzrd", "label": "day.jpg"}, srv["submits"][-1]["meta"]
                sid = {j: library.query_one("SELECT remote FROM jobs WHERE id=?", (j,))["remote"].split("|")[1]
                       for j in (l1, l2)}
                view = (await get("/api/tf/queue"))["local"]
                assert view[0]["foreign"] is True and view[0]["client"] == "lab" and view[0]["job_id"] == "lab1"
                assert [v.get("job") for v in view[1:]] == [l1, l2], view
                assert (await post("/api/jobs/reorder", {"backend": "local", "order": [l2, l1]}))["order"] == [l2, l1]
                assert srv["orders"][-1] == ["lab1", sid[l2], sid[l1]], srv["orders"]
                assert jobs.get(l2)["lane_pos"] == 1
                await post("/api/jobs/move", {"job": l1, "backend": "modal"})
                assert sid[l1] in srv["cancels"]
                await until(lambda: st(l1) == "done", "local → Modal")
                rec = library.render(mp4_of(l1))["record"]
                assert rec["textureflow"]["backend"] == "modal" and rec["args"]["backend"] == "modal"
                assert modal_calls[-1][1]["seed"] == 777
                srv["go"] |= {sid[l2]}
                await until(lambda: st(l2) == "done", "the reordered local job")
                del srv["jobs"]["lab1"]

                # 13. an older server (no meta part, no /v1/queue): resent without meta, the view falls back,
                # reorder refuses.
                srv["legacy"] = True
                old = (await call(c, "texture_flow", {"images": [style], **base["draft"], "seed": 777,
                                                      "backend": "local", "wait": False}))["job"]
                await until(lambda: st(old) == "queued" and jobs.get(old)["lane_pos"] == 0, "legacy submit")
                assert srv["submits"][-1]["meta"] is None and textureflow_local._meta_ok is False
                assert [v.get("job") for v in (await get("/api/tf/queue"))["local"]] == [old]
                err = await post("/api/jobs/reorder", {"backend": "local", "order": [old]}, 400)
                assert "older version" in err["error"], err
                srv["go"].add(library.query_one("SELECT remote FROM jobs WHERE id=?", (old,))["remote"].split("|")[1])
                await until(lambda: st(old) == "done", "legacy job")
                srv["legacy"], textureflow_local._meta_ok = False, True

                # 14. desktop unreachable: the job is held in the hub (no fail-fast once SSH is set), the
                # lane reads off → service_down → `textureflow-ctl start` over ssh (once) → submitted.
                ctl_calls: list = []
                desk = {"service": "off"}

                def fake_ctl(argv):
                    ctl_calls.append(argv)
                    if desk["service"] == "off":
                        return subprocess.CompletedProcess(argv, 255, "", "ssh: connect to host beast: timed out")
                    result = "ok"
                    if argv[-1].endswith(" start"):
                        desk["service"], result = "active", "started"
                        os.environ["TEXTUREFLOW_LOCAL_URL"] = url  # the service comes up
                    return subprocess.CompletedProcess(argv, 0, json.dumps(
                        {"result": result, "service": desk["service"], "gpu_lock": "free", "holder": None,
                         "want": False, "health": None, "detail": None}) + "\n", "")

                saved_ctl, saved_poll = tf_queue.run_ctl, tf_queue.HOLD_POLL_S
                tf_queue.run_ctl, tf_queue.HOLD_POLL_S = fake_ctl, 0.05
                os.environ["TEXTUREFLOW_LOCAL_SSH"] = "beast"
                os.environ["TEXTUREFLOW_LOCAL_URL"] = f"http://127.0.0.1:{_free_port()}"
                try:
                    tf_queue.S.local.health_t = tf_queue.S.local.ctl_t = 0
                    assert (await get("/api/tf/backends"))["local"]["state"] == "off"
                    held = await call(c, "texture_flow", {"images": [style], **base["draft"], "backend": "local",
                                                          "wait": False})
                    hj = held["job"]
                    await until(lambda: (jobs.get(hj)["progress"] or {}).get("phase") == "waiting_backend", "held")
                    assert st(hj) == "queued" and jobs.get(hj)["lane_pos"] == 0
                    assert not any(x[-1].endswith(" start") for x in ctl_calls)  # off: nothing to start
                    desk["service"] = "inactive"  # the desktop wakes, the service is down
                    tf_queue.S.local.ctl_t = 0
                    await until(lambda: st(hj) == "done", "the held job", timeout=20)
                    assert sum(1 for x in ctl_calls if x[-1].endswith(" start")) == 1
                    assert ctl_calls[0][:6] == ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                                                "beast"] and ctl_calls[0][-1] == "~/.local/bin/textureflow-ctl status"
                    n = len(ctl_calls)
                    out = await post("/api/tf/local", {"action": "start"})  # up already: no ssh
                    assert out["state"] == "ready" and len(ctl_calls) == n, (out, ctl_calls[n:])
                    await post("/api/tf/local", {"action": "stop"}, 400)
                finally:
                    tf_queue.run_ctl, tf_queue.HOLD_POLL_S = saved_ctl, saved_poll
                    del os.environ["TEXTUREFLOW_LOCAL_SSH"]
                    os.environ["TEXTUREFLOW_LOCAL_URL"] = url

                # 15. Modal progress from the Dict, only when the deployed spec() advertises it.
                tf_spec._latest = {"capabilities": ["progress_dict"]}
                modal.Dict.from_name = FakeDict
                modal_gate["event"] = asyncio.Event()
                pj = (await call(c, "texture_flow", {"images": [style], **base["draft"], "wait": False}))["job"]
                await until(lambda: ((jobs.get(pj)["progress"] or {}).get("stage") == "sample"), "Dict progress")
                pr = jobs.get(pj)["progress"]
                assert pr["source"] == "live" and pr["phase"] == "rendering" and (pr["step"], pr["total"]) == (2, 10), pr
                assert 0 < pr["pct"] < 1 and pr["eta_s"] > 0, pr
                modal_gate["event"].set()
                await until(lambda: st(pj) == "done", "the Dict-tracked render")
                assert modal_kw[-1] == {"progress_key": pj} and ("pop", pj) in dict_reads
                assert dict_reads[0] == ("from_name", ("textureflow-progress",), {"environment_name": "wzrd"})
                tf_spec._latest, modal_gate["event"] = None, None
                modal.Cls.from_name = no_modal

                # 16. hub restart (Q8): a local job a dead hub submitted is re-adopted and collected.
                srv["jobs"]["ljx"] = {"state": "done", "script": "ok", "polls": 0, "args": {"images": ["day.jpg"]},
                                      "changed": asyncio.Event(), "meta": {}, "error": None}
                orphan = jobs.start("texture_flow", {"images": [style], "backend": "local"}, project="tf-t",
                                    dest=str(library.inbox()))
                dead = subprocess.Popen(["true"])
                dead.wait()
                with library.tx() as cx:
                    cx.execute("UPDATE jobs SET pid=?, remote='local|ljx' WHERE id=?", (dead.pid, orphan))
                assert st(orphan) == "lost"
                assert await tf_queue.adopt_all() == [orphan]
                await until(lambda: st(orphan) == "done", "the re-adopted job")
                rec = library.render(mp4_of(orphan))["record"]
                assert rec["textureflow"]["backend"] == "local" and rec["job"] == orphan, rec
                assert await tf_queue.adopt_all() == []
                await api.aclose()
        finally:
            mute.close()
            await runner.cleanup()

    def spec_sources():
        """Cache (per app) beats the snapshot; a bad cache or an app without one
        falls back; refresh writes the cache and never raises."""
        loaded, fetch = tf_spec._current, tf_spec.fetch
        try:
            assert tf_spec._load().source == "snapshot"
            snap = json.loads(tf_spec.SNAPSHOT_PATH.read_text())
            newer = {**snap, "git_sha": "abc1234", "spec": {**snap["spec"], "version": "1.1.0"}}
            tf_spec.fetch = lambda app, env: newer
            tf_spec.refresh_in_background().join(10)
            cached = tf_spec.cache_path("textureflow")
            assert json.loads(cached.read_text()) == newer
            got = tf_spec._load()
            assert got.source == f"cache {cached}" and got.spec["version"] == "1.1.0", got.source
            os.environ["TEXTUREFLOW_APP"] = "wzrd-textureflow"  # the fallback: no spec(), no cache
            assert tf_spec._load().source == "snapshot"
            del os.environ["TEXTUREFLOW_APP"]

            def gone(app, env):
                raise RuntimeError("Function not found")
            tf_spec.fetch = gone
            tf_spec.refresh_in_background().join(10)  # logs, keeps the cache
            assert json.loads(cached.read_text()) == newer
            cached.write_text("{not json")
            assert tf_spec._load().source == "snapshot"
        finally:
            tf_spec._current, tf_spec.fetch = loaded, fetch

    def eta_sanity():
        """The stage model (tf_eta): plausible totals, drafts/finishes cheaper, a tracker that only moves
        forward, and the lab's 3090 samples file used as priors when present."""
        from wzrd_mcp import tf_eta

        full, draft = tf_eta.predict_total({}, "modal"), tf_eta.predict_total({"draft": True}, "modal")
        assert 30 < full < 600 and draft < full < tf_eta.predict_total({}, "local"), (full, draft)
        assert tf_eta.predict_total({"n_seconds": 12}, "modal") > full
        laps = dict(tf_eta.plan({"finish_from": "x"}, "modal"))
        assert "resume" in laps and "sample" not in laps and "upscale_pass" in laps
        tr = tf_eta.Tracker(tf_eta.plan({}, "modal"), "modal", 10.0)
        tr.set_phase("starting", 1000.0)
        snaps = []
        for t, (stage, step, total) in enumerate([("styles", None, None), ("KSampler", 5, 10), ("sample", None, None),
                                                  ("KSampler", 2, 6), ("esrgan", 1, 4), ("rife", 3, 4)]):
            tr.on_stage(stage, step, total, 1001.0 + 5 * t)
            snaps.append(tr.snapshot(1002.0 + 5 * t))
        assert [s["stage"] for s in snaps] == ["control_input", "sample", "vae_decode", "upscale_pass", "esrgan", "rife"]
        pcts = [s["pct"] for s in snaps]
        assert pcts == sorted(pcts) and 0 < pcts[0] and pcts[-1] < 1, pcts
        saved = tf_eta.SAMPLES_PATH
        small = {"n_seconds": 6, "density": 0.15, "use_upscale": True}
        try:
            tf_eta.SAMPLES_PATH = tmp / "samples.json"
            tf_eta.invalidate()
            without = dict(tf_eta.plan(small, "local"))["sample"]
            tf_eta.SAMPLES_PATH.write_text(json.dumps({"samples": [
                {"params": {"n_seconds": 6, "n_steps": 10}, "stage_s": {"sample": 900.0},
                 "info": {"size": [512, 288], "n_frames": 48, "steps": 10}},
                {"params": {"n_seconds": 12, "n_steps": 10}, "stage_s": {"sample": 1800.0},
                 "info": {"size": [512, 288], "n_frames": 96, "steps": 10}}]}))
            tf_eta.invalidate()
            assert dict(tf_eta.plan(small, "local"))["sample"] > 2 * without, without  # the slow samples count
        finally:
            tf_eta.SAMPLES_PATH = saved
            tf_eta.invalidate()

    loaded = tf_spec.current()
    try:
        asyncio.run(run())
        tf_spec._current = loaded
        tools._tf_tool()  # back to the loaded spec's tool
        spec_sources()
        eta_sanity()
    finally:
        tf_spec._current = loaded
        modal.Cls.from_name, modal.Function.from_name, modal.Dict.from_name = saved_modal
        tf_spec._latest = None
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    from wzrd_mcp import assets, hub, library, project, prompts
    from wzrd_mcp.server import mcp

    _require(TEST_NIGHT_IMAGE, TEST_DAY_IMAGE)
    tmp = Path(tempfile.mkdtemp(prefix="wzrd_setup_"))
    saved = {k: os.environ.get(k) for k in ("WZRD_MEDIA_ROOT", "WZRD_PROJECTS_ROOT")}
    os.environ["WZRD_MEDIA_ROOT"] = str(tmp / "media")
    os.environ["WZRD_PROJECTS_ROOT"] = str(tmp / "projects")
    show = "setup-t"

    async def call(tool, args):  # as the app's wizard does: /api/tool
        return await hub.run_tool(tool, {**args, "project": show} if tool != "set_project" else args)

    async def run():
        async with Client(mcp) as c:
            info = await call("set_project", {"name": show, "canvas": "16:9"})
            assert set(project.LAYOUT) <= set(info["layout"]) and info["assets"] == {}
            photos = (await call("import_photos", {"paths": [os.path.abspath(TEST_NIGHT_IMAGE),
                                                            os.path.abspath(TEST_DAY_IMAGE)],
                                                  "conform": False}))["photos"]
            assets.pin("photo_lit", photos[0]["path"], by="app", name=show)
            pinned = assets.pin("photo_ambient", photos[1]["path"], by="app", name=show)
            assert pinned["photo_lit"]["exists"] and pinned["photo_lit"]["id"]
            try:
                await call("rectify_surface", {"lit_photo": "asset:nope"})
                raise AssertionError("unknown asset was accepted")
            except ToolError as exc:
                assert "photo_lit" in str(exc), exc
            res = await call("rectify_surface", {"lit_photo": "asset:photo_lit",
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
            res2 = await call("rectify_surface", {"lit_photo": "asset:photo_lit", "corners": nudged})
            assert res2["manual_corners"] and res2["surface_ambient"] is None and res2["run"] != res["run"]
            assert all(abs(a - b) < 1e-5 for p, q in zip(res2["corners"], nudged) for a, b in zip(p, q))
            # prompts: the shipped WZRD default, then a project fork listed first
            found = prompts.list_prompts("segmentation", show)
            assert found[0]["scope"] == "wzrd" and prompts.PLACEHOLDER in found[0]["prompt"]
            prompts.save_prompt("segmentation", "default", "x {n_regions}", "project", show)
            found = prompts.list_prompts("segmentation", show)
            assert [p["scope"] for p in found] == ["project", "wzrd"], found
            # what an agent reads of the show: canvas, assets, regions (none before a pack)
            r = json.loads((await c.call_tool("show_info", {"project": show})).content[0].text)
            assert r["canvas"] and r["assets"]["photo_lit"]["exists"] and r["regions"] == [], r

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
    "rectify": test_rectify,
    "subtract_video": test_subtract_video,
    "islands": test_islands,
    "layerpack": test_layerpack,
    "concat": test_concat,
    "check_frames": test_check_frames,
    "clips": test_clips,
    "mcp": test_mcp,
    "textureflow": test_textureflow,
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
