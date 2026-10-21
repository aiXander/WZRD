"""TextureFlow on our own Modal account — the Eden-free render endpoint.

Pinned ComfyUI + 13 node packs running the TextureFlow graph vendored from
edenartlab/workflows@fcbd170 (`workflow/`). One GPU class, one method:

    TextureFlow.render(args: dict, files: dict[str, bytes]) -> dict

`args` is any subset of `workflow/api.yaml`'s parameters plus `output_codec`
("h264" default | "hevc10", see OUTPUT_CODECS); api.yaml defaults are
applied server-side and every value is validated before ComfyUI sees it. Media
args (`images`, `control_input`, `diffusion_mask`) are http(s) URLs or keys into
`files`. Returns the 4 mp4s as bytes (also copied to volume
`textureflow-outputs/<request_id>/`).

Every CLI call passes `--env wzrd`:

    uv run modal run    --env wzrd deploy/textureflow/app.py::download_models
    uv run modal deploy --env wzrd deploy/textureflow/app.py
    uv run modal run    --env wzrd deploy/textureflow/app.py::smoke
    uv run modal run    --env wzrd deploy/textureflow/app.py::matrix

Reference: docs/reference/textureflow.md.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import modal
import modal.experimental

HERE = Path(__file__).parent
COMFY = "/root/ComfyUI"
WEIGHTS = "/weights"
OUTPUTS = "/outputs"
PORT = 8188
# RTX PRO 6000 (Blackwell, 96 GB): fastest *and* cheapest per render in the lab's 2026-10-21 sweep
# (15 s HD: 209 s / $0.18 vs L40S 358 s / $0.19). Needs BLACKWELL_PATCHES (applied in install_nodes).
# Gigs can still pick another card per call: TextureFlow.with_options(gpu="H100").
GPU = "RTX-PRO-6000"
BOOT_TIMEOUT_S = 120
# Node-pack requirements resolve as of the March 2026 Eden deploy, not today.
EXCLUDE_NEWER = "2026-03-17T23:59:59Z"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# Output node id -> name in the response (the names the WZRD client uses).
OUTPUT_NODES = {
    "249": "output_video",
    "226": "first_pass_video",
    "131": "control_signal_preview",
    "69": "mapping_motion_preview",
}

# `output_codec` (not in api.yaml, which stays verbatim): VHS settings for the
# two diffusion outputs (249 final, 226 first pass). The graph ships HEVC 10-bit,
# which many decoders can't play; h264 is the default (x264 crf 19 ≈ x265 crf 22).
OUTPUT_CODECS = {
    "h264": {"format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19},
    "hevc10": {"format": "video/h265-mp4", "pix_fmt": "yuv420p10le", "crf": 22},
}
CODEC_NODES = ("249", "226")
EXTRA_PARAMS = {"output_codec": {"type": "string", "default": "h264", "choices": list(OUTPUT_CODECS)}}

# Fast boot: GPU memory snapshot taken after one tiny warm-up render, so a cold
# container restores with SD1.5 + AnimateLCM + IP-Adapter + CLIP-ViT-H + a
# ControlNet + RIFE already in VRAM. Any redeploy re-takes it on the next cold start.
GPU_SNAPSHOT = True
WARMUP_ARGS = {
    "images": ["w0.png", "w1.png", "w2.png"], "n_seconds": 2, "width": 320, "height": 320, "n_steps": 4,
    "use_upscale": False, "upscale_esrgan": False, "use_controlnet1": True, "control_input": "shape.png",
    "preprocessor1": "Scribble_XDoG_Preprocessor", "seed": 0,
}


def _warmup_files() -> dict[str, bytes]:
    """Generated warm-up inputs (no network, no volume): 3 colour images + a ring."""
    import io

    from PIL import Image, ImageDraw

    def png(im) -> bytes:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    files = {f"w{i}.png": png(Image.new("RGB", (512, 512), c))
             for i, c in enumerate([(200, 60, 40), (40, 160, 90), (50, 70, 210)])}
    shape = Image.new("RGB", (320, 320), "black")
    ImageDraw.Draw(shape).ellipse((60, 60, 260, 260), outline="white", width=24)
    files["shape.png"] = png(shape)
    return files

app = modal.App("wzrd-textureflow")
models_vol = modal.Volume.from_name("textureflow-models", create_if_missing=True)
outputs_vol = modal.Volume.from_name("textureflow-outputs", create_if_missing=True)
warm_state = modal.Dict.from_name("textureflow-warm", create_if_missing=True)  # {"until": epoch}

if modal.is_local():
    SNAPSHOT = json.loads((HERE / "workflow" / "snapshot.json").read_text())
else:  # in a container the image is hydrated by id; the definition is not rebuilt
    SNAPSHOT = {"comfyui": "", "comfyui_repo": "", "custom_nodes": {}}


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def install_nodes(nodes: dict[str, str], requirements: bool = True, patches: dict | None = None) -> None:
    """git clone + checkout each pinned pack, then (`requirements=True`, the
    unlocked path) its requirements under the torch constraints. Mirrors
    eve/comfyui.py install_custom_node minus retries; like eve, a failing
    requirements file is logged, not fatal."""
    failed = []
    for url, sha in nodes.items():
        dest = Path(COMFY) / "custom_nodes" / url.rstrip("/").split("/")[-1]
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)
        subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init", "--recursive"], check=True)
        for req in sorted(dest.rglob("requirements*")) if requirements else []:
            if req.suffix not in (".txt", ".pip") or "with-cupy" in req.name or ".git" in req.parts:
                continue
            print(f"[nodes] {dest.name}: {req.relative_to(dest)}", flush=True)
            r = subprocess.run(["uv", "pip", "install", "--system", "-r", str(req),
                                "-c", "/root/constraints.txt", "--exclude-newer", EXCLUDE_NEWER])
            if r.returncode:
                failed.append(str(req))
    # Print-spam flag (eve patches it too).
    helper = Path(COMFY) / "custom_nodes/ControlFlowUtils/helper.py"
    if helper.exists():
        helper.write_text(helper.read_text().replace("DEBUG_MODE = True", "DEBUG_MODE = False"))
    for rel, (before, after) in (patches or {}).items():
        f = Path(COMFY) / "custom_nodes" / rel
        text = f.read_text()
        if before not in text:
            raise RuntimeError(f"Blackwell patch no longer applies to {rel} (pack bumped?)")
        f.write_text(text.replace(before, after))
    print(f"[nodes] requirements that failed to install: {failed or 'none'}", flush=True)


# Blackwell (sm_100/sm_120): xformers 0.0.35's only fp32 attention kernel is built for <= sm_90, and
# DepthAnythingV2's DINOv2 runs in fp32 -> "No operator found for memory_efficient_attention_forward".
# Fall back to the plain attention there; older GPUs keep xformers (bit-identical). Same patch as
# textureflow-lab's vendor/ tree (its vendor/VENDOR.md).
BLACKWELL_PATCHES = {
    "comfyui_controlnet_aux/src/custom_controlnet_aux/depth_anything_v2/dinov2_layers/attention.py": (
        "        if not XFORMERS_AVAILABLE:\n            assert attn_bias is None",
        "        import torch\n\n"
        "        no_fp32_kernel = x.is_cuda and x.dtype == torch.float32 and torch.cuda.get_device_capability(x.device)[0] >= 10\n"
        "        if not XFORMERS_AVAILABLE or no_fp32_kernel:\n            assert attn_bias is None",
    ),
}


# Every Python dep after torch comes from requirements.lock (exact pins, so a
# rebuild can't drift). To refresh the lock after bumping a node pack: set
# LOCKED = False (re-resolves the packs' requirements as of EXCLUDE_NEWER, then
# Eve's late pins), deploy, run `::lock`, set it back.
LOCKED = True

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0", "libmagic1", "libegl1",
                 # pycairo (svglib, via comfyui_controlnet_aux's requirements) has no Linux
                 # wheel; without these that whole requirements file fails to install.
                 "build-essential", "pkg-config", "libcairo2-dev")
    .pip_install("uv==0.12.19")
    .add_local_file(HERE / "constraints.txt", "/root/constraints.txt", copy=True)
    .run_commands(
        "uv pip install --system --index-url " + TORCH_INDEX
        + " torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers==0.0.35",
        "uv pip install --system sageattention==1.0.6 -c /root/constraints.txt",
    )
)
if LOCKED:
    image = image.add_local_file(HERE / "requirements.lock", "/root/requirements.lock", copy=True).run_commands(
        "uv pip install --system -r /root/requirements.lock --extra-index-url " + TORCH_INDEX
        + " --index-strategy unsafe-best-match"
    )
image = image.run_commands(
    f"git clone -q {SNAPSHOT['comfyui_repo']} {COMFY}",
    f"git -C {COMFY} checkout -q {SNAPSHOT['comfyui']}",
    *([] if LOCKED else [
        f"uv pip install --system -r {COMFY}/requirements.txt -c /root/constraints.txt --exclude-newer {EXCLUDE_NEWER}"]),
).run_function(install_nodes, kwargs={"nodes": SNAPSHOT["custom_nodes"], "requirements": not LOCKED,
                                      "patches": BLACKWELL_PATCHES})  # kwargs = part of the layer hash
if not LOCKED:  # Eve's late pins, applied after the nodes (some packs rely on them).
    image = image.run_commands(
        "uv pip install --system -c /root/constraints.txt diffusers==0.31.0 moviepy==1.0.3"
        " accelerate==1.4.0 peft==0.14.0 transformers==4.49.0 safetensors==0.5.3"
    )
image = (
    image.env({"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})  # every weight is baked; fail loudly, never lazy-download
    # Last, so editing the graph / api.yaml never rebuilds the heavy layers.
    .add_local_dir(HERE / "workflow", "/root/workflow")
    .add_local_file(HERE / "models.json", "/root/models.json")
)

fetch_image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install(
    "requests==2.32.3", "pillow==11.0.0"
).add_local_file(HERE / "models.json", "/root/models.json")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@app.function(image=fetch_image, volumes={WEIGHTS: models_vol}, cpu=4, timeout=3 * 3600)
def download_models(verify: bool = False) -> None:
    """Fill only the missing weights from public URLs (sha256-verified).
    `verify=True` also re-hashes files already present."""
    import hashlib

    import requests

    rows = json.loads(Path("/root/models.json").read_text())["files"]
    for row in rows:
        dest = Path(WEIGHTS) / row["path"]
        if dest.exists() and (row.get("bytes") in (None, dest.stat().st_size)) and not verify:
            print(f"present   {row['path']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if row["url"] == "generate":
            if not dest.exists():
                _generate_dummy(dest)
            print(f"generated {row['path']}")
            continue
        if not dest.exists():
            if not row["url"]:
                print(f"MISSING   {row['path']} (no public source; seed it from Eden, see seed_from_eden.py)")
                continue
            tmp = dest.with_name(dest.name + ".part")
            with requests.get(row["url"], stream=True, timeout=60, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(16 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            status = "fetched"
        else:
            status = "verified"
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if row.get("sha256") and digest != row["sha256"]:
            dest.unlink()
            raise RuntimeError(f"sha256 mismatch for {row['path']}: got {digest}, want {row['sha256']}")
        print(f"{status:9} {row['path']}  {dest.stat().st_size / 1e9:.2f} GB  sha256={digest}")
    models_vol.commit()


def _generate_dummy(dest: Path) -> None:
    if dest.suffix == ".mp4":
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=512x512:d=2:r=8",
                        "-pix_fmt", "yuv420p", str(dest)], check=True, capture_output=True)
    else:
        from PIL import Image

        Image.new("RGB", (512, 512), "white").save(dest)


def link_weights() -> None:
    """Symlink every file in the weights volume to the same ComfyUI-relative path."""
    root = Path(WEIGHTS)
    for src in root.rglob("*"):
        if not src.is_file() or src.name.endswith(".part"):
            continue
        dst = Path(COMFY) / src.relative_to(root)
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


# ---------------------------------------------------------------------------
# Args: api.yaml defaults + validation + graph injection
# ---------------------------------------------------------------------------

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")  # what LoadRandomImage accepts (lowercase)
VIDEO_EXTS = (".mp4", ".mov")  # what Eden_AllMediaLoader decodes as video
MEDIA_PARAMS = ("images", "control_input", "diffusion_mask")


def _is_media(p: dict) -> bool:
    return "anyOf" in p or (p.get("type") == "array" and p.get("items", {}).get("type") in ("image", "video"))


def resolve_args(args: dict, spec: dict, files: dict) -> dict:
    """api.yaml defaults for omitted args, then type/range/choice validation.
    Raises ValueError listing every problem; nothing reaches ComfyUI first."""
    params = {**spec["parameters"], **EXTRA_PARAMS}
    errors = [f"unknown arg '{k}'" for k in args if k not in params]
    out: dict = {}
    for name, p in params.items():
        v = args.get(name)
        if name == "seed" and v in (None, -1, "random"):
            out[name] = random.randint(0, 2**31 - 1)
            continue
        if v is None:
            if "default" in p:
                out[name] = p["default"]
            elif p.get("required"):
                errors.append(f"'{name}' is required")
            continue
        err = _check(name, v, p, files)
        if err:
            errors.append(err)
        else:
            out[name] = _coerce(v, p)
    if errors:
        raise ValueError("Invalid TextureFlow args:\n  - " + "\n  - ".join(errors))
    return out


def _check(name: str, v, p: dict, files: dict) -> str | None:
    def media_ok(x) -> bool:
        return isinstance(x, str) and (x.startswith(("http://", "https://")) or x in files)

    if _is_media(p):
        if p.get("type") == "array":
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                return f"'{name}' must be a list of URLs / file keys"
            lo, hi = p.get("min_length", 0), p.get("max_length", 10**9)
            if not lo <= len(v) <= hi:
                return f"'{name}' needs {lo}–{hi} items, got {len(v)}"
            bad = [x for x in v if not media_ok(x)]
            return f"'{name}': not an http(s) URL or a key in files: {bad}" if bad else None
        return None if media_ok(v) else f"'{name}': {v!r} is not an http(s) URL or a key in files"
    t = p.get("type")
    if "choices" in p:
        return None if v in p["choices"] else f"'{name}': {v!r} is not one of {p['choices']}"
    if t == "boolean":
        return None if isinstance(v, bool) else f"'{name}' must be true/false, got {v!r}"
    if t in ("integer", "float"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"'{name}' must be a number, got {v!r}"
        if t == "integer" and float(v) != int(v):
            return f"'{name}' must be an integer, got {v!r}"
        lo, hi = p.get("minimum"), p.get("maximum")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            return f"'{name}' must be in [{lo}, {hi}], got {v!r}"
        if t == "integer" and p.get("step") and lo is not None and (int(v) - lo) % p["step"]:
            return f"'{name}' must be {lo} + a multiple of {p['step']}, got {v!r}"
        return None
    if t == "string":
        return None if isinstance(v, str) else f"'{name}' must be a string"
    return None


def _coerce(v, p: dict):
    t = p.get("type")
    if t == "integer":
        return int(v)
    if t == "float":
        return float(v)
    if t == "array" and isinstance(v, str):
        return [v]
    return v


def inject(workflow: dict, spec: dict, values: dict) -> dict:
    """Write each resolved value into its node per api.yaml's comfyui block
    (comma-separated subfields + remap). Port of eve/comfyui.py
    _inject_args_into_workflow without the LoRA/Mongo branches."""
    for name, p in spec["parameters"].items():
        if name not in values:
            continue
        c, value = p["comfyui"], values[name]
        for sub in str(c["subfield"]).split(","):
            workflow[str(c["node_id"])][c["field"]][sub.strip()] = value
        for remap in c.get("remap") or []:
            mapped = remap["map"].get(value)
            if mapped is not None:
                for sub in str(remap["subfield"]).split(","):
                    workflow[str(remap["node_id"])][remap["field"]][sub.strip()] = mapped
    if "output_codec" in values:
        for node_id in CODEC_NODES:
            workflow[node_id]["inputs"].update(OUTPUT_CODECS[values["output_codec"]])
    return workflow


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "file"
    stem, ext = os.path.splitext(name)
    return stem[:80] + ext.lower()  # lowercase ext: LoadRandomImage skips .PNG/.WEBP


def _fetch(ref: str, files: dict, dest: Path) -> Path:
    import requests

    if ref in files:
        data, name = files[ref], os.path.basename(ref)
    else:
        r = requests.get(ref, timeout=120, headers={"User-Agent": "wzrd-textureflow/1"})
        if r.status_code != 200:
            raise ValueError(f"could not download {ref}: HTTP {r.status_code}")
        data = r.content
        name = os.path.basename(ref.split("?")[0].split("#")[0])
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / _safe_name(name)
    path.write_bytes(data)
    return path


def _normalize(path: Path, allow_video: bool) -> Path:
    """Make the file loadable by the Eden nodes: a decodable image with a
    lowercase png/jpg/webp/bmp ext, or (media params) .gif / .mp4 / .mov.
    Anything else that decodes (extensionless, .avif, .webm, …) is converted."""
    from PIL import Image, ImageSequence

    ext = path.suffix.lower()
    if allow_video and ext in VIDEO_EXTS + (".gif",):
        if ext != ".gif" and not _has_video_stream(path):
            raise ValueError(f"{path.name}: not a decodable video")
        return path
    try:
        with Image.open(path) as im:
            im.verify()
        is_image = True
    except Exception:  # noqa: BLE001 — PIL raises a zoo of types on junk
        is_image = False
    if is_image:
        if ext in IMG_EXTS:
            return path
        with Image.open(path) as im:
            if allow_video and getattr(im, "n_frames", 1) > 1:
                out = path.with_suffix(".gif")
                im.save(out, save_all=True)
            else:
                out = path.with_suffix(".png")
                next(ImageSequence.Iterator(im)).convert("RGB").save(out)
        if out != path:
            path.unlink(missing_ok=True)
        return out
    if allow_video and _has_video_stream(path):
        out = path.with_suffix(".mp4")
        if out == path:
            out = path.with_name(path.stem + "_t.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "16", str(out)], check=True)
        return out
    raise ValueError(f"{path.name}: not a decodable image{' or video' if allow_video else ''}")


def _has_video_stream(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


# ---------------------------------------------------------------------------
# The GPU class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS: models_vol, OUTPUTS: outputs_vol},
    timeout=3600,
    startup_timeout=600,
    scaledown_window=300,
    min_containers=0,
    max_containers=3,
    # GPU memory snapshot: boot + warm-up run once per deploy (and GPU type);
    # later cold starts restore ComfyUI with its models already in VRAM.
    enable_memory_snapshot=GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": GPU_SNAPSHOT},
)
@modal.concurrent(max_inputs=1)
class TextureFlow:
    @modal.enter(snap=True)
    def start(self):
        import yaml

        t0 = time.time()
        link_weights()
        self.spec = yaml.safe_load(Path("/root/workflow/api.yaml").read_text())
        self.graph = json.loads(Path("/root/workflow/workflow_api.json").read_text())
        # --highvram keeps every loaded model on the GPU (it also switches off
        # ComfyUI's dynamic-VRAM loader), so the warm-up below leaves them resident.
        self.proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch", "--highvram"],
            cwd=COMFY,
        )
        self._wait_healthy(BOOT_TIMEOUT_S)
        self.served = 0
        print(f"[textureflow] ComfyUI up in {time.time() - t0:.1f}s", flush=True)
        if GPU_SNAPSHOT:
            w = self._render(WARMUP_ARGS, _warmup_files(), persist=False)
            print(f"[textureflow] warm-up render {w['timings']['render_s']}s", flush=True)
        self.boot_s = round(time.time() - t0, 2)
        self.restore_s = None

    @modal.enter(snap=False)
    def restored(self):
        """Runs on every container start, after a snapshot restore (or right
        after `start` when no snapshot was used)."""
        t0 = time.time()
        random.seed()  # the RNG state is in the snapshot: unseeded, every restore draws the same "random" seeds
        link_weights()
        self._wait_healthy(30)
        self.restore_s = round(time.time() - t0, 2)

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while not self._healthy():
            if self.proc.poll() is not None or time.time() > deadline:
                modal.experimental.stop_fetching_inputs()
                raise RuntimeError(f"ComfyUI did not come up within {timeout_s}s (exit={self.proc.poll()})")
            time.sleep(0.5)

    def _healthy(self) -> bool:
        import requests

        try:
            return requests.get(f"http://127.0.0.1:{PORT}/system_stats", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    @modal.method()
    def render(self, args: dict, files: dict | None = None) -> dict:
        cold, self.served = self.served == 0, self.served + 1
        res = self._render(args, files or {}, persist=True)
        res["timings"].update(cold_start=cold, boot_s=self.boot_s, restore_s=self.restore_s)
        return res

    def _render(self, args: dict, files: dict, persist: bool) -> dict:
        """One ComfyUI run. `persist=False` (the snapshot warm-up) skips the outputs volume."""
        import requests

        t0 = time.time()
        values = resolve_args(dict(args), self.spec, files)  # ValueError before any work
        request_id = f"tf-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        in_dir = Path(COMFY) / "input" / request_id
        out_root, tmp_root = Path(COMFY) / "output", Path(COMFY) / "temp"
        before = {p for d in (out_root, tmp_root) if d.exists() for p in d.rglob("*")}

        try:
            graph_values = dict(values)
            for name in MEDIA_PARAMS:
                if name not in values:
                    continue
                if name == "images":
                    folder = in_dir / "images"
                    for i, ref in enumerate(values[name]):
                        p = _normalize(_fetch(ref, files, in_dir / "raw"), allow_video=False)
                        folder.mkdir(parents=True, exist_ok=True)
                        shutil.move(p, folder / f"{i:06d}_{p.name}")
                    graph_values[name] = str(folder)
                else:
                    p = _normalize(_fetch(values[name], files, in_dir / name), allow_video=True)
                    graph_values[name] = str(p)
            workflow = inject(json.loads(json.dumps(self.graph)), self.spec, graph_values)
            t_prep = time.time()

            if self.proc.poll() is not None or not self._healthy():
                modal.experimental.stop_fetching_inputs()
                raise RuntimeError("ComfyUI server is down; container stops taking inputs")
            r = requests.post(f"http://127.0.0.1:{PORT}/prompt",
                              json={"prompt": workflow, "client_id": request_id}, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected the prompt: {r.text}")
            prompt_id = r.json()["prompt_id"]
            t_post = time.time()
            try:
                entry = self._wait(prompt_id)
            except BaseException:
                requests.post(f"http://127.0.0.1:{PORT}/interrupt", timeout=5)
                raise

            outputs, rid_dir = {}, Path(OUTPUTS) / request_id
            for node_id, name in OUTPUT_NODES.items():
                for g in (entry.get("outputs", {}).get(node_id) or {}).get("gifs", []):
                    src = Path(g.get("fullpath") or out_root / g.get("subfolder", "") / g["filename"])
                    if src.suffix.lower() not in (".mp4", ".mov", ".webm"):
                        continue
                    data = src.read_bytes()
                    if persist:
                        rid_dir.mkdir(parents=True, exist_ok=True)
                        (rid_dir / f"{name}{src.suffix}").write_bytes(data)
                    outputs[name] = {"filename": src.name, "bytes": data,
                                     "mime": "video/quicktime" if src.suffix == ".mov" else "video/mp4"}
            if persist:
                (rid_dir / "params.json").write_text(json.dumps(values, indent=2))
                outputs_vol.commit()

            ts = {m[0]: m[1].get("timestamp") for m in entry.get("status", {}).get("messages", [])}
            start_ms, end_ms = ts.get("execution_start"), ts.get("execution_success")
            return {
                "request_id": request_id,
                "outputs": outputs,
                "params": values,
                "timings": {
                    "prep_s": round(t_prep - t0, 2),
                    "queue_s": round(start_ms / 1000 - t_post, 2) if start_ms else None,
                    "render_s": round((end_ms - start_ms) / 1000, 2) if start_ms and end_ms else None,
                    "total_s": round(time.time() - t0, 2),
                },
            }
        finally:
            shutil.rmtree(in_dir, ignore_errors=True)
            for d in (out_root, tmp_root):
                for p in sorted(set(d.rglob("*")) - before, reverse=True) if d.exists() else []:
                    p.unlink() if p.is_file() or p.is_symlink() else shutil.rmtree(p, ignore_errors=True)

    def _wait(self, prompt_id: str) -> dict:
        import requests

        while True:
            if self.proc.poll() is not None:
                modal.experimental.stop_fetching_inputs()
                raise RuntimeError(f"ComfyUI crashed mid-render (exit={self.proc.returncode})")
            h = requests.get(f"http://127.0.0.1:{PORT}/history/{prompt_id}", timeout=10).json()
            entry = h.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    for kind, msg in status.get("messages", []):
                        if kind == "execution_error":
                            raise RuntimeError(
                                f"ComfyUI execution_error in node {msg.get('node_id')} ({msg.get('node_type')}): "
                                f"{msg.get('exception_type')}: {msg.get('exception_message')}"
                            )
                    raise RuntimeError(f"ComfyUI error: {status}")
                if status.get("completed", True):
                    return entry
            time.sleep(1)


# ---------------------------------------------------------------------------
# Show-day warmth: keep one container up for N minutes, then cool down
# ---------------------------------------------------------------------------

ctl_image = modal.Image.debian_slim(python_version="3.11")
MAX_WARM_MIN = 12 * 60


@app.function(image=ctl_image, timeout=60)
def set_warm(minutes: float) -> dict:
    """Keep one TextureFlow container running for `minutes` from now (0 = cool
    down now). Calling again moves the deadline. The cool-down timer runs here
    on Modal, so a closed laptop can't leave a GPU on. No redeploy involved; a
    redeploy also resets to scale-to-zero."""
    minutes = max(0.0, min(float(minutes), MAX_WARM_MIN))
    until = time.time() + minutes * 60 if minutes else 0.0
    warm_state["until"] = until
    TextureFlow().update_autoscaler(min_containers=1 if until else 0)
    if until:
        cool_down.spawn()
    return {"warm": bool(until), "until_epoch": until, "minutes": minutes}


@app.function(image=ctl_image, timeout=MAX_WARM_MIN * 60 + 600)
def cool_down() -> None:
    """Sleep until the (possibly extended) deadline, then back to scale-to-zero."""
    while (left := warm_state.get("until", 0.0) - time.time()) > 0:
        time.sleep(min(left, 60))
    TextureFlow().update_autoscaler(min_containers=0)


@app.local_entrypoint()
def warm(minutes: float = 60):
    """`modal run --env wzrd app.py::warm --minutes 120` (0 = cool down now). Targets the deployed app."""
    fn = modal.Function.from_name("wzrd-textureflow", "set_warm", environment_name="wzrd")
    print(fn.remote(minutes))


@app.function(image=image, timeout=300)
def freeze() -> str:
    """`uv pip freeze` of the render image (CPU only) — the source of requirements.lock."""
    return subprocess.run(["uv", "pip", "freeze", "--system"], capture_output=True, text=True, check=True).stdout


@app.local_entrypoint()
def lock():
    text = freeze.remote()
    (HERE / "requirements.lock").write_text(
        "# uv pip freeze of the working wzrd-textureflow image (Phase 1, 2026-10-09).\n" + text)
    print(f"{len(text.splitlines())} packages -> {HERE / 'requirements.lock'}")


@app.function(image=image, gpu=GPU, volumes={WEIGHTS: models_vol}, timeout=3600, scaledown_window=300,
              max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.web_server(PORT, startup_timeout=300, requires_proxy_auth=True)
def ui():
    """ComfyUI's browser UI on the same image, for graph debugging/editing.
    Load workflow/workflow.json, tweak, export API format, re-vendor."""
    link_weights()
    subprocess.Popen(["python", "main.py", "--listen", "0.0.0.0", "--port", str(PORT)], cwd=COMFY)


# ---------------------------------------------------------------------------
# Local entrypoints: smoke + arg matrix
# ---------------------------------------------------------------------------

TEST_INPUTS = HERE / "test_inputs"
RESULTS = HERE / "test_results"
STYLE_URLS = [
    "https://d14i3advvh2bvd.cloudfront.net/53c044a1d7fcc46587ba82ff95c1241f56a4b1d62f95241569d26a15d7aff39a_2560.webp",
    "https://d14i3advvh2bvd.cloudfront.net/66890d4e0d7d975d4a10e0dc9ba0b9e61d15d2deaaf1ceeb0c5a372f8f50e122_2560.webp",
    "https://d14i3advvh2bvd.cloudfront.net/c0379a280e07ab777193b4148e652bb1f830c1f757fd09d35e824b62c12050b9_2560.webp",
]
# test.json from the Eden workspace == the 2026-03-17 reference render (seed 42)
SMOKE_ARGS = {
    "images": STYLE_URLS, "n_seconds": 2.5, "n_steps": 4, "width": 384, "height": 384, "seed": 42,
    "use_upscale": False, "upscale_esrgan": False, "use_controlnet1": False,
    "preprocessor1": "DepthAnythingV2Preprocessor", "controlnet_strength1": 0.5,
}


def _save(case: str, res: dict) -> dict:
    d = RESULTS / case
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, o in res["outputs"].items():
        p = d / f"{name}{Path(o['filename']).suffix}"
        p.write_bytes(o["bytes"])
        paths[name] = str(p)
    (d / "result.json").write_text(json.dumps(
        {"request_id": res["request_id"], "params": res["params"], "timings": res["timings"], "outputs": paths},
        indent=2))
    return paths


def _client(deployed: bool):
    if deployed:
        return modal.Cls.from_name("wzrd-textureflow", "TextureFlow", environment_name="wzrd")()
    return TextureFlow()


@app.local_entrypoint()
def smoke(deployed: bool = False, gpu: str = "", out: str = "smoke"):
    """`--gpu A100-40GB` renders on another GPU type (parity experiments)."""
    t0 = time.time()
    cls = _client(deployed)
    if gpu:
        cls = (modal.Cls.from_name("wzrd-textureflow", "TextureFlow", environment_name="wzrd") if deployed
               else TextureFlow).with_options(gpu=gpu)()
    res = cls.render.remote(SMOKE_ARGS, {})
    paths = _save(out, res)
    print(json.dumps({"timings": res["timings"], "wall_s": round(time.time() - t0, 1),
                      "seed": res["params"]["seed"], "outputs": paths}, indent=2))


def _matrix_cases() -> list[tuple[str, dict, dict, str | None]]:
    """(case, args, files, expect_error_substring). Short renders by default:
    2 s / 320² (api.yaml minimum) / 4 steps / upscale off."""
    ti = TEST_INPUTS
    b = lambda p: (ti / p).read_bytes()  # noqa: E731
    style = {f"style{i}.jpg": b(n) for i, n in enumerate(["img_00019.jpg", "img_00021.jpg", "img_00024.jpg"])}
    base = {"images": list(style), "n_seconds": 2, "width": 320, "height": 320, "n_steps": 4,
            "use_upscale": False, "upscale_esrgan": False, "seed": 7}
    shape = {"shape.png": b("shape.png")}
    cn = {**base, "use_controlnet1": True, "control_input": "shape.png", "controlnet_strength1": 0.6}
    f_style_shape = {**style, **shape}
    video_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    image_url = "https://storage.googleapis.com/public-assets-xander/A_workbox/init_imgs/img_00024.jpg"
    cases = []
    for pp in ["Scribble_XDoG_Preprocessor", "CannyEdgePreprocessor", "DepthAnythingV2Preprocessor",
               "AnyLineArtPreprocessor_aux", "OpenposePreprocessor", "none"]:
        cases.append((f"pre_{pp}", {**cn, "preprocessor1": pp}, f_style_shape, None))
    # Pose transfer needs a person: public-domain Wikimedia "Pyei Sandy.jpg" (960px thumb), portrait 2:3.
    cases.append(("pose_person", {**cn, "preprocessor1": "OpenposePreprocessor", "control_input": "person.jpg",
                                  "controlnet_strength1": 0.8, "height": 480},
                  {**style, "person.jpg": b("person.jpg")}, None))
    spec = json_yaml(HERE / "workflow" / "api.yaml")
    for mm in spec["parameters"]["mapping_mode"]["choices"]:
        cases.append((f"map_{mm}", {**base, "mapping_mode": mm}, style, None))
    for bm in spec["parameters"]["base_model"]["choices"]:
        cases.append((f"model_{Path(bm).stem.split('_')[0]}", {**base, "base_model": bm}, style, None))
    wide = {"wide.png": b("wide.png")}
    for fs in spec["parameters"]["control_input_fit_strategy"]["choices"]:
        cases.append((f"fit_{fs.replace(' / ', '_')}",
                      {**cn, "control_input": "wide.png", "control_input_fit_strategy": fs}, {**style, **wide}, None))
    cases += [
        ("control_mp4_bytes", {**cn, "control_input": "control.mp4"}, {**style, "control.mp4": b("control.mp4")}, None),
        ("control_image_url", {**cn, "control_input": image_url}, style, None),
        ("control_mp4_url", {**cn, "control_input": video_url, "preprocessor1": "CannyEdgePreprocessor"}, style, None),
        ("mask_image_bytes", {**base, "diffusion_mask": "mask.png"}, {**style, "mask.png": b("mask.png")}, None),
        ("mask_mp4_bytes", {**base, "diffusion_mask": "mask.mp4"}, {**style, "mask.mp4": b("mask.mp4")}, None),
        ("mask_mp4_url", {**base, "diffusion_mask": video_url}, style, None),
        ("map_shape_to_ip_masks", {**cn, "control_input": "regions.png", "map_shape_input_to_ip_masks": True},
         {**style, "regions.png": b("regions.png")}, None),
        ("upscale_esrgan", {**base, "use_upscale": True, "upscale_esrgan": True, "upscale_resolution": 1024}, style, None),
        ("upscale_latent_only", {**base, "use_upscale": True, "upscale_esrgan": False, "upscale_resolution": 1088},
         style, None),
        ("denoise_0.7", {**cn, "denoise": 0.7, "motion_scale": 0.9}, f_style_shape, None),
        ("style_1", {**base, "images": ["style0.jpg"]}, style, None),
        ("style_6", {**base, "images": [*style, *STYLE_URLS]}, style, None),
        ("size_512x320", {**base, "width": 512, "height": 320}, style, None),
        ("n_seconds_24", {**base, "n_seconds": 24}, style, None),
        ("n_steps_14", {**base, "n_steps": 14}, style, None),
        ("seed_omitted_a", {k: v for k, v in base.items() if k != "seed"}, style, None),
        ("seed_omitted_b", {**base, "seed": -1}, style, None),
        ("uppercase_exts", {**base, "images": ["A.PNG", "B.WEBP", "C.JPG"]},
         {"A.PNG": b("upper/A.PNG"), "B.WEBP": b("upper/B.WEBP"), "C.JPG": b("img_00019.jpg")}, None),
        ("odd_formats", {**base, "images": ["noext", "anim.gif"]},
         {"noext": b("img_00021.jpg"), "anim.gif": b("anim.gif")}, None),
        ("defaults_only", {"images": list(style)}, style, None),
        # Invalid: must be rejected by validation, before ComfyUI runs.
        ("bad_width_range", {**base, "width": 256}, style, "'width' must be in"),
        ("bad_width_step", {**base, "width": 330}, style, "multiple of 32"),
        ("bad_n_seconds", {**base, "n_seconds": 30}, style, "'n_seconds' must be in"),
        ("bad_enum_mapping", {**base, "mapping_mode": "spiral"}, style, "'mapping_mode'"),
        ("bad_enum_preproc", {**base, "preprocessor1": "Sobel"}, style, "'preprocessor1'"),
        ("bad_enum_densepose", {**base, "preprocessor1": "DensePosePreprocessor"}, style, "'preprocessor1'"),
        ("bad_type_bool", {**base, "use_upscale": "yes"}, style, "'use_upscale' must be true/false"),
        ("bad_unknown_arg", {**base, "n_second": 3}, style, "unknown arg 'n_second'"),
        ("bad_images_7", {**base, "images": [*style, *STYLE_URLS, STYLE_URLS[0]]}, style, "needs 1–6 items"),
        ("bad_images_missing", {k: v for k, v in base.items() if k != "images"}, style, "'images' is required"),
        ("bad_media_ref", {**base, "diffusion_mask": "nope.png"}, style, "not an http(s) URL or a key in files"),
        ("bad_image_bytes", {**base, "images": ["junk.png"]}, {"junk.png": b"not an image"},
         "not a decodable image"),
    ]
    return cases


def json_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


@app.local_entrypoint()
def matrix(only: str = "", deployed: bool = False, workers: int = 3, out: str = ""):
    """NEVER run the full matrix without Xander's explicit permission: it is ~57
    GPU renders and no longer needed routinely; use `smoke` or `--only`.

    Exercise every api.yaml param + enum value; write test_results/ and a
    summary table. `--only a,b` runs a subset (substring match); `--out dir` writes
    under test_results/dir/ instead of over the previous run."""
    from concurrent.futures import ThreadPoolExecutor

    cases = _matrix_cases()
    if only:
        keys = only.split(",")
        cases = [c for c in cases if any(k in c[0] for k in keys)]
    cls = _client(deployed)

    def run(case):
        name, args, files, expect = case
        t0 = time.time()
        try:
            res = cls.render.remote(args, files)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
            ok = bool(expect) and expect in msg
            return {"case": name, "ok": ok, "wall_s": round(time.time() - t0, 1), "detail": msg[:400]}
        paths = _save(f"{out}/{name}" if out else name, res)
        ok = not expect and "output_video" in res["outputs"]
        return {"case": name, "ok": ok, "wall_s": round(time.time() - t0, 1),
                "render_s": res["timings"]["render_s"], "cold": res["timings"]["cold_start"],
                "seed": res["params"]["seed"], "n_outputs": len(paths),
                "output": paths.get("output_video", ""),
                "detail": "" if not expect else "expected a rejection, got a render"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, cases))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / out).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / out / f"matrix-{stamp}.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['case']:<42} wall={r['wall_s']:>6}s "
              f"render={r.get('render_s')}s cold={r.get('cold')} seed={r.get('seed')} {r.get('detail', '')}")
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} passed -> {RESULTS / out}/matrix-{stamp}.json")
