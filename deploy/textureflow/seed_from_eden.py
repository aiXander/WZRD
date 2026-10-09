"""One-time seed: copy TextureFlow's weights from Eden's private S3 into our volume.

Run once, by hand (not part of the deploy):

    uv run modal run --env wzrd deploy/textureflow/seed_from_eden.py

Reads Eden's STAGE AWS keys from ~/Documents/GitHub/Eden/eve/.env.STAGE on this
machine and hands only those three values to the remote function as an
*ephemeral* secret — nothing is stored in our Modal workspace or committed.
The copy runs server-side (S3 → volume), nothing passes through the laptop.

Volume layout: `textureflow-models` mirrors ComfyUI-relative paths
(`models/checkpoints/...`, `input/...`, `custom_nodes/...`), so the deploy can
symlink `<mount>/<path>` → `ComfyUI/<path>` one-to-one.

Writes `eden_seed_manifest.json` next to this file (path, S3 key, bytes,
sha256) for folding into `models.json`. Re-running skips files already present.
See docs/TODO/textureflow-modal.md §1d.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

EDEN_ENV_FILE = Path("~/Documents/GitHub/Eden/eve/.env.STAGE").expanduser()
BUCKET = "edenartlab-lfs"
S3_PREFIX = "comfyui/models2/"
MOUNT = "/weights"

# ComfyUI-relative target path -> key under S3_PREFIX. Only what the wzrd
# TextureFlow graph needs (the rest of Eden's downloads.json is unused).
FILES = {
    "models/checkpoints/SD15/juggernaut_reborn.safetensors": "checkpoints/juggernaut_reborn.safetensors",
    "models/animatediff_models/AnimateLCM_sd15_t2v.ckpt": "animatediff_models/AnimateLCM_sd15_t2v.ckpt",
    "models/loras/AnimateLCM_sd15_t2v_lora.safetensors": "loras/AnimateLCM_sd15_t2v_lora.safetensors",
    "models/ipadapter/ip-adapter-plus_sd15.safetensors": "ipadapter/ip-adapter-plus_sd15.safetensors",
    "models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    "models/controlnet/control_v11p_sd15_scribble.pth": "controlnet/control_v11p_sd15_scribble.pth",
    "models/controlnet/control_v11p_sd15_canny.pth": "controlnet/control_v11p_sd15_canny.pth",
    "models/controlnet/control_v11p_sd15_lineart.pth": "controlnet/control_v11p_sd15_lineart.pth",
    "models/controlnet/control_v11p_sd15_openpose.pth": "controlnet/control_v11p_sd15_openpose.pth",
    "models/controlnet/control_v11f1p_sd15_depth.pth": "controlnet/control_v11f1p_sd15_depth.pth",
    "models/controlnet/controlnetQRPatternQR_v2Sd15.safetensors": "controlnet/controlnetQRPatternQR_v2Sd15.safetensors",
    "models/upscale_models/RealESRGAN_x2plus.pth": "upscale_models/RealESRGAN_x2plus.pth",
    "models/embeddings/NEG_EMBED_STABLE_YOGI_V3.pt": "embeddings/NEG_EMBED_STABLE_YOGI_V3.pt",
    "input/a_black_video.mp4": "assets/a_black_video.mp4",
    "input/a_white_image.jpg": "assets/a_white_image.jpg",
}

app = modal.App("wzrd-textureflow-seed")
vol = modal.Volume.from_name("textureflow-models", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("boto3")


def _eden_secret() -> modal.Secret:
    # The module is re-imported inside the container, where neither the .env
    # file nor python-dotenv exists. Still declare one Secret there (Modal
    # requires the same dependency list on both sides); the container gets the
    # real values via the object id hydrated from the local run.
    if not modal.is_local():
        return modal.Secret.from_dict({})
    from dotenv import dotenv_values

    env = dotenv_values(EDEN_ENV_FILE)
    keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME")
    return modal.Secret.from_dict({k: env[k] for k in keys if env.get(k)})


@app.function(image=image, volumes={MOUNT: vol}, secrets=[_eden_secret()], cpu=4, timeout=3600)
def seed(files: dict[str, str]) -> list[dict]:
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION_NAME") or "us-east-1")
    cfg = TransferConfig(max_concurrency=16, multipart_chunksize=64 * 1024 * 1024)

    def one(item):
        target, rel_key = item
        key = S3_PREFIX + rel_key
        dest = Path(MOUNT) / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_name(dest.name + ".part")
            s3.download_file(BUCKET, key, str(tmp), Config=cfg)
            tmp.rename(dest)
            status = "copied"
        else:
            status = "present"
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        print(f"{status:8} {dest.stat().st_size / 1e9:6.2f} GB  {target}", flush=True)
        return {"path": target, "s3_key": key, "bytes": dest.stat().st_size,
                "sha256": h.hexdigest(), "status": status}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, files.items()))
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    results = seed.remote(FILES)
    out = Path(__file__).with_name("eden_seed_manifest.json")
    out.write_text(json.dumps(sorted(results, key=lambda r: r["path"]), indent=2) + "\n")
    total = sum(r["bytes"] for r in results)
    print(f"{len(results)} files, {total / 1e9:.1f} GB in volume textureflow-models -> {out}")
