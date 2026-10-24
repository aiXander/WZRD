"""TextureFlow ETA: a per-stage time model × history (render-queue plan Q6).

A render is the pipeline's ordered **laps** (the lab's `info["stage_s"]` keys: `styles`, `control_input`, …,
`sample`, `vae_decode`, `upscale_pass`, `esrgan`, `rife`, `encode`). Each lap's seconds are modelled per
backend as `c + k · x`, where `x` is the lap's work in pixel-frames:

| lap | x |
|---|---|
| `sample` | frames × first-pass px × steps |
| `upscale_pass` | frames × latent-pass px |
| `esrgan` | frames × output px |
| `rife`, `encode` | output frames × output px (a draft: the first pass's px) |
| everything else | frames × first-pass px (mostly the constant) |

`(c, k)` are least-squares fits over **priors** — the 3090's explore-batch samples
(`tf_eta_3090_samples.json` beside this file, the lab's export, loaded if present; plus one built-in bench) and
two RTX PRO 6000 anchors from the lab's `docs/reference/benchmarks.md` — plus **WZRD's own records** (every
texture_flow record keeps `textureflow.stage_s` + sizes since the queue landed, Q7), refit at most once a
minute. `Tracker` turns a plan + the backend's live stage reports into `{pct, eta_s}`.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ._log import _emit

FPS = 8.0  # the diffusion frame rate (lab pipeline.FPS)
SAMPLES_PATH = Path(__file__).with_name("tf_eta_3090_samples.json")
SETUP_LAPS = ("styles", "control_input", "style_mapping", "diffusion_mask", "base_model", "ipadapter",
              "controlnet", "animatediff_setup", "start_latent", "resume", "auto_levels")
ORDER = ("styles", "control_input", "style_mapping", "diffusion_mask", "base_model", "controlnet", "sample",
         "resume", "vae_decode", "auto_levels", "upscale_pass", "esrgan", "rife", "encode")
#: Seconds outside the laps: submit + container/worker pickup + download + library save + HAP.
OVERHEAD_S = {"modal": 12.0, "local": 8.0}
MODAL_COLD_S = 20.0  # a cold Modal container (boot + snapshot restore), added when nothing ran lately
REFIT_EVERY_S = 60.0
RECORDS_LIMIT = 300

# The 3090 without the samples file: `bench_hd_15s` (lab docs/reference/local-gpu.md, 2026-10-22).
_3090_FALLBACK = [{
    "frames": 120, "fp": (768, 448), "out": (1920, 1120), "out_frames": 360, "steps": 8, "up_res": 1280,
    "stage_s": {"styles": 1.0, "control_input": 1.0, "style_mapping": 1.0, "base_model": 1.0, "controlnet": 1.0,
                "sample": 71.0, "vae_decode": 9.0, "auto_levels": 0.5, "upscale_pass": 183.0, "esrgan": 7.0,
                "rife": 48.0, "encode": 87.0},
}]
# RTX PRO 6000 (Modal production), lab benchmarks.md 2026-10-21/22 at today's defaults (NAG, latent pass 1024·6).
_MODAL_ANCHORS = [
    {"frames": 68, "fp": (1024, 576), "out": (1920, 1080), "out_frames": 204, "steps": 10, "up_res": 1024,
     "stage_s": {"styles": 1.0, "control_input": 1.0, "style_mapping": 1.5, "base_model": 1.0, "controlnet": 1.5,
                 "sample": 22.0, "vae_decode": 2.0, "auto_levels": 0.5, "upscale_pass": 34.0, "esrgan": 8.0,
                 "rife": 3.5, "encode": 7.0}},
    {"frames": 120, "fp": (768, 448), "out": (1920, 1120), "out_frames": 360, "steps": 8, "up_res": 1280,
     "stage_s": {"styles": 1.0, "control_input": 1.5, "style_mapping": 2.0, "base_model": 1.0, "controlnet": 2.0,
                 "sample": 24.0, "vae_decode": 3.0, "auto_levels": 0.5, "upscale_pass": 52.0, "esrgan": 13.0,
                 "rife": 7.0, "encode": 8.0}},
]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _defaults() -> dict:
    try:
        from . import tf_spec

        return {k: p.get("default") for k, p in tf_spec.current().params.items()}
    except Exception:  # noqa: BLE001
        return {}


def _snap(aspect: float, pixels: float) -> tuple[int, int]:
    """lab contract.snap_size: a (w, h) of that aspect holding ≈ pixels, each side on a 16-px grid."""
    h = (pixels / aspect) ** 0.5
    return max(16, round(h * aspect / 16) * 16), max(16, round(h / 16) * 16)


def _num(v: Any, default: float) -> float:
    try:
        return float(v) if v is not None and not isinstance(v, bool) else default
    except (TypeError, ValueError):
        return default


def features(args: dict) -> dict:
    """Work sizes of a render from its args (caller-set, recipe-filled, or a draft's resolved params);
    unset args take the loaded spec's defaults."""
    a = {**_defaults(), **{k: v for k, v in args.items() if v is not None}}
    ow, oh = int(_num(a.get("output_width"), 1920)), int(_num(a.get("output_height"), 1080))
    use_up, esr = bool(a.get("use_upscale", True)), bool(a.get("upscale_esrgan", True))
    px = _num(a.get("density"), 0.33) * 1e6 if use_up else ow * oh / (4 if esr else 1)
    fp = _snap(ow / max(1, oh), px)
    frames = max(1, int(round(_num(a.get("n_seconds"), 6.0) * FPS)))
    steps = max(4, int(round(_num(a.get("n_steps"), 10) * _num(a.get("denoise"), 1.0) ** 0.75)))
    draft = bool(a.get("draft"))
    out = fp if draft else (ow, oh)
    return {"frames": frames, "fp": fp, "out": out, "out_frames": frames * int(_num(a.get("rife_multiplier"), 4)),
            "steps": steps, "up_res": int(_num(a.get("upscale_resolution"), 1024)),
            "use_upscale": use_up and not draft, "esrgan": esr and not draft, "draft": draft,
            "finish": bool(a.get("finish_from"))}


def _x(lap: str, f: dict) -> float:
    """The lap's work, in megapixel-frames (megapixel-frame-steps for `sample`)."""
    fp_px = f["fp"][0] * f["fp"][1] / 1e6
    out_px = f["out"][0] * f["out"][1] / 1e6
    if lap == "sample":
        return f["frames"] * fp_px * f["steps"]
    if lap == "upscale_pass":
        return f["frames"] * max(f["up_res"] ** 2 * 9 / 16 / 1e6, fp_px)
    if lap == "esrgan":
        return f["frames"] * out_px
    if lap in ("rife", "encode"):
        return f["out_frames"] * out_px
    return f["frames"] * fp_px


def laps(f: dict) -> list[str]:
    """The laps this render runs, in order (lab pipeline.texture_flow, native path)."""
    skip = {"controlnet", "sample"} if f["finish"] else {"resume"}
    if not f["use_upscale"]:
        skip.add("upscale_pass")
    if not f["esrgan"]:
        skip.add("esrgan")
    return [lap for lap in ORDER if lap not in skip]


# ---------------------------------------------------------------------------
# Observations → fits
# ---------------------------------------------------------------------------
def _obs_from(params: dict, info: dict) -> Optional[dict]:
    """One observation `{f, stage_s}` from resolved params + the server's info sizes (exact when present)."""
    stage_s = info.get("stage_s")
    if not isinstance(stage_s, dict) or not stage_s:
        return None
    f = features({**params, "finish_from": None, "draft": info.get("stage") == "draft"})
    size = info.get("size") or info.get("first_pass_size")
    final = info.get("final_size") or info.get("output_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        f["fp"] = (int(size[0]), int(size[1]))
    if isinstance(final, (list, tuple)) and len(final) == 2:
        f["out"] = (int(final[0]), int(final[1]))
    for k, key in (("n_frames", "frames"), ("n_output_frames", "out_frames"), ("steps", "steps")):
        if isinstance(info.get(k), (int, float)) and info[k] > 0:
            f[key] = int(info[k])
    return {"f": f, "stage_s": {k: float(v) for k, v in stage_s.items() if isinstance(v, (int, float))}}


def _anchor_obs(anchors: list[dict]) -> list[dict]:
    out = []
    for a in anchors:
        f = {"frames": a["frames"], "fp": a["fp"], "out": a["out"], "out_frames": a["out_frames"],
             "steps": a["steps"], "up_res": a["up_res"]}
        out.append({"f": f, "stage_s": a["stage_s"]})
    return out


def _load_samples() -> list[dict]:
    """The lab's 3090 export (L5): a list (or {samples|renders: list}) of {params, stage_s, info?, …}."""
    if not SAMPLES_PATH.is_file():
        return []
    try:
        data = json.loads(SAMPLES_PATH.read_text())
    except (OSError, ValueError) as exc:
        _emit(f"⚠ tf_eta: {SAMPLES_PATH.name} unreadable ({exc}) — using the built-in 3090 bench")
        return []
    if isinstance(data, dict):
        data = data.get("samples") or data.get("renders") or data.get("results") or []
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or item.get("exp"):
            continue  # lab-only experiment knobs: not what production renders
        info = {**{k: item[k] for k in ("size", "final_size", "n_frames", "n_output_frames", "steps",
                                         "first_pass_size", "output_size") if k in item},
                **(item.get("info") or {})}
        info.setdefault("stage_s", item.get("stage_s"))
        ob = _obs_from(item.get("params") or item.get("args") or {}, info)
        if ob:
            out.append(ob)
    return out


def _record_obs(backend: str) -> list[dict]:
    from . import library

    try:
        rows = library.query(
            "SELECT record FROM renders WHERE tool='texture_flow' "
            "AND json_extract(record, '$.textureflow.backend')=? "
            "AND json_extract(record, '$.textureflow.stage_s') IS NOT NULL ORDER BY created DESC LIMIT ?",
            (backend, RECORDS_LIMIT))
    except Exception:  # noqa: BLE001 — no db yet
        return []
    out = []
    for r in rows:
        tf = (library._loads(r["record"], {}) or {}).get("textureflow") or {}
        ob = _obs_from(tf.get("params") or {}, {**tf, "size": tf.get("first_pass_size"),
                                                 "final_size": tf.get("output_size")})
        if ob:
            out.append(ob)
    return out


def _fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares `y = c + k x` with c, k ≥ 0."""
    n = len(points)
    if not n:
        return 0.0, 0.0
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if n >= 2 and sxx > 1e-12:
        k = sum((x - mx) * (y - my) for x, y in points) / sxx
        c = my - k * mx
        if k >= 0 and c >= 0:
            return c, k
        if k < 0:
            return my, 0.0
    sx2 = sum(x * x for x, _ in points)
    return 0.0, (sum(x * y for x, y in points) / sx2) if sx2 > 0 else 0.0


class Model:
    """Per-lap `(c, k)` for one backend, plus how many observations it was fitted on."""

    def __init__(self, obs: list[dict]):
        per: dict[str, list[tuple[float, float]]] = {}
        for ob in obs:
            for lap, s in ob["stage_s"].items():
                per.setdefault(lap, []).append((_x(lap, ob["f"]), s))
        self.coef = {lap: _fit(pts) for lap, pts in per.items()}
        self.n = len(obs)

    def predict(self, lap: str, f: dict) -> float:
        c, k = self.coef.get(lap, (0.5 if lap in SETUP_LAPS else 0.0, 0.0))
        return max(0.0, c + k * _x(lap, f))


_models: dict[str, tuple[float, Model]] = {}
_lock = threading.Lock()


def model(backend: str) -> Model:
    now = time.time()
    with _lock:
        hit = _models.get(backend)
        if hit and now - hit[0] < REFIT_EVERY_S:
            return hit[1]
    # 3090: the lab's export + the built-in bench (the only one with ESRGAN; the explore batches ran none)
    priors = (_load_samples() + _anchor_obs(_3090_FALLBACK)) if backend == "local" else _anchor_obs(_MODAL_ANCHORS)
    m = Model(priors + _record_obs(backend))
    with _lock:
        _models[backend] = (now, m)
    return m


def invalidate() -> None:
    with _lock:
        _models.clear()


# ---------------------------------------------------------------------------
# Plans + live tracking
# ---------------------------------------------------------------------------
def plan(args: dict, backend: str) -> list[tuple[str, float]]:
    """`[(lap, predicted s), …]` for a render of *args* on *backend*."""
    f = features(args)
    m = model(backend)
    return [(lap, round(m.predict(lap, f), 2)) for lap in laps(f)]


def predict_total(args: dict, backend: str, cold: bool = False) -> float:
    return round(sum(s for _, s in plan(args, backend)) + OVERHEAD_S.get(backend, 10.0)
                 + (MODAL_COLD_S if cold and backend == "modal" else 0.0), 1)


TICK_LAPS = {"esrgan": "esrgan", "rife": "rife"}  # progress-bar tick names → the lap running


class Tracker:
    """One job's progress: phase + the backend's stage reports → `{phase, stage, step, total, pct, eta_s,
    source}` (the hub API's `progress` block)."""

    def __init__(self, laps_: list[tuple[str, float]], backend: str, overhead_s: float):
        self.laps = list(laps_)
        self.names = [n for n, _ in self.laps]
        self.pred = dict(self.laps)
        self.backend = backend
        self.overhead_s = overhead_s
        self.phase = "queued"
        self.stage: Optional[str] = None  # the server's last reported stage name
        self.step: Optional[int] = None
        self.total: Optional[int] = None
        self.done: set[str] = set()
        self.running: Optional[str] = None  # the lap running now (our reading of `stage`)
        self.lap_t: Optional[float] = None  # when `running` started
        self.t_sent: Optional[float] = None
        self.t_live: Optional[float] = None  # first live report
        self.t_post: Optional[float] = None  # downloading / saving started
        self.live = False

    def replan(self, laps_: list[tuple[str, float]], backend: str, overhead_s: float) -> None:
        self.__init__(laps_, backend, overhead_s)

    @property
    def predicted(self) -> float:
        return sum(self.pred.values()) + self.overhead_s

    def set_phase(self, phase: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if phase == self.phase:
            return False
        if phase in ("waiting_backend", "starting", "rendering") and self.t_sent is None:
            self.t_sent = now
        if phase in ("downloading", "saving"):
            self.t_post = self.t_post or now
            self.done |= set(self.names)
            self.running, self.step, self.total = None, None, None
        self.phase = phase
        return True

    def on_stage(self, stage: Optional[str], step: Optional[int], total: Optional[int],
                 now: Optional[float] = None) -> bool:
        """A live report (local status poll / Modal Dict). Returns True when the stage changed."""
        now = time.time() if now is None else now
        if not stage:
            return False
        self.live = True
        self.t_live = self.t_live or now
        if self.phase in ("queued", "waiting_backend", "starting"):
            self.set_phase("rendering", now)
        changed = stage != self.stage
        self.stage, self.step, self.total = stage, step, total
        prev = self.running
        if step is None and stage in self.pred:  # a lap finished: the next one runs
            i = self.names.index(stage)
            self.done |= set(self.names[: i + 1])
            self.running = next((n for n in self.names[i + 1:] if n not in self.done), None)
            self.step = self.total = None
        elif stage == "KSampler":  # diffusion ticks: the first pass, else the latent pass
            self.running = "upscale_pass" if ("sample" in self.done or "sample" not in self.pred) else "sample"
        elif stage in TICK_LAPS:
            self.running = TICK_LAPS[stage]
        elif step is None:  # an unknown lap name (the ComfyUI path's `ipadapter`, …): counts as setup
            self.running = self.running or (self.names[0] if self.names else None)
        if self.running and self.running in self.pred:
            for n in self.names[: self.names.index(self.running)]:
                self.done.add(n)
        if self.running != prev:
            self.lap_t = now
        return changed

    def _frac(self, now: float) -> float:
        if not self.running:
            return 0.0
        if self.step is not None and self.total:
            return max(0.0, min(1.0, self.step / self.total))
        p = self.pred.get(self.running) or 0.0
        if p <= 0 or self.lap_t is None:
            return 0.0
        return min(0.95, (now - self.lap_t) / p)

    def snapshot(self, now: Optional[float] = None, wait_s: float = 0.0) -> dict:
        """The API's progress block. *wait_s*: the queue's predicted wait ahead (queued jobs)."""
        now = time.time() if now is None else now
        total = self.predicted
        out = {"phase": self.phase, "stage": self.running or self.stage, "step": self.step,
               "total": self.total, "pct": None, "eta_s": None, "source": "live" if self.live else "estimate"}
        if self.phase == "queued":
            out.update(pct=0.0, eta_s=round(wait_s + total, 1), stage=None)
            return out
        if self.phase in ("downloading", "saving"):
            post = max(1.0, self.overhead_s / 2)
            spent = now - (self.t_post or now)
            out.update(pct=round(min(0.99, 0.97 + 0.02 * min(1.0, spent / post)), 3),
                       eta_s=round(max(1.0, post - spent), 1), stage=None)
            return out
        laps_total = sum(self.pred.values()) or 1.0
        if self.live:
            done_s = sum(self.pred[n] for n in self.done if n in self.pred)
            run_s = (self.pred.get(self.running, 0.0) * self._frac(now)) if self.running else 0.0
            pct = min(0.97, (done_s + run_s) / laps_total)
            remaining = laps_total - done_s - run_s
            # observed speed vs the model over the laps done so far
            ratio = 1.0
            if done_s >= 3.0 and self.t_live is not None and self.lap_t is not None:
                ratio = max(0.33, min(3.0, (self.lap_t - self.t_live) / done_s))
            eta = remaining * ratio + self.overhead_s / 2
        else:
            t0 = self.t_sent or now
            pct = min(0.95, (now - t0) / total) if total > 0 else None
            eta = max(1.0, total - (now - t0)) if total > 0 else None
        out.update(pct=round(pct, 3) if pct is not None else None,
                   eta_s=round(max(0.0, eta), 1) if eta is not None else None)
        return out

    def remaining(self, now: Optional[float] = None) -> float:
        """Seconds until done (for the wait of the jobs queued behind this one)."""
        snap = self.snapshot(now)
        return snap["eta_s"] if snap["eta_s"] is not None else self.predicted
