"""The TextureFlow render queue: the hub's Modal lane + a view of the desktop's own queue (render-queue plan Q1–Q8).

Every `texture_flow` call is *tracked* here from the moment it passed its checks (args, spend cap) until its
files are saved: its backend, its place, and a `tf_eta.Tracker` for the progress block of the job dict.

- **Modal lane (owned by the hub).** The call parks on `acquire_modal(job)`; the dispatcher grants the head of
  the lane a slot while fewer than the operator's Parallel GPUs setting run (`modal_slots()`: library.db
  `settings.tf_modal_slots`, 1..3 = the app's `max_containers`, default 1). A spawned Modal call can't be
  reordered, so jobs beyond the slots wait here.
- **Local lane (owned by the desktop job server).** A local job is submitted as soon as the server answers
  `/v1/health`; the server's SQLite queue orders it (`queue_position` = `lane_pos`; reorder goes to its
  `/v1/queue/order`). The hub only *holds* a local job while the server is unreachable (phase
  `waiting_backend`); if the desktop answers SSH but the service doesn't, `textureflow-ctl start` runs over
  `TEXTUREFLOW_LOCAL_SSH` at most once a minute. The server's GPU worker boots on demand and exits after 10 idle
  min by itself (health `worker: idle`; `waiting_gpu` while a lab batch holds the GPU) — the hub never stops it.
- **Moves:** Modal → local takes the job out of the hub lane and submits it; local → Modal cancels the queued
  server job and spawns from the call's in-memory inputs. A finish is pinned to its draft's backend.
- **Hub restart (Q8):** submitted local jobs keep rendering on the desktop; `adopt_all()` (hub start) re-adopts
  this library's unfinished local jobs (`jobs.remote = local|<server id>`) and collects their results. Modal jobs
  still queued in the hub read `lost`.

Reference: docs/reference/textureflow.md "The render queue".
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import jobs, tf_eta
from ._log import _emit

BACKENDS = ("modal", "local")
SSH = "/usr/bin/ssh"  # absolute: the hub is spawned by a Finder-launched app with no shell PATH
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=4")
CTL_DEFAULT = "~/.local/bin/textureflow-ctl"  # the remote shell expands ~ (a non-login PATH lacks ~/.local/bin)
CTL_TIMEOUT_S = 60.0
START_EVERY_S = 60.0  # at most one `textureflow-ctl start` a minute while local jobs are held
HEALTH_MAX_AGE_S = 2.0
CTL_MAX_AGE_S = 20.0
HOLD_POLL_S = 2.0
MODAL_WARM_S = 300.0  # deploy.py scaledown_window: a container lingers this long after a render
LOCAL_BOOT_S = 45.0  # queue-wait guess when the desktop's worker isn't loaded (idle / starting / held)
TICK_S = 1.0
HELD_RANK = 1_000_000  # lane_rank of a held local job (after everything already on the server)
UP_WORKER = ("ready", "booting", "idle", "waiting_gpu", "down")  # the API answers: submit (down fails it at once)


class Moved(Exception):
    """The operator moved this queued job to the other backend: the call re-routes."""


MAX_MODAL_SLOTS = 3  # the app's max_containers (textureflow-lab deploy.py)
SLOTS_SETTING = "tf_modal_slots"


def modal_slots() -> int:
    """How many Modal GPU containers may render at once: the operator's setting (library.db `settings`,
    1..MAX_MODAL_SLOTS); `TEXTUREFLOW_MODAL_SLOTS` only seeds it, default 1."""
    from . import library

    try:
        v = library.get_setting(SLOTS_SETTING, None)
        if v is None:
            v = int(os.getenv("TEXTUREFLOW_MODAL_SLOTS") or 1)
        return max(1, min(MAX_MODAL_SLOTS, int(v)))
    except (TypeError, ValueError):
        return 1


def set_modal_slots(n: Any) -> dict:
    """`POST /api/tf/modal {slots}`: takes effect on the next dispatch; lowering it never cancels a running
    job (the lane drains down to the new limit). Returns the modal block."""
    from . import library

    if isinstance(n, bool) or not isinstance(n, (int, float)) or int(n) != n or not 1 <= n <= MAX_MODAL_SLOTS:
        raise ValueError(f"slots must be an integer 1-{MAX_MODAL_SLOTS}")
    library.set_setting(SLOTS_SETTING, int(n))
    _kick()
    jobs.bump()
    return _modal_block()


def ssh_host() -> Optional[str]:
    return (os.getenv("TEXTUREFLOW_LOCAL_SSH") or "").strip() or None


def local_configured() -> bool:
    return bool((os.getenv("TEXTUREFLOW_LOCAL_URL") or "").strip()
                and (os.getenv("TEXTUREFLOW_LOCAL_TOKEN") or "").strip())


# ---------------------------------------------------------------------------
# SSH control of the desktop (`textureflow-ctl`)
# ---------------------------------------------------------------------------
def run_ctl(argv: list[str]) -> subprocess.CompletedProcess:
    """Run one ssh command (tests replace this)."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=CTL_TIMEOUT_S, stdin=subprocess.DEVNULL)


async def ctl(action: str) -> Optional[dict]:
    """`textureflow-ctl <action>` on the desktop → its JSON line, or None (SSH unreachable / no answer)."""
    host = ssh_host()
    if not host:
        return None
    path = os.getenv("TEXTUREFLOW_LOCAL_CTL") or CTL_DEFAULT
    argv = [SSH, *SSH_OPTS, host, f"{path} {action}"]
    try:
        p = await asyncio.to_thread(run_ctl, argv)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _emit(f"[tf-queue] ssh {host} textureflow-ctl {action}: {type(exc).__name__}")
        return None
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    try:
        out = json.loads(lines[-1]) if lines else None
    except ValueError:
        out = None
    if not isinstance(out, dict):
        err = (p.stderr or "").strip().splitlines()
        _emit(f"[tf-queue] ssh {host} textureflow-ctl {action}: exit {p.returncode}"
              + (f" — {err[-1][:200]}" if err else ""))
        return None
    return out


class LocalLane:
    """What the hub knows about the desktop job server (health, cached ≈ 2 s; SSH status ≈ 20 s)."""

    def __init__(self) -> None:
        self.state = "unknown"
        self.detail: Optional[str] = None
        self.gpu: Optional[str] = None
        self.gpu_holder: Optional[str] = None
        self.vram_free_gb: Optional[float] = None
        self.reachable = False
        self.health_t = 0.0
        self.ctl_t = 0.0
        self.ctl_reply: Optional[dict] = None
        self.last_start = 0.0
        self.inflight: Optional[asyncio.Future] = None  # the health probe running now (callers share it)

    def _set(self, state: str, detail: Optional[str] = None) -> None:
        if (state, detail) != (self.state, self.detail):
            _emit(f"[tf-queue] local: {state}" + (f" ({detail})" if detail else ""))
        self.state, self.detail = state, detail

    async def refresh(self, max_age: float = HEALTH_MAX_AGE_S) -> None:
        from . import textureflow_local

        if not local_configured():
            self.reachable = False
            self._set("unconfigured", "TEXTUREFLOW_LOCAL_URL / TEXTUREFLOW_LOCAL_TOKEN not set")
            return
        now = time.time()
        if self.inflight is not None and not self.inflight.done():
            await asyncio.shield(self.inflight)  # a probe is on its way: its answer is ours too
            return
        if now - self.health_t < max_age:
            return
        self.health_t = now
        self.inflight = asyncio.get_running_loop().create_future()
        try:
            await self._probe(textureflow_local)
        finally:
            fut, self.inflight = self.inflight, None
            if fut is not None and not fut.done():
                fut.set_result(None)

    async def _probe(self, textureflow_local: Any) -> None:
        now = time.time()
        info, err = await textureflow_local.probe()
        if info is not None:
            self.reachable = True
            self.gpu = info.get("gpu") or self.gpu
            self.vram_free_gb = info.get("vram_free_gb")
            holder = info.get("gpu_holder")
            self.gpu_holder = (holder.get("cmd") if isinstance(holder, dict) else holder) or None
            worker = info.get("worker")
            if worker == "down":
                self._set("down", "the render worker is down (the server retries its boot)")
            elif worker == "waiting_gpu":
                self._set("waiting_gpu", f"another process holds the GPU: {(self.gpu_holder or '?')[:120]}")
            elif worker == "booting":
                self._set("starting", "the render worker is loading models")
            elif worker == "idle":
                self._set("idle", "server up; the worker boots on the next job")
            elif info.get("busy") or info.get("queue"):
                self._set("busy", f"rendering (queue {info.get('queue')})")
            else:
                self._set("ready")
            return
        self.reachable = False
        if "401" in (err or ""):
            self._set("down", "the server refused TEXTUREFLOW_LOCAL_TOKEN (401)")
            return
        if not ssh_host():
            self._set("off", "unreachable — is the desktop on? (TEXTUREFLOW_LOCAL_SSH unset: the hub can't start it)")
            return
        if now - self.ctl_t >= CTL_MAX_AGE_S or self.ctl_reply is None:
            self.ctl_t = now
            self.ctl_reply = await ctl("status")
        self.from_ctl(self.ctl_reply)

    def from_ctl(self, c: Optional[dict]) -> None:
        if c is None:
            self._set("off", f"no answer over ssh {ssh_host()} — desktop off or asleep")
            return
        svc = c.get("service")
        if svc in ("active", "activating"):
            self._set("starting", "the job server is starting")
        elif svc == "not_installed":
            self._set("service_down", "the job server is not installed on the desktop (lab deploy/local/install.sh)")
        elif svc == "failed":
            self._set("service_down", "the job server unit failed (journalctl --user -u textureflow)")
        else:
            self._set("service_down", c.get("detail") or f"the job server is {svc or 'not running'}")

    async def start(self) -> Optional[dict]:
        self.last_start = time.time()
        c = await ctl("start")
        self.ctl_reply, self.ctl_t = c, time.time()
        if c and c.get("result") in ("started", "already_running", "ok"):
            self._set("starting", "the job server is starting")
        else:
            self.from_ctl(c)
            if c and c.get("detail"):
                self.detail = c["detail"]
        self.health_t = 0.0
        return c


# ---------------------------------------------------------------------------
# Tracked jobs
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    job: str
    backend: str
    pinned: bool
    args: dict  # what the ETA reads (a finish: the draft's resolved params + the finish args)
    est: dict  # pre-estimate per backend, USD
    allow_over_cap: bool
    label: Optional[str]
    tracker: tf_eta.Tracker
    granted: Optional[asyncio.Event] = None  # Modal: set when the job gets a slot (or moves to local)
    sent: bool = False  # Modal: holds a slot
    local: Optional[str] = None  # local: None | "held" | "queued" | "running" | "done"
    server_id: Optional[str] = None
    queue_position: Optional[int] = None
    moved: bool = False
    last_bump: float = 0.0
    rank: Any = field(default=None)


class _State:
    def __init__(self) -> None:
        self.modal: list[str] = []  # the Modal lane: queued job ids in order
        self.modal_running: set[str] = set()
        self.held: list[str] = []  # local jobs the hub holds (server unreachable)
        self.entries: dict[str, Entry] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.wake: Optional[asyncio.Event] = None
        self.task: Optional[asyncio.Task] = None
        self.lock: Optional[asyncio.Lock] = None
        self.local = LocalLane()
        self.modal_last_end = 0.0
        self.tick_t = 0.0


S = _State()


def _ensure() -> None:
    """Bind to the running loop and start the dispatcher (a new loop — a test's asyncio.run — starts fresh)."""
    loop = asyncio.get_running_loop()
    if S.loop is not loop:
        local = S.local
        S.__init__()
        S.local = local
        S.loop = loop
        S.wake = asyncio.Event()
        S.lock = asyncio.Lock()
    if S.task is None or S.task.done():
        S.task = loop.create_task(_run(), name="wzrd-tf-queue")


def _kick() -> None:
    if S.wake is not None:
        S.wake.set()


def _plan(args: dict, backend: str) -> tuple[list, float]:
    return tf_eta.plan(args, backend), tf_eta.OVERHEAD_S.get(backend, 10.0)


def _set_rank(e: Entry, rank: Optional[float]) -> None:
    if rank != e.rank:
        e.rank = rank
        jobs.update(e.job, lane_rank=rank)


def _persist_modal() -> None:
    for i, j in enumerate(S.modal):
        if j in S.entries:
            _set_rank(S.entries[j], float(i))


def _persist_held() -> None:
    for i, j in enumerate(S.held):
        if j in S.entries:
            _set_rank(S.entries[j], float(HELD_RANK + i))


async def track(job: str, backend: str, args: dict, *, pinned: bool, est: dict,
                allow_over_cap: bool = False, label: Optional[str] = None) -> Entry:
    """Start tracking a texture_flow job (call `finish(job)` when the call ends)."""
    _ensure()
    laps, overhead = await asyncio.to_thread(_plan, args, backend)
    e = Entry(job, backend, pinned, dict(args), dict(est), allow_over_cap, label,
              tf_eta.Tracker(laps, backend, overhead))
    S.entries[job] = e
    jobs.update(job, backend=backend)
    if backend == "modal":  # in the lane now; `acquire_modal` waits for the grant
        e.granted = asyncio.Event()
        S.modal.append(job)
        _persist_modal()
    else:  # queued until the server takes it (`local_submitted` → its queue position)
        _set_rank(e, float(HELD_RANK))
    _kick()
    return e


def entry(job: Optional[str]) -> Optional[Entry]:
    return S.entries.get(job or "")


async def acquire_modal(job: str) -> str:
    """Queue *job* in the Modal lane and wait for a slot. Returns the backend it ends up on: "modal", or
    "local" when the operator moved it while it waited. Cancelling the caller takes it out of the lane."""
    _ensure()
    e = S.entries[job]
    if job in S.modal_running:
        return "modal"  # granted already
    if e.granted is None or e.granted.is_set():
        e.granted = asyncio.Event()
    e.tracker.set_phase("queued")
    if job not in S.modal:
        S.modal.append(job)
        _persist_modal()
    _kick()
    try:
        await e.granted.wait()
    except BaseException:
        if job in S.modal:
            S.modal.remove(job)
            _persist_modal()
        S.modal_running.discard(job)
        _kick()
        raise
    return e.backend


def _grant_modal(job: str) -> None:
    e = S.entries[job]
    S.modal.remove(job)
    S.modal_running.add(job)
    e.sent = True
    e.tracker.set_phase("starting")
    _set_rank(e, None)
    jobs.update(job, progress=json.dumps(e.tracker.snapshot()))
    _persist_modal()
    assert e.granted is not None
    e.granted.set()


def release(job: str) -> None:
    """The remote side is done with *job*: a Modal slot frees (progress keeps tracking until `finish`)."""
    e = S.entries.get(job)
    if e and job in S.modal_running:
        S.modal_running.discard(job)
        e.sent = False
        S.modal_last_end = time.time()
        _kick()
    if e and e.local in ("queued", "running"):
        e.local = "done"


def finish(job: str) -> None:
    """The call ended (any way): forget it, persisting its last progress."""
    release(job)
    if job in S.held:
        S.held.remove(job)
        _persist_held()
    if job in S.modal:
        S.modal.remove(job)
        _persist_modal()
    e = S.entries.pop(job, None)
    if e:
        jobs.update(job, lane_rank=None, progress=json.dumps(e.tracker.snapshot()))
    _kick()


def report(job: Optional[str], *, phase: Optional[str] = None, stage: Optional[str] = None,
           step: Optional[int] = None, total: Optional[int] = None) -> None:
    """A phase change and/or a live stage report. Persists on a change; otherwise bumps `/api/jobs` at most
    once a second per job."""
    e = S.entries.get(job or "")
    if not e:
        return
    now = time.time()
    before = (e.tracker.phase, e.tracker.running)
    if phase:
        e.tracker.set_phase(phase, now)
    if stage is not None:
        e.tracker.on_stage(stage, step, total, now)
    if (e.tracker.phase, e.tracker.running) != before:
        e.last_bump = now
        jobs.update(e.job, progress=json.dumps(e.tracker.snapshot(now)))
    elif now - e.last_bump >= TICK_S:
        e.last_bump = now
        jobs.bump()


# ---------------------------------------------------------------------------
# Local: hold until the server answers, then follow the server's queue
# ---------------------------------------------------------------------------
async def hold_local(job: str) -> None:
    """Wait until the desktop server answers (holding the job in the hub meanwhile). Raises `Moved` when the
    operator moves the job to Modal while it is held."""
    _ensure()
    e = S.entries[job]
    while True:
        if e.backend != "local":
            raise Moved(job)
        await S.local.refresh()
        if S.local.reachable:
            if job in S.held:
                S.held.remove(job)
                _persist_held()
            return
        if job not in S.held:
            S.held.append(job)
            e.local = "held"
            _persist_held()
            report(job, phase="waiting_backend")
            _kick()
        await asyncio.sleep(HOLD_POLL_S)


def local_submitted(job: str, server_id: str, queue_position: Optional[int]) -> None:
    e = S.entries.get(job)
    if not e:
        return
    e.server_id, e.local = server_id, "queued"
    local_status(job, {"state": "queued", "queue_position": queue_position})


def local_status(job: str, st: dict) -> None:
    """One v1 status reply of this job's server job → its place + progress."""
    e = S.entries.get(job)
    if not e:
        return
    state = st.get("state")
    if state == "queued":
        e.local = "queued"
        e.queue_position = st.get("queue_position")
        _set_rank(e, float(e.queue_position or 0))
        report(job, phase="queued")
    elif state == "running":
        e.local, e.queue_position = "running", None
        _set_rank(e, None)
        if st.get("stage"):
            report(job, stage=st.get("stage"), step=st.get("step"), total=st.get("total"))
        else:
            report(job, phase="starting")
    elif state in ("done", "error", "cancelled"):
        _set_rank(e, None)


def moved_away(job: str) -> bool:
    e = S.entries.get(job)
    return bool(e and e.moved and e.backend != "local")


def consume_move(job: str) -> None:
    e = S.entries.get(job)
    if e:
        e.moved = False


# ---------------------------------------------------------------------------
# Reads (any thread)
# ---------------------------------------------------------------------------
def _wait_ahead(e: Entry, now: float) -> float:
    if e.backend == "modal":
        if e.job not in S.modal:
            return 0.0
        i = S.modal.index(e.job)
        ahead = sum(S.entries[j].tracker.remaining(now) for j in list(S.modal_running) if j in S.entries)
        ahead += sum(S.entries[j].tracker.predicted for j in S.modal[:i] if j in S.entries)
        return ahead / modal_slots()
    if e.local == "queued":
        # the server's queue: jobs ahead (anyone's) at roughly this job's own size, plus the running one
        n = e.queue_position or 0
        running = [x for x in S.entries.values() if x.local == "running"]
        run_s = running[0].tracker.remaining(now) if running else e.tracker.predicted / 2
        boot = LOCAL_BOOT_S if S.local.state in ("idle", "starting", "waiting_gpu") else 0.0
        return n * e.tracker.predicted + run_s + boot
    if e.local == "held":
        return LOCAL_BOOT_S + S.held.index(e.job) * e.tracker.predicted if e.job in S.held else LOCAL_BOOT_S
    return 0.0


def progress(job: str) -> Optional[dict]:
    """The live progress block of a job this hub tracks, else None."""
    e = S.entries.get(job)
    if not e:
        return None
    now = time.time()
    return e.tracker.snapshot(now, _wait_ahead(e, now))


def lane_pos(job: str) -> Optional[int]:
    e = S.entries.get(job)
    if not e:
        return None
    if e.backend == "modal":
        return S.modal.index(job) if job in S.modal else None
    if e.local == "queued":
        return e.queue_position
    if e.local == "held" and job in S.held:
        n_server = sum(1 for x in S.entries.values() if x.local == "queued")
        return n_server + S.held.index(job)
    return None


def busy() -> bool:
    return bool(S.entries)


# ---------------------------------------------------------------------------
# Operator actions (the /api routes)
# ---------------------------------------------------------------------------
def _ordered(current: list[str], order: list[str]) -> list[str]:
    first: list[str] = []
    for j in order:
        if j in current and j not in first:
            first.append(j)
    return first + [j for j in current if j not in first]


async def reorder(backend: str, order: list[str]) -> dict:
    from . import textureflow_local

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    _ensure()
    if backend == "modal":
        S.modal = _ordered(S.modal, order)
        _persist_modal()
        _kick()
        return {"order": list(S.modal)}
    # local: the server's queue (our jobs take our slots in the new order; foreign jobs keep theirs), then held
    ours = {e.server_id: e.job for e in S.entries.values() if e.local == "queued" and e.server_id}
    sid_of = {j: sid for sid, j in ours.items()}
    wanted = list(dict.fromkeys(j for j in order if j in sid_of))
    if ours:
        q = await textureflow_local.queue()
        if q is None:
            raise ValueError("the desktop job server has no /v1/queue (an older version) — it can't reorder yet")
        server_order = [x["job_id"] for x in q.get("queued") or [] if x.get("state", "queued") == "queued"]
        mine_in_place = [sid for sid in server_order if sid in ours]
        mine_new = [sid_of[j] for j in wanted if sid_of[j] in mine_in_place] \
            + [sid for sid in mine_in_place if ours[sid] not in wanted]
        it = iter(mine_new)
        new_order = [next(it) if sid in ours else sid for sid in server_order]
        await textureflow_local.queue_order(new_order)
        for pos, sid in enumerate(new_order):
            if sid in ours:
                e = S.entries[ours[sid]]
                e.queue_position = pos
                _set_rank(e, float(pos))
                jobs.bump()
    S.held = _ordered(S.held, order)
    _persist_held()
    result = [ours[s] for s in sorted(ours, key=lambda s: S.entries[ours[s]].queue_position or 0)] + list(S.held)
    return {"order": result}


async def move(job: str, backend: str) -> None:
    """A queued, unpinned job changes lane (appended at the end)."""
    from . import textureflow_local

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    _ensure()
    e = S.entries.get(job)
    queued = e is not None and ((e.backend == "modal" and job in S.modal)
                                or (e.backend == "local" and e.local in ("held", "queued")))
    if not queued:
        raise ValueError(f"job {job} is not queued in this hub — only a queued job can move")
    assert e is not None
    if e.pinned:
        raise ValueError(f"job {job} finishes a draft: it runs on the draft's backend ({e.backend}) and can't move")
    if backend == e.backend:
        return
    if backend == "local" and not local_configured():
        raise ValueError("the local lane is unconfigured (TEXTUREFLOW_LOCAL_URL / TEXTUREFLOW_LOCAL_TOKEN)")
    usd = float(e.est.get(backend, 0.0))
    if backend == "modal":
        total, cap = jobs.spent_today(exclude=job), jobs.daily_cap()
        if total + usd > cap and not e.allow_over_cap:
            raise ValueError(f"daily spend cap: ${total:.2f} of today's ${cap:.2f} is spent and this render costs "
                             f"~${usd:.2f} on Modal — raise the cap first")
        if e.local == "queued" and e.server_id:
            reply = await textureflow_local.cancel(e.server_id)
            if reply.get("state") not in ("cancelled", "queued"):
                raise ValueError(f"job {job} already started on the desktop ({reply.get('state')}) — it stays there")
        if job in S.held:
            S.held.remove(job)
            _persist_held()
        e.local, e.server_id, e.queue_position = None, None, None
        e.backend, e.moved = "modal", True
        S.modal.append(job)
        _persist_modal()
        jobs.update(job, remote=None)
    else:
        if job in S.modal:
            S.modal.remove(job)
            _persist_modal()
        e.backend = "local"
        if e.granted is not None:
            e.granted.set()  # acquire_modal returns "local"
    laps, overhead = _plan(e.args, backend)
    e.tracker.replan(laps, backend, overhead)
    jobs.move_job(job, backend, round(usd, 4))
    _kick()


def _modal_block() -> dict:
    now = time.time()
    running = len(S.modal_running)
    if running:
        state = "ready"
    elif now - S.modal_last_end < MODAL_WARM_S:
        state = "warm"
    elif S.modal_last_end:
        state = "cold"
    else:
        state = "unknown"
    return {"state": state, "slots": modal_slots(), "max_slots": MAX_MODAL_SLOTS, "running": running,
            "queued": len(S.modal)}


def _local_block() -> dict:
    lane = S.local
    ours = [e for e in S.entries.values() if e.backend == "local"]
    return {"state": lane.state, "detail": lane.detail, "gpu": lane.gpu, "gpu_holder": lane.gpu_holder,
            "vram_free_gb": lane.vram_free_gb, "running": sum(1 for e in ours if e.local == "running"),
            "queued": sum(1 for e in ours if e.local in ("held", "queued", None)), "ssh": bool(ssh_host())}


async def backends() -> dict:
    _ensure()
    await S.local.refresh()
    return {"modal": _modal_block(), "local": _local_block()}


async def local_action(action: str) -> dict:
    _ensure()
    if action != "start":
        raise ValueError("action must be 'start' (the server's worker frees the GPU by itself when idle)")
    if not ssh_host():
        raise ValueError("TEXTUREFLOW_LOCAL_SSH is not set — the hub can't start the desktop server")
    assert S.lock is not None
    async with S.lock:
        await S.local.refresh(0)
        if not S.local.reachable:
            await S.local.start()
    _kick()
    return _local_block()


def _foreign(x: dict) -> dict:
    step, total = x.get("step"), x.get("total")
    phase = {"queued": "queued", "running": "rendering" if x.get("stage") else "starting"}.get(x.get("state"),
                                                                                              x.get("state"))
    return {"foreign": True, "job_id": x.get("job_id"), "client": x.get("client"), "label": x.get("label"),
            "state": x.get("state"), "progress": {
                "phase": phase, "stage": x.get("stage"), "step": step, "total": total,
                "pct": round(step / total, 3) if isinstance(step, int) and total else None, "eta_s": None,
                "source": "live" if x.get("stage") else "estimate"}}


async def queue_view() -> dict:
    """`GET /api/tf/queue`: both lanes in run order (running first)."""
    from . import library, textureflow_local

    _ensure()

    def dicts(ids: list[str]) -> list[dict]:
        out = []
        for j in ids:
            try:
                out.append(jobs.get(j))
            except Exception:  # noqa: BLE001 — finished meanwhile
                pass
        return out

    modal = await asyncio.to_thread(dicts, list(S.modal_running) + list(S.modal))
    local: list[dict] = []
    q = await textureflow_local.queue() if local_configured() and S.local.reachable else None
    if q is not None:
        running = q.get("running")
        server_jobs = ([running] if isinstance(running, dict) else list(running or [])) + list(q.get("queued") or [])
        ids = [x.get("job_id") for x in server_jobs if isinstance(x, dict)]
        rows = await asyncio.to_thread(
            library.query, f"SELECT id, remote FROM jobs WHERE remote IN ({','.join('?' * len(ids))})",
            [f"local|{i}" for i in ids]) if ids else []
        wzrd = {r["remote"].split("|", 1)[1]: r["id"] for r in rows}
        for x in server_jobs:
            if not isinstance(x, dict):
                continue
            j = wzrd.get(x.get("job_id"))
            if j:
                local.extend(await asyncio.to_thread(dicts, [j]))
            else:
                local.append(_foreign(x))
        if q.get("gpu_holder"):
            h = q["gpu_holder"]
            S.local.gpu_holder = h.get("cmd") if isinstance(h, dict) else h
    else:  # no /v1/queue (an older server, or unreachable): our own jobs, in the order we know
        ours = [e for e in S.entries.values() if e.backend == "local" and e.local in ("running", "queued")]
        ours.sort(key=lambda e: (e.local != "running", e.queue_position or 0))
        local = await asyncio.to_thread(dicts, [e.job for e in ours])
    local.extend(await asyncio.to_thread(dicts, list(S.held)))
    return {"modal": modal, "local": local}


# ---------------------------------------------------------------------------
# The dispatcher + hub start
# ---------------------------------------------------------------------------
async def _step() -> None:
    now = time.time()
    while S.modal and len(S.modal_running) < modal_slots():
        _grant_modal(S.modal[0])
    if S.held and ssh_host():
        assert S.lock is not None
        async with S.lock:
            await S.local.refresh()
            if (not S.local.reachable and S.local.state == "service_down"
                    and now - S.local.last_start >= START_EVERY_S):
                await S.local.start()
    if any(e.sent or e.local in ("queued", "running") for e in S.entries.values()) and now - S.tick_t >= TICK_S:
        S.tick_t = now
        jobs.bump()  # running jobs' eta / pct move with time


async def _run() -> None:
    while True:
        try:
            await _step()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — the dispatcher must survive anything
            _emit(f"[tf-queue] dispatcher step failed: {exc!r}")
        assert S.wake is not None
        try:
            await asyncio.wait_for(S.wake.wait(), TICK_S if S.entries else None)
        except asyncio.TimeoutError:
            pass
        S.wake.clear()


async def adopt_all() -> list[str]:
    """Hub start (Q8): re-adopt this library's unfinished local jobs left by a previous hub — they keep
    rendering on the desktop; collect their results into the library."""
    from . import library

    _ensure()
    rows = await asyncio.to_thread(
        library.query, "SELECT * FROM jobs WHERE tool='texture_flow' AND ended IS NULL AND remote LIKE 'local|%'")
    adopted = []
    for row in rows:
        if jobs._alive(row["pid"]) and row["pid"] != os.getpid():
            continue  # another live process (a --stdio server) owns it
        if row["id"] in S.entries or row["id"] in jobs._tasks:
            continue
        from .tools import adopt_local

        jobs.adopt(row["id"])
        task = asyncio.get_running_loop().create_task(adopt_local(dict(row)), name=f"wzrd-adopt-{row['id']}")
        jobs._tasks[row["id"]] = task
        adopted.append(row["id"])
    if adopted:
        _emit(f"[tf-queue] re-adopted {len(adopted)} local job(s) from a previous hub: {', '.join(adopted)}")
    return adopted
