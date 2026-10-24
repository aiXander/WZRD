// Dev-only mock of the TextureFlow render queue (render queue plan, the hub
// API contract): `/api/tf/backends`, `/api/tf/queue`, `/api/tf/local`,
// `/api/jobs/reorder`, `/api/jobs/move`, plus simulated texture_flow jobs
// (and one lab agent's foreign job on the desktop) merged into the real
// `/api/jobs` feed. Opt in with `?hubmock=1` (`api/hub.ts::hubCall`); every
// other route goes to the real hub. A mocked texture_flow `/api/render`
// never reaches the hub, so nothing is spent.
//
// The simulation: Modal has 3 slots, local 1 (only while its server is
// ready); a job runs starting → sample (12 steps) → upscale → encode →
// downloading → saving over its `dur` seconds, then leaves the queue.

import type { ForeignJob, Job, JobProgress, LocalLane, LocalState, QueueEntry, TfBackend, TfBackends } from './hub';

type Call = <T>(method: 'GET' | 'POST' | 'PUT', path: string, body?: unknown) => Promise<T>;

type MockJob = {
  id: string;
  backend: TfBackend;
  status: Job['status'];
  args: Record<string, unknown>;
  created: number;
  rank: number;
  pinned: boolean;
  dur: number;
  runStart: number | null;
  ended: number | null;
  /** Report live stages (else the bar is the estimate model). */
  live: boolean;
};

// Modal's slots are operator-set (`POST /api/tf/modal`, default 1, max 3); local is one GPU.
const MAX_MODAL = 3;
const SLOTS: Record<TfBackend, number> = { modal: 1, local: 1 };
// Fractions of a job's duration per phase / stage.
const PLAN: { phase: JobProgress['phase']; stage: string | null; frac: number; steps: number | null }[] = [
  { phase: 'starting', stage: null, frac: 0.08, steps: null },
  { phase: 'rendering', stage: 'sample', frac: 0.5, steps: 12 },
  { phase: 'rendering', stage: 'upscale', frac: 0.22, steps: 4 },
  { phase: 'rendering', stage: 'encode', frac: 0.08, steps: null },
  { phase: 'downloading', stage: null, frac: 0.07, steps: null },
  { phase: 'saving', stage: null, frac: 0.05, steps: null },
];

let seq = 0;
let version = 0;
let local: { state: LocalState; until: number } = { state: 'ready', until: 0 };
const now = () => Date.now() / 1000;

function mk(backend: TfBackend, args: Record<string, unknown>, dur: number, startedAgo: number | null, live = true): MockJob {
  const t = now();
  return {
    id: `mock_${(++seq).toString(36)}`,
    backend,
    status: startedAgo == null ? 'queued' : 'running',
    args,
    created: t - (startedAgo ?? 0) - 5,
    rank: seq,
    pinned: !!args.finish_from,
    dur,
    runStart: startedAgo == null ? null : t - startedAgo,
    ended: null,
    live,
  };
}

const jobs: MockJob[] = [
  mk('modal', { recipe: 'ink_bloom', draft: true, n_seconds: 8 }, 140, 95),
  mk('modal', { recipe: 'ink_bloom', draft: true, n_seconds: 8 }, 140, 40, false),
  mk('modal', { finish_from: 'tf_20261024_1412_ab12', backend: 'modal' }, 180, 12),
  mk('modal', { recipe: 'chrome_moss', n_seconds: 10 }, 300, null),
  mk('modal', { recipe: 'chrome_moss', draft: true, n_seconds: 10 }, 150, null),
  mk('local', { images: [], draft: true, n_seconds: 8, backend: 'local' }, 260, 120),
  mk('local', { images: [], draft: true, n_seconds: 8, backend: 'local' }, 200, null),
  mk('local', { recipe: 'lava_lamp', draft: true, backend: 'local' }, 260, null),
  mk('local', { finish_from: 'tf_20261024_1350_cd34', backend: 'local' }, 420, null),
  mk('local', { prompt: 'slow ink tendrils on black', images: [], draft: true, backend: 'local' }, 260, null),
];

/** Mock jobs without style images borrow stills from the Studio target folder
 *  (read from the real hub), so cards show thumbs. */
async function fillImages(real: Call) {
  const want = jobs.filter((j) => Array.isArray(j.args.images) && !(j.args.images as string[]).length);
  if (!want.length) return;
  try {
    const sel = await real<{ target: string | null }>('GET', '/api/selection');
    if (!sel.target) return;
    const l = await real<{ items: { kind: string | null; path: string }[] }>(
      'GET',
      `/api/library/list?dir=${encodeURIComponent(sel.target)}`
    );
    const stills = l.items.filter((i) => i.kind === 'image').map((i) => i.path);
    if (stills.length) want.forEach((j, k) => (j.args.images = [stills[k % stills.length]]));
  } catch {
    /* hub down: no thumbs */
  }
}

function lane(b: TfBackend) {
  return jobs.filter((j) => j.backend === b && (j.status === 'queued' || j.status === 'running'));
}
const queued = (b: TfBackend) => lane(b).filter((j) => j.status === 'queued').sort((a, c) => a.rank - c.rank);
const running = (b: TfBackend) => lane(b).filter((j) => j.status === 'running');

function tick() {
  const t = now();
  if (local.state === 'starting' && t >= local.until) local = { state: 'ready', until: 0 };
  if (local.state === 'idle' && queued('local').length) local = { state: 'starting', until: t + 8 };
  for (const j of jobs) {
    if (j.status === 'running' && j.runStart != null && t - j.runStart >= j.dur) {
      j.status = 'done';
      j.ended = t;
    }
  }
  for (const b of ['modal', 'local'] as TfBackend[]) {
    if (b === 'local' && !['ready', 'busy'].includes(local.state)) continue;
    let free = SLOTS[b] - running(b).length;
    for (const j of queued(b)) {
      if (free-- <= 0) break;
      j.status = 'running';
      j.runStart = t;
    }
  }
  if (local.state === 'ready' || local.state === 'busy') {
    local.state = running('local').length ? 'busy' : 'ready';
  }
}

/** A lab agent's render on the desktop, queued behind WZRD's running one. */
const foreign: ForeignJob = {
  foreign: true,
  job_id: 'srv_7f3a',
  client: 'lab · explore03',
  label: 'explore03 #14 · flow_noise 0.6',
  state: 'queued',
  progress: { phase: 'queued', stage: null, step: null, total: null, pct: null, eta_s: 410, source: 'estimate' },
};

function laneEntries(b: TfBackend): QueueEntry[] {
  const rows: QueueEntry[] = [...running(b), ...queued(b)].map(toJob);
  // The foreign job sits second in the server's queue.
  if (b === 'local') rows.splice(Math.min(rows.length, running('local').length + 1), 0, foreign);
  return rows;
}

function progressOf(j: MockJob, waitAhead: number): JobProgress {
  if (j.status === 'queued') {
    const waiting = j.backend === 'local' && !['ready', 'busy'].includes(local.state);
    return {
      phase: waiting ? 'waiting_backend' : 'queued',
      stage: null,
      step: null,
      total: null,
      pct: null,
      eta_s: Math.round(waitAhead + j.dur),
      source: 'estimate',
    };
  }
  const el = now() - (j.runStart ?? now());
  const pct = Math.min(0.99, el / j.dur);
  let acc = 0;
  let cur = PLAN[0];
  let within = 0;
  for (const p of PLAN) {
    if (pct < acc + p.frac) {
      cur = p;
      within = (pct - acc) / p.frac;
      break;
    }
    acc += p.frac;
  }
  const live = j.live && cur.phase === 'rendering';
  return {
    phase: cur.phase,
    stage: live ? cur.stage : null,
    step: live && cur.steps ? Math.min(cur.steps, 1 + Math.floor(within * cur.steps)) : null,
    total: live && cur.steps ? cur.steps : null,
    pct: cur.phase === 'starting' && !j.live ? null : pct,
    eta_s: Math.max(0, Math.round(j.dur - el)),
    source: j.live ? 'live' : 'estimate',
  };
}

function toJob(j: MockJob): Job {
  // Wait ahead of a queued job = the remaining time of what's ahead, spread over the slots.
  let waitAhead = 0;
  let pos: number | null = null;
  if (j.status === 'queued') {
    const q = queued(j.backend);
    pos = q.indexOf(j);
    const t = now();
    const busy = running(j.backend).reduce((s, r) => s + Math.max(0, r.dur - (t - (r.runStart ?? t))), 0);
    waitAhead = (busy + q.slice(0, pos).reduce((s, r) => s + r.dur, 0)) / SLOTS[j.backend];
  }
  const p = j.status === 'queued' || j.status === 'running' ? progressOf(j, waitAhead) : null;
  return {
    job: j.id,
    tool: 'texture_flow',
    status: j.status,
    by: 'app',
    args: j.args,
    started: j.runStart ?? j.created,
    ended: j.ended,
    duration_s: j.ended && j.runStart ? j.ended - j.runStart : null,
    est_usd: j.backend === 'modal' ? 0.1 : 0,
    error: null,
    outputs: [],
    dest: null,
    project: null,
    session: null,
    elapsed_s: j.runStart ? now() - j.runStart : 0,
    eta_s: p?.eta_s ?? null,
    backend: j.backend,
    lane_pos: pos,
    pinned: j.pinned,
    progress: p,
  };
}

function backends(): TfBackends {
  const loc: LocalLane = {
    state: local.state,
    detail: local.state === 'starting' ? 'textureflow-ctl start on rednax-beast' : null,
    gpu: 'RTX 3090',
    gpu_holder: null,
    vram_free_gb: local.state === 'busy' ? 3.1 : 21.4,
    running: running('local').length,
    queued: queued('local').length,
  };
  return {
    modal: {
      state: running('modal').length ? 'warm' : 'cold',
      slots: SLOTS.modal,
      max_slots: MAX_MODAL,
      running: running('modal').length,
      queued: queued('modal').length,
    },
    local: loc,
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Handler = (body: any, path: string, real: Call) => unknown;

/** A mock job by id, or undefined (then the real hub answers). */
const mine = (id: string) => jobs.find((x) => x.id === id);

const ROUTES: Record<string, Handler> = {
  '/api/tf/backends': () => backends(),
  '/api/tf/queue': () => ({ modal: laneEntries('modal'), local: laneEntries('local') }),
  '/api/tf/modal': (body) => {
    const n = Number(body?.slots);
    if (!Number.isInteger(n) || n < 1 || n > MAX_MODAL) throw new Error(`slots must be 1..${MAX_MODAL}`);
    SLOTS.modal = n;
    tick();
    return backends().modal;
  },
  '/api/tf/local': (body) => {
    if (body?.action === 'start' && local.state === 'service_down') local = { state: 'starting', until: now() + 8 };
    return backends().local;
  },
  // Dev aid: force a local state (`hubCall('POST', '/__mock/local', {state})`).
  '/__mock/local': (body) => {
    local = { state: body.state, until: now() + 8 };
    return backends().local;
  },
  '/api/jobs/reorder': (body) => {
    const order: string[] = body.order;
    const q = queued(body.backend);
    const ranks = q.map((j) => j.rank).sort((a, b) => a - b);
    const at = (j: MockJob) => (order.includes(j.id) ? order.indexOf(j.id) : 1e9 + j.rank);
    const sorted = [...q].sort((a, b) => at(a) - at(b));
    sorted.forEach((j, k) => (j.rank = ranks[k]));
    return { order: sorted.map((j) => j.id) };
  },
  '/api/jobs/move': (body) => {
    const j = mine(body.job)!;
    if (j.status !== 'queued') throw new Error('only a queued job can change lanes');
    if (j.pinned) throw new Error("a finish runs where its draft's bundle lives");
    j.backend = body.backend;
    j.rank = ++seq;
    return toJob(j);
  },
  '/api/jobs/cancel': (body) => {
    const j = mine(body.job)!;
    j.status = 'cancelled';
    j.ended = now();
    return { job: j.id, status: 'cancelled' };
  },
  '/api/render': (body) => {
    const b: TfBackend = body.args?.backend === 'local' ? 'local' : 'modal';
    const made = Array.from({ length: Number(body.count ?? 1) }, () => {
      const j = mk(b, { ...body.args }, body.args?.draft ? 140 : 300, null, b === 'local');
      jobs.push(j);
      return { job: j.id, status: 'queued', dest: null, eta_s: null };
    });
    return { jobs: made };
  },
  '/api/jobs': async (body, path, real) => {
    // Answer about once a second (the hub's progress cadence) — a mock long-poll.
    if (/[?&]version=/.test(path)) await sleep(1000);
    tick();
    await fillImages(real);
    let feed: { jobs: Job[]; spend: any } = { jobs: [], spend: null };
    try {
      feed = await real<any>('GET', path.replace(/wait=\d+/, 'wait=0'), body);
    } catch {
      /* hub down: mock jobs only */
    }
    const ours = jobs.filter((j) => j.status !== 'done' || (j.ended ?? 0) > now() - 600).map(toJob);
    return {
      version: version + 1,
      now: now(),
      jobs: [...ours, ...feed.jobs],
      spend: feed.spend ?? { today_usd: 1.24, cap_usd: 20, running_usd: 0.3, warn_at: 0.8, warn: false, over: false },
    };
  },
};

/** Requests the real hub answers even in mock mode: non-mock jobs, other tools' renders. */
function passThrough(route: string, body: any): boolean {
  if (route === '/api/jobs/move' || route === '/api/jobs/cancel') return !mine(body?.job);
  if (route === '/api/render') return body?.tool !== 'texture_flow';
  return !(route in ROUTES);
}

export async function mockHubCall<T>(method: 'GET' | 'POST' | 'PUT', path: string, body: any, real: Call): Promise<T> {
  tick();
  const route = path.split('?')[0];
  if (passThrough(route, body)) return real<T>(method, path, body);
  const v = await ROUTES[route](body, path, real);
  version += 1;
  return v as T;
}
