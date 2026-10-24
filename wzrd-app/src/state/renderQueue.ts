// The TextureFlow render queue, app side (render queue plan, lane U): one
// ordered lane per backend — Modal (a few slots, the hub's lane) and Local
// (the desktop job server's own queue, which other clients such as lab
// agents share) — plus the backends' state and the operator's queue
// actions: reorder, to top, move to the other lane, cancel, start the
// desktop's job service.
//
// - Lanes come from `GET /api/tf/queue` (run order, running first; the local
//   lane can hold `foreign` entries — another client's job, shown read-only).
//   It and `/api/tf/backends` are polled every 2 s while something holds
//   `useRenderQueue` (the queue panel, the Render panel, the jobs pill), and
//   again whenever the `/api/jobs` feed changes. A hub without the route:
//   lanes are derived from the job feed instead (`laneOf`).
// - A reorder is optimistic: the dragged order shows at once and holds until
//   a queue snapshot newer than the hub's reply arrives.

import { useEffect } from 'react';
import { create } from 'zustand';
import {
  cancelJob,
  moveJob,
  reorderJobs,
  tfBackends,
  tfLocalStart,
  tfModalSlots,
  tfQueue,
  type ForeignJob,
  type Job,
  type LocalState,
  type ModalState,
  type QueueEntry,
  type TfBackend,
  type TfBackends,
  type TfQueue,
} from '../api/hub';
import { hubUp, refreshJobs, useHub } from './hub';
import { useStore } from './store';

export const BACKENDS: TfBackend[] = ['modal', 'local'];
export const BACKEND_LABEL: Record<TfBackend, string> = { modal: 'Modal', local: 'Local' };

type Override = { ids: string[]; resolvedAt: number };

type QueueState = {
  backends: TfBackends | null;
  /** The hub's lanes (`/api/tf/queue`), null until read or when the hub lacks the route. */
  queue: TfQueue | null;
  /** When `queue` arrived (ms). */
  queueAt: number;
  /** Optimistic lane orders after a drag, until a newer snapshot. */
  order: Partial<Record<TfBackend, Override>>;
  /** Studio's queue pane is open. */
  open: boolean;
};

const OPEN_KEY = 'wzrd.studio.queue';
const PREF_KEY = 'wzrd.tf.backend';

function readOpen(): boolean {
  try {
    return localStorage.getItem(OPEN_KEY) !== '0';
  } catch {
    return true;
  }
}

export const useQueue = create<QueueState>(() => ({
  backends: null,
  queue: null,
  queueAt: 0,
  order: {},
  open: readOpen(),
}));

export function setQueueOpen(open: boolean) {
  useQueue.setState({ open });
  try {
    localStorage.setItem(OPEN_KEY, open ? '1' : '0');
  } catch {
    /* per-viewer convenience only */
  }
}

// ---------- the backend the Render panel sends to (remembered per machine) ----------

export function preferredBackend(): TfBackend {
  try {
    return localStorage.getItem(PREF_KEY) === 'local' ? 'local' : 'modal';
  } catch {
    return 'modal';
  }
}

export function setPreferredBackend(b: TfBackend) {
  try {
    localStorage.setItem(PREF_KEY, b);
  } catch {
    /* per-viewer convenience only */
  }
}

// ---------- lanes ----------

export const isForeign = (e: QueueEntry): e is ForeignJob => (e as ForeignJob).foreign === true;
export const entryId = (e: QueueEntry) => (isForeign(e) ? `foreign:${e.job_id}` : e.job);
/** A WZRD job still waiting in its lane (the only kind the operator can drag). */
export const isQueued = (e: QueueEntry): e is Job => !isForeign(e) && e.status === 'queued';

/** The lane a texture_flow job sits in (hubs before the queue: its args). */
export function jobBackend(j: Job): TfBackend {
  const b = j.backend ?? (j.args?.backend as string | undefined);
  return b === 'local' ? 'local' : 'modal';
}

const inQueue = (j: Job) => j.tool === 'texture_flow' && (j.status === 'queued' || j.status === 'running');

/** Fallback for hubs without `/api/tf/queue`: running first, then queued by `lane_pos`. */
function laneOf(jobs: Job[], backend: TfBackend): Job[] {
  const mine = jobs.filter((j) => inQueue(j) && jobBackend(j) === backend);
  const running = mine.filter((j) => j.status === 'running').sort((a, b) => a.started - b.started);
  const queued = mine
    .filter((j) => j.status === 'queued')
    .sort((a, b) => (a.lane_pos ?? 1e9) - (b.lane_pos ?? 1e9) || a.started - b.started);
  return [...running, ...queued];
}

/** The WZRD queued entries re-sorted into the dragged order, in the slots
 *  they held (foreign and running entries keep their places). */
function applyOverride(lane: QueueEntry[], o: Override): QueueEntry[] {
  const rank = new Map(o.ids.map((id, i) => [id, i]));
  const queued = lane.filter(isQueued).sort((a, b) => (rank.get(a.job) ?? 1e9) - (rank.get(b.job) ?? 1e9));
  let k = 0;
  return lane.map((e) => (isQueued(e) ? queued[k++] : e));
}

/** Both lanes, re-rendering on snapshot, feed and override changes. */
export function useLanes(): Record<TfBackend, QueueEntry[]> {
  const jobs = useHub((s) => s.jobs);
  const queue = useQueue((s) => s.queue);
  const queueAt = useQueue((s) => s.queueAt);
  const order = useQueue((s) => s.order);
  const lane = (b: TfBackend): QueueEntry[] => {
    const base: QueueEntry[] = queue ? queue[b] ?? [] : laneOf(jobs, b);
    const o = order[b];
    return o && queueAt <= o.resolvedAt ? applyOverride(base, o) : base;
  };
  return { modal: lane('modal'), local: lane('local') };
}

// ---------- polling (while held) ----------

let holders = 0;
let timer: ReturnType<typeof setInterval> | undefined;
let unsubFeed: (() => void) | undefined;
let noQueueRoute = false;

async function refreshQueue() {
  if (!hubUp()) return;
  const [b, q] = await Promise.allSettled([tfBackends(), noQueueRoute ? Promise.reject() : tfQueue()]);
  useQueue.setState({
    ...(b.status === 'fulfilled' ? { backends: b.value } : {}),
    ...(q.status === 'fulfilled' ? { queue: q.value, queueAt: Date.now() } : { queue: null }),
  });
  // A hub without the route (404): stop asking until the page reloads.
  if (q.status === 'rejected' && /404|not found|no route/i.test(String(q.reason))) noQueueRoute = true;
}

/** Hold the queue + backends poll (2 s, and on every job-feed change) while mounted. */
export function useRenderQueue(active = true) {
  const up = useHub((s) => s.status?.state === 'up');
  useEffect(() => {
    if (!active || !up) return;
    holders += 1;
    if (holders === 1) {
      refreshQueue();
      timer = setInterval(refreshQueue, 2000);
      unsubFeed = useHub.subscribe((s, prev) => {
        if (s.jobsVersion !== prev.jobsVersion && s.jobsVersion != null) refreshQueue();
      });
    }
    return () => {
      holders -= 1;
      if (holders === 0) {
        if (timer) clearInterval(timer);
        unsubFeed?.();
        timer = undefined;
        unsubFeed = undefined;
      }
    };
  }, [active, up]);
}

type Tone = 'ok' | 'warn' | 'bad' | 'off' | 'info';

/** Lamp tone + a short word + one line of explanation for a backend's state. */
export function backendLook(b: TfBackend, st: TfBackends | null): { tone: Tone; word: string; line: string } {
  if (!st) return { tone: 'off', word: '…', line: 'Backend state unknown (the hub has no queue routes yet).' };
  if (b === 'modal') {
    const s: ModalState = st.modal.state;
    const looks: Record<ModalState, { tone: Tone; word: string; line: string }> = {
      ready: { tone: 'ok', word: 'ready', line: 'Modal GPU ready.' },
      warm: { tone: 'ok', word: 'warm', line: `Modal GPU warm — ${st.modal.slots} renders at a time.` },
      cold: { tone: 'info', word: 'cold', line: 'Modal GPU cold — the first render adds about a minute.' },
      unknown: { tone: 'off', word: 'no data', line: 'No Modal render since the hub started — the first one may be a cold start (≈ 1 min extra).' },
    };
    return looks[s] ?? looks.unknown;
  }
  const l = st.local;
  const gpu = l.gpu ? ` (${l.gpu})` : '';
  const map: Record<LocalState, { tone: Tone; word: string; line: string }> = {
    unconfigured: { tone: 'off', word: 'not set up', line: 'No desktop configured (TEXTUREFLOW_LOCAL_URL / _SSH in .env).' },
    off: { tone: 'bad', word: 'off', line: 'Desktop unreachable — it may be off. Renders wait in the queue until it answers.' },
    service_down: { tone: 'bad', word: 'service down', line: 'The desktop answers but its job service is down. Start it; queued renders wait.' },
    starting: { tone: 'warn', word: 'starting', line: `Starting the job service${gpu}…` },
    idle: { tone: 'ok', word: 'idle', line: `Desktop up${gpu}, GPU released — it loads the models on the next render (≈ 1 min extra).` },
    waiting_gpu: {
      tone: 'warn',
      word: 'GPU held',
      line: `Another process holds the GPU${l.gpu_holder ? ` (${l.gpu_holder})` : ''}; renders start when it lets go.`,
    },
    ready: { tone: 'ok', word: 'ready', line: `Desktop GPU ready${gpu}, $0 per render.` },
    busy: { tone: 'ok', word: 'busy', line: `Desktop GPU rendering${gpu}.` },
    down: { tone: 'bad', word: 'down', line: 'The job server is not answering; queued renders wait.' },
  };
  const look = map[l.state] ?? map.down;
  return l.detail ? { ...look, line: `${look.line} ${l.detail}` } : look;
}

// ---------- operator actions ----------

const toast = (t: string, tone: 'info' | 'warn' | 'bad' | 'ok' = 'info') => useStore.getState().toast(t, tone);

/** Put a lane's queued jobs in this order (drag, to top). */
export async function reorderLane(backend: TfBackend, ids: string[]) {
  useQueue.setState((s) => ({ order: { ...s.order, [backend]: { ids, resolvedAt: Infinity } } }));
  try {
    await reorderJobs(backend, ids);
  } catch (e) {
    toast(`Reorder failed: ${e}`, 'bad');
  }
  // The override holds until a snapshot newer than this reply arrives.
  useQueue.setState((s) => ({ order: { ...s.order, [backend]: { ids, resolvedAt: Date.now() + 300 } } }));
  refreshJobs();
  refreshQueue();
}

export function toTop(job: Job, lane: QueueEntry[]) {
  const queued = lane.filter(isQueued).map((j) => j.job);
  reorderLane(jobBackend(job), [job.job, ...queued.filter((id) => id !== job.job)]);
}

export async function moveToOtherLane(job: Job) {
  const to: TfBackend = jobBackend(job) === 'modal' ? 'local' : 'modal';
  try {
    await moveJob(job.job, to);
    toast(`Moved to ${BACKEND_LABEL[to]}`, 'ok');
  } catch (e) {
    toast(`Move refused: ${e}`, 'bad');
  }
  refreshJobs();
  refreshQueue();
}

export async function cancelQueued(job: Job) {
  try {
    await cancelJob(job.job);
  } catch (e) {
    toast(`Cancel failed: ${e}`, 'bad');
  }
  refreshJobs();
  refreshQueue();
}

/** Parallel Modal GPUs (1..max_slots): optimistic, then the hub's reply. Lowering it
 *  never cancels a running render — the lane drains down to the new limit. */
export async function setModalSlots(slots: number) {
  const prev = useQueue.getState().backends;
  if (prev) useQueue.setState({ backends: { ...prev, modal: { ...prev.modal, slots } } });
  try {
    const modal = await tfModalSlots(slots);
    useQueue.setState((s) => ({ backends: s.backends ? { ...s.backends, modal } : s.backends }));
  } catch (e) {
    toast(`Parallel GPUs not changed: ${e}`, 'bad');
    refreshQueue();
  }
}

/** Start the desktop's job service (only offered while it's `service_down`). */
export async function startLocal() {
  try {
    const local = await tfLocalStart();
    useQueue.setState((s) => ({ backends: s.backends ? { ...s.backends, local } : s.backends }));
  } catch (e) {
    toast(`Start failed: ${e}`, 'bad');
  }
  refreshQueue();
}

// ---------- card text ----------

export type JobKind = 'draft' | 'upscale' | 'full';

export function jobKind(j: Job): JobKind {
  if (j.args?.finish_from) return 'upscale';
  return j.args?.draft ? 'draft' : 'full';
}

const STAGE_WORD: Record<string, string> = {
  sample: 'sampling',
  sampling: 'sampling',
  first_pass: 'first pass',
  upscale: 'upscaling',
  esrgan: 'ESRGAN',
  rife: 'interpolating',
  interpolate: 'interpolating',
  encode: 'encoding',
  decode: 'decoding',
  load: 'loading models',
};

/** "sampling 7/12", "upscaling", "downloading", "queued · #2", … */
export function phaseText(j: Job, localState?: LocalState): string {
  const p = j.progress;
  if (j.status === 'queued') {
    if (p?.phase === 'waiting_backend') {
      return jobBackend(j) === 'local'
        ? localState === 'waiting_gpu'
          ? 'waiting for the GPU'
          : 'waiting for the desktop'
        : 'waiting for Modal';
    }
    return j.lane_pos != null ? `queued · #${j.lane_pos + 1}` : 'queued';
  }
  if (!p) return 'running';
  if (p.phase === 'rendering') {
    const word = p.stage ? STAGE_WORD[p.stage] ?? p.stage.replace(/_/g, ' ') : 'rendering';
    return p.step != null && p.total ? `${word} ${p.step}/${p.total}` : word;
  }
  return { queued: 'queued', waiting_backend: 'waiting', starting: 'starting', downloading: 'downloading', saving: 'saving' }[p.phase] ?? p.phase;
}

/** Seconds → "45 s", "3:05", "1 h 12". */
export function fmtEta(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '';
  const v = Math.max(0, Math.round(s));
  if (v < 60) return `${v} s`;
  if (v < 3600) return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}`;
  return `${Math.floor(v / 3600)} h ${String(Math.floor((v % 3600) / 60)).padStart(2, '0')}`;
}

/** Seconds until a lane is empty: the longest ETA in it (queued ETAs include the wait ahead). */
export function laneEta(lane: QueueEntry[]): number | null {
  const etas = lane
    .map((e) => (isForeign(e) ? e.progress?.eta_s : e.progress?.eta_s ?? e.eta_s))
    .filter((v): v is number => v != null);
  return etas.length ? Math.max(...etas) : null;
}

/** 0..1 across running jobs (null when none reports a percentage). */
export function aggregatePct(jobs: Job[]): number | null {
  const ps = jobs
    .filter((j) => j.status === 'running')
    .map((j) => j.progress?.pct ?? fracFromEta(j))
    .filter((v): v is number => v != null);
  return ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : null;
}

function fracFromEta(j: Job): number | null {
  const el = j.elapsed_s ?? 0;
  return j.eta_s != null && el + j.eta_s > 0 ? el / (el + j.eta_s) : null;
}

/** A running job's progress 0..1: the hub's pct, else elapsed / (elapsed + ETA). */
export function jobPct(j: Job): number | null {
  return j.progress?.pct ?? (j.status === 'running' ? fracFromEta(j) : null);
}
