// Hub lamp + job feed state (studio plan A1). Its own small store, so the
// engine store stays per-project: the hub outlives every project switch.
//
// Status: pushed by the shell as `hub:status` (src-tauri/src/hub.rs — its
// monitor re-checks `/api/hello` every 3 s); the browser seat gets no Tauri
// events, so it polls `hub_status` instead.
//
// Jobs: one long-poll loop on `/api/jobs` (returns when the job set
// changes, else after 20 s), run only while something holds `useJobsFeed`
// — the Studio room and the jobs pill. Running jobs refresh their elapsed /
// ETA every poll, which `useJobsFeed` shortens to 2 s while any job runs.

import { useEffect } from 'react';
import { create } from 'zustand';
import { IN_TAURI } from '../api/transport';
import { useStore } from './store';
import {
  hubEnsure,
  hubRestart,
  hubStatus,
  jobsFeed,
  onHubStatus,
  putSelection,
  type HubStatus,
  type Job,
  type Spend,
} from '../api/hub';

type HubState = {
  status: HubStatus | null;
  jobs: Job[];
  spend: Spend | null;
  jobsVersion: number | null;
  /** Wall clock of the hub at the last feed (elapsed is relative to it). */
  jobsAt: number;
  setStatus: (s: HubStatus) => void;
  setFeed: (version: number, jobs: Job[], spend: Spend) => void;
};

export const useHub = create<HubState>((set) => ({
  status: null,
  jobs: [],
  spend: null,
  jobsVersion: null,
  jobsAt: 0,
  setStatus: (status) => set({ status }),
  setFeed: (jobsVersion, jobs, spend) => set({ jobsVersion, jobs, spend, jobsAt: Date.now() }),
}));

export const hubUp = () => useHub.getState().status?.state === 'up';

/** The open show's folder name — what MCP tools take as `project=`. */
export function showName(): string | null {
  const dir = useStore.getState().project?.dir;
  return dir ? dir.replace(/\/+$/, '').split('/').pop() ?? null : null;
}

/** App-level: keep the lamp current for the app's life, and tell the hub
 *  which show is open (L5 `selection.project`): an agent tool called
 *  without `project=` acts on it. Written on every show switch and every
 *  time the hub comes (back) up. */
export function useHubWire() {
  const up = useHub((s) => s.status?.state === 'up');
  const dir = useStore((s) => s.project?.dir);
  useEffect(() => {
    const project = showName();
    if (!up || !project) return;
    putSelection({ project }).catch((e) => console.warn('selection.project:', e));
  }, [up, dir]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let timer: ReturnType<typeof setInterval> | undefined;
    const refresh = () =>
      hubStatus()
        .then((s) => useHub.getState().setStatus(s))
        .catch(() => {});
    refresh();
    if (IN_TAURI) {
      onHubStatus((s) => useHub.getState().setStatus(s)).then((u) => (unlisten = u));
    } else {
      timer = setInterval(refresh, 3000);
    }
    return () => {
      unlisten?.();
      if (timer) clearInterval(timer);
    };
  }, []);
}

export async function ensureHub() {
  useHub.getState().setStatus({ ...(useHub.getState().status ?? emptyStatus()), state: 'starting', error: null });
  try {
    useHub.getState().setStatus(await hubEnsure());
  } catch (e) {
    useHub.getState().setStatus({ ...emptyStatus(), state: 'down', error: String(e) });
  }
}

export async function restartHub() {
  useHub.getState().setStatus({ ...(useHub.getState().status ?? emptyStatus()), state: 'starting', error: null });
  try {
    useHub.getState().setStatus(await hubRestart());
  } catch (e) {
    useHub.getState().setStatus({ ...emptyStatus(), state: 'down', error: String(e) });
  }
}

function emptyStatus(): HubStatus {
  return { state: 'down', pid: null, version: null, media_root: '', error: null };
}

// ---------- the job feed (refcounted long-poll) ----------

let holders = 0;
let loopRunning = false;
let kick: (() => void) | null = null;

async function feedLoop() {
  if (loopRunning) return;
  loopRunning = true;
  try {
    while (holders > 0) {
      if (!hubUp()) {
        await sleep(2000);
        continue;
      }
      const st = useHub.getState();
      const running = st.jobs.some((j) => j.status === 'running');
      try {
        // While something runs, poll briefly so elapsed / ETA tick.
        const feed = await jobsFeed(st.jobsVersion ?? undefined, running ? 2 : 20);
        useHub.getState().setFeed(feed.version, feed.jobs, feed.spend);
      } catch {
        await sleep(2000);
      }
    }
  } finally {
    loopRunning = false;
  }
}

function sleep(ms: number) {
  return new Promise<void>((r) => {
    const t = setTimeout(r, ms);
    kick = () => {
      clearTimeout(t);
      r();
    };
  });
}

/** Refresh the job feed now (after starting / cancelling a job). */
export function refreshJobs() {
  useHub.setState({ jobsVersion: null });
  kick?.();
  if (holders > 0) feedLoop();
}

/** Hold the job feed while mounted. */
export function useJobsFeed(active = true) {
  useEffect(() => {
    if (!active) return;
    holders += 1;
    feedLoop();
    return () => {
      holders -= 1;
    };
  }, [active]);
}
