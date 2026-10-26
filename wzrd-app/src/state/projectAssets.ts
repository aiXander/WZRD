// One shared `ProjectInfo` per show (show-setup A1/G4): its named assets
// (`<project>/assets.json`), roles, folder map and canvas, from the hub's
// `GET /api/project`. Every component that shows assets (AssetPicker,
// MediaGallery pin chips, RecordPanel, the setup wizard) reads this cache, so
// a pin from one shows in the others at once.
//
// Polled every 3 s per show while anything holds it (refcounted
// `useProjectInfo`), so a pin from another seat or the wizard shows up live. The hub is the
// only writer of assets.json; after an app-side pin call `refreshProjectInfo`.

import { useEffect } from 'react';
import { create } from 'zustand';
import { projectInfo, type ProjectInfo } from '../api/project';
import { hubUp, useHub } from './hub';

const POLL_MS = 3000;

type S = {
  infos: Record<string, ProjectInfo>;
  errors: Record<string, string | null>;
};

const useInfos = create<S>(() => ({ infos: {}, errors: {} }));

const holders = new Map<string, number>();
const timers = new Map<string, ReturnType<typeof setInterval>>();
const inflight = new Map<string, Promise<void>>();

export function refreshProjectInfo(project: string): Promise<void> {
  const running = inflight.get(project);
  if (running) return running;
  const p = (async () => {
    if (!hubUp()) {
      useInfos.setState((s) => ({ errors: { ...s.errors, [project]: 'The content hub is down' } }));
      return;
    }
    try {
      const info = await projectInfo(project);
      useInfos.setState((s) => ({
        infos: { ...s.infos, [project]: info },
        errors: { ...s.errors, [project]: null },
      }));
    } catch (e) {
      useInfos.setState((s) => ({ errors: { ...s.errors, [project]: String(e) } }));
    }
  })().finally(() => inflight.delete(project));
  inflight.set(project, p);
  return p;
}

function hold(project: string) {
  holders.set(project, (holders.get(project) ?? 0) + 1);
  if (!timers.has(project)) {
    refreshProjectInfo(project);
    timers.set(
      project,
      setInterval(() => refreshProjectInfo(project), POLL_MS)
    );
  }
}

function release(project: string) {
  const n = (holders.get(project) ?? 1) - 1;
  if (n > 0) return void holders.set(project, n);
  holders.delete(project);
  clearInterval(timers.get(project));
  timers.delete(project);
}

/** The show's info, kept fresh while mounted. `project` null = nothing. */
export function useProjectInfo(project: string | null): {
  info: ProjectInfo | null;
  error: string | null;
  refresh: () => Promise<void>;
} {
  const info = useInfos((s) => (project ? s.infos[project] ?? null : null));
  const error = useInfos((s) => (project ? s.errors[project] ?? null : null));
  const up = useHub((s) => s.status?.state === 'up');
  useEffect(() => {
    if (!project) return;
    hold(project);
    return () => release(project);
  }, [project]);
  // The hub came (back) up: don't wait for the next tick.
  useEffect(() => {
    if (project && up) refreshProjectInfo(project);
  }, [project, up]);
  return { info, error, refresh: () => (project ? refreshProjectInfo(project) : Promise.resolve()) };
}

/** Absolute path → the asset names pinned to it. */
export function pinsByPath(info: ProjectInfo | null): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  if (!info) return out;
  for (const [name, a] of Object.entries(info.assets ?? {})) {
    if (!a?.path) continue;
    (out[a.path] ??= []).push(name);
  }
  return out;
}

/** The well-known asset names (G2) — the fallback while `roles` hasn't loaded. */
export const DEFAULT_ROLES = ['photo_lit', 'photo_ambient', 'surface_lit', 'surface', 'surface_ambient', 'segmentation'];
