// The hub + media library from the webview (studio plan L1–L5, A1).
//
// The hub (`python -m wzrd_mcp` on 127.0.0.1:8787, spawned/attached by the
// shell — src-tauri/src/hub.rs) owns library.db, jobs, the Studio selection
// and the spend cap. Writes go through `hubCall` (the shell's thin `/api`
// proxy). Reads that must work while the hub is down — folder listings,
// one render's record, tags — fall back to the shell's read-only library.db
// reader (src-tauri/src/library.rs), which returns the hub's exact shapes.
//
// Media reach the page as URLs, never base64: Tauri's asset protocol in the
// shell, the devport's `/file` route in the browser seat.

import { convertFileSrc } from '@tauri-apps/api/core';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { IN_TAURI, call, listenEvent } from './transport';

export type HubStatus = {
  state: 'up' | 'starting' | 'down' | 'foreign';
  pid: number | null;
  version: string | null;
  media_root: string;
  error: string | null;
};

/** One media file + its render (`library._item` in the hub). */
export type LibraryItem = {
  path: string;
  name: string;
  dir: string;
  render_id: string;
  variant: 'master' | 'hap' | null;
  fp: string | null;
  size: number | null;
  mtime: number | null;
  kind: 'image' | 'video' | null;
  codec: string | null;
  w: number | null;
  h: number | null;
  frames: number | null;
  fps: number | null;
  duration_s: number | null;
  missing: boolean;
  /** The engine can play it (HAP or a still). */
  playable: boolean;
  /** Made by WZRD (tagged), vs a foreign file identified by fingerprint. */
  wzrd: boolean;
  tool: string | null;
  created: string | null;
  project: string | null;
  region: string | null;
  est_usd: number | null;
  score: number | null;
  tags: string[];
  agent_tags: string[];
  notes: string | null;
  /** Absolute path of the cached thumbnail, or null. */
  thumb: string | null;
};

export type FolderListing = {
  dir: string;
  root: string | null;
  dirs: { name: string; path: string }[];
  items: LibraryItem[];
  /** Media on disk the database doesn't know yet (being indexed). */
  pending: string[];
};

export type RenderRecord = {
  id: string;
  kind: 'wzrd' | 'foreign' | null;
  tool: string | null;
  created: string | null;
  project: string | null;
  region: string | null;
  job: string | null;
  parent: string | null;
  /** The L2 creation record: `{tool, args, inputs, est_usd, canvas, …}`. */
  record: Record<string, any>;
  score: number | null;
  tags: string[];
  agent_tags: string[];
  notes: string | null;
  signals: Record<string, any>;
  purged: boolean;
  thumb: string | null;
  files: LibraryItem[];
};

export type Root = { path: string; home: boolean; added?: string; exists?: boolean };

export type Selection = {
  styles: string[];
  items: string[];
  target: string | null;
  project: string | null;
};

export type Job = {
  job: string;
  tool: string;
  status: 'running' | 'done' | 'failed' | 'cancelled' | 'lost';
  by: 'agent' | 'app';
  args: Record<string, any>;
  started: number;
  ended: number | null;
  duration_s: number | null;
  est_usd: number;
  error: string | null;
  outputs: string[];
  dest: string | null;
  project: string | null;
  session: string | null;
  elapsed_s?: number;
  eta_s?: number | null;
};

export type Spend = {
  today_usd: number;
  cap_usd: number;
  running_usd: number;
  warn_at: number;
  warn: boolean;
  over: boolean;
};

type JobsFeed = { version: number; now: number; jobs: Job[]; spend: Spend };

// ---------- hub lifecycle ----------

export const hubStatus = () => call<HubStatus>('hub_status');
export const hubEnsure = () => call<HubStatus>('hub_ensure');
export const hubRestart = () => call<HubStatus>('hub_restart');
export const onHubStatus = (h: (s: HubStatus) => void): Promise<UnlistenFn> =>
  listenEvent<HubStatus>('hub:status', h);

/** One `/api` call through the shell. Rejects with the hub's `{error}`. */
export const hubCall = <T>(method: 'GET' | 'POST' | 'PUT', path: string, body?: unknown) =>
  call<T>('hub_call', { method, path, body: body ?? null });

const q = (params: Record<string, string | number | undefined>) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join('&');

// ---------- library reads (hub when up, read-only DB when down) ----------

/** A folder. With the hub up this also indexes what's new (S3: index on
 *  open); down, it lists what library.db already knows. */
export async function listFolder(dir: string, hubUp: boolean): Promise<FolderListing> {
  if (hubUp) {
    try {
      return await hubCall<FolderListing>('GET', `/api/library/list?${q({ dir })}`);
    } catch (e) {
      console.warn('hub list failed, reading library.db:', e);
    }
  }
  return call<FolderListing>('library_dir', { dir });
}

export const libraryRoots = () => call<Root[]>('library_roots');
export const libraryRecord = (id: string) => call<RenderRecord>('library_record', { id });
export const libraryTags = () => call<Record<string, number>>('library_tags');

// ---------- library writes (hub only) ----------

export const indexFolder = (path: string, deep: boolean) =>
  hubCall<{ dir: string; files: number; queued: number; pending: number }>('POST', '/api/library/index', {
    path,
    deep,
  });
export type LabelPatch = {
  score?: number | null;
  tags?: string[];
  add_tags?: string[];
  remove_tags?: string[];
  notes?: string | null;
};
export const labelRenders = (refs: { ids?: string[]; paths?: string[] }, patch: LabelPatch) =>
  hubCall<{ renders: unknown[] }>('POST', '/api/library/label', { ...refs, ...patch });
export const moveFiles = (paths: string[], dest: string) =>
  hubCall<{ moved: unknown[] }>('POST', '/api/library/move', { paths, dest });
export const deleteFiles = (paths: string[]) => hubCall<unknown>('POST', '/api/library/delete', { paths });
export const deleteScored = (dir: string, maxScore: number, dryRun = false) =>
  hubCall<{ deleted?: string[]; would_delete?: string[]; count?: number }>('POST', '/api/library/delete_scored', {
    dir,
    max_score: maxScore,
    dry_run: dryRun,
  });
export const addRoot = (path: string) => hubCall<{ path: string; roots: Root[] }>('POST', '/api/roots', { path });
export const removeRoot = (path: string) => hubCall<{ roots: Root[] }>('POST', '/api/roots/remove', { path });

// ---------- selection (L5), jobs, spend ----------

export const getSelection = () => hubCall<Selection>('GET', '/api/selection');
export const putSelection = (patch: Partial<Selection>) => hubCall<Selection>('PUT', '/api/selection', patch);
/** Long-poll: resolves when the job set changed since `version` (or after `wait` s). */
export const jobsFeed = (version?: number, wait = 20) =>
  hubCall<JobsFeed>('GET', `/api/jobs?${q({ version, wait: version === undefined ? 0 : wait, limit: 60 })}`);

// ---------- media URLs ----------

/** A URL the page can load an absolute library path from (thumbnail, mp4). */
export function mediaUrl(path: string): string {
  if (IN_TAURI) return convertFileSrc(path);
  return `/__shell/file?path=${encodeURIComponent(path)}`;
}
