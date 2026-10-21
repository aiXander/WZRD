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
  /** The args its job was submitted with (the tool's own inputs), or null. */
  job_args?: Record<string, any> | null;
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
/** Items for many paths (deck slots): indexed rows, else bare `{path, name}`;
 *  `exists` says whether the file is on disk. */
export const libraryItems = (paths: string[]) =>
  call<(Partial<LibraryItem> & { path: string; exists: boolean })[]>('library_items', { paths });
/** The newest playable renders made for a show (D1's incoming strip). */
export const libraryIncoming = (project: string, limit = 24) =>
  call<LibraryItem[]>('library_incoming', { project, limit });

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

// ---------- rendering (H2: every render is a non-blocking job) ----------

/** The render-tracked tools `/api/render` runs. */
export type RenderTool = 'texture_flow' | 'nano_banana' | 'kling_v25_image_to_video' | 'to_hap';
type StartedJob = { job: string; status: string; dest: string | null; eta_s: number | null; note?: string };
export type Estimate = { tool: string; usd: number; spend: Spend };

/** Start `count` jobs of `tool` (each its own seed unless `args.seed` is set). */
export const startRender = (tool: RenderTool, args: Record<string, unknown>, count = 1) =>
  hubCall<{ jobs: StartedJob[] }>('POST', '/api/render', { tool, args, count });
export const estimateRender = (tool: RenderTool, args: Record<string, unknown>, count = 1) =>
  hubCall<Estimate>('POST', '/api/estimate', { tool, args, count });
export const cancelJob = (job: string) => hubCall<unknown>('POST', '/api/jobs/cancel', { job });

export type Recipe = {
  name: string;
  when_to_use?: string;
  look?: string;
  images?: number;
  preview?: string | null;
};
export const listRecipes = () => hubCall<{ root: string; recipes: Recipe[] }>('GET', '/api/recipes');

/** One TextureFlow arg as the endpoint publishes it (textureflow-lab `tflab/spec.yaml`). */
export type TfParam = {
  type?: string;
  default?: unknown;
  minimum?: number;
  maximum?: number;
  step?: number;
  choices?: string[];
  label?: string;
  description?: string;
};
/** The spec `texture_flow` was built from (the hub's cache of the app's `spec()`, else its snapshot). */
export type TfSpec = { app: string | null; version: string; source: string; parameters: Record<string, TfParam> };
export const getTextureflowSpec = () => hubCall<TfSpec>('GET', '/api/textureflow/spec');

// ---------- media URLs ----------

/** A URL the page can load an absolute library path from (thumbnail, mp4). */
export function mediaUrl(path: string): string {
  if (IN_TAURI) return convertFileSrc(path);
  return `/__shell/file?path=${encodeURIComponent(path)}`;
}
