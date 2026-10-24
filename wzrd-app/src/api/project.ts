// Show setup + project assets from the webview (docs/TODO/show-setup.md).
//
// Everything here is a hub `/api` call through the shell's `hub_call` proxy
// (src-tauri/src/hub.rs) — no new shell commands. The hub is the only writer
// of `<project>/assets.json`, the prompt files and the trash.

import { hubCall, type LibraryItem } from './hub';

const q = (params: Record<string, string | number | undefined | null>) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join('&');

// ---------- project info + named assets (contract A1) ----------

/** One named asset of a show (`<project>/assets.json`). */
export type Asset = {
  name: string;
  /** Absolute path, resolved (a moved file is followed by render id / fingerprint). */
  path: string;
  id: string | null;
  fp: string | null;
  kind: 'image' | 'video' | null;
  note: string;
  pinned: string;
  by: 'app' | 'agent';
  /** False when the file is gone and couldn't be followed. */
  exists: boolean;
};

/** The well-known asset names and what each is for (`assets.ROLES`). */
export type Roles = Record<string, string>;

export type ProjectInfo = {
  project: string;
  project_dir: string;
  exists: boolean;
  /** Locked canvas `[w, h]`, or null (a legacy show, or not locked yet). */
  canvas: [number, number] | null;
  /** The folder map (G4): folder name → absolute path + what goes there. */
  layout: Record<string, { path: string; what: string }>;
  assets: Record<string, Asset>;
  roles: Roles;
  /** Canvas choices for a new show: aspect name → its 1K size. */
  canvases: { name: string; w: number; h: number }[];
  /** True once `scene.json` exists (the engine can open the show). */
  has_scene: boolean;
  scene_path: string;
  existing_projects: string[];
};

export const projectInfo = (project?: string | null) =>
  hubCall<ProjectInfo>('GET', `/api/project?${q({ project })}`);

/** A region's mask + the show's surface cut to its placement box (the Render panel's auto-fill). */
export type RegionInputs = { region: string; source_box: number[]; mask: string; shape: string | null; shape_from: string | null };
export const regionInputs = (project: string, region: string) =>
  hubCall<RegionInputs>('GET', `/api/project/region_inputs?${q({ project, region })}`);

export const pinAsset = (project: string, name: string, path: string, note?: string) =>
  hubCall<{ assets: Record<string, Asset> }>('POST', '/api/project/assets', { project, name, path, note });
export const unpinAsset = (project: string, name: string) =>
  hubCall<{ assets: Record<string, Asset> }>('POST', '/api/project/assets', { project, name, path: null });

// ---------- any non-render tool, synchronously (G8) ----------

/** Run one MCP tool in the hub and get its JSON result. Render tools
 *  (nano_banana, texture_flow, kling_*, to_hap) go through `startRender`. */
export const runTool = <T = Record<string, any>>(tool: string, args: Record<string, unknown>) =>
  hubCall<T>('POST', '/api/tool', { tool, args });

export type RectifyArgs = {
  project: string;
  lit_photo: string;
  ambient_photo?: string | null;
  /** Normalised [x, y] in the lit photo, TL, TR, BR, BL. Omit = auto-detect. */
  corners?: [number, number][] | null;
  margin?: number;
  min_projection_fraction?: number;
  max_brightness?: number;
  detail_boost?: number;
  chroma_correction?: number;
  gradient_source?: number;
};
export type RectifyResult = {
  run: string;
  surface_lit: string;
  surface: string;
  surface_ambient: string | null;
  corners: [number, number][];
  /** True when `corners` came from the caller, false = auto-detected. */
  manual_corners: boolean;
  align: { method: string | null; confidence: number | null; num_inliers: number | null } | null;
  size: [number, number];
};
export const rectifySurface = (args: RectifyArgs) => runTool<RectifyResult>('rectify_surface', args);

export type Region = {
  region_mask: string;
  region_crop_mask?: string;
  region_surface_image?: string;
  source_box: { x: number; y: number; width: number; height: number } | null;
  aspect_ratio?: string;
  color?: string;
  area_fraction?: number;
};
export type RegionsResult = { regions: Region[]; regions_dir: string; metadata_path: string | null };

// ---------- prompt templates (contract P1) ----------

export type Prompt = {
  kind: string;
  name: string;
  scope: 'project' | 'wzrd';
  prompt: string;
  notes: string | null;
  updated: string | null;
  path: string;
};

export const listPrompts = (kind: string, project?: string | null) =>
  hubCall<{ prompts: Prompt[] }>('GET', `/api/prompts?${q({ kind, project })}`);
export const savePrompt = (p: {
  kind: string;
  name: string;
  prompt: string;
  scope: 'project' | 'wzrd';
  project?: string | null;
  notes?: string | null;
}) => hubCall<Prompt>('POST', '/api/prompts', p);

/** `{n_regions}` is the one placeholder (plain replace, prompts may hold braces). */
export const fillPrompt = (text: string, nRegions: number) => text.split('{n_regions}').join(String(nRegions));

// ---------- trash (contract T1) ----------

export type TrashItem = {
  id: string;
  /** Where it lived (and returns to on restore). */
  path: string;
  name: string;
  dir: string;
  render_id: string | null;
  size: number | null;
  /** Epoch seconds. */
  deleted: number;
  /** Epoch seconds when it's flushed for good. */
  expires: number;
  kind: LibraryItem['kind'];
  thumb: string | null;
};

export const trashList = () => hubCall<{ ttl_s: number; items: TrashItem[] }>('GET', '/api/trash');
export const trashRestore = (ids: string[]) =>
  hubCall<{ restored: { id: string; path: string }[]; errors: string[] }>('POST', '/api/trash/restore', { ids });
export const trashFlush = (ids?: string[]) => hubCall<{ flushed: number }>('POST', '/api/trash/flush', ids ? { ids } : {});
