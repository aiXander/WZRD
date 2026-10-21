// The show-setup wizard's state (docs/TODO/show-setup.md, W1).
//
// The wizard works on its *own* show (`project`, default = the open one) and
// passes `project=` on every hub call (G1) — the engine keeps playing the open
// show meanwhile. What a step computed but hasn't pinned yet (the last surface
// run, the last region extract, the prompt being edited) lives here per show,
// so jumping between steps — or closing and reopening the wizard — keeps it.
// Everything durable is the hub's: pinned assets in `<project>/assets.json`,
// files in the show's folders.

import { create } from 'zustand';
import type { RectifyResult, RegionsResult } from '../api/project';
import { showName } from './hub';

export type SetupStep = 'show' | 'photos' | 'surface' | 'segmentation' | 'regions';
export const SETUP_STEPS: { id: SetupStep; label: string }[] = [
  { id: 'show', label: 'Show' },
  { id: 'photos', label: 'Photos' },
  { id: 'surface', label: 'Surface' },
  { id: 'segmentation', label: 'Segmentation' },
  { id: 'regions', label: 'Regions' },
];

/** The detection + darken knobs of `rectify_surface` (defaults = the tool's
 *  own, wzrd_mcp/tools.py::rectify_surface). */
export type SurfaceSettings = {
  margin: number;
  min_projection_fraction: number;
  max_brightness: number;
  detail_boost: number;
  chroma_correction: number;
  gradient_source: number;
};
export const SURFACE_DEFAULTS: SurfaceSettings = {
  margin: 0.01,
  min_projection_fraction: 0.1,
  max_brightness: 0.2,
  detail_boost: 1.15,
  chroma_correction: 0.5,
  gradient_source: 0.5,
};

export type Corners = [number, number][];

export type SurfaceDraft = {
  settings: SurfaceSettings;
  /** The quad on the lit photo (normalised, TL TR BR BL), or null = none drawn yet. */
  corners: Corners | null;
  /** True once a handle was moved: Run sends `corners`; false = auto-detect. */
  manual: boolean;
  /** The run on screen (fresh or a trial loaded back). */
  run: RectifyResult | null;
  /** The lit photo the corners belong to (a new lit photo drops them). */
  litPath: string | null;
};

export type RegionsSettings = {
  max_colors: number;
  min_area_fraction: number;
  delta_e_threshold: number;
  background_threshold: number;
  merge_same_color: boolean;
};
export const REGIONS_DEFAULTS: RegionsSettings = {
  max_colors: 8,
  min_area_fraction: 0.01,
  delta_e_threshold: 5,
  background_threshold: 15,
  merge_same_color: true,
};

export type RegionsDraft = {
  settings: RegionsSettings;
  /** The last extract and the inputs it came from. */
  result: RegionsResult | null;
  segmentation: string | null;
  surface: string | null;
  /** Parallel to `result.regions`. */
  ids: string[];
  labels: string[];
  packName: string;
  built: { pack_dir: string; layer_count: number; note?: string; scene_json?: string } | null;
};

export type SegDraft = {
  /** `${scope}:${name}` of the template the text was forked from. */
  from: string | null;
  text: string | null;
  nRegions: number;
  count: number;
  resolution: '1K' | '2K';
  thinking: 'minimal' | 'medium';
  input: string | null;
};
const SEG_DEFAULTS: SegDraft = {
  from: null,
  text: null,
  nRegions: 6,
  count: 4,
  resolution: '1K',
  thinking: 'medium',
  input: null,
};

type SetupState = {
  open: boolean;
  /** The show this wizard works on; null = none yet (a new show being made). */
  project: string | null;
  step: SetupStep;
  /** Show step opens on the "new show" form. */
  newShow: boolean;
  surface: Record<string, SurfaceDraft>;
  regions: Record<string, RegionsDraft>;
  seg: Record<string, SegDraft>;
};

export const useSetup = create<SetupState>(() => ({
  open: false,
  project: null,
  step: 'show',
  newShow: false,
  surface: {},
  regions: {},
  seg: {},
}));

export function openSetup(opts: { project?: string | null; step?: SetupStep; newShow?: boolean } = {}) {
  const project = opts.newShow ? null : opts.project ?? showName();
  const prev = useSetup.getState();
  useSetup.setState({
    open: true,
    project,
    newShow: !!opts.newShow || !project,
    // Reopening on the same show resumes where you were; another show starts at Show.
    step: opts.newShow || !project ? 'show' : opts.step ?? (prev.project === project ? prev.step : 'show'),
  });
}

export const closeSetup = () => useSetup.setState({ open: false });
export const setSetupStep = (step: SetupStep) => useSetup.setState({ step });
export const setSetupProject = (project: string) => useSetup.setState({ project, newShow: false });
export const setNewShow = (newShow: boolean) => useSetup.setState({ newShow });

// ---------- per-show drafts ----------

// Constant empties, so a selector over a show with no draft yet returns the
// same object every time.
const EMPTY_SURFACE: SurfaceDraft = { settings: SURFACE_DEFAULTS, corners: null, manual: false, run: null, litPath: null };

export function surfaceDraft(s: SetupState, project: string): SurfaceDraft {
  return s.surface[project] ?? EMPTY_SURFACE;
}
export function patchSurface(project: string, patch: Partial<SurfaceDraft>) {
  useSetup.setState((s) => ({ surface: { ...s.surface, [project]: { ...surfaceDraft(s, project), ...patch } } }));
}

const EMPTY_REGIONS: RegionsDraft = {
  settings: REGIONS_DEFAULTS,
  result: null,
  segmentation: null,
  surface: null,
  ids: [],
  labels: [],
  packName: '',
  built: null,
};

export function regionsDraft(s: SetupState, project: string): RegionsDraft {
  return s.regions[project] ?? EMPTY_REGIONS;
}
export function patchRegions(project: string, patch: Partial<RegionsDraft>) {
  useSetup.setState((s) => ({ regions: { ...s.regions, [project]: { ...regionsDraft(s, project), ...patch } } }));
}

export function segDraft(s: SetupState, project: string): SegDraft {
  return s.seg[project] ?? SEG_DEFAULTS;
}
export function patchSeg(project: string, patch: Partial<SegDraft>) {
  useSetup.setState((s) => ({ seg: { ...s.seg, [project]: { ...segDraft(s, project), ...patch } } }));
}

/** A region id the engine and the layer pack accept: lowercase slug. */
export function slugId(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48);
}
