// Thin wrappers around Tauri's invoke() for every engine command. Keeping
// the surface small + typed means the rest of the front-end doesn't import
// '@tauri-apps/api/core' directly — handy if we ever swap transport.

import type { UnlistenFn } from '@tauri-apps/api/event';
import { call as invoke, listenEvent } from './transport';

export type PackLayer = {
  id: string;
  slice: number;
  mask_path: string;
  label: string | null;
  tags: string[];
  bbox: [number, number, number, number] | null;
  centroid: [number, number] | null;
  z: number;
};

export type PackInfo = {
  pack_dir: string;
  width: number;
  height: number;
  layers: PackLayer[];
  groups: { id: string; members: string[] }[];
};

export type WgslDiagnostic = {
  severity: 'error' | 'warning';
  line: number;
  column: number;
  end_line: number;
  end_column: number;
  message: string;
};

export type WgslValidateResult = {
  ok: boolean;
  diagnostics: WgslDiagnostic[];
};

export type EngineStatus = {
  running: boolean;
  ws_addr: string | null;
  last_error: string | null;
};

/** §5.6 — one probed pipeline's verdict. */
export type ProbeVerdict = {
  key: string;
  label: string;
  predicted_p95_ms: number;
  band: 'green' | 'yellow' | 'red';
  thumbnail_b64: string | null;
};

/** §5.6 — pre-flight probe report riding `hot_reload` + apply replies. */
export type ProbeReport = {
  compiled: boolean;
  predicted_p95_ms: number;
  band: 'green' | 'yellow' | 'red';
  thumbnail_b64: string | null;
  verdicts: ProbeVerdict[];
};

/** §5.6 — sticky `deck` channel payload (two-leg state). */
export type DeckPayload = {
  promote: 'idle' | 'pending' | 'ramping';
  mix: number;
  fade_ms: number | null;
  quantize: string | null;
  preview_source: 'live' | 'design';
  two_leg: boolean;
};

export type ProbeThresholds = { a_ms: number; b_ms: number };

/** Perf-monitor PM1 — p50/p95/max in ms over the engine's last ~300 frames. */
export type PerfPct = { p50: number; p95: number; max: number };

/**
 * Perf-monitor PM1 — the 2 Hz `perf` channel (not sticky) and the
 * `perf.snapshot` result. `cpu` is keyed by render-thread stage
 * (`inbound.cmds`, `tick.live`, `encode`, `acquire`, `pace`, `host`, …);
 * `frame.work` = Σ stages − acquire − pace; `headroom` =
 * 1 − max(work p95, gpu.total p95) / budget.
 */
export type PerfPayload = {
  budget_ms: number;
  headroom: number | null;
  presenting: boolean;
  frames: number;
  cpu: Record<string, PerfPct>;
  frame: { interval: PerfPct; work: PerfPct; unaccounted: PerfPct; monitor: PerfPct };
  dropped: { window: number; total: number };
  /** PM2 GPU timestamps; null when off (`--gpu-timing 0` / `WZRD_GPU_TIMING=0`)
   *  or the adapter lacks timestamp queries. Sampled 1 frame in `every`. */
  gpu: PerfGpu | null;
  /** CPU % of one core; null where the platform has no thread CPU clock. */
  threads: { name: string; cpu_pct: number | null }[];
  clips: {
    label: string;
    decode_p50: number | null;
    decode_p95: number | null;
    /** Wanted frame not ready (decoder behind) since the previous emit. */
    stale: number;
    stale_total: number;
    queue: number;
  }[];
  clips_idle: number;
  latency: {
    knob_submit_p95: number | null;
    rpc_queue_p95: number | null;
    osc_age_p50: number | null;
  };
  process: {
    cpu_pct: number | null;
    rss_mb: number | null;
    gpu_mb: number | null;
    resources: Record<string, number> | null;
  };
  telemetry_bps: { ws: Record<string, number>; webview: Record<string, number> };
  last_change: { epoch: number; rev: number; actor: string; facet: string } | null;
  spikes: { trigger_ms: number; total: number; suppressed: number };
};

/** Perf-monitor PM2 — `perf.gpu`. `total` = Σ per-submit first-begin → last-end
 *  (the budget truth); `passes` keyed `warp_bake`, `composite.live`,
 *  `composite.design`, `final`, `preview`, `probe` — only those that ran in the
 *  window; main-submit passes are charged from the previous pass's end, so they
 *  tile `total`. Clip uploads are untimed. */
export type PerfGpu = {
  every: number;
  /** Timed frames in the aggregate window (~10 s). */
  frames: number;
  /** Sampled frames skipped because every readback slot was in flight. */
  skipped: number;
  total: PerfPct | null;
  passes: Record<string, PerfPct & { n: number }>;
};

/** Perf-monitor PM1 — one frame of a spike capture (offset 0 = the spike). */
export type PerfSpikeFrame = {
  offset: number;
  interval_ms: number;
  work_ms: number;
  presenting: boolean;
  stages: Record<string, number>;
  ran: {
    cmds: string[];
    cmd_count: number;
    reloads: number;
    probe_steps: number;
    clip_opens: number;
    rev: number;
  };
};

/** Perf-monitor PM1 — sticky `perf_spike`: a frame over `trigger_ms`, ± 5 neighbours. */
export type PerfSpike = {
  ts_ms: number;
  frame_ms: number;
  trigger_ms: number;
  budget_ms: number;
  worst_stage: string;
  worst_stage_ms: number;
  frames: PerfSpikeFrame[];
  suppressed: number;
};

/** §5.14 — one extra (non-corner) warp handle. */
export type WarpPoint = {
  id: string;
  /** Where the handle grabs the content, source uv. */
  anchor: [number, number];
  /** Where the operator dragged it, dest uv (may lie outside [0,1]). */
  dest: [number, number];
  /** Support radius in dest-normalized units. */
  radius: number;
};

/** §5.14 — sticky `alignment` channel payload / `alignment.get` result. */
export type AlignmentDoc = {
  version: number;
  enabled: boolean;
  /** `#rrggbb`. Non-black floods the surface with light — warn while it is. */
  background: string;
  /** Dest positions of source corners (0,0) (1,0) (1,1) (0,1). Always 4. */
  corners: [number, number][];
  points: WarpPoint[];
  /**
   * Solved RBF coefficients, one per point, same order — read-only derived
   * state so a client can draw the *actual* field instead of re-solving it.
   * Never persisted, ignored on input.
   */
  weights?: [number, number][];
  /** Projector swapchain size in px — what a one-pixel nudge means. */
  output: [number, number];
  points_max: number;
  test_pattern: 'none' | 'grid' | 'border' | 'corners';
  solve_ok: boolean;
};

/** A partial merge for `alignment.set`; anchor omitted ⇒ engine anchors it. */
export type AlignmentPatch = {
  enabled?: boolean;
  background?: string;
  corners?: [number, number][];
  points?: { id?: string; anchor?: [number, number]; dest: [number, number]; radius?: number }[];
};

/**
 * §5.10 — one design mutation on the sticky `changes` channel. The webview
 * re-pulls the affected facet when `actor !== 'ui'` (agent/watcher edits),
 * closing the reverse-sync loop so both seats always agree.
 */
export type ChangeEntry = {
  epoch: number;
  rev: number;
  ts_ms: number;
  actor: 'ui' | 'agent' | 'system';
  facet: 'bindings' | 'effects' | 'layers';
  summary: string;
};

// ---------- commands ----------

export const engineStatus = () => invoke<EngineStatus>('engine_status');
export const packInfo = () => invoke<PackInfo>('pack_info');
export const sceneGetState = (leg?: 'live' | 'design') =>
  invoke<{ json: string; leg: string; epoch: number; rev: number }>('scene_get_state', {
    leg: leg ?? null,
  });
/**
 * Live tempo + downbeat. Tempo changes keep the beat grid continuous;
 * `resync` puts a bar line at "now"; `sync` snaps design's clock onto live's.
 * Omit `leg` to move both legs together.
 */
export const transportSet = (opts: { bpm?: number; resync?: boolean; sync?: boolean; leg?: LegName }) =>
  invoke<unknown>('transport_set', {
    bpm: opts.bpm ?? null,
    resync: opts.resync ?? false,
    sync: opts.sync ?? false,
    leg: opts.leg ?? null,
  });
/** One leg's knob values + override table (incl. `@opacity`/`@mute`/`@solo`). */
export const paramList = (leg: LegName) =>
  invoke<{
    sliders: Record<string, number>;
    overrides: Record<string, Record<string, number>>;
    leg: string;
  }>('param_list', { leg });

/** A saved look: a plain scene file in `<scene_dir>/looks/` + thumbnail. */
export type LookEntry = { name: string; thumb_b64: string | null };
export const looksList = () => invoke<LookEntry[]>('looks_list');
export const lookRead = (name: string) => invoke<string>('look_read', { name });
export const lookWrite = (name: string, scene: string, thumbB64: string | null) =>
  invoke<void>('look_write', { name, scene, thumbB64 });
export const lookDelete = (name: string) => invoke<void>('look_delete', { name });
/**
 * §5.6 promote — crossfade the projector to the design composite, then adopt
 * design's plan into the live slot. quantize 'bar' (default) starts the fade
 * on the next bar boundary; 'now' starts immediately.
 */
export const promote = (fadeMs: number, quantize: 'bar' | 'now') =>
  invoke<{ ok: boolean; state: string }>('promote', { fadeMs, quantize });
/** §5.6 pull — hard-copy live's scene back into design. */
export const pull = () => invoke<{ ok: boolean }>('pull');
/** §5.6 — which composite the native preview samples (LIVE ⇄ DESIGN). */
export const previewSetSource = (source: 'live' | 'design') =>
  invoke<{ ok: boolean; source: string }>('preview_set_source', { source });
/** §5.6 probe thresholds A < B (ms of predicted full-res p95). */
export const probeGetThresholds = () =>
  invoke<ProbeThresholds>('probe_get_thresholds');
export const probeSetThresholds = (aMs: number, bMs: number) =>
  invoke<ProbeThresholds>('probe_set_thresholds', { aMs, bMs });
/** §5.14 alignment layer — engine-wide, never per leg, never scene content. */
export const alignmentGet = () => invoke<AlignmentDoc>('alignment_get');
/**
 * Partial merge. Send `corners` alone during a corner drag and the engine
 * carries the extra handles with the content; send `points` to replace the
 * handle list. Throws with a prescriptive message on rejection — the previous
 * alignment keeps rendering.
 */
export const alignmentSet = (patch: AlignmentPatch) =>
  invoke<AlignmentDoc>('alignment_set', {
    enabled: patch.enabled ?? null,
    background: patch.background ?? null,
    corners: patch.corners ?? null,
    points: patch.points ?? null,
  });
export const alignmentReset = () => invoke<AlignmentDoc>('alignment_reset');
/** §3.6 test pattern, generated in source space so it warps with the content. */
export const alignmentSetTestPattern = (
  pattern: 'none' | 'grid' | 'border' | 'corners'
) => invoke<AlignmentDoc>('alignment_set_test_pattern', { pattern });

export const sceneLoad = (jsonText: string) =>
  invoke<unknown>('scene_load', { jsonText });
export const sceneReload = () => invoke<unknown>('scene_reload');
export const wgslValidate = (source: string) =>
  invoke<WgslValidateResult>('wgsl_validate', { source });
/** §5.6 — which leg a control write targets (the deck toggle's position). */
export type LegName = 'live' | 'design';

/**
 * Live knob path — no scene rebuild, engine picks it up next frame.
 * §5.6: per-leg; pass the deck toggle's leg (engine default: design).
 */
export const paramSet = (name: string, value: number, leg?: LegName) =>
  invoke<unknown>('param_set', { name, value, leg: leg ?? null });
/**
 * §5.5 live per-binding override — pins any scalar param (const or
 * driver-bound) without a rebuild; `null` clears it. Persisted in the
 * session sidecar, never written into scene.json. §5.6: per-leg.
 */
export const paramOverride = (
  binding: string,
  param: string,
  value: number | null,
  leg?: LegName
) => invoke<unknown>('param_override', { binding, param, value, leg: leg ?? null });
/**
 * §5.4 masters — brightness | speed | saturation | audioListen.
 * §5.6: per-leg; the deck toggle picks which leg your faders drive.
 */
export const masterSet = (name: string, value: number, leg?: LegName) =>
  invoke<unknown>('master_set', { name, value, leg: leg ?? null });
/** §5.5 effect input descriptors (ranges/steps/widgets). */
export const effectDescribe = (name?: string) =>
  invoke<unknown>('effect_describe', { name: name ?? null });
/** §5.3 explicit session sidecar save. */
export const sessionSave = () => invoke<{ ok: boolean; path: string }>('session_save');
export const effectUpsert = (name: string, wgsl: string, descriptor: unknown | null) =>
  invoke<unknown>('effect_upsert', { name, wgsl, descriptor });
export const effectRemove = (name: string) =>
  invoke<unknown>('effect_remove', { name });
export const lastPayload = <T>(channel: string) =>
  invoke<T | null>('last_payload', { channel });
export const listEffects = () => invoke<string[]>('list_effects');
export const readEffect = (name: string) =>
  invoke<{ name: string; wgsl: string; descriptor: string | null }>('read_effect', {
    name,
  });
export const readSceneFile = () => invoke<string>('read_scene_file');
export const writeSceneFile = (contents: string) =>
  invoke<void>('write_scene_file', { contents });
export const readMaskPng = (maskPath: string) =>
  invoke<string>('read_mask_png', { maskPath });
/**
 * What the native preview shows (engine `PreviewParams`). Omitted fields take
 * Perform's defaults: the deck-toggle leg, no warp, no photo, no dim.
 */
export type PreviewParams = {
  /** Which composite; omitted/null = follow the deck toggle. */
  source?: 'live' | 'design' | null;
  /** The exact projector final pass: promote mix, live masters, real warp, test pattern. */
  program?: boolean;
  /** Final multiply, 1 = off. */
  dim?: number;
  /** Surface photo added under the light (linear, additive), 0 = off. */
  photo_weight?: number;
  /** The composite itself, 0 = photo only. */
  light_weight?: number;
};
/**
 * Place the native (lossless, full-rate) preview over a measured layout slot
 * with what it should show, or hide it. CSS px, viewport-relative. Use
 * `<NativeSlot>` rather than calling this directly.
 */
export const previewSetSlot = (slot: {
  x: number;
  y: number;
  width: number;
  height: number;
  visible: boolean;
  params?: PreviewParams;
}) => invoke<void>('preview_set_slot', { slot: { id: 'main', ...slot } });
/** `underlay`: the preview sits under a transparent page, HTML draws over it.
 *  `window`: `WZRD_PREVIEW=window`, the old child window above the page. */
export const previewModeGet = () => invoke<'underlay' | 'window'>('preview_mode');

// ---------- projects (A1: re-root the in-process engine) ----------

/** One openable scene: a project folder's scene.json, or a recent scene file. */
export type ProjectEntry = {
  name: string;
  scene_path: string;
  dir: string | null;
  /** Last opened on this machine (settings.json `recent`), unix ms. */
  opened_ms: number | null;
  current: boolean;
  /** Under the projects root (vs. a recent scene elsewhere). */
  in_root: boolean;
};
/** The open project + whether its engine runs (and why not). */
export type ProjectInfo = {
  name: string;
  scene_path: string;
  dir: string;
  running: boolean;
  error: string | null;
  elapsed_ms?: number;
};
export const projectsList = () =>
  invoke<{ root: string; projects: ProjectEntry[] }>('projects_list');
export const projectCurrent = () => invoke<ProjectInfo>('project_current');
/** Tear the engine down and boot it on another scene (a folder or a file).
 *  On failure the shell re-opens the previous project and throws. */
export const projectOpen = (scenePath: string) =>
  invoke<ProjectInfo>('project_open', { scenePath });
/** Re-boot the engine on the current scene — crash recovery. */
export const engineRestart = () => invoke<ProjectInfo>('engine_restart');

// ---------- clips (A2: region × clip-slot grid) ----------

/** C2 provenance sidecar — readers ignore unknown keys. */
export type ClipSidecar = {
  schema?: number;
  tool?: string;
  args?: Record<string, any>;
  inputs?: string[];
  parent?: string | null;
  created?: string;
  est_usd?: number;
  region?: string | null;
  source_box?: [number, number, number, number] | null;
  rating?: 'favourite' | 'reject';
  [k: string]: unknown;
};
/** One C3 HAP clip; paths are relative to the scene dir (C5's `image.clip`). */
export type ClipEntry = {
  path: string;
  name: string;
  mtime_ms: number;
  sidecar: ClipSidecar | null;
  /** The source mp4's sidecar (via `sidecar.parent`): prompt, tool, model. */
  source: ClipSidecar | null;
  thumb: string | null;
};
/** Newest first, plus the operator's row filing (clipgrid.json). */
export const clipsList = () =>
  invoke<{ clips: ClipEntry[]; rows: Record<string, string | null> }>('clips_list');
export const clipThumb = (path: string) => invoke<string>('clip_thumb', { path });
/** File a clip onto a region row (`null` = unassigned). App state only. */
export const clipFile = (path: string, region: string | null) =>
  invoke<void>('clip_file', { path, region });

// ---------- operator-UI layout (per project, `<project>/ui_layout.json`) ----------

/** The project's saved panel layout (`ui/layoutStore.ts::LayoutDoc`), or
 *  null. Named by scene path so a late write can't land in another project. */
export const uiLayoutRead = (scenePath: string) => invoke<unknown>('ui_layout_read', { scenePath });
export const uiLayoutWrite = (scenePath: string, layout: unknown) =>
  invoke<void>('ui_layout_write', { scenePath, layout });

// ---------- events ----------

export type TelemetryFrame = {
  channel: string;
  payload: any;
};

/** `engine:telemetry` carries a batch (the shell flushes at most every
 *  ~33 ms — one webview eval per batch); `handler` sees one frame at a time,
 *  in order. Streaming channels arrive only while demanded
 *  (`api/telemetryDemand.ts`). */
export function onTelemetry(handler: (frame: TelemetryFrame) => void): Promise<UnlistenFn> {
  return listenEvent<TelemetryFrame[]>('engine:telemetry', (batch) => {
    for (const frame of batch) handler(frame);
  });
}

export function onEngineStatus(
  handler: (s: { running: boolean; ws_addr?: string }) => void
): Promise<UnlistenFn> {
  return listenEvent<{ running: boolean; ws_addr?: string }>('engine:status', handler);
}

/** Shell-only: fired after every project open / restart (Tauri seat). The
 *  browser seat sees the same switch as an engine-socket reconnect. */
export function onProject(handler: (p: ProjectInfo) => void): Promise<UnlistenFn> {
  return listenEvent<ProjectInfo>('engine:project', handler);
}
