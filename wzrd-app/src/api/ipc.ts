// Thin wrappers around Tauri's invoke() for every engine command. Keeping
// the surface small + typed means the rest of the front-end doesn't import
// '@tauri-apps/api/core' directly — handy if we ever swap transport.

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

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
export const sceneGetState = () =>
  invoke<{ json: string; leg: string; epoch: number; rev: number }>('scene_get_state');
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
 * Collapse Step 3 — position the native (lossless, full-rate) preview
 * window over a measured layout slot, or hide it. CSS px, viewport-relative.
 */
export const previewSetBounds = (
  x: number,
  y: number,
  width: number,
  height: number,
  visible: boolean
) => invoke<void>('preview_set_bounds', { x, y, width, height, visible });

// ---------- events ----------

export type TelemetryFrame = {
  channel: string;
  payload: any;
};

export function onTelemetry(handler: (frame: TelemetryFrame) => void): Promise<UnlistenFn> {
  return listen<TelemetryFrame>('engine:telemetry', (evt) => handler(evt.payload));
}

export function onEngineStatus(
  handler: (s: { running: boolean; ws_addr?: string }) => void
): Promise<UnlistenFn> {
  return listen<{ running: boolean; ws_addr?: string }>('engine:status', (evt) =>
    handler(evt.payload)
  );
}
