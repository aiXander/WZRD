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

// ---------- commands ----------

export const engineStatus = () => invoke<EngineStatus>('engine_status');
export const packInfo = () => invoke<PackInfo>('pack_info');
export const sceneGetState = () => invoke<{ json: string }>('scene_get_state');
export const sceneLoad = (jsonText: string) =>
  invoke<unknown>('scene_load', { jsonText });
export const sceneReload = () => invoke<unknown>('scene_reload');
export const wgslValidate = (source: string) =>
  invoke<WgslValidateResult>('wgsl_validate', { source });
/** Live knob path — no scene rebuild, engine picks it up next frame. */
export const paramSet = (name: string, value: number) =>
  invoke<unknown>('param_set', { name, value });
/**
 * §5.5 live per-binding override — pins any scalar param (const or
 * driver-bound) without a rebuild; `null` clears it. Persisted in the
 * session sidecar, never written into scene.json.
 */
export const paramOverride = (binding: string, param: string, value: number | null) =>
  invoke<unknown>('param_override', { binding, param, value });
/** §5.4 masters — brightness | speed | saturation | audioListen. */
export const masterSet = (name: string, value: number) =>
  invoke<unknown>('master_set', { name, value });
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
