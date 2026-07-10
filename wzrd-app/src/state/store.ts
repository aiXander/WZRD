// Single global store for engine-derived live state. Telemetry events feed
// in via `useEngineWire` (in App.tsx); routes read what they need with
// `useStore(s => s.fps)` etc.
//
// Sticky channels (hot_reload, audio_freshness, fps, connectivity) get
// their last value retained so a route mounting late sees the right pill
// color immediately. Noisy channels (preview, audio, drivers, log,
// frame_stats) are best-effort — we keep the latest value but don't fan
// history.

import { create } from 'zustand';
import type { PackInfo } from '../api/ipc';

export type FpsPayload = {
  fps: number;
  frame_time_ms: number;
  /** false ⇒ projector window occluded; engine self-paces offscreen (~30 Hz) by design. */
  presenting?: boolean;
};
export type AudioFreshness = {
  state: 'fresh' | 'stale' | 'down';
  last_packet_ms: number;
};
export type HotReload = {
  target: string;
  ok: boolean;
  elapsed_ms: number;
  message: string | null;
};
export type AudioPayload = {
  band_low: number;
  band_mid: number;
  band_high: number;
  onset_low: number;
  onset_mid: number;
  onset_high: number;
};
export type FrameStats = {
  fps: number;
  frame_time_ms_p50: number;
  frame_time_ms_p95: number;
  frame_time_ms_p99: number;
  mask_slice_count: number;
  pipeline_count: number;
  pass_count: number;
  /** false ⇒ occluded, intentional offscreen self-pacing. */
  presenting?: boolean;
};
export type DriverRow = {
  binding_id: string;
  param_name: string;
  source: string;
  value: number;
  affects: number;
};
export type LogLine = {
  level: string;
  target: string;
  message: string;
  ts_ms: number;
};
export type Connectivity = {
  osc: { status: string; detail: string | null };
  file_watcher: { status: string; detail: string | null };
  ws: { status: string; detail: string | null };
};

interface Store {
  // engine lifecycle
  engineRunning: boolean;
  setEngineRunning: (v: boolean) => void;

  // pack
  pack: PackInfo | null;
  setPack: (p: PackInfo | null) => void;

  // scene text (Monaco source of truth)
  sceneJson: string;
  setSceneJson: (s: string) => void;
  sceneDirty: boolean;
  setSceneDirty: (d: boolean) => void;

  // active monaco tab
  activeTab: { kind: 'scene' } | { kind: 'effect'; name: string };
  setActiveTab: (t: Store['activeTab']) => void;

  // effects file list
  effects: string[];
  setEffects: (e: string[]) => void;

  // route
  route: 'prepare' | 'perform' | 'debug';
  setRoute: (r: Store['route']) => void;

  // telemetry — sticky/best-effort latest
  fps: FpsPayload | null;
  setFps: (v: FpsPayload) => void;

  audioFresh: AudioFreshness | null;
  setAudioFresh: (v: AudioFreshness) => void;

  hotReload: HotReload | null;
  setHotReload: (v: HotReload) => void;

  audio: AudioPayload | null;
  setAudio: (v: AudioPayload) => void;

  frameStats: FrameStats | null;
  setFrameStats: (v: FrameStats) => void;

  drivers: DriverRow[];
  setDrivers: (v: DriverRow[]) => void;

  connectivity: Connectivity | null;
  setConnectivity: (v: Connectivity) => void;

  preview: { width: number; height: number; data_b64: string } | null;
  setPreview: (v: Store['preview']) => void;

  log: LogLine[];
  appendLog: (line: LogLine) => void;

  hotReloadHistory: HotReload[];
  appendHotReload: (h: HotReload) => void;

  // selected binding (Phase 4.2 inspector)
  selectedBindingId: string | null;
  setSelectedBindingId: (id: string | null) => void;

  // selected *layer* (surface canvas region) — distinct from bindings: a
  // layer is a mask region in the pack; a binding is a scene entry whose
  // selector may resolve to several layers.
  selectedLayerId: string | null;
  setSelectedLayerId: (id: string | null) => void;
}

const LOG_CAP = 2000;
const HOT_CAP = 200;

export const useStore = create<Store>((set) => ({
  engineRunning: false,
  setEngineRunning: (v) => set({ engineRunning: v }),

  pack: null,
  setPack: (p) => set({ pack: p }),

  sceneJson: '',
  setSceneJson: (s) => set({ sceneJson: s }),
  sceneDirty: false,
  setSceneDirty: (d) => set({ sceneDirty: d }),

  activeTab: { kind: 'scene' },
  setActiveTab: (t) => set({ activeTab: t }),

  effects: [],
  setEffects: (e) => set({ effects: e }),

  route: 'prepare',
  setRoute: (r) => set({ route: r }),

  fps: null,
  setFps: (v) => set({ fps: v }),

  audioFresh: null,
  setAudioFresh: (v) => set({ audioFresh: v }),

  hotReload: null,
  setHotReload: (v) => set({ hotReload: v }),

  audio: null,
  setAudio: (v) => set({ audio: v }),

  frameStats: null,
  setFrameStats: (v) => set({ frameStats: v }),

  drivers: [],
  setDrivers: (v) => set({ drivers: v }),

  connectivity: null,
  setConnectivity: (v) => set({ connectivity: v }),

  preview: null,
  setPreview: (v) => set({ preview: v }),

  log: [],
  appendLog: (line) =>
    set((s) => {
      const next = [...s.log, line];
      if (next.length > LOG_CAP) next.splice(0, next.length - LOG_CAP);
      return { log: next };
    }),

  hotReloadHistory: [],
  appendHotReload: (h) =>
    set((s) => {
      const next = [h, ...s.hotReloadHistory];
      if (next.length > HOT_CAP) next.length = HOT_CAP;
      return { hotReloadHistory: next };
    }),

  selectedBindingId: null,
  setSelectedBindingId: (id) => set({ selectedBindingId: id }),

  selectedLayerId: null,
  setSelectedLayerId: (id) => set({ selectedLayerId: id }),
}));
