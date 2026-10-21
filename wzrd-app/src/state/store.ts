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
import type {
  AlignmentDoc,
  ChangeEntry,
  DeckPayload,
  LegName,
  LookEntry,
  PackInfo,
  PerfPayload,
  PerfSpike,
  ProbeReport,
  ProjectInfo,
} from '../api/ipc';

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
  /** §5.6 — pre-flight probe verdict, when one ran for this reload. */
  probe?: ProbeReport | null;
  /** §5.10 correlation stamp: boot epoch, design rev, causing actor. */
  epoch?: number;
  rev?: number;
  actor?: string;
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
  /** §5.5 — a live param.set {binding,param} override is pinning this param. */
  overridden?: boolean;
};
export type Masters = {
  brightness: number;
  speed: number;
  saturation: number;
  audioListen: number;
};
/** §5.6 — the `masters` channel now carries both legs' values. */
export type MastersState = {
  live: Masters;
  design: Masters;
  /** §5.4 crossfade-time master (seconds) — engine-wide, not per leg. */
  crossfade: number;
};
/** `transport` channel — both legs' tempo + musical position (beats). */
export type TransportPayload = {
  live: { bpm: number; beats: number };
  design: { bpm: number; beats: number };
};
/** binding → param → value: a leg's §5.5 override table, incl. the reserved
 *  layer-mixer keys `@opacity` / `@mute` / `@solo`. */
export type OverrideTable = Record<string, Record<string, number>>;
/** `effect.describe` input descriptor. */
export type EffectInput = {
  name: string;
  type: 'float' | 'color' | string;
  default?: any;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  unit?: string | null;
  widget?: string | null;
};
export type Catalog = Record<string, { kind: string; inputs: EffectInput[] }>;
export type Toast = {
  id: number;
  text: string;
  tone: 'info' | 'warn' | 'bad' | 'ok';
  /** A button in the toast (e.g. Undo after a delete); clicking it runs + drops the toast. */
  action?: { label: string; run: () => void };
};
export type Route = 'studio' | 'prepare' | 'align' | 'perform' | 'debug';

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
  /** Why the engine is down (boot error / last panic), if it is. */
  engineError: string | null;
  setEngineError: (e: string | null) => void;

  // A1 — the open project. Switching re-roots the engine; `resetProject`
  // drops everything that belonged to the previous engine before hydrate.
  project: ProjectInfo | null;
  setProject: (p: ProjectInfo | null) => void;
  resetProject: () => void;

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

  // route. Studio (make + label content), Prepare and Perform are the tab bar; Align (done once at load-in)
  // and Debug live one click down in the TopBar's ⚙ menu.
  route: Route;
  setRoute: (r: Route) => void;

  // Live leg's scene text — compared against the design scene to light TAKE
  // when design holds something the crowd hasn't seen.
  liveSceneJson: string;
  setLiveSceneJson: (s: string) => void;

  transport: TransportPayload | null;
  setTransport: (t: TransportPayload) => void;

  // Per-leg override tables — an optimistic mirror of `param.list` (knob
  // pins + the layer mixer). Writes land here first, the engine follows.
  overrides: Record<LegName, OverrideTable>;
  setOverrides: (leg: LegName, t: OverrideTable) => void;
  patchOverride: (leg: LegName, binding: string, param: string, value: number | null) => void;

  catalog: Catalog;
  setCatalog: (c: Catalog) => void;

  looks: LookEntry[];
  setLooks: (l: LookEntry[]) => void;
  /** Look last recalled into design (cued) / taken live. */
  cuedLook: string | null;
  liveLook: string | null;
  setCuedLook: (n: string | null) => void;
  setLiveLook: (n: string | null) => void;

  toasts: Toast[];
  toast: (text: string, tone?: Toast['tone'], opts?: { action?: Toast['action']; ms?: number }) => void;
  dropToast: (id: number) => void;

  showHelp: boolean;
  setShowHelp: (v: boolean) => void;

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

  // Perf-monitor PM1 — `perf` (2 Hz aggregate) + sticky `perf_spike`.
  perf: PerfPayload | null;
  setPerf: (v: PerfPayload) => void;
  perfSpike: PerfSpike | null;
  setPerfSpike: (v: PerfSpike | null) => void;

  drivers: DriverRow[];
  setDrivers: (v: DriverRow[]) => void;

  connectivity: Connectivity | null;
  setConnectivity: (v: Connectivity) => void;

  // §5.4/§5.6 masters — sticky, both legs; engine emits on startup, every
  // master.set, and promote/pull control-state copies.
  masters: MastersState | null;
  setMasters: (v: MastersState) => void;

  // §5.6 two-deck state — sticky `deck` channel (promote phase, mix,
  // preview source). Null until the engine's first emission arrives.
  deck: DeckPayload | null;
  setDeck: (v: DeckPayload) => void;

  // §5.14 alignment document — sticky `alignment` channel. The engine is the
  // only writer of alignment.json; this is a mirror plus the optimistic local
  // edits an in-flight drag has already applied (see state/alignment.ts).
  alignment: AlignmentDoc | null;
  setAlignment: (v: AlignmentDoc | null) => void;
  /** Last rejected `alignment.set`, shown inline on the Align tab. */
  alignmentError: string | null;
  setAlignmentError: (e: string | null) => void;
  /** Which handle the canvas has selected — `null` | 'c0'..'c3' | a point id. */
  selectedHandle: string | null;
  setSelectedHandle: (h: string | null) => void;

  preview: { width: number; height: number; data_b64: string } | null;
  setPreview: (v: Store['preview']) => void;

  log: LogLine[];
  appendLog: (line: LogLine) => void;

  hotReloadHistory: HotReload[];
  appendHotReload: (h: HotReload) => void;

  // selected binding (Phase 4.2 inspector)
  selectedBindingId: string | null;
  setSelectedBindingId: (id: string | null) => void;

  // binding the pointer is currently over in the inspector. Transient, and
  // the mirror of the canvas's own hover: the surface highlights whichever
  // regions this binding's selector resolves to, so "which region am I
  // editing?" is answered by looking, not by reading an id.
  hoveredBindingId: string | null;
  setHoveredBindingId: (id: string | null) => void;

  // selected *layer* (surface canvas region) — distinct from bindings: a
  // layer is a mask region in the pack; a binding is a scene entry whose
  // selector may resolve to several layers.
  selectedLayerId: string | null;
  setSelectedLayerId: (id: string | null) => void;

  // §5.10 reverse-sync: the last agent-authored change re-pulled into this
  // store. Non-null means the store scene came from the agent and is not
  // yet adopted into scene.json (the "Adopt" button in the top bar).
  agentEdit: ChangeEntry | null;
  setAgentEdit: (e: ChangeEntry | null) => void;
}

const LOG_CAP = 2000;
let toastSeq = 0;

function initialRoute(): Route {
  try {
    const r = localStorage.getItem('wzrd.route');
    // Never reopen on Align: it's a load-in step, and its arrow keys nudge
    // the projector warp — not where a launch mid-gig should land.
    if (r === 'studio' || r === 'prepare' || r === 'perform' || r === 'debug') return r;
  } catch {
    /* ignore */
  }
  return 'perform';
}
const HOT_CAP = 200;

/** Project-scoped state, as it is before the first hydrate. Telemetry pills
 *  (fps, audio, connectivity), route, log and toasts survive a re-root. */
const PROJECT_INITIAL = {
  pack: null,
  sceneJson: '',
  sceneDirty: false,
  activeTab: { kind: 'scene' } as const,
  effects: [],
  liveSceneJson: '',
  transport: null,
  overrides: { live: {}, design: {} },
  catalog: {},
  looks: [],
  cuedLook: null,
  liveLook: null,
  hotReload: null,
  hotReloadHistory: [],
  drivers: [],
  masters: null,
  deck: null,
  perf: null,
  perfSpike: null,
  alignment: null,
  alignmentError: null,
  selectedHandle: null,
  preview: null,
  selectedBindingId: null,
  hoveredBindingId: null,
  selectedLayerId: null,
  agentEdit: null,
} satisfies Partial<Store>;

export const useStore = create<Store>((set) => ({
  engineRunning: false,
  setEngineRunning: (v) => set({ engineRunning: v }),
  engineError: null,
  setEngineError: (e) => set({ engineError: e }),

  project: null,
  setProject: (p) => set({ project: p }),
  resetProject: () => set({ ...PROJECT_INITIAL }),

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

  route: initialRoute(),
  setRoute: (r) => {
    try {
      localStorage.setItem('wzrd.route', r);
    } catch {
      /* private mode — fine */
    }
    set({ route: r });
  },

  liveSceneJson: '',
  setLiveSceneJson: (s) => set({ liveSceneJson: s }),

  transport: null,
  setTransport: (t) => set({ transport: t }),

  overrides: { live: {}, design: {} },
  setOverrides: (leg, t) => set((s) => ({ overrides: { ...s.overrides, [leg]: t } })),
  patchOverride: (leg, binding, param, value) =>
    set((s) => {
      const table = { ...s.overrides[leg] };
      const row = { ...(table[binding] ?? {}) };
      if (value == null) delete row[param];
      else row[param] = value;
      if (Object.keys(row).length) table[binding] = row;
      else delete table[binding];
      return { overrides: { ...s.overrides, [leg]: table } };
    }),

  catalog: {},
  setCatalog: (c) => set({ catalog: c }),

  looks: [],
  setLooks: (l) => set({ looks: l }),
  cuedLook: null,
  liveLook: null,
  setCuedLook: (n) => set({ cuedLook: n }),
  setLiveLook: (n) => set({ liveLook: n }),

  toasts: [],
  toast: (text, tone = 'info', opts) => {
    const id = ++toastSeq;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { id, text, tone, action: opts?.action }] }));
    const ms = opts?.ms ?? (opts?.action ? 10000 : tone === 'bad' ? 6000 : 3200);
    window.setTimeout(() => useStore.getState().dropToast(id), ms);
  },
  dropToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  showHelp: false,
  setShowHelp: (v) => set({ showHelp: v }),

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

  perf: null,
  setPerf: (v) => set({ perf: v }),
  perfSpike: null,
  setPerfSpike: (v) => set({ perfSpike: v }),

  drivers: [],
  setDrivers: (v) => set({ drivers: v }),

  connectivity: null,
  setConnectivity: (v) => set({ connectivity: v }),

  masters: null,
  setMasters: (v) => set({ masters: v }),

  deck: null,
  setDeck: (v) => set({ deck: v }),

  alignment: null,
  setAlignment: (v) => set({ alignment: v }),
  alignmentError: null,
  setAlignmentError: (e) => set({ alignmentError: e }),
  selectedHandle: null,
  setSelectedHandle: (h) => set({ selectedHandle: h }),

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

  hoveredBindingId: null,
  setHoveredBindingId: (id) => set({ hoveredBindingId: id }),

  selectedLayerId: null,
  setSelectedLayerId: (id) => set({ selectedLayerId: id }),

  agentEdit: null,
  setAgentEdit: (e) => set({ agentEdit: e }),
}));
