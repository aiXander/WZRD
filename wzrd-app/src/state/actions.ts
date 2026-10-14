// Every operator action in one place. Components call these; these call the
// engine. Three rules shape them:
//
//  1. Hands go where the deck points. Layer faders, knob pins and solo/mute
//     write the *control leg* (the deck toggle). Masters are the exception:
//     they are the operator's final stage, so they write BOTH legs and a
//     TAKE never makes them jump (design spec: "masters live only in live").
//  2. Structure is authored in DESIGN. Effect choice, region targeting,
//     colours and modulation routing are scene edits; scene edits always land
//     on the design leg (engine blanket rule) and reach the wall via TAKE.
//  3. Live writes are throttled, not debounced: a drag streams ~40 updates/s
//     to the engine. (The old 25 ms *debounce* never fired mid-drag — pointer
//     events arrive every ~16 ms — so faders only "landed" when you paused.)

import {
  lookDelete,
  lookRead,
  looksList,
  lookWrite,
  masterSet,
  paramList,
  paramOverride,
  previewSetSource,
  promote,
  pull,
  sceneGetState,
  transportSet,
  engineRestart,
  projectOpen,
  type LegName,
} from '../api/ipc';
import { commitSceneMutation, commitSceneNow, commitSceneText, settleCommits } from './sceneCommit';
import { canonical, isDriver, layerTitle, parseScene, uniqueId, type Binding } from './scene';
import { useStore, type Masters, type OverrideTable } from './store';

// ---------- throttle ----------

const lastSent: Record<string, number> = {};
const trailing: Record<string, number> = {};
/** Leading + trailing throttle per key: first event goes now, the latest
 *  value always lands, never more than one send per `ms`. */
function throttled(key: string, ms: number, fn: () => void) {
  const now = performance.now();
  const wait = (lastSent[key] ?? 0) + ms - now;
  if (trailing[key] != null) window.clearTimeout(trailing[key]);
  if (wait <= 0) {
    lastSent[key] = now;
    fn();
  } else {
    trailing[key] = window.setTimeout(() => {
      delete trailing[key];
      lastSent[key] = performance.now();
      fn();
    }, wait);
  }
}

const warn = (what: string) => (e: unknown) => console.warn(what, e);

/** A re-root drops throttled writes still aimed at the previous engine and
 *  the blackout/flash/freeze memories of the previous room. */
export function resetActionState() {
  for (const k of Object.keys(trailing)) {
    window.clearTimeout(trailing[k]);
    delete trailing[k];
  }
  preBlackout = preFlash = preFreeze = null;
  deletedLayers.length = 0;
  notifyUndo();
}

// ---------- projects (A1) ----------

/** Switch projects: settle in-flight scene edits into the old project, then
 *  re-root the engine. Hydrate follows from the engine coming back (App). */
export async function openProject(scenePath: string) {
  const st = useStore.getState();
  await settleCommits();
  try {
    const info = await projectOpen(scenePath);
    st.toast(`Opened ${info.name} · ${info.elapsed_ms ?? '?'} ms`, 'ok');
  } catch (e) {
    st.toast(String(e), 'bad');
  }
}

/** Re-boot the engine on the current project — the crash recovery button. */
export async function restartEngine() {
  const st = useStore.getState();
  await settleCommits();
  try {
    const info = await engineRestart();
    st.toast(`Engine restarted · ${info.elapsed_ms ?? '?'} ms`, 'ok');
  } catch (e) {
    st.toast(String(e), 'bad');
  }
}

export function controlLeg(): LegName {
  return useStore.getState().deck?.preview_source ?? 'design';
}

// ---------- deck ----------

export async function setDeck(leg: LegName) {
  const st = useStore.getState();
  if (st.deck?.preview_source === leg) return;
  try {
    await previewSetSource(leg);
    // Optimistic: the sticky `deck` echo confirms within a frame.
    if (st.deck) st.setDeck({ ...st.deck, preview_source: leg });
    hydrateOverrides(leg);
  } catch (e) {
    st.toast(String(e), 'bad');
  }
}

export type Quantize = 'bar' | 'now';
let quantize: Quantize = 'bar';
export const getQuantize = () => quantize;
export function setQuantize(q: Quantize) {
  quantize = q;
}

/** TAKE — crossfade the projector to design over the FADE master. */
export async function take(opts?: { cut?: boolean; now?: boolean }) {
  const st = useStore.getState();
  const fadeS = opts?.cut ? 0 : st.masters?.crossfade ?? 0.5;
  const q: Quantize = opts?.cut || opts?.now ? 'now' : quantize;
  try {
    await promote(Math.round(fadeS * 1000), q);
    const cued = st.cuedLook;
    if (cued) st.setLiveLook(cued);
  } catch (e) {
    st.toast(`TAKE refused — ${e}`, 'warn');
  }
}

/** PULL — copy what's live (scene + controls) into design. */
export async function pullLive() {
  const st = useStore.getState();
  try {
    await pull();
    st.setCuedLook(st.liveLook);
    await resyncDesign();
    st.toast('Design reset to what’s live', 'info');
  } catch (e) {
    st.toast(`PULL refused — ${e}`, 'warn');
  }
}

/** Re-read both legs' scenes + override tables (after promote/pull/boot). */
async function resyncDesign() {
  const st = useStore.getState();
  try {
    const [d, l] = await Promise.all([sceneGetState('design'), sceneGetState('live')]);
    if (d?.json) st.setSceneJson(d.json);
    if (l?.json) st.setLiveSceneJson(l.json);
  } catch (e) {
    console.warn('resync scenes', e);
  }
  hydrateOverrides('design');
  hydrateOverrides('live');
}

export async function refreshLiveScene() {
  try {
    const l = await sceneGetState('live');
    if (l?.json) useStore.getState().setLiveSceneJson(l.json);
  } catch (e) {
    console.warn('live scene', e);
  }
}

export function hydrateOverrides(leg: LegName) {
  paramList(leg)
    .then((r) => useStore.getState().setOverrides(leg, r?.overrides ?? {}))
    .catch(warn('param.list'));
}

/** Design holds something the crowd hasn't seen: a different scene, or
 *  different per-leg overrides (mixer `@mute`/`@opacity`/`@solo`, pinned
 *  knobs) — TAKE copies those too, so a design-only mute is a pending
 *  change. Tempo is ignored: TAP moves both legs live, only scene.json's
 *  copy trails on design. (Shared `ui.slider` knob values aren't mirrored
 *  in the store, so they don't count.) */
export function designDiffers(
  designJson: string,
  liveJson: string,
  designOverrides?: OverrideTable,
  liveOverrides?: OverrideTable
): boolean {
  if (!designJson || !liveJson) return false;
  const strip = (j: string) => {
    const s = parseScene(j);
    if (!s) return j;
    delete s.transport;
    return canonical(JSON.stringify(s));
  };
  if (strip(designJson) !== strip(liveJson)) return true;
  if (!designOverrides || !liveOverrides) return false;
  return overridesKey(designOverrides) !== overridesKey(liveOverrides);
}

/** An override table as a comparable string: no-op mixer entries (`@mute`
 *  0, `@solo` 0, `@opacity` 1) and empty rows dropped, keys sorted. */
function overridesKey(t: OverrideTable): string {
  const NEUTRAL: Record<string, number> = { '@mute': 0, '@solo': 0, '@opacity': 1 };
  const out: OverrideTable = {};
  for (const [binding, row] of Object.entries(t)) {
    const kept = Object.entries(row).filter(([p, v]) => NEUTRAL[p] !== v);
    if (kept.length) out[binding] = Object.fromEntries(kept);
  }
  return canonical(JSON.stringify(out));
}

// ---------- masters (both legs) ----------

export function setMaster(name: keyof Masters, value: number) {
  const st = useStore.getState();
  if (st.masters) {
    st.setMasters({
      ...st.masters,
      live: { ...st.masters.live, [name]: value },
      design: { ...st.masters.design, [name]: value },
    });
  }
  throttled(`master:${name}`, 25, () => {
    masterSet(name, value, 'live').catch(warn('master.set live'));
    masterSet(name, value, 'design').catch(warn('master.set design'));
  });
}

export function setCrossfade(seconds: number) {
  const st = useStore.getState();
  if (st.masters) st.setMasters({ ...st.masters, crossfade: seconds });
  throttled('master:crossfade', 25, () => {
    masterSet('crossfade', seconds).catch(warn('master.set crossfade'));
  });
}

/** BLACKOUT: brightness to 0 on the live leg, remembering where it was. */
let preBlackout: number | null = null;
export function toggleBlackout() {
  const m = useStore.getState().masters?.live;
  if (!m) return;
  if (m.brightness > 0.001) {
    preBlackout = m.brightness;
    setMaster('brightness', 0);
  } else {
    setMaster('brightness', preBlackout ?? 1);
    preBlackout = null;
  }
}

/** FLASH (momentary): master to full-blast while held. */
let preFlash: number | null = null;
export function flash(on: boolean) {
  const m = useStore.getState().masters?.live;
  if (!m) return;
  if (on && preFlash == null) {
    preFlash = m.brightness;
    setMaster('brightness', 2);
  } else if (!on && preFlash != null) {
    setMaster('brightness', preFlash);
    preFlash = null;
  }
}

/** FREEZE: stop time (speed master 0) — press again to thaw. */
let preFreeze: number | null = null;
export function toggleFreeze() {
  const m = useStore.getState().masters?.live;
  if (!m) return;
  if (m.speed > 0.0001) {
    preFreeze = m.speed;
    setMaster('speed', 0);
  } else {
    setMaster('speed', preFreeze ?? 1);
    preFreeze = null;
  }
}

// ---------- layer mixer + knob pins (control leg) ----------

export const MIX = { opacity: '@opacity', mute: '@mute', solo: '@solo' } as const;

export function pin(binding: string, param: string, value: number | null, leg = controlLeg()) {
  useStore.getState().patchOverride(leg, binding, param, value);
  throttled(`pin:${leg}:${binding}:${param}`, 25, () => {
    paramOverride(binding, param, value, leg).catch(warn('param.set override'));
  });
}

export function mixOf(binding: string, leg = controlLeg()) {
  const row = useStore.getState().overrides[leg][binding] ?? {};
  return {
    opacity: row[MIX.opacity] ?? 1,
    mute: (row[MIX.mute] ?? 0) > 0.5,
    solo: (row[MIX.solo] ?? 0) > 0.5,
  };
}

export function setOpacity(binding: string, v: number) {
  pin(binding, MIX.opacity, v >= 0.999 ? null : v);
}
export function toggleMute(binding: string) {
  pin(binding, MIX.mute, mixOf(binding).mute ? null : 1);
}
export function toggleSolo(binding: string, exclusive = true) {
  const leg = controlLeg();
  const on = !mixOf(binding, leg).solo;
  if (exclusive && on) {
    // Plain click = exclusive solo (Resolume); ⇧-click adds to the solo set.
    const table = useStore.getState().overrides[leg];
    for (const b of Object.keys(table)) {
      if (b !== binding && (table[b][MIX.solo] ?? 0) > 0.5) pin(b, MIX.solo, null, leg);
    }
  }
  pin(binding, MIX.solo, on ? 1 : null, leg);
}

// ---------- structure (design scene edits) ----------

function mutateBinding(id: string, fn: (b: Binding) => Binding | void) {
  commitSceneMutation((scene) => {
    const i = (scene.bindings ?? []).findIndex((b: Binding) => b.id === id);
    if (i >= 0) scene.bindings[i] = fn(scene.bindings[i]) ?? scene.bindings[i];
    return scene;
  });
}

export function setSceneParam(binding: string, param: string, value: any) {
  mutateBinding(binding, (b) => {
    b.params = { ...(b.params ?? {}), [param]: value };
  });
}

/** Clear both selections (layer + region) — Esc in Prepare, the
 *  inspector's ×, a click on the empty map. */
export function clearSelection() {
  const st = useStore.getState();
  st.setSelectedBindingId(null);
  st.setSelectedLayerId(null);
}

// ---------- layer delete + undo (design scene edits) ----------
//
// Deleting a layer is one click/key (Prepare row ×, inspector, Perform
// mixer on DESIGN, Delete/⌫), so it must be cheap to take back: each delete
// stacks the removed binding + its index, ⌘Z (or the Layers panel's Undo)
// puts the newest back. DESIGN-only like every structural edit — the wall
// keeps the layer until TAKE. The stack is per project (resetActionState).

const deletedLayers: { binding: Binding; index: number }[] = [];
const undoListeners = new Set<() => void>();
const notifyUndo = () => undoListeners.forEach((f) => f());

export function subscribeLayerUndo(fn: () => void): () => void {
  undoListeners.add(fn);
  return () => undoListeners.delete(fn);
}
/** Id of the layer ⌘Z would restore, or null. */
export const lastDeletedLayer = (): string | null => deletedLayers.at(-1)?.binding.id ?? null;

export function deleteLayer(id: string): boolean {
  const st = useStore.getState();
  const list = parseScene(st.sceneJson)?.bindings ?? [];
  const index = list.findIndex((b) => b.id === id);
  if (index < 0) {
    st.toast(`“${id}” isn’t in DESIGN — nothing to delete`, 'warn');
    return false;
  }
  const binding = structuredClone(list[index]);
  commitSceneMutation((s) => {
    s.bindings = (s.bindings ?? []).filter((b: Binding) => b.id !== id);
    return s;
  });
  deletedLayers.push({ binding, index });
  if (deletedLayers.length > 20) deletedLayers.shift();
  notifyUndo();
  if (st.selectedBindingId === id) st.setSelectedBindingId(null);
  if (st.hoveredBindingId === id) st.setHoveredBindingId(null);
  st.toast(`Deleted “${layerTitle(binding)}” from DESIGN — ⌘Z restores it`, 'info');
  return true;
}

/** Delete from wherever the hands are (Perform's mixer, the ⌫ key). On LIVE
 *  this is the "edit in DESIGN" move: hands go to DESIGN, nothing is
 *  deleted yet — the row the operator meant is right there to delete. */
export function deleteLayerOnHands(id: string): boolean {
  if (controlLeg() === 'live') {
    void setDeck('design');
    useStore.getState().toast('Layers are deleted on DESIGN — hands moved there. Delete again, then TAKE.', 'info');
    return false;
  }
  return deleteLayer(id);
}

export function undoDeleteLayer(): boolean {
  const d = deletedLayers.pop();
  if (!d) return false;
  notifyUndo();
  const st = useStore.getState();
  const taken = new Set((parseScene(st.sceneJson)?.bindings ?? []).map((b) => b.id));
  const id = taken.has(d.binding.id) ? uniqueId(d.binding.id, taken) : d.binding.id;
  commitSceneMutation((s) => {
    const list = s.bindings ?? [];
    list.splice(Math.min(d.index, list.length), 0, { ...d.binding, id });
    s.bindings = list;
    return s;
  });
  st.setSelectedBindingId(id);
  st.toast(`Restored “${layerTitle({ ...d.binding, id })}”`, 'ok');
  return true;
}

/** Modulation sources — the words a VJ thinks in. Audio sources are fixed
 *  driver specs; the two clock sources are a *kind* whose rate and shape are
 *  dialled separately (`setModRate` / `setModShape`): Tempo follows the BPM
 *  grid (clock.beats below a bar, clock.bars from a bar up), Free runs in
 *  seconds (clock.phase). */
export const MOD_SOURCES: {
  id: string;
  label: string;
  group: 'audio' | 'clock';
  title: string;
  spec: Record<string, any>;
}[] = [
  { id: 'kick', label: 'Kick', group: 'audio', title: 'Flash on each low hit, then decay', spec: { driver: 'audio.onset', band: 'low', decay: 0.18 } },
  { id: 'snare', label: 'Snare', group: 'audio', title: 'Flash on each mid hit, then decay', spec: { driver: 'audio.onset', band: 'mid', decay: 0.15 } },
  { id: 'hats', label: 'Hats', group: 'audio', title: 'Flash on each high hit, then decay', spec: { driver: 'audio.onset', band: 'high', decay: 0.08 } },
  { id: 'low', label: 'Bass', group: 'audio', title: 'Follow the low-band level', spec: { driver: 'audio.band', band: 'low' } },
  { id: 'mid', label: 'Mids', group: 'audio', title: 'Follow the mid-band level', spec: { driver: 'audio.band', band: 'mid' } },
  { id: 'high', label: 'Highs', group: 'audio', title: 'Follow the high-band level', spec: { driver: 'audio.band', band: 'high' } },
  { id: 'tempo', label: 'Tempo', group: 'clock', title: 'A shape locked to the beat grid (beats / bars)', spec: { driver: 'clock.bars', n: 4 } },
  { id: 'free', label: 'Free', group: 'clock', title: 'A shape in seconds, independent of the BPM', spec: { driver: 'clock.phase', rate: 0.125 } },
];

/** Engine `shape` names (render-engine.md §2.1). Clock drivers only. */
export type ModShape = 'saw' | 'ramp_down' | 'pingpong' | 'sine' | 'square' | 'random';
export const MOD_SHAPES: { id: ModShape; label: string; title: string }[] = [
  { id: 'pingpong', label: 'Ping-pong', title: 'Up and back down at constant speed — no jump' },
  { id: 'sine', label: 'Sine', title: 'Up and back down, eased at both ends — the smoothest' },
  { id: 'saw', label: 'Ramp up', title: 'Rest → peak, then jumps back to rest' },
  { id: 'ramp_down', label: 'Ramp down', title: 'Peak → rest, then jumps back to peak' },
  { id: 'square', label: 'Square', title: 'Peak for the first half of each cycle, rest for the second' },
  { id: 'random', label: 'Random', title: 'Glides to a new random spot every cycle. Click again for a different sequence' },
];
/** What new clock modulations get: the operator wants smooth textures. */
const DEFAULT_MOD_SHAPE: ModShape = 'pingpong';

/** Rate choices: Tempo in bars (¼ bar = 1 beat), Free in seconds. */
export const TEMPO_RATES = [0.25, 0.5, 1, 2, 4, 8, 16];
export const FREE_RATES = [1, 2, 4, 8, 16, 32, 64];

const isClock = (v: any) => isDriver(v) && /^clock\.(bars|beats|phase)$/.test(v.driver);

export function modSourceOf(v: any): (typeof MOD_SOURCES)[number] | null {
  if (!isDriver(v)) return null;
  const id =
    v.driver === 'clock.bars' || v.driver === 'clock.beats'
      ? 'tempo'
      : v.driver === 'clock.phase'
      ? 'free'
      : MOD_SOURCES.find((m) => m.spec.driver === v.driver && m.spec.band === v.band)?.id;
  return MOD_SOURCES.find((m) => m.id === id) ?? null;
}

/** A clock driver's cycle length: bars for Tempo, seconds for Free. */
export function modPeriod(v: any): number | null {
  if (!isDriver(v)) return null;
  if (v.driver === 'clock.bars') return Number(v.n ?? 1);
  if (v.driver === 'clock.beats') return Number(v.n ?? 1) / 4;
  if (v.driver === 'clock.phase') return 1 / Math.max(1e-6, Number(v.rate ?? 0.1));
  return null;
}

export function modShapeOf(v: any): ModShape | null {
  return isClock(v) ? ((v.shape as ModShape) ?? 'saw') : null;
}

function clockKeys(kind: 'tempo' | 'free', period: number): Record<string, any> {
  if (kind === 'free') return { driver: 'clock.phase', rate: 1 / period };
  return period < 1 ? { driver: 'clock.beats', n: period * 4 } : { driver: 'clock.bars', n: period };
}

/** Rewrite one modulated param's driver object (no-op if it isn't one). */
function patchMod(binding: string, param: string, fn: (prev: Record<string, any>) => Record<string, any>) {
  mutateBinding(binding, (b) => {
    const prev = b.params?.[param];
    if (!isDriver(prev)) return;
    b.params = { ...(b.params ?? {}), [param]: fn(prev) };
  });
}

/** Route a modulation source onto a param. A fresh routing sweeps from the
 *  current value toward the far end of the param's range; re-routing keeps
 *  the range, and clock → clock keeps the shape. */
export function modulate(
  binding: string,
  param: string,
  sourceId: string | null,
  current: number,
  lo: number,
  hi: number
) {
  // A pin would mask the modulation — drop it on the design leg.
  pin(binding, param, null, 'design');
  mutateBinding(binding, (b) => {
    const prev = b.params?.[param];
    let next: any;
    if (!sourceId) {
      next = isDriver(prev) && Array.isArray(prev.range) ? prev.range[0] : current;
    } else {
      const src = MOD_SOURCES.find((m) => m.id === sourceId)!;
      const span = hi - lo || 1;
      let to = current + (hi - current) * 0.75;
      // Already near the top: sweep downward instead of a dead range.
      if (Math.abs(to - current) < span * 0.05) to = current - (current - lo) * 0.75;
      const range = isDriver(prev) && Array.isArray(prev.range) ? prev.range : [current, to];
      let spec: Record<string, any> = { ...src.spec };
      if (src.group === 'clock') {
        const same = modSourceOf(prev)?.id === src.id;
        const period = same ? modPeriod(prev)! : src.id === 'tempo' ? 4 : 8;
        const shape = modShapeOf(prev) ?? DEFAULT_MOD_SHAPE;
        spec = { ...clockKeys(src.id as 'tempo' | 'free', period), shape };
        if (shape === 'random') spec.seed = isDriver(prev) && prev.seed != null ? prev.seed : newSeed();
      }
      next = { ...spec, range: range.map((x: number) => Math.min(hi, Math.max(lo, x))) };
    }
    b.params = { ...(b.params ?? {}), [param]: next };
  });
}

export function setModRange(binding: string, param: string, from: number, to: number) {
  patchMod(binding, param, (prev) => ({ ...prev, range: [from, to] }));
}

const newSeed = () => Math.floor(Math.random() * 10000);

/** Clock modulations only. Picking `random` again re-rolls its sequence. */
export function setModShape(binding: string, param: string, shape: ModShape) {
  patchMod(binding, param, (prev) => {
    if (!isClock(prev)) return prev;
    const { seed: _seed, ...rest } = prev;
    return shape === 'random' ? { ...rest, shape, seed: newSeed() } : { ...rest, shape };
  });
}

/** Cycle length for a clock modulation — bars (Tempo) or seconds (Free). */
export function setModRate(binding: string, param: string, period: number) {
  patchMod(binding, param, (prev) => {
    const kind = modSourceOf(prev)?.id;
    if (kind !== 'tempo' && kind !== 'free') return prev;
    const { driver: _d, n: _n, rate: _r, ...rest } = prev;
    return { ...clockKeys(kind, period), ...rest };
  });
}

/** How long each onset hit glows, in seconds. */
export function setModDecay(binding: string, param: string, decay: number) {
  patchMod(binding, param, (prev) => (prev.driver === 'audio.onset' ? { ...prev, decay } : prev));
}

/** Write the design leg's knob pins into scene.json params and clear them —
 *  "what I dialled in" becomes the scene (looks/saves carry it). */
export function bakeDesignPins(): string | null {
  const st = useStore.getState();
  const scene = parseScene(st.sceneJson);
  if (!scene) return null;
  const table = st.overrides.design;
  for (const b of scene.bindings ?? []) {
    const pins = table[b.id];
    if (!pins) continue;
    for (const [param, v] of Object.entries(pins)) {
      if (param.startsWith('@')) continue;
      if (!(param in (b.params ?? {}))) continue;
      // A pin on a modulated param froze it — baking keeps it frozen.
      b.params![param] = v;
      pin(b.id, param, null, 'design');
    }
  }
  const text = JSON.stringify(scene, null, 2);
  commitSceneText(text);
  return text;
}

// ---------- tempo ----------

const taps: number[] = [];
export function tapTempo() {
  const now = performance.now();
  if (taps.length && now - taps[taps.length - 1] > 2000) taps.length = 0;
  taps.push(now);
  if (taps.length > 8) taps.shift();
  if (taps.length < 3) return;
  const iv = (taps[taps.length - 1] - taps[0]) / (taps.length - 1);
  setBpm(Math.round((60000 / iv) * 10) / 10);
}

export function setBpm(bpm: number) {
  const b = Math.min(400, Math.max(20, bpm));
  const st = useStore.getState();
  if (st.transport) {
    st.setTransport({
      live: { ...st.transport.live, bpm: b },
      design: { ...st.transport.design, bpm: b },
    });
  }
  throttled('bpm', 60, () => {
    transportSet({ bpm: b }).catch(warn('transport.set'));
  });
  // scene.json is what a cold boot reads — persist the tempo there too
  // (design leg; live already has it from transport.set).
  throttled('bpm-scene', 600, () => {
    commitSceneMutation((scene) => {
      scene.transport = { ...(scene.transport ?? {}), bpm: b };
      return scene;
    });
  });
}

export function resyncBeat() {
  transportSet({ resync: true }).catch(warn('transport.set resync'));
}

// ---------- looks (the clip bank) ----------

export async function loadLooks() {
  try {
    useStore.getState().setLooks(await looksList());
  } catch (e) {
    console.warn('looks_list', e);
  }
}

export async function saveLook(name: string) {
  const st = useStore.getState();
  const text = bakeDesignPins() ?? st.sceneJson;
  const thumb = st.preview?.data_b64 ?? null;
  try {
    await lookWrite(name, text, thumb);
    st.setCuedLook(name);
    await loadLooks();
    st.toast(`Saved look “${name}”`, 'ok');
  } catch (e) {
    st.toast(`Save failed — ${e}`, 'bad');
  }
}

/** Cue a look into DESIGN (the preview); `go` also TAKEs it live. */
export async function recallLook(name: string, go = false) {
  const st = useStore.getState();
  try {
    const text = await lookRead(name);
    // Pins on design would mask the look's own values.
    for (const [b, row] of Object.entries(st.overrides.design)) {
      for (const p of Object.keys(row)) if (!p.startsWith('@')) pin(b, p, null, 'design');
    }
    // Await the design apply (incl. its probe) so a GO fades to the look,
    // not to whatever design held a moment ago.
    await commitSceneNow(text);
    st.setCuedLook(name);
    if (go) await take();
  } catch (e) {
    st.toast(`Couldn’t load “${name}” — ${e}`, 'bad');
  }
}

export async function deleteLook(name: string) {
  try {
    await lookDelete(name);
    await loadLooks();
  } catch (e) {
    useStore.getState().toast(String(e), 'bad');
  }
}
