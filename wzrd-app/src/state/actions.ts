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
  type LegName,
} from '../api/ipc';
import { commitSceneMutation, commitSceneNow, commitSceneText } from './sceneCommit';
import { canonical, isDriver, parseScene, type Binding } from './scene';
import { useStore, type Masters } from './store';

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

/** Design holds something the crowd hasn't seen. Tempo is ignored: TAP
 *  moves both legs live, only scene.json's copy trails on design. */
export function designDiffers(designJson: string, liveJson: string): boolean {
  if (!designJson || !liveJson) return false;
  const strip = (j: string) => {
    const s = parseScene(j);
    if (!s) return j;
    delete s.transport;
    return canonical(JSON.stringify(s));
  };
  return strip(designJson) !== strip(liveJson);
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

/** Modulation presets — the sources a VJ thinks in. */
export const MOD_SOURCES: {
  id: string;
  label: string;
  group: 'audio' | 'clock';
  spec: Record<string, any>;
}[] = [
  { id: 'kick', label: 'Kick', group: 'audio', spec: { driver: 'audio.onset', band: 'low', decay: 0.18 } },
  { id: 'snare', label: 'Snare', group: 'audio', spec: { driver: 'audio.onset', band: 'mid', decay: 0.15 } },
  { id: 'hats', label: 'Hats', group: 'audio', spec: { driver: 'audio.onset', band: 'high', decay: 0.08 } },
  { id: 'low', label: 'Bass', group: 'audio', spec: { driver: 'audio.band', band: 'low' } },
  { id: 'mid', label: 'Mids', group: 'audio', spec: { driver: 'audio.band', band: 'mid' } },
  { id: 'high', label: 'Highs', group: 'audio', spec: { driver: 'audio.band', band: 'high' } },
  { id: 'beat', label: 'Beat', group: 'clock', spec: { driver: 'clock.beats', n: 1 } },
  { id: 'bar', label: 'Bar', group: 'clock', spec: { driver: 'clock.bars', n: 1 } },
  { id: 'bar4', label: '4 Bars', group: 'clock', spec: { driver: 'clock.bars', n: 4 } },
  { id: 'lfo', label: 'Slow', group: 'clock', spec: { driver: 'clock.phase', rate: 0.125 } },
];

export function modSourceOf(v: any): (typeof MOD_SOURCES)[number] | null {
  if (!isDriver(v)) return null;
  return (
    MOD_SOURCES.find((m) =>
      Object.entries(m.spec).every(([k, val]) => k === 'decay' || k === 'rate' || v[k] === val)
    ) ?? null
  );
}

/** Route a modulation source onto a param (range defaults to value → max). */
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
      const range =
        isDriver(prev) && Array.isArray(prev.range)
          ? prev.range
          : [current, current + (hi - current) * 0.75 || hi];
      next = { ...src.spec, range: range.map((x: number) => Math.min(hi, Math.max(lo, x))) };
    }
    b.params = { ...(b.params ?? {}), [param]: next };
  });
}

export function setModRange(binding: string, param: string, from: number, to: number) {
  mutateBinding(binding, (b) => {
    const prev = b.params?.[param];
    if (!isDriver(prev)) return;
    b.params = { ...(b.params ?? {}), [param]: { ...prev, range: [from, to] } };
  });
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
