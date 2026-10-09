// Read-side helpers over the design scene + pack: parse once, name things in
// surface-language, resolve selectors, read effect descriptors.

import { useMemo } from 'react';
import type { PackInfo, PackLayer } from '../api/ipc';
import { useStore, type EffectInput } from './store';

export type Binding = {
  id: string;
  select: any;
  effect: any;
  params?: Record<string, any>;
};
export type Scene = {
  version?: number;
  pack?: string;
  transport?: { bpm?: number };
  bindings?: Binding[];
  [k: string]: any;
};

export function parseScene(json: string): Scene | null {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

export function useScene(): Scene | null {
  const json = useStore((s) => s.sceneJson);
  return useMemo(() => parseScene(json), [json]);
}

/** Stable key-sorted serialization, for "is design == live?" checks. */
export function canonical(json: string): string {
  const sort = (v: any): any =>
    Array.isArray(v)
      ? v.map(sort)
      : v && typeof v === 'object'
      ? Object.fromEntries(
          Object.keys(v)
            .sort()
            .map((k) => [k, sort(v[k])])
        )
      : v;
  try {
    return JSON.stringify(sort(JSON.parse(json)));
  } catch {
    return json;
  }
}

export function effectName(e: any): string {
  if (typeof e === 'string') return e;
  if (e && typeof e === 'object' && e.inline) return e.name ? `${e.name}` : 'inline';
  return '?';
}

/** Declared inputs for a binding's effect (catalog, or the inline list). */
export function inputsFor(effect: any, catalog: Record<string, { inputs: EffectInput[] }>): EffectInput[] {
  if (effect && typeof effect === 'object' && effect.inline) {
    return Array.isArray(effect.inputs) ? effect.inputs : [];
  }
  if (typeof effect === 'string') return catalog[effect]?.inputs ?? [];
  return [];
}

/** One stable hue per pack layer index — the region's colour everywhere. */
export function hueForIndex(i: number): number {
  return (i * 47 + 190) % 360;
}

export function regionName(pack: PackInfo | null, id: string): string {
  const l = pack?.layers.find((x) => x.id === id);
  return l?.label && l.label !== l.id ? l.label : prettify(id);
}

export function prettify(id: string): string {
  return id.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function selectorCovers(select: any, pack: PackInfo, layer: PackLayer): boolean {
  const s = select ?? {};
  if (s.all === true) return true;
  if (s.id === layer.id) return true;
  if (s.tag && layer.tags.includes(s.tag)) return true;
  if (s.group) {
    const g = (pack.groups ?? []).find((grp) => grp.id === s.group);
    if (g && g.members.includes(layer.id)) return true;
  }
  return false;
}

/** Regions a binding lights up (pick ignored — member set, like the engine). */
export function regionsOf(binding: Binding | undefined | null, pack: PackInfo | null): PackLayer[] {
  if (!binding || !pack) return [];
  return pack.layers.filter((l) => selectorCovers(binding.select, pack, l));
}

/** Human description of a selector: "Main facade", "#tower ×2", "All". */
export function targetLabel(binding: Binding, pack: PackInfo | null): string {
  const s = binding.select ?? {};
  let base: string;
  if (s.all) base = 'Everything';
  else if (s.id != null) base = regionName(pack, s.id);
  else if (s.tag != null) base = `#${s.tag}`;
  else if (s.group != null) base = `group ${s.group}`;
  else base = '—';
  if (s.pick) base += s.pick.mode === 'random_each' ? ' · random' : ' · pick one';
  return base;
}

/** Human name for a binding — the id is the stable key, this is the label. */
export function layerTitle(binding: Binding): string {
  return prettify(binding.id);
}

export function isDriver(v: any): v is { driver: string; [k: string]: any } {
  return !!v && typeof v === 'object' && typeof v.driver === 'string';
}

/** Hue for a binding: its first region's hue (falls back to index). */
export function bindingHue(binding: Binding, pack: PackInfo | null, index: number): number {
  const regs = regionsOf(binding, pack);
  if (pack && regs.length === 1) {
    return hueForIndex(pack.layers.findIndex((l) => l.id === regs[0].id));
  }
  return (index * 67 + 20) % 360;
}

export function uniqueId(base: string, taken: Set<string>): string {
  const clean = base.toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'layer';
  if (!taken.has(clean)) return clean;
  for (let i = 2; ; i++) if (!taken.has(`${clean}_${i}`)) return `${clean}_${i}`;
}
