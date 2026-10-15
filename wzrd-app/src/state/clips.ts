// The two scene writes a deck slot makes on DESIGN (gig C5, studio D1): put
// a clip on a region's binding, or stop it. The decks themselves (which
// clips sit in which row) live in `state/decks.ts`.
//
// Semantics, one rule: a region row is a radio group — each region plays at
// most one deck clip on DESIGN. Click an idle clip = it replaces whatever the
// region played; click the playing one again (or the row's ■ slot) = stop.

import { commitSceneMutation } from './sceneCommit';
import { inputsFor, parseScene, uniqueId, type Binding } from './scene';
import { useStore } from './store';

/** region id → clip paths its bindings play, in one leg's scene. Only
 *  bindings that target exactly that region count (a tag/group binding
 *  isn't "this region's clip"). */
export function clipsByRegion(sceneJson: string): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const b of parseScene(sceneJson)?.bindings ?? []) {
    const region = b.select?.id;
    const clip = b.params?.image?.clip;
    if (typeof region === 'string' && typeof clip === 'string') (out[region] ??= []).push(clip);
  }
  return out;
}

/**
 * Put `clip` on `region` in DESIGN (C5), through sceneCommit like every
 * scene write: set `image.clip` on the region's first binding whose effect
 * declares an `image` input (keeping its placement/rate), else add
 * `clip_<region>` playing it through the built-in `videoClip`. Returns the
 * binding id. Taking it live stays the operator's TAKE; stopClip is the reverse.
 */
export function bindClip(clip: string, region: string): string {
  const { catalog } = useStore.getState();
  const hasImage = (b: Binding) => inputsFor(b.effect, catalog).some((i) => i.type === 'image');
  let id = '';
  commitSceneMutation((s) => {
    s.bindings = s.bindings ?? [];
    const target = s.bindings.find((b: Binding) => b.select?.id === region && hasImage(b));
    if (target) {
      const prev = target.params?.image && typeof target.params.image === 'object' ? target.params.image : {};
      target.params = { ...(target.params ?? {}), image: { placement: 'region', rate: 1, ...prev, clip } };
      id = target.id;
    } else {
      id = uniqueId(`clip_${region}`, new Set(s.bindings.map((b: Binding) => b.id)));
      const gain = catalog.videoClip?.inputs.find((i) => i.name === 'gain')?.default;
      s.bindings.push({
        id,
        select: { id: region },
        effect: 'videoClip',
        params: {
          image: { clip, placement: 'region', rate: 1 },
          ...(gain != null ? { gain: Number(gain) } : {}),
        },
      });
    }
    return s;
  });
  useStore.getState().setSelectedBindingId(id);
  return id;
}

/**
 * Stop what `region` plays on DESIGN — `clip` only, or any clip when
 * omitted (the row's ■ slot). Through sceneCommit like every scene write.
 * "Stop" keeps an author's work: a built-in `videoClip` binding exists only
 * to show its clip, so it goes (the next click recreates it); any other
 * effect with an `image` input just loses its `image` param (C5: an absent
 * image samples transparent) and keeps every other setting. Returns how
 * many bindings changed. Like bindClip, the wall keeps playing until TAKE.
 */
export function stopClip(region: string, clip?: string): number {
  const st = useStore.getState();
  const hits = (parseScene(st.sceneJson)?.bindings ?? []).filter((b) => {
    const c = b.params?.image?.clip;
    return b.select?.id === region && typeof c === 'string' && (clip == null || c === clip);
  });
  if (hits.length === 0) return 0;
  const ids = new Set(hits.map((b) => b.id));
  const removed = new Set(hits.filter((b) => b.effect === 'videoClip').map((b) => b.id));
  commitSceneMutation((s) => {
    s.bindings = (s.bindings ?? []).filter((b: Binding) => !removed.has(b.id));
    for (const b of s.bindings as Binding[]) {
      if (ids.has(b.id) && b.params) {
        const { image: _drop, ...rest } = b.params;
        b.params = rest;
      }
    }
    return s;
  });
  if (st.selectedBindingId && removed.has(st.selectedBindingId)) st.setSelectedBindingId(null);
  return hits.length;
}
