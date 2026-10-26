// Prepare's inspector for one design binding: name + Duplicate / Delete /
// deselect in the header (they used to sit under the dials list, out of
// sight), target ("Lights up"), the grouped effect picker, then the
// author-mode dials.

import { useMemo } from 'react';
import { clearSelection, deleteLayer } from '../state/actions';
import { commitSceneMutation } from '../state/sceneCommit';
import { regionName, uniqueId, type Binding } from '../state/scene';
import { useStore, type EffectInput } from '../state/store';
import { Btn, InlineEdit, Seg } from '../ui/primitives';
import { EffectPicker } from './EffectPicker';
import { LayerParams, toHex } from './LayerParams';

/** Scene params for an effect's inputs, keeping values `prev` already has. */
export function defaultsFor(inputs: EffectInput[], prev: Record<string, any> = {}): Record<string, any> {
  const out: Record<string, any> = {};
  for (const i of inputs) {
    if (prev[i.name] !== undefined) out[i.name] = prev[i.name];
    // An `image` input has no default: a clip arrives from the clip grid.
    else if (i.type === 'image') continue;
    else if (i.type === 'color') out[i.name] = toHex(i.default);
    else out[i.name] = Number(i.default) || 0;
  }
  return out;
}

export function LayerInspector({ binding, bindings }: { binding: Binding; bindings: Binding[] }) {
  const pack = useStore((s) => s.pack);
  const catalog = useStore((s) => s.catalog);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const sel = binding.select ?? {};
  const kind: 'id' | 'tag' | 'group' | 'all' =
    'id' in sel ? 'id' : 'tag' in sel ? 'tag' : 'group' in sel ? 'group' : 'all';
  const tags = useMemo(() => Array.from(new Set(pack?.layers.flatMap((l) => l.tags) ?? [])).sort(), [pack]);
  const groups = pack?.groups.map((g) => g.id) ?? [];
  const ids = pack?.layers.map((l) => l.id) ?? [];

  function update(fn: (b: Binding) => void) {
    commitSceneMutation((s) => {
      const i = (s.bindings ?? []).findIndex((b: Binding) => b.id === binding.id);
      if (i >= 0) fn(s.bindings[i]);
      return s;
    });
  }

  function setSelect(k: typeof kind, value?: string) {
    const next: any =
      k === 'all'
        ? { all: true }
        : k === 'id'
        ? { id: value ?? ids[0] ?? '' }
        : k === 'tag'
        ? { tag: value ?? tags[0] ?? '' }
        : { group: value ?? groups[0] ?? '' };
    if (sel.pick) next.pick = sel.pick;
    update((b) => {
      b.select = next;
    });
  }

  function setPick(mode: 'none' | 'random_each' | 'random_static') {
    update((b) => {
      const s = { ...b.select };
      if (mode === 'none') delete s.pick;
      else if (mode === 'random_each') s.pick = { mode, rate: { driver: 'clock.bars', n: 4 } };
      else s.pick = { mode };
      b.select = s;
    });
  }

  function switchEffect(name: string) {
    update((b) => {
      b.effect = name;
      b.params = defaultsFor(catalog[name]?.inputs ?? [], b.params ?? {});
      // Drop params the new effect doesn't declare — the engine rejects them.
      const declared = new Set((catalog[name]?.inputs ?? []).map((i) => i.name));
      for (const k of Object.keys(b.params)) if (!declared.has(k)) delete b.params[k];
    });
  }

  function rename(next: string) {
    const id = uniqueId(next, new Set(bindings.filter((b) => b.id !== binding.id).map((b) => b.id)));
    update((b) => {
      b.id = id;
    });
    setSelected(id);
  }

  function duplicate() {
    const id = uniqueId(binding.id + '_copy', new Set(bindings.map((b) => b.id)));
    commitSceneMutation((s) => {
      const i = s.bindings.findIndex((b: Binding) => b.id === binding.id);
      s.bindings.splice(i + 1, 0, { ...structuredClone(binding), id });
      return s;
    });
    setSelected(id);
  }

  const current = typeof binding.effect === 'string' ? binding.effect : null;
  const pickMode: 'none' | 'random_each' | 'random_static' = sel.pick?.mode ?? 'none';

  return (
    <div className="flex flex-col">
      <div className="px-3 pt-3 pb-2 flex flex-col gap-2 border-b border-line">
        <div className="flex items-center gap-1.5">
          <InlineEdit value={binding.id} onCommit={rename} className="text-[14px] font-semibold flex-1" />
          <Btn size="xs" onClick={duplicate} title="Duplicate this layer">
            Duplicate
          </Btn>
          <Btn size="xs" tone="danger" onClick={() => deleteLayer(binding.id)} title="Delete layer (Delete / ⌫) — ⌘Z restores it">
            Delete
          </Btn>
          <Btn size="xs" tone="ghost" square onClick={clearSelection} title="Deselect (Esc)">
            ×
          </Btn>
        </div>
        <div className="label">Lights up</div>
        <Seg
          value={kind}
          onChange={(k) => setSelect(k)}
          options={[
            { id: 'id', label: 'Region' },
            { id: 'tag', label: 'Tag' },
            { id: 'group', label: 'Group' },
            { id: 'all', label: 'All' },
          ]}
        />
        {kind !== 'all' && (
          <select
            value={sel[kind] ?? ''}
            onChange={(e) => setSelect(kind, e.target.value)}
            className="w-full h-7"
          >
            {kind === 'id' &&
              ids.map((id) => (
                <option key={id} value={id}>
                  {regionName(pack, id)}
                </option>
              ))}
            {kind === 'tag' &&
              tags.map((t) => (
                <option key={t} value={t}>
                  #{t}
                </option>
              ))}
            {kind === 'group' &&
              groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
          </select>
        )}
        {kind !== 'id' && (
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-fg3">Members</span>
            <Seg
              size="xs"
              value={pickMode}
              onChange={setPick}
              options={[
                { id: 'none', label: 'All at once' },
                { id: 'random_each', label: 'One / 4 bars', title: 'A different random member every 4 bars' },
                { id: 'random_static', label: 'One', title: 'One random member, fixed' },
              ]}
            />
          </div>
        )}
      </div>

      <div className="px-3 pt-2.5 pb-3 border-b border-line">
        <EffectPicker current={current} onPick={switchEffect} />
        {!current && <div className="mt-1.5 text-[11px] text-fg3">Inline WGSL effect — edit it under Code.</div>}
      </div>

      <div className="min-h-[200px]">
        <LayerParams mode="author" bindingId={binding.id} />
      </div>
    </div>
  );
}

/** Inspector body for a selected region nothing lights. */
export function DarkRegionInspector({ regionId, onCreate }: { regionId: string; onCreate: (id: string) => void }) {
  const pack = useStore((s) => s.pack);
  return (
    <div className="p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="text-[13px] font-semibold flex-1 truncate">{regionName(pack, regionId)}</div>
        <Btn size="xs" tone="ghost" square onClick={clearSelection} title="Deselect (Esc)">
          ×
        </Btn>
      </div>
      <div className="text-[12px] text-fg3 leading-snug">This region is dark — no layer lights it.</div>
      <Btn tone="design" active onClick={() => onCreate(regionId)}>
        + Light it up
      </Btn>
    </div>
  );
}
