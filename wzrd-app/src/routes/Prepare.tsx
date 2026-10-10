// Prepare — daylight-me's workbench (spec §7: "preparation, not
// configuration"). Always the DESIGN leg: nothing here touches the wall
// until TAKE.
//
//   ┌ layers ┬ surface map (primary) ─────────────┬ inspector ┐
//   │ regions│                                     │ target    │
//   │        │                                     │ effect    │
//   │        │                                     │ dials     │
//   └────────┴─ drawer: clip grid | code (scene.json + WGSL) ┴┘
//
// Building a layer is two clicks: a region on the map, an effect card.
// Putting a rendered clip on a region is one: its slot in the clip grid.

import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { SurfaceMap } from '../components/SurfaceMap';
import { ClipGrid } from '../components/ClipGrid';
import { LayerParams } from '../components/LayerParams';
import { RegionGlyph } from '../components/LayerStack';
import { setDeck } from '../state/actions';
import { commitSceneMutation } from '../state/sceneCommit';
import { getMask, useMaskVersion } from '../state/masks';
import {
  bindingHue,
  effectName,
  hueForIndex,
  layerTitle,
  prettify,
  regionName,
  selectorCovers,
  targetLabel,
  uniqueId,
  useScene,
  type Binding,
} from '../state/scene';
import { useStore, type EffectInput } from '../state/store';
import { toHex } from '../components/LayerParams';
import { Btn, InlineEdit, Panel, Seg } from '../ui/primitives';

const MonacoPanel = lazy(() => import('../components/MonacoPanel').then((m) => ({ default: m.MonacoPanel })));

function defaultsFor(inputs: EffectInput[], prev: Record<string, any> = {}): Record<string, any> {
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

export function Prepare() {
  const pack = useStore((s) => s.pack);
  const catalog = useStore((s) => s.catalog);
  const selected = useStore((s) => s.selectedBindingId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const setRegion = useStore((s) => s.setSelectedLayerId);
  const setHovered = useStore((s) => s.setHoveredBindingId);
  const toast = useStore((s) => s.toast);
  const scene = useScene();
  const bindings = scene?.bindings ?? [];
  const [armed, setArmed] = useState(false);
  const [drawer, setDrawerState] = useState<'clips' | 'code' | null>(() => {
    const v = localStorage.getItem('wzrd.prepare.drawer');
    return v === 'code' || v === 'none' ? (v === 'none' ? null : 'code') : 'clips';
  });
  const setDrawer = (d: 'clips' | 'code' | null) => {
    const next = drawer === d ? null : d;
    localStorage.setItem('wzrd.prepare.drawer', next ?? 'none');
    setDrawerState(next);
  };

  // Authoring is a DESIGN act — put the hands there on the way in.
  useEffect(() => {
    if (useStore.getState().deck?.preview_source === 'live') {
      setDeck('design');
      toast('Hands on DESIGN for authoring — the wall is untouched until TAKE', 'info');
    }
  }, [toast]);
  useEffect(() => () => setHovered(null), [setHovered]);

  const defaultEffect = useMemo(() => {
    const users = Object.entries(catalog).filter(([, e]) => e.kind === 'user').map(([n]) => n);
    return users.sort()[0] ?? 'tint';
  }, [catalog]);

  function createLayer(regionId: string) {
    const label = regionName(pack, regionId);
    const id = uniqueId(label, new Set(bindings.map((b) => b.id)));
    commitSceneMutation((s) => {
      s.bindings = [
        ...(s.bindings ?? []),
        {
          id,
          select: { id: regionId },
          effect: defaultEffect,
          params: defaultsFor(catalog[defaultEffect]?.inputs ?? []),
        },
      ];
      return s;
    });
    setSelected(id);
    setRegion(regionId);
    setArmed(false);
  }

  function onRegionClick(regionId: string | null) {
    setRegion(regionId);
    if (!regionId || !pack) return;
    if (armed) return createLayer(regionId);
    const layer = pack.layers.find((l) => l.id === regionId)!;
    const covering = bindings.filter((b) => selectorCovers(b.select, pack, layer));
    if (covering.length) {
      // Cycle through the layers lighting this region on repeated clicks.
      const i = covering.findIndex((b) => b.id === selected);
      setSelected(covering[(i + 1) % covering.length].id);
    } else {
      setSelected(null);
    }
  }

  const current = bindings.find((b) => b.id === selected) ?? null;
  const selectedRegion = useStore((s) => s.selectedLayerId);
  const darkRegion =
    !current && selectedRegion && pack && !bindings.some((b) => selectorCovers(b.select, pack, pack.layers.find((l) => l.id === selectedRegion)!))
      ? selectedRegion
      : null;

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 grid grid-cols-[248px_minmax(0,1fr)_340px] gap-2 p-2">
        {/* Left: layers + regions */}
        <div className="min-h-0 flex flex-col gap-2">
          <Panel
            title="Layers"
            right={
              <Btn
                size="sm"
                tone="design"
                active={armed}
                onClick={() => {
                  if (selectedRegion && !armed) createLayer(selectedRegion);
                  else setArmed((a) => !a);
                }}
                title="New layer on the selected region — or click a region on the map next"
              >
                + New
              </Btn>
            }
            className="flex-[3]"
            bodyClassName="overflow-auto"
          >
            <div className="flex flex-col p-1.5 gap-1" onMouseLeave={() => setHovered(null)}>
              {bindings.length === 0 && (
                <div className="p-2 text-[11px] text-fg3 leading-snug">
                  The surface is dark. Click a region on the map, then <b className="text-fg2">+ New</b>.
                </div>
              )}
              {bindings.map((b, i) => (
                <button
                  key={b.id}
                  onMouseEnter={() => setHovered(b.id)}
                  onClick={() => setSelected(b.id)}
                  className={
                    'flex items-center gap-2 h-10 px-1.5 rounded-md border text-left ' +
                    (b.id === selected ? 'bg-raised border-design/60' : 'border-transparent hover:bg-panel2')
                  }
                >
                  <RegionGlyph binding={b} pack={pack} hue={bindingHue(b, pack, i)} size={26} />
                  <div className="min-w-0">
                    <div className="text-[12px] font-medium truncate">{layerTitle(b)}</div>
                    <div className="text-[10px] text-fg3 truncate">
                      {targetLabel(b, pack)} · {prettify(effectName(b.effect))}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </Panel>
          <Panel title="Regions" className="flex-[2]" bodyClassName="overflow-auto">
            <RegionList bindings={bindings} onPick={onRegionClick} onCreate={createLayer} />
          </Panel>
        </div>

        {/* Center: the surface */}
        <Panel className="overflow-hidden" bodyClassName="min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <SurfaceMap bindings={bindings} onRegionClick={onRegionClick} armed={armed} />
          </div>
          <div className="flex items-center gap-2 h-8 px-2 border-t border-line shrink-0">
            <Btn size="sm" tone="ghost" active={drawer === 'clips'} onClick={() => setDrawer('clips')} title="Rendered clips per region — click one to play it on DESIGN">
              ▦ Clips
            </Btn>
            <Btn size="sm" tone="ghost" active={drawer === 'code'} onClick={() => setDrawer('code')} title="scene.json + effect shaders">
              {'</>'} Code
            </Btn>
            <span className="text-[10px] text-fg4">
              {drawer === 'clips'
                ? 'Click a clip to play it on its region (DESIGN), drag it to another row to re-file it, TAKE to go live.'
                : 'Plain files: scene.json + effects/*/shader.wgsl — the AI and a text editor author the same scene.'}
            </span>
          </div>
        </Panel>

        {/* Right: inspector */}
        <Panel title={current ? 'Layer' : 'Inspector'} bodyClassName="overflow-auto">
          {current ? (
            <Inspector binding={current} bindings={bindings} />
          ) : darkRegion ? (
            <div className="p-4 flex flex-col gap-3">
              <div className="text-[13px] font-semibold">{regionName(pack, darkRegion)}</div>
              <div className="text-[12px] text-fg3 leading-snug">This region is dark — no layer lights it.</div>
              <Btn tone="design" active onClick={() => createLayer(darkRegion)}>
                + Light it up
              </Btn>
            </div>
          ) : (
            <div className="p-4 text-[12px] text-fg3 leading-relaxed">
              Pick a layer on the left, or click any region on the surface.
            </div>
          )}
        </Panel>
      </div>

      {drawer === 'code' && (
        <div className="h-[42%] min-h-[220px] mx-2 mb-2 rounded-lg border border-line overflow-hidden">
          <Suspense fallback={<div className="p-3 text-[11px] text-fg3">loading editor…</div>}>
            <MonacoPanel />
          </Suspense>
        </div>
      )}
      {drawer === 'clips' && (
        <div className="h-[34%] min-h-[180px] mx-2 mb-2 rounded-lg border border-line bg-panel overflow-hidden">
          <ClipGrid />
        </div>
      )}
    </div>
  );
}

function RegionList({
  bindings,
  onPick,
  onCreate,
}: {
  bindings: Binding[];
  onPick: (id: string) => void;
  onCreate: (id: string) => void;
}) {
  const pack = useStore((s) => s.pack);
  const sel = useStore((s) => s.selectedLayerId);
  useMaskVersion();
  if (!pack) return null;
  return (
    <div className="flex flex-col p-1.5 gap-0.5">
      {pack.layers.map((l, i) => {
        const n = bindings.filter((b) => selectorCovers(b.select, pack, l)).length;
        const m = getMask(l.id);
        const hue = hueForIndex(i);
        return (
          <div
            key={l.id}
            onClick={() => onPick(l.id)}
            onDoubleClick={() => onCreate(l.id)}
            title={`${l.id}${l.tags.length ? ' · #' + l.tags.join(' #') : ''}\nclick: select · double-click: new layer here`}
            className={
              'flex items-center gap-2 h-7 px-1.5 rounded-md ' + (sel === l.id ? 'bg-raised' : 'hover:bg-panel2')
            }
          >
            <span
              className="w-4 h-3 shrink-0 rounded-[2px]"
              style={{
                background: `hsl(${hue} 85% 62%)`,
                WebkitMaskImage: m ? `url(${m.thumb})` : undefined,
                WebkitMaskSize: '100% 100%',
              }}
            />
            <span className="text-[11px] truncate flex-1">{regionName(pack, l.id)}</span>
            {l.tags.slice(0, 1).map((t) => (
              <span key={t} className="text-[9px] text-fg4">
                #{t}
              </span>
            ))}
            <span className={'num text-[10px] w-4 text-right ' + (n ? 'text-design' : 'text-fg4')}>{n || '·'}</span>
          </div>
        );
      })}
    </div>
  );
}

function Inspector({ binding, bindings }: { binding: Binding; bindings: Binding[] }) {
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

  function remove() {
    commitSceneMutation((s) => {
      s.bindings = s.bindings.filter((b: Binding) => b.id !== binding.id);
      return s;
    });
    setSelected(null);
  }

  const effects = Object.entries(catalog).sort(([a, x], [b, y]) =>
    x.kind === y.kind ? a.localeCompare(b) : x.kind === 'user' ? -1 : 1
  );
  const current = typeof binding.effect === 'string' ? binding.effect : null;
  const pickMode: 'none' | 'random_each' | 'random_static' = sel.pick?.mode ?? 'none';

  return (
    <div className="flex flex-col">
      <div className="px-3 pt-3 pb-2 flex flex-col gap-2 border-b border-line">
        <InlineEdit value={binding.id} onCommit={rename} className="text-[14px] font-semibold" />
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
        <div className="label mb-2">Effect</div>
        <div className="grid grid-cols-2 gap-1.5">
          {effects.map(([name, e]) => (
            <button
              key={name}
              onClick={() => switchEffect(name)}
              className={
                'h-11 rounded-md border px-2 text-left flex flex-col justify-center ' +
                (current === name ? 'border-design bg-design/10' : 'border-line bg-panel2 hover:border-line2')
              }
            >
              <span className="text-[12px] font-medium truncate">{prettify(name)}</span>
              <span className="text-[9px] text-fg3">
                {e.kind === 'user' ? 'project' : 'built-in'} · {e.inputs.length} dial{e.inputs.length === 1 ? '' : 's'}
              </span>
            </button>
          ))}
          {!current && <div className="col-span-2 text-[11px] text-fg3">Inline WGSL effect — edit it under Code.</div>}
        </div>
      </div>

      <div className="min-h-[200px]">
        <LayerParams mode="author" bindingId={binding.id} />
      </div>

      <div className="flex gap-2 p-3 border-t border-line">
        <Btn size="sm" onClick={duplicate}>
          Duplicate
        </Btn>
        <div className="flex-1" />
        <Btn size="sm" tone="danger" onClick={remove}>
          Delete layer
        </Btn>
      </div>
    </div>
  );
}
