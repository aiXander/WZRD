// Binding inspector — structured editor for the binding currently selected
// in the binding list. Monaco stays the source of truth; this panel exposes
// dropdowns / sliders / pickers that mutate `scene.json` and write back.
//
// Phase 4.2 scope: editable selector kind, effect dropdown, scalar param
// rows with literal vs. driver toggle. Driver picker enumerates the bus.
// "Add binding" button at the top opens a fresh row with sensible defaults.

import { useEffect, useMemo, useState } from 'react';
import { commitSceneText } from '../state/sceneCommit';
import { effectDescribe } from '../api/ipc';
import { useStore } from '../state/store';

const BUILTIN_EFFECTS = ['tint', 'hueCycle', 'flash', 'wobble'];

type EffectInput = { name: string; type: string; default?: any };
type EffectCatalog = Record<string, EffectInput[]>;

function toHex(v: any): string {
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) {
    const [r, g, b] = v;
    const c = (x: number) =>
      Math.round(Math.min(1, Math.max(0, x ?? 0)) * 255)
        .toString(16)
        .padStart(2, '0');
    return `#${c(r)}${c(g)}${c(b)}`;
  }
  return '#ffffff';
}

function defaultForInput(input: EffectInput): any {
  if (input.type === 'color') return toHex(input.default);
  const n = Number(input.default);
  return Number.isFinite(n) ? n : 0;
}

/** Declared inputs for a binding's effect: catalog for named effects,
 *  the embedded `inputs` array for inline ones. */
function inputsForEffect(effect: any, catalog: EffectCatalog): EffectInput[] {
  if (effect && typeof effect === 'object' && effect.inline) {
    return Array.isArray(effect.inputs) ? effect.inputs : [];
  }
  if (typeof effect === 'string') return catalog[effect] ?? [];
  return [];
}
const DRIVER_TYPES = [
  'const',
  'clock.bars',
  'clock.beats',
  'clock.phase',
  'clock.time',
  'audio.band',
  'audio.onset',
  'ui.slider',
];

type Binding = {
  id: string;
  select: any;
  effect: any;
  params?: Record<string, any>;
};

function parseScene(json: string): any | null {
  try {
    return JSON.parse(json);
  } catch {
    return null;
  }
}

export function BindingInspector() {
  const sceneJson = useStore((s) => s.sceneJson);
  const pack = useStore((s) => s.pack);
  const selected = useStore((s) => s.selectedBindingId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const setHovered = useStore((s) => s.setHoveredBindingId);
  const effects = useStore((s) => s.effects);
  const drivers = useStore((s) => s.drivers);

  // Leaving the inspector entirely (route switch, unmount) must not strand a
  // highlight on the surface canvas.
  useEffect(() => () => setHovered(null), [setHovered]);

  // Effect catalog (declared inputs incl. defaults) via `effect.describe`.
  // Drives param defaults on effect switch and the add-param choices, so
  // the inspector can never author params an effect doesn't declare.
  const [catalog, setCatalog] = useState<EffectCatalog>({});
  useEffect(() => {
    let cancelled = false;
    effectDescribe()
      .then((res: any) => {
        if (cancelled || !res?.effects) return;
        const next: EffectCatalog = {};
        for (const e of res.effects) next[e.name] = e.inputs ?? [];
        setCatalog(next);
      })
      .catch((e) => console.warn('effect.describe', e));
    return () => {
      cancelled = true;
    };
    // Re-fetch when the effect list changes (hot-reloaded user effects).
  }, [effects]);

  const scene = useMemo(() => parseScene(sceneJson), [sceneJson]);
  const bindings: Binding[] = scene?.bindings ?? [];
  const current = useMemo(
    () => bindings.find((b) => b.id === selected) ?? null,
    [bindings, selected]
  );

  function commit(mutator: (s: any) => any) {
    const next = mutator(structuredClone(scene));
    const text = JSON.stringify(next, null, 2);
    // Optimistic local update + debounced engine push / disk write — rapid
    // keystrokes and slider drags collapse into one plan rebuild.
    commitSceneText(text);
  }

  function addBinding() {
    commit((s) => {
      const newBinding: Binding = {
        id: `binding_${(s.bindings?.length ?? 0) + 1}`,
        select: { all: true },
        effect: 'tint',
        params: { color: '#ffffff' },
      };
      s.bindings = [...(s.bindings ?? []), newBinding];
      setSelected(newBinding.id);
      return s;
    });
  }

  function removeBinding(id: string) {
    commit((s) => {
      s.bindings = (s.bindings ?? []).filter((b: Binding) => b.id !== id);
      if (selected === id) setSelected(null);
      return s;
    });
  }

  if (!scene) {
    return (
      <div className="text-xs text-zinc-500">
        scene.json failed to parse — fix it in the editor and the inspector will reappear.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 text-xs">
      <div className="flex items-center justify-between">
        <div className="text-zinc-500">Bindings · {bindings.length}</div>
        <button
          className="px-2 py-1 rounded bg-ink-600 hover:bg-ink-500 text-zinc-100"
          onClick={addBinding}
        >
          + Add
        </button>
      </div>

      {/* Hovering a row highlights the regions it drives on the surface
          canvas; the row list clears the highlight as a whole so moving
          between rows doesn't flicker through a null. */}
      <ul
        className="border border-ink-700 rounded divide-y divide-ink-700"
        onMouseLeave={() => setHovered(null)}
      >
        {bindings.map((b) => (
          <li
            key={b.id}
            className={
              'px-2 py-1 cursor-pointer flex items-center justify-between ' +
              (selected === b.id ? 'bg-ink-600 text-zinc-100' : 'hover:bg-ink-700')
            }
            onMouseEnter={() => setHovered(b.id)}
            onClick={() => setSelected(b.id)}
          >
            <div className="truncate">
              <span className="text-zinc-200">{b.id}</span>
              <span className="text-zinc-500"> · {effectName(b.effect)}</span>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeBinding(b.id);
              }}
              className="text-zinc-500 hover:text-accent-red"
              title="Remove binding"
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      {current && (
        <div
          onMouseEnter={() => setHovered(current.id)}
          onMouseLeave={() => setHovered(null)}
        >
        <BindingEditor
          binding={current}
          pack={pack}
          effects={[...BUILTIN_EFFECTS, ...effects]}
          catalog={catalog}
          drivers={drivers}
          onChange={(next) =>
            commit((s) => {
              const idx = (s.bindings ?? []).findIndex(
                (b: Binding) => b.id === current.id
              );
              if (idx >= 0) s.bindings[idx] = next;
              return s;
            })
          }
        />
        </div>
      )}
    </div>
  );
}

function effectName(e: any): string {
  if (typeof e === 'string') return e;
  if (e && typeof e === 'object' && e.inline) return 'inline';
  return '?';
}

function BindingEditor({
  binding,
  pack,
  effects,
  catalog,
  drivers,
  onChange,
}: {
  binding: Binding;
  pack: any;
  effects: string[];
  catalog: EffectCatalog;
  drivers: Array<{ binding_id: string; param_name: string; value: number; source: string }>;
  onChange: (b: Binding) => void;
}) {
  // Key presence, not truthiness — `{ id: "" }` must still read as 'id'.
  const sel = binding.select ?? {};
  const selectKind =
    'id' in sel ? 'id' : 'tag' in sel ? 'tag' : 'group' in sel ? 'group' : 'all';

  function update(patch: Partial<Binding>) {
    onChange({ ...binding, ...patch });
  }

  function updateSelect(kind: 'all' | 'id' | 'tag' | 'group', value?: string) {
    const next: any =
      kind === 'all'
        ? { all: true }
        : kind === 'id'
        ? { id: value ?? ids[0] ?? '' }
        : kind === 'tag'
        ? { tag: value ?? tagList[0] ?? '' }
        : { group: value ?? groups[0] ?? '' };
    // `pick` is orthogonal to the member-set kind — switching kind must not
    // silently drop it (it flattened pick_bloom to `{ id: "" }` once).
    if (sel.pick) next.pick = sel.pick;
    update({ select: next });
  }

  function updateParam(name: string, value: any) {
    update({ params: { ...(binding.params ?? {}), [name]: value } });
  }

  const declaredInputs = inputsForEffect(binding.effect, catalog);
  const missingInputs = declaredInputs.filter(
    (i) => !(i.name in (binding.params ?? {}))
  );

  /** Switching effect rebuilds params from the new effect's declared
   *  defaults — carrying the old effect's params is a guaranteed engine
   *  rejection ("effect X has no param Y"). Same-named params survive. */
  function switchEffect(name: string) {
    if (name === '__inline__') {
      update({
        effect: {
          inline: true,
          name: binding.id,
          wgsl: 'fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> { let a = mask * 0.5; return vec4<f32>(a, a, a, a); }',
          inputs: [],
        },
        params: {},
      });
      return;
    }
    const inputs = catalog[name] ?? [];
    const params: Record<string, any> = {};
    for (const input of inputs) {
      const prev = (binding.params ?? {})[input.name];
      params[input.name] = prev !== undefined ? prev : defaultForInput(input);
    }
    update({ effect: name, params });
  }

  const ids: string[] = pack?.layers.map((l: any) => l.id) ?? [];
  /** Dropdown text: human label when the pack carries one, else the id. */
  const layerText = (id: string) => {
    const l = pack?.layers.find((x: any) => x.id === id);
    return l?.label && l.label !== l.id ? `${l.label} · ${id}` : id;
  };
  const tags = new Set<string>();
  pack?.layers.forEach((l: any) => l.tags.forEach((t: string) => tags.add(t)));
  const tagList = Array.from(tags).sort();
  const groups: string[] = pack?.groups.map((g: any) => g.id) ?? [];

  return (
    <div className="border border-ink-700 rounded p-3 flex flex-col gap-3 bg-ink-800">
      <Row label="id">
        <input
          className="bg-ink-900 px-1 py-0.5 rounded border border-ink-600 w-full"
          value={binding.id}
          onChange={(e) => update({ id: e.target.value })}
        />
      </Row>

      <Row label="select">
        <div className="flex gap-1">
          <select
            className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5"
            value={selectKind}
            onChange={(e) =>
              updateSelect(e.target.value as 'all' | 'id' | 'tag' | 'group')
            }
          >
            <option value="all">all</option>
            <option value="id">id</option>
            <option value="tag">tag</option>
            <option value="group">group</option>
          </select>
          {selectKind === 'id' && (
            <select
              className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5 flex-1"
              value={binding.select.id}
              onChange={(e) => updateSelect('id', e.target.value)}
            >
              {!ids.includes(binding.select.id) && (
                <option value={binding.select.id}>—</option>
              )}
              {ids.map((id) => (
                <option key={id} value={id}>
                  {layerText(id)}
                </option>
              ))}
            </select>
          )}
          {selectKind === 'tag' && (
            <select
              className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5 flex-1"
              value={binding.select.tag}
              onChange={(e) => updateSelect('tag', e.target.value)}
            >
              {!tagList.includes(binding.select.tag) && (
                <option value={binding.select.tag}>—</option>
              )}
              {tagList.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          )}
          {selectKind === 'group' && (
            <select
              className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5 flex-1"
              value={binding.select.group}
              onChange={(e) => updateSelect('group', e.target.value)}
            >
              {!groups.includes(binding.select.group) && (
                <option value={binding.select.group}>—</option>
              )}
              {groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          )}
        </div>
      </Row>

      <Row label="effect">
        {typeof binding.effect === 'string' ? (
          <select
            className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5 w-full"
            value={binding.effect}
            onChange={(e) => switchEffect(e.target.value)}
          >
            {effects.map((eName) => (
              <option key={eName} value={eName}>
                {eName}
              </option>
            ))}
            <option value="__inline__">(inline WGSL)</option>
          </select>
        ) : (
          <div className="text-zinc-400">inline WGSL — edit in Monaco scene tab</div>
        )}
      </Row>

      <div className="border-t border-ink-700 pt-2 text-zinc-500">params</div>
      <div className="flex flex-col gap-2">
        {Object.entries(binding.params ?? {}).map(([name, value]) => (
          <ParamRow
            key={name}
            name={name}
            value={value}
            declaredType={declaredInputs.find((i) => i.name === name)?.type}
            onChange={(v) => updateParam(name, v)}
            liveValue={
              drivers.find(
                (d) => d.binding_id === binding.id && d.param_name === name
              )?.value
            }
          />
        ))}
        {/* Only declared-but-unset inputs can be added — inventing names
            (the old `param_N` button) is a guaranteed engine rejection. */}
        {missingInputs.map((input) => (
          <button
            key={input.name}
            className="text-zinc-400 hover:text-zinc-100 text-left"
            onClick={() => updateParam(input.name, defaultForInput(input))}
          >
            + {input.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 items-center">
      <div className="w-16 text-zinc-500">{label}</div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

function ParamRow({
  name,
  value,
  declaredType,
  onChange,
  liveValue,
}: {
  name: string;
  value: any;
  declaredType?: string;
  onChange: (v: any) => void;
  liveValue?: number;
}) {
  // The declared input type bounds what this row may author — a float param
  // must never become a color string and vice versa (the engine rejects the
  // whole scene on a type mismatch). Unknown type → allow everything.
  const allowColor = declaredType !== 'float';
  const allowScalar = declaredType !== 'color';
  const isDriver = value && typeof value === 'object' && 'driver' in value;
  const driverKind: string =
    (isDriver && typeof value.driver === 'string' ? value.driver : 'const');

  function setDriverKind(kind: string) {
    if (kind === 'const') {
      onChange(0);
      return;
    }
    const skeleton: any = { driver: kind };
    if (kind.startsWith('clock.bars') || kind.startsWith('clock.beats')) {
      skeleton.n = 8;
    } else if (kind === 'clock.phase') {
      skeleton.rate = 0.1;
    } else if (kind.startsWith('audio.')) {
      skeleton.band = 'low';
      if (kind === 'audio.onset') skeleton.decay = 0.15;
    } else if (kind === 'ui.slider') {
      skeleton.name = name;
      skeleton.default = 0.5;
    }
    onChange(skeleton);
  }

  return (
    <div className="flex flex-col gap-1 border border-ink-700 rounded p-2">
      <div className="flex items-center gap-2">
        <div className="text-zinc-300 flex-1">{name}</div>
        <select
          className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5 text-[11px]"
          value={isDriver ? driverKind : (typeof value === 'string' ? 'color' : 'const')}
          onChange={(e) => {
            if (e.target.value === 'color') {
              onChange('#ffffff');
            } else {
              setDriverKind(e.target.value);
            }
          }}
        >
          {allowScalar && <option value="const">number</option>}
          {allowColor && <option value="color">color</option>}
          {allowScalar &&
            DRIVER_TYPES.filter((d) => d !== 'const').map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
        </select>
      </div>
      {typeof value === 'number' && (
        <input
          type="number"
          step="0.01"
          className="bg-ink-900 px-1 py-0.5 rounded border border-ink-600"
          value={value}
          onChange={(e) => {
            // An emptied field parses to NaN → serializes to null → the
            // engine rejects the whole scene. Don't commit until numeric.
            const n = parseFloat(e.target.value);
            if (Number.isFinite(n)) onChange(n);
          }}
        />
      )}
      {typeof value === 'string' && (
        <input
          type="color"
          className="w-full h-7 bg-ink-900 rounded"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {isDriver && (
        <div className="flex flex-col gap-1">
          {value.driver === 'clock.bars' || value.driver === 'clock.beats' ? (
            <NumberField
              label="n"
              value={value.n ?? 1}
              onChange={(n) => onChange({ ...value, n })}
            />
          ) : null}
          {value.driver === 'clock.phase' ? (
            <NumberField
              label="rate"
              value={value.rate ?? 0.1}
              onChange={(rate) => onChange({ ...value, rate })}
            />
          ) : null}
          {value.driver === 'audio.band' || value.driver === 'audio.onset' ? (
            <select
              className="bg-ink-900 border border-ink-600 rounded px-1 py-0.5"
              value={value.band ?? 'low'}
              onChange={(e) => onChange({ ...value, band: e.target.value })}
            >
              <option value="low">low</option>
              <option value="mid">mid</option>
              <option value="high">high</option>
            </select>
          ) : null}
          {value.driver === 'audio.onset' ? (
            <NumberField
              label="decay"
              value={value.decay ?? 0.15}
              onChange={(decay) => onChange({ ...value, decay })}
            />
          ) : null}
          {value.driver === 'ui.slider' ? (
            <>
              <TextField
                label="name"
                value={value.name ?? ''}
                onChange={(s) => onChange({ ...value, name: s })}
              />
              <NumberField
                label="default"
                value={value.default ?? 0}
                onChange={(d) => onChange({ ...value, default: d })}
              />
            </>
          ) : null}
          {liveValue !== undefined && (
            <div className="h-1 bg-ink-700 rounded overflow-hidden">
              <div
                className="h-full bg-accent-violet"
                style={{ width: `${Math.min(100, Math.max(0, liveValue * 100))}%` }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px]">
      <span className="w-12 text-zinc-500">{label}</span>
      <input
        type="number"
        step="0.01"
        className="bg-ink-900 px-1 py-0.5 rounded border border-ink-600 flex-1"
        value={value}
        onChange={(e) => {
          const n = parseFloat(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
      />
    </label>
  );
}
function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px]">
      <span className="w-12 text-zinc-500">{label}</span>
      <input
        className="bg-ink-900 px-1 py-0.5 rounded border border-ink-600 flex-1"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
