// The parameter panel for one layer — every dial its effect declares, with
// the effect's own ranges (descriptor min/max), never guessed ones.
//
// Each scalar is one of:
//   still      a fader. Dragging *pins* it on the leg your hands are on
//              (zero rebuild, next frame). An amber tick marks the scene value;
//              double-click releases the pin.
//   modulated  a range bar: FROM (rest) → TO (peak) with the live value
//              riding inside, plus the source chip (Kick, Bass, a clock
//              shape glyph + "4 bars"…) that opens the ModEditor popover.
//              Ranges, sources, shapes and rates are structure → DESIGN only.
// Colours are structure too (scene edits) → DESIGN only.
//
// `mode="author"` (Prepare) writes straight into scene.json instead of
// pinning — authoring is what the scene *is*; pins are how you play it.

import { useMemo, useState } from 'react';
import { paramSet, type LegName } from '../api/ipc';
import { useTelemetryDemand } from '../api/telemetryDemand';
import { pin, setDeck, setModRange, setSceneParam } from '../state/actions';
import {
  bindingHue,
  effectName,
  inputsFor,
  isDriver,
  layerTitle,
  parseScene,
  prettify,
  targetLabel,
  type Binding,
} from '../state/scene';
import { useStore, type EffectInput } from '../state/store';
import { Fader } from '../ui/Fader';
import { RangeFader } from '../ui/RangeFader';
import { Popover } from '../ui/Popover';
import { RegionGlyph } from './LayerStack';
import { ModChip, ModEditor } from './ModEditor';

type Mode = 'perform' | 'author';

export function LayerParams({ mode = 'perform', bindingId }: { mode?: Mode; bindingId?: string | null }) {
  const pack = useStore((s) => s.pack);
  const catalog = useStore((s) => s.catalog);
  const selected = useStore((s) => s.selectedBindingId);
  const deckLeg = useStore((s) => s.deck?.preview_source ?? 'design');
  const leg: LegName = mode === 'author' ? 'design' : deckLeg;
  const json = useStore((s) => (leg === 'live' ? s.liveSceneJson || s.sceneJson : s.sceneJson));
  const scene = useMemo(() => parseScene(json), [json]);
  const id = bindingId ?? selected;
  const index = (scene?.bindings ?? []).findIndex((b) => b.id === id);
  const binding = index >= 0 ? scene!.bindings![index] : null;

  if (!binding) {
    return <div className="p-4 text-[12px] text-fg3">Select a layer to play its controls.</div>;
  }

  const inputs = inputsFor(binding.effect, catalog);
  // Params the scene sets that the descriptor doesn't declare still show.
  const declared = new Set(inputs.map((i) => i.name));
  const extras: EffectInput[] = Object.entries(binding.params ?? {})
    .filter(([k]) => !declared.has(k))
    .map(([name, v]) => ({
      name,
      type: typeof v === 'string' ? 'color' : v && typeof v === 'object' && 'clip' in v ? 'image' : 'float',
    }));
  const all = [...inputs, ...extras];
  const hue = bindingHue(binding, pack, index);
  const structural = leg === 'design';

  return (
    <div className="flex flex-col min-h-0 h-full">
      {mode === 'perform' && (
      <div className="flex items-center gap-2.5 px-3 py-2 border-b border-line shrink-0">
        <RegionGlyph binding={binding} pack={pack} hue={hue} size={28} />
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold truncate">{layerTitle(binding)}</div>
          <div className="text-[10px] text-fg3 truncate">
            {targetLabel(binding, pack)} · <span className="text-fg2">{prettify(effectName(binding.effect))}</span>
          </div>
        </div>
        {!structural && (
          <button
            className="text-[10px] text-design/90 hover:text-design border border-design/30 hover:border-design/60 rounded px-1.5 h-5"
            title="Effect, colours and modulation are edited on the DESIGN leg (then TAKE). Switch hands to DESIGN."
            onClick={() => setDeck('design')}
          >
            edit in DESIGN
          </button>
        )}
      </div>
      )}
      <div className="flex-1 min-h-0 overflow-auto p-2.5 flex flex-col gap-1.5">
        {all.length === 0 && <div className="text-[11px] text-fg3 p-1">This effect has no controls.</div>}
        {all.map((input) =>
          input.type === 'image' ? (
            <ImageRow key={input.name} value={binding.params?.[input.name]} />
          ) : input.type === 'color' ? (
            <ColorRow key={input.name} binding={binding} input={input} editable={structural} />
          ) : (
            <ScalarRow key={input.name} binding={binding} input={input} leg={leg} mode={mode} structural={structural} />
          )
        )}
      </div>
    </div>
  );
}

/** A clip input (C5): shows what plays; clips are picked in the clip grid. */
function ImageRow({ value }: { value: any }) {
  const clip: string | null = value && typeof value === 'object' ? value.clip ?? null : null;
  return (
    <div className="flex items-center gap-2 h-8 px-2 rounded-md bg-panel2 border border-line text-[11px]">
      <span className="text-fg3 shrink-0">Clip</span>
      <span className="num truncate flex-1 text-fg" title={clip ?? undefined}>
        {clip ? clip.split('/').pop() : 'none — pick one in the clip grid'}
      </span>
      {clip && value.placement === 'canvas' && <span className="text-[9px] text-fg4">canvas</span>}
      {clip && value.rate != null && value.rate !== 1 && <span className="num text-[9px] text-fg4">×{value.rate}</span>}
    </div>
  );
}

function rangeFor(input: EffectInput, v: number): [number, number] {
  let lo = input.min ?? null;
  let hi = input.max ?? null;
  if (lo == null || hi == null) {
    const mag = Math.max(Math.abs(v), Math.abs(Number(input.default) || 0));
    const m = mag <= 1 ? 1 : Math.ceil(mag * 2);
    lo = lo ?? (v < 0 ? -m : 0);
    hi = hi ?? m;
  }
  // A scene value outside the declared range still has to be reachable.
  return [Math.min(lo, v), Math.max(hi, v)];
}

function ScalarRow({
  binding,
  input,
  leg,
  mode,
  structural,
}: {
  binding: Binding;
  input: EffectInput;
  leg: LegName;
  mode: Mode;
  structural: boolean;
}) {
  const raw = binding.params?.[input.name];
  const def = Number(input.default ?? 0) || 0;
  const pinned = useStore((s) => s.overrides[leg][binding.id]?.[input.name]);
  const driver = isDriver(raw) ? raw : null;
  const slider = driver?.driver === 'ui.slider' ? driver : null;
  const modded = driver && !slider;
  // The live driver value, as a primitive: the `drivers` channel replaces
  // its rows every frame (10 Hz), so selecting the row object re-rendered
  // every scalar row each frame. Only driven rows ask for the channel.
  useTelemetryDemand('drivers', !!driver);
  const liveValue = useStore((s) =>
    driver
      ? s.drivers.find((d) => d.binding_id === binding.id && d.param_name === input.name)?.value
      : undefined
  );
  const [menu, setMenu] = useState<HTMLElement | null>(null);
  const label = prettify(input.name) + (input.unit ? ` (${input.unit})` : '');
  const range: [number, number] | null = modded
    ? Array.isArray(driver!.range)
      ? [driver!.range[0], driver!.range[1]]
      : [0, 1]
    : null;
  const sceneVal = typeof raw === 'number' ? raw : range ? range[0] : slider ? slider.default ?? def : def;
  const [lo, hi] = rangeFor(input, range ? Math.max(range[0], range[1]) : sceneVal);
  const [lo2] = rangeFor(input, range ? Math.min(range[0], range[1]) : sceneVal);
  const min = Math.min(lo, lo2);

  let control: React.ReactNode;
  if (slider) {
    control = (
      <Fader
        label={label}
        value={liveValue ?? slider.default ?? 0}
        min={min}
        max={hi}
        color="rgb(var(--deck))"
        onChange={(v) => paramSet(slider.name, v, leg).catch(() => {})}
        title={`Shared knob “${slider.name}”`}
      />
    );
  } else if (modded && pinned == null) {
    control = (
      <RangeFader
        label={label}
        from={range![0]}
        to={range![1]}
        min={min}
        max={hi}
        live={liveValue ?? null}
        disabled={!structural}
        onChange={(f, t) => setModRange(binding.id, input.name, f, t)}
        title={
          structural
            ? 'Drag an end to set rest (left mark) and peak (bright mark); drag the band to move both.'
            : 'Modulation depth is set on DESIGN.'
        }
      />
    );
  } else if (mode === 'author' && !modded) {
    control = (
      <Fader
        label={label}
        value={pinned ?? sceneVal}
        min={min}
        max={hi}
        defaultValue={def}
        color="rgb(var(--design))"
        onChange={(v) => {
          if (pinned != null) pin(binding.id, input.name, null, 'design');
          setSceneParam(binding.id, input.name, v);
        }}
        title={`Scene value. Double-click: effect default (${def}).`}
      />
    );
  } else {
    // Pin path (perform): zero-rebuild live knob on the leg your hands are on.
    const value = pinned ?? (modded ? liveValue ?? sceneVal : sceneVal);
    control = (
      <Fader
        label={
          <>
            {label}
            {pinned != null && <span className="ml-1.5 text-amber text-[9px] font-semibold tracking-wider">PINNED</span>}
          </>
        }
        value={value}
        min={min}
        max={hi}
        color={pinned != null ? 'rgb(var(--amber))' : 'rgb(var(--deck))'}
        ghost={pinned != null && !modded ? sceneVal : null}
        onChange={(v) => pin(binding.id, input.name, v, leg)}
        onReset={() => pin(binding.id, input.name, null, leg)}
        title={
          modded
            ? 'Dragging freezes the modulation at your value (PINNED). Double-click releases it.'
            : 'Double-click: back to the scene value'
        }
      />
    );
  }

  return (
    <div className="relative grid grid-cols-[minmax(0,1fr)_66px] gap-1.5 items-center">
      {control}
      <ModChip raw={raw} disabled={!structural || !!slider} onClick={(el) => setMenu((m) => (m ? null : el))} />
      {menu && (
        <Popover anchor={menu} width={288} onClose={() => setMenu(null)}>
          <ModEditor
            bindingId={binding.id}
            param={input.name}
            raw={raw}
            current={typeof raw === 'number' ? raw : sceneVal}
            lo={min}
            hi={hi}
            onClose={() => setMenu(null)}
          />
        </Popover>
      )}
    </div>
  );
}

function ColorRow({ binding, input, editable }: { binding: Binding; input: EffectInput; editable: boolean }) {
  const raw = binding.params?.[input.name] ?? input.default ?? '#ffffff';
  const hex = toHex(raw);
  const dark = hex.toLowerCase() === '#000000';
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_66px] gap-1.5 items-center">
      <label
        className={
          'relative flex items-center gap-2 h-[26px] rounded-[5px] border border-line bg-panel2 px-2 ' +
          (editable ? 'hover:border-line2 cursor-pointer' : 'opacity-70')
        }
        title={editable ? 'Pick a colour' : 'Colours are edited on DESIGN'}
      >
        <span className="text-[11px] text-fg2 flex-1 truncate">{prettify(input.name)}</span>
        {dark && <span className="text-[9px] text-amber" title="Black adds no light on an additive surface">no light</span>}
        <span className="num text-[10px] text-fg3">{hex}</span>
        <span className="w-8 h-4 rounded-sm border border-line2" style={{ background: hex }} />
        {editable && (
          <input
            type="color"
            value={hex}
            className="absolute inset-0 opacity-0 cursor-pointer"
            onChange={(e) => setSceneParam(binding.id, input.name, e.target.value)}
          />
        )}
      </label>
      <div />
    </div>
  );
}

export function toHex(v: any): string {
  if (typeof v === 'string') {
    const raw = v.startsWith('#') ? v.slice(1) : v;
    if (raw.length === 3) return '#' + raw.split('').map((c) => c + c).join('');
    return '#' + raw.slice(0, 6).padEnd(6, '0');
  }
  if (Array.isArray(v)) {
    const c = (x: number) =>
      Math.round(Math.min(1, Math.max(0, x ?? 0)) * 255)
        .toString(16)
        .padStart(2, '0');
    return `#${c(v[0])}${c(v[1])}${c(v[2])}`;
  }
  return '#ffffff';
}
