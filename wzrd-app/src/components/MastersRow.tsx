// §5.4 masters row — always visible in Perform. Engine-level, operator-owned
// globals the AI can never touch: they live outside scene.json entirely and
// persist via the session sidecar. Values arrive on the sticky `masters`
// telemetry channel; a slider drag wins locally so the knob never jumps
// under the finger.
//
// §5.6 full-control-switch: the deck toggle picks which leg these faders
// drive — tuning DESIGN never touches the show; PROMOTE copies the design
// values live. Switching legs re-reads that leg's values (local drag state
// resets so the faders snap to the newly selected leg).

import { useEffect, useState } from 'react';
import { masterSet, type LegName } from '../api/ipc';
import { useStore, type Masters } from '../state/store';

type MasterDef = {
  key: keyof Masters;
  label: string;
  min: number;
  max: number;
  default: number;
};

const MASTERS: MasterDef[] = [
  { key: 'brightness', label: 'BRIGHT', min: 0, max: 2, default: 1 },
  { key: 'speed', label: 'SPEED', min: 0, max: 4, default: 1 },
  { key: 'saturation', label: 'SAT', min: 0, max: 2, default: 1 },
  { key: 'audioListen', label: 'AUDIO', min: 0, max: 1, default: 1 },
];

// Trailing throttle so drags don't flood the IPC bridge (same cadence as the
// driver rack's param path).
const timers: Record<string, number> = {};
function sendMaster(name: string, value: number, leg: LegName) {
  if (timers[name] != null) window.clearTimeout(timers[name]);
  timers[name] = window.setTimeout(() => {
    delete timers[name];
    masterSet(name, value, leg).catch((e) => console.warn('master.set', name, e));
  }, 25);
}

export function MastersRow() {
  const masters = useStore((s) => s.masters);
  const leg: LegName = useStore((s) => s.deck?.preview_source) ?? 'design';
  return (
    <div className="flex items-center gap-5">
      <span
        className={
          'text-[10px] tracking-widest ' +
          (leg === 'live' ? 'text-red-300' : 'text-emerald-300')
        }
        title="these faders drive the leg selected by the deck toggle"
      >
        MASTERS · {leg.toUpperCase()}
      </span>
      {MASTERS.map((def) => (
        <MasterSlider
          key={def.key}
          def={def}
          leg={leg}
          liveValue={masters ? masters[leg]?.[def.key] : undefined}
        />
      ))}
    </div>
  );
}

function MasterSlider({
  def,
  leg,
  liveValue,
}: {
  def: MasterDef;
  leg: LegName;
  liveValue: number | undefined;
}) {
  // Local value wins while (and after) dragging — a stale telemetry frame
  // must never move a fader mid-set.
  const [local, setLocal] = useState<number | null>(null);
  // Switching legs must snap the fader to the newly selected leg's value.
  useEffect(() => setLocal(null), [leg]);
  const shown = local ?? liveValue ?? def.default;
  const isDefault = Math.abs(shown - def.default) < 1e-4;

  function set(v: number) {
    setLocal(v);
    sendMaster(def.key, v, leg);
  }

  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <button
        className={
          'text-[10px] tracking-wider w-12 text-left ' +
          (isDefault ? 'text-zinc-500' : 'text-amber-300')
        }
        title={`double-click to reset to ${def.default}`}
        onDoubleClick={() => set(def.default)}
      >
        {def.label}
      </button>
      <input
        type="range"
        min={def.min}
        max={def.max}
        step={0.005}
        value={shown}
        className="flex-1 accent-amber-400"
        onChange={(e) => set(parseFloat(e.target.value))}
      />
      <span className="w-9 text-right font-mono text-[11px] text-zinc-200">
        {shown.toFixed(2)}
      </span>
    </div>
  );
}
