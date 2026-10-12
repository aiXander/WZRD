// Masters — the operator's final stage, invisible to the AI. Written to both
// legs at once (actions.setMaster) so a TAKE never makes them jump, which is
// why this panel has no leg colour: it's always "the room".
//
// Next to it: what the room hears. Band meters, onset lamps (kick / snare /
// hats), and the AUDIO master that scales how hard the surface listens.

import { useTelemetryDemand } from '../api/telemetryDemand';
import { flash, setMaster, toggleBlackout, toggleFreeze } from '../state/actions';
import { useStore, type Masters } from '../state/store';
import { Fader } from '../ui/Fader';
import { Btn, Kbd, Lamp, Meter } from '../ui/primitives';

type Def = {
  key: keyof Masters;
  label: string;
  max: number;
  def: number;
  color: string;
  fmt: (v: number) => string;
  title: string;
};

const pct = (v: number) => `${Math.round(v * 100)}%`;
const MASTERS: Def[] = [
  {
    key: 'brightness',
    label: 'Master',
    max: 2,
    def: 1,
    color: 'rgb(var(--fg))',
    fmt: pct,
    title: 'Overall light output (0–200%). Double-click: 100%. B = blackout.',
  },
  {
    key: 'saturation',
    label: 'Sat',
    max: 2,
    def: 1,
    color: 'rgb(var(--design))',
    fmt: pct,
    title: 'Colour saturation (0 = white light, 200% = punchy). Double-click: 100%.',
  },
  {
    key: 'speed',
    label: 'Speed',
    max: 4,
    def: 1,
    color: 'rgb(var(--amber))',
    fmt: (v) => `${v.toFixed(2)}×`,
    title: 'Time-stretch every animation (0 = freeze, 4× = rush). Double-click: 1×.',
  },
  {
    key: 'audioListen',
    label: 'Audio',
    max: 1,
    def: 1,
    color: 'rgb(var(--mod))',
    fmt: pct,
    title: 'How hard the surface listens — scales every audio-driven parameter. Double-click: 100%.',
  },
];

export function MastersPanel() {
  const m = useStore((s) => s.masters?.live);
  const dark = (m?.brightness ?? 1) < 0.001;
  const frozen = (m?.speed ?? 1) < 0.0001;
  return (
    <div className="flex flex-col h-full min-h-0 p-3 gap-2.5">
      <div className="flex-1 min-h-[120px] grid grid-cols-4 gap-2">
        {MASTERS.map((d) => (
          <Fader
            key={d.key}
            orientation="v"
            label={d.label}
            value={m?.[d.key] ?? d.def}
            min={0}
            max={d.max}
            defaultValue={d.def}
            color={d.color}
            format={d.fmt}
            title={d.title}
            onChange={(v) => setMaster(d.key, v)}
          />
        ))}
      </div>
      <div className="grid grid-cols-3 gap-1.5">
        <Btn
          tone="danger"
          active={dark}
          onClick={() => toggleBlackout()}
          title="Kill the light (master to 0) — press again to restore. (B)"
        >
          {dark ? 'RESTORE' : 'BLACK'} <Kbd>B</Kbd>
        </Btn>
        <button
          onPointerDown={() => flash(true)}
          onPointerUp={() => flash(false)}
          onPointerLeave={() => flash(false)}
          title="Hold for a burst — master to 200% while pressed (hold F)"
          className="h-7 rounded-md border border-line bg-panel2 text-xs font-medium text-fg2 hover:border-fg3 hover:text-fg active:bg-white active:text-black active:border-white inline-flex items-center justify-center gap-1.5"
        >
          FLASH <Kbd>F</Kbd>
        </button>
        <Btn
          tone="design"
          active={frozen}
          onClick={() => toggleFreeze()}
          title="Freeze every animation (speed 0) — press again to thaw. (Z)"
        >
          FREEZE <Kbd>Z</Kbd>
        </Btn>
      </div>
    </div>
  );
}

export function AudioPanel() {
  const a = useStore((s) => s.audio);
  const fresh = useStore((s) => s.audioFresh?.state);
  const listen = useStore((s) => s.masters?.live.audioListen ?? 1);
  // 30 Hz levels only while the strip can show them (not while the feed is down).
  useTelemetryDemand('audio', fresh !== 'down');
  if (!a || fresh === 'down') {
    return (
      <div className="px-3 py-3 text-[11px] text-fg3 leading-snug">
        No audio. Start the feature server (<span className="num text-fg2">uv run audio-server</span>) — audio-reactive layers rest at their base value until it’s up.
      </div>
    );
  }
  // Meters show the raw room; the dim overlay shows what the AUDIO master
  // lets through to the surface.
  const band = (v: number, c: string, label: string) => (
    <div className="flex flex-col items-center gap-1">
      <div className="relative h-12 flex gap-[2px]">
        <Meter value={v} color={`rgb(var(--${c}) / 0.35)`} className="h-full" />
        <Meter value={v * listen} color={`rgb(var(--${c}))`} className="h-full" />
      </div>
      <div className="text-[9px] font-semibold tracking-wider text-fg3">{label}</div>
    </div>
  );
  return (
    <div className="flex items-end justify-between px-3 py-2.5 gap-2">
      <div className="flex gap-2.5">
        {band(a.band_low, 'low', 'LOW')}
        {band(a.band_mid, 'mid', 'MID')}
        {band(a.band_high, 'high', 'HIGH')}
      </div>
      <div className="flex gap-2 pb-0.5">
        <Lamp value={a.onset_low * listen} color="rgb(var(--low))" label="KICK" />
        <Lamp value={a.onset_mid * listen} color="rgb(var(--mid))" label="SNARE" />
        <Lamp value={a.onset_high * listen} color="rgb(var(--high))" label="HATS" />
      </div>
    </div>
  );
}
