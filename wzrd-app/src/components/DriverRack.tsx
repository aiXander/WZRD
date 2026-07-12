// Driver rack — every param in the active scene, playable.
//
// Row control depends on what the param is:
//   - `ui.slider` driver  → live slider → `param.set {name}` RPC (no
//     rebuild, next frame). Scene-authored shared knobs.
//   - literal number      → live slider → §5.5 `param.set {binding, param}`
//     override (no rebuild; persisted in the session sidecar, never written
//     into scene.json implicitly — ↺ clears back to the scene value).
//   - colour string       → colour picker → debounced scene commit.
//   - clock./audio. driver→ read-only live value bar (the source drives it).
//
// Live values arrive on the `drivers` telemetry channel (~10 Hz) — since
// §5.6 the engine emits the rows of whichever leg the deck toggle selects,
// and every knob write here targets that same leg (full-control-switch:
// tuning DESIGN never touches the show; PROMOTE copies the tuning live).

import { useMemo, useRef, useState } from 'react';
import { paramOverride, paramSet, type LegName } from '../api/ipc';
import { commitSceneMutation } from '../state/sceneCommit';
import { useStore, type DriverRow } from '../state/store';

/** The leg every rack write targets — the deck toggle's position. */
function controlLeg(): LegName {
  return useStore.getState().deck?.preview_source ?? 'design';
}

// Trailing throttle so slider drags don't flood the IPC bridge.
const throttleTimers: Record<string, number> = {};
function sendParam(name: string, value: number) {
  if (throttleTimers[name] != null) window.clearTimeout(throttleTimers[name]);
  throttleTimers[name] = window.setTimeout(() => {
    delete throttleTimers[name];
    paramSet(name, value, controlLeg()).catch((e) => console.warn('param.set', name, e));
  }, 25);
}

function sendOverride(binding: string, param: string, value: number) {
  const key = `${binding}::${param}`;
  if (throttleTimers[key] != null) window.clearTimeout(throttleTimers[key]);
  throttleTimers[key] = window.setTimeout(() => {
    delete throttleTimers[key];
    paramOverride(binding, param, value, controlLeg()).catch((e) =>
      console.warn('param.set override', key, e)
    );
  }, 25);
}

type SceneParam = {
  binding_id: string;
  param_name: string;
  raw: any; // the JSON value in scene.json
};

type Row = {
  key: string;
  binding_id: string;
  param_name: string;
  raw: any;
  live: DriverRow | null;
};

export function DriverRack() {
  const sceneJson = useStore((s) => s.sceneJson);
  const drivers = useStore((s) => s.drivers);
  const selectedBinding = useStore((s) => s.selectedBindingId);

  const sceneParams = useMemo<SceneParam[]>(() => {
    try {
      const scene = JSON.parse(sceneJson);
      const out: SceneParam[] = [];
      for (const b of scene.bindings ?? []) {
        for (const [name, v] of Object.entries(b.params ?? {})) {
          out.push({ binding_id: b.id, param_name: name, raw: v });
        }
      }
      return out;
    } catch {
      return [];
    }
  }, [sceneJson]);

  const rows = useMemo<Row[]>(() => {
    const liveMap = new Map(
      drivers.map((d) => [`${d.binding_id}::${d.param_name}`, d])
    );
    return sceneParams.map((p) => {
      const key = `${p.binding_id}::${p.param_name}`;
      return { key, ...p, live: liveMap.get(key) ?? null };
    });
  }, [sceneParams, drivers]);

  if (rows.length === 0) {
    return (
      <div className="text-xs text-zinc-500">
        no params in the current scene — add a binding in Prepare (⌘1)
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <header className="text-xs text-zinc-500 pb-1">
        Drivers · {rows.length} param{rows.length === 1 ? '' : 's'}
      </header>
      <div className="flex flex-col divide-y divide-ink-700/60">
        {rows.map((r) => (
          <RowView key={r.key} row={r} highlight={r.binding_id === selectedBinding} />
        ))}
      </div>
    </div>
  );
}

function RowView({ row, highlight }: { row: Row; highlight: boolean }) {
  const { raw } = row;
  const isDriver = raw && typeof raw === 'object' && 'driver' in raw;
  const driverKind: string | null = isDriver ? raw.driver : null;

  let control: React.ReactNode;
  let sourceLabel: string;

  if (driverKind === 'ui.slider') {
    sourceLabel = `ui.slider(${raw.name ?? row.param_name})`;
    control = (
      <UiSliderControl
        sliderName={raw.name ?? row.param_name}
        defaultValue={raw.default ?? 0}
        liveValue={row.live?.value}
      />
    );
  } else if (isDriver) {
    sourceLabel = row.live?.source ?? driverKind ?? '?';
    control = <LiveBar value={row.live?.value ?? 0} />;
  } else if (typeof raw === 'number') {
    sourceLabel = row.live?.overridden ? 'const · override' : 'const';
    control = (
      <ConstControl
        bindingId={row.binding_id}
        paramName={row.param_name}
        value={raw}
        liveValue={row.live?.value}
        overridden={!!row.live?.overridden}
      />
    );
  } else if (typeof raw === 'string') {
    sourceLabel = 'color';
    control = (
      <ColorControl
        bindingId={row.binding_id}
        paramName={row.param_name}
        value={raw}
      />
    );
  } else {
    sourceLabel = 'unknown';
    control = <span className="text-zinc-500 text-xs">—</span>;
  }

  return (
    <div
      className={
        'grid grid-cols-[16rem_11rem_1fr_4rem] items-center gap-x-3 py-1.5 text-xs ' +
        (highlight ? 'bg-ink-800/60 rounded' : '')
      }
    >
      <div className="truncate">
        <span className="text-zinc-100">{row.binding_id}</span>
        <span className="text-zinc-500"> · {row.param_name}</span>
      </div>
      <div className="truncate text-zinc-400">{sourceLabel}</div>
      <div className="min-w-0">{control}</div>
      <div className="text-right text-zinc-500">
        {row.live && row.live.affects > 0 ? `→ ${row.live.affects}` : ''}
      </div>
    </div>
  );
}

/** Live, zero-rebuild knob: param.set → engine picks it up next frame. */
function UiSliderControl({
  sliderName,
  defaultValue,
  liveValue,
}: {
  sliderName: string;
  defaultValue: number;
  liveValue: number | undefined;
}) {
  // Local value wins while (and after) dragging so the knob never jumps
  // under the finger when a stale telemetry frame arrives.
  const [local, setLocal] = useState<number | null>(null);
  const shown = local ?? liveValue ?? defaultValue;
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={0}
        max={1}
        step={0.005}
        value={shown}
        className="flex-1 accent-violet-400"
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          setLocal(v);
          sendParam(sliderName, v);
        }}
      />
      <span className="w-10 text-right font-mono text-zinc-200">
        {shown.toFixed(2)}
      </span>
    </div>
  );
}

/**
 * Literal scene value: live §5.5 override (zero rebuild, session-persisted).
 * The scene value stays untouched — ↺ clears the override so the param
 * falls back to it next frame. "Write knobs back into scene.json" remains a
 * separate, explicit authoring action (Monaco / binding inspector).
 */
function ConstControl({
  bindingId,
  paramName,
  value,
  liveValue,
  overridden,
}: {
  bindingId: string;
  paramName: string;
  value: number;
  liveValue: number | undefined;
  overridden: boolean;
}) {
  // Local value wins while (and after) dragging so the knob never jumps
  // under the finger when a stale telemetry frame arrives.
  const [local, setLocal] = useState<number | null>(null);
  const shown = local ?? (overridden ? liveValue ?? value : value);

  // Slider bounds adapt to the value's magnitude so both a 0..1 intensity
  // and a freq=18 stay draggable. Bounds are sticky per mount so the scale
  // doesn't warp mid-drag.
  const boundsRef = useRef<{ min: number; max: number } | null>(null);
  if (!boundsRef.current) {
    const mag = Math.abs(value);
    const max = mag <= 1 ? 1 : Math.ceil(mag * 2);
    boundsRef.current = { min: value < 0 ? -max : 0, max };
  }
  const { min, max } = boundsRef.current;

  function commit(v: number) {
    if (!Number.isFinite(v)) return;
    setLocal(v);
    sendOverride(bindingId, paramName, v);
  }

  function reset() {
    setLocal(null);
    paramOverride(bindingId, paramName, null, controlLeg()).catch((e) =>
      console.warn('param.set clear', bindingId, paramName, e)
    );
  }

  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 200}
        value={shown}
        className={'flex-1 ' + (overridden || local != null ? 'accent-amber-400' : 'accent-violet-400')}
        onChange={(e) => commit(parseFloat(e.target.value))}
      />
      <input
        type="number"
        step="0.01"
        value={shown}
        className="w-16 bg-ink-900 px-1 py-0.5 rounded border border-ink-600 font-mono text-right"
        onChange={(e) => commit(parseFloat(e.target.value))}
      />
      <button
        className={
          'w-5 text-center ' +
          (overridden || local != null
            ? 'text-amber-300 hover:text-amber-100'
            : 'text-transparent pointer-events-none')
        }
        title={`clear override (back to scene value ${value})`}
        onClick={reset}
      >
        ↺
      </button>
    </div>
  );
}

function ColorControl({
  bindingId,
  paramName,
  value,
}: {
  bindingId: string;
  paramName: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="color"
        value={normalizeHex(value)}
        className="h-6 w-10 bg-transparent cursor-pointer"
        onChange={(e) =>
          commitSceneMutation((scene) => {
            const b = (scene.bindings ?? []).find((b: any) => b.id === bindingId);
            if (b) b.params = { ...(b.params ?? {}), [paramName]: e.target.value };
            return scene;
          })
        }
      />
      <span className="font-mono text-zinc-400">{value}</span>
    </div>
  );
}

/** Read-only bar for clock./audio.-driven params (the source drives them). */
function LiveBar({ value }: { value: number }) {
  const inUnit = value >= 0 && value <= 1;
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 bg-ink-700 rounded overflow-hidden">
        {inUnit && (
          <div className="h-full bg-accent-violet" style={{ width: `${pct}%` }} />
        )}
      </div>
      <span className="w-12 text-right font-mono text-zinc-300">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function normalizeHex(s: string): string {
  // <input type=color> only accepts #rrggbb.
  const raw = s.startsWith('#') ? s.slice(1) : s;
  if (raw.length === 3) {
    return '#' + raw.split('').map((c) => c + c).join('');
  }
  if (raw.length >= 6) return '#' + raw.slice(0, 6);
  return '#ffffff';
}
