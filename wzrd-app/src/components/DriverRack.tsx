// Driver rack — single scrollable list of every driver-bound param in the
// active scene. Per row: `binding_id · param_name`, source pill, live value
// bar, inline control if applicable. Phase 4.2 ships read-only-with-live
// values; param mutation goes through the binding inspector for now.
//
// The list is built from two sources:
//   - the *engine-emitted* `drivers` telemetry channel (authoritative live
//     values, includes the `affects` chip);
//   - the parsed scene.json (covers params not currently being driven, so a
//     `const(0.4)` row still shows).

import { useMemo } from 'react';
import { useStore, type DriverRow } from '../state/store';

type Row = DriverRow & { fromScene?: boolean };

function describeSource(source: string): string {
  return source;
}

function valueBar(v: number) {
  const pct = Math.min(100, Math.max(0, v * 100));
  return (
    <div className="h-1 w-full bg-ink-700 rounded overflow-hidden">
      <div
        className="h-full bg-accent-violet"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function DriverRack() {
  const sceneJson = useStore((s) => s.sceneJson);
  const drivers = useStore((s) => s.drivers);

  const fromScene = useMemo<Row[]>(() => {
    try {
      const scene = JSON.parse(sceneJson);
      const out: Row[] = [];
      for (const b of scene.bindings ?? []) {
        for (const [name, v] of Object.entries(b.params ?? {})) {
          let source = 'const';
          let value = 0;
          if (typeof v === 'number') {
            source = `const(${v})`;
            value = v;
          } else if (v && typeof v === 'object' && (v as any).driver) {
            source = (v as any).driver;
          } else if (typeof v === 'string') {
            source = 'color';
          }
          out.push({
            binding_id: b.id,
            param_name: name,
            source,
            value,
            affects: 0,
            fromScene: true,
          });
        }
      }
      return out;
    } catch {
      return [];
    }
  }, [sceneJson]);

  // Merge: any row present in `drivers` overrides the scene one (it has live values).
  const merged: Row[] = useMemo(() => {
    const liveKey = (r: { binding_id: string; param_name: string }) =>
      `${r.binding_id}::${r.param_name}`;
    const liveMap = new Map(drivers.map((d) => [liveKey(d), d]));
    return fromScene.map((r) => {
      const live = liveMap.get(liveKey(r));
      return live ? { ...live, fromScene: false } : r;
    });
  }, [drivers, fromScene]);

  return (
    <div className="flex flex-col gap-2">
      <header className="text-xs text-zinc-500">
        Drivers · {merged.length} param{merged.length === 1 ? '' : 's'}
      </header>
      <div className="grid grid-cols-[1fr_1fr_2fr_auto] gap-x-3 gap-y-2 text-xs">
        <div className="text-zinc-500">binding · param</div>
        <div className="text-zinc-500">source</div>
        <div className="text-zinc-500">value</div>
        <div className="text-zinc-500 text-right">affects</div>
        {merged.map((r) => (
          <RowView key={`${r.binding_id}:${r.param_name}`} row={r} />
        ))}
      </div>
    </div>
  );
}

function RowView({ row }: { row: Row }) {
  return (
    <>
      <div className="truncate">
        <span className="text-zinc-100">{row.binding_id}</span>
        <span className="text-zinc-500"> · {row.param_name}</span>
      </div>
      <div className="truncate text-zinc-300">{describeSource(row.source)}</div>
      <div className="flex items-center gap-2">
        <div className="w-12 text-right font-mono text-zinc-300">
          {row.value.toFixed(2)}
        </div>
        <div className="flex-1">{valueBar(row.value)}</div>
      </div>
      <div className="text-right text-zinc-400">
        {row.affects > 0 ? `→ ${row.affects}` : '—'}
      </div>
    </>
  );
}
