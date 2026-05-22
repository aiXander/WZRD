// Debug route — vertical stack of collapsible panels.
//
// Phase 4.2 scope: connectivity, render stats, driver bus snapshot,
// hot-reload events, log stream, pack & scene state. Cut-without-touching:
// the route is self-contained and can be gated behind a flag in release.
//
// The page I'll stare at most while iterating on shaders, per the spec.

import { useMemo, useState } from 'react';
import { useStore } from '../state/store';

function Panel({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-ink-700 rounded">
      <button
        className="w-full px-3 py-1.5 text-xs text-left bg-ink-800 border-b border-ink-700 flex items-center gap-2"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="text-zinc-400">{open ? '▾' : '▸'}</span>
        <span className="text-zinc-200">{title}</span>
      </button>
      {open && <div className="p-3">{children}</div>}
    </div>
  );
}

function statusDot(s: string | undefined) {
  if (s === 'ok' || s === 'fresh') return 'bg-accent-green';
  if (s === 'warn' || s === 'stale') return 'bg-accent-amber';
  return 'bg-accent-red';
}

export function DebugRoute() {
  const connectivity = useStore((s) => s.connectivity);
  const frameStats = useStore((s) => s.frameStats);
  const fps = useStore((s) => s.fps);
  const drivers = useStore((s) => s.drivers);
  const hotHistory = useStore((s) => s.hotReloadHistory);
  const log = useStore((s) => s.log);
  const pack = useStore((s) => s.pack);
  const sceneJson = useStore((s) => s.sceneJson);
  const audioFresh = useStore((s) => s.audioFresh);

  const [logLevel, setLogLevel] = useState<'all' | 'warn' | 'error'>('all');
  const filteredLog = useMemo(
    () =>
      log.filter((l) => {
        if (logLevel === 'all') return true;
        if (logLevel === 'warn')
          return l.level === 'warn' || l.level === 'error';
        return l.level === 'error';
      }),
    [log, logLevel]
  );

  return (
    <div className="p-4 space-y-3 overflow-auto h-full">
      <Panel title="Connectivity" defaultOpen>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className={`dot w-2 h-2 rounded-full ${statusDot(audioFresh?.state)}`} />
            <span className="text-zinc-300">OSC audio feed</span>
            <span className="text-zinc-500">
              {audioFresh?.state ?? '—'}
              {audioFresh?.last_packet_ms
                ? ` · last ${(audioFresh.last_packet_ms / 1000).toFixed(1)}s`
                : ''}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`dot w-2 h-2 rounded-full ${statusDot(
                connectivity?.file_watcher.status
              )}`}
            />
            <span className="text-zinc-300">file watcher</span>
            <span className="text-zinc-500">
              {connectivity?.file_watcher.detail ?? '—'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`dot w-2 h-2 rounded-full ${statusDot(connectivity?.ws.status)}`} />
            <span className="text-zinc-300">WS IPC</span>
            <span className="text-zinc-500">
              {connectivity?.ws.detail ?? 'tauri ↔ render-core'}
            </span>
          </div>
        </div>
      </Panel>

      <Panel title="Render stats" defaultOpen>
        <div className="grid grid-cols-4 gap-3 text-xs">
          <Stat label="FPS" value={fps?.fps.toFixed(0) ?? '—'} />
          <Stat label="frame p50" value={frameStats ? `${frameStats.frame_time_ms_p50.toFixed(2)} ms` : '—'} />
          <Stat label="frame p95" value={frameStats ? `${frameStats.frame_time_ms_p95.toFixed(2)} ms` : '—'} />
          <Stat label="frame p99" value={frameStats ? `${frameStats.frame_time_ms_p99.toFixed(2)} ms` : '—'} />
          <Stat label="mask slices" value={frameStats?.mask_slice_count ?? pack?.layers.length ?? '—'} />
          <Stat label="pipelines" value={frameStats?.pipeline_count ?? '—'} />
          <Stat label="passes" value={frameStats?.pass_count ?? '—'} />
        </div>
      </Panel>

      <Panel title="Driver bus snapshot" defaultOpen>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-zinc-500 text-left">
              <th>binding</th>
              <th>param</th>
              <th>source</th>
              <th className="text-right">value</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((d) => (
              <tr key={`${d.binding_id}::${d.param_name}`} className="border-t border-ink-700">
                <td>{d.binding_id}</td>
                <td>{d.param_name}</td>
                <td className="text-zinc-400">{d.source}</td>
                <td className="text-right font-mono">{d.value.toFixed(3)}</td>
              </tr>
            ))}
            {drivers.length === 0 && (
              <tr>
                <td colSpan={4} className="text-zinc-500 py-2">
                  no live drivers — wait for the engine to emit `drivers` events
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>

      <Panel title="Hot-reload events" defaultOpen>
        <div className="text-xs flex flex-col divide-y divide-ink-700">
          {hotHistory.map((h, i) => (
            <div key={i} className="py-1 flex items-center gap-2">
              <span
                className={
                  'dot w-2 h-2 rounded-full ' +
                  (h.ok ? 'bg-accent-green' : 'bg-accent-red')
                }
              />
              <span className="font-mono">{h.target}</span>
              <span className="text-zinc-500">{h.elapsed_ms.toFixed(0)}ms</span>
              {h.message && (
                <span className="text-accent-red truncate flex-1">
                  {h.message.split('\n')[0]}
                </span>
              )}
            </div>
          ))}
          {hotHistory.length === 0 && (
            <div className="text-zinc-500">no events yet</div>
          )}
        </div>
      </Panel>

      <Panel title="Log stream">
        <div className="flex items-center gap-2 mb-2 text-xs">
          <span className="text-zinc-500">filter</span>
          {(['all', 'warn', 'error'] as const).map((lvl) => (
            <button
              key={lvl}
              className={
                'px-2 py-0.5 rounded ' +
                (logLevel === lvl ? 'bg-ink-600 text-zinc-100' : 'text-zinc-400')
              }
              onClick={() => setLogLevel(lvl)}
            >
              {lvl}
            </button>
          ))}
        </div>
        <div className="font-mono text-[11px] h-64 overflow-auto bg-ink-900 p-2 rounded">
          {filteredLog.length === 0 && (
            <div className="text-zinc-500">no log lines</div>
          )}
          {filteredLog.map((l, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-zinc-500">
                {new Date(l.ts_ms).toISOString().slice(11, 19)}
              </span>
              <span
                className={
                  l.level === 'error'
                    ? 'text-accent-red'
                    : l.level === 'warn'
                    ? 'text-accent-amber'
                    : 'text-zinc-300'
                }
              >
                {l.level}
              </span>
              <span className="text-zinc-500">{l.target}</span>
              <span className="text-zinc-200 flex-1 truncate">{l.message}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Pack & scene state">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-zinc-500 mb-1">pack.json</div>
            <pre className="bg-ink-900 p-2 rounded overflow-auto max-h-96 text-[11px]">
              {pack ? JSON.stringify(pack, null, 2) : '—'}
            </pre>
          </div>
          <div>
            <div className="text-zinc-500 mb-1">scene.json (live)</div>
            <pre className="bg-ink-900 p-2 rounded overflow-auto max-h-96 text-[11px]">
              {sceneJson}
            </pre>
          </div>
        </div>
      </Panel>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[10px] text-zinc-500">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}
