// Debug route — vertical stack of collapsible panels.
//
// Phase 4.2 scope: connectivity, render stats, driver bus snapshot,
// hot-reload events, log stream, pack & scene state. Cut-without-touching:
// the route is self-contained and can be gated behind a flag in release.
//
// The page I'll stare at most while iterating on shaders, per the spec.

import { useEffect, useMemo, useState } from 'react';
import { probeGetThresholds, probeSetThresholds, type PerfPayload, type PerfSpike } from '../api/ipc';
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
    <div className="border border-line rounded-lg bg-panel overflow-hidden">
      <button
        className="w-full h-8 px-3 text-left border-b border-line flex items-center gap-2 label hover:text-fg"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="text-fg2">{open ? '▾' : '▸'}</span>
        <span>{title}</span>
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
    <div className="p-2 space-y-2 overflow-auto h-full">
      <Panel title="Connectivity" defaultOpen>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className={`dot w-2 h-2 rounded-full ${statusDot(audioFresh?.state)}`} />
            <span className="text-fg2">OSC audio feed</span>
            <span className="text-fg3">
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
            <span className="text-fg2">file watcher</span>
            <span className="text-fg3">
              {connectivity?.file_watcher.detail ?? '—'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`dot w-2 h-2 rounded-full ${statusDot(connectivity?.ws.status)}`} />
            <span className="text-fg2">WS IPC</span>
            <span className="text-fg3">
              {connectivity?.ws.detail ?? 'tauri ↔ render-core'}
            </span>
          </div>
        </div>
      </Panel>

      <Panel title="Render stats" defaultOpen>
        <div className="grid grid-cols-4 gap-3 text-xs">
          <Stat
            label="mode"
            value={
              frameStats == null
                ? '—'
                : frameStats.presenting === false
                ? 'offscreen (window hidden)'
                : 'presenting'
            }
          />
          <Stat label="FPS" value={fps?.fps.toFixed(0) ?? '—'} />
          <Stat label="frame p50" value={frameStats ? `${frameStats.frame_time_ms_p50.toFixed(2)} ms` : '—'} />
          <Stat label="frame p95" value={frameStats ? `${frameStats.frame_time_ms_p95.toFixed(2)} ms` : '—'} />
          <Stat label="frame p99" value={frameStats ? `${frameStats.frame_time_ms_p99.toFixed(2)} ms` : '—'} />
          <Stat label="mask slices" value={frameStats?.mask_slice_count ?? pack?.layers.length ?? '—'} />
          <Stat label="pipelines" value={frameStats?.pipeline_count ?? '—'} />
          <Stat label="passes" value={frameStats?.pass_count ?? '—'} />
        </div>
      </Panel>

      <Panel title="Performance (frame profiler)" defaultOpen>
        <PerfPanel />
      </Panel>

      <Panel title="Driver bus snapshot" defaultOpen>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-fg3 text-left">
              <th>binding</th>
              <th>param</th>
              <th>source</th>
              <th className="text-right">value</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((d) => (
              <tr key={`${d.binding_id}::${d.param_name}`} className="border-t border-line">
                <td>{d.binding_id}</td>
                <td>{d.param_name}</td>
                <td className="text-fg2">{d.source}</td>
                <td className="text-right font-mono">{d.value.toFixed(3)}</td>
              </tr>
            ))}
            {drivers.length === 0 && (
              <tr>
                <td colSpan={4} className="text-fg3 py-2">
                  no live drivers — wait for the engine to emit `drivers` events
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>

      <Panel title="Hot-reload events" defaultOpen>
        <div className="text-xs flex flex-col divide-y divide-line">
          {hotHistory.map((h, i) => (
            <div key={i} className="py-1 flex items-center gap-2">
              <span
                className={
                  'dot w-2 h-2 rounded-full ' +
                  (h.ok ? 'bg-accent-green' : 'bg-accent-red')
                }
              />
              <span className="font-mono">{h.target}</span>
              <span className="text-fg3">{h.elapsed_ms.toFixed(0)}ms</span>
              {h.probe && (
                <span
                  className={
                    'px-1.5 rounded text-[10px] tracking-wider ' +
                    (h.probe.band === 'green'
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : h.probe.band === 'yellow'
                      ? 'bg-amber-500/20 text-amber-300'
                      : 'bg-red-500/20 text-red-300')
                  }
                  title={h.probe.verdicts
                    .map((v) => `${v.label}: ${v.predicted_p95_ms.toFixed(1)} ms (${v.band})`)
                    .join('\n')}
                >
                  probe {h.probe.band} · {h.probe.predicted_p95_ms.toFixed(1)}ms
                </span>
              )}
              {h.message && (
                <span className="text-accent-red truncate flex-1">
                  {h.message.split('\n')[0]}
                </span>
              )}
            </div>
          ))}
          {hotHistory.length === 0 && (
            <div className="text-fg3">no events yet</div>
          )}
        </div>
      </Panel>

      <Panel title="Pre-flight probe thresholds" defaultOpen>
        <ProbeThresholdsPanel />
      </Panel>

      <Panel title="Log stream">
        <div className="flex items-center gap-2 mb-2 text-xs">
          <span className="text-fg3">filter</span>
          {(['all', 'warn', 'error'] as const).map((lvl) => (
            <button
              key={lvl}
              className={
                'px-2 py-0.5 rounded ' +
                (logLevel === lvl ? 'bg-line2 text-fg' : 'text-fg2')
              }
              onClick={() => setLogLevel(lvl)}
            >
              {lvl}
            </button>
          ))}
        </div>
        <div className="font-mono text-[11px] h-64 overflow-auto bg-bg p-2 rounded">
          {filteredLog.length === 0 && (
            <div className="text-fg3">no log lines</div>
          )}
          {filteredLog.map((l, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-fg3">
                {new Date(l.ts_ms).toISOString().slice(11, 19)}
              </span>
              <span
                className={
                  l.level === 'error'
                    ? 'text-accent-red'
                    : l.level === 'warn'
                    ? 'text-accent-amber'
                    : 'text-fg2'
                }
              >
                {l.level}
              </span>
              <span className="text-fg3">{l.target}</span>
              <span className="text-fg flex-1 truncate">{l.message}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Pack & scene state">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-fg3 mb-1">pack.json</div>
            <pre className="bg-bg p-2 rounded overflow-auto max-h-96 text-[11px]">
              {pack ? JSON.stringify(pack, null, 2) : '—'}
            </pre>
          </div>
          <div>
            <div className="text-fg3 mb-1">scene.json (live)</div>
            <pre className="bg-bg p-2 rounded overflow-auto max-h-96 text-[11px]">
              {sceneJson}
            </pre>
          </div>
        </div>
      </Panel>
    </div>
  );
}

/// §5.6 — venue-level probe thresholds A < B (predicted full-res p95, ms).
/// Persisted engine-side in the session sidecar; frame budget @60 Hz is
/// 16.6 ms. Green < A, yellow A..B (flagged, still enters design), red > B
/// (refused entry).
function ProbeThresholdsPanel() {
  const [a, setA] = useState<string>('');
  const [b, setB] = useState<string>('');
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    probeGetThresholds()
      .then((t) => {
        setA(String(t.a_ms));
        setB(String(t.b_ms));
      })
      .catch((e) => setStatus(String(e)));
  }, []);

  function save() {
    const aMs = parseFloat(a);
    const bMs = parseFloat(b);
    if (!isFinite(aMs) || !isFinite(bMs)) {
      setStatus('thresholds must be numbers');
      return;
    }
    probeSetThresholds(aMs, bMs)
      .then((t) => {
        setA(String(t.a_ms));
        setB(String(t.b_ms));
        setStatus('saved (persists in session.json)');
      })
      .catch((e) => setStatus(String(e)));
  }

  return (
    <div className="flex items-center gap-3 text-xs">
      <label className="flex items-center gap-1.5">
        <span className="text-fg3">A (green below)</span>
        <input
          className="w-16 bg-bg border border-line2 rounded px-1.5 py-0.5 font-mono"
          value={a}
          onChange={(e) => setA(e.target.value)}
        />
        <span className="text-fg3">ms</span>
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-fg3">B (red above)</span>
        <input
          className="w-16 bg-bg border border-line2 rounded px-1.5 py-0.5 font-mono"
          value={b}
          onChange={(e) => setB(e.target.value)}
        />
        <span className="text-fg3">ms</span>
      </label>
      <button
        className="px-2.5 py-0.5 rounded border border-line2 text-fg2 hover:bg-raised"
        onClick={save}
      >
        save
      </button>
      <span className="text-fg3">
        budget 16.6 ms @ 60 Hz · yellow enters design flagged · red is refused
      </span>
      {status && <span className="text-fg2">{status}</span>}
    </div>
  );
}

const ms = (v: number | null | undefined, digits = 2) =>
  v == null ? '—' : `${v.toFixed(digits)} ms`;

/// Perf-monitor PM1 — plain readout of the `perf` channel + the last
/// `perf_spike`. Headroom is the headline (fps can't show it: a 2 ms and a
/// 15 ms frame both read 60). The graphed Perf panel is PM4.
function PerfPanel() {
  const perf = useStore((s) => s.perf);
  const spike = useStore((s) => s.perfSpike);
  if (!perf) {
    return (
      <div className="text-fg3 text-xs">
        waiting for the engine's `perf` channel (2 Hz; off when started with --perf off)
      </div>
    );
  }
  return (
    <div className="text-xs space-y-3">
      <PerfSummary perf={perf} />
      <div className="grid grid-cols-2 gap-4">
        <StageTable perf={perf} />
        <div className="space-y-3">
          <ThreadsTable perf={perf} />
          <ClipsTable perf={perf} />
          <TelemetryBps perf={perf} />
        </div>
      </div>
      <SpikeLine spike={spike} triggerMs={perf.spikes.trigger_ms} />
    </div>
  );
}

function PerfSummary({ perf }: { perf: PerfPayload }) {
  const h = perf.headroom;
  const tint =
    h == null ? 'text-fg3' : h > 0.4 ? 'text-accent-green' : h > 0.15 ? 'text-accent-amber' : 'text-accent-red';
  const res = perf.process.resources;
  return (
    <div className="grid grid-cols-6 gap-3">
      <div className="flex flex-col gap-0.5">
        <div className="text-[10px] text-fg3">headroom</div>
        <div className={`font-mono ${tint}`}>{h == null ? '—' : `${Math.round(h * 100)} %`}</div>
      </div>
      <Stat label="cpu work p50 / p95" value={`${perf.frame.work.p50.toFixed(2)} / ${ms(perf.frame.work.p95)}`} />
      <Stat label={`interval p95 (budget ${perf.budget_ms.toFixed(1)})`} value={ms(perf.frame.interval.p95)} />
      <Stat label="dropped (window / total)" value={`${perf.dropped.window} / ${perf.dropped.total}`} />
      <Stat label="mode" value={perf.presenting ? 'presenting' : 'offscreen'} />
      <Stat label="monitor p95 / unaccounted p95" value={`${perf.frame.monitor.p95.toFixed(3)} / ${ms(perf.frame.unaccounted.p95, 3)}`} />
      <Stat label="knob→submit p95" value={ms(perf.latency.knob_submit_p95, 1)} />
      <Stat label="rpc queue p95" value={ms(perf.latency.rpc_queue_p95, 1)} />
      <Stat label="osc age p50" value={ms(perf.latency.osc_age_p50, 0)} />
      <Stat label="process cpu" value={perf.process.cpu_pct == null ? '—' : `${perf.process.cpu_pct.toFixed(1)} %`} />
      <Stat label="rss / gpu (own)" value={`${perf.process.rss_mb ?? '—'} / ${perf.process.gpu_mb ?? '—'} MB`} />
      <Stat
        label="wgpu live (buf / tex / bg / pipe)"
        value={res ? `${res.buffers} / ${res.textures} / ${res.bind_groups} / ${res.render_pipelines}` : '—'}
      />
    </div>
  );
}

function StageTable({ perf }: { perf: PerfPayload }) {
  const rows = Object.entries(perf.cpu).sort((a, b) => b[1].p95 - a[1].p95);
  return (
    <table className="w-full">
      <PerfHead cols={['stage', 'p50', 'p95', 'max']} />
      <tbody className="font-mono">
        {rows.map(([name, p]) => {
          const wait = name === 'acquire' || name === 'pace';
          return (
            <tr key={name} className={`border-t border-line ${wait ? 'text-fg3' : ''}`}>
              <td>
                {name}
                {wait && ' (wait)'}
              </td>
              <td className="text-right">{p.p50.toFixed(3)}</td>
              <td className="text-right">{p.p95.toFixed(3)}</td>
              <td className="text-right">{p.max.toFixed(3)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ThreadsTable({ perf }: { perf: PerfPayload }) {
  return (
    <table className="w-full">
      <PerfHead cols={['thread', 'cpu % (1 core)']} />
      <tbody className="font-mono">
        {perf.threads.map((t, i) => (
          <tr key={`${t.name}-${i}`} className="border-t border-line">
            <td className="truncate">{t.name}</td>
            <td className="text-right">{t.cpu_pct == null ? '—' : t.cpu_pct.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ClipsTable({ perf }: { perf: PerfPayload }) {
  if (perf.clips.length === 0) {
    return (
      <div className="text-fg3">
        no active clip decoders{perf.clips_idle ? ` (${perf.clips_idle} idle)` : ''}
      </div>
    );
  }
  return (
    <table className="w-full">
      <PerfHead cols={['clip', 'decode p95', 'stale (0.5 s / total)', 'queue']} />
      <tbody className="font-mono">
        {perf.clips.map((c, i) => (
          <tr key={`${c.label}-${i}`} className="border-t border-line">
            <td className="truncate">{c.label}</td>
            <td className="text-right">{ms(c.decode_p95)}</td>
            <td className={`text-right ${c.stale > 0 ? 'text-accent-red' : ''}`}>
              {c.stale} / {c.stale_total}
            </td>
            <td className="text-right">{c.queue}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PerfHead({ cols }: { cols: string[] }) {
  return (
    <thead>
      <tr className="text-fg3 text-left">
        {cols.map((c, i) => (
          <th key={c} className={i > 0 ? 'text-right' : undefined}>
            {c}
          </th>
        ))}
      </tr>
    </thead>
  );
}

function TelemetryBps({ perf }: { perf: PerfPayload }) {
  const fmt = (m: Record<string, number>) =>
    Object.entries(m)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k} ${(v / 1024).toFixed(1)}k`)
      .join(' · ') || '—';
  return (
    <div className="space-y-0.5">
      <div>
        <span className="text-fg3">telemetry → webview </span>
        <span className="font-mono">{fmt(perf.telemetry_bps.webview)}</span>
      </div>
      <div>
        <span className="text-fg3">telemetry → ws </span>
        <span className="font-mono">{fmt(perf.telemetry_bps.ws)}</span>
      </div>
    </div>
  );
}

function SpikeLine({ spike, triggerMs }: { spike: PerfSpike | null; triggerMs: number }) {
  if (!spike) {
    return <div className="text-fg3">no spike over {triggerMs.toFixed(1)} ms captured</div>;
  }
  const hit = spike.frames.find((f) => f.offset === 0);
  return (
    <div className="border-t border-line pt-2">
      <span className="text-accent-amber">last spike</span>{' '}
      <span className="text-fg3">{new Date(spike.ts_ms).toISOString().slice(11, 19)}</span>{' '}
      <span className="font-mono">
        {spike.frame_ms.toFixed(1)} ms (trigger {spike.trigger_ms.toFixed(1)}) — worst {spike.worst_stage}{' '}
        {spike.worst_stage_ms.toFixed(1)} ms
      </span>
      {hit && (
        <span className="text-fg2">
          {' '}
          · ran: {hit.ran.cmds.length ? hit.ran.cmds.join(', ') : 'no cmds'}, reloads {hit.ran.reloads},
          probe {hit.ran.probe_steps}, clip opens {hit.ran.clip_opens}, rev {hit.ran.rev}
        </span>
      )}
      {spike.suppressed > 0 && <span className="text-fg3"> · {spike.suppressed} suppressed</span>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[10px] text-fg3">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}
