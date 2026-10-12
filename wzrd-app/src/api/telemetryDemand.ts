// Per-channel telemetry demand. The streaming channels — `preview`,
// `audio`, `transport`, `drivers`, `frame_stats`, `perf` — reach the webview
// only while something on screen shows them; the engine skips the work
// behind an undemanded channel entirely (JPEG readback + design-leg render
// for `preview`, payload building for the rest). Sticky channels and `log`
// always flow.
//
// Components hold demand with `useTelemetryDemand(channel, active)`. Counts
// live here; the shell gets the *union* (`telemetry_demand {channels}`),
// coalesced to one call per microtask and only when it changes — so a page
// reload that skipped unmount cleanups can't leak demand, and a room switch
// that unmounts + remounts the same consumers sends nothing.

import { useEffect } from 'react';
import { call } from './transport';

export type DemandChannel = 'preview' | 'audio' | 'transport' | 'drivers' | 'frame_stats' | 'perf';

const counts = new Map<DemandChannel, number>();
/** Last union the shell accepted (`null` = never sent: always send once). */
let sent: string | null = null;
let queued = false;

function flush() {
  queued = false;
  const channels = [...counts].filter(([, n]) => n > 0).map(([c]) => c).sort();
  const key = channels.join(',');
  if (key === sent) return;
  sent = key;
  call<void>('telemetry_demand', { channels }).catch(() => {
    sent = null; // retry with the next change
  });
}

function schedule() {
  if (queued) return;
  queued = true;
  queueMicrotask(flush);
}

/** Take (`on`) or release one unit of demand on `channel`. */
export function demandChannel(channel: DemandChannel, on: boolean) {
  counts.set(channel, Math.max(0, (counts.get(channel) ?? 0) + (on ? 1 : -1)));
  schedule();
}

/** Hold demand on `channel` while mounted and `active`. */
export function useTelemetryDemand(channel: DemandChannel, active = true) {
  useEffect(() => {
    if (!active) return;
    demandChannel(channel, true);
    return () => demandChannel(channel, false);
  }, [channel, active]);
}

// Boot: replace whatever a previous page (pre-reload) left demanded.
schedule();
