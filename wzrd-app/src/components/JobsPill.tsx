// The jobs pill (studio plan A2, S15; render queue plan U3): running /
// queued renders per TextureFlow backend (M = Modal, L = the desktop GPU;
// other tools count under Modal's side as "other"), a thin aggregate
// progress bar along its bottom, and today's spend against the daily cap,
// in the TopBar of every room. Click: the render queue's two lanes
// (`studio/RenderQueue.tsx`, compact) + "Open in Studio", then the recent
// jobs with their args, and Cancel for a running one (H2's
// `/api/jobs/cancel`). Holds the job feed outside Studio (Studio holds it
// anyway); the queue poll runs only while the popover is open.

import { useRef, useState } from 'react';
import { cancelJob, type Job } from '../api/hub';
import { refreshJobs, useHub, useJobsFeed } from '../state/hub';
import { aggregatePct, jobBackend, setQueueOpen, useQueue } from '../state/renderQueue';
import { useStore } from '../state/store';
import { RenderQueue } from './studio/RenderQueue';
import { Btn } from '../ui/primitives';
import { Popover } from '../ui/Popover';

const TONE: Record<Job['status'], string> = {
  queued: 'text-fg2',
  running: 'text-deck',
  done: 'text-ok',
  failed: 'text-bad',
  lost: 'text-amber',
  cancelled: 'text-fg3',
};

export function JobsPill() {
  const up = useHub((s) => s.status?.state === 'up');
  const jobs = useHub((s) => s.jobs);
  const spend = useHub((s) => s.spend);
  const route = useStore((s) => s.route);
  const queueOpen = useQueue((s) => s.open);
  const anchor = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  // Outside Studio the pill keeps a cheap long-poll too, so an agent's
  // render shows up in any room.
  useJobsFeed(up && route !== 'studio');
  if (!up) return null;
  const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued');
  const count = (b: 'modal' | 'local', st: Job['status']) =>
    active.filter((j) => j.tool === 'texture_flow' && jobBackend(j) === b && j.status === st).length;
  const lanes = (['modal', 'local'] as const)
    .map((b) => ({ b, run: count(b, 'running'), q: count(b, 'queued') }))
    .filter((l) => l.run || l.q);
  const other = active.filter((j) => j.tool !== 'texture_flow' && j.status === 'running').length;
  const pct = aggregatePct(active);
  // TextureFlow jobs in a lane show as cards above; the list is everything else.
  const history = jobs.filter((j) => !(j.tool === 'texture_flow' && (j.status === 'running' || j.status === 'queued')));
  const spendTone = spend?.over ? 'text-bad' : spend?.warn ? 'text-amber' : 'text-fg2';
  return (
    <>
      <button
        ref={anchor}
        onClick={() => setOpen((o) => !o)}
        title={`Render jobs · spend today${spend ? ` $${spend.today_usd.toFixed(2)} of $${spend.cap_usd.toFixed(0)} cap` : ''}`}
        className="relative overflow-hidden flex items-center gap-2 h-8 px-2.5 shrink-0 rounded-lg bg-bg border border-line text-[11px] text-fg2 hover:text-fg"
      >
        {lanes.length || other ? (
          <>
            {lanes.map((l) => (
              <span
                key={l.b}
                className="num flex items-center gap-1"
                title={`${l.b === 'modal' ? 'Modal' : 'Local (desktop GPU)'}: ${l.run} running, ${l.q} queued`}
              >
                <span className="text-fg3 font-semibold">{l.b === 'modal' ? 'M' : 'L'}</span>
                <span className={l.run ? 'text-deck font-semibold' : 'text-fg3'}>
                  {l.run ? <span className="animate-pulse">●</span> : null}
                  {l.run}
                </span>
                {l.q > 0 && <span className="text-fg3">+{l.q}</span>}
              </span>
            ))}
            {other > 0 && (
              <span className="num text-deck font-semibold" title={`${other} other render${other === 1 ? '' : 's'} running`}>
                <span className="animate-pulse">●</span> {other}
              </span>
            )}
          </>
        ) : (
          <span className="text-fg3">jobs</span>
        )}
        {spend && <span className={'num ' + spendTone}>${spend.today_usd.toFixed(2)}</span>}
        {pct != null && (
          <span className="absolute left-0 bottom-0 h-[2px] bg-deck transition-[width] duration-700" style={{ width: `${pct * 100}%` }} />
        )}
      </button>
      {open && (
        <Popover anchor={anchor.current} onClose={() => setOpen(false)} width={640}>
          {/* font-sans: the portal lands in <body>, which is font-mono. */}
          <div className="h-[280px] flex flex-col mb-2 font-sans">
            <RenderQueue compact />
          </div>
          {route !== 'studio' || !queueOpen ? (
            <div className="flex justify-end -mt-1 mb-2">
              <Btn
                size="xs"
                onClick={() => {
                  setQueueOpen(true);
                  useStore.getState().setRoute('studio');
                  setOpen(false);
                }}
                title="Dock the queue under Studio's grid"
              >
                Open in Studio
              </Btn>
            </div>
          ) : null}
          <div className="label px-1 pb-1.5 flex items-center border-t border-line pt-2">
            Recent jobs
            <span className="flex-1" />
            {spend && (
              <span className={'normal-case tracking-normal ' + spendTone}>
                ${spend.today_usd.toFixed(2)} of ${spend.cap_usd.toFixed(2)} today
              </span>
            )}
          </div>
          <div className="max-h-[260px] overflow-y-auto flex flex-col gap-1">
            {history.length === 0 && <div className="px-1 text-[12px] text-fg3">No other jobs yet today.</div>}
            {history.map((j) => (
              <JobRow key={j.job} job={j} />
            ))}
          </div>
        </Popover>
      )}
    </>
  );
}

function JobRow({ job }: { job: Job }) {
  const [busy, setBusy] = useState(false);
  const when = new Date(job.started * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return (
    <div className="px-1.5 py-1 rounded border border-line">
      <div className="flex items-center gap-2 text-[12px]">
        <span className={'font-semibold ' + TONE[job.status]}>{job.status}</span>
        <span className="text-fg">{job.tool}</span>
        <span className="text-fg3 text-[10px]">
          {job.by} · {when}
        </span>
        <span className="flex-1" />
        {job.status === 'running' && (
          <span className="num text-[10px] text-fg3">
            {Math.round(job.elapsed_s ?? 0)} s{job.eta_s != null ? ` / ~${Math.round(job.eta_s)} s left` : ''}
          </span>
        )}
        {job.est_usd > 0 && <span className="num text-[10px] text-fg3">${job.est_usd.toFixed(2)}</span>}
        {(job.status === 'running' || job.status === 'queued') && (
          <Btn
            size="xs"
            tone="danger"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await cancelJob(job.job);
                refreshJobs();
              } catch (e) {
                useStore.getState().toast(`Cancel failed: ${e}`, 'bad');
              } finally {
                setBusy(false);
              }
            }}
          >
            Cancel
          </Btn>
        )}
      </div>
      <div className="text-[10px] text-fg3 truncate" title={JSON.stringify(job.args, null, 1)}>
        {Object.entries(job.args)
          .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
          .join(' · ')}
      </div>
      {job.error && <div className="text-[10px] text-bad truncate" title={job.error}>{job.error}</div>}
    </div>
  );
}
