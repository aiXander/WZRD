// The jobs pill (studio plan A2, S15): running render jobs and today's
// spend against the daily cap, in the TopBar of every room. Click: the
// running / finished / failed / lost / cancelled jobs with their args, and
// Cancel for a running one (H2's `/api/jobs/cancel`). Holds the job feed
// only while the popover is open — or while Studio holds it anyway.

import { useRef, useState } from 'react';
import { hubCall, type Job } from '../api/hub';
import { refreshJobs, useHub, useJobsFeed } from '../state/hub';
import { useStore } from '../state/store';
import { Btn } from '../ui/primitives';
import { Popover } from '../ui/Popover';

const TONE: Record<Job['status'], string> = {
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
  const anchor = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  // Outside Studio the pill keeps a cheap long-poll too, so an agent's
  // render shows up in any room.
  useJobsFeed(up && route !== 'studio');
  if (!up) return null;
  const running = jobs.filter((j) => j.status === 'running').length;
  const spendTone = spend?.over ? 'text-bad' : spend?.warn ? 'text-amber' : 'text-fg2';
  return (
    <>
      <button
        ref={anchor}
        onClick={() => setOpen((o) => !o)}
        title={`Render jobs · spend today${spend ? ` $${spend.today_usd.toFixed(2)} of $${spend.cap_usd.toFixed(0)} cap` : ''}`}
        className="flex items-center gap-2 h-8 px-2.5 shrink-0 rounded-lg bg-bg border border-line text-[11px] text-fg2 hover:text-fg"
      >
        {running > 0 ? (
          <span className="text-deck font-semibold">
            <span className="animate-pulse">●</span> {running}
          </span>
        ) : (
          <span className="text-fg3">jobs</span>
        )}
        {spend && <span className={'num ' + spendTone}>${spend.today_usd.toFixed(2)}</span>}
      </button>
      {open && (
        <Popover anchor={anchor.current} onClose={() => setOpen(false)} width={420}>
          <div className="label px-1 pb-1.5 flex items-center">
            Jobs
            <span className="flex-1" />
            {spend && (
              <span className={'normal-case tracking-normal ' + spendTone}>
                ${spend.today_usd.toFixed(2)} of ${spend.cap_usd.toFixed(2)} today
              </span>
            )}
          </div>
          <div className="max-h-[420px] overflow-y-auto flex flex-col gap-1">
            {jobs.length === 0 && <div className="px-1 text-[12px] text-fg3">No jobs yet today.</div>}
            {jobs.map((j) => (
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
        {job.status === 'running' && (
          <Btn
            size="xs"
            tone="danger"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await hubCall('POST', '/api/jobs/cancel', { job: job.job });
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
