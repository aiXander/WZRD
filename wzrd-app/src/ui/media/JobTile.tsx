// A running render shown as a tile in the folder it lands in (studio plan
// A2, reused by MediaGallery): elapsed / ETA, a progress hairline, ✕ cancel.

import { cancelJob, type Job } from '../../api/hub';
import { refreshJobs } from '../../state/hub';
import { useStore } from '../../state/store';

export function JobTile({ job, aspect = '16 / 9' }: { job: Job; aspect?: string }) {
  const elapsed = job.elapsed_s ?? 0;
  const eta = job.eta_s;
  const frac = eta != null && elapsed + eta > 0 ? elapsed / (elapsed + eta) : null;
  return (
    <div
      className="relative rounded-md border border-deck/50 bg-bg overflow-hidden"
      title={`${job.tool} by ${job.by} — ${JSON.stringify(job.args).slice(0, 300)}`}
    >
      <div className="flex flex-col items-center justify-center gap-1" style={{ aspectRatio: aspect }}>
        <span className="text-[11px] text-deck font-semibold animate-pulse">rendering</span>
        <span className="num text-[10px] text-fg3">
          {Math.round(elapsed)} s{eta != null ? ` · ~${Math.round(eta)} s left` : ''}
        </span>
      </div>
      {frac != null && <div className="absolute bottom-0 left-0 h-[2px] bg-deck" style={{ width: `${frac * 100}%` }} />}
      <div className="px-1.5 py-1 text-[10px] text-fg2 flex items-center gap-1 min-w-0">
        <span className="truncate flex-1">
          {job.tool} · {job.by}
        </span>
        <button
          className="text-fg3 hover:text-bad shrink-0"
          title="Cancel this render (no output; Modal / FAL cancel the remote call)"
          onClick={(e) => {
            e.stopPropagation();
            cancelJob(job.job)
              .then(refreshJobs)
              .catch((err) => useStore.getState().toast(`Cancel failed: ${err}`, 'bad'));
          }}
        >
          ✕
        </button>
      </div>
    </div>
  );
}
