// The TextureFlow render queue (render queue plan, lane U item 2): one lane
// per backend — Modal (a few renders at once) and Local (the desktop GPU,
// one at a time) — side by side, or one of them (Both | Modal | Local,
// remembered per machine). Docked under Studio's grid and shown in the jobs
// pill's popover (`compact`).
//
// The local lane also shows other clients' jobs on the desktop (a lab
// agent's batch: `foreign`), read-only.
//
// A card per job: what it is (recipe / prompt / first style image; an
// Upscale shows its draft's thumb), a Draft / Upscale / Full badge, phase +
// stage ("sampling 7/12"), a progress bar (indeterminate without a pct,
// striped + "≈" when the hub only estimates) and the ETA. Queued cards drag
// to reorder (pointer events, so it works without HTML5 drag-and-drop), ⤒ to
// the top, ⇄ to the other lane (not a finish: it's bound to its draft's
// backend), ✕ cancel. Lane header: the backend's lamp + state, counts, time
// to clear, and Start while the desktop's job service is down; the Modal
// lane's header holds the parallel-GPUs stepper (`ModalSlots`, also in the
// Render panel) — hidden on a hub without `POST /api/tf/modal`.

import { useEffect, useRef, useState } from 'react';
import { listRecipes, mediaUrl, type ForeignJob, type Job, type QueueEntry, type TfBackend } from '../../api/hub';
import {
  BACKENDS,
  BACKEND_LABEL,
  backendLook,
  cancelQueued,
  entryId,
  fmtEta,
  isForeign,
  isQueued,
  jobBackend,
  jobKind,
  jobPct,
  laneEta,
  moveToOtherLane,
  phaseText,
  reorderLane,
  setModalSlots,
  startLocal,
  toTop,
  useLanes,
  useQueue,
  useRenderQueue,
  type JobKind,
} from '../../state/renderQueue';
import { useStudio } from '../../state/studio';
import { Btn, Dot, Seg } from '../../ui/primitives';

type Show = 'both' | TfBackend;
const SHOW_KEY = 'wzrd.queue.lanes';

function readShow(): Show {
  try {
    const v = localStorage.getItem(SHOW_KEY);
    return v === 'modal' || v === 'local' ? v : 'both';
  } catch {
    return 'both';
  }
}

export function RenderQueue({ compact = false, onClose }: { compact?: boolean; onClose?: () => void }) {
  const [show, setShowState] = useState<Show>(readShow);
  const lanes = useLanes();
  useRenderQueue();
  const setShow = (v: Show) => {
    setShowState(v);
    try {
      localStorage.setItem(SHOW_KEY, v);
    } catch {
      /* per-viewer convenience only */
    }
  };
  const visible = show === 'both' ? BACKENDS : [show];
  const total = lanes.modal.length + lanes.local.length;
  return (
    <section className={'h-full min-h-0 flex flex-col ' + (compact ? '' : 'border border-line rounded-lg bg-panel')}>
      <header className={'flex items-center gap-2 shrink-0 ' + (compact ? 'pb-1.5' : 'h-8 px-3 border-b border-line')}>
        <div className="label">Render queue</div>
        <span className="text-[10px] text-fg3 num">{total ? `${total} job${total === 1 ? '' : 's'}` : 'empty'}</span>
        <div className="flex-1" />
        <Seg<Show>
          size="xs"
          value={show}
          onChange={setShow}
          options={[
            { id: 'both', label: 'Both' },
            { id: 'modal', label: `Modal${lanes.modal.length ? ` ${lanes.modal.length}` : ''}` },
            { id: 'local', label: `Local${lanes.local.length ? ` ${lanes.local.length}` : ''}` },
          ]}
        />
        {onClose && (
          <button className="text-fg3 hover:text-fg text-[12px] px-1" onClick={onClose} title="Hide the queue (the jobs pill still shows it)">
            ✕
          </button>
        )}
      </header>
      <div
        className={'flex-1 min-h-0 grid gap-2 ' + (compact ? '' : 'p-2')}
        style={{ gridTemplateColumns: `repeat(${visible.length}, minmax(0, 1fr))` }}
      >
        {visible.map((b) => (
          <Lane key={b} backend={b} jobs={lanes[b]} compact={compact} />
        ))}
      </div>
    </section>
  );
}

function Lane({ backend, jobs, compact }: { backend: TfBackend; jobs: QueueEntry[]; compact: boolean }) {
  const backends = useQueue((s) => s.backends);
  const look = backendLook(backend, backends);
  const running = jobs.filter((j) => (isForeign(j) ? j.state === 'running' : j.status === 'running')).length;
  const queued = jobs.length - running;
  const eta = laneEta(jobs);
  const local = backends?.local;
  const drag = useLaneDrag(backend, jobs);
  const [busy, setBusy] = useState(false);
  const start = async () => {
    setBusy(true);
    await startLocal();
    setBusy(false);
  };
  const canStart = backend === 'local' && local?.state === 'service_down';
  return (
    <div className="min-h-0 min-w-0 flex flex-col rounded-md border border-line bg-bg/60">
      <div className="flex items-center gap-1.5 h-7 px-2 border-b border-line shrink-0 min-w-0" title={look.line}>
        <Dot tone={look.tone} />
        <span className="text-[11px] font-semibold text-fg">{BACKEND_LABEL[backend]}</span>
        <span className="text-[10px] text-fg3 truncate">{look.word}</span>
        <span className="flex-1" />
        <span className="text-[10px] text-fg3 num shrink-0">
          {running} run · {queued} queued{eta != null ? ` · ${fmtEta(eta)}` : ''}
        </span>
        {backend === 'modal' && <ModalSlots compact />}
        {canStart && (
          <Btn size="xs" tone="deck" disabled={busy} onClick={start} title="Start the desktop's job service (textureflow-ctl start over SSH)">
            Start
          </Btn>
        )}
      </div>
      {look.tone !== 'ok' && backend === 'local' && !compact && (
        <div className="px-2 py-1 text-[10px] text-fg3 leading-snug border-b border-line shrink-0">{look.line}</div>
      )}
      <div ref={drag.listRef} className={'flex-1 min-h-0 overflow-y-auto p-1.5 flex flex-col gap-1 ' + (compact ? 'max-h-[340px]' : '')}>
        {jobs.length === 0 && (
          <div className="text-[11px] text-fg4 px-1 py-2">
            {backend === 'modal' ? 'Nothing on Modal.' : 'Nothing on the desktop.'}
          </div>
        )}
        {jobs.map((j) =>
          isForeign(j) ? (
            <ForeignCard key={entryId(j)} job={j} insertBefore={drag.insertAt === entryId(j)} />
          ) : (
            <Card
              key={j.job}
              job={j}
              lane={jobs}
              dragging={drag.dragId === j.job}
              dy={drag.dragId === j.job ? drag.dy : 0}
              insertBefore={drag.insertAt === j.job}
              onPointerDown={j.status === 'queued' ? (e) => drag.start(e, j.job) : undefined}
              localState={local?.state}
            />
          )
        )}
        {drag.insertAt === '__end' && <div className="h-0.5 rounded bg-deck shrink-0" />}
      </div>
    </div>
  );
}

/** Pointer-driven reorder of a lane's queued cards. Cards register by
 *  `data-job`; the insertion point is the first queued card whose middle is
 *  below the pointer. Starts after 4 px so a click stays a click. */
function useLaneDrag(backend: TfBackend, jobs: QueueEntry[]) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<{ id: string; dy: number; insertAt: string | null } | null>(null);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  function start(e: React.PointerEvent, id: string) {
    if (e.button !== 0 || (e.target as HTMLElement).closest('button')) return;
    const y0 = e.clientY;
    const el = e.currentTarget as HTMLElement;
    const pid = e.pointerId;
    let active = false;
    let insertAt: string | null = null;
    const queuedIds = () => jobsRef.current.filter(isQueued).map((j) => j.job);
    const move = (ev: PointerEvent) => {
      const dy = ev.clientY - y0;
      if (!active) {
        if (Math.abs(dy) < 4) return;
        active = true;
        el.setPointerCapture(pid);
        document.body.style.cursor = 'grabbing';
      }
      insertAt = '__end';
      for (const qid of queuedIds()) {
        if (qid === id) continue;
        const r = listRef.current?.querySelector<HTMLElement>(`[data-job="${qid}"]`)?.getBoundingClientRect();
        if (r && ev.clientY < r.top + r.height / 2) {
          insertAt = qid;
          break;
        }
      }
      // Dropping right where it was is no move.
      const ids = queuedIds();
      const at = ids.indexOf(id);
      if (insertAt === (ids[at + 1] ?? '__end')) insertAt = null;
      setState({ id, dy, insertAt });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      document.body.style.cursor = '';
      setState(null);
      if (!active || insertAt == null) return;
      const ids = queuedIds().filter((x) => x !== id);
      const at = insertAt === '__end' ? ids.length : ids.indexOf(insertAt);
      ids.splice(at < 0 ? ids.length : at, 0, id);
      reorderLane(backend, ids);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }

  return { listRef, start, dragId: state?.id ?? null, dy: state?.dy ?? 0, insertAt: state?.insertAt ?? null };
}

const KIND: Record<JobKind, { label: string; cls: string; title: string }> = {
  draft: { label: 'DRAFT', cls: 'bg-amber/20 text-amber border-amber/50', title: "The final's first pass — Upscale it later" },
  upscale: { label: 'UPSCALE', cls: 'bg-deck/20 text-deck border-deck/50', title: "Finishing a draft (runs where the draft's bundle lives)" },
  full: { label: 'FULL', cls: 'bg-panel2 text-fg2 border-line2', title: 'A full render: first pass + upscale' },
};

function Card({
  job,
  lane,
  dragging,
  dy,
  insertBefore,
  onPointerDown,
  localState,
}: {
  job: Job;
  lane: QueueEntry[];
  dragging: boolean;
  dy: number;
  insertBefore: boolean;
  onPointerDown?: (e: React.PointerEvent) => void;
  localState?: string;
}) {
  const kind = jobKind(job);
  const { text, thumb } = useJobLabel(job);
  const queued = job.status === 'queued';
  const p = job.progress;
  const pct = jobPct(job);
  const estimate = p?.source === 'estimate';
  const eta = p?.eta_s ?? job.eta_s ?? null;
  const other: TfBackend = jobBackend(job) === 'modal' ? 'local' : 'modal';
  return (
    <>
      {insertBefore && <div className="h-0.5 rounded bg-deck shrink-0" />}
      <div
        data-job={job.job}
        onPointerDown={onPointerDown}
        style={dragging ? { transform: `translateY(${dy}px)`, zIndex: 10 } : undefined}
        className={
          'group relative shrink-0 flex gap-2 p-1.5 rounded border bg-panel2 min-w-0 ' +
          (queued ? 'border-line cursor-grab ' : 'border-deck/40 ') +
          (dragging ? 'opacity-80 shadow-[0_8px_24px_rgba(0,0,0,.5)] cursor-grabbing' : '')
        }
        title={JSON.stringify(job.args, null, 1).slice(0, 600)}
      >
        <Thumb src={thumb} />
        <div className="flex-1 min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className={'text-[8.5px] font-bold tracking-wider px-1 rounded border shrink-0 ' + KIND[kind].cls} title={KIND[kind].title}>
              {KIND[kind].label}
            </span>
            <span className="text-[11px] text-fg truncate min-w-0 flex-1">{text}</span>
            {job.by === 'agent' && <span className="text-[9px] text-fg3 shrink-0" title="Started by an agent">✦</span>}
          </div>
          <div className="flex items-center gap-1.5 text-[10px] min-w-0">
            <span className={'truncate ' + (queued ? 'text-fg3' : 'text-fg2')}>{phaseText(job, localState as never)}</span>
            <span className="flex-1" />
            {eta != null && (
              <span className="num text-fg3 shrink-0" title={estimate ? 'Estimated from the time model' : 'From the live stage'}>
                {estimate ? '≈' : ''}
                {fmtEta(eta)}
                {queued ? '' : ' left'}
              </span>
            )}
          </div>
          {!queued && <Bar pct={pct} estimate={estimate} />}
        </div>
        <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity bg-panel2/95 rounded">
          {queued && (
            <>
              <IconBtn
                onClick={() => toTop(job, lane)}
                disabled={job.lane_pos === 0}
                title="To the top of this lane"
              >
                ⤒
              </IconBtn>
              <IconBtn
                onClick={() => moveToOtherLane(job)}
                disabled={!!job.pinned}
                title={job.pinned ? "A finish runs where its draft's bundle lives — it can't change backend" : `Move to ${BACKEND_LABEL[other]}`}
              >
                ⇄
              </IconBtn>
            </>
          )}
          <IconBtn onClick={() => cancelQueued(job)} title={queued ? 'Remove from the queue' : 'Cancel this render'} danger>
            ✕
          </IconBtn>
        </div>
      </div>
    </>
  );
}

function IconBtn({
  children,
  onClick,
  title,
  disabled,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      disabled={disabled}
      title={title}
      className={
        'w-5 h-5 rounded border border-line2 bg-panel text-[11px] leading-none text-fg2 ' +
        (disabled ? 'opacity-30 cursor-default' : danger ? 'hover:text-bad hover:border-bad/60' : 'hover:text-fg hover:border-fg4')
      }
    >
      {children}
    </button>
  );
}

const slotsTitle = (n: number) =>
  `Parallel GPUs: ${n} — up to ${n} Modal render${n === 1 ? '' : 's'} billed at once. ` +
  'Lowering it never cancels a running render; the lane drains down to the new limit.';

/** How many Modal GPU containers may render at once (1..max_slots). `compact`:
 *  a − n + stepper for the lane header; else a labelled slider (Render panel).
 *  Renders nothing on a hub that doesn't publish `max_slots`. */
export function ModalSlots({ compact = false }: { compact?: boolean }) {
  const modal = useQueue((s) => s.backends?.modal);
  if (!modal?.max_slots || modal.max_slots < 2) return null;
  const { slots, max_slots: max } = modal;
  const set = (n: number) => n !== slots && n >= 1 && n <= max && setModalSlots(n);
  if (compact) {
    return (
      <span className="flex items-center gap-0.5 shrink-0" title={slotsTitle(slots)}>
        <span className="text-[10px] text-fg3 mr-0.5">GPUs</span>
        <IconBtn onClick={() => set(slots - 1)} disabled={slots <= 1} title="One GPU fewer">
          −
        </IconBtn>
        <span className="num text-[11px] text-fg w-3 text-center">{slots}</span>
        <IconBtn onClick={() => set(slots + 1)} disabled={slots >= max} title={`One GPU more (max ${max})`}>
          +
        </IconBtn>
      </span>
    );
  }
  return (
    <div className="flex flex-col gap-0.5" title={slotsTitle(slots)}>
      <span className="text-[11px] text-fg2">
        Parallel GPUs: <span className="num text-fg">{slots}</span>
        <span className="text-fg3"> — up to {slots} render{slots === 1 ? '' : 's'} billed at once</span>
      </span>
      <input
        type="range"
        min={1}
        max={max}
        step={1}
        value={slots}
        onChange={(e) => set(Number(e.target.value))}
        className="w-full accent-[rgb(var(--deck))]"
      />
    </div>
  );
}

/** Another client's job on the desktop GPU: shown in its place, never touched. */
function ForeignCard({ job, insertBefore }: { job: ForeignJob; insertBefore: boolean }) {
  const p = job.progress;
  const running = job.state === 'running';
  return (
    <>
      {insertBefore && <div className="h-0.5 rounded bg-deck shrink-0" />}
      <div
        className="shrink-0 flex gap-2 p-1.5 rounded border border-dashed border-line2 bg-panel/60 min-w-0"
        title={`Another client's job on the desktop (${job.client ?? 'unknown'}) — WZRD can't move or cancel it`}
      >
        <div className="w-9 h-9 shrink-0 rounded border border-dashed border-line2 flex items-center justify-center text-[9px] text-fg4">
          ext
        </div>
        <div className="flex-1 min-w-0 flex flex-col gap-0.5">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="text-[8.5px] font-bold tracking-wider px-1 rounded border border-line2 text-fg3 shrink-0">
              {(job.client ?? 'other').toUpperCase().slice(0, 14)}
            </span>
            <span className="text-[11px] text-fg2 truncate min-w-0 flex-1">{job.label ?? job.job_id}</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-fg3 min-w-0">
            <span className="truncate">{running ? 'rendering' : job.state}</span>
            <span className="flex-1" />
            {p?.eta_s != null && <span className="num shrink-0">≈{fmtEta(p.eta_s)}</span>}
          </div>
          {running && <Bar pct={p?.pct ?? null} estimate={p?.source !== 'live'} />}
        </div>
      </div>
    </>
  );
}

/** Progress: a filled bar at `pct`, a sweeping sliver when unknown; striped when estimated. */
function Bar({ pct, estimate }: { pct: number | null; estimate: boolean }) {
  const fill = estimate
    ? 'bg-[repeating-linear-gradient(135deg,rgb(var(--deck)/.75)_0_4px,rgb(var(--deck)/.35)_4px_8px)]'
    : 'bg-deck';
  return (
    <div className="relative h-[3px] rounded-full bg-line overflow-hidden mt-0.5">
      {pct == null ? (
        <div className={'absolute inset-y-0 left-0 w-[40%] rounded-full animate-sweep ' + fill} />
      ) : (
        <div className={'absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ' + fill} style={{ width: `${Math.max(2, pct * 100)}%` }} />
      )}
    </div>
  );
}

function Thumb({ src }: { src: string | null }) {
  const [broken, setBroken] = useState(false);
  useEffect(() => setBroken(false), [src]);
  return (
    <div className="w-9 h-9 shrink-0 rounded bg-black border border-line overflow-hidden flex items-center justify-center">
      {src && !broken ? (
        <img src={mediaUrl(src)} loading="lazy" draggable={false} onError={() => setBroken(true)} className="w-full h-full object-cover" />
      ) : (
        <span className="text-[10px] text-fg4">TF</span>
      )}
    </div>
  );
}

// ---------- what a card calls its job ----------

let recipePreviews: Record<string, string | null> | null = null;
let recipeLoad: Promise<void> | null = null;

function useRecipePreviews(): Record<string, string | null> {
  const [, bump] = useState(0);
  useEffect(() => {
    if (recipePreviews) return;
    recipeLoad ??= listRecipes()
      .then((r) => {
        recipePreviews = Object.fromEntries(r.recipes.map((x) => [x.name, x.preview ?? null]));
      })
      .catch(() => {
        recipePreviews = {};
      });
    recipeLoad.then(() => bump((n) => n + 1));
  }, []);
  return recipePreviews ?? {};
}

/** Label + thumb: an Upscale names (and shows) its draft — found in the open
 *  folder by its request id, which the draft's file name carries; else the
 *  recipe / prompt / first style image. */
function useJobLabel(job: Job): { text: string; thumb: string | null } {
  const previews = useRecipePreviews();
  const a = job.args ?? {};
  const finish = a.finish_from as string | undefined;
  const draft = useStudio((s) => (finish ? s.listing?.items.find((i) => i.name.includes(finish)) ?? null : null));
  const images = Array.isArray(a.images) ? (a.images as string[]) : [];
  const firstImage = images.find((p) => typeof p === 'string' && !/^https?:/.test(p)) ?? null;
  if (finish) {
    return {
      text: draft ? draft.name.replace(/\.[^.]+$/, '') : `draft ${finish.slice(-8)}`,
      thumb: draft ? draft.thumb ?? null : null,
    };
  }
  const recipe = a.recipe as string | undefined;
  const prompt = (a.prompt as string | undefined)?.trim();
  const secs = a.n_seconds != null ? ` · ${a.n_seconds} s` : '';
  const text = recipe
    ? `${recipe}${images.length ? ` + ${images.length} style${images.length === 1 ? '' : 's'}` : ''}${secs}`
    : prompt
    ? `“${prompt}”${secs}`
    : firstImage
    ? `${firstImage.split('/').pop()}${images.length > 1 ? ` +${images.length - 1}` : ''}${secs}`
    : `texture_flow${secs}`;
  return { text, thumb: firstImage ?? (recipe ? previews[recipe] ?? null : null) };
}
