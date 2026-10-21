// "Trash (n)" (show-setup G6 / T1): what was deleted in the last 30 min,
// across the whole library, with Restore per item, Restore all and "Empty
// now" (two-click). The list is polled every 5 s only while the popover is
// open; the count refreshes on mount, on open and after every app-side
// delete or restore (`useTrash.version`).

import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../api/hub';
import { trashFlush, trashList, trashRestore, type TrashItem } from '../../api/project';
import { useHub } from '../../state/hub';
import { useStore } from '../../state/store';
import { short } from '../../state/studio';
import { Popover } from '../Popover';
import { Btn } from '../primitives';
import { trashChanged, useTrash } from './trash';

const POLL_MS = 5000;

const hhmm = (epochS: number) => {
  const d = new Date(epochS * 1000);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
};

export function RecentlyDeleted({ className = '' }: { className?: string }) {
  const up = useHub((s) => s.status?.state === 'up');
  const version = useTrash((s) => s.version);
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<TrashItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [armEmpty, setArmEmpty] = useState(false);
  const [busy, setBusy] = useState(false);
  const btn = useRef<HTMLButtonElement | null>(null);

  const load = () =>
    trashList()
      .then((r) => {
        setItems(r.items);
        setError(null);
      })
      .catch((e) => setError(String(e)));

  useEffect(() => {
    if (up) load();
  }, [up, version]);

  useEffect(() => {
    if (!open || !up) return;
    load();
    const t = setInterval(load, POLL_MS);
    return () => clearInterval(t);
  }, [open, up]);

  useEffect(() => {
    if (!armEmpty) return;
    const t = setTimeout(() => setArmEmpty(false), 3000);
    return () => clearTimeout(t);
  }, [armEmpty]);

  const toast = useStore.getState().toast;
  const restore = async (ids: string[]) => {
    setBusy(true);
    try {
      const r = await trashRestore(ids);
      if (r.errors?.length) toast(`Restore: ${r.errors.join('; ')}`, 'warn');
      else toast(`Restored ${r.restored.length} file${r.restored.length === 1 ? '' : 's'}`, 'ok');
    } catch (e) {
      toast(`Restore failed: ${e}`, 'bad');
    } finally {
      setBusy(false);
      trashChanged();
    }
  };
  const empty = async () => {
    setBusy(true);
    try {
      const r = await trashFlush();
      toast(`Emptied the trash (${r.flushed} file${r.flushed === 1 ? '' : 's'} gone for good)`, 'ok');
    } catch (e) {
      toast(`Empty failed: ${e}`, 'bad');
    } finally {
      setBusy(false);
      setArmEmpty(false);
      trashChanged();
    }
  };

  const n = items?.length ?? 0;
  return (
    <>
      <button
        ref={btn}
        disabled={!up}
        onClick={() => setOpen((v) => !v)}
        title={up ? 'Recently deleted — restorable for 30 min' : 'The trash needs the content hub'}
        className={
          'h-6 px-2 rounded-md border text-[11px] font-medium whitespace-nowrap shrink-0 ' +
          (open ? 'bg-raised text-fg border-line2' : 'bg-panel2 text-fg2 border-line hover:text-fg hover:border-line2') +
          (!up ? ' opacity-40 pointer-events-none' : '') +
          ' ' +
          className
        }
      >
        Trash{n ? <span className="num text-fg3"> ({n})</span> : ''}
      </button>
      {open && (
        <Popover anchor={btn.current} onClose={() => setOpen(false)} width={360}>
          <div className="flex items-center gap-2 mb-2">
            <div className="label">Recently deleted</div>
            <div className="flex-1" />
            {n > 1 && (
              <Btn size="xs" disabled={busy} onClick={() => restore(items!.map((i) => i.id))}>
                Restore all
              </Btn>
            )}
            {n > 0 && (
              <Btn
                size="xs"
                tone="danger"
                active={armEmpty}
                disabled={busy}
                onClick={() => (armEmpty ? empty() : setArmEmpty(true))}
                title="Delete everything in the trash for good, now"
              >
                {armEmpty ? 'Empty — sure?' : 'Empty now'}
              </Btn>
            )}
          </div>
          {error ? (
            <div className="text-[11px] text-bad">{error}</div>
          ) : !items ? (
            <div className="text-[11px] text-fg3">Loading…</div>
          ) : !n ? (
            <div className="text-[11px] text-fg3">Nothing deleted in the last 30 min.</div>
          ) : (
            <div className="flex flex-col gap-1 max-h-[360px] overflow-y-auto -mx-1 px-1">
              {items.map((it) => (
                <div key={it.id} className="flex items-center gap-2 min-w-0 py-0.5">
                  <div className="w-12 h-8 rounded bg-black border border-line shrink-0 overflow-hidden flex items-center justify-center">
                    {it.thumb ? (
                      <img src={mediaUrl(it.thumb)} className="w-full h-full object-contain" draggable={false} />
                    ) : (
                      <span className="text-[9px] text-fg4">{it.kind === 'video' ? '▶' : '—'}</span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0 leading-tight">
                    <div className="text-[11px] text-fg truncate" title={it.path}>
                      {it.name}
                    </div>
                    <div className="text-[10px] text-fg3 truncate" title={it.dir}>
                      {short(it.dir)} · until {hhmm(it.expires)}
                    </div>
                  </div>
                  <Btn size="xs" disabled={busy} onClick={() => restore([it.id])} title={`Put it back in ${it.dir}`}>
                    Restore
                  </Btn>
                </div>
              ))}
            </div>
          )}
        </Popover>
      )}
    </>
  );
}
