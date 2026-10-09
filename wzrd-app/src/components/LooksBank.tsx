// Looks — the clip bank. A look is a saved scene (plain file in
// `<scene_dir>/looks/`) with a thumbnail grabbed at save time. Resolume's
// clip grid, semantically: each slot is a whole surface state.
//
//   click        cue it into DESIGN (preview it, keep tweaking)
//   double-click / ▶   cue + TAKE (on the bar, over FADE)
//   ⌥-click      delete (no dialogs mid-set)
//   1–9 with ⇧   cue slot N · ⇧⌥ + N = GO

import { useEffect, useRef, useState } from 'react';
import { deleteLook, loadLooks, recallLook, saveLook } from '../state/actions';
import { useStore } from '../state/store';
import { useAutopilot } from '../state/autopilot';

export function LooksBank() {
  const looks = useStore((s) => s.looks);
  const cued = useStore((s) => s.cuedLook);
  const live = useStore((s) => s.liveLook);
  const [naming, setNaming] = useState(false);
  const [armDelete, setArmDelete] = useState<string | null>(null);

  useEffect(() => {
    loadLooks();
  }, []);

  return (
    <div className="flex items-stretch gap-2 h-full px-2 py-2 overflow-x-auto">
      <AutoBlock count={looks.length} />
      {looks.map((l, i) => {
        const isLive = l.name === live;
        const isCued = l.name === cued;
        return (
          <div
            key={l.name}
            onClick={(e) => {
              if (e.altKey) {
                if (armDelete === l.name) {
                  deleteLook(l.name);
                  setArmDelete(null);
                } else setArmDelete(l.name);
                return;
              }
              setArmDelete(null);
              recallLook(l.name);
            }}
            onDoubleClick={() => recallLook(l.name, true)}
            title={`${l.name}\nclick: cue in design · double-click: GO (take live) · ⌥-click twice: delete`}
            className={
              'group relative shrink-0 w-[118px] rounded-md overflow-hidden border-2 bg-black transition-colors ' +
              (armDelete === l.name
                ? 'border-bad'
                : isLive
                ? 'border-live'
                : isCued
                ? 'border-design'
                : 'border-line hover:border-line2')
            }
          >
            {l.thumb_b64 ? (
              <img
                alt=""
                draggable={false}
                className="absolute inset-0 w-full h-full object-cover"
                src={`data:image/jpeg;base64,${l.thumb_b64}`}
              />
            ) : (
              <div className="absolute inset-0 dim-grid" />
            )}
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-1.5 pt-3 pb-1">
              <div className="text-[10px] font-medium text-white truncate">{l.name}</div>
            </div>
            <div className="absolute top-1 left-1 num text-[9px] text-white/70 bg-black/50 rounded px-1">{i + 1}</div>
            {isLive && (
              <div className="absolute top-1 right-1 text-[8px] font-bold tracking-wider bg-live text-white rounded px-1">LIVE</div>
            )}
            {!isLive && isCued && (
              <div className="absolute top-1 right-1 text-[8px] font-bold tracking-wider bg-design text-black rounded px-1">CUED</div>
            )}
            {armDelete === l.name && (
              <div className="absolute inset-0 bg-bad/40 flex items-center justify-center text-[10px] font-bold text-white">
                ⌥-click again to delete
              </div>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                recallLook(l.name, true);
              }}
              title="GO — cue and TAKE live"
              className="absolute bottom-1 right-1 w-6 h-6 rounded-full bg-live text-white text-[10px] opacity-0 group-hover:opacity-100 transition-opacity"
            >
              ▶
            </button>
          </div>
        );
      })}
      {naming ? (
        <NameField
          onDone={(n) => {
            setNaming(false);
            if (n) saveLook(n);
          }}
          suggestion={`Look ${looks.length + 1}`}
        />
      ) : (
        <button
          onClick={() => setNaming(true)}
          title="Save what DESIGN shows now (knob pins baked in) as a look"
          className="shrink-0 w-[118px] rounded-md border-2 border-dashed border-line2 text-fg3 hover:text-fg hover:border-design/60 flex flex-col items-center justify-center gap-0.5"
        >
          <span className="text-lg leading-none">+</span>
          <span className="text-[10px] font-semibold tracking-wider">SAVE LOOK</span>
        </button>
      )}
      {looks.length === 0 && !naming && (
        <div className="self-center text-[11px] text-fg3 max-w-[340px] leading-snug pl-1">
          Looks are saved scenes you can fire mid-set. Build something in DESIGN, then <b className="text-fg2">Save look</b>.
        </div>
      )}
    </div>
  );
}

function NameField({ onDone, suggestion }: { onDone: (n: string | null) => void; suggestion: string }) {
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    ref.current?.select();
  }, []);
  return (
    <div className="shrink-0 w-[160px] rounded-md border-2 border-design bg-panel2 p-2 flex flex-col justify-center gap-1.5">
      <div className="label">Name this look</div>
      <input
        ref={ref}
        autoFocus
        defaultValue={suggestion}
        spellCheck={false}
        className="h-7 rounded bg-bg border border-line2 px-2 text-[12px] outline-none focus:border-design"
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === 'Enter') onDone((e.target as HTMLInputElement).value.trim().replace(/[^A-Za-z0-9 _.-]/g, '') || null);
          if (e.key === 'Escape') onDone(null);
        }}
        onBlur={() => onDone(null)}
      />
    </div>
  );
}

function AutoBlock({ count }: { count: number }) {
  const { on, bars, setOn, setBars } = useAutopilot();
  return (
    <div className="shrink-0 w-[92px] flex flex-col justify-between pr-2 mr-0.5 border-r border-line">
      <div className="label">Looks</div>
      <button
        disabled={count < 2}
        onClick={() => setOn(!on)}
        title={count < 2 ? 'Save at least two looks to run auto-pilot' : 'Auto-pilot: take the next look every N bars, on the bar line'}
        className={
          'h-7 rounded-md border text-[11px] font-bold tracking-wider ' +
          (on
            ? 'bg-ok/20 border-ok text-ok'
            : 'bg-panel2 border-line text-fg3 hover:text-fg hover:border-line2') +
          (count < 2 ? ' opacity-40' : '')
        }
      >
        {on ? '● AUTO' : 'AUTO'}
      </button>
      <select
        value={bars}
        onChange={(e) => setBars(Number(e.target.value))}
        className="h-6 text-[11px] py-0"
        title="Bars per look"
      >
        {[4, 8, 16, 32, 64].map((n) => (
          <option key={n} value={n}>
            {n} bars
          </option>
        ))}
      </select>
    </div>
  );
}
