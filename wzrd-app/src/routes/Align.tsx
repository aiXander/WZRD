// Align route (⌘2) — the geometric stage between the composite and the
// projector. Sits between Prepare and Perform because that is the order of a
// load-in: build the look, land it on the wall, then play.
//
// Everything here is engine state, not scene state: nothing on this tab is
// written into scene.json, nothing is per leg, and nothing survives into
// another venue. The engine owns `alignment.json` and writes it on its own
// debounce — there is no save button on purpose.

import { useEffect } from 'react';
import { useStore } from '../state/store';
import { WarpCanvas } from '../components/WarpCanvas';
import {
  ingestAlignment,
  resetAlignment,
  setAlignmentBackground,
  setAlignmentEnabled,
  setTestPattern,
} from '../state/alignment';
import { alignmentGet, type AlignmentDoc } from '../api/ipc';

const PATTERNS: { id: AlignmentDoc['test_pattern']; label: string; hint: string }[] = [
  { id: 'none', label: 'Off', hint: 'Show the actual composite' },
  { id: 'grid', label: 'Grid', hint: 'Generated in source space — bends exactly as the content does' },
  { id: 'border', label: 'Border', hint: 'Just the source rectangle outline' },
  { id: 'corners', label: 'Corners', hint: 'Corner marks + centre crosshair' },
];

export function Align() {
  const doc = useStore((s) => s.alignment);
  const error = useStore((s) => s.alignmentError);
  const setError = useStore((s) => s.setAlignmentError);

  // The `alignment` channel is sticky, so App.tsx normally hydrates this from
  // `lastPayload` before the tab is ever opened. Ask directly if it didn't:
  // the channel only emits at boot and on edits, so a store that missed the
  // snapshot would otherwise sit on the placeholder until someone made an
  // edit — and this tab is where an engine relaunch is most likely to be
  // noticed. One cheap request beats a dead tab.
  useEffect(() => {
    if (doc) return;
    let cancelled = false;
    alignmentGet()
      .then((d) => {
        if (!cancelled) ingestAlignment(d);
      })
      .catch((e) => console.warn('alignment_get:', e));
    return () => {
      cancelled = true;
    };
  }, [doc]);

  const backgroundLit = !!doc && doc.background.toLowerCase() !== '#000000';

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-ink-700 px-3 py-2 text-xs">
        <label className="flex items-center gap-2 text-zinc-300">
          <input
            type="checkbox"
            checked={doc?.enabled ?? false}
            disabled={!doc}
            onChange={(e) => setAlignmentEnabled(e.target.checked)}
          />
          Warp enabled
        </label>

        <span className="text-ink-600">|</span>

        <div className="flex items-center gap-1">
          <span className="text-zinc-500">Test pattern</span>
          {PATTERNS.map((p) => (
            <button
              key={p.id}
              title={p.hint}
              disabled={!doc}
              onClick={() => void setTestPattern(p.id).catch((e) => setError(String(e)))}
              className={
                'rounded px-2 py-1 ' +
                (doc?.test_pattern === p.id
                  ? 'bg-ink-600 text-zinc-100'
                  : 'text-zinc-400 hover:bg-ink-700 hover:text-zinc-100')
              }
            >
              {p.label}
            </button>
          ))}
        </div>

        <span className="text-ink-600">|</span>

        <label className="flex items-center gap-2 text-zinc-300">
          <span className="text-zinc-500">Background</span>
          <input
            type="color"
            className="h-6 w-8 bg-transparent"
            value={doc?.background ?? '#000000'}
            disabled={!doc}
            onChange={(e) => setAlignmentBackground(e.target.value)}
          />
        </label>
        {backgroundLit && (
          // A non-black background paints every dest pixel whose source falls
          // outside the composite — i.e. it floods the physical surface with
          // light and breaks the additive thesis. It persists, so the warning
          // has to be visible for as long as it is set.
          <span
            className="rounded border border-amber-700 bg-amber-900/60 px-2 py-1 text-[11px] text-amber-200"
            title="Non-black background lights up the whole surface. Alignment aid only — set it back to #000000 before the show."
          >
            ⚠ BACKGROUND IS LIT
            <button
              className="ml-2 underline hover:text-amber-100"
              onClick={() => setAlignmentBackground('#000000')}
            >
              black
            </button>
          </span>
        )}

        <div className="flex-1" />

        {doc && (
          <span className="text-zinc-500">
            {doc.points.length}/{doc.points_max} handles
          </span>
        )}
        <button
          disabled={!doc}
          onClick={() => void resetAlignment().catch((e) => setError(String(e)))}
          className="rounded px-2 py-1 text-zinc-400 hover:bg-ink-700 hover:text-zinc-100"
          title="Identity corners, no handles, black background. Leaves the enabled flag alone."
        >
          Reset
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 border-b border-red-900/60 bg-red-950/40 px-3 py-1.5 text-[11px] text-red-300">
          {/* The engine rejected the edit and kept rendering the previous
              alignment — nothing on the wall moved. */}
          <span className="flex-1">{error}</span>
          <button className="underline hover:text-red-100" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      <div className="min-h-0 flex-1">
        <WarpCanvas />
      </div>
    </div>
  );
}
