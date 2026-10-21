// Align route (⌘3, in the TopBar's ⚙ menu) — the geometric stage between
// the composite and the projector. A load-in step, done once per venue, so
// it's one click down rather than in the tab bar, and the app never reopens
// on it (store.ts::initialRoute).
//
// Everything here is engine state, not scene state: nothing on this tab is
// written into scene.json, nothing is per leg, and nothing survives into
// another venue. The engine owns `alignment.json` and writes it on its own
// debounce — there is no save button on purpose.

import { nativeUnderlay } from '../api/previewMode';
import { useEffect } from 'react';
import { useStore } from '../state/store';
import { WarpCanvas } from '../components/WarpCanvas';
import { Btn, Panel, Seg } from '../ui/primitives';
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
  { id: 'white', label: 'White', hint: 'The whole projector raster full white — for the lit surface photo' },
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
    <div className="flex h-full min-h-0 gap-2 p-2">
      <Panel hole={nativeUnderlay()} className="flex-1 overflow-hidden" bodyClassName="min-h-0 flex flex-col">
        {error && (
          <div className="flex items-start gap-2 border-b border-bad/40 bg-bad/10 px-3 py-1.5 text-[11px] text-fg">
            {/* The engine rejected the edit and kept rendering the previous
                alignment — nothing on the wall moved. */}
            <span className="flex-1">{error}</span>
            <button className="underline text-fg2 hover:text-fg" onClick={() => setError(null)}>
              dismiss
            </button>
          </div>
        )}
        <div className="min-h-0 flex-1">
          <WarpCanvas />
        </div>
      </Panel>

      <div className="w-[280px] shrink-0 flex flex-col gap-2">
        <Panel title="Projector bumped?">
          <div className="p-3 text-[12px] leading-relaxed text-fg2">
            Drag the four corners until the light lands back on the surface — no re-segmenting. Turn on the{' '}
            <b className="text-fg">Grid</b> to see the warp on the wall. Arrows nudge 1&nbsp;px, ⇧ 10.
          </div>
        </Panel>

        <Panel title="Warp">
          <div className="p-3 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[12px] text-fg2">Output warp</span>
              <Seg
                value={doc?.enabled ? 'on' : 'off'}
                onChange={(v) => doc && setAlignmentEnabled(v === 'on')}
                options={[
                  { id: 'off', label: 'Off', title: 'Bypass — content goes out unwarped' },
                  { id: 'on', label: 'On' },
                ]}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[12px] text-fg2">Extra handles</span>
              <span className="num text-[12px] text-fg">
                {doc ? `${doc.points.length} / ${doc.points_max}` : '–'}
              </span>
            </div>
            <Btn
              size="sm"
              disabled={!doc}
              onClick={() => void resetAlignment().catch((e) => setError(String(e)))}
              title="Identity corners, no handles, black background. Leaves the on/off switch alone."
            >
              Reset to identity
            </Btn>
          </div>
        </Panel>

        <Panel title="Test pattern">
          <div className="p-3 grid grid-cols-2 gap-1.5">
            {PATTERNS.map((p) => (
              <Btn
                key={p.id}
                size="sm"
                tone="deck"
                active={doc?.test_pattern === p.id}
                disabled={!doc}
                title={p.hint}
                onClick={() => void setTestPattern(p.id).catch((e) => setError(String(e)))}
              >
                {p.label}
              </Btn>
            ))}
          </div>
        </Panel>

        <Panel title="Background">
          <div className="p-3 flex flex-col gap-2">
            <label className="relative flex items-center gap-2 h-7 rounded-md border border-line bg-panel2 px-2 cursor-pointer hover:border-line2">
              <span className="text-[11px] text-fg2 flex-1">Fill outside the content</span>
              <span className="num text-[10px] text-fg3">{doc?.background ?? '#000000'}</span>
              <span className="w-6 h-4 rounded-sm border border-line2" style={{ background: doc?.background ?? '#000' }} />
              <input
                type="color"
                className="absolute inset-0 opacity-0 cursor-pointer"
                value={doc?.background ?? '#000000'}
                disabled={!doc}
                onChange={(e) => setAlignmentBackground(e.target.value)}
              />
            </label>
            {backgroundLit && (
              // A non-black background floods the physical surface with light
              // and breaks the additive thesis. It persists, so the warning
              // stays visible for as long as it is set.
              <div className="rounded-md border border-amber/60 bg-amber/10 px-2 py-1.5 text-[11px] text-amber leading-snug">
                ⚠ The background is lit — the whole surface glows. Alignment aid only.
                <button className="ml-1 underline hover:text-fg" onClick={() => setAlignmentBackground('#000000')}>
                  Back to black
                </button>
              </div>
            )}
          </div>
        </Panel>
        <div className="flex-1" />
        <div className="px-1 text-[10px] text-fg4 leading-snug">
          Alignment is venue state (alignment.json), not scene content — it never changes with TAKE or looks.
        </div>
      </div>
    </div>
  );
}
