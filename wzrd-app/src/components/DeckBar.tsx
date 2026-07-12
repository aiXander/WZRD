// §5.6 deck bar — the two-deck controls that live directly under the
// preview hero in Perform:
//
//   [LIVE ⇄ DESIGN]  which composite the native preview samples
//   [PROMOTE]        crossfade the projector to the design leg
//   [BAR|NOW]        quantize toggle: fade starts on the next bar boundary
//   [fade select]    CUT / 0.5s / 2s / 8s
//   [PULL]           hard-copy live back into design
//
// The toggle can't overlay the preview itself: the native child window
// covers that rect, so HTML there is invisible. State arrives on the sticky
// `deck` channel; a ramping fade shows a progress wash on the button.

import { useState } from 'react';
import { promote, pull, previewSetSource } from '../api/ipc';
import { useStore } from '../state/store';

const FADES: { label: string; ms: number }[] = [
  { label: 'CUT', ms: 0 },
  { label: '0.5s', ms: 500 },
  { label: '2s', ms: 2000 },
  { label: '8s', ms: 8000 },
];

export function DeckBar() {
  const deck = useStore((s) => s.deck);
  const [fadeMs, setFadeMs] = useState(500);
  const [quantize, setQuantize] = useState<'bar' | 'now'>('bar');
  const [error, setError] = useState<string | null>(null);

  const twoLeg = deck?.two_leg ?? false;
  const source = deck?.preview_source ?? 'design';
  const phase = deck?.promote ?? 'idle';
  const mix = deck?.mix ?? 0;

  function run(p: Promise<unknown>) {
    setError(null);
    p.catch((e) => setError(String(e)));
  }

  if (!twoLeg) {
    return (
      <div className="flex items-center gap-3 text-[11px] text-zinc-500">
        <span className="tracking-widest">DECK</span>
        <span>single-leg engine (no control surface) — promote unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <span className="text-[10px] tracking-widest text-zinc-500">DECK</span>

      {/* Preview source toggle — which leg the native preview shows. */}
      <div className="flex rounded overflow-hidden border border-ink-600 text-[11px]">
        {(['live', 'design'] as const).map((s) => (
          <button
            key={s}
            className={
              'px-3 py-1 tracking-wider ' +
              (source === s
                ? s === 'live'
                  ? 'bg-accent-red/30 text-red-200'
                  : 'bg-accent-green/30 text-emerald-200'
                : 'bg-ink-800 text-zinc-500 hover:text-zinc-300')
            }
            title={
              s === 'live'
                ? 'preview the projector output (masters applied, no warp)'
                : 'preview the design scratchpad (un-mastered)'
            }
            onClick={() => run(previewSetSource(s))}
          >
            {s.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="h-5 w-px bg-ink-600" />

      {/* Promote — with a mix-progress wash while ramping. */}
      <button
        className={
          'relative px-4 py-1 rounded border text-[11px] tracking-wider overflow-hidden ' +
          (phase === 'idle'
            ? 'border-amber-500/60 text-amber-300 hover:bg-amber-500/10'
            : 'border-amber-400 text-amber-200')
        }
        disabled={phase === 'ramping'}
        title="crossfade the projector to the design leg, then keep iterating"
        onClick={() => run(promote(fadeMs, quantize))}
      >
        {phase === 'ramping' && (
          <span
            className="absolute inset-y-0 left-0 bg-amber-500/25"
            style={{ width: `${Math.round(mix * 100)}%` }}
          />
        )}
        <span className="relative">
          {phase === 'pending'
            ? 'PROMOTE · waiting for bar'
            : phase === 'ramping'
            ? `PROMOTING ${(mix * 100).toFixed(0)}%`
            : 'PROMOTE'}
        </span>
      </button>

      {/* Quantize toggle */}
      <div className="flex rounded overflow-hidden border border-ink-600 text-[11px]">
        {(['bar', 'now'] as const).map((q) => (
          <button
            key={q}
            className={
              'px-2.5 py-1 tracking-wider ' +
              (quantize === q
                ? 'bg-ink-600 text-zinc-100'
                : 'bg-ink-800 text-zinc-500 hover:text-zinc-300')
            }
            title={
              q === 'bar'
                ? 'fade starts on the next bar boundary (lands on a downbeat)'
                : 'fade starts immediately'
            }
            onClick={() => setQuantize(q)}
          >
            {q.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Fade time */}
      <div className="flex rounded overflow-hidden border border-ink-600 text-[11px]">
        {FADES.map((f) => (
          <button
            key={f.ms}
            className={
              'px-2.5 py-1 ' +
              (fadeMs === f.ms
                ? 'bg-ink-600 text-zinc-100'
                : 'bg-ink-800 text-zinc-500 hover:text-zinc-300')
            }
            onClick={() => setFadeMs(f.ms)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="h-5 w-px bg-ink-600" />

      <button
        className="px-3 py-1 rounded border border-ink-600 text-[11px] tracking-wider text-zinc-400 hover:text-zinc-200 hover:bg-ink-700"
        disabled={phase === 'ramping'}
        title="reset the design leg to what's live (discards the draft)"
        onClick={() => run(pull())}
      >
        PULL
      </button>

      {error && (
        <span className="text-[11px] text-accent-red truncate flex-1" title={error}>
          {error}
        </span>
      )}
    </div>
  );
}
