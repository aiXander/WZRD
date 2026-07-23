// §5.6 deck bar — the two-deck controls that live directly under the
// preview hero in Perform:
//
//   [LIVE ⇄ DESIGN]  which composite the native preview samples
//   [PROMOTE]        crossfade the projector to the design leg
//   [BAR|NOW]        quantize toggle: fade starts on the next bar boundary
//   [FADE slider]    §5.4 crossfade-time master — log 0s…30s, CUT at bottom
//   [PULL]           hard-copy live back into design
//
// The toggle can't overlay the preview itself: the native child window
// covers that rect, so HTML there is invisible. State arrives on the sticky
// `deck` channel; a ramping fade shows a progress wash on the button.

import { useState } from 'react';
import { promote, pull, previewSetSource, masterSet } from '../api/ipc';
import { useStore } from '../state/store';

// §5.4 crossfade-time master — a logarithmic 0s…30s fader. A pure log scale
// can't reach zero, so the very bottom of the travel snaps to CUT (0s) and
// the log ramp starts at MIN_S just above it.
const MIN_S = 0.05; // 50 ms — smallest non-CUT fade
const MAX_S = 30; // spec: master crossfade time up to 30 s

function posToSeconds(pos: number): number {
  if (pos <= 0) return 0;
  return MIN_S * Math.pow(MAX_S / MIN_S, pos);
}
function secondsToPos(s: number): number {
  if (s <= MIN_S) return 0; // CUT / sub-min collapse to the bottom
  return Math.min(1, Math.log(s / MIN_S) / Math.log(MAX_S / MIN_S));
}
function fmtFade(s: number): string {
  if (s <= 0) return 'CUT';
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  return `${s.toFixed(s < 10 ? 1 : 0)}s`;
}

// Trailing throttle so a slider drag doesn't flood the IPC bridge (matches
// the masters row / driver rack cadence).
let crossfadeTimer: number | undefined;
function sendCrossfade(seconds: number) {
  if (crossfadeTimer != null) window.clearTimeout(crossfadeTimer);
  crossfadeTimer = window.setTimeout(() => {
    crossfadeTimer = undefined;
    masterSet('crossfade', seconds).catch((e) => console.warn('master.set crossfade', e));
  }, 25);
}

export function DeckBar() {
  const deck = useStore((s) => s.deck);
  // §5.4 crossfade-time master arrives on the sticky `masters` channel
  // (engine-wide, not per leg). Local drag wins so the fader never jumps
  // under the finger.
  const crossfade = useStore((s) => s.masters?.crossfade);
  const [localPos, setLocalPos] = useState<number | null>(null);
  const [quantize, setQuantize] = useState<'bar' | 'now'>('bar');
  const [error, setError] = useState<string | null>(null);

  const twoLeg = deck?.two_leg ?? false;
  const source = deck?.preview_source ?? 'design';
  const phase = deck?.promote ?? 'idle';
  const mix = deck?.mix ?? 0;

  const fadeSeconds = localPos != null ? posToSeconds(localPos) : crossfade ?? 0.5;
  const fadePos = localPos ?? secondsToPos(crossfade ?? 0.5);

  function setFade(pos: number) {
    setLocalPos(pos);
    sendCrossfade(posToSeconds(pos));
  }

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
                ? 'preview + control the LIVE leg (the show): masters, knobs and overrides drive the projector'
                : 'preview + control the DESIGN leg (the scratchpad): nothing you tune here touches the show until PROMOTE'
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
        onClick={() => run(promote(Math.round(fadeSeconds * 1000), quantize))}
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

      {/* §5.4 crossfade-time master — logarithmic 0s…30s fader (CUT at the
          bottom). Engine-wide default promote fade; persists in the session
          sidecar. Double-click the label to reset to 0.5s. */}
      <div className="flex items-center gap-2 w-44">
        <button
          className={
            'text-[10px] tracking-wider ' +
            (Math.abs(fadeSeconds - 0.5) < 1e-3 ? 'text-zinc-500' : 'text-amber-300')
          }
          title="crossfade-time master — double-click to reset to 0.5s"
          onDoubleClick={() => setFade(secondsToPos(0.5))}
        >
          FADE
        </button>
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={fadePos}
          className="flex-1 accent-amber-400"
          title="promote crossfade time (logarithmic, 0s…30s)"
          onChange={(e) => setFade(parseFloat(e.target.value))}
        />
        <span className="w-10 text-right font-mono text-[11px] text-zinc-200">
          {fmtFade(fadeSeconds)}
        </span>
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
