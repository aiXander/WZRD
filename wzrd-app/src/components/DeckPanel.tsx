// The deck — the only crossing between the two legs.
//
//   [ DESIGN | LIVE ]   where your hands are (and what the monitor shows)
//   cue monitor         what TAKE would send (design), shown while on LIVE
//   [ TAKE ]            crossfade the wall to design over FADE, on the bar
//   FADE · BAR/NOW · CUT · PULL
//
// TAKE arms (pulses red) whenever design holds something the crowd hasn't
// seen yet — the one deliberately loud control (spec: "committing to the
// crowd should feel like a decision").

import { useMemo, useState } from 'react';
import {
  designDiffers,
  getQuantize,
  pullLive,
  setCrossfade,
  setDeck,
  setQuantize,
  take,
  type Quantize,
} from '../state/actions';
import { useStore } from '../state/store';
import { Btn, Kbd, Seg } from '../ui/primitives';
import { Fader } from '../ui/Fader';
import { JpegFeed } from './NativePreview';

const MIN_S = 0.05;
const MAX_S = 30;
const posToSeconds = (p: number) => (p <= 0.001 ? 0 : MIN_S * Math.pow(MAX_S / MIN_S, p));
const secondsToPos = (s: number) =>
  s <= MIN_S ? 0 : Math.min(1, Math.log(s / MIN_S) / Math.log(MAX_S / MIN_S));
function fmtFade(s: number): string {
  if (s <= 0) return 'CUT';
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  return `${s.toFixed(s < 10 ? 1 : 0)}s`;
}

export function DeckSwitch({ size = 'lg' }: { size?: 'sm' | 'lg' }) {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const big = size === 'lg';
  return (
    <div className={'grid grid-cols-2 rounded-lg border border-line bg-bg p-[3px] gap-[3px] ' + (big ? 'h-11' : 'h-7')}>
      <button
        onClick={() => setDeck('design')}
        title="Hands on DESIGN — the private scratch leg. Nothing here reaches the wall until TAKE. (D)"
        className={
          'rounded-md font-semibold tracking-[0.12em] transition-colors ' +
          (big ? 'text-[12px]' : 'text-[10px] px-2') +
          ' ' +
          (leg === 'design' ? 'bg-design text-black' : 'text-fg3 hover:text-fg hover:bg-raised')
        }
      >
        DESIGN
      </button>
      <button
        onClick={() => setDeck('live')}
        title="Hands on LIVE — faders and knobs move the projector right now. (L)"
        className={
          'rounded-md font-semibold tracking-[0.12em] transition-colors flex items-center justify-center gap-1.5 ' +
          (big ? 'text-[12px]' : 'text-[10px] px-2') +
          ' ' +
          (leg === 'live' ? 'bg-live text-white' : 'text-fg3 hover:text-fg hover:bg-raised')
        }
      >
        <span className={'inline-block w-1.5 h-1.5 rounded-full ' + (leg === 'live' ? 'bg-white animate-pulse' : 'bg-live/70')} />
        LIVE
      </button>
    </div>
  );
}

function useTakeState() {
  const deck = useStore((s) => s.deck);
  const designJson = useStore((s) => s.sceneJson);
  const liveJson = useStore((s) => s.liveSceneJson);
  const differs = useMemo(() => designDiffers(designJson, liveJson), [designJson, liveJson]);
  return {
    phase: deck?.promote ?? 'idle',
    mix: deck?.mix ?? 0,
    differs,
    twoLeg: deck?.two_leg ?? false,
  };
}

export function TakeButton({ size = 'lg' }: { size?: 'sm' | 'lg' }) {
  const { phase, mix, differs } = useTakeState();
  const fade = useStore((s) => s.masters?.crossfade ?? 0.5);
  const big = size === 'lg';
  const label =
    phase === 'pending' ? 'ON THE BAR…' : phase === 'ramping' ? `TAKING ${Math.round(mix * 100)}%` : 'TAKE';
  return (
    <button
      onClick={() => take()}
      disabled={phase === 'ramping'}
      title={`Crossfade the wall to DESIGN over ${fmtFade(fade)}${getQuantize() === 'bar' ? ', starting on the next bar' : ''} (T · ⇧T = cut)`}
      className={
        'relative overflow-hidden rounded-lg border font-bold tracking-[0.18em] transition-colors ' +
        (big ? 'h-14 text-[15px] w-full' : 'h-7 px-3 text-[11px]') +
        ' ' +
        (phase !== 'idle'
          ? 'border-live bg-live/20 text-white'
          : differs
          ? 'border-live bg-live/15 text-white animate-arm hover:bg-live/30'
          : 'border-line2 bg-panel2 text-fg3 hover:text-fg hover:border-live/60')
      }
    >
      {phase === 'ramping' && (
        <span className="absolute inset-y-0 left-0 bg-live/60" style={{ width: `${Math.round(mix * 100)}%` }} />
      )}
      {phase === 'pending' && <span className="absolute inset-0 bg-live/20 animate-pulse" />}
      <span className="relative flex items-center justify-center gap-2">
        {label}
        {big && phase === 'idle' && <span className="text-[10px] font-medium tracking-normal opacity-70">{fmtFade(fade)}</span>}
      </span>
    </button>
  );
}

export function DeckPanel() {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const crossfade = useStore((s) => s.masters?.crossfade ?? 0.5);
  const cuedLook = useStore((s) => s.cuedLook);
  const { phase, differs, twoLeg } = useTakeState();
  const [q, setQ] = useState<Quantize>(getQuantize());

  if (!twoLeg) {
    return (
      <div className="p-3 text-2xs text-fg3">
        Single-leg engine (no control surface) — TAKE / PULL unavailable.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 p-3">
      <DeckSwitch />

      {/* Cue monitor: on LIVE the big monitor shows the wall, so this shows
          what's waiting in design. On DESIGN the big monitor already is it. */}
      {leg === 'live' ? (
        <div className="rounded-md border border-design/40 overflow-hidden bg-black">
          <div className="flex items-center justify-between px-2 h-6 bg-design/10 border-b border-design/30">
            <span className="text-[9px] font-semibold tracking-[0.14em] text-design">CUED · DESIGN</span>
            <span className="text-[10px] text-fg3 truncate ml-2">{cuedLook ?? (differs ? 'unsaved edits' : 'same as live')}</span>
          </div>
          <div className="h-[118px] flex items-center justify-center">
            <JpegFeed />
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-line bg-panel2 px-3 py-2 text-[11px] leading-snug text-fg2">
          {differs ? (
            <>
              <span className="text-live font-semibold">Design differs from live.</span> TAKE sends it to the wall.
            </>
          ) : (
            <>Design matches what’s live. Edit freely — nothing reaches the wall until TAKE.</>
          )}
        </div>
      )}

      <TakeButton />

      <div className="flex items-center gap-2">
        <Fader
          className="flex-1"
          label="FADE"
          value={secondsToPos(crossfade)}
          min={0}
          max={1}
          defaultValue={secondsToPos(0.5)}
          format={(p) => fmtFade(posToSeconds(p))}
          color="rgb(var(--live))"
          onChange={(p) => setCrossfade(posToSeconds(p))}
          title="TAKE crossfade time — log scale, CUT at the far left. Double-click: 0.5s"
        />
        <Seg
          value={q}
          onChange={(v) => {
            setQ(v);
            setQuantize(v);
          }}
          options={[
            { id: 'bar', label: 'BAR', title: 'TAKE starts on the next bar line (lands on a downbeat)' },
            { id: 'now', label: 'NOW', title: 'TAKE starts immediately' },
          ]}
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Btn tone="danger" onClick={() => take({ cut: true })} disabled={phase === 'ramping'} title="Hard cut to design, no fade, no bar wait (⇧T)">
          CUT <Kbd>⇧T</Kbd>
        </Btn>
        <Btn
          onClick={() => pullLive()}
          disabled={phase === 'ramping'}
          title="Copy what's live (scene + knobs) back into design — start editing from the wall. Discards design edits. (P)"
        >
          ⟲ PULL <Kbd>P</Kbd>
        </Btn>
      </div>
    </div>
  );
}
