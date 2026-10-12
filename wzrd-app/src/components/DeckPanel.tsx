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

import { useEffect, useMemo, useState } from 'react';
import { demandChannel, useTelemetryDemand } from '../api/telemetryDemand';
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
    <div className={'grid grid-cols-2 shrink-0 rounded-lg border border-line bg-bg p-[3px] gap-[3px] ' + (big ? 'h-11' : 'h-7')}>
      <button
        onClick={() => setDeck('design')}
        title="Hands on DESIGN — the private scratch leg. Nothing here reaches the wall until TAKE. (D)"
        className={
          'rounded-md font-semibold tracking-[0.12em] whitespace-nowrap transition-colors ' +
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
          'rounded-md font-semibold tracking-[0.12em] whitespace-nowrap transition-colors flex items-center justify-center gap-1.5 ' +
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
  const designOv = useStore((s) => s.overrides.design);
  const liveOv = useStore((s) => s.overrides.live);
  const differs = useMemo(
    () => designDiffers(designJson, liveJson, designOv, liveOv),
    [designJson, liveJson, designOv, liveOv]
  );
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
        // No overflow-hidden here: `.arm-glow`'s pulse lives outside the box.
        'relative shrink-0 whitespace-nowrap rounded-lg border font-bold tracking-[0.18em] transition-colors ' +
        (big ? 'h-14 text-[15px] w-full' : 'h-7 px-3 text-[11px]') +
        ' ' +
        (phase !== 'idle'
          ? 'border-live bg-live/20 text-white'
          : differs
          ? 'border-live bg-live/15 text-white arm-glow hover:bg-live/30'
          : 'border-line2 bg-panel2 text-fg3 hover:text-fg hover:border-live/60')
      }
    >
      {phase !== 'idle' && (
        <span className="absolute inset-0 rounded-[inherit] overflow-hidden pointer-events-none">
          {phase === 'ramping' && (
            <span className="absolute inset-y-0 left-0 bg-live/60" style={{ width: `${Math.round(mix * 100)}%` }} />
          )}
          {phase === 'pending' && <span className="absolute inset-0 bg-live/20 animate-pulse" />}
        </span>
      )}
      <span className="relative flex items-center justify-center gap-2">
        {label}
        {big && phase === 'idle' && <span className="text-[10px] font-medium tracking-normal opacity-70">{fmtFade(fade)}</span>}
      </span>
    </button>
  );
}

/** How often the cue thumbnail refreshes while design just plays (clips,
 *  clocks and audio keep it moving; edits refresh it at once). */
const CUE_REFRESH_MS = 1000;
/** Give up on a refresh whose frame never arrives (engine busy/down). */
const CUE_GRAB_TIMEOUT_MS = 1500;

/**
 * The cue thumbnail: the design composite, sampled — not streamed. Holding
 * the JPEG `preview` channel makes the engine render the whole design leg
 * every frame plus a 15 fps readback + encode, for a ~118 px picture in the
 * normal show posture (hands on LIVE). Instead this holds demand just long
 * enough for one fresh frame: on mount, shortly after any design change
 * (applied reload, pins, scene, cue, take/pull), and every
 * `CUE_REFRESH_MS`. Hovering it "peeks" at full rate.
 */
function CueMonitor({ label }: { label: string }) {
  const preview = useStore((s) => s.preview);
  const [peek, setPeek] = useState(false);
  useTelemetryDemand('preview', peek);
  useCueFrames();
  return (
    <div
      className="rounded-md border border-design/40 overflow-hidden bg-black"
      onMouseEnter={() => setPeek(true)}
      onMouseLeave={() => setPeek(false)}
      title="What TAKE would send. Refreshes on every design change and about once a second; hover to watch it live."
    >
      <div className="flex items-center justify-between px-2 h-6 bg-design/10 border-b border-design/30">
        <span className="text-[9px] font-semibold tracking-[0.14em] text-design">CUED · DESIGN</span>
        <span className="text-[10px] text-fg3 truncate ml-2">{peek ? 'live peek' : label}</span>
      </div>
      <div className="h-[118px] flex items-center justify-center">
        {preview ? (
          <img
            alt=""
            draggable={false}
            className="w-full h-full object-contain"
            src={`data:image/jpeg;base64,${preview.data_b64}`}
          />
        ) : (
          <span className="text-2xs text-fg4">no preview frames</span>
        )}
      </div>
    </div>
  );
}

/** One-shot `preview` grabs while mounted (see `CueMonitor`). */
function useCueFrames() {
  useEffect(() => {
    let holding = false;
    let guard = 0;
    let soon = 0;
    const release = () => {
      window.clearTimeout(guard);
      if (!holding) return;
      holding = false;
      demandChannel('preview', false);
    };
    const grab = () => {
      if (holding || document.hidden) return;
      holding = true;
      demandChannel('preview', true);
      guard = window.setTimeout(release, CUE_GRAB_TIMEOUT_MS);
    };
    // Coalesce a burst of edits (a fader drag) into one grab after it.
    const grabSoon = () => {
      window.clearTimeout(soon);
      soon = window.setTimeout(grab, 250);
    };
    const unsub = useStore.subscribe((s, prev) => {
      // A frame arrived while we held demand: it was rendered after we asked.
      if (holding && s.preview !== prev.preview) release();
      if (
        s.hotReload !== prev.hotReload ||
        s.sceneJson !== prev.sceneJson ||
        s.overrides.design !== prev.overrides.design ||
        s.cuedLook !== prev.cuedLook ||
        s.deck?.promote !== prev.deck?.promote
      ) {
        grabSoon();
      }
    });
    grab();
    const tick = window.setInterval(grab, CUE_REFRESH_MS);
    return () => {
      unsub();
      window.clearInterval(tick);
      window.clearTimeout(soon);
      release();
    };
  }, []);
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
        <CueMonitor label={cuedLook ?? (differs ? 'unsaved edits' : 'same as live')} />
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
