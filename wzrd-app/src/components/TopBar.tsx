// The top bar is the same in every room: where you are, which leg your
// hands are on, the musical clock, and the health of the rig — everything
// spec §10 wants legible at a glance from the booth.

import { useEffect, useRef, useState } from 'react';
import { open as openExternal } from '@tauri-apps/plugin-shell';
import { IN_TAURI } from '../api/transport';
import { resyncBeat, setBpm, tapTempo } from '../state/actions';
import { adoptAgentScene } from '../state/sceneCommit';
import { useStore, type Route } from '../state/store';
import { Dot, Kbd } from '../ui/primitives';
import { DeckSwitch, TakeButton } from './DeckPanel';
import { ProjectMenu } from './ProjectMenu';
import { restartEngine } from '../state/actions';

const ROUTES: { id: Route; label: string; key: string; hint: string }[] = [
  { id: 'prepare', label: 'Prepare', key: '⌘1', hint: 'Build the look: regions, effects, code' },
  { id: 'align', label: 'Align', key: '⌘2', hint: 'Land the light on the surface' },
  { id: 'perform', label: 'Perform', key: '⌘3', hint: 'Play the show' },
];

export function TopBar() {
  const route = useStore((s) => s.route);
  const setRoute = useStore((s) => s.setRoute);
  const agentEdit = useStore((s) => s.agentEdit);
  const setShowHelp = useStore((s) => s.setShowHelp);

  return (
    <header className="relative flex items-center gap-3 h-12 px-3 bg-panel border-b border-line shrink-0">
      {/* deck-coloured hairline: the room's mode, visible from anywhere */}
      <div className="absolute inset-x-0 top-0 h-[2px] bg-deck" />
      <div className="flex items-center gap-2 pr-1">
        <Logo />
        <span className="text-[13px] font-bold tracking-[0.22em]">WZRD</span>
      </div>
      <ProjectMenu />
      <nav className="flex items-center gap-0.5 rounded-lg bg-bg border border-line p-[3px]">
        {ROUTES.map((r) => (
          <button
            key={r.id}
            onClick={() => setRoute(r.id)}
            title={`${r.hint} (${r.key})`}
            className={
              'h-7 px-3 rounded-md text-[12px] font-medium transition-colors ' +
              (route === r.id ? 'bg-raised text-fg shadow-inset' : 'text-fg3 hover:text-fg')
            }
          >
            {r.label}
          </button>
        ))}
      </nav>

      {route !== 'perform' && (
        <div className="flex items-center gap-2 pl-2">
          <DeckSwitch size="sm" />
          <TakeButton size="sm" />
        </div>
      )}

      <div className="flex-1" />

      {agentEdit && (
        <button
          onClick={() => adoptAgentScene().catch((e) => console.error('adopt agent scene:', e))}
          className="h-7 px-2.5 text-[11px] font-semibold rounded-md bg-amber/15 text-amber border border-amber/60 hover:bg-amber/25"
          title={`The AI edited the design scene (r${agentEdit.rev}): ${agentEdit.summary}\nClick to save it to scene.json`}
        >
          ✦ AI edit · ADOPT
        </button>
      )}

      <Tempo />
      <Health />

      <button
        onClick={() => setRoute('debug')}
        title="Engine internals: logs, reloads, frame timing (⌘4)"
        className={
          'h-7 w-7 rounded-md text-[13px] border ' +
          (route === 'debug' ? 'bg-raised border-line2 text-fg' : 'border-transparent text-fg3 hover:text-fg hover:bg-raised')
        }
      >
        ⚙
      </button>
      <button
        onClick={() => setShowHelp(true)}
        title="Keyboard shortcuts (?)"
        className="h-7 w-7 rounded-md text-[12px] font-semibold border border-transparent text-fg3 hover:text-fg hover:bg-raised"
      >
        ?
      </button>
    </header>
  );
}

function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden>
      <defs>
        <linearGradient id="wz" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="rgb(var(--design))" />
          <stop offset="1" stopColor="rgb(var(--mod))" />
        </linearGradient>
      </defs>
      <path d="M10 1 L19 10 L10 19 L1 10 Z" fill="none" stroke="url(#wz)" strokeWidth="1.6" />
      <path d="M10 6 L14 10 L10 14 L6 10 Z" fill="url(#wz)" />
    </svg>
  );
}

function Tempo() {
  const t = useStore((s) => s.transport?.live);
  const bpm = t?.bpm ?? 120;
  const beats = t?.beats ?? 0;
  const beatInBar = ((Math.floor(beats) % 4) + 4) % 4;
  const frac = beats - Math.floor(beats);
  const [editing, setEditing] = useState(false);
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (editing) ref.current?.select();
  }, [editing]);

  return (
    <div className="flex items-center gap-1.5 h-8 pl-2 pr-1 rounded-lg bg-bg border border-line">
      <div className="flex gap-[3px]" title="Beat clock (live leg) — the bar line is the first lamp">
        {[0, 1, 2, 3].map((i) => {
          const on = i === beatInBar;
          const glow = on ? 1 - frac * 0.85 : 0;
          return (
            <span
              key={i}
              className={'w-2 h-2 rounded-full ' + (i === 0 ? 'bg-live' : 'bg-fg')}
              style={{ opacity: 0.15 + glow * 0.85, boxShadow: glow > 0.5 ? `0 0 8px ${i === 0 ? 'rgb(var(--live))' : 'white'}` : undefined }}
            />
          );
        })}
      </div>
      {editing ? (
        <input
          ref={ref}
          defaultValue={bpm.toFixed(1)}
          className="num w-14 h-6 bg-panel2 border border-line2 rounded px-1 text-[13px] outline-none"
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === 'Enter') {
              const v = parseFloat((e.target as HTMLInputElement).value);
              if (Number.isFinite(v)) setBpm(v);
              setEditing(false);
            }
            if (e.key === 'Escape') setEditing(false);
          }}
          onBlur={() => setEditing(false)}
        />
      ) : (
        <button
          onDoubleClick={() => setEditing(true)}
          onWheel={(e) => setBpm(bpm + (e.deltaY < 0 ? 1 : -1) * (e.shiftKey ? 0.1 : 1))}
          title="Tempo — double-click to type, scroll to nudge (⇧ fine)"
          className="num w-[52px] text-right text-[14px] font-semibold text-fg"
        >
          {bpm.toFixed(1)}
        </button>
      )}
      <span className="text-[9px] text-fg3 font-semibold tracking-wider">BPM</span>
      <button
        onPointerDown={() => tapTempo()}
        title="Tap tempo — tap 3+ times on the beat (Tab)"
        className="h-6 px-2 rounded-md bg-panel2 border border-line2 text-[10px] font-bold tracking-wider text-fg2 hover:text-fg active:bg-deck active:text-black"
      >
        TAP
      </button>
      <button
        onPointerDown={() => resyncBeat()}
        title="Put the bar line here — press on the ONE (`)"
        className="h-6 px-1.5 rounded-md bg-panel2 border border-line2 text-[10px] font-bold tracking-wider text-fg2 hover:text-fg active:bg-live active:text-white"
      >
        1
      </button>
    </div>
  );
}

function Health() {
  const fresh = useStore((s) => s.audioFresh);
  const fps = useStore((s) => s.fps);
  const hot = useStore((s) => s.hotReload);
  const running = useStore((s) => s.engineRunning);
  const engineError = useStore((s) => s.engineError);
  const audioTone = fresh?.state === 'fresh' ? 'ok' : fresh?.state === 'stale' ? 'warn' : fresh ? 'bad' : 'off';
  const fpsTone = !fps ? 'off' : fps.presenting === false ? 'info' : fps.fps > 55 ? 'ok' : fps.fps > 40 ? 'warn' : 'bad';
  const hotTone = !hot ? 'off' : hot.ok ? (hot.probe?.band === 'yellow' ? 'warn' : 'ok') : 'bad';
  return (
    <div className="flex items-center h-8 rounded-lg bg-bg border border-line divide-x divide-line">
      <button
        className="flex items-center gap-1.5 px-2.5 h-full text-[11px] text-fg2 hover:text-fg"
        title={`Audio feed: ${fresh?.state ?? 'unknown'}. Click to open the audio server's tuning page.`}
        onClick={() => IN_TAURI && openExternal('http://127.0.0.1:8765/').catch(() => {})}
      >
        <Dot tone={audioTone} /> Audio
      </button>
      <div
        className="flex items-center gap-1.5 px-2.5 h-full text-[11px] text-fg2"
        title={
          fps?.presenting === false
            ? 'Projector window hidden — engine self-paces offscreen (~30 Hz) by design; the preview stays live.'
            : `Projector frame rate · ${fps?.frame_time_ms.toFixed(1) ?? '–'} ms/frame`
        }
      >
        <Dot tone={fpsTone} />
        <span className="num">{fps ? Math.round(fps.fps) : '–'}</span>
        <span className="text-fg3 text-[10px]">FPS</span>
      </div>
      <div
        className="flex items-center gap-1.5 px-2.5 h-full text-[11px] text-fg2 max-w-[220px]"
        title={
          hot
            ? `${hot.target} ${hot.ok ? 'applied' : 'FAILED — the last good version keeps playing'}` +
              (hot.probe ? `\nGPU probe: ${hot.probe.band}, ~${hot.probe.predicted_p95_ms.toFixed(1)} ms/frame predicted` : '') +
              (hot.message ? '\n' + hot.message : '')
            : 'No reloads yet'
        }
      >
        <Dot tone={hotTone} />
        <span className="truncate">
          {!hot ? 'Ready' : !hot.ok ? 'Error — kept last good' : hot.probe?.band === 'yellow' ? 'Applied · heavy' : 'Applied'}
        </span>
      </div>
      {running ? (
        <div className="flex items-center gap-1.5 px-2.5 h-full text-[11px] text-fg2" title="Engine running">
          <Dot tone="ok" />
          Engine
        </div>
      ) : (
        <button
          onClick={() => restartEngine()}
          className="flex items-center gap-1.5 px-2.5 h-full text-[11px] text-bad font-semibold hover:bg-bad/15 rounded-r-lg"
          title={`Engine down${engineError ? ': ' + engineError : ''}\nClick to restart it on this project`}
        >
          <Dot tone="bad" />
          Engine ↻
        </button>
      )}
    </div>
  );
}

export function HelpOverlay() {
  const show = useStore((s) => s.showHelp);
  const setShow = useStore((s) => s.setShowHelp);
  if (!show) return null;
  const rows: [string, string][] = [
    ['D / L', 'Hands on DESIGN / LIVE'],
    ['T', 'TAKE — fade the wall to design (on the bar)'],
    ['⇧T', 'CUT — instant take'],
    ['P', 'PULL — copy live back into design'],
    ['B', 'Blackout / restore'],
    ['F (hold)', 'Flash — full blast while held'],
    ['Z', 'Freeze all motion / thaw'],
    ['Tab', 'Tap tempo'],
    ['`', 'Bar line here (resync)'],
    ['↑ ↓', 'Select layer'],
    ['1 – 9', 'Select layer N'],
    ['M / S', 'Mute / solo selected layer'],
    ['[ ]', 'Selected layer opacity down / up'],
    ['⇧1 – ⇧9', 'Cue look N into design'],
    ['⌘1 ⌘2 ⌘3', 'Prepare · Align · Perform'],
    ['⌘S', 'Save design (bakes knob pins into scene.json)'],
  ];
  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center animate-in" onClick={() => setShow(false)}>
      <div className="w-[460px] rounded-xl border border-line2 bg-panel shadow-2xl p-5" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center mb-3">
          <div className="text-[14px] font-semibold">Keyboard</div>
          <div className="flex-1" />
          <Kbd>esc</Kbd>
        </div>
        <div className="grid grid-cols-[110px_1fr] gap-y-1.5 text-[12px]">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <div className="num text-fg">{k}</div>
              <div className="text-fg2">{v}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-[11px] text-fg3 leading-snug">
          Faders: press to jump, ⇧-drag for fine, double-click to reset, scroll to nudge.
        </div>
      </div>
    </div>
  );
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const drop = useStore((s) => s.dropToast);
  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          onClick={() => drop(t.id)}
          className={
            'pointer-events-auto max-w-[560px] rounded-lg border px-3 py-2 text-[12px] shadow-xl animate-in bg-raised ' +
            (t.tone === 'bad'
              ? 'border-bad/70 text-fg'
              : t.tone === 'warn'
              ? 'border-amber/70 text-fg'
              : t.tone === 'ok'
              ? 'border-ok/60 text-fg'
              : 'border-line2 text-fg')
          }
        >
          {t.text}
        </div>
      ))}
    </div>
  );
}
