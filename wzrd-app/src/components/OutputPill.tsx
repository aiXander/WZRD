// OUTPUT — the projector switch, Resolume-style: output is off at every
// launch and turned on here (pill → display = two clicks). The popover also
// holds the projector-facing settings: flips + fit (stored in the project's
// alignment, so they travel with the install), cursor hiding and display
// sleep (per machine). The warp itself stays in the Align room.

import { useEffect, useRef, useState } from 'react';
import { setOutputTransform } from '../state/alignment';
import {
  identifyDisplays,
  outputOff,
  outputToDisplay,
  outputWindow,
  refreshOutput,
  setOutputOptions,
  startOutputSync,
  useOutput,
  type DisplayInfo,
} from '../state/output';
import { useStore } from '../state/store';
import type { OutputFit } from '../api/ipc';
import { Btn, Dot, Seg } from '../ui/primitives';
import { Popover } from '../ui/Popover';

let synced = false;

const fmtMode = (d: DisplayInfo) =>
  `${d.width}×${d.height}` + (d.refresh_hz ? ` · ${Math.round(d.refresh_hz)} Hz` : '');

const scaled = (d: DisplayInfo) => d.native != null && (d.native[0] !== d.width || d.native[1] !== d.height);

export function OutputPill() {
  const state = useOutput((s) => s.state);
  const displays = useOutput((s) => s.displays);
  const error = useOutput((s) => s.error);
  const alignment = useStore((s) => s.alignment);
  const setRoute = useStore((s) => s.setRoute);
  const anchor = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (synced) return;
    synced = true;
    startOutputSync();
  }, []);
  // Fresh display list each time the popover opens (the browser seat has no
  // pushes; in the app it's cheap insurance after a replug).
  useEffect(() => {
    if (open) void refreshOutput();
  }, [open]);

  const target = state?.target ?? 'off';
  const live = displays.find((d) => d.name === state?.display);
  const tone = state?.lost ? 'bad' : target === 'display' ? 'ok' : target === 'window' ? 'info' : 'off';
  const detail =
    target === 'display' ? state?.display ?? 'display' : target === 'window' ? 'window' : state?.lost ? 'lost' : 'off';

  return (
    <>
      <button
        ref={anchor}
        onClick={() => setOpen((o) => !o)}
        title={
          state?.lost
            ? `Output display "${state.lost}" disconnected — output is off`
            : target === 'off'
            ? 'Projector output is off — click to send it to a display'
            : `Projector output on ${detail}${live ? ` (${fmtMode(live)})` : ''}`
        }
        className={
          'flex items-center gap-1.5 px-2 xl:px-2.5 h-8 shrink-0 rounded-lg bg-bg border text-[11px] ' +
          (state?.lost
            ? 'border-bad/60 text-bad font-semibold'
            : target === 'off'
            ? 'border-line text-fg3 hover:text-fg'
            : 'border-ok/50 text-fg')
        }
      >
        <Dot tone={tone} />
        <span className="font-semibold tracking-wide">OUTPUT</span>
        <span className={'max-w-[9rem] truncate ' + (target === 'off' ? '' : 'text-fg2')}>{detail}</span>
      </button>
      {open && (
        <Popover anchor={anchor.current} onClose={() => setOpen(false)} width={340}>
          <div className="flex items-center px-1 pb-1.5">
            <div className="label flex-1">Output</div>
            <Btn size="xs" tone="ghost" onClick={() => identifyDisplays()} title="Flash each display's number on it">
              Identify
            </Btn>
          </div>

          <div className="flex flex-col gap-0.5">
            <Row active={target === 'off'} onClick={() => outputOff()} title="Off" sub="Engine keeps running, nothing is sent" />
            <Row
              active={target === 'window'}
              onClick={() => outputWindow()}
              title="Window"
              sub="A movable window — testing without a projector"
            />
            {displays.map((d) => {
              const holdsUi = d.operator;
              return (
                <Row
                  key={d.name}
                  active={target === 'display' && state?.display === d.name}
                  disabled={holdsUi}
                  onClick={() => outputToDisplay(d.name)}
                  title={`${d.index + 1} · ${d.name}`}
                  sub={
                    fmtMode(d) +
                    (holdsUi ? ' · holds this UI — use Window' : d.builtin ? ' · built-in' : '') +
                    (state?.last_display === d.name && target !== 'display' ? ' · last used' : '')
                  }
                  warn={scaled(d)}
                />
              );
            })}
          </div>

          {state?.lost && (
            <Note tone="bad">“{state.lost}” disconnected, so output was switched off. Reconnect it and pick it again.</Note>
          )}
          {live && scaled(live) && live.native && (
            <Note tone="warn">
              macOS sends {live.width}×{live.height} but this display's native mode is {live.native[0]}×{live.native[1]}.
              Set it to native in System Settings › Displays for a sharp image.
            </Note>
          )}
          {error && <Note tone="bad">{error}</Note>}

          <div className="label px-1 pt-3 pb-1.5 mt-2 border-t border-line">Projector</div>
          {alignment ? (
            <div className="flex flex-col gap-2 px-1">
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-fg3 w-10">Flip</span>
                <Btn
                  size="sm"
                  active={!!alignment.flip_h}
                  onClick={() => setOutputTransform({ flip_h: !alignment.flip_h })}
                  title="Mirror left↔right — undoes a projector in rear-projection mode"
                >
                  ⇋ Horizontal
                </Btn>
                <Btn
                  size="sm"
                  active={!!alignment.flip_v}
                  onClick={() => setOutputTransform({ flip_v: !alignment.flip_v })}
                  title="Mirror top↔bottom — undoes a ceiling-mounted projector. Both flips = rotate 180°"
                >
                  ⇵ Vertical
                </Btn>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-fg3 w-10">Fit</span>
                <Seg<OutputFit>
                  value={alignment.fit ?? 'stretch'}
                  onChange={(fit) => setOutputTransform({ fit })}
                  options={[
                    { id: 'stretch', label: 'Stretch', title: 'Fill the output, ignoring aspect' },
                    { id: 'fit', label: 'Fit', title: 'Keep aspect, black bars' },
                    { id: 'fill', label: 'Fill', title: 'Keep aspect, crop the overflow' },
                  ]}
                />
                <span className="text-[10px] text-fg3 num">
                  {alignment.output[0]}×{alignment.output[1]}
                </span>
              </div>
              <div className="text-[10px] text-fg3 leading-snug">
                Saved with this project. Flips don't move your alignment corners.
              </div>
            </div>
          ) : (
            <div className="px-1 text-[11px] text-fg3">Open a project to set flips and fit.</div>
          )}

          <div className="label px-1 pt-3 pb-1.5 mt-2 border-t border-line">While live</div>
          <Check
            checked={state?.hide_cursor ?? true}
            onChange={(v) => setOutputOptions({ hideCursor: v })}
            label="Hide the cursor over the output"
          />
          <Check
            checked={state?.keep_awake ?? true}
            onChange={(v) => setOutputOptions({ keepAwake: v })}
            label="Keep displays awake"
          />

          <div className="flex items-center gap-1.5 px-1 pt-3 mt-2 border-t border-line">
            <Btn
              size="sm"
              onClick={() => {
                setRoute('align');
                setOpen(false);
              }}
              title="Corner pin + warp handles (⌘3)"
            >
              Align corners…
            </Btn>
          </div>
        </Popover>
      )}
    </>
  );
}

function Row({
  active,
  disabled,
  onClick,
  title,
  sub,
  warn,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  title: string;
  sub: string;
  warn?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={
        'flex items-center gap-2 px-2 py-1.5 rounded-md text-left border ' +
        (active ? 'bg-panel2 border-line2' : 'border-transparent hover:bg-panel2') +
        (disabled ? ' opacity-45 pointer-events-none' : '')
      }
    >
      <Dot tone={active ? 'ok' : 'off'} />
      <div className="min-w-0 flex-1">
        <div className="text-[12px] font-medium truncate">
          {title}
          {warn && <span className="ml-1.5 text-amber" title="Not the display's native mode">●</span>}
        </div>
        <div className="text-[10px] text-fg3 truncate">{sub}</div>
      </div>
    </button>
  );
}

function Note({ tone, children }: { tone: 'warn' | 'bad'; children: React.ReactNode }) {
  return (
    <div
      className={
        'mt-2 mx-1 px-2 py-1.5 rounded-md text-[11px] leading-snug border ' +
        (tone === 'bad' ? 'border-bad/50 text-bad bg-bad/10' : 'border-amber/50 text-amber bg-amber/10')
      }
    >
      {children}
    </div>
  );
}

function Check({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 px-2 py-1 cursor-pointer text-[12px]">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-[rgb(var(--deck))]"
      />
      {label}
    </label>
  );
}
