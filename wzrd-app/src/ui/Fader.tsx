// The one continuous control in the app — a Resolume-style value bar.
//
//   horizontal: label left, value right, both *inside* the bar (dense param
//               lists read like a mixing desk, not a form)
//   vertical:   a mixer fader with a cap, value on top, label below
//
// Interaction (identical everywhere, so muscle memory transfers):
//   press        jump to the pointer and keep dragging
//   ⇧ + drag     fine: relative, 1/10 speed (no jump on press)
//   double-click reset (`onReset`, else back to `defaultValue`)
//   ← → ↑ ↓      nudge when focused (1% of the range, ⇧ = 0.1%)
//
// No mouse wheel, on purpose (operator feedback 2026-10-14): a wheel over a
// fader must scroll the panel it sits in — a scroll through the layer mixer
// used to nudge every fader the pointer crossed. The value path is
// `onChange`/`onCommit` → an action in state/actions.ts, never DOM-only, so a
// hardware controller (MIDI, docs/TODO/midi-controller.md) drives the same
// actions without touching this component.
//
// "A knob never jumps under my finger": while a drag is in flight the bar
// shows its own value and ignores the prop, so a stale telemetry frame can't
// yank it. Optional overlays: `mod` (a modulation range band + the live value
// riding inside it) and `ghost` (where the scene value sits when an override
// pins the param elsewhere).

import { useCallback, useRef, useState } from 'react';

export type FaderProps = {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  onCommit?: (v: number) => void;
  onReset?: () => void;
  defaultValue?: number;
  label?: React.ReactNode;
  format?: (v: number) => string;
  /** CSS colour for the fill (e.g. `rgb(var(--deck))`). */
  color?: string;
  orientation?: 'h' | 'v';
  size?: 'sm' | 'md' | 'lg';
  step?: number;
  disabled?: boolean;
  /** Draw from 0 instead of `min` when the range straddles zero. */
  bipolar?: boolean;
  /** Modulation: the range band (from, to) and the live value inside it. */
  mod?: { from: number; to: number; live?: number } | null;
  /** Faint tick where the un-overridden value sits. */
  ghost?: number | null;
  /** Live readout dot riding the bar (e.g. a driver's current output). */
  live?: number | null;
  title?: string;
  className?: string;
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function defaultFormat(v: number, span: number): string {
  const a = Math.abs(v);
  if (span >= 50 || a >= 100) return v.toFixed(0);
  if (span >= 5 || a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function Fader({
  value,
  min,
  max,
  onChange,
  onCommit,
  onReset,
  defaultValue,
  label,
  format,
  color = 'rgb(var(--deck))',
  orientation = 'h',
  size = 'md',
  step,
  disabled,
  bipolar,
  mod,
  ghost,
  live,
  title,
  className = '',
}: FaderProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const dragRef = useRef<{ startPos: number; startVal: number; fine: boolean } | null>(null);
  const span = max - min || 1;
  const shown = drag ?? value;
  const fmt = format ?? ((v: number) => defaultFormat(v, span));
  const vert = orientation === 'v';

  const quant = useCallback(
    (v: number) => {
      const c = clamp(v, min, max);
      return step ? Math.round(c / step) * step : c;
    },
    [min, max, step]
  );

  const fracAt = (e: { clientX: number; clientY: number }) => {
    const r = ref.current!.getBoundingClientRect();
    return vert ? 1 - (e.clientY - r.top) / r.height : (e.clientX - r.left) / r.width;
  };
  const posOf = (e: { clientX: number; clientY: number }) => (vert ? -e.clientY : e.clientX);
  const lengthPx = () => {
    const r = ref.current!.getBoundingClientRect();
    return vert ? r.height : r.width;
  };

  function emit(v: number) {
    setDrag(v);
    onChange(v);
  }

  function onPointerDown(e: React.PointerEvent) {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    ref.current?.focus();
    const fine = e.shiftKey;
    dragRef.current = { startPos: posOf(e), startVal: value, fine };
    if (!fine) emit(quant(min + fracAt(e) * span));
    else setDrag(value);
  }
  function onPointerMove(e: React.PointerEvent) {
    const d = dragRef.current;
    if (!d) return;
    if (d.fine || e.shiftKey) {
      // Re-anchor when ⇧ is pressed mid-drag so the switch never jumps.
      if (!d.fine) {
        d.fine = true;
        d.startPos = posOf(e);
        d.startVal = drag ?? value;
      }
      const delta = ((posOf(e) - d.startPos) / lengthPx()) * span * 0.1;
      emit(quant(d.startVal + delta));
    } else {
      emit(quant(min + fracAt(e) * span));
    }
  }
  function onPointerUp() {
    if (!dragRef.current) return;
    dragRef.current = null;
    const v = drag ?? value;
    setDrag(null);
    onCommit?.(v);
  }
  function reset() {
    if (disabled) return;
    if (onReset) onReset();
    else if (defaultValue != null) {
      onChange(defaultValue);
      onCommit?.(defaultValue);
    }
  }
  function nudge(dir: number, fine: boolean) {
    const v = quant(value + dir * span * (fine ? 0.001 : 0.01));
    onChange(v);
    onCommit?.(v);
  }

  const f = (v: number) => clamp((v - min) / span, 0, 1);
  const origin = bipolar && min < 0 && max > 0 ? f(0) : 0;
  const pos = f(shown);
  const lo = Math.min(origin, pos);
  const hi = Math.max(origin, pos);
  const active = drag != null;

  const common = {
    ref,
    tabIndex: disabled ? -1 : 0,
    title,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel: onPointerUp,
    onDoubleClick: reset,
    onKeyDown: (e: React.KeyboardEvent) => {
      const k = e.key;
      if (k === 'ArrowRight' || k === 'ArrowUp') nudge(1, e.shiftKey);
      else if (k === 'ArrowLeft' || k === 'ArrowDown') nudge(-1, e.shiftKey);
      else return;
      e.preventDefault();
      e.stopPropagation();
    },
  };

  if (vert) {
    return (
      <div className={'flex flex-col items-center gap-1.5 select-none ' + className}>
        <div className={'num text-[11px] ' + (active ? 'text-fg' : 'text-fg2')}>{fmt(shown)}</div>
        <div
          {...common}
          className={
            'relative flex-1 w-full min-h-[60px] rounded-md bg-panel2 border border-line overflow-hidden touch-none ' +
            (disabled ? 'opacity-40' : 'hover:border-line2')
          }
        >
          {/* 25% ticks, like a printed fader scale */}
          {[0.25, 0.5, 0.75].map((t) => (
            <div key={t} className="absolute left-0 right-0 h-px bg-line" style={{ bottom: `${t * 100}%` }} />
          ))}
          <div
            className="absolute left-0 right-0"
            style={{ bottom: `${lo * 100}%`, height: `${(hi - lo) * 100}%`, background: color, opacity: 0.28 }}
          />
          {ghost != null && (
            <div className="absolute left-0 right-0 h-px bg-fg3" style={{ bottom: `${f(ghost) * 100}%` }} />
          )}
          {/* cap */}
          <div
            className="absolute left-0.5 right-0.5 h-[6px] -mb-[3px] rounded-sm"
            style={{
              bottom: `${pos * 100}%`,
              background: color,
              boxShadow: active ? `0 0 12px ${color}` : undefined,
            }}
          />
        </div>
        {label != null && <div className="label text-center leading-tight">{label}</div>}
      </div>
    );
  }

  const h = size === 'sm' ? 'h-[20px]' : size === 'lg' ? 'h-[34px]' : 'h-[26px]';
  return (
    <div
      {...common}
      className={
        'relative rounded-[5px] bg-panel2 border border-line overflow-hidden touch-none select-none ' +
        h +
        ' ' +
        (disabled ? 'opacity-40 ' : 'hover:border-line2 ') +
        className
      }
    >
      <div
        className="absolute top-0 bottom-0"
        style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%`, background: color, opacity: active ? 0.42 : 0.3 }}
      />
      {mod && (
        <>
          <div
            className="absolute top-0 bottom-0 bg-mod/20 border-x border-mod/70"
            style={{ left: `${Math.min(f(mod.from), f(mod.to)) * 100}%`, width: `${Math.abs(f(mod.to) - f(mod.from)) * 100}%` }}
          />
          {mod.live != null && (
            <div
              className="absolute top-0 bottom-0 w-[2px] bg-mod shadow-[0_0_8px_rgb(var(--mod))]"
              style={{ left: `calc(${f(mod.live) * 100}% - 1px)` }}
            />
          )}
        </>
      )}
      {ghost != null && (
        <div className="absolute top-1 bottom-1 w-px bg-fg3/70" style={{ left: `${f(ghost) * 100}%` }} />
      )}
      {!mod && (
        <div
          className="absolute top-0 bottom-0 w-[2px]"
          style={{ left: `calc(${pos * 100}% - 1px)`, background: color, boxShadow: active ? `0 0 10px ${color}` : undefined }}
        />
      )}
      {live != null && (
        <div
          className="absolute bottom-0 h-[3px] w-[3px] rounded-full bg-fg"
          style={{ left: `calc(${f(live) * 100}% - 1px)` }}
        />
      )}
      <div className="absolute inset-0 flex items-center justify-between px-2 pointer-events-none gap-2">
        <span className={'truncate ' + (size === 'sm' ? 'text-[10px]' : 'text-[11px]') + ' text-fg2'}>{label}</span>
        <span className={'num shrink-0 ' + (size === 'sm' ? 'text-[10px]' : 'text-[11px]') + (active ? ' text-fg' : ' text-fg2')}>
          {fmt(shown)}
        </span>
      </div>
    </div>
  );
}
