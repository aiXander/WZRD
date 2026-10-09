// Modulation depth: where a param rests (FROM) and where the music throws it
// (TO). The band is the sweep, the bright marker rides it live. Press near
// an end to drag that end; press inside the band to slide the whole sweep.

import { useRef, useState } from 'react';
import { defaultFormat } from './Fader';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function RangeFader({
  from,
  to,
  min,
  max,
  live,
  label,
  onChange,
  disabled,
  title,
}: {
  from: number;
  to: number;
  min: number;
  max: number;
  live?: number | null;
  label?: React.ReactNode;
  onChange: (from: number, to: number) => void;
  disabled?: boolean;
  title?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [local, setLocal] = useState<[number, number] | null>(null);
  const drag = useRef<{ mode: 'from' | 'to' | 'both'; start: number; f0: number; t0: number } | null>(null);
  const span = max - min || 1;
  const [f, t] = local ?? [from, to];
  const frac = (v: number) => clamp((v - min) / span, 0, 1);
  const valAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return min + clamp((clientX - r.left) / r.width, 0, 1) * span;
  };

  function down(e: React.PointerEvent) {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const v = valAt(e.clientX);
    const r = ref.current!.getBoundingClientRect();
    const px = (x: number) => frac(x) * r.width;
    const p = e.clientX - r.left;
    const dF = Math.abs(p - px(f));
    const dT = Math.abs(p - px(t));
    const inside = p > Math.min(px(f), px(t)) + 8 && p < Math.max(px(f), px(t)) - 8;
    const mode = inside ? 'both' : dF <= dT ? 'from' : 'to';
    drag.current = { mode, start: v, f0: f, t0: t };
    if (mode !== 'both') move(e);
  }
  function move(e: React.PointerEvent) {
    const d = drag.current;
    if (!d) return;
    const v = valAt(e.clientX);
    let nf = d.f0;
    let nt = d.t0;
    if (d.mode === 'from') nf = v;
    else if (d.mode === 'to') nt = v;
    else {
      const delta = clamp(v - d.start, min - Math.min(d.f0, d.t0), max - Math.max(d.f0, d.t0));
      nf = d.f0 + delta;
      nt = d.t0 + delta;
    }
    setLocal([nf, nt]);
    onChange(nf, nt);
  }
  function up() {
    drag.current = null;
    setLocal(null);
  }

  const lo = Math.min(frac(f), frac(t));
  const hi = Math.max(frac(f), frac(t));
  const fmt = (v: number) => defaultFormat(v, span);
  return (
    <div
      ref={ref}
      title={title}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      className={
        'relative h-[26px] rounded-[5px] bg-panel2 border border-line overflow-hidden touch-none select-none ' +
        (disabled ? 'opacity-60' : 'hover:border-line2')
      }
    >
      <div className="absolute top-0 bottom-0 bg-mod/20" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
      {/* FROM end (rest) — solid; TO end (peak) — glowing */}
      <div className="absolute top-0 bottom-0 w-[2px] bg-mod/60" style={{ left: `calc(${frac(f) * 100}% - 1px)` }} />
      <div className="absolute top-0 bottom-0 w-[2px] bg-mod" style={{ left: `calc(${frac(t) * 100}% - 1px)` }} />
      {live != null && (
        <div
          className="absolute top-[3px] bottom-[3px] w-[4px] rounded-full bg-white shadow-[0_0_10px_rgb(var(--mod))]"
          style={{ left: `calc(${frac(live) * 100}% - 2px)` }}
        />
      )}
      <div className="absolute inset-0 flex items-center justify-between px-2 pointer-events-none gap-2">
        <span className="truncate text-[11px] text-fg2">{label}</span>
        <span className="num shrink-0 text-[11px] text-fg2">
          {fmt(f)} <span className="text-mod">→</span> {fmt(t)}
        </span>
      </div>
    </div>
  );
}
