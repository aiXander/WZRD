// The modulation editor for one scalar param: the chip on the row (source +
// shape glyph + rate at a glance) and the popover behind it — a live scope,
// the source picker (Audio / Clock), shape + rate for clocks, decay for
// onsets, invert, remove. Everything here is a scene edit → DESIGN only (the
// caller disables the chip on LIVE).
//
// Perf: the scope draws on a canvas from a store subscription + rAF, only
// while the popover is open — it never re-renders React per telemetry frame
// and asks for no channel the driven row doesn't already demand (`drivers`).

import { useEffect, useRef } from 'react';
import {
  FREE_RATES,
  MOD_SHAPES,
  MOD_SOURCES,
  TEMPO_RATES,
  modPeriod,
  modShapeOf,
  modSourceOf,
  modulate,
  setModDecay,
  setModRange,
  setModRate,
  setModShape,
  type ModShape,
} from '../state/actions';
import { isDriver } from '../state/scene';
import { useStore } from '../state/store';
import { Fader, defaultFormat } from '../ui/Fader';
import { Seg } from '../ui/primitives';

// ---------- shapes ----------

const fract = (x: number) => x - Math.floor(x);

/** UI mirror of the engine's `Shape::eval` (drivers.rs) for glyphs. The
 *  random one only has to *look* random — the engine's hash differs. */
function shapeAt(shape: ModShape, pos: number, seed = 3): number {
  const k = Math.floor(pos);
  const x = pos - k;
  switch (shape) {
    case 'saw':
      return x;
    case 'ramp_down':
      return 1 - x;
    case 'pingpong':
      return 1 - Math.abs(2 * x - 1);
    case 'sine':
      return 0.5 - 0.5 * Math.cos(2 * Math.PI * x);
    case 'square':
      return x < 0.5 ? 1 : 0;
    case 'random': {
      const h = (i: number) => fract(Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453);
      const s = x * x * (3 - 2 * x);
      return h(k) + (h(k + 1) - h(k)) * s;
    }
  }
}

const glyphCache = new Map<string, string>();
function glyphPath(shape: ModShape, w: number, h: number, cycles: number): string {
  const key = `${shape}:${w}:${h}:${cycles}`;
  const hit = glyphCache.get(key);
  if (hit) return hit;
  const pad = 1.5;
  const ts: number[] = [];
  for (let i = 0; i <= 48; i++) ts.push((i / 48) * cycles);
  // Sample both sides of every jump so ramps/squares draw true verticals.
  for (let k = 0; k <= cycles; k++) ts.push(k - 1e-4, k + 0.5 - 1e-4, k + 0.5);
  const pts = ts
    .filter((t) => t >= 0 && t <= cycles - 1e-4)
    .sort((a, b) => a - b)
    .map((t) => `${((t / cycles) * (w - 2 * pad) + pad).toFixed(2)},${((1 - shapeAt(shape, t)) * (h - 2 * pad) + pad).toFixed(2)}`);
  const d = 'M' + pts.join('L');
  glyphCache.set(key, d);
  return d;
}

function ShapeGlyph({
  shape,
  width = 22,
  height = 12,
  cycles = 2,
  className = '',
}: {
  shape: ModShape;
  width?: number;
  height?: number;
  cycles?: number;
  className?: string;
}) {
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={'shrink-0 ' + className} aria-hidden>
      <path d={glyphPath(shape, width, height, cycles)} fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinejoin="round" />
    </svg>
  );
}

// ---------- labels ----------

const plural = (n: number, unit: string) => `${+n.toFixed(2)} ${unit}${n === 1 ? '' : 's'}`;

/** Short rate label: "1 beat", "4 bars", "8 s". */
function rateLabel(raw: any): string | null {
  const p = modPeriod(raw);
  if (p == null) return null;
  if (raw.driver === 'clock.phase') return `${+p.toFixed(p < 10 ? 1 : 0)} s`;
  return p < 1 ? plural(p * 4, 'beat') : plural(p, 'bar');
}

const fracLabel = (p: number) => (p === 0.25 ? '¼' : p === 0.5 ? '½' : String(p));

// ---------- chip ----------

/** The row's modulation chip: `+ MOD`, or the source at a glance. */
export function ModChip({
  raw,
  disabled,
  onClick,
}: {
  raw: any;
  disabled: boolean;
  onClick: (el: HTMLElement) => void;
}) {
  const driver = isDriver(raw) && raw.driver !== 'ui.slider' ? raw : null;
  const src = modSourceOf(driver);
  const shape = modShapeOf(driver);
  const text = driver ? (shape ? rateLabel(driver) : src?.label ?? driver.driver.replace(/^.*\./, '')) : null;
  const long = driver ? [src?.label, shape && MOD_SHAPES.find((s) => s.id === shape)?.label, shape && rateLabel(driver)].filter(Boolean).join(' · ') : '';
  return (
    <button
      onClick={disabled ? undefined : (e) => onClick(e.currentTarget)}
      title={
        disabled
          ? driver
            ? `Driven by ${long}. Change it on DESIGN.`
            : 'Modulation is wired on DESIGN.'
          : driver
          ? `Driven by ${long} — click to edit or remove`
          : 'Make this move: route a kick, a band or a clock shape onto it'
      }
      className={
        'h-[26px] rounded-[5px] border text-[10px] font-semibold tracking-wide px-1.5 flex items-center justify-center gap-1 min-w-0 ' +
        (driver
          ? 'border-mod/60 bg-mod/15 text-mod'
          : disabled
          ? 'border-line text-fg4'
          : 'border-line2 text-fg3 hover:text-mod hover:border-mod/60') +
        (disabled ? ' cursor-not-allowed' : '')
      }
    >
      {driver ? (
        <>
          {shape ? <ShapeGlyph shape={shape} width={14} height={10} /> : <span>∿</span>}
          <span className="truncate">{text}</span>
        </>
      ) : (
        '+ MOD'
      )}
    </button>
  );
}

// ---------- editor ----------

export function ModEditor({
  bindingId,
  param,
  raw,
  current,
  lo,
  hi,
  onClose,
}: {
  bindingId: string;
  param: string;
  raw: any;
  /** The param's still value — where a fresh modulation starts. */
  current: number;
  lo: number;
  hi: number;
  onClose: () => void;
}) {
  const driver = isDriver(raw) && raw.driver !== 'ui.slider' ? raw : null;
  const src = modSourceOf(driver);
  const shape = modShapeOf(driver);
  const period = modPeriod(driver);
  const range: [number, number] = driver && Array.isArray(driver.range) ? [driver.range[0], driver.range[1]] : [0, 1];
  const pick = (id: string) => modulate(bindingId, param, id, current, lo, hi);
  const fmt = (v: number) => defaultFormat(v, hi - lo || 1);

  const srcBtn = (m: (typeof MOD_SOURCES)[number]) => (
    <button
      key={m.id}
      title={m.title}
      onClick={() => pick(m.id)}
      className={
        'h-7 rounded-md text-[11px] font-medium border ' +
        (src?.id === m.id ? 'bg-mod text-black border-mod' : 'bg-panel2 border-line text-fg2 hover:border-mod/60 hover:text-fg')
      }
    >
      {m.label}
    </button>
  );
  const rates = src?.id === 'free' ? FREE_RATES : TEMPO_RATES;

  return (
    <div className="flex flex-col gap-2.5">
      {driver && <Scope bindingId={bindingId} param={param} range={range} />}

      <Section label="Audio">
        <div className="grid grid-cols-3 gap-1">{MOD_SOURCES.filter((m) => m.group === 'audio').map(srcBtn)}</div>
      </Section>
      <Section label="Clock">
        <div className="grid grid-cols-2 gap-1">{MOD_SOURCES.filter((m) => m.group === 'clock').map(srcBtn)}</div>
      </Section>

      {shape && (
        <>
          <Section label="Shape" right={MOD_SHAPES.find((s) => s.id === shape)?.label}>
            <div className="grid grid-cols-6 gap-1">
              {MOD_SHAPES.map((s) => (
                <button
                  key={s.id}
                  title={`${s.label} — ${s.title}`}
                  onClick={() => setModShape(bindingId, param, s.id)}
                  className={
                    'h-7 rounded-md border flex items-center justify-center ' +
                    (shape === s.id
                      ? 'bg-mod/20 border-mod text-mod'
                      : 'bg-panel2 border-line text-fg3 hover:text-fg hover:border-mod/60')
                  }
                >
                  <ShapeGlyph shape={s.id} />
                </button>
              ))}
            </div>
          </Section>
          <Section label="Cycle" right={rateLabel(driver)}>
            <div className="flex items-center gap-1.5">
              <Seg
                size="xs"
                value={String(period)}
                options={rates.map((r) => ({
                  id: String(r),
                  label: src?.id === 'free' ? String(r) : fracLabel(r),
                  title: src?.id === 'free' ? `${r} s per cycle` : r < 1 ? plural(r * 4, 'beat') + ' per cycle' : plural(r, 'bar') + ' per cycle',
                }))}
                onChange={(v) => setModRate(bindingId, param, Number(v))}
              />
              <span className="text-[10px] text-fg4">{src?.id === 'free' ? 'sec' : 'bars'}</span>
            </div>
          </Section>
        </>
      )}

      {driver?.driver === 'audio.onset' && (
        <Section label="Decay">
          <Fader
            size="sm"
            label="glow"
            value={driver.decay ?? 0.15}
            min={0.02}
            max={1.5}
            color="rgb(var(--mod))"
            format={(v) => `${Math.round(v * 1000)}ms`}
            onChange={(v) => setModDecay(bindingId, param, v)}
            title="How long each hit glows"
          />
        </Section>
      )}

      {driver && (
        <div className="flex items-center gap-2 pt-2 border-t border-line">
          <span className="num text-[11px] text-fg2 flex-1">
            rest {fmt(range[0])} <span className="text-mod">→</span> peak {fmt(range[1])}
          </span>
          <button
            className="h-6 px-2 rounded-md border border-line bg-panel2 text-[10px] text-fg2 hover:text-fg hover:border-line2"
            title="Swap rest and peak — the source then pushes the value the other way"
            onClick={() => setModRange(bindingId, param, range[1], range[0])}
          >
            ⇄ invert
          </button>
          <button
            className="h-6 px-2 rounded-md border border-line bg-panel2 text-[10px] text-fg3 hover:text-bad hover:border-bad/60"
            title="Back to a still value (the rest end)"
            onClick={() => {
              modulate(bindingId, param, null, current, lo, hi);
              onClose();
            }}
          >
            remove
          </button>
        </div>
      )}
    </div>
  );
}

function Section({ label, right, children }: { label: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="label">{label}</span>
        {right && <span className="text-[10px] text-fg3">{right}</span>}
      </div>
      {children}
    </div>
  );
}

// ---------- scope ----------

const SCOPE_WINDOW_MS = 4000;

/** The last 4 s of the param's actual value, normalised rest (bottom) →
 *  peak (top). Samples arrive with the `drivers` channel (~10 Hz). */
function Scope({ bindingId, param, range }: { bindingId: string; param: string; range: [number, number] }) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const rangeRef = useRef(range);
  rangeRef.current = range;

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext('2d');
    if (!el || !ctx) return;
    const buf: { t: number; v: number }[] = [];
    const sample = (rows: { binding_id: string; param_name: string; value: number }[]) => {
      const r = rows.find((d) => d.binding_id === bindingId && d.param_name === param);
      if (r) buf.push({ t: performance.now(), v: r.value });
    };
    sample(useStore.getState().drivers);
    const unsub = useStore.subscribe((s, p) => {
      if (s.drivers !== p.drivers) sample(s.drivers);
    });
    const mod = getComputedStyle(document.documentElement).getPropertyValue('--mod').trim() || '232 107 255';
    const rgb = (a: number) => `rgba(${mod.split(/\s+/).join(',')},${a})`;
    const dpr = window.devicePixelRatio || 1;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (el.width !== Math.round(w * dpr)) {
        el.width = Math.round(w * dpr);
        el.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const now = performance.now();
      const t0 = now - SCOPE_WINDOW_MS;
      while (buf.length > 2 && buf[1].t < t0) buf.shift();
      const [from, to] = rangeRef.current;
      const span = to - from;
      const norm = (v: number) => (Math.abs(span) < 1e-9 ? 0.5 : Math.min(1, Math.max(0, (v - from) / span)));
      const pad = 3;
      const y = (v: number) => pad + (1 - norm(v)) * (h - 2 * pad);
      const x = (t: number) => ((t - t0) / SCOPE_WINDOW_MS) * w;
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      if (!buf.length) return;
      ctx.beginPath();
      buf.forEach((s, i) => (i ? ctx.lineTo(x(s.t), y(s.v)) : ctx.moveTo(Math.max(0, x(s.t)), y(s.v))));
      const last = buf[buf.length - 1];
      ctx.lineTo(w, y(last.v));
      ctx.strokeStyle = rgb(1);
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.lineTo(w, h);
      ctx.lineTo(Math.max(0, x(buf[0].t)), h);
      ctx.closePath();
      ctx.fillStyle = rgb(0.12);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(w - 3, y(last.v), 2.5, 0, Math.PI * 2);
      ctx.fill();
    };
    raf = requestAnimationFrame(draw);
    return () => {
      unsub();
      cancelAnimationFrame(raf);
    };
  }, [bindingId, param]);

  return (
    <div className="relative h-11 rounded-md bg-panel2 border border-line overflow-hidden">
      <canvas ref={canvas} className="absolute inset-0 w-full h-full" />
      <span className="absolute left-1.5 top-0.5 text-[8px] uppercase tracking-wider text-fg4">peak</span>
      <span className="absolute left-1.5 bottom-0.5 text-[8px] uppercase tracking-wider text-fg4">rest</span>
    </div>
  );
}
