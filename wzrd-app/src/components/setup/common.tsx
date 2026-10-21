// Small building blocks shared by the show-setup steps: a labelled field, a
// slider with its value and a double-click reset, the step's two-column body,
// a spinner, a framed note.

import { mediaUrl } from '../../api/hub';

export function Field({ label, hint, children }: { label: React.ReactNode; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    // A div, not a <label>: a label forwards clicks on its text to its first
    // button (a Seg would flip to its first option).
    <div className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="text-[10px] text-fg3 leading-snug">{hint}</span>}
    </div>
  );
}

/** A native range with its value; double-click resets to `def`. */
export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  def,
  digits = 2,
  hint,
  pct,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  def?: number;
  digits?: number;
  hint?: string;
  /** Show the value as a percentage. */
  pct?: boolean;
}) {
  const shown = pct ? `${(value * 100).toFixed(Math.max(0, digits - 2))} %` : value.toFixed(digits);
  return (
    <div className="flex flex-col gap-1 min-w-0" title={hint}>
      <div className="flex items-baseline gap-2">
        <span className="label flex-1 truncate">{label}</span>
        <span className={'num text-[11px] ' + (def !== undefined && Math.abs(value - def) > 1e-9 ? 'text-fg' : 'text-fg3')}>{shown}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => def !== undefined && onChange(def)}
        className="w-full accent-[rgb(var(--deck))]"
      />
    </div>
  );
}

/** A step's body: the working area left, settings right. */
export function StepBody({ main, side }: { main: React.ReactNode; side: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-0">
      <div className="flex-1 min-w-0 min-h-0 overflow-y-auto p-4 flex flex-col gap-4">{main}</div>
      <aside className="w-[300px] shrink-0 border-l border-line overflow-y-auto p-4 flex flex-col gap-3.5 text-[12px] bg-panel2/40">
        {side}
      </aside>
    </div>
  );
}

export function SectionTitle({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="text-[12px] font-semibold text-fg truncate">{children}</span>
      <div className="flex-1" />
      {right}
    </div>
  );
}

export function Spinner({ className = '' }: { className?: string }) {
  return (
    <span
      className={'inline-block w-3.5 h-3.5 rounded-full border-2 border-fg3/40 border-t-fg animate-spin ' + className}
      aria-label="working"
    />
  );
}

export function Note({ tone = 'info', children }: { tone?: 'info' | 'warn' | 'bad' | 'ok'; children: React.ReactNode }) {
  const c =
    tone === 'bad'
      ? 'border-bad/60 bg-bad/10 text-fg'
      : tone === 'warn'
      ? 'border-amber/60 bg-amber/10 text-fg'
      : tone === 'ok'
      ? 'border-ok/50 bg-ok/10 text-fg'
      : 'border-line bg-panel2 text-fg2';
  return <div className={'rounded-md border px-3 py-2 text-[11px] leading-snug ' + c}>{children}</div>;
}

/** An image from disk, by absolute path. */
export function Img({ path, className = '', alt = '' }: { path: string; className?: string; alt?: string }) {
  return <img src={mediaUrl(path)} alt={alt} draggable={false} className={className} />;
}

export { baseName } from '../../ui/media/types';
