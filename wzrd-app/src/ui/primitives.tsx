// Small shared building blocks. Everything visual in the app composes these
// plus <Fader>, so spacing, type and colour stay one system.

import { useRef } from 'react';

type BtnTone = 'ghost' | 'solid' | 'deck' | 'live' | 'design' | 'danger' | 'amber';

const TONE: Record<BtnTone, { on: string; off: string }> = {
  ghost: {
    on: 'bg-raised text-fg border-line2',
    off: 'bg-transparent text-fg2 border-transparent hover:bg-raised/70 hover:text-fg',
  },
  solid: {
    on: 'bg-line2 text-fg border-line2',
    off: 'bg-panel2 text-fg2 border-line hover:bg-raised hover:text-fg hover:border-line2',
  },
  deck: {
    on: 'bg-deck/20 text-fg border-deck/70',
    off: 'bg-panel2 text-fg2 border-line hover:border-deck/50 hover:text-fg',
  },
  live: {
    on: 'bg-live text-white border-live',
    off: 'bg-panel2 text-fg2 border-line hover:border-live/60 hover:text-fg',
  },
  design: {
    on: 'bg-design text-black border-design',
    off: 'bg-panel2 text-fg2 border-line hover:border-design/60 hover:text-fg',
  },
  danger: {
    on: 'bg-bad/25 text-fg border-bad',
    off: 'bg-panel2 text-fg2 border-line hover:border-bad/70 hover:text-bad',
  },
  amber: {
    on: 'bg-amber/25 text-amber border-amber/70',
    off: 'bg-panel2 text-fg2 border-line hover:border-amber/60 hover:text-amber',
  },
};

export function Btn({
  children,
  onClick,
  active = false,
  tone = 'solid',
  size = 'md',
  title,
  disabled,
  className = '',
  square,
}: {
  children: React.ReactNode;
  onClick?: (e: React.MouseEvent) => void;
  active?: boolean;
  tone?: BtnTone;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  title?: string;
  disabled?: boolean;
  className?: string;
  square?: boolean;
}) {
  const sz =
    size === 'xs'
      ? square
        ? 'h-5 w-5 text-[10px]'
        : 'h-5 px-1.5 text-[10px]'
      : size === 'sm'
      ? square
        ? 'h-6 w-6 text-[11px]'
        : 'h-6 px-2 text-[11px]'
      : size === 'lg'
      ? 'h-10 px-4 text-[13px]'
      : square
      ? 'h-7 w-7 text-xs'
      : 'h-7 px-2.5 text-xs';
  const t = TONE[tone];
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={
        'inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors duration-75 whitespace-nowrap ' +
        sz +
        ' ' +
        (active ? t.on : t.off) +
        (disabled ? ' opacity-40 pointer-events-none' : '') +
        ' ' +
        className
      }
    >
      {children}
    </button>
  );
}

/** Segmented control. */
export function Seg<T extends string>({
  value,
  options,
  onChange,
  size = 'sm',
  className = '',
}: {
  value: T;
  options: { id: T; label: React.ReactNode; title?: string }[];
  onChange: (v: T) => void;
  size?: 'xs' | 'sm';
  className?: string;
}) {
  return (
    <div className={'inline-flex rounded-md border border-line bg-panel2 p-[2px] ' + className}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          title={o.title}
          onClick={() => onChange(o.id)}
          className={
            'rounded-[4px] font-medium transition-colors ' +
            (size === 'xs' ? 'h-[18px] px-1.5 text-[10px]' : 'h-[22px] px-2 text-[11px]') +
            ' ' +
            (value === o.id ? 'bg-raised text-fg shadow-inset' : 'text-fg3 hover:text-fg')
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Panel({
  title,
  right,
  children,
  className = '',
  bodyClassName = '',
  hole = false,
}: {
  title?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  /** A `<NativeSlot>` lives inside: paint no background, or it would hide
   *  the native preview under the page. */
  hole?: boolean;
}) {
  return (
    <section
      className={'flex flex-col min-h-0 min-w-0 border border-line rounded-lg ' + (hole ? '' : 'bg-panel ') + className}
    >
      {(title || right) && (
        <header className="flex items-center gap-2 h-8 px-3 border-b border-line shrink-0">
          <div className="label flex items-center gap-2 min-w-0 truncate">{title}</div>
          <div className="flex-1" />
          {right}
        </header>
      )}
      <div className={'flex-1 min-h-0 ' + bodyClassName}>{children}</div>
    </section>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[16px] h-[16px] px-1 rounded border border-line2 bg-panel2 text-[9px] text-fg3 font-mono leading-none">
      {children}
    </kbd>
  );
}

export function Dot({ tone }: { tone: 'ok' | 'warn' | 'bad' | 'off' | 'info' }) {
  const c =
    tone === 'ok'
      ? 'bg-ok shadow-[0_0_6px_rgb(var(--ok))]'
      : tone === 'warn'
      ? 'bg-amber shadow-[0_0_6px_rgb(var(--amber))]'
      : tone === 'bad'
      ? 'bg-bad shadow-[0_0_6px_rgb(var(--bad))]'
      : tone === 'info'
      ? 'bg-design'
      : 'bg-fg4';
  return <span className={'inline-block w-[7px] h-[7px] rounded-full ' + c} />;
}

/** Vertical level meter (0..1). */
export function Meter({ value, color, className = '' }: { value: number; color: string; className?: string }) {
  const pct = Math.min(1, Math.max(0, value)) * 100;
  return (
    <div className={'relative w-2 rounded-sm bg-panel2 overflow-hidden ' + className}>
      <div
        className="absolute left-0 right-0 bottom-0 transition-[height] duration-[60ms] ease-linear"
        style={{ height: `${pct}%`, background: color }}
      />
    </div>
  );
}

/** Onset lamp — lights with the envelope. */
export function Lamp({ value, color, label }: { value: number; color: string; label: string }) {
  const v = Math.min(1, Math.max(0, value));
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="w-5 h-5 rounded-full border border-line2"
        style={{
          background: v > 0.04 ? color : 'transparent',
          opacity: 0.25 + v * 0.75,
          boxShadow: v > 0.2 ? `0 0 ${6 + v * 14}px ${color}` : undefined,
        }}
      />
      <div className="text-[9px] font-semibold tracking-wider text-fg3">{label}</div>
    </div>
  );
}

/** Inline editable text (double-click to edit). */
export function InlineEdit({
  value,
  onCommit,
  className = '',
}: {
  value: string;
  onCommit: (v: string) => void;
  className?: string;
}) {
  const ref = useRef<HTMLInputElement | null>(null);
  return (
    <input
      ref={ref}
      defaultValue={value}
      key={value}
      spellCheck={false}
      className={
        'bg-transparent border border-transparent rounded px-1 -mx-1 outline-none focus:border-line2 focus:bg-panel2 min-w-0 ' +
        className
      }
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Enter') ref.current?.blur();
        if (e.key === 'Escape') {
          if (ref.current) ref.current.value = value;
          ref.current?.blur();
        }
      }}
      onBlur={(e) => {
        const v = e.target.value.trim();
        if (v && v !== value) onCommit(v);
        else e.target.value = value;
      }}
    />
  );
}
