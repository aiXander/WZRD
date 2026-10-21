// Ask for an asset name (show-setup A1): a popover with a text field and
// suggestion chips (the well-known roles first). Names follow the hub's rule
// `^[a-z0-9][a-z0-9_-]{0,47}$`; typing is lower-cased and spaces become `_`.
// Enter submits, Esc / outside click closes.

import { useEffect, useRef, useState } from 'react';
import { Popover } from '../Popover';
import { Btn } from '../primitives';

const ASSET_NAME = /^[a-z0-9][a-z0-9_-]{0,47}$/;

export function NamePrompt({
  anchor,
  suggestions,
  initial = '',
  title = 'Pin as…',
  onSubmit,
  onClose,
}: {
  anchor: HTMLElement | null;
  suggestions: string[];
  initial?: string;
  title?: string;
  onSubmit: (name: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState(initial);
  const ref = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  const ok = ASSET_NAME.test(name);
  const submit = (n: string) => {
    if (!ASSET_NAME.test(n)) return;
    onSubmit(n);
    onClose();
  };
  return (
    <Popover anchor={anchor} onClose={onClose} width={280}>
      <div className="label mb-1.5">{title}</div>
      <div className="flex gap-1.5">
        <input
          ref={ref}
          value={name}
          spellCheck={false}
          onChange={(e) => setName(e.target.value.toLowerCase().replace(/\s+/g, '_'))}
          onKeyDown={(e) => {
            if (e.key === 'Escape') return; // the popover closes itself
            e.stopPropagation();
            if (e.key === 'Enter') {
              e.preventDefault();
              submit(name);
            }
          }}
          placeholder={suggestions[0] ?? 'asset_name'}
          className={
            'flex-1 min-w-0 h-7 bg-bg border rounded px-2 text-[12px] font-mono outline-none ' +
            (name && !ok ? 'border-bad/70' : 'border-line2 focus:border-deck/60')
          }
        />
        <Btn size="md" tone="deck" disabled={!ok} onClick={() => submit(name)}>
          Pin
        </Btn>
      </div>
      {name && !ok && (
        <div className="mt-1 text-[10px] text-bad">a–z, 0–9, _ and -; starts with a letter or digit; ≤ 48</div>
      )}
      {suggestions.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => submit(s)}
              className={
                'text-[11px] font-mono px-1.5 py-[1px] rounded border ' +
                (s === name ? 'border-deck/70 text-fg bg-deck/15' : 'border-line2 text-fg2 hover:text-fg hover:border-fg4')
              }
              title={`Pin as ${s}`}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </Popover>
  );
}
