// The small input Studio's T / N / M keys open (A2): add tags (autocomplete
// from every tag in use — S5), edit notes, or move the selection to another
// folder. Enter applies to every selected item, Esc cancels. Mouse-free on
// purpose: label 20 renders in under a minute.

import { useEffect, useMemo, useRef, useState } from 'react';
import { libraryTags } from '../../api/hub';
import { addTags, moveTo, setNotes, short, targets, useStudio } from '../../state/studio';

export type PromptKind = 'tag' | 'note' | 'move';

const TITLE: Record<PromptKind, string> = {
  tag: 'Add tags — comma separated, Enter to apply',
  note: 'Notes — ⌘Enter to save',
  move: 'Move to folder — pick one, Enter to move',
};

export function KeyPrompt({ kind, onClose }: { kind: PromptKind; onClose: () => void }) {
  const items = useMemo(() => targets(), []);
  const [text, setText] = useState(kind === 'note' ? items[0]?.notes ?? '' : '');
  const [allTags, setAllTags] = useState<string[]>([]);
  const [pick, setPick] = useState(0);
  const ref = useRef<HTMLInputElement & HTMLTextAreaElement>(null);
  useEffect(() => {
    ref.current?.focus();
    if (kind === 'tag') libraryTags().then((t) => setAllTags(Object.keys(t))).catch(() => {});
  }, [kind]);

  const folders = useMoveTargets(kind === 'move');
  const last = text.split(',').pop()!.trim().toLowerCase();
  const suggestions =
    kind === 'tag'
      ? allTags.filter((t) => last && t.toLowerCase().startsWith(last) && !items[0]?.tags.includes(t)).slice(0, 6)
      : kind === 'move'
      ? folders.filter((f) => f.toLowerCase().includes(text.toLowerCase())).slice(0, 10)
      : [];

  const apply = () => {
    if (kind === 'tag') addTags(items, text.split(','));
    else if (kind === 'note') setNotes(items, text);
    else {
      const dest = suggestions[pick] ?? (text.startsWith('/') ? text : null);
      if (dest) moveTo(items, dest);
    }
    onClose();
  };

  const onKey = (e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'Escape') return onClose();
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setPick((p) => Math.max(0, Math.min(suggestions.length - 1, p + (e.key === 'ArrowDown' ? 1 : -1))));
      return;
    }
    if (e.key === 'Tab' && kind === 'tag' && suggestions[pick]) {
      e.preventDefault();
      const parts = text.split(',');
      parts[parts.length - 1] = ' ' + suggestions[pick];
      setText(parts.join(',').replace(/^ /, '') + ', ');
      return;
    }
    if (e.key === 'Enter' && (kind !== 'note' || e.metaKey)) {
      e.preventDefault();
      apply();
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-start justify-center pt-[18vh]" onMouseDown={onClose}>
      <div className="w-[480px] rounded-xl border border-line2 bg-panel shadow-2xl p-3" onMouseDown={(e) => e.stopPropagation()}>
        <div className="text-[11px] text-fg3 mb-1.5">
          {TITLE[kind]} · {items.length} item{items.length === 1 ? '' : 's'}
        </div>
        {kind === 'note' ? (
          <textarea
            ref={ref}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            rows={4}
            className="w-full bg-bg border border-line2 rounded px-2 py-1 text-[13px] outline-none resize-none"
          />
        ) : (
          <input
            ref={ref}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setPick(0);
            }}
            onKeyDown={onKey}
            placeholder={kind === 'tag' ? 'moody, fire, slow' : 'type to filter folders, or an absolute path'}
            className="w-full h-8 bg-bg border border-line2 rounded px-2 text-[13px] outline-none"
          />
        )}
        {suggestions.length > 0 && (
          <div className="mt-1 flex flex-col">
            {suggestions.map((s, i) => (
              <button
                key={s}
                onMouseDown={(e) => {
                  e.preventDefault();
                  setPick(i);
                }}
                onDoubleClick={apply}
                className={'text-left px-2 py-1 rounded text-[12px] truncate ' + (i === pick ? 'bg-raised text-fg' : 'text-fg2')}
                title={s}
              >
                {kind === 'move' ? short(s) : s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** Folders to move into: the open folder's subfolders, its parent's, and
 *  every folder the tree has loaded. */
function useMoveTargets(active: boolean): string[] {
  const dir = useStudio((s) => s.dir);
  const listing = useStudio((s) => s.listing);
  const expanded = useStudio((s) => s.expanded);
  return useMemo(() => {
    if (!active) return [];
    const set = new Set<string>();
    listing?.dirs.forEach((d) => set.add(d.path));
    Object.entries(expanded).forEach(([p, kids]) => {
      set.add(p);
      kids.forEach((k) => set.add(k.path));
    });
    if (dir) set.delete(dir);
    return [...set].sort();
  }, [active, dir, listing, expanded]);
}
