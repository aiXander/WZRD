// The inspector's effect cards, structured: a "New" section (project effects
// no group lists yet — where agent-authored shaders land), the operator's
// named, foldable groups, then the unfiled built-ins. Right-click a card for
// Rename / Update thumbnail / Move to / Delete, a group header for Rename /
// Delete; cards also drag onto any section. State + disk: state/effectLibrary.ts.

import { useCallback, useState } from 'react';
import { prettify } from '../state/scene';
import { useStore } from '../state/store';
import {
  createGroup,
  deleteEffect,
  deleteGroup,
  groupOf,
  moveEffect,
  renameEffect,
  renameGroup,
  toggleGroup,
  updateThumbnail,
  useEffectLibrary,
} from '../state/effectLibrary';
import { ContextMenu, type MenuItem } from '../ui/ContextMenu';
import { Btn } from '../ui/primitives';

const DRAG_TYPE = 'application/x-wzrd-effect';

type Section = { id: string; name: string; effects: string[]; collapsed: boolean; fixed: boolean; accent?: boolean };
type Menu = { x: number; y: number; items: MenuItem[] };

export function EffectPicker({ current, onPick }: { current: string | null; onPick: (name: string) => void }) {
  const catalog = useStore((s) => s.catalog);
  const doc = useEffectLibrary((s) => s.doc);
  const thumbs = useEffectLibrary((s) => s.thumbs);
  const [menu, setMenu] = useState<Menu | null>(null);
  const [renamingGroup, setRenamingGroup] = useState<string | null>(null);
  const [renamingEffect, setRenamingEffect] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const closeMenu = useCallback(() => setMenu(null), []);

  const filed = new Set(doc?.groups.flatMap((g) => g.effects) ?? []);
  const unfiled = (kind: 'user' | 'builtin') =>
    Object.keys(catalog)
      .filter((n) => catalog[n].kind === kind && !filed.has(n))
      .sort((a, b) => a.localeCompare(b));
  const sections: Section[] = [
    { id: 'new', name: 'New', effects: unfiled('user'), collapsed: !!doc?.newCollapsed, fixed: true, accent: true },
    ...(doc?.groups ?? []).map((g) => ({
      id: g.id,
      name: g.name,
      effects: g.effects.filter((n) => catalog[n]),
      collapsed: !!g.collapsed,
      fixed: false,
    })),
    { id: 'builtin', name: 'Built-in', effects: unfiled('builtin'), collapsed: !!doc?.builtinCollapsed, fixed: true },
  ].filter((s) => !s.fixed || s.effects.length > 0);

  function newGroupWith(name?: string) {
    const id = createGroup();
    if (name) moveEffect(name, id);
    setRenamingGroup(id);
  }

  function cardMenu(e: React.MouseEvent, name: string) {
    e.preventDefault();
    const user = catalog[name]?.kind === 'user';
    const at = groupOf(doc, name);
    const items: MenuItem[] = [];
    if (user) {
      items.push(
        { label: 'Rename', onClick: () => setRenamingEffect(name) },
        {
          label: 'Update thumbnail',
          title: 'Snapshot of the layer showing this effect right now — its light alone, no surface photo',
          onClick: () => void updateThumbnail(name).then((ok) => ok && useStore.getState().toast('Thumbnail updated', 'ok')),
        },
        'separator'
      );
    }
    items.push({ heading: 'Move to' });
    for (const g of doc?.groups ?? []) items.push({ label: g.name, checked: at === g.id, onClick: () => moveEffect(name, g.id) });
    if (at) items.push({ label: user ? 'New (unsorted)' : 'Built-in', onClick: () => moveEffect(name, null) });
    items.push({ label: 'New group…', onClick: () => newGroupWith(name) });
    if (user) items.push('separator', { label: 'Delete', danger: true, onClick: () => void deleteEffect(name) });
    setMenu({ x: e.clientX, y: e.clientY, items });
  }

  function headerMenu(e: React.MouseEvent, s: Section) {
    e.preventDefault();
    if (s.fixed) return;
    setMenu({
      x: e.clientX,
      y: e.clientY,
      items: [
        { label: 'Rename group', onClick: () => setRenamingGroup(s.id) },
        {
          label: 'Delete group',
          danger: true,
          title: 'Its effects move back to New',
          onClick: () => deleteGroup(s.id),
        },
      ],
    });
  }

  const dropProps = (s: Section) => ({
    onDragOver: (e: React.DragEvent) => {
      if (!e.dataTransfer.types.includes(DRAG_TYPE)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      if (dropTarget !== s.id) setDropTarget(s.id);
    },
    onDragLeave: (e: React.DragEvent) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropTarget(null);
    },
    onDrop: (e: React.DragEvent) => {
      const name = e.dataTransfer.getData(DRAG_TYPE);
      setDropTarget(null);
      if (!name) return;
      e.preventDefault();
      moveEffect(name, s.fixed ? null : s.id);
    },
  });

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center">
        <div className="label flex-1">Effect</div>
        <Btn size="xs" tone="ghost" onClick={() => newGroupWith()} title="New group — right-click a card to file it, or drag it onto a group">
          + Group
        </Btn>
      </div>
      {sections.map((s) => (
        <div
          key={s.id}
          {...dropProps(s)}
          className={'rounded-md ' + (dropTarget === s.id ? 'ring-1 ring-design bg-design/5' : '')}
        >
          <div
            className="group flex items-center gap-1.5 h-6 cursor-pointer select-none"
            onClick={() => renamingGroup !== s.id && toggleGroup(s.id)}
            onContextMenu={(e) => headerMenu(e, s)}
            onDoubleClick={() => !s.fixed && setRenamingGroup(s.id)}
            title={s.fixed ? undefined : 'Click to fold · double-click or right-click to rename'}
          >
            <span className="w-3 text-[9px] text-fg3">{s.collapsed ? '▶' : '▼'}</span>
            {renamingGroup === s.id ? (
              <RenameInput
                value={s.name}
                onDone={(v) => {
                  if (v) renameGroup(s.id, v);
                  setRenamingGroup(null);
                }}
                className="text-[11px] font-semibold"
              />
            ) : (
              <span className={'text-[11px] font-semibold truncate ' + (s.accent ? 'text-design' : 'text-fg2')}>{s.name}</span>
            )}
            <span className="text-[10px] text-fg4">{s.effects.length}</span>
            {s.collapsed && current && s.effects.includes(current) && (
              <span className="h-1.5 w-1.5 rounded-full bg-design" title={`${prettify(current)} is in here`} />
            )}
          </div>
          {!s.collapsed && (
            <div className="grid grid-cols-2 gap-1.5 pb-1">
              {s.effects.map((name) => (
                <EffectCard
                  key={name}
                  name={name}
                  dials={catalog[name]?.inputs.length ?? 0}
                  thumb={thumbs[name]}
                  active={current === name}
                  renaming={renamingEffect === name}
                  onPick={() => onPick(name)}
                  onMenu={(e) => cardMenu(e, name)}
                  onRenamed={(v) => {
                    setRenamingEffect(null);
                    if (v) void renameEffect(name, v);
                  }}
                />
              ))}
              {s.effects.length === 0 && (
                <div className="col-span-2 rounded-md border border-dashed border-line px-2 py-2 text-[10px] text-fg4">
                  Empty — drag an effect here, or right-click one → Move to
                </div>
              )}
            </div>
          )}
        </div>
      ))}
      {menu && <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={closeMenu} />}
    </div>
  );
}

function EffectCard({
  name,
  dials,
  thumb,
  active,
  renaming,
  onPick,
  onMenu,
  onRenamed,
}: {
  name: string;
  dials: number;
  thumb?: string;
  active: boolean;
  renaming: boolean;
  onPick: () => void;
  onMenu: (e: React.MouseEvent) => void;
  onRenamed: (v: string | null) => void;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      draggable={!renaming}
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_TYPE, name);
        e.dataTransfer.effectAllowed = 'move';
      }}
      onClick={() => !renaming && onPick()}
      onKeyDown={(e) => !renaming && (e.key === 'Enter' || e.key === ' ') && onPick()}
      onContextMenu={onMenu}
      title="Right-click: rename, thumbnail, move, delete"
      className={
        'h-12 rounded-md border pl-1 pr-2 text-left flex items-center gap-2 cursor-pointer min-w-0 ' +
        (active ? 'border-design bg-design/10' : 'border-line bg-panel2 hover:border-line2')
      }
    >
      {thumb ? (
        <img src={thumb} alt="" draggable={false} className="h-10 w-10 shrink-0 rounded object-cover bg-black" />
      ) : (
        <div className="h-10 w-10 shrink-0 rounded bg-black/60 border border-line" />
      )}
      <div className="flex min-w-0 flex-1 flex-col justify-center">
        {renaming ? (
          <RenameInput value={prettify(name)} onDone={onRenamed} className="text-[12px] font-medium" />
        ) : (
          <span className="text-[12px] font-medium truncate">{prettify(name)}</span>
        )}
        <span className="text-[9px] text-fg3">
          {dials} dial{dials === 1 ? '' : 's'}
        </span>
      </div>
    </div>
  );
}

/** Autofocused one-shot name field: Enter/blur commits, Esc cancels (null). */
function RenameInput({ value, onDone, className = '' }: { value: string; onDone: (v: string | null) => void; className?: string }) {
  const [done, setDone] = useState(false);
  const finish = (v: string | null) => {
    if (done) return;
    setDone(true);
    onDone(v && v.trim() && v.trim() !== value ? v.trim() : null);
  };
  return (
    <input
      autoFocus
      defaultValue={value}
      spellCheck={false}
      onFocus={(e) => e.currentTarget.select()}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === 'Enter') finish(e.currentTarget.value);
        if (e.key === 'Escape') finish(null);
      }}
      onBlur={(e) => finish(e.currentTarget.value)}
      className={'min-w-0 w-full bg-panel border border-line2 rounded px-1 outline-none ' + className}
    />
  );
}
