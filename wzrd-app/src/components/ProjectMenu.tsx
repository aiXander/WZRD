// A1 — project switcher + engine recovery (gig plan D13).
//
// A project is a folder with a scene.json under the projects root; recent
// scenes (anywhere on disk) come first. Opening one re-roots the in-process
// engine on that folder — its own pack, effects, looks, alignment and
// session (masters, knobs) — in a few hundred ms. The projector goes dark
// for that moment, so opening takes two clicks: arm, then confirm.

import { useEffect, useRef, useState } from 'react';
import { projectsList, type ProjectEntry } from '../api/ipc';
import { openProject, restartEngine, saveProjectAs } from '../state/actions';
import { relinkMissing, useDecks } from '../state/decks';
import { useStore } from '../state/store';
import { Popover } from '../ui/Popover';
import { Btn, Dot } from '../ui/primitives';

function ago(ms: number | null): string {
  if (!ms) return '';
  const s = (Date.now() - ms) / 1000;
  if (s < 90) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function ProjectMenu() {
  const project = useStore((s) => s.project);
  const running = useStore((s) => s.engineRunning);
  const anchor = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        ref={anchor}
        onClick={() => setOpen((o) => !o)}
        title={project ? `${project.scene_path}\nSwitch project · restart engine` : 'Open a project'}
        className={
          'flex items-center gap-1.5 h-7 max-w-[220px] min-w-0 px-2 rounded-md border text-[12px] ' +
          (open ? 'bg-raised border-line2 text-fg' : 'border-line text-fg2 hover:text-fg hover:bg-raised')
        }
      >
        {!running && <Dot tone="bad" />}
        <span className="truncate font-medium">{project?.name || 'No project'}</span>
        <span className="text-fg3 text-[10px]">▾</span>
      </button>
      {open && <ProjectList anchor={anchor.current} onClose={() => setOpen(false)} />}
    </>
  );
}

function ProjectList({ anchor, onClose }: { anchor: HTMLElement | null; onClose: () => void }) {
  const running = useStore((s) => s.engineRunning);
  const engineError = useStore((s) => s.engineError);
  const [list, setList] = useState<{ root: string; projects: ProjectEntry[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [armed, setArmed] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    projectsList()
      .then(setList)
      .catch((e) => setErr(String(e)));
  }, []);

  async function go(p: ProjectEntry) {
    if (armed !== p.scene_path) return setArmed(p.scene_path);
    setBusy(true);
    await openProject(p.scene_path);
    setBusy(false);
    onClose();
  }

  async function restart() {
    setBusy(true);
    await restartEngine();
    setBusy(false);
    onClose();
  }

  return (
    <Popover anchor={anchor} onClose={onClose} width={360}>
      <div className="flex items-baseline gap-2 px-1 pb-2">
        <span className="label">Projects</span>
        <span className="text-[10px] text-fg4 truncate" title={list?.root}>
          {list?.root}
        </span>
      </div>
      <div className="flex flex-col gap-0.5 max-h-[50vh] overflow-auto">
        {err && <div className="p-2 text-[11px] text-bad">{err}</div>}
        {list && list.projects.length === 0 && (
          <div className="p-2 text-[11px] text-fg3 leading-snug">
            No project folder with a scene.json under the projects root yet — build one with the MCP
            (<b className="text-fg2">build_layerpack</b> writes a starter scene).
          </div>
        )}
        {list?.projects.map((p) => {
          const isArmed = armed === p.scene_path;
          return (
            <button
              key={p.scene_path}
              disabled={busy || p.current}
              onClick={() => go(p)}
              title={p.scene_path}
              className={
                'flex items-center gap-2 px-2 py-1.5 rounded-md border text-left ' +
                (p.current
                  ? 'border-transparent bg-panel2 cursor-default'
                  : isArmed
                  ? 'border-amber/70 bg-amber/10'
                  : 'border-transparent hover:bg-panel2')
              }
            >
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-medium truncate">{p.name}</div>
                <div className="text-[10px] text-fg3 truncate">
                  {isArmed
                    ? 'Click again to open — the wall goes dark for a moment'
                    : [p.in_root ? null : p.dir, ago(p.opened_ms)].filter(Boolean).join(' · ') || 'never opened here'}
                </div>
              </div>
              {p.current && (
                <span className="flex items-center gap-1 text-[10px] text-fg2">
                  <Dot tone={running ? 'ok' : 'bad'} /> open
                </span>
              )}
              {isArmed && <span className="text-[10px] font-bold text-amber">OPEN</span>}
            </button>
          );
        })}
      </div>
      <SaveAs busy={busy} setBusy={setBusy} onDone={onClose} />
      <div className="flex items-center gap-2 mt-2 pt-2 border-t border-line">
        <Btn size="sm" tone={running ? 'solid' : 'danger'} active={!running} onClick={restart} title="Re-boot the engine on this project (session + alignment are saved first)">
          {busy ? 'Working…' : '↻ Restart engine'}
        </Btn>
        <RelinkButton />
        {!running && engineError && (
          <span className="text-[10px] text-bad truncate" title={engineError}>
            {engineError}
          </span>
        )}
      </div>
    </Popover>
  );
}

/** D3: find clips that moved and point the show at them. Shows the count
 *  of missing deck slots when there are any. */
function RelinkButton() {
  const n = useDecks((s) => s.missing.size);
  const [busy, setBusy] = useState(false);
  return (
    <Btn
      size="sm"
      tone={n ? 'danger' : 'solid'}
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await relinkMissing();
        setBusy(false);
      }}
      title="Find clips whose file moved (same render id, same fingerprint, then a walk of the library roots) and fix the decks, looks and scene"
    >
      {busy ? 'Searching…' : `Relink missing${n ? ` (${n})` : ''}`}
    </Btn>
  );
}

/** "Save as…" (S21): everything autosaves; this forks the show into a
 *  sibling folder (clips referenced, not copied) and opens the fork. */
function SaveAs({ busy, setBusy, onDone }: { busy: boolean; setBusy: (b: boolean) => void; onDone: () => void }) {
  const project = useStore((s) => s.project);
  const [name, setName] = useState<string | null>(null);
  if (!project) return null;
  const submit = async () => {
    if (!name?.trim()) return;
    setBusy(true);
    const ok = await saveProjectAs(name.trim());
    setBusy(false);
    if (ok) onDone();
  };
  return (
    <div className="flex items-center gap-2 mt-2 pt-2 border-t border-line">
      {name == null ? (
        <Btn size="sm" disabled={busy} onClick={() => setName(`${project.dir.split('/').pop()}-copy`)} title="Fork this show (scene, effects, decks, looks, session, alignment, pack — clips stay referenced) and open the copy. Everything else saves itself.">
          Save as…
        </Btn>
      ) : (
        <>
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') submit();
              if (e.key === 'Escape') setName(null);
            }}
            onFocus={(e) => e.target.select()}
            className="flex-1 min-w-0 h-7 px-2 text-[12px] rounded-md bg-bg border border-line2 outline-none"
            placeholder="new show folder name"
          />
          <Btn size="sm" tone="solid" disabled={busy || !name.trim()} onClick={submit}>
            {busy ? 'Saving…' : 'Save'}
          </Btn>
        </>
      )}
    </div>
  );
}

/** Full-width strip under the top bar while the engine is down: the reason
 *  and the one click that brings it back. */
export function EngineDownBanner() {
  const running = useStore((s) => s.engineRunning);
  const engineError = useStore((s) => s.engineError);
  const project = useStore((s) => s.project);
  const [busy, setBusy] = useState(false);
  if (running || !project) return null;
  return (
    <div className="flex items-center gap-3 h-9 px-3 shrink-0 bg-bad/15 border-b border-bad/60 text-[12px]">
      <Dot tone="bad" />
      <span className="font-semibold">Engine stopped</span>
      <span className="text-fg2 truncate flex-1" title={engineError ?? undefined}>
        {engineError ?? 'The render thread exited.'} — scene, looks, session and alignment on disk are intact.
      </span>
      <Btn
        size="sm"
        tone="danger"
        active
        onClick={async () => {
          setBusy(true);
          await restartEngine();
          setBusy(false);
        }}
      >
        {busy ? 'Restarting…' : '↻ Restart engine'}
      </Btn>
    </div>
  );
}
