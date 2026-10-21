// Step 1 — Show: pick the show this wizard works on, or make a new one
// (name + canvas → `set_project`). A show's canvas locks once, at creation;
// a legacy show without one can lock it here. The folder map and the pinned
// assets show what's already there.

import { useEffect, useMemo, useState } from 'react';
import { projectInfo, runTool, type ProjectInfo } from '../../api/project';
import { refreshProjectInfo } from '../../state/projectAssets';
import { setNewShow, setSetupProject, setSetupStep, slugId, useSetup } from '../../state/setup';
import { showName } from '../../state/hub';
import { useStore } from '../../state/store';
import { Btn } from '../../ui/primitives';
import { Lightbox } from '../../ui/media/Lightbox';
import { MediaTile } from '../../ui/media/MediaTile';
import type { MediaRef } from '../../ui/media/types';
import { Field, Note, SectionTitle, Spinner, StepBody } from './common';

export function ShowStep({ info }: { info: ProjectInfo | null }) {
  const project = useSetup((s) => s.project);
  const newShow = useSetup((s) => s.newShow);
  const open = showName();
  // Canvas choices + the show list come with any /api/project answer; with
  // no show picked yet, ask for the open one's.
  const [catalog, setCatalog] = useState<ProjectInfo | null>(null);
  useEffect(() => {
    if (info) return;
    projectInfo()
      .then(setCatalog)
      .catch(() => {});
  }, [info]);
  const cat = info ?? catalog;
  const shows = cat?.existing_projects ?? [];

  return (
    <StepBody
      main={
        <>
          <SectionTitle>Which show?</SectionTitle>
          <p className="text-[12px] text-fg2 -mt-2 leading-snug">
            Setup works on one show at a time. The wall keeps playing the open show
            {open ? <> (<b className="text-fg">{open}</b>)</> : null} while you work here.
          </p>
          <div className="flex flex-wrap gap-1.5">
            {shows.map((name) => (
              <button
                key={name}
                onClick={() => setSetupProject(name)}
                className={
                  'h-8 px-3 rounded-md border text-[12px] font-medium ' +
                  (name === project && !newShow
                    ? 'bg-deck/20 border-deck/70 text-fg'
                    : 'bg-panel2 border-line text-fg2 hover:text-fg hover:border-line2')
                }
              >
                {name}
                {name === open && <span className="ml-1.5 text-[10px] text-fg3">open</span>}
              </button>
            ))}
            <button
              onClick={() => setNewShow(true)}
              className={
                'h-8 px-3 rounded-md border border-dashed text-[12px] font-medium ' +
                (newShow ? 'bg-deck/20 border-deck/70 text-fg' : 'border-line2 text-fg2 hover:text-fg')
              }
            >
              + New show
            </button>
          </div>
          {newShow ? <NewShowForm catalog={cat} /> : info ? <ShowSummary info={info} /> : null}
        </>
      }
      side={<SideHelp />}
    />
  );
}

function SideHelp() {
  return (
    <>
      <span className="label">The steps</span>
      <ol className="flex flex-col gap-2 text-[11px] text-fg2 leading-snug list-decimal pl-4">
        <li><b className="text-fg">Photos</b> — the lit photo (projector full white, room dark) and, if you can, the same shot with the room lit.</li>
        <li><b className="text-fg">Surface</b> — crop the projector's throw out of the photo, straighten it, darken it. Pin the best run.</li>
        <li><b className="text-fg">Segmentation</b> — Nano Banana paints the surface as flat colour regions.</li>
        <li><b className="text-fg">Regions</b> — each colour becomes a mask; name them and build the layer pack.</li>
      </ol>
      <p className="text-[10px] text-fg3 leading-snug">
        Every step is a plain tool call, so an agent can run any of them too — its results show up here.
      </p>
    </>
  );
}

function NewShowForm({ catalog }: { catalog: ProjectInfo | null }) {
  const [name, setName] = useState('');
  const canvases = catalog?.canvases ?? [];
  const [canvas, setCanvas] = useState<string>('16:9');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const slug = slugId(name);
  const taken = !!slug && (catalog?.existing_projects ?? []).includes(slug);

  const create = async () => {
    if (!slug || taken) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await runTool<{ project?: string }>('set_project', { name: slug, canvas });
      const made = r.project ?? slug;
      setSetupProject(made);
      refreshProjectInfo(made);
      useStore.getState().toast(`Made show ${made}`, 'ok');
      setSetupStep('photos');
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 max-w-[460px] rounded-lg border border-line bg-panel2/50 p-4">
      <Field label="Name" hint={slug && slug !== name ? <>Folder: <b className="text-fg2">{slug}</b></> : 'Lowercase, no spaces — it becomes the folder name.'}>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && create()}
          placeholder="e.g. wintercircus-nov"
          className="h-8 px-2 rounded-md bg-bg border border-line2 text-[12px] outline-none"
        />
      </Field>
      <Field label="Canvas" hint="The pixel frame every surface, map, mask and pack is made in. Pick the projector's aspect. It can never change.">
        <div className="flex flex-wrap gap-1">
          {(canvases.length ? canvases : [{ name: '16:9', w: 1376, h: 768 }]).map((c) => (
            <button
              key={c.name}
              onClick={() => setCanvas(c.name)}
              title={`${c.w}×${c.h}`}
              className={
                'h-7 px-2 rounded-md border text-[11px] ' +
                (canvas === c.name ? 'bg-raised border-line2 text-fg' : 'border-line text-fg3 hover:text-fg')
              }
            >
              {c.name}
              <span className="ml-1 text-[9px] text-fg4 num">{c.w}×{c.h}</span>
            </button>
          ))}
        </div>
      </Field>
      {taken && <Note tone="warn">A show called {slug} already exists — pick it above instead.</Note>}
      {err && <Note tone="bad">{err}</Note>}
      <div>
        <Btn tone="deck" active disabled={!slug || taken || busy} onClick={create}>
          {busy ? <Spinner /> : null} Make show
        </Btn>
      </div>
    </div>
  );
}

function ShowSummary({ info }: { info: ProjectInfo }) {
  const [lock, setLock] = useState('16:9');
  const [busy, setBusy] = useState(false);
  const [lb, setLb] = useState<number | null>(null);
  const assets = useMemo(() => Object.values(info.assets), [info.assets]);
  const refs: MediaRef[] = assets.map((a) => ({ path: a.path, name: a.name, kind: a.kind }));
  const canvasName = info.canvas
    ? info.canvases.find((c) => c.w === info.canvas![0] && c.h === info.canvas![1])?.name
    : null;

  const lockCanvas = async () => {
    setBusy(true);
    try {
      await runTool('set_project', { name: info.project, canvas: lock });
      refreshProjectInfo(info.project);
    } catch (e) {
      useStore.getState().toast(`Couldn't lock the canvas: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3 flex-wrap text-[12px]">
        <span className="label">Canvas</span>
        {info.canvas ? (
          <span className="text-fg">
            <span className="num">{info.canvas[0]}×{info.canvas[1]}</span>
            {canvasName ? <span className="text-fg3"> · {canvasName}</span> : null}
            <span className="ml-2 text-[10px] font-semibold tracking-wider text-fg3">LOCKED</span>
          </span>
        ) : (
          <>
            <span className="text-amber">not locked yet</span>
            <select value={lock} onChange={(e) => setLock(e.target.value)} className="h-7 bg-bg border border-line rounded px-1 text-[12px]">
              {info.canvases.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} · {c.w}×{c.h}
                </option>
              ))}
            </select>
            <Btn size="sm" disabled={busy} onClick={lockCanvas} title="Lock this show's canvas — once, for good">
              Lock canvas
            </Btn>
          </>
        )}
        <span className="text-fg3 text-[11px] ml-auto">{info.has_scene ? 'Has a scene — the engine can open it' : 'No scene yet (Regions builds one)'}</span>
      </div>

      <div className="flex flex-col gap-2">
        <span className="label">Pinned assets</span>
        {assets.length === 0 ? (
          <span className="text-[11px] text-fg3">None yet — the next steps pin them.</span>
        ) : (
          <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))' }}>
            {assets.map((a, i) => (
              <div key={a.name} className="flex flex-col gap-1 min-w-0">
                <MediaTile media={refs[i]} onClick={() => setLb(i)} onOpen={() => setLb(i)} />
                <span className={'text-[10px] truncate ' + (a.exists ? 'text-fg2' : 'text-bad')} title={a.path}>
                  {a.name}
                  {a.exists ? '' : ' · missing'}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <details className="text-[11px]">
        <summary className="cursor-pointer label select-none">Folder map</summary>
        <div className="mt-2 grid grid-cols-[110px_1fr] gap-x-3 gap-y-1">
          {Object.entries(info.layout).map(([k, v]) => (
            <div key={k} className="contents">
              <span className="text-fg font-medium truncate" title={v.path}>
                {k}/
              </span>
              <span className="text-fg3 truncate" title={v.path}>
                {v.what}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-2 text-[10px] text-fg4 truncate" title={info.project_dir}>
          {info.project_dir}
        </div>
      </details>

      {lb != null && <Lightbox items={refs} index={lb} onIndex={setLb} onClose={() => setLb(null)} />}
    </div>
  );
}
