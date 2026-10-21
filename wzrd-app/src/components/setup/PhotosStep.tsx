// Step 2 — Photos: the lit photo (projector full white, room dark) and the
// ambient one (same spot, room lit — optional). A photo comes in through
// `import_photos` (whole frame, full resolution: `conform: false`, detection
// needs every pixel) into the show's captures/, then is pinned as
// `photo_lit` / `photo_ambient`.
//
// "Project white" puts the engine's `white` test pattern on the open show's
// output (alignment.setTestPattern) — it's turned back off when you leave
// the step or close the wizard.

import { useEffect, useRef, useState } from 'react';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { pinAsset, runTool, type ProjectInfo } from '../../api/project';
import { IN_TAURI } from '../../api/transport';
import { setTestPattern } from '../../state/alignment';
import { outputToDisplay, outputWindow, refreshOutput, useOutput } from '../../state/output';
import { refreshProjectInfo } from '../../state/projectAssets';
import { useStore } from '../../state/store';
import { Btn } from '../../ui/primitives';
import { Lightbox } from '../../ui/media/Lightbox';
import { MediaGallery } from '../../ui/media/MediaGallery';
import { useFileDrop } from '../../ui/media/useFileDrop';
import type { MediaRef } from '../../ui/media/types';
import { Img, Note, SectionTitle, Spinner, StepBody, baseName } from './common';

type Slot = 'photo_lit' | 'photo_ambient';
const SLOTS: { id: Slot; label: string; what: string; optional?: boolean }[] = [
  { id: 'photo_lit', label: 'Lit', what: 'Projector on, full white, room dark' },
  { id: 'photo_ambient', label: 'Ambient', what: 'Same spot, room lit, projector off', optional: true },
];
const PHOTO_EXT = ['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp', 'tif', 'tiff'];

export function PhotosStep({ info }: { info: ProjectInfo }) {
  const project = info.project;
  const [target, setTarget] = useState<Slot>(() => (info.assets.photo_lit ? 'photo_ambient' : 'photo_lit'));
  const [busy, setBusy] = useState<Slot | null>(null);
  const [picked, setPicked] = useState<MediaRef | null>(null);
  const [lb, setLb] = useState<MediaRef | null>(null);

  /** Bring files into captures/ and pin them, in slot order from `first`. */
  const importInto = async (first: Slot, paths: string[] | null) => {
    setBusy(first);
    try {
      const r = await runTool<{ photos: { path: string }[] }>(
        'import_photos',
        paths ? { paths, conform: false, project } : { count: 1, conform: false, project }
      );
      const order: Slot[] = first === 'photo_lit' ? ['photo_lit', 'photo_ambient'] : ['photo_ambient', 'photo_lit'];
      const photos = r.photos.slice(0, 2);
      for (let i = 0; i < photos.length; i++) await pinAsset(project, order[i], photos[i].path);
      refreshProjectInfo(project);
      useStore.getState().toast(
        photos.length > 1 ? 'Imported both photos (lit, then ambient)' : `${first === 'photo_lit' ? 'Lit' : 'Ambient'} photo: ${baseName(photos[0]?.path ?? '')}`,
        'ok'
      );
      if (first === 'photo_lit' && photos.length === 1 && !info.assets.photo_ambient) setTarget('photo_ambient');
    } catch (e) {
      useStore.getState().toast(`Import failed: ${e}`, 'bad');
    } finally {
      setBusy(null);
    }
  };

  const pinExisting = async (slot: Slot, path: string) => {
    try {
      await pinAsset(project, slot, path);
      refreshProjectInfo(project);
    } catch (e) {
      useStore.getState().toast(`Pin failed: ${e}`, 'bad');
    }
  };

  // Finder drops land in the targeted slot (two files = lit + ambient).
  const { hovering } = useFileDrop((paths) => {
    const photos = paths.filter((p) => PHOTO_EXT.includes(p.split('.').pop()?.toLowerCase() ?? ''));
    if (!photos.length) return useStore.getState().toast('Drop a photo (jpg, png, heic…)', 'warn');
    void importInto(photos.length > 1 ? 'photo_lit' : target, photos);
  }, !busy);

  const choose = async (slot: Slot) => {
    let path: string | null = null;
    if (IN_TAURI) {
      const r = await openDialog({ multiple: false, title: 'Choose a photo', filters: [{ name: 'Photos', extensions: PHOTO_EXT }] });
      path = typeof r === 'string' ? r : null;
    } else {
      path = window.prompt('Absolute path of the photo');
    }
    if (path) void importInto(slot, [path]);
  };

  return (
    <StepBody
      main={
        <>
          <div className="grid grid-cols-2 gap-4">
            {SLOTS.map((s) => {
              const a = info.assets[s.id];
              const isTarget = target === s.id;
              return (
                <div
                  key={s.id}
                  onClick={() => setTarget(s.id)}
                  className={
                    'flex flex-col rounded-lg border bg-panel2/40 min-w-0 ' +
                    (hovering && isTarget ? 'border-deck ring-2 ring-deck/60' : isTarget ? 'border-line2' : 'border-line')
                  }
                >
                  <div className="flex items-baseline gap-2 px-3 pt-2.5 pb-2">
                    <span className="text-[13px] font-semibold">{s.label}</span>
                    {s.optional && <span className="text-[10px] text-fg3">optional</span>}
                    <span className="text-[11px] text-fg3 truncate">{s.what}</span>
                    <div className="flex-1" />
                    {a && <span className="text-[10px] text-ok font-semibold">✓ pinned</span>}
                  </div>
                  <div
                    className="relative mx-3 rounded-md overflow-hidden bg-black border border-line flex items-center justify-center cursor-pointer"
                    style={{ aspectRatio: '4 / 3' }}
                    onDoubleClick={() => a && setLb({ path: a.path, name: s.label, kind: 'image' })}
                  >
                    {a && a.exists ? (
                      <Img path={a.path} className="w-full h-full object-contain" />
                    ) : (
                      <span className="text-[11px] text-fg3 px-6 text-center leading-snug">
                        {a ? 'The pinned file is missing.' : isTarget ? 'Drop a photo here (or anywhere on this step)' : 'Click to target this slot, then drop'}
                      </span>
                    )}
                    {busy === s.id && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center gap-2 text-[11px]">
                        <Spinner /> importing…
                      </div>
                    )}
                    {hovering && isTarget && (
                      <div className="absolute inset-0 bg-deck/20 flex items-center justify-center text-[12px] font-semibold">
                        Drop → {s.label}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 p-3">
                    <Btn size="sm" disabled={!!busy} onClick={() => choose(s.id)}>
                      Choose file…
                    </Btn>
                    <Btn size="sm" disabled={!!busy} onClick={() => importInto(s.id, null)} title="Take the newest photo in ~/Downloads (where AirDrop lands)">
                      Newest from Downloads
                    </Btn>
                    {a && (
                      <span className="text-[10px] text-fg3 truncate ml-auto" title={a.path}>
                        {baseName(a.path)}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex flex-col gap-2">
            <SectionTitle
              right={
                picked ? (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] text-fg3 truncate max-w-[200px]">{picked.name ?? baseName(picked.path)}</span>
                    <Btn size="xs" onClick={() => pinExisting('photo_lit', picked.path)}>
                      Use as Lit
                    </Btn>
                    <Btn size="xs" onClick={() => pinExisting('photo_ambient', picked.path)}>
                      Use as Ambient
                    </Btn>
                  </div>
                ) : (
                  <span className="text-[10px] text-fg3">Click one to use it</span>
                )
              }
            >
              Already in captures/
            </SectionTitle>
            <MediaGallery
              dir={info.layout.captures?.path}
              project={project}
              roles={['photo_lit', 'photo_ambient']}
              selectedPath={picked?.path ?? null}
              onSelect={setPicked}
              onPinned={() => refreshProjectInfo(project)}
              minTile={130}
              emptyText="No photos imported yet."
            />
          </div>
          {lb && <Lightbox items={[lb]} index={0} onIndex={() => {}} onClose={() => setLb(null)} />}
        </>
      }
      side={<ShootingSide />}
    />
  );
}

function ShootingSide() {
  const pattern = useStore((s) => s.alignment?.test_pattern);
  const engineUp = useStore((s) => s.engineRunning);
  const out = useOutput((s) => s.state);
  const displays = useOutput((s) => s.displays);
  const [err, setErr] = useState<string | null>(null);
  const turnedOn = useRef(false);
  const white = pattern === 'white';

  useEffect(() => {
    void refreshOutput();
    // Leaving the step (or closing the wizard) takes the white off the wall.
    return () => {
      if (turnedOn.current && useStore.getState().alignment?.test_pattern === 'white') {
        setTestPattern('none').catch(() => {});
      }
    };
  }, []);

  const toggle = async () => {
    setErr(null);
    try {
      await setTestPattern(white ? 'none' : 'white');
      turnedOn.current = !white;
    } catch (e) {
      setErr(String(e));
    }
  };

  const outputOff = (out?.target ?? 'off') === 'off';
  const free = displays.filter((d) => !d.operator);

  return (
    <>
      <span className="label">Shooting the photos</span>
      <ol className="flex flex-col gap-1.5 text-[11px] text-fg2 leading-snug list-decimal pl-4">
        <li>Aim the projector at the surface.</li>
        <li>Open the zoom / lens so the light covers as much of the surface as it can.</li>
        <li>Project full white (below) and darken the room.</li>
        <li>Shoot from as close to the projector as you can — beside or just under the lens.</li>
        <li><b className="text-fg">Lit</b>: that photo. <b className="text-fg">Ambient</b>: same spot, room lights on, projector off.</li>
        <li>AirDrop them to this Mac and use "Newest from Downloads".</li>
      </ol>

      <div className="flex flex-col gap-2 pt-2 border-t border-line">
        <span className="label">Project white</span>
        <Btn tone={white ? 'amber' : 'solid'} active={white} disabled={!engineUp} onClick={toggle} title="Fill the whole projector raster with white (the open show's output)">
          {white ? '■ White is on — turn off' : '□ Project full white'}
        </Btn>
        {!engineUp && <span className="text-[10px] text-bad">The engine is stopped — restart it from the project menu.</span>}
        {outputOff && (
          <Note tone="warn">
            The projector output is off, so the white won't reach the wall. Send it to a display:
            <div className="flex flex-wrap gap-1 mt-1.5">
              {free.map((d) => (
                <Btn key={d.name} size="xs" onClick={() => outputToDisplay(d.name)}>
                  {d.name}
                </Btn>
              ))}
              <Btn size="xs" tone="ghost" onClick={() => outputWindow()} title="A window on this screen (testing)">
                Window
              </Btn>
            </div>
          </Note>
        )}
        {err && <Note tone="bad">{err}</Note>}
        <span className="text-[10px] text-fg3 leading-snug">It turns off when you leave this step.</span>
      </div>
    </>
  );
}
