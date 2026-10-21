// Show setup (docs/TODO/show-setup.md, W1): the venue-day steps before
// anything renders — Show · Photos · Surface · Segmentation · Regions — as one
// modal, opened from the project menu ("New show…", "Show setup…") and ⚙
// Setup. It works on its own show (state/setup.ts `project`, default = the
// open one) and passes `project=` on every hub call; the engine keeps playing
// the open show. Steps are freely clickable; a step shows ✓ once its assets
// are pinned (the hub's `<project>/assets.json`, polled, so an agent's pins
// show up too).

import { useEffect } from 'react';
import type { ProjectInfo } from '../../api/project';
import { ensureHub, showName, useHub } from '../../state/hub';
import { useProjectInfo } from '../../state/projectAssets';
import { SETUP_STEPS, closeSetup, setSetupStep, useSetup, type SetupStep } from '../../state/setup';
import { Btn } from '../../ui/primitives';
import { Modal } from '../../ui/media/Modal';
import { Note, Spinner } from './common';
import { PhotosStep } from './PhotosStep';
import { RegionsStep } from './RegionsStep';
import { SegmentationStep } from './SegmentationStep';
import { ShowStep } from './ShowStep';
import { SurfaceStep } from './SurfaceStep';

/** Mounted once in App.tsx. */
export function SetupHost() {
  const open = useSetup((s) => s.open);
  return open ? <SetupWizard /> : null;
}

function stepDone(step: SetupStep, info: ProjectInfo | null): boolean {
  if (!info?.exists) return false;
  const has = (n: string) => !!info.assets[n]?.exists;
  switch (step) {
    case 'show':
      return !!info.canvas;
    case 'photos':
      return has('photo_lit');
    case 'surface':
      return has('surface_lit') && has('surface');
    case 'segmentation':
      return has('segmentation');
    case 'regions':
      return info.has_scene;
  }
}

function SetupWizard() {
  const project = useSetup((s) => s.project);
  const step = useSetup((s) => s.step);
  const newShow = useSetup((s) => s.newShow);
  const hubState = useHub((s) => s.status?.state);
  const { info, error } = useProjectInfo(project);
  const idx = SETUP_STEPS.findIndex((s) => s.id === step);
  const needsShow = step !== 'show' && (!project || newShow);

  // A step past Show with no show picked → back to Show.
  useEffect(() => {
    if (needsShow) setSetupStep('show');
  }, [needsShow]);

  const title = (
    <span className="flex items-baseline gap-2">
      Show setup
      {project && !newShow && (
        <>
          <span className="text-fg3 font-normal">·</span>
          <span className="text-fg">{project}</span>
          {project !== showName() && <span className="text-[10px] font-normal text-fg3">(not the open show)</span>}
        </>
      )}
    </span>
  );

  const body = (() => {
    if (hubState !== 'up') {
      return (
        <div className="p-6 max-w-[520px] flex flex-col gap-3">
          <Note tone="warn">Show setup runs its tools in the content hub, which isn't running.</Note>
          <div>
            <Btn tone="deck" active disabled={hubState === 'starting'} onClick={() => ensureHub()}>
              {hubState === 'starting' ? (
                <>
                  <Spinner /> Starting…
                </>
              ) : (
                'Start the hub'
              )}
            </Btn>
          </div>
        </div>
      );
    }
    if (step === 'show') return <ShowStep info={newShow ? null : info} />;
    if (!info) {
      return (
        <div className="p-6 flex items-center gap-2 text-[12px] text-fg3">
          {error ? <Note tone="bad">{error}</Note> : <><Spinner /> Loading {project}…</>}
        </div>
      );
    }
    switch (step) {
      case 'photos':
        return <PhotosStep info={info} />;
      case 'surface':
        return <SurfaceStep info={info} />;
      case 'segmentation':
        return <SegmentationStep info={info} />;
      case 'regions':
        return <RegionsStep info={info} />;
    }
  })();

  const canMove = !!project && !newShow;

  return (
    <Modal
      title={title}
      onClose={closeSetup}
      width={1080}
      height="min(88vh, 860px)"
      footer={
        <>
          <span className="text-[10px] text-fg3 truncate min-w-0" title={info?.project_dir}>
            {canMove && info ? info.project_dir : 'Pick or make a show to start'}
          </span>
          <div className="flex-1" />
          <Btn disabled={idx <= 0 || !canMove} onClick={() => setSetupStep(SETUP_STEPS[idx - 1].id)}>
            ← Back
          </Btn>
          {idx < SETUP_STEPS.length - 1 ? (
            <Btn tone="deck" active disabled={!canMove} onClick={() => setSetupStep(SETUP_STEPS[idx + 1].id)}>
              Next: {SETUP_STEPS[idx + 1].label} →
            </Btn>
          ) : (
            <Btn tone="solid" onClick={closeSetup}>
              Done
            </Btn>
          )}
        </>
      }
    >
      <div className="flex flex-col h-full min-h-0">
        <nav className="flex items-center gap-1 px-4 py-2 border-b border-line shrink-0">
          {SETUP_STEPS.map((s, i) => {
            const done = canMove && stepDone(s.id, info);
            const active = s.id === step;
            const disabled = s.id !== 'show' && !canMove;
            return (
              <div key={s.id} className="flex items-center gap-1">
                {i > 0 && <span className="w-4 h-px bg-line2" />}
                <button
                  disabled={disabled}
                  onClick={() => setSetupStep(s.id)}
                  className={
                    'flex items-center gap-1.5 h-7 pl-1.5 pr-2.5 rounded-full border text-[12px] font-medium transition-colors ' +
                    (active
                      ? 'bg-deck/20 border-deck/70 text-fg'
                      : disabled
                      ? 'border-line text-fg4 cursor-default'
                      : 'border-line text-fg2 hover:text-fg hover:border-line2')
                  }
                >
                  <span
                    className={
                      'inline-flex items-center justify-center w-[18px] h-[18px] rounded-full text-[10px] font-bold ' +
                      (done ? 'bg-ok/90 text-black' : active ? 'bg-deck text-black' : 'bg-panel2 text-fg3 border border-line')
                    }
                  >
                    {done ? '✓' : i + 1}
                  </span>
                  {s.label}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="flex-1 min-h-0">{body}</div>
      </div>
    </Modal>
  );
}
