// Step 4 — Segmentation: Nano Banana paints the lit crop as flat colour
// regions. The prompt is a fork of a template (`segmentation` kind: this
// show's own first, then WZRD's), `{n_regions}` filled from the stepper;
// "Save to project" / "Save to WZRD" keep an edited prompt as a template.
// Renders are hub jobs (`/api/render`, nano_banana) landing in the show's
// segmentation/ folder — their tiles show up in the gallery while they run.

import { useEffect, useMemo, useRef, useState } from 'react';
import { estimateRender, startRender, type Estimate } from '../../api/hub';
import { fillPrompt, listPrompts, savePrompt, type Prompt, type ProjectInfo } from '../../api/project';
import { refreshJobs } from '../../state/hub';
import { refreshProjectInfo } from '../../state/projectAssets';
import { patchSeg, segDraft, useSetup } from '../../state/setup';
import { useStore } from '../../state/store';
import { Btn, Seg } from '../../ui/primitives';
import { AssetPicker } from '../../ui/media/AssetPicker';
import { MediaGallery } from '../../ui/media/MediaGallery';
import { NamePrompt } from '../../ui/media/NamePrompt';
import { Field, Note, SectionTitle, Slider, Spinner, StepBody } from './common';
import { SpendLine, wouldExceedCap } from '../studio/SpendLine';

const KIND = 'segmentation';
const keyOf = (p: Prompt) => `${p.scope}:${p.name}`;

export function SegmentationStep({ info }: { info: ProjectInfo }) {
  const project = info.project;
  const d = useSetup((s) => segDraft(s, project));
  const [prompts, setPrompts] = useState<Prompt[] | null>(null);
  const [promptErr, setPromptErr] = useState<string | null>(null);
  const [saveScope, setSaveScope] = useState<{ scope: 'project' | 'wzrd'; anchor: HTMLElement } | null>(null);
  const [est, setEst] = useState<Estimate | null>(null);
  const [estErr, setEstErr] = useState<string | null>(null);
  const [overCap, setOverCap] = useState(false);
  const [busy, setBusy] = useState(false);
  const set = (p: Parameters<typeof patchSeg>[1]) => patchSeg(project, p);

  const input = d.input ?? info.assets.surface_lit?.path ?? null;
  const dest = info.layout.segmentation?.path ?? null;

  const loadPrompts = async (select?: string) => {
    try {
      const r = await listPrompts(KIND, project);
      setPrompts(r.prompts);
      setPromptErr(null);
      const cur = useSetup.getState().seg[project];
      const pick = (select && r.prompts.find((p) => keyOf(p) === select)) || (!cur?.from && r.prompts[0]) || null;
      if (pick) set({ from: keyOf(pick), text: select ? cur?.text ?? pick.prompt : pick.prompt });
    } catch (e) {
      setPromptErr(String(e));
    }
  };
  useEffect(() => {
    void loadPrompts();
  }, [project]);

  const source = prompts?.find((p) => keyOf(p) === d.from) ?? null;
  const text = d.text ?? source?.prompt ?? '';
  const edited = !!source && text !== source.prompt;
  const filled = fillPrompt(text, d.nRegions);

  const args = useMemo((): Record<string, unknown> | null => {
    if (!input || !filled.trim() || !dest) return null;
    return {
      prompt: filled,
      image_urls: [input],
      aspect_ratio: 'auto',
      resolution: d.resolution,
      thinking_level: d.thinking,
      dest,
      project,
    };
  }, [input, filled, dest, d.resolution, d.thinking, project]);

  const argsKey = JSON.stringify([args, d.count]);
  useEffect(() => {
    setEst(null);
    setEstErr(null);
    setOverCap(false);
    if (!args) return;
    const t = setTimeout(() => {
      estimateRender('nano_banana', args, d.count)
        .then(setEst)
        .catch((e) => setEstErr(String(e)));
    }, 350);
    return () => clearTimeout(t);
  }, [argsKey]);

  const wouldExceed = wouldExceedCap(est);

  const go = async (allowOverCap: boolean) => {
    if (!args) return;
    setBusy(true);
    try {
      const r = await startRender('nano_banana', allowOverCap ? { ...args, allow_over_cap: true } : args, d.count);
      useStore.getState().toast(`Started ${r.jobs.length} segmentation render${r.jobs.length === 1 ? '' : 's'}`, 'ok');
      setOverCap(false);
      refreshJobs();
    } catch (e) {
      useStore.getState().toast(`Render refused: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };

  const save = async (scope: 'project' | 'wzrd', name: string) => {
    try {
      const p = await savePrompt({ kind: KIND, name, prompt: text, scope, project });
      useStore.getState().toast(`Saved prompt "${p.name}" to ${scope === 'project' ? 'this show' : 'WZRD'}`, 'ok');
      await loadPrompts(keyOf(p));
    } catch (e) {
      useStore.getState().toast(`Save failed: ${e}`, 'bad');
    }
  };
  const defaultName = source && source.name !== 'default' ? source.name : '';
  const nameSuggestions = (scope: 'project' | 'wzrd') =>
    (prompts ?? []).filter((p) => p.scope === scope && p.name !== 'default').map((p) => p.name);

  const missing = !input
    ? 'Pick the input image (the lit crop)'
    : !filled.trim()
    ? 'Write a prompt'
    : !dest
    ? 'This show has no segmentation folder'
    : null;

  return (
    <StepBody
      main={
        <>
          <div className="grid grid-cols-[220px_1fr] gap-4">
            <Field label="Input" hint="The lit crop from the Surface step — sharp shapes segment best.">
              <AssetPicker project={project} value={input} onChange={(p) => set({ input: p })} kind="image" suggest={['surface_lit', 'photo_lit']} />
            </Field>
            <div className="flex flex-col gap-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="label">Prompt</span>
                <select
                  value={d.from ?? ''}
                  onChange={(e) => {
                    const p = prompts?.find((x) => keyOf(x) === e.target.value);
                    if (p) set({ from: keyOf(p), text: p.prompt });
                  }}
                  className="h-6 bg-bg border border-line rounded px-1 text-[11px] min-w-0 max-w-[260px]"
                >
                  {(prompts ?? []).map((p) => (
                    <option key={keyOf(p)} value={keyOf(p)}>
                      {p.name} · {p.scope === 'project' ? 'this show' : 'WZRD'}
                    </option>
                  ))}
                </select>
                {edited && <span className="text-[10px] text-amber font-semibold">edited</span>}
                <div className="flex-1" />
                <Btn size="xs" tone="ghost" disabled={!edited} onClick={() => source && set({ text: source.prompt })} title="Back to the template's text">
                  Reset
                </Btn>
                <Btn size="xs" onClick={(e) => setSaveScope({ scope: 'project', anchor: e.currentTarget as HTMLElement })} title="Keep this prompt as a template for this show">
                  Save to project
                </Btn>
                <Btn size="xs" onClick={(e) => setSaveScope({ scope: 'wzrd', anchor: e.currentTarget as HTMLElement })} title="Keep this prompt as a template for every show">
                  Save to WZRD
                </Btn>
              </div>
              <textarea
                value={text}
                onChange={(e) => set({ text: e.target.value })}
                rows={14}
                spellCheck={false}
                className="w-full min-h-[220px] bg-bg border border-line rounded px-2 py-1.5 text-[12px] leading-snug outline-none resize-y font-mono"
              />
              <span className="text-[10px] text-fg3">
                <code className="text-fg2">{'{n_regions}'}</code> becomes {d.nRegions}; the input is <code className="text-fg2">image_1</code>.
                {source?.notes ? ` · ${source.notes}` : ''}
              </span>
              {promptErr && <Note tone="bad">{promptErr}</Note>}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <SectionTitle right={<span className="text-[10px] text-fg3">★ pins one as the show's segmentation map</span>}>
              Results
            </SectionTitle>
            <MediaGallery
              dir={dest}
              project={project}
              roles={['segmentation']}
              onPinned={() => refreshProjectInfo(project)}
              compareWith={input}
              minTile={180}
              emptyText="Nothing rendered yet — Generate makes the first maps."
            />
          </div>
          {saveScope && (
            <NamePrompt
              anchor={saveScope.anchor}
              title={saveScope.scope === 'project' ? 'Save to this show as…' : 'Save to WZRD as…'}
              initial={defaultName}
              suggestions={nameSuggestions(saveScope.scope)}
              onSubmit={(name) => save(saveScope.scope, name)}
              onClose={() => setSaveScope(null)}
            />
          )}
        </>
      }
      side={
        <>
          <span className="label">Regions</span>
          <div className="flex items-center gap-2">
            <Btn size="sm" square onClick={() => set({ nRegions: Math.max(1, d.nRegions - 1) })}>
              −
            </Btn>
            <span className="num text-[16px] font-semibold w-8 text-center">{d.nRegions}</span>
            <Btn size="sm" square onClick={() => set({ nRegions: Math.min(24, d.nRegions + 1) })}>
              +
            </Btn>
            <span className="text-[10px] text-fg3 leading-snug">how many colour regions to ask for</span>
          </div>
          <Slider label="How many renders" value={d.count} min={1} max={8} step={1} def={4} digits={0} onChange={(v) => set({ count: v })} />
          <Field label="Resolution">
            <Seg<'1K' | '2K'>
              value={d.resolution}
              onChange={(v) => set({ resolution: v })}
              options={[
                { id: '1K', label: '1K' },
                { id: '2K', label: '2K' },
              ]}
            />
          </Field>
          <Field label="Thinking" hint="Minimal is ~30 % faster; medium follows the prompt more closely.">
            <Seg<'minimal' | 'medium'>
              value={d.thinking}
              onChange={(v) => set({ thinking: v })}
              options={[
                { id: 'minimal', label: 'Minimal' },
                { id: 'medium', label: 'Medium' },
              ]}
            />
          </Field>
          <div className="flex-1" />
          <div className="text-[11px] leading-tight pt-2 border-t border-line">
            <SpendLine missing={missing} estErr={estErr} est={est} />
          </div>
          {overCap ? (
            <Btn tone="danger" size="lg" className="w-full" disabled={busy} onClick={() => go(true)} title="Render even though it takes today past the daily cap">
              Past the cap — render anyway
            </Btn>
          ) : (
            <Btn
              tone="deck"
              active
              size="lg"
              className="w-full"
              disabled={!args || busy || !!missing}
              onClick={() => (wouldExceed ? setOverCap(true) : go(false))}
            >
              {busy ? (
                <>
                  <Spinner /> Starting…
                </>
              ) : (
                `Generate ${d.count}`
              )}
            </Btn>
          )}
        </>
      }
    />
  );
}
