// Studio's Render panel (studio plan R1, S16): start renders from the app,
// no chat and no terminal. Every render is a non-blocking hub job (H2 —
// `POST /api/render`), visible in the jobs pill and as a tile in the folder
// it lands in (the open folder = the target, S1), exactly like an agent's.
//
// - TextureFlow: a recipe (its style images + settings are the defaults;
//   what's set here wins), or the style tray's images; the target region
//   sets the aspect (long side 640, step 32) and is recorded as "made for";
//   length / motion / seed; Draft (4 steps, no upscale, ≈ 30 s) or Final.
//   Optional shape input (`control_input`, an AssetPicker over the open
//   show's named assets — `surface_lit` / `surface` first) with "Use as
//   shape" → ControlNet (`use_controlnet1` + `controlnet_strength1`
//   0.35–0.65), and a `diffusion_mask`. Sent only when set (show-setup M1).
// - Image (Nano Banana): a prompt + the tray as references.
// - Video (Kling v2.5): a prompt + the focused still as the start frame,
//   5 or 10 s.
// The cost estimate comes from the hub before Go (`/api/estimate`); past
// today's cap the app asks, and only then sends `allow_over_cap` (S15).

import { useEffect, useMemo, useState } from 'react';
import {
  estimateRender,
  listRecipes,
  mediaUrl,
  startRender,
  type Estimate,
  type Recipe,
  type RenderTool,
} from '../../api/hub';
import { refreshJobs, useHub } from '../../state/hub';
import { regionName } from '../../state/scene';
import { useStore } from '../../state/store';
import { itemByPath, short, useStudio } from '../../state/studio';
import { Btn, Seg } from '../../ui/primitives';
import { Fader } from '../../ui/Fader';
import { AssetPicker } from '../../ui/media/AssetPicker';
import { SpendLine, wouldExceedCap } from './SpendLine';

type Kind = 'flow' | 'image' | 'video';
const TOOL: Record<Kind, RenderTool> = {
  flow: 'texture_flow',
  image: 'nano_banana',
  video: 'kling_v25_image_to_video',
};

/** A region's box → TextureFlow width/height: its aspect, long side 640,
 *  multiples of 32, each side ≥ 320 (the tool's range). */
function sizeFor(bbox: [number, number, number, number] | null | undefined): { width: number; height: number } | null {
  if (!bbox) return null;
  const w = Math.max(1, bbox[2] - bbox[0]);
  const h = Math.max(1, bbox[3] - bbox[1]);
  const snap = (v: number) => Math.min(1280, Math.max(320, Math.round(v / 32) * 32));
  return w >= h ? { width: snap(640), height: snap((640 * h) / w) } : { width: snap((640 * w) / h), height: snap(640) };
}

export function RenderPanel() {
  const hubUp = useHub((s) => s.status?.state === 'up');
  const pack = useStore((s) => s.pack);
  const styles = useStudio((s) => s.selection.styles);
  const focus = useStudio((s) => s.focus);
  const [kind, setKind] = useState<Kind>('flow');
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipe, setRecipe] = useState<string>('');
  const [region, setRegion] = useState<string>('');
  const [seconds, setSeconds] = useState(6);
  const [motion, setMotion] = useState(1.1);
  const [seed, setSeed] = useState('');
  const [final, setFinal] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [count, setCount] = useState(4);
  const [est, setEst] = useState<Estimate | null>(null);
  const [estErr, setEstErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [overCap, setOverCap] = useState(false);
  const [controlInput, setControlInput] = useState<string | null>(null);
  const [useShape, setUseShape] = useState(true);
  const [cnStrength, setCnStrength] = useState(0.45);
  const [diffusionMask, setDiffusionMask] = useState<string | null>(null);
  const showDir = useStore((s) => s.project?.dir);
  const show = showDir ? showDir.replace(/\/+$/, '').split('/').pop() ?? null : null;

  useEffect(() => {
    if (!hubUp) return;
    listRecipes()
      .then((r) => setRecipes(r.recipes))
      .catch((e) => console.warn('recipes', e));
  }, [hubUp]);

  const focused = itemByPath(focus);
  const startFrame = focused?.kind === 'image' ? focused.path : null;
  const regionLayer = pack?.layers.find((l) => l.id === region);

  const args = useMemo((): Record<string, unknown> | null => {
    const common: Record<string, unknown> = region ? { region } : {};
    if (kind === 'flow') {
      if (!recipe && styles.length === 0) return null;
      const a: Record<string, unknown> = { ...common, n_seconds: seconds, motion_scale: motion };
      if (recipe) a.recipe = recipe;
      if (styles.length) a.images = styles.slice(0, 6);
      const size = sizeFor(regionLayer?.bbox);
      if (size) Object.assign(a, size);
      if (seed.trim()) a.seed = Number(seed);
      if (!final) Object.assign(a, { n_steps: 4, use_upscale: false, upscale_esrgan: false });
      if (controlInput) {
        a.control_input = controlInput;
        if (useShape) Object.assign(a, { use_controlnet1: true, controlnet_strength1: Math.round(cnStrength * 100) / 100 });
      }
      if (diffusionMask) a.diffusion_mask = diffusionMask;
      return a;
    }
    if (!prompt.trim()) return null;
    if (kind === 'image') {
      return { ...common, prompt: prompt.trim(), ...(styles.length ? { image_urls: styles.slice(0, 10) } : {}) };
    }
    if (!startFrame) return null;
    return { ...common, prompt: prompt.trim(), image_url: startFrame, duration: seconds >= 8 ? '10' : '5' };
  }, [kind, recipe, styles, region, regionLayer, seconds, motion, seed, final, prompt, startFrame, controlInput, useShape, cnStrength, diffusionMask]);

  // The estimate follows the form (debounced).
  const argsKey = JSON.stringify([kind, args, count]);
  useEffect(() => {
    setEst(null);
    setEstErr(null);
    setOverCap(false);
    if (!hubUp || !args) return;
    const t = setTimeout(() => {
      estimateRender(TOOL[kind], args, count)
        .then(setEst)
        .catch((e) => setEstErr(String(e)));
      // (to_hap has no estimate — it's free; it never runs from here.)
    }, 350);
    return () => clearTimeout(t);
  }, [argsKey, hubUp]);

  const wouldExceed = wouldExceedCap(est);

  const go = async (allowOverCap: boolean) => {
    if (!args) return;
    setBusy(true);
    try {
      const r = await startRender(TOOL[kind], allowOverCap ? { ...args, allow_over_cap: true } : args, count);
      useStore.getState().toast(`Started ${r.jobs.length} ${kind === 'flow' ? 'TextureFlow' : kind} job${r.jobs.length === 1 ? '' : 's'} → ${short(r.jobs[0]?.dest ?? '')}`, 'ok');
      setOverCap(false);
      refreshJobs();
    } catch (e) {
      useStore.getState().toast(`Render refused: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };

  if (!hubUp) {
    return <div className="p-3 text-[12px] text-fg3">Rendering needs the content hub — start it from the Hub lamp.</div>;
  }

  const missing =
    kind === 'flow'
      ? args
        ? null
        : 'Pick a recipe or drop style images in the tray'
      : !prompt.trim()
      ? 'Write a prompt'
      : kind === 'video' && !startFrame
      ? 'Select a still in the grid as the start frame'
      : null;

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 overflow-y-auto p-3 flex flex-col gap-3 text-[12px]">
        <Seg<Kind>
          value={kind}
          onChange={setKind}
          options={[
            { id: 'flow', label: 'TextureFlow' },
            { id: 'image', label: 'Image' },
            { id: 'video', label: 'Video' },
          ]}
        />
        {kind === 'flow' && (
          <Field label="Recipe">
            <select value={recipe} onChange={(e) => setRecipe(e.target.value)} className="w-full h-7 bg-bg border border-line rounded px-1 text-[12px]">
              <option value="">— none (style tray only) —</option>
              {recipes.map((r) => (
                <option key={r.name} value={r.name} title={r.when_to_use}>
                  {r.name}
                  {r.look ? ` · ${r.look}` : ''}
                </option>
              ))}
            </select>
            {recipe && (
              <RecipeHint recipe={recipes.find((r) => r.name === recipe)} />
            )}
          </Field>
        )}
        {kind !== 'flow' && (
          <Field label="Prompt">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder={kind === 'image' ? 'image_1 … refer to the tray images in order' : 'static camera, dark background, slow motion of …'}
              className="w-full bg-bg border border-line rounded px-2 py-1 text-[12px] outline-none resize-none"
            />
          </Field>
        )}
        <Field label={kind === 'video' ? 'Start frame' : 'Styles'}>
          {kind === 'video' ? (
            <span className="text-fg2 truncate block">{startFrame ? short(startFrame) : '— select a still in the grid —'}</span>
          ) : (
            <span className="text-fg2">
              {styles.length ? `${styles.length} from the tray${styles.length > (kind === 'flow' ? 6 : 10) ? ' (first ones used)' : ''}` : 'none in the tray'}
              {kind === 'flow' && recipe && styles.length ? ' — override the recipe’s' : ''}
            </span>
          )}
        </Field>
        <Field label="Made for">
          <select value={region} onChange={(e) => setRegion(e.target.value)} className="w-full h-7 bg-bg border border-line rounded px-1 text-[12px]">
            <option value="">— no region —</option>
            {(pack?.layers ?? []).map((l) => (
              <option key={l.id} value={l.id}>
                {regionName(pack, l.id)}
              </option>
            ))}
          </select>
          {kind === 'flow' && regionLayer && (
            <span className="text-[10px] text-fg3">
              {(() => {
                const s = sizeFor(regionLayer.bbox);
                return s ? `${s.width}×${s.height} (the region's aspect)` : '';
              })()}
            </span>
          )}
        </Field>
        {kind !== 'image' && (
          <Field label={`Length · ${seconds} s`}>
            <input
              type="range"
              min={kind === 'video' ? 5 : 2}
              max={kind === 'video' ? 10 : 24}
              step={kind === 'video' ? 5 : 1}
              value={seconds}
              onChange={(e) => setSeconds(Number(e.target.value))}
              className="w-full accent-[rgb(var(--deck))]"
            />
          </Field>
        )}
        {kind === 'flow' && (
          <>
            <Field label={`Motion · ${motion.toFixed(2)}`}>
              <input
                type="range"
                min={0.7}
                max={1.4}
                step={0.05}
                value={motion}
                onChange={(e) => setMotion(Number(e.target.value))}
                className="w-full accent-[rgb(var(--deck))]"
              />
            </Field>
            {show && (
              <>
                <Group label="Shape input">
                  <AssetPicker
                    project={show}
                    value={controlInput}
                    onChange={setControlInput}
                    suggest={['surface_lit', 'surface']}
                    placeholder="— none —"
                    compact
                  />
                  {controlInput && (
                    <div className="flex items-center gap-2 mt-0.5">
                      <label className="flex items-center gap-1.5 text-[11px] text-fg2 shrink-0 cursor-pointer" title="ControlNet: the render follows this image's shapes / edges. Off = it only shows through black mask areas (and seeds img2img when denoise < 1)">
                        <input type="checkbox" checked={useShape} onChange={(e) => setUseShape(e.target.checked)} className="accent-[rgb(var(--deck))]" />
                        Use as shape (ControlNet)
                      </label>
                      {useShape && (
                        <Fader
                          value={cnStrength}
                          min={0.35}
                          max={0.65}
                          step={0.01}
                          defaultValue={0.45}
                          onChange={setCnStrength}
                          label="strength"
                          format={(v) => v.toFixed(2)}
                          size="sm"
                          className="flex-1 min-w-0"
                          title="Subtle 0.35–0.45, strong 0.45–0.65; higher tends to look bad"
                        />
                      )}
                    </div>
                  )}
                </Group>
                <Group label="Mask (white = animated)">
                  <AssetPicker
                    project={show}
                    value={diffusionMask}
                    onChange={setDiffusionMask}
                    placeholder="— none —"
                    compact
                  />
                </Group>
              </>
            )}
            <div className="flex gap-3">
              <Field label="Seed">
                <input
                  value={seed}
                  onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                  placeholder="random"
                  className="w-24 h-7 bg-bg border border-line rounded px-2 text-[12px] num"
                />
              </Field>
              <Field label="Quality">
                <Seg<'draft' | 'final'>
                  value={final ? 'final' : 'draft'}
                  onChange={(v) => setFinal(v === 'final')}
                  options={[
                    { id: 'draft', label: 'Draft ≈30 s' },
                    { id: 'final', label: 'Final ≈6 min' },
                  ]}
                />
              </Field>
            </div>
          </>
        )}
        <Field label={`How many · ${count}`}>
          <input
            type="range"
            min={1}
            max={12}
            step={1}
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
            className="w-full accent-[rgb(var(--deck))]"
          />
        </Field>
      </div>
      <footer className="flex items-center gap-2 p-2 border-t border-line shrink-0">
        <div className="flex-1 min-w-0 text-[11px] leading-tight">
          <SpendLine missing={missing} estErr={estErr} est={est} />
        </div>
        {overCap ? (
          <Btn tone="danger" disabled={busy} onClick={() => go(true)} title="Render even though it takes today past the daily cap">
            Past the cap — render anyway
          </Btn>
        ) : (
          <Btn
            tone="deck"
            disabled={!args || busy || !!missing}
            onClick={() => (wouldExceed ? setOverCap(true) : go(false))}
            title="Start the jobs; they land in the open folder"
          >
            {busy ? 'Starting…' : `Render ${count}`}
          </Btn>
        )}
      </footer>
    </div>
  );
}

function RecipeHint({ recipe }: { recipe?: Recipe }) {
  if (!recipe) return null;
  return (
    <div className="flex gap-2 mt-1.5">
      {recipe.preview && <img src={mediaUrl(recipe.preview)} className="w-14 h-14 object-cover rounded border border-line shrink-0" />}
      <span className="text-[10px] text-fg3 leading-snug">{recipe.when_to_use}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

/** `Field` without the <label>: for controls holding buttons / faders, where
 *  a label would forward every click to its first button (the picker). */
function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
    </div>
  );
}
