// A2 — the clip grid's data (gig plan D12): the project's HAP clips (C3,
// polled every 2 s so agent renders appear without a reload), which region
// row each sits in (C2 sidecar `region`, overridden by the operator's
// drag-filing in clipgrid.json), which are NEW, and the one scene write the
// grid makes: put a clip on a region's binding (C5) on DESIGN.
//
// NEW is in memory per project: every clip present at the first load is
// "seen"; anything that arrives later is NEW until it is clicked.

import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { clipFile, clipsList, clipThumb, type ClipEntry, type PackInfo } from '../api/ipc';
import { commitSceneMutation } from './sceneCommit';
import { inputsFor, parseScene, uniqueId, type Binding } from './scene';
import { useStore } from './store';

const POLL_MS = 2000;

type ClipsState = {
  /** scene_path the list belongs to — a switch starts a fresh list. */
  project: string | null;
  clips: ClipEntry[];
  /** Operator filing: clip path → region id | null (unassigned). */
  rows: Record<string, string | null>;
  seen: Set<string>;
  /** `path|mtime` → JPEG data URL. */
  thumbs: Record<string, string>;
};

export const useClips = create<ClipsState>(() => ({
  project: null,
  clips: [],
  rows: {},
  seen: new Set(),
  thumbs: {},
}));

const signature = (cs: ClipEntry[]) =>
  cs.map((c) => `${c.path}:${c.mtime_ms}:${c.thumb ?? ''}:${c.sidecar ? 1 : 0}`).join('|');

async function poll() {
  const project = useStore.getState().project?.scene_path ?? null;
  if (!project) return;
  let res: Awaited<ReturnType<typeof clipsList>>;
  try {
    res = await clipsList();
  } catch {
    return;
  }
  // A switch landed while the scan was in flight: this list is stale.
  if (useStore.getState().project?.scene_path !== project) return;
  const cur = useClips.getState();
  if (cur.project !== project) {
    useClips.setState({
      project,
      clips: res.clips,
      rows: res.rows ?? {},
      seen: new Set(res.clips.map((c) => c.path)),
      thumbs: {},
    });
    return;
  }
  if (signature(cur.clips) !== signature(res.clips) || JSON.stringify(cur.rows) !== JSON.stringify(res.rows)) {
    useClips.setState({ clips: res.clips, rows: res.rows ?? {} });
  }
}

/** Mounted once (App): keeps the list fresh in every room, so NEW is
 *  accurate whenever the grid is opened. */
export function useClipsPoller() {
  useEffect(() => {
    poll();
    const t = window.setInterval(poll, POLL_MS);
    const unsub = useStore.subscribe((s, prev) => {
      if (s.project?.scene_path !== prev.project?.scene_path) poll();
    });
    return () => {
      window.clearInterval(t);
      unsub();
    };
  }, []);
}

function markSeen(path: string) {
  const seen = useClips.getState().seen;
  if (seen.has(path)) return;
  useClips.setState({ seen: new Set(seen).add(path) });
}

/** The row a clip sits in: operator filing first, then the sidecar's
 *  region; a region the pack doesn't have → unassigned (null). */
export function rowOf(clip: ClipEntry, rows: Record<string, string | null>, pack: PackInfo | null): string | null {
  const region = clip.path in rows ? rows[clip.path] : clip.sidecar?.region ?? null;
  return region && pack?.layers.some((l) => l.id === region) ? region : null;
}

export function fileClip(path: string, region: string | null) {
  useClips.setState((s) => ({ rows: { ...s.rows, [path]: region } }));
  clipFile(path, region).catch((e) => useStore.getState().toast(`Could not re-file clip: ${e}`, 'bad'));
}

/** region id → clip paths its bindings play, in one leg's scene. Only
 *  bindings that target exactly that region count (a tag/group binding
 *  isn't "this region's clip"). */
export function clipsByRegion(sceneJson: string): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const b of parseScene(sceneJson)?.bindings ?? []) {
    const region = b.select?.id;
    const clip = b.params?.image?.clip;
    if (typeof region === 'string' && typeof clip === 'string') (out[region] ??= []).push(clip);
  }
  return out;
}

/**
 * Put `clip` on `region` in DESIGN (C5), through sceneCommit like every
 * scene write: set `image.clip` on the region's first binding whose effect
 * declares an `image` input (keeping its placement/rate), else add
 * `clip_<region>` playing it through the built-in `videoClip`. Returns the
 * binding id. Taking it live stays the operator's TAKE.
 */
export function bindClip(clip: string, region: string): string {
  const { catalog } = useStore.getState();
  const hasImage = (b: Binding) => inputsFor(b.effect, catalog).some((i) => i.type === 'image');
  let id = '';
  commitSceneMutation((s) => {
    s.bindings = s.bindings ?? [];
    const target = s.bindings.find((b: Binding) => b.select?.id === region && hasImage(b));
    if (target) {
      const prev = target.params?.image && typeof target.params.image === 'object' ? target.params.image : {};
      target.params = { ...(target.params ?? {}), image: { placement: 'region', rate: 1, ...prev, clip } };
      id = target.id;
    } else {
      id = uniqueId(`clip_${region}`, new Set(s.bindings.map((b: Binding) => b.id)));
      const gain = catalog.videoClip?.inputs.find((i) => i.name === 'gain')?.default;
      s.bindings.push({
        id,
        select: { id: region },
        effect: 'videoClip',
        params: {
          image: { clip, placement: 'region', rate: 1 },
          ...(gain != null ? { gain: Number(gain) } : {}),
        },
      });
    }
    return s;
  });
  markSeen(clip);
  useStore.getState().setSelectedBindingId(id);
  return id;
}

/** Thumbnail data URL for a clip (null while loading / when it has none). */
export function useThumb(clip: ClipEntry): string | null {
  const key = `${clip.path}|${clip.mtime_ms}`;
  const cached = useClips((s) => s.thumbs[key] ?? null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (cached || !clip.thumb || failed) return;
    let live = true;
    clipThumb(clip.thumb)
      .then((b64) => {
        if (live) useClips.setState((s) => ({ thumbs: { ...s.thumbs, [key]: `data:image/jpeg;base64,${b64}` } }));
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [key, cached, clip.thumb, failed]);
  return cached;
}
