// The Code drawer: one Monaco editor over the project's plain files —
// scene.json + every effects/<name>/shader.wgsl, one tab (and one Monaco
// model, so undo never crosses files) each.
//
// Saving is per file and explicit: ⌘S inside the drawer or the Save button
// saves the *open tab*, never the whole project.
//   - scene.json → `commitSceneNow` (the one scene commit path: engine
//     `scene.load` first, disk only after the engine accepted it).
//   - a shader   → `effect_upsert`: the engine naga-validates before touching
//     disk (a bad shader comes back as line-mapped errors, nothing written),
//     writes effects/<name>/shader.wgsl and hot-reloads the design leg
//     through the probe. The descriptor is sent as null = keep the file's.
// The result shows inline under the tabs (errors stay until the next edit).
//
// ⌘S routing (trap): the editor's own `addCommand` handles it while the text
// has focus — Monaco stops the event there, and shortcuts.ts ignores keys
// from `.monaco-editor` anyway. With focus elsewhere in the drawer (a tab,
// the Save button) the root's onKeyDown takes it and stops propagation, so
// the global "save design" ⌘S never fires from in here. `addCommand` is
// registered once at mount, so it calls through `saveRef` — a closure
// captured at mount would save whatever tab was open back then.
//
// Unsaved shader edits live in a module store (per project), so closing the
// drawer or switching rooms never drops them. `wgsl.validate` runs debounced
// while typing and squiggles naga diagnostics into the model.

import './monacoSetup';
import { Editor, type Monaco } from '@monaco-editor/react';
import type { editor as mEditor, languages as mLanguages } from 'monaco-editor';
import { useEffect, useMemo, useRef, useState } from 'react';
import { create } from 'zustand';
import { effectUpsert, listEffects, readEffect, wgslValidate } from '../api/ipc';
import { commitSceneNow } from '../state/sceneCommit';
import { useStore } from '../state/store';

type TabKind = { kind: 'scene' } | { kind: 'effect'; name: string };
type Status = { tab: string; tone: 'ok' | 'warn' | 'bad' | 'busy'; text: string };

const tabKey = (t: TabKind) => (t.kind === 'scene' ? 'scene.json' : `effects/${t.name}/shader.wgsl`);

/** Open shader buffers: `saved` = what's on disk (as last read/saved). */
type Buf = { wgsl: string; saved: string };
const useBuffers = create<{ project: string | null; bufs: Record<string, Buf> }>(() => ({
  project: null,
  bufs: {},
}));
const setBuf = (name: string, f: (b: Buf | undefined) => Buf) =>
  useBuffers.setState((s) => ({ bufs: { ...s.bufs, [name]: f(s.bufs[name]) } }));

export function MonacoPanel() {
  const editorRef = useRef<mEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const saveRef = useRef<() => void>(() => {});
  const project = useStore((s) => s.project?.scene_path ?? null);
  const sceneJson = useStore((s) => s.sceneJson);
  const setSceneJson = useStore((s) => s.setSceneJson);
  const sceneDirty = useStore((s) => s.sceneDirty);
  const setSceneDirty = useStore((s) => s.setSceneDirty);
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const effects = useStore((s) => s.effects);
  const setEffects = useStore((s) => s.setEffects);
  const toast = useStore((s) => s.toast);
  const bufs = useBuffers((s) => s.bufs);
  const [status, setStatus] = useState<Status | null>(null);
  const validateTimer = useRef<number | null>(null);
  const key = tabKey(activeTab);

  // A project switch starts a fresh buffer set (its effects are other files).
  useEffect(() => {
    if (useBuffers.getState().project !== project) useBuffers.setState({ project, bufs: {} });
  }, [project]);

  // Load an effect's source lazily when its tab is selected.
  useEffect(() => {
    if (activeTab.kind !== 'effect') return;
    const name = activeTab.name;
    if (bufs[name]) return;
    readEffect(name)
      .then((data) => {
        if (!useBuffers.getState().bufs[name]) setBuf(name, () => ({ wgsl: data.wgsl, saved: data.wgsl }));
      })
      .catch((e) => setStatus({ tab: tabKey(activeTab), tone: 'bad', text: `Could not read ${name}: ${e}` }));
  }, [activeTab, bufs]);

  const loaded = activeTab.kind === 'scene' || !!bufs[activeTab.name];
  const currentValue = activeTab.kind === 'scene' ? sceneJson : bufs[activeTab.name]?.wgsl ?? '// loading…';
  const language = activeTab.kind === 'scene' ? 'json' : 'wgsl';

  function onMount(editor: mEditor.IStandaloneCodeEditor, monaco: Monaco) {
    editorRef.current = editor;
    monacoRef.current = monaco;
    // Register a tiny WGSL grammar — Monaco bundles JSON, not WGSL. We pick
    // up syntax highlighting through a regex-only fallback for now.
    if (!monaco.languages.getLanguages().some((l) => l.id === 'wgsl')) {
      monaco.languages.register({ id: 'wgsl' });
      monaco.languages.setMonarchTokensProvider('wgsl', WGSL_GRAMMAR);
      monaco.languages.setLanguageConfiguration('wgsl', {
        comments: { lineComment: '//', blockComment: ['/*', '*/'] },
        brackets: [
          ['{', '}'],
          ['[', ']'],
          ['(', ')'],
        ],
        autoClosingPairs: [
          { open: '{', close: '}' },
          { open: '[', close: ']' },
          { open: '(', close: ')' },
          { open: '"', close: '"' },
        ],
      });
    }
    // ⌘S / Ctrl+S → save the open tab (through the ref: see the header).
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => saveRef.current());
  }

  async function saveActive() {
    const tab = activeTab;
    const k = tabKey(tab);
    if (tab.kind === 'scene') {
      const text = useStore.getState().sceneJson;
      try {
        JSON.parse(text);
      } catch (e) {
        setStatus({ tab: k, tone: 'bad', text: `Not saved — scene.json doesn't parse: ${(e as Error).message}` });
        return;
      }
      setStatus({ tab: k, tone: 'busy', text: 'Applying scene.json…' });
      try {
        await commitSceneNow(text);
        setStatus({ tab: k, tone: 'ok', text: 'scene.json applied to DESIGN and saved to disk' });
      } catch (e) {
        setStatus({ tab: k, tone: 'bad', text: `Engine rejected scene.json (disk keeps the last good one): ${e}` });
      }
      return;
    }
    const name = tab.name;
    const buf = useBuffers.getState().bufs[name];
    if (!buf) return;
    const wgsl = buf.wgsl;
    setStatus({ tab: k, tone: 'busy', text: `Validating + saving ${name}…` });
    try {
      const r = (await effectUpsert(name, wgsl, null)) as { probe?: { band?: string } | null; warnings?: string[] } | null;
      setBuf(name, (b) => ({ wgsl: b?.wgsl ?? wgsl, saved: wgsl }));
      const band = r?.probe?.band;
      const warn = r?.warnings?.length ? r.warnings.join('; ') : band === 'yellow' ? 'heavy shader (probe: yellow)' : null;
      setStatus({
        tab: k,
        tone: warn ? 'warn' : 'ok',
        text: `Saved effects/${name}/shader.wgsl — hot-reloaded on DESIGN${warn ? ` · ${warn}` : ''}`,
      });
      toast(`Saved ${name}/shader.wgsl`, 'ok');
    } catch (e) {
      // Nothing was written: bad WGSL (line-mapped) or the probe refused it.
      setStatus({ tab: k, tone: 'bad', text: `Not saved — ${e}` });
      runValidate(wgsl);
    }
  }
  saveRef.current = () => void saveActive();

  function onChange(v?: string) {
    if (v === undefined) return;
    // Route by the model that changed, never by `activeTab`: on a tab switch
    // the wrapper can fire a change for the new model (e.g. the transient
    // `loading` one) while its subscription still holds the previous
    // render's closure — routing by tab once wrote "// loading…" into
    // sceneJson and blanked every layer.
    const path = editorRef.current?.getModel()?.uri.path ?? '';
    const tab: TabKind | null =
      path === '/' + tabKey({ kind: 'scene' })
        ? { kind: 'scene' }
        : /^\/effects\/(.+)\/shader\.wgsl$/.test(path)
        ? { kind: 'effect', name: path.slice('/effects/'.length, -'/shader.wgsl'.length) }
        : null;
    if (!tab) return;
    if (status?.tab === tabKey(tab) && status.tone !== 'busy') setStatus(null);
    if (tab.kind === 'scene') {
      setSceneJson(v);
      setSceneDirty(true);
    } else {
      const name = tab.name;
      setBuf(name, (b) => ({ saved: b?.saved ?? '', wgsl: v }));
    }
    if (tab.kind === 'effect') {
      if (validateTimer.current) clearTimeout(validateTimer.current);
      validateTimer.current = window.setTimeout(() => runValidate(v), 180);
    }
  }

  async function runValidate(source: string) {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    // Capture the model now: a tab switch during the round trip must not
    // squiggle the wrong file.
    const model = editor?.getModel();
    if (!monaco || !model) return;
    try {
      const r = await wgslValidate(source);
      if (model.isDisposed()) return;
      const markers: mEditor.IMarkerData[] = r.diagnostics.map((d) => ({
        severity: d.severity === 'error' ? monaco.MarkerSeverity.Error : monaco.MarkerSeverity.Warning,
        message: d.message,
        startLineNumber: d.line,
        startColumn: d.column,
        endLineNumber: d.end_line,
        endColumn: Math.max(d.end_column, d.column + 1),
      }));
      monaco.editor.setModelMarkers(model, 'wgsl-naga', markers);
    } catch (e) {
      console.warn('wgsl.validate failed', e);
    }
  }

  const tabs: TabKind[] = useMemo(
    () => [{ kind: 'scene' }, ...effects.map((name) => ({ kind: 'effect' as const, name }))],
    [effects]
  );

  function isDirty(t: TabKind): boolean {
    if (t.kind === 'scene') return sceneDirty;
    const b = bufs[t.name];
    return !!b && b.wgsl !== b.saved;
  }
  const dirty = isDirty(activeTab);
  const shown = status?.tab === key ? status : null;

  return (
    <div
      className="flex flex-col h-full min-h-0"
      onKeyDown={(e) => {
        // ⌘S with focus on a tab / button in the drawer (the editor's own
        // command covers the text area). Stop it reaching the global handler.
        if ((e.metaKey || e.ctrlKey) && e.key === 's' && !(e.target as HTMLElement).closest('.monaco-editor')) {
          e.preventDefault();
          e.stopPropagation();
          saveRef.current();
        }
      }}
    >
      <div className="flex items-center gap-1 px-2 h-9 bg-panel border-b border-line overflow-x-auto shrink-0">
        {tabs.map((t) => {
          const active = tabKey(t) === key;
          const d = isDirty(t);
          return (
            <button
              key={tabKey(t)}
              onClick={() => setActiveTab(t)}
              title={d ? 'Unsaved changes — ⌘S saves this file' : undefined}
              className={
                'h-6 px-2 text-[11px] rounded-md font-mono shrink-0 ' +
                (active ? 'bg-raised text-fg' : 'text-fg3 hover:text-fg hover:bg-raised/60')
              }
            >
              {d && <span className="text-amber mr-1">●</span>}
              {t.kind === 'scene' ? 'scene.json' : `${t.name}/shader.wgsl`}
            </button>
          );
        })}
        <div className="ml-auto flex items-center gap-2 shrink-0 pl-2">
          {dirty && <span className="text-[10px] text-amber">● modified</span>}
          <button
            className={
              'h-6 px-2 rounded-md text-[11px] font-semibold border ' +
              (dirty
                ? 'border-design/70 bg-design/15 text-design hover:bg-design/25'
                : 'border-line text-fg3 hover:text-fg hover:border-line2')
            }
            disabled={!loaded || shown?.tone === 'busy'}
            onClick={() => saveRef.current()}
            title={activeTab.kind === 'scene' ? 'Apply scene.json to DESIGN and write it (⌘S)' : 'Validate, write and hot-reload this shader (⌘S)'}
          >
            Save <span className="text-fg4 font-normal">⌘S</span>
          </button>
          <button
            className="text-xs text-fg3 hover:text-fg px-1"
            onClick={() => {
              // Re-scan the effects dir and re-read every clean buffer (an
              // agent may have rewritten a shader on disk); dirty ones stay.
              listEffects().then(setEffects);
              useBuffers.setState((s) => ({
                bufs: Object.fromEntries(Object.entries(s.bufs).filter(([, b]) => b.wgsl !== b.saved)),
              }));
            }}
            title="Re-scan effects directory and reload unmodified shaders from disk"
          >
            ↻
          </button>
        </div>
      </div>
      {shown && (
        <div
          className={
            'px-2 py-1 text-[11px] font-mono border-b border-line shrink-0 whitespace-pre-wrap break-words max-h-[30%] overflow-auto ' +
            (shown.tone === 'bad'
              ? 'text-bad bg-bad/10'
              : shown.tone === 'warn'
                ? 'text-amber bg-amber/10'
                : shown.tone === 'ok'
                  ? 'text-ok bg-ok/10'
                  : 'text-fg3')
          }
        >
          {shown.text}
        </div>
      )}
      <div className="flex-1 min-h-0">
        <Editor
          theme="wzrd"
          path={loaded ? key : 'loading'}
          language={language}
          value={currentValue}
          onChange={onChange}
          onMount={onMount}
          options={{
            minimap: { enabled: false },
            fontSize: 12.5,
            padding: { top: 8 },
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
            renderLineHighlight: 'gutter',
            scrollBeyondLastLine: false,
            tabSize: 2,
            readOnly: !loaded,
          }}
        />
      </div>
    </div>
  );
}

// Hand-rolled Monarch grammar for WGSL. Covers keywords + the engine prelude
// types (`FrameState`, `LayerParams`, `sample_mask`, `f_param`, `c_param`)
// so the operator's eye lands on the right things.
const WGSL_GRAMMAR: mLanguages.IMonarchLanguage = {
  defaultToken: '',
  keywords: [
    'fn', 'let', 'var', 'const', 'if', 'else', 'for', 'while', 'loop',
    'continue', 'break', 'return', 'struct', 'true', 'false',
    'switch', 'case', 'default', 'in', 'out',
  ],
  typeKeywords: [
    'f32', 'f16', 'i32', 'u32', 'bool',
    'vec2', 'vec3', 'vec4',
    'mat2x2', 'mat3x3', 'mat4x4',
    'array', 'sampler', 'texture_2d', 'texture_2d_array',
  ],
  operators: [
    '=', '>', '<', '!', '~', '?', ':',
    '==', '<=', '>=', '!=', '&&', '||', '+', '-', '*', '/', '%',
    '+=', '-=', '*=', '/=', '%=', '&', '|', '^', '<<', '>>',
  ],
  symbols: /[=><!~?:&|+\-*/^%]+/,
  tokenizer: {
    root: [
      [/[a-z_$][\w$]*/, {
        cases: {
          '@keywords': 'keyword',
          '@typeKeywords': 'type',
          '@default': 'identifier',
        },
      }],
      [/[A-Z][\w$]*/, 'type.identifier'],
      [/\/\/.*$/, 'comment'],
      [/\/\*/, 'comment', '@comment'],
      [/"([^"\\]|\\.)*$/, 'string.invalid'],
      [/"/, 'string', '@string'],
      [/\d+\.\d*([eE][+\-]?\d+)?/, 'number.float'],
      [/\d+/, 'number'],
      [/[{}()[\]]/, '@brackets'],
      [/@symbols/, {
        cases: { '@operators': 'operator', '@default': '' },
      }],
    ],
    comment: [
      [/[^/*]+/, 'comment'],
      [/\*\//, 'comment', '@pop'],
      [/[/*]/, 'comment'],
    ],
    string: [
      [/[^\\"]+/, 'string'],
      [/\\./, 'string.escape'],
      [/"/, 'string', '@pop'],
    ],
  },
};
