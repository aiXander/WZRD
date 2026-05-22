// Monaco editor mediator. Owns the tab list (scene.json + every
// effects/*.wgsl) and bridges saves to the engine.
//
// Phase 4.1: scene.json saves go through `scene_load` (immediate engine
// reload, no disk write); we *also* persist to disk so the headless agent
// path keeps working. Effect saves use `effect_upsert` which writes to
// disk under <scene_dir>/effects/<name>/.
//
// `wgsl.validate` runs debounced on the active editor's source for any
// .wgsl tab, returning naga diagnostics into Monaco's marker system.

import { Editor, type Monaco } from '@monaco-editor/react';
import type { editor as mEditor, languages as mLanguages } from 'monaco-editor';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  effectUpsert,
  listEffects,
  readEffect,
  sceneLoad,
  wgslValidate,
  writeSceneFile,
} from '../api/ipc';
import { useStore } from '../state/store';

type TabKind = { kind: 'scene' } | { kind: 'effect'; name: string };

function tabLabel(t: TabKind, dirty: boolean): string {
  const base = t.kind === 'scene' ? 'scene.json' : `${t.name}/shader.wgsl`;
  return dirty ? `● ${base}` : base;
}

export function MonacoPanel() {
  const editorRef = useRef<mEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const sceneJson = useStore((s) => s.sceneJson);
  const setSceneJson = useStore((s) => s.setSceneJson);
  const sceneDirty = useStore((s) => s.sceneDirty);
  const setSceneDirty = useStore((s) => s.setSceneDirty);
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const effects = useStore((s) => s.effects);
  const setEffects = useStore((s) => s.setEffects);

  // Buffer state for each open effect file. Keyed by effect name.
  const [effectBuffers, setEffectBuffers] = useState<
    Record<string, { wgsl: string; descriptor: string | null; dirty: boolean }>
  >({});
  const validateTimer = useRef<number | null>(null);

  // Load an effect's content lazily when its tab is selected.
  useEffect(() => {
    if (activeTab.kind !== 'effect') return;
    const name = activeTab.name;
    if (effectBuffers[name]) return;
    readEffect(name)
      .then((data) =>
        setEffectBuffers((b) => ({
          ...b,
          [name]: { wgsl: data.wgsl, descriptor: data.descriptor, dirty: false },
        }))
      )
      .catch((e) => console.warn('read_effect', name, e));
  }, [activeTab, effectBuffers]);

  const currentValue = useMemo(() => {
    if (activeTab.kind === 'scene') return sceneJson;
    return effectBuffers[activeTab.name]?.wgsl ?? '// loading…';
  }, [activeTab, sceneJson, effectBuffers]);

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

    // ⌘S / Ctrl+S → save active tab.
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, async () => {
      await saveActive();
    });
  }

  async function saveActive() {
    const editor = editorRef.current;
    if (!editor) return;
    const value = editor.getValue();
    if (activeTab.kind === 'scene') {
      try {
        await sceneLoad(value);
        await writeSceneFile(value);
        setSceneJson(value);
        setSceneDirty(false);
      } catch (e) {
        console.error('scene save failed', e);
      }
    } else {
      const name = activeTab.name;
      const descRaw = effectBuffers[name]?.descriptor;
      let descriptor: unknown = null;
      if (descRaw) {
        try {
          descriptor = JSON.parse(descRaw);
        } catch {
          /* ignore — engine will reject and report via hot_reload */
        }
      }
      try {
        await effectUpsert(name, value, descriptor);
        setEffectBuffers((b) => ({
          ...b,
          [name]: { ...b[name]!, wgsl: value, dirty: false },
        }));
      } catch (e) {
        console.error('effect save failed', e);
      }
    }
  }

  function onChange(v?: string) {
    if (v === undefined) return;
    if (activeTab.kind === 'scene') {
      setSceneJson(v);
      setSceneDirty(true);
    } else {
      const name = activeTab.name;
      setEffectBuffers((b) => ({
        ...b,
        [name]: { ...(b[name] ?? { descriptor: null, dirty: false }), wgsl: v, dirty: true },
      }));
    }
    if (language === 'wgsl') {
      if (validateTimer.current) {
        clearTimeout(validateTimer.current);
      }
      validateTimer.current = window.setTimeout(() => runValidate(v), 180);
    }
  }

  async function runValidate(source: string) {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    if (!editor || !monaco) return;
    try {
      const r = await wgslValidate(source);
      const model = editor.getModel();
      if (!model) return;
      const markers: mEditor.IMarkerData[] = r.diagnostics.map((d) => ({
        severity:
          d.severity === 'error'
            ? monaco.MarkerSeverity.Error
            : monaco.MarkerSeverity.Warning,
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
    return !!effectBuffers[t.name]?.dirty;
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 px-2 py-1 bg-ink-800 border-b border-ink-700 overflow-x-auto">
        {tabs.map((t) => {
          const active =
            (activeTab.kind === 'scene' && t.kind === 'scene') ||
            (activeTab.kind === 'effect' &&
              t.kind === 'effect' &&
              t.name === activeTab.name);
          return (
            <button
              key={t.kind === 'scene' ? 'scene' : `effect:${t.name}`}
              onClick={() => setActiveTab(t)}
              className={
                'px-2 py-1 text-xs rounded ' +
                (active ? 'bg-ink-600 text-zinc-100' : 'text-zinc-400 hover:bg-ink-700')
              }
            >
              {tabLabel(t, isDirty(t))}
            </button>
          );
        })}
        <button
          className="ml-auto text-xs text-zinc-400 hover:text-zinc-100"
          onClick={() => listEffects().then(setEffects)}
          title="Re-scan effects directory"
        >
          ↻
        </button>
      </div>
      <div className="flex-1 min-h-0">
        <Editor
          theme="vs-dark"
          language={language}
          value={currentValue}
          onChange={onChange}
          onMount={onMount}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            fontFamily:
              'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
            renderLineHighlight: 'gutter',
            scrollBeyondLastLine: false,
            tabSize: 2,
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
