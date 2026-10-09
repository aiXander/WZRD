// Bundle Monaco instead of letting @monaco-editor/react fetch it from a CDN —
// venues are often offline, and a CDN-loaded editor is a blank grey box.

import { loader } from '@monaco-editor/react';
// Editor core + every editor feature (find, folding, …) but only the JSON
// language — the full `monaco-editor` entry drags in ~80 languages (8 MB).
// WGSL gets its own Monarch grammar in MonacoPanel.
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import 'monaco-editor/esm/vs/editor/editor.all.js';
import 'monaco-editor/esm/vs/language/json/monaco.contribution';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';

self.MonacoEnvironment = {
  getWorker(_id: string, label: string) {
    return label === 'json' ? new JsonWorker() : new EditorWorker();
  },
};

monaco.editor.defineTheme('wzrd', {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '6b7283', fontStyle: 'italic' },
    { token: 'keyword', foreground: 'e86bff' },
    { token: 'type', foreground: '3cc8ff' },
    { token: 'type.identifier', foreground: '3cc8ff' },
    { token: 'number', foreground: 'ffb02e' },
    { token: 'number.float', foreground: 'ffb02e' },
    { token: 'string', foreground: '9be15d' },
    { token: 'string.key.json', foreground: 'a8aebd' },
    { token: 'string.value.json', foreground: '9be15d' },
  ],
  colors: {
    'editor.background': '#0d0e12',
    'editor.lineHighlightBackground': '#14161c',
    'editorGutter.background': '#0d0e12',
    'editorLineNumber.foreground': '#3a3f4c',
    'editorLineNumber.activeForeground': '#a8aebd',
    'editor.selectionBackground': '#3cc8ff33',
    'editorCursor.foreground': '#3cc8ff',
    'editorIndentGuide.background1': '#1a1d24',
  },
});

loader.config({ monaco });
