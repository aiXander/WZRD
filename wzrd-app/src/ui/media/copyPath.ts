// One-click "copy the full path" for library files: paste it straight into an
// agent ("turn this image into a shader"). Several paths = one per line.

import { useStore } from '../../state/store';

async function writeText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    // Fall through: a webview that refuses the async API still allows execCommand in a click.
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand('copy');
  ta.remove();
  if (!ok) throw new Error('clipboard unavailable');
}

export async function copyPaths(paths: string[]): Promise<void> {
  const toast = useStore.getState().toast;
  try {
    await writeText(paths.join('\n'));
    toast(paths.length === 1 ? `Copied path: ${paths[0]}` : `Copied ${paths.length} paths`, 'ok');
  } catch (e) {
    toast(`Copy failed: ${e}`, 'bad');
  }
}
