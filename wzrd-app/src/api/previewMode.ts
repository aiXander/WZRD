// How the native preview is hosted, resolved once at boot (before the first
// render) so components can branch synchronously:
//   underlay — a GPU view under a transparent page; `<NativeSlot>` shows the
//              native image in every room and HTML draws over it.
//   window   — `WZRD_PREVIEW=window`: the old child window above the page.
//              Only Perform's monitor is native; Prepare/Align draw the JPEG.
//   none     — plain browser (dev bridge): the JPEG everywhere.

import { previewModeGet, previewSetSlot } from './ipc';
import { IN_TAURI } from './transport';

type PreviewMode = 'underlay' | 'window' | 'none';

let mode: PreviewMode = 'none';

export const previewMode = () => mode;
/** True when every room's image comes from the native underlay. */
export const nativeUnderlay = () => mode === 'underlay';

export async function initPreviewMode() {
  if (!IN_TAURI) return;
  mode = await previewModeGet().catch(() => 'window' as const);
  // A page reload skips React's unmount cleanup, so the shell may still hold
  // the previous page's slot (visible, stale params). Start from hidden; the
  // first mounted <NativeSlot> shows it again.
  await previewSetSlot({ x: 0, y: 0, width: 0, height: 0, visible: false }).catch(() => {});
  // index.css: the page goes transparent so the underlay shows through
  // wherever nothing paints. The native window background is the app --bg.
  if (mode === 'underlay') document.documentElement.dataset.underlay = '';
}
