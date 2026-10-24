// Studio's bottom strip (A2): the style tray and the target folder — the
// visible half of the hub's Studio selection (L5). An agent asked for
// "two renders with these styles" reads exactly this (`studio_selection()`),
// and renders land in the target folder.
// Drag library tiles here (pointer drag, `ui/media/pathDrag.ts`), or
// "+ Styles" in the record panel; × removes.
//
// "Renders land in" is a card: a fixed folder the operator picks (the native
// folder dialog, which can create folders; browser seat: a typed path) —
// browsing other folders never changes it. "show" opens it in the grid.

import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { mediaUrl } from '../../api/hub';
import { IN_TAURI } from '../../api/transport';
import { useHub } from '../../state/hub';
import { addStyles, openDir, removeStyle, setTarget, short, useStudio } from '../../state/studio';
import { usePathDrop } from '../../ui/media/pathDrag';

export function StyleTray() {
  const styles = useStudio((s) => s.selection.styles);
  const { dropProps, hovering } = usePathDrop(addStyles);
  return (
    <section
      {...dropProps}
      className={
        'shrink-0 h-[68px] flex items-center gap-3 px-3 border rounded-lg bg-panel transition-colors ' +
        (hovering ? 'border-deck bg-deck/10' : 'border-line')
      }
    >
      <div className="label shrink-0 w-12">Styles</div>
      <div className="flex-1 min-w-0 flex items-center gap-1.5 overflow-x-auto">
        {styles.length === 0 && (
          <span className="text-[11px] text-fg4">
            {hovering ? 'Drop to add as styles' : 'Drag style images here — what “these styles” means to the agent and the Render panel'}
          </span>
        )}
        {styles.map((p) => (
          <div key={p} className="group relative h-12 aspect-square shrink-0 rounded border border-line2 overflow-hidden bg-black" title={p}>
            <img src={mediaUrl(p)} loading="lazy" className="w-full h-full object-cover" draggable={false} />
            <button
              onClick={() => removeStyle(p)}
              className="absolute top-0 right-0 w-4 h-4 text-[10px] leading-none bg-black/70 text-fg2 hidden group-hover:block hover:text-bad"
              title="Remove from the tray"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <TargetCard />
    </section>
  );
}

/** "Renders land in": click to pick (or create) the output folder; "show" opens it in the grid. */
function TargetCard() {
  const target = useStudio((s) => s.selection.target);
  const up = useHub((s) => s.status?.state === 'up');
  const mediaRoot = useHub((s) => s.status?.media_root);
  const choose = async () => {
    let path: string | null = null;
    if (IN_TAURI) {
      const r = await openDialog({
        directory: true,
        multiple: false,
        canCreateDirectories: true,
        defaultPath: target ?? mediaRoot ?? undefined,
        title: 'Renders land in…',
      });
      path = typeof r === 'string' ? r : null;
    } else {
      path = window.prompt('Absolute folder path renders should land in', target ?? mediaRoot ?? '');
    }
    if (path) setTarget(path.replace(/\/+$/, ''));
  };
  return (
    <div className="shrink-0 max-w-[40%] min-w-0 flex items-stretch rounded-md border border-line hover:border-line2 bg-panel2">
      <button
        type="button"
        onClick={choose}
        disabled={!up}
        className="min-w-0 flex items-center gap-2 pl-2 pr-2.5 py-1 text-left group disabled:opacity-50"
        title={`${target ?? 'inbox/ (the library home)'}\nClick to choose where new renders land — it stays until you change it`}
      >
        <svg width="16" height="14" viewBox="0 0 16 14" className="shrink-0 text-fg3 group-hover:text-deck" fill="none" stroke="currentColor" strokeWidth="1.3" aria-hidden>
          <path d="M1 3.2V12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V4.6a1 1 0 0 0-1-1H7.4L5.9 1.8a1 1 0 0 0-.8-.4H2a1 1 0 0 0-1 1z" />
        </svg>
        <span className="min-w-0 flex flex-col">
          <span className="label">Renders land in</span>
          <span className="text-[11px] text-fg2 group-hover:text-fg truncate">{target ? short(target) : 'inbox/ (click to choose)'}</span>
        </span>
      </button>
      {target && (
        <button
          type="button"
          onClick={() => openDir(target)}
          className="shrink-0 px-2 border-l border-line text-[10px] text-fg3 hover:text-fg"
          title="Open this folder in the grid"
        >
          show
        </button>
      )}
    </div>
  );
}
