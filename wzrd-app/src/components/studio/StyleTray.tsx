// Studio's bottom strip (A2): the style tray and the target folder — the
// visible half of the hub's Studio selection (L5). An agent asked for
// "two renders with these styles" reads exactly this (`studio_selection()`),
// and renders land in the target folder (= the folder open in Studio, S1).
// Drop library tiles here, or "+ Styles" in the record panel; × removes.

import { mediaUrl } from '../../api/hub';
import { addStyles, removeStyle, short, useStudio } from '../../state/studio';
import { LIBRARY_DRAG } from './MediaGrid';

export function StyleTray() {
  const styles = useStudio((s) => s.selection.styles);
  const target = useStudio((s) => s.selection.target);
  return (
    <section
      className="shrink-0 h-[68px] flex items-center gap-3 px-3 border border-line rounded-lg bg-panel"
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes(LIBRARY_DRAG)) {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
        }
      }}
      onDrop={(e) => {
        const raw = e.dataTransfer.getData(LIBRARY_DRAG);
        if (!raw) return;
        e.preventDefault();
        addStyles(JSON.parse(raw));
      }}
    >
      <div className="label shrink-0 w-12">Styles</div>
      <div className="flex-1 min-w-0 flex items-center gap-1.5 overflow-x-auto">
        {styles.length === 0 && (
          <span className="text-[11px] text-fg4">Drop style images here — what “these styles” means to the agent and the Render panel</span>
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
      <div className="shrink-0 text-right max-w-[38%] min-w-0">
        <div className="label">Renders land in</div>
        <div className="text-[11px] text-fg2 truncate" title={target ?? ''}>
          {target ? short(target) : 'inbox/'}
        </div>
      </div>
    </section>
  );
}
