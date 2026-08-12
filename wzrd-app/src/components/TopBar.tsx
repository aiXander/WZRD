// Top bar: brand + route tabs + pack/scene labels.
//
// The status strip lives in its own row below this — keeping nav and live
// indicators visually separate so the pills are always the same place
// regardless of which route is active.

import { useStore } from '../state/store';
import { adoptAgentScene } from '../state/sceneCommit';

// Load-in order: build the look, land it on the wall, play, inspect.
const ROUTES = [
  { id: 'prepare', label: 'Prepare', hotkey: '⌘1' },
  { id: 'align', label: 'Align', hotkey: '⌘2' },
  { id: 'perform', label: 'Perform', hotkey: '⌘3' },
  { id: 'debug', label: 'Debug', hotkey: '⌘4' },
] as const;

export function TopBar() {
  const route = useStore((s) => s.route);
  const setRoute = useStore((s) => s.setRoute);
  const pack = useStore((s) => s.pack);
  const agentEdit = useStore((s) => s.agentEdit);

  return (
    <header className="flex items-center gap-4 px-4 py-2 bg-ink-800 border-b border-ink-600">
      <div className="text-sm font-semibold tracking-wide text-zinc-200">WZRD</div>
      <nav className="flex items-center gap-1">
        {ROUTES.map((r) => (
          <button
            key={r.id}
            onClick={() => setRoute(r.id)}
            className={
              'px-3 py-1 text-xs rounded ' +
              (route === r.id
                ? 'bg-ink-600 text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-100 hover:bg-ink-700')
            }
            title={r.hotkey}
          >
            {r.label}
            <span className="ml-2 text-[10px] text-zinc-500">{r.hotkey}</span>
          </button>
        ))}
      </nav>
      <div className="flex-1" />
      {agentEdit && (
        // §5.10 — an agent authored the design scene (rev shown); the store
        // is re-synced but scene.json is not. Adoption is the deliberate
        // human act that persists it.
        <button
          onClick={() => {
            adoptAgentScene().catch((e) => console.error('adopt agent scene:', e));
          }}
          className="px-2 py-1 text-[11px] rounded bg-amber-900/60 text-amber-200 border border-amber-700 hover:bg-amber-800/60"
          title={`Agent edit r${agentEdit.rev}: ${agentEdit.summary}\nSave the agent-authored scene to scene.json`}
        >
          ADOPT AGENT SCENE · r{agentEdit.rev}
        </button>
      )}
      {pack && (
        <div className="text-xs text-zinc-400 truncate max-w-[40rem]">
          <span className="text-zinc-500">pack:</span>{' '}
          <span className="text-zinc-200">{pack.pack_dir}</span>
          <span className="ml-3 text-zinc-500">
            {pack.layers.length} layers · {pack.width}×{pack.height}
          </span>
        </div>
      )}
    </header>
  );
}
