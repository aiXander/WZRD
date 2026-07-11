// Prepare route — surface canvas (left) + Monaco editor (middle) +
// binding inspector (right). The surface canvas + inspector arrive in 4.2;
// 4.1 ships the middle column only and uses a corner thumbnail in the
// surface column. Both shapes coexist so the file doesn't churn at 4.2
// flip-over.

import { MonacoPanel } from '../components/MonacoPanel';
import { SurfaceCanvas } from '../components/SurfaceCanvas';
import { BindingInspector } from '../components/BindingInspector';

export function Prepare() {
  return (
    <div className="grid grid-cols-[40fr_35fr_25fr] h-full min-h-0">
      {/* min-w-0 on every column: grid children default to min-width:auto,
          so variable-width content (selected-layer label, inspector rows)
          would otherwise resize the whole grid on every click. */}
      <section className="border-r border-ink-700 min-h-0 min-w-0 flex flex-col">
        <header className="text-xs text-zinc-500 px-3 py-1.5 border-b border-ink-700">
          Surface
        </header>
        <div className="flex-1 min-h-0 overflow-auto p-3">
          <SurfaceCanvas />
        </div>
      </section>

      <section className="border-r border-ink-700 min-h-0 min-w-0 flex flex-col">
        <header className="text-xs text-zinc-500 px-3 py-1.5 border-b border-ink-700">
          Editor
        </header>
        <div className="flex-1 min-h-0">
          <MonacoPanel />
        </div>
      </section>

      <section className="min-h-0 min-w-0 flex flex-col">
        <header className="text-xs text-zinc-500 px-3 py-1.5 border-b border-ink-700">
          Binding Inspector
        </header>
        <div className="flex-1 min-h-0 overflow-auto p-3">
          <BindingInspector />
        </div>
      </section>
    </div>
  );
}
