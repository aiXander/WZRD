// Soft delete from the app (show-setup G6 / T1): the hub moves files to a
// trash and flushes them after 30 min. Every app-side delete goes through
// `softDelete`, which toasts with an **Undo** (restores exactly what this
// call trashed) and bumps `useTrash.version` so a `<RecentlyDeleted>`
// button re-counts.

import { create } from 'zustand';
import { deleteFiles } from '../../api/hub';
import { trashRestore } from '../../api/project';
import { useStore } from '../../state/store';

export const useTrash = create<{ version: number }>(() => ({ version: 0 }));
export const trashChanged = () => useTrash.setState((s) => ({ version: s.version + 1 }));

type Trashed = { id: string; path: string }[];

const plural = (n: number, w = 'file') => `${n} ${w}${n === 1 ? '' : 's'}`;

/** Offer Undo for what a delete call trashed (its `trashed` list). */
export function undoToast(text: string, trashed: Trashed | undefined, after?: () => void) {
  const toast = useStore.getState().toast;
  trashChanged();
  if (!trashed?.length) return toast(text, 'ok');
  toast(text, 'ok', {
    action: {
      label: 'Undo',
      run: async () => {
        try {
          const r = await trashRestore(trashed.map((t) => t.id));
          if (r.errors?.length) toast(`Restore: ${r.errors.join('; ')}`, 'warn');
          else toast(`Restored ${plural(r.restored.length)}`, 'ok');
        } catch (e) {
          toast(`Restore failed: ${e}`, 'bad');
        }
        trashChanged();
        after?.();
      },
    },
  });
}

/** Delete media files (to the trash) + an Undo toast. Resolves true on success. */
export async function softDelete(paths: string[], after?: () => void): Promise<boolean> {
  if (!paths.length) return false;
  try {
    const r = (await deleteFiles(paths)) as { trashed?: Trashed } | null;
    undoToast(`Moved ${plural(paths.length)} to the trash — restorable for 30 min`, r?.trashed, after);
    return true;
  } catch (e) {
    useStore.getState().toast(`Delete failed: ${e}`, 'bad');
    return false;
  }
}
