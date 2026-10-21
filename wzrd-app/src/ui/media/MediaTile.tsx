// One media tile (show-setup M1): a static thumb (the cached thumb, else the
// image itself; a video without a thumb = ▶ placeholder — never autoplay),
// the name, caller badges (top-left), hover actions (⤢ open, ★ pin, ✕
// delete — each shown only when its handler is given), asset-name chips
// (`pins`) along the bottom of the picture, a ring when selected.

import { memo, useState } from 'react';
import { mediaUrl } from '../../api/hub';
import { baseName, kindOf, thumbOf, type MediaRef } from './types';

export const MediaTile = memo(function MediaTile({
  media,
  selected = false,
  onClick,
  onOpen,
  badges,
  pins,
  onPin,
  onDelete,
  aspect = '16 / 9',
  fit = 'contain',
}: {
  media: MediaRef;
  selected?: boolean;
  onClick?: (e: React.MouseEvent) => void;
  onOpen?: () => void;
  badges?: React.ReactNode;
  pins?: string[];
  onPin?: (anchor: HTMLElement) => void;
  onDelete?: () => void;
  aspect?: string;
  fit?: 'cover' | 'contain';
}) {
  const [broken, setBroken] = useState(false);
  const thumb = thumbOf(media);
  const kind = kindOf(media);
  const name = media.name ?? baseName(media.path);
  return (
    <div
      onClick={onClick}
      onDoubleClick={onOpen}
      title={name}
      className={
        'group relative rounded-md overflow-hidden border select-none bg-bg ' +
        (onClick || onOpen ? 'cursor-pointer ' : '') +
        (selected ? 'border-deck ring-1 ring-deck' : 'border-line hover:border-line2')
      }
    >
      <div className="relative bg-black flex items-center justify-center" style={{ aspectRatio: aspect }}>
        {thumb && !broken ? (
          <img
            src={mediaUrl(thumb)}
            loading="lazy"
            decoding="async"
            draggable={false}
            onError={() => setBroken(true)}
            className={'w-full h-full ' + (fit === 'cover' ? 'object-cover' : 'object-contain')}
          />
        ) : kind === 'video' ? (
          <span className="text-[20px] text-fg3">▶</span>
        ) : (
          <span className="text-[10px] text-fg3 px-2 text-center break-all">{media.item?.codec ?? name}</span>
        )}
        {kind === 'video' && thumb && !broken && (
          <span className="absolute bottom-1 right-1 text-[9px] px-1 rounded bg-black/70 text-fg2">▶</span>
        )}
        {badges && <div className="absolute top-1 left-1 flex gap-0.5 items-start">{badges}</div>}
        {(onOpen || onPin || onDelete) && (
          <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            {onOpen && (
              <TileBtn title="Open full-screen" onClick={() => onOpen()}>
                ⤢
              </TileBtn>
            )}
            {onPin && (
              <TileBtn title="Pin to the show as a named asset…" onClick={(el) => onPin(el)}>
                ★
              </TileBtn>
            )}
            {onDelete && (
              <TileBtn title="Delete (to the trash — restorable for 30 min)" danger onClick={() => onDelete()}>
                ✕
              </TileBtn>
            )}
          </div>
        )}
        {pins && pins.length > 0 && (
          <div className="absolute bottom-1 left-1 right-6 flex flex-wrap gap-0.5 overflow-hidden max-h-[34px]">
            {pins.map((p) => (
              <span
                key={p}
                className="text-[9px] font-mono px-1 rounded bg-amber/90 text-black whitespace-nowrap"
                title={`Pinned as ${p}`}
              >
                ★ {p}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="px-1.5 py-1 text-[10px] text-fg2 truncate">{name}</div>
    </div>
  );
});

function TileBtn({
  title,
  onClick,
  danger,
  children,
}: {
  title: string;
  onClick: (el: HTMLElement) => void;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick(e.currentTarget);
      }}
      onDoubleClick={(e) => e.stopPropagation()}
      className={
        'w-6 h-6 rounded bg-black/75 text-[12px] leading-none text-fg2 border border-line2 ' +
        (danger ? 'hover:text-bad hover:border-bad/70' : 'hover:text-fg hover:border-fg4')
      }
    >
      {children}
    </button>
  );
}
