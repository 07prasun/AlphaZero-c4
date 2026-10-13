import { useState } from "react";
import { COLS, ROWS, winLine, type State } from "../engine/game";

export const INK = "#161616";
export const RED = "#e0482f";

interface Props {
  s: State;
  canMove: boolean;
  onMove: (c: number) => void;
  heat?: number[] | null; // per-column visit share [0,1]
}

export function Board({ s, canMove, onMove, heat }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const line = new Set(winLine(s) ?? []);
  const cell = 52;
  const W = COLS * cell;
  const H = ROWS * cell;
  const maxHeat = heat ? Math.max(...heat, 1e-9) : 1;

  return (
    <div className="inline-block">
      {/* heat strip */}
      <div className="flex" style={{ width: W }}>
        {Array.from({ length: COLS }, (_, c) => {
          const h = heat?.[c] ?? 0;
          return (
            <div key={c} className="flex flex-col items-center justify-end" style={{ width: cell, height: 30 }}>
              {heat && (
                <>
                  <span className="text-[10px] tabular-nums text-neutral-500">{h > 0 ? Math.round(h * 100) : ""}</span>
                  <div className="w-6 bg-neutral-800" style={{ height: Math.max(h > 0 ? 1 : 0, (h / maxHeat) * 14), opacity: 0.25 + 0.75 * (h / maxHeat) }} />
                </>
              )}
            </div>
          );
        })}
      </div>
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        className={canMove ? "cursor-pointer" : "cursor-default"}
        onMouseLeave={() => setHover(null)}
      >
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="#fff" stroke="#d4d4d4" />
        {Array.from({ length: COLS }, (_, c) => (
          <rect
            key={"col" + c}
            x={c * cell}
            y={0}
            width={cell}
            height={H}
            fill={canMove && hover === c && s.h[c] < ROWS ? "#f6f6f4" : "transparent"}
            onMouseEnter={() => setHover(c)}
            onClick={() => canMove && s.h[c] < ROWS && onMove(c)}
          />
        ))}
        {Array.from({ length: ROWS * COLS }, (_, i) => {
          const r = Math.floor(i / COLS);
          const c = i % COLS;
          const v = s.b[i];
          const cx = c * cell + cell / 2;
          const cy = (ROWS - 1 - r) * cell + cell / 2;
          const ghost = canMove && hover === c && s.h[c] === r && !v;
          return (
            <g key={i} pointerEvents="none">
              <circle cx={cx} cy={cy} r={cell * 0.38} fill="none" stroke="#ececec" />
              {v !== 0 && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={cell * 0.38}
                  fill={v === 1 ? INK : RED}
                  opacity={s.done && line.size && !line.has(i) ? 0.35 : 1}
                />
              )}
              {ghost && <circle cx={cx} cy={cy} r={cell * 0.38} fill={s.p === 1 ? INK : RED} opacity={0.15} />}
              {s.last === i && <circle cx={cx} cy={cy} r={4} fill="#fff" />}
            </g>
          );
        })}
      </svg>
      <div className="flex text-[10px] text-neutral-400 tabular-nums" style={{ width: W }}>
        {Array.from({ length: COLS }, (_, c) => (
          <div key={c} className="text-center" style={{ width: cell }}>
            {c + 1}
          </div>
        ))}
      </div>
    </div>
  );
}
