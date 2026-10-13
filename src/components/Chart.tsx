import { useState } from "react";

export interface Series {
  name: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
}

interface Props {
  series: Series[];
  height?: number;
  yDomain?: [number, number];
  yFmt?: (v: number) => string;
  xLabel?: string;
}

const W = 560;

export function Chart({ series, height = 170, yDomain, yFmt = (v) => v.toFixed(0), xLabel = "gen" }: Props) {
  const [hx, setHx] = useState<number | null>(null);
  const H = height;
  const pad = { l: 44, r: 10, t: 10, b: 22 };
  const all = series.flatMap((s) => s.points.filter((p) => Number.isFinite(p[1])));
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const x0 = xs.length ? Math.min(...xs) : 0;
  const x1 = xs.length ? Math.max(...xs, x0 + 1) : 1;
  let [y0, y1] = yDomain ?? [ys.length ? Math.min(...ys) : 0, ys.length ? Math.max(...ys) : 1];
  if (!yDomain) {
    const span = Math.max(1e-6, y1 - y0);
    y0 -= span * 0.08;
    y1 += span * 0.08;
    if (y1 - y0 < 1e-6) y1 = y0 + 1;
  }
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y: number) => H - pad.b - ((y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const ticks = [0, 1, 2, 3, 4].map((i) => y0 + ((y1 - y0) * i) / 4);
  const xticks = Array.from(new Set([0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(x0 + (x1 - x0) * f))));

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const x = x0 + ((px - pad.l) / (W - pad.l - pad.r)) * (x1 - x0);
    setHx(Math.max(x0, Math.min(x1, Math.round(x))));
  };

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" onMouseMove={onMove} onMouseLeave={() => setHx(null)}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke="#eee" />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize="10" fill="#999">
              {yFmt(t)}
            </text>
          </g>
        ))}
        {xticks.map((t) => (
          <text key={t} x={sx(t)} y={H - 6} textAnchor="middle" fontSize="10" fill="#999">
            {t}
          </text>
        ))}
        <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10" fill="#bbb">
          {xLabel}
        </text>
        {series.map((s) => {
          const pts = s.points.filter((p) => Number.isFinite(p[1]));
          if (!pts.length) return null;
          const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth="1.5" strokeDasharray={s.dashed ? "4 3" : undefined} />
              {pts.length < 40 && pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r="1.8" fill={s.color} />)}
            </g>
          );
        })}
        {hx !== null && <line x1={sx(hx)} x2={sx(hx)} y1={pad.t} y2={H - pad.b} stroke="#ccc" strokeDasharray="2 2" />}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-neutral-500 mt-1">
        {series.map((s) => {
          const p = hx !== null ? s.points.find((q) => q[0] === hx) : s.points[s.points.length - 1];
          return (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <span className="inline-block w-3 h-[2px]" style={{ background: s.color }} />
              {s.name}
              <span className="text-neutral-900 tabular-nums">{p && Number.isFinite(p[1]) ? yFmt(p[1]) : "–"}</span>
            </span>
          );
        })}
        {hx !== null && <span className="ml-auto tabular-nums">gen {hx}</span>}
      </div>
    </div>
  );
}
