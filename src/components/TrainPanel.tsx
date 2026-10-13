import type { Config, Coordinator, Snapshot } from "../engine/coordinator";
import { Chart } from "./Chart";
import { INK, RED } from "./Board";

const hw = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;

function Num({
  label,
  k,
  cfg,
  coord,
  opts,
}: {
  label: string;
  k: keyof Config;
  cfg: Config;
  coord: Coordinator;
  opts: number[];
}) {
  return (
    <label className="flex items-center justify-between gap-2">
      <span className="text-neutral-500">{label}</span>
      <select
        className="border border-neutral-300 bg-white px-1 py-0.5 tabular-nums"
        value={cfg[k]}
        onChange={(e) => coord.setConfig({ [k]: Number(e.target.value) } as Partial<Config>)}
      >
        {opts.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

const fmtK = (n: number) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n)));

export function TrainPanel({ coord, snap }: { coord: Coordinator; snap: Snapshot }) {
  const h = snap.history;
  const last = h[h.length - 1];
  const cfg = snap.config;

  const download = () => {
    const blob = new Blob([coord.exportModel()], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `az4-gen${snap.gen}.json`;
    a.click();
  };

  return (
    <section>
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-sm font-semibold tracking-tight">train</h2>
        <span className="text-[11px] text-neutral-500">
          {snap.running ? (snap.stopping ? "finishing generation…" : snap.phase) : "idle"}
          {snap.running && <span className="inline-block w-1.5 h-1.5 ml-2 rounded-full bg-emerald-500 animate-pulse align-middle" />}
        </span>
      </div>

      <div className="flex flex-wrap gap-2 text-xs mb-4">
        {snap.running ? (
          <button className="btn btn-primary" onClick={() => coord.stop()} disabled={snap.stopping}>
            pause
          </button>
        ) : (
          <button className="btn btn-primary" onClick={() => coord.start()}>
            {snap.gen === 0 ? "start self-play" : "resume"}
          </button>
        )}
        <button
          className="btn"
          onClick={() => {
            if (confirm("Discard all generations and start from a random network?")) coord.reset();
          }}
        >
          reset
        </button>
        <button className="btn" onClick={download}>
          export weights
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-neutral-200 border border-neutral-200 mb-4 text-xs">
        {[
          ["generation", String(snap.gen)],
          ["Elo", last ? Math.round(last.elo).toString() : "0"],
          ["self-play games", fmtK(snap.totalGames)],
          ["NN evals / s", snap.posPerSec ? fmtK(snap.posPerSec) : "–"],
        ].map(([k, v]) => (
          <div key={k} className="bg-white px-3 py-2">
            <div className="text-neutral-400 text-[10px] uppercase tracking-wider">{k}</div>
            <div className="text-lg tabular-nums leading-tight">{v}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs mb-5">
        <Num label="workers" k="workers" cfg={cfg} coord={coord} opts={Array.from({ length: Math.max(1, hw) }, (_, i) => i + 1)} />
        <Num label="sims / move" k="sims" cfg={cfg} coord={coord} opts={[25, 50, 80, 120, 200, 400]} />
        <Num label="games / gen" k="gamesPerGen" cfg={cfg} coord={coord} opts={[16, 32, 48, 96, 192]} />
        <Num label="eval games" k="evalGames" cfg={cfg} coord={coord} opts={[10, 20, 40, 80]} />
        <Num label="eval sims" k="evalSims" cfg={cfg} coord={coord} opts={[32, 64, 128, 256]} />
        <Num label="ref MCTS sims" k="refSims" cfg={cfg} coord={coord} opts={[100, 200, 400, 1000]} />
      </div>

      <h3 className="label">Elo progression</h3>
      <Chart series={[{ name: "Elo", color: INK, points: h.map((r) => [r.gen, r.elo]) }]} height={180} />

      <h3 className="label mt-5">match score</h3>
      <Chart
        series={[
          { name: "vs prev gen", color: "#999", points: h.slice(1).map((r) => [r.gen, r.vsPrev]), dashed: true },
          { name: `vs pure MCTS`, color: RED, points: h.map((r) => [r.gen, r.vsRef]) },
          { name: "vs random", color: "#3b82f6", points: h.map((r) => [r.gen, r.vsRandom]) },
        ]}
        yDomain={[0, 1]}
        yFmt={(v) => v.toFixed(2)}
        height={140}
      />

      <h3 className="label mt-5">training loss</h3>
      <Chart
        series={[
          { name: "policy CE", color: INK, points: h.map((r) => [r.gen, r.lp]) },
          { name: "value MSE", color: RED, points: h.map((r) => [r.gen, r.lv]) },
        ]}
        yFmt={(v) => v.toFixed(2)}
        height={140}
      />

      <h3 className="label mt-5">generations</h3>
      <div className="max-h-56 overflow-auto border-t border-neutral-200">
        <table className="w-full text-[11px] tabular-nums">
          <thead className="sticky top-0 bg-white text-neutral-400">
            <tr>
              <th className="text-left font-normal py-1">gen</th>
              <th className="text-right font-normal">Elo</th>
              <th className="text-right font-normal">vs prev</th>
              <th className="text-right font-normal">vs mcts</th>
              <th className="text-right font-normal">len</th>
              <th className="text-right font-normal">P1 win</th>
              <th className="text-right font-normal">buffer</th>
              <th className="text-right font-normal">time</th>
            </tr>
          </thead>
          <tbody>
            {[...h].reverse().map((r) => (
              <tr key={r.gen} className="border-t border-neutral-100">
                <td className="py-0.5">{r.gen}</td>
                <td className="text-right">{Math.round(r.elo)}</td>
                <td className="text-right">{Number.isFinite(r.vsPrev) && r.gen > 0 ? r.vsPrev.toFixed(2) : "–"}</td>
                <td className="text-right">{Number.isFinite(r.vsRef) ? r.vsRef.toFixed(2) : "–"}</td>
                <td className="text-right">{Number.isFinite(r.plies) ? r.plies.toFixed(1) : "–"}</td>
                <td className="text-right">{Number.isFinite(r.firstWin) ? r.firstWin.toFixed(2) : "–"}</td>
                <td className="text-right">{r.buffer ? fmtK(r.buffer) : "–"}</td>
                <td className="text-right">{r.secs ? r.secs.toFixed(1) + "s" : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="label mt-5">log</h3>
      <pre className="text-[10.5px] leading-relaxed text-neutral-600 max-h-40 overflow-auto whitespace-pre-wrap border-t border-neutral-200 pt-1">
        {snap.log.join("\n") || "—"}
      </pre>
    </section>
  );
}
