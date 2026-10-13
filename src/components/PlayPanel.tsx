import { useEffect, useMemo, useRef, useState } from "react";
import { COLS, clone, newState, play, type Player, type State } from "../engine/game";
import type { Coordinator, Snapshot } from "../engine/coordinator";
import type { SearchResult } from "../engine/mcts";
import { Board, INK, RED } from "./Board";

type Analysis = SearchResult & { ms: number; ply: number; who: "ai" | "hint"; gen: number };

function replay(moves: number[]): State {
  const s = newState();
  for (const m of moves) play(s, m);
  return s;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function PlayPanel({ coord, snap }: { coord: Coordinator; snap: Snapshot }) {
  const [moves, setMoves] = useState<number[]>([]);
  const [human, setHuman] = useState<Player>(1);
  const [oppGen, setOppGen] = useState<"latest" | number>("latest");
  const [sims, setSims] = useState(400);
  const [hints, setHints] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [thinking, setThinking] = useState(false);
  const [tally, setTally] = useState({ w: 0, d: 0, l: 0 });
  const token = useRef(0);
  const s = useMemo(() => replay(moves), [moves]);
  const aiTurn = !s.done && s.p !== human;

  const resolveGen = (): number => (oppGen === "latest" ? snap.gen : oppGen);
  const weights = () => (oppGen === "latest" ? coord.current : coord.weightsFor(oppGen) ?? coord.current);

  // AI move
  useEffect(() => {
    if (!aiTurn) return;
    const t = ++token.current;
    setThinking(true);
    const gen = resolveGen();
    coord.analyze(weights(), clone(s), sims).then((r) => {
      if (t !== token.current) return;
      let best = 0;
      for (let a = 1; a < COLS; a++) if (r.visits[a] > r.visits[best]) best = a;
      setAnalysis({ ...r, ply: s.n, who: "ai", gen });
      setThinking(false);
      setMoves((m) => [...m, best]);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moves, human, aiTurn]);

  // hints on human turn
  useEffect(() => {
    if (aiTurn || s.done || !hints) return;
    const t = ++token.current;
    const gen = resolveGen();
    coord.analyze(weights(), clone(s), sims).then((r) => {
      if (t !== token.current) return;
      setAnalysis({ ...r, ply: s.n, who: "hint", gen });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moves, hints, aiTurn]);

  // results tally
  const counted = useRef(false);
  useEffect(() => {
    if (!s.done) {
      counted.current = false;
      return;
    }
    if (counted.current) return;
    counted.current = true;
    setTally((t) =>
      s.winner === 0 ? { ...t, d: t.d + 1 } : s.winner === human ? { ...t, w: t.w + 1 } : { ...t, l: t.l + 1 },
    );
  }, [s.done, s.winner, human]);

  const humanMove = (c: number) => {
    if (aiTurn || s.done) return;
    setMoves((m) => [...m, c]);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      const k = parseInt(e.key, 10);
      if (k >= 1 && k <= 7 && s.h[k - 1] < 6) humanMove(k - 1);
      if (e.key === "u") undo();
      if (e.key === "n") newGame(human);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const newGame = (side: Player) => {
    token.current++;
    setThinking(false);
    setAnalysis(null);
    setHuman(side);
    setMoves([]);
  };
  const undo = () => {
    token.current++;
    setThinking(false);
    setAnalysis(null);
    setMoves((m) => {
      let k = m.length;
      // back to the human's previous turn
      do k--;
      while (k > 0 && replay(m.slice(0, k)).p !== human);
      return m.slice(0, Math.max(0, k));
    });
  };

  const showA = analysis && (analysis.who === "ai" ? analysis.ply === s.n - 1 : analysis.ply === s.n);
  const tot = analysis ? analysis.visits.reduce((a, b) => a + b, 0) : 0;
  const heat = showA && analysis!.who === "hint" ? analysis!.visits.map((v) => v / tot) : null;

  // eval from human perspective
  let humanEval: number | null = null;
  // search value is from the searcher's side-to-move view; the AI searched for itself, hints search for you
  if (showA && analysis) humanEval = analysis.who === "ai" ? -analysis.value : analysis.value;

  let status = "";
  if (s.done) status = s.winner === 0 ? "draw." : s.winner === human ? "you win." : "AlphaZero wins.";
  else if (thinking) status = `thinking… (${sims} sims)`;
  else status = "your move — click a column or press 1–7";

  return (
    <section>
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-sm font-semibold tracking-tight">play</h2>
        <span className="text-[11px] text-neutral-500 tabular-nums">
          you {tally.w}–{tally.d}–{tally.l} az
        </span>
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs mb-4 items-center">
        <label className="flex items-center gap-1.5">
          opponent
          <select
            className="border border-neutral-300 bg-white px-1 py-0.5"
            value={String(oppGen)}
            onChange={(e) => setOppGen(e.target.value === "latest" ? "latest" : Number(e.target.value))}
          >
            <option value="latest">latest (gen {snap.gen})</option>
            {[...snap.gens]
              .sort((a, b) => b - a)
              .map((g) => (
                <option key={g} value={g}>
                  gen {g}
                  {snap.history.find((h) => h.gen === g) ? ` · ${Math.round(snap.history.find((h) => h.gen === g)!.elo)}` : ""}
                </option>
              ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          sims
          <select className="border border-neutral-300 bg-white px-1 py-0.5" value={sims} onChange={(e) => setSims(Number(e.target.value))}>
            {[1, 25, 100, 400, 1600].map((n) => (
              <option key={n} value={n}>
                {n === 1 ? "1 (raw policy)" : n}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={hints} onChange={(e) => setHints(e.target.checked)} />
          analysis
        </label>
      </div>

      <Board s={s} canMove={!aiTurn && !s.done} onMove={humanMove} heat={heat} />

      <div className="mt-3 flex items-center gap-3 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: human === 1 ? INK : RED }} />
          you
        </span>
        <span className="text-neutral-500">{status}</span>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-xs">
        <button className="btn" onClick={() => newGame(1)}>
          new · I go first
        </button>
        <button className="btn" onClick={() => newGame(-1)}>
          new · AI goes first
        </button>
        <button className="btn" onClick={undo} disabled={moves.length === 0}>
          undo
        </button>
      </div>

      {/* eval bar */}
      <div className="mt-5">
        <div className="flex justify-between text-[11px] text-neutral-500 mb-1">
          <span>eval (your view)</span>
          <span className="tabular-nums">
            {humanEval === null ? "–" : `${humanEval >= 0 ? "+" : ""}${humanEval.toFixed(2)} · P(win)≈${pct((humanEval + 1) / 2)}`}
          </span>
        </div>
        <div className="h-1.5 bg-neutral-100 relative">
          <div
            className="absolute inset-y-0 left-0 transition-all duration-300"
            style={{ width: humanEval === null ? "50%" : pct((humanEval + 1) / 2), background: human === 1 ? INK : RED }}
          />
          <div className="absolute inset-y-[-2px] left-1/2 w-px bg-neutral-400" />
        </div>
      </div>

      {/* search table */}
      <div className="mt-5">
        <div className="text-[11px] text-neutral-500 mb-1">
          {showA
            ? analysis!.who === "ai"
              ? `AI search (gen ${analysis!.gen}) for its last move · ${tot} sims · ${analysis!.ms.toFixed(0)} ms`
              : `hint: gen ${analysis!.gen} search for your move · ${tot} sims · ${analysis!.ms.toFixed(0)} ms`
            : "search stats appear after the AI moves (or enable analysis)"}
        </div>
        <table className="w-full text-[11px] tabular-nums">
          <thead>
            <tr className="text-neutral-400">
              <td className="py-0.5 w-12">col</td>
              {Array.from({ length: COLS }, (_, c) => (
                <td key={c} className="text-right">
                  {c + 1}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["prior P", (c: number) => (showA ? pct(analysis!.prior[c]) : "")],
                ["visits N", (c: number) => (showA ? String(analysis!.visits[c]) : "")],
                ["value Q", (c: number) => (showA && Number.isFinite(analysis!.q[c]) ? analysis!.q[c].toFixed(2) : "")],
              ] as [string, (c: number) => string][]
            ).map(([name, f]) => (
              <tr key={name} className="border-t border-neutral-100">
                <td className="py-0.5 text-neutral-500">{name}</td>
                {Array.from({ length: COLS }, (_, c) => {
                  const best = showA && analysis!.visits[c] === Math.max(...analysis!.visits) && analysis!.visits[c] > 0;
                  return (
                    <td key={c} className={"text-right " + (best ? "font-semibold" : "")}>
                      {f(c) || "·"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {showA && (
          <div className="text-[11px] text-neutral-500 mt-1 tabular-nums">
            net value v(root) = {analysis!.netValue.toFixed(2)} · search value = {analysis!.value.toFixed(2)} (side to move)
          </div>
        )}
      </div>
    </section>
  );
}
