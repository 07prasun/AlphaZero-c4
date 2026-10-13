/// <reference lib="webworker" />
import { match, netEvaluator, search, selfPlay, type AgentSpec } from "./mcts";
import type { State } from "./game";

type Msg =
  | { id: number; type: "selfplay"; weights: Float32Array; games: number; sims: number }
  | { id: number; type: "match"; a: AgentSpec; b: AgentSpec; games: number; simsA: number; simsB: number; parity: number }
  | { id: number; type: "analyze"; weights: Float32Array; state: State; sims: number };

self.onmessage = (e: MessageEvent<Msg>) => {
  const m = e.data;
  try {
    if (m.type === "selfplay") {
      const t0 = performance.now();
      const r = selfPlay(m.weights, m.games, m.sims);
      (self as unknown as Worker).postMessage({ id: m.id, ok: true, result: { ...r, ms: performance.now() - t0 } }, [
        r.samples.x.buffer,
        r.samples.pi.buffer,
        r.samples.z.buffer,
      ]);
    } else if (m.type === "match") {
      const r = match(m.a, m.b, m.games, m.simsA, m.simsB, m.parity);
      self.postMessage({ id: m.id, ok: true, result: r });
    } else if (m.type === "analyze") {
      const t0 = performance.now();
      const r = search(m.state, netEvaluator(m.weights), { sims: m.sims });
      self.postMessage({ id: m.id, ok: true, result: { ...r, ms: performance.now() - t0 } });
    }
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: String(err) });
  }
};
