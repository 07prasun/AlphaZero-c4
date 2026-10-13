import { CELLS, COLS, INPUT, clone, encode, legal, play, type State } from "./game";
import { Net } from "./nn";

export type Evaluator = (s: State) => { p: ArrayLike<number>; v: number };

export interface SearchOpts {
  sims: number;
  cpuct?: number;
  noise?: boolean;
  alpha?: number;
  eps?: number;
}

export interface SearchResult {
  visits: number[];
  q: number[];
  prior: number[];
  value: number; // root value, side-to-move perspective, [-1,1]
  netValue: number; // raw network value at root
}

class Node {
  P = new Float32Array(COLS);
  N = new Float32Array(COLS);
  W = new Float32Array(COLS);
  kids: (Node | null)[] = new Array(COLS).fill(null);
  legal: number[] = [];
  n = 0;
}

function expand(node: Node, s: State, ev: Evaluator): number {
  node.legal = legal(s);
  const { p, v } = ev(s);
  let sum = 0;
  for (const a of node.legal) sum += p[a];
  for (const a of node.legal) node.P[a] = sum > 0 ? p[a] / sum : 1 / node.legal.length;
  return v;
}

function select(node: Node, c: number): number {
  const sq = Math.sqrt(Math.max(1, node.n));
  let best = -1;
  let bestU = -Infinity;
  for (const a of node.legal) {
    const n = node.N[a];
    const q = n > 0 ? node.W[a] / n : 0;
    const u = q + (c * node.P[a] * sq) / (1 + n);
    if (u > bestU) {
      bestU = u;
      best = a;
    }
  }
  return best;
}

function gamma(alpha: number): number {
  if (alpha < 1) return gamma(alpha + 1) * Math.pow(Math.random(), 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0;
    let v = 0;
    do {
      const u1 = Math.random() || 1e-12;
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export function search(root: State, ev: Evaluator, o: SearchOpts): SearchResult {
  const c = o.cpuct ?? 1.5;
  const rootNode = new Node();
  const netValue = expand(rootNode, root, ev);
  if (o.noise) {
    const eps = o.eps ?? 0.25;
    const ns = rootNode.legal.map(() => gamma(o.alpha ?? 1.0));
    const t = ns.reduce((a, b) => a + b, 0);
    rootNode.legal.forEach((a, i) => {
      rootNode.P[a] = (1 - eps) * rootNode.P[a] + (eps * ns[i]) / t;
    });
  }
  const pathN: Node[] = [];
  const pathA: number[] = [];
  for (let sim = 0; sim < o.sims; sim++) {
    const s = clone(root);
    let node = rootNode;
    pathN.length = 0;
    pathA.length = 0;
    let v = 0;
    for (;;) {
      const a = select(node, c);
      pathN.push(node);
      pathA.push(a);
      play(s, a);
      if (s.done) {
        v = s.winner !== 0 ? -1 : 0; // side to move at s just lost (or draw)
        break;
      }
      let child = node.kids[a];
      if (!child) {
        child = new Node();
        node.kids[a] = child;
        v = expand(child, s, ev);
        break;
      }
      node = child;
    }
    for (let i = pathN.length - 1; i >= 0; i--) {
      v = -v;
      const nd = pathN[i];
      const a = pathA[i];
      nd.N[a] += 1;
      nd.W[a] += v;
      nd.n += 1;
    }
  }
  const visits: number[] = [];
  const q: number[] = [];
  const prior: number[] = [];
  let tw = 0;
  for (let a = 0; a < COLS; a++) {
    visits.push(rootNode.N[a]);
    q.push(rootNode.N[a] > 0 ? rootNode.W[a] / rootNode.N[a] : NaN);
    prior.push(rootNode.P[a]);
    tw += rootNode.W[a];
  }
  return { visits, q, prior, value: rootNode.n > 0 ? tw / rootNode.n : netValue, netValue };
}

// ---------- evaluators ----------

export function netEvaluator(weights: Float32Array): Evaluator {
  const net = new Net(weights);
  const x = new Float32Array(INPUT);
  return (s) => {
    const p = new Float32Array(COLS);
    encode(s, x);
    const v = net.forward(x, 0, p);
    return { p, v };
  };
}

const uniform = new Float32Array(COLS).fill(1 / COLS);
/** Classic pure MCTS: uniform prior + random rollout. Used as a fixed reference opponent. */
export const rolloutEvaluator: Evaluator = (s) => {
  const me = s.p;
  const t = clone(s);
  while (!t.done) {
    const l = legal(t);
    play(t, l[(Math.random() * l.length) | 0]);
  }
  return { p: uniform, v: t.winner === 0 ? 0 : t.winner === me ? 1 : -1 };
};

// ---------- move selection ----------

export function pickMove(visits: number[], temp: number): number {
  if (temp <= 0.01) {
    let best = 0;
    for (let a = 1; a < visits.length; a++) if (visits[a] > visits[best]) best = a;
    return best;
  }
  const w = visits.map((v) => Math.pow(v, 1 / temp));
  const t = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * t;
  for (let a = 0; a < w.length; a++) {
    r -= w[a];
    if (r <= 0 && w[a] > 0) return a;
  }
  return visits.indexOf(Math.max(...visits));
}

// ---------- self-play ----------

export interface Samples {
  x: Float32Array;
  pi: Float32Array;
  z: Float32Array;
  n: number;
}

export function selfPlay(weights: Float32Array, games: number, sims: number, tempMoves = 10) {
  const ev = netEvaluator(weights);
  const xs: Float32Array[] = [];
  const pis: Float32Array[] = [];
  const zs: number[] = [];
  let plies = 0;
  let w1 = 0;
  let w2 = 0;
  let dr = 0;
  for (let g = 0; g < games; g++) {
    const s = newGame();
    const hist: { x: Float32Array; xm: Float32Array; pi: Float32Array; p: number }[] = [];
    while (!s.done) {
      const r = search(s, ev, { sims, noise: true, alpha: 1.0, eps: 0.25 });
      const tot = r.visits.reduce((a, b) => a + b, 0);
      const pi = new Float32Array(COLS);
      for (let a = 0; a < COLS; a++) pi[a] = r.visits[a] / tot;
      const x = new Float32Array(INPUT);
      const xm = new Float32Array(INPUT);
      encode(s, x);
      encode(s, xm, 0, true);
      hist.push({ x, xm, pi, p: s.p });
      play(s, pickMove(r.visits, s.n < tempMoves ? 1 : 0));
    }
    plies += s.n;
    if (s.winner === 1) w1++;
    else if (s.winner === -1) w2++;
    else dr++;
    for (const h of hist) {
      const z = s.winner === 0 ? 0 : s.winner === h.p ? 1 : -1;
      xs.push(h.x, h.xm);
      const pim = new Float32Array(COLS);
      for (let a = 0; a < COLS; a++) pim[a] = h.pi[COLS - 1 - a];
      pis.push(h.pi, pim);
      zs.push(z, z);
    }
  }
  const n = xs.length;
  const X = new Float32Array(n * INPUT);
  const PI = new Float32Array(n * COLS);
  xs.forEach((x, i) => X.set(x, i * INPUT));
  pis.forEach((p, i) => PI.set(p, i * COLS));
  return { samples: { x: X, pi: PI, z: Float32Array.from(zs), n } as Samples, plies, w1, w2, dr };
}

function newGame(): State {
  return {
    b: new Int8Array(CELLS),
    h: new Int8Array(COLS),
    p: 1,
    n: 0,
    winner: 0,
    done: false,
    last: -1,
  };
}

// ---------- evaluation matches ----------

export type AgentSpec = Float32Array | "rollout" | "random";

function agentMove(spec: AgentSpec, ev: Evaluator | null, s: State, sims: number, temp: number): number {
  if (spec === "random") {
    const l = legal(s);
    return l[(Math.random() * l.length) | 0];
  }
  const r = search(s, ev!, { sims });
  return pickMove(r.visits, temp);
}

/** Plays `games` games alternating colors. Returns score of A (win=1, draw=.5). */
export function match(a: AgentSpec, b: AgentSpec, games: number, simsA: number, simsB: number, startParity = 0) {
  const evA = a instanceof Float32Array ? netEvaluator(a) : a === "rollout" ? rolloutEvaluator : null;
  const evB = b instanceof Float32Array ? netEvaluator(b) : b === "rollout" ? rolloutEvaluator : null;
  let w = 0;
  let d = 0;
  let l = 0;
  for (let g = 0; g < games; g++) {
    const aSide = (g + startParity) % 2 === 0 ? 1 : -1;
    const s = newGame();
    while (!s.done) {
      // light randomness in the opening so games differ
      const temp = s.n < 4 ? 1 : 0;
      const mv = s.p === aSide ? agentMove(a, evA, s, simsA, temp) : agentMove(b, evB, s, simsB, temp);
      play(s, mv);
    }
    if (s.winner === 0) d++;
    else if (s.winner === aSide) w++;
    else l++;
  }
  return { w, d, l };
}
