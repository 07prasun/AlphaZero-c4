import SelfPlayWorker from "./selfplay.worker.ts?worker&inline";
import TrainerWorker from "./trainer.worker.ts?worker&inline";
import { initParams } from "./nn";
import type { AgentSpec, Samples, SearchResult } from "./mcts";
import type { State } from "./game";

// ---------- worker plumbing ----------

class Handle {
  w: Worker;
  busy = false;
  retire = false;
  private seq = 0;
  private pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
  constructor(w: Worker) {
    this.w = w;
    w.onmessage = (e) => {
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      if (e.data.ok) p.res(e.data.result);
      else p.rej(new Error(e.data.error));
    };
  }
  call<T = any>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.seq;
    return new Promise<T>((res, rej) => {
      this.pending.set(id, { res, rej });
      this.w.postMessage({ ...msg, id }, transfer);
    });
  }
  terminate() {
    this.w.terminate();
  }
}

class Pool {
  hs: Handle[] = [];
  private q: { msg: Record<string, unknown>; res: (v: any) => void; rej: (e: any) => void }[] = [];
  constructor(n: number) {
    this.resize(n);
  }
  resize(n: number) {
    while (this.hs.length < n) this.hs.push(new Handle(new SelfPlayWorker()));
    while (this.hs.length > n) {
      const h = this.hs.pop()!;
      if (h.busy) h.retire = true; // let in-flight job finish, then terminate
      else h.terminate();
    }
  }
  get size() {
    return this.hs.length;
  }
  run<T = any>(msg: Record<string, unknown>): Promise<T> {
    return new Promise<T>((res, rej) => {
      this.q.push({ msg, res, rej });
      this.pump();
    });
  }
  private pump() {
    for (const h of this.hs) {
      if (h.busy || !this.q.length) continue;
      const job = this.q.shift()!;
      h.busy = true;
      h.call(job.msg)
        .then(job.res, job.rej)
        .finally(() => {
          h.busy = false;
          if (h.retire) h.terminate();
          this.pump();
        });
    }
  }
  busyCount() {
    return this.hs.filter((h) => h.busy).length;
  }
}

// ---------- Elo ----------

const expected = (r: number, o: number) => 1 / (1 + Math.pow(10, (o - r) / 400));

/** MLE rating for a new player vs fixed-rated opponents, with a 1-draw prior vs `prior` to keep it finite. */
export function fitElo(results: { opp: number; w: number; d: number; l: number }[], prior: number): number {
  const f = (R: number) => {
    let s = 0.5 - expected(R, prior);
    for (const g of results) s += g.w + 0.5 * g.d - (g.w + g.d + g.l) * expected(R, g.opp);
    return s;
  };
  let lo = -3000;
  let hi = 8000;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) > 0) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---------- coordinator ----------

export interface Config {
  workers: number;
  sims: number;
  gamesPerGen: number;
  maxSteps: number;
  batch: number;
  lr: number;
  evalGames: number;
  evalSims: number;
  refSims: number;
}

export interface GenRecord {
  gen: number;
  elo: number;
  vsPrev: number; // score vs previous gen [0,1]
  vsRef: number; // score vs pure-MCTS reference [0,1]
  vsRandom: number;
  lp: number;
  lv: number;
  buffer: number;
  games: number; // cumulative self-play games
  plies: number; // avg game length this gen
  firstWin: number; // first-player win rate in self-play
  secs: number; // wall time of generation
}

export interface Snapshot {
  running: boolean;
  stopping: boolean;
  phase: string;
  gen: number;
  history: GenRecord[];
  log: string[];
  config: Config;
  gens: number[]; // available checkpoints
  totalGames: number;
  totalPositions: number;
  posPerSec: number;
}

const LS_KEY = "az4.v1";

function f32ToB64(a: Float32Array): string {
  const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}
function b64ToF32(b: string): Float32Array {
  const s = atob(b);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}

const hw = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;

export class Coordinator {
  pool: Pool;
  trainer: Handle;
  player: Handle; // dedicated worker for interactive play, stays responsive while training
  checkpoints = new Map<number, Float32Array>();
  current: Float32Array;
  snap: Snapshot;
  private listeners = new Set<() => void>();

  constructor() {
    const config: Config = {
      workers: Math.max(1, Math.min(8, hw - 2)),
      sims: 80,
      gamesPerGen: 48,
      maxSteps: 300,
      batch: 128,
      lr: 1e-3,
      evalGames: 20,
      evalSims: 64,
      refSims: 200,
    };
    this.pool = new Pool(config.workers);
    this.trainer = new Handle(new TrainerWorker());
    this.player = new Handle(new SelfPlayWorker());
    this.current = initParams();
    this.snap = {
      running: false,
      stopping: false,
      phase: "idle",
      gen: 0,
      history: [],
      log: [],
      config,
      gens: [0],
      totalGames: 0,
      totalPositions: 0,
      posPerSec: 0,
    };
    if (!this.restore()) {
      this.checkpoints.set(0, this.current);
      this.snap.history = [
        { gen: 0, elo: 0, vsPrev: 0.5, vsRef: NaN, vsRandom: NaN, lp: NaN, lv: NaN, buffer: 0, games: 0, plies: NaN, firstWin: NaN, secs: 0 },
      ];
      this.log("gen 0: random init, Elo anchored at 0");
    }
    this.trainer.call({ type: "init", weights: this.current });
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  getSnapshot = () => this.snap;

  private set(p: Partial<Snapshot>) {
    this.snap = { ...this.snap, ...p };
    this.listeners.forEach((f) => f());
  }
  private log(s: string) {
    const t = new Date().toLocaleTimeString([], { hour12: false });
    this.set({ log: [`${t}  ${s}`, ...this.snap.log].slice(0, 80) });
  }

  setConfig(p: Partial<Config>) {
    const config = { ...this.snap.config, ...p };
    if (p.workers) this.pool.resize(p.workers);
    this.set({ config });
  }

  weightsFor(gen: number): Float32Array | undefined {
    return this.checkpoints.get(gen);
  }

  analyze(weights: Float32Array, state: State, sims: number): Promise<SearchResult & { ms: number }> {
    return this.player.call({ type: "analyze", weights, state, sims });
  }

  start() {
    if (this.snap.running) return;
    this.set({ running: true, stopping: false });
    this.loop().catch((e) => {
      this.log("error: " + e.message);
      this.set({ running: false, stopping: false, phase: "idle" });
    });
  }
  stop() {
    if (this.snap.running) this.set({ stopping: true });
  }

  async reset() {
    this.stop();
    while (this.snap.running) await new Promise((r) => setTimeout(r, 100));
    localStorage.removeItem(LS_KEY);
    this.checkpoints.clear();
    this.current = initParams();
    this.checkpoints.set(0, this.current);
    await this.trainer.call({ type: "init", weights: this.current });
    this.set({
      gen: 0,
      gens: [0],
      totalGames: 0,
      totalPositions: 0,
      posPerSec: 0,
      log: [],
      history: [
        { gen: 0, elo: 0, vsPrev: 0.5, vsRef: NaN, vsRandom: NaN, lp: NaN, lv: NaN, buffer: 0, games: 0, plies: NaN, firstWin: NaN, secs: 0 },
      ],
    });
    this.log("reset: gen 0, random init");
  }

  private async loop() {
    while (!this.snap.stopping) {
      await this.generation();
    }
    this.set({ running: false, stopping: false, phase: "idle" });
    this.log("paused");
  }

  private async generation() {
    const cfg = this.snap.config;
    const t0 = performance.now();
    const gen = this.snap.gen + 1;
    const W = this.current;

    // 1. distributed self-play
    this.set({ phase: `self-play · ${cfg.gamesPerGen} games × ${cfg.sims} sims on ${this.pool.size} workers` });
    const chunk = Math.max(1, Math.ceil(cfg.gamesPerGen / (this.pool.size * 3)));
    const jobs: Promise<any>[] = [];
    for (let g = 0; g < cfg.gamesPerGen; g += chunk) {
      const n = Math.min(chunk, cfg.gamesPerGen - g);
      jobs.push(
        this.pool.run({ type: "selfplay", weights: W, games: n, sims: cfg.sims }).then(async (r) => {
          const s: Samples = r.samples;
          await this.trainer.call({ type: "add", ...s }, [s.x.buffer, s.pi.buffer, s.z.buffer]);
          return r;
        }),
      );
    }
    const res = await Promise.all(jobs);
    const plies = res.reduce((a, r) => a + r.plies, 0);
    const w1 = res.reduce((a, r) => a + r.w1, 0);
    const newSamples = plies * 2;
    const spSecs = (performance.now() - t0) / 1000;
    const positions = plies * cfg.sims;

    // 2. train
    const steps = Math.max(20, Math.min(cfg.maxSteps, Math.round((newSamples * 6) / cfg.batch)));
    this.set({ phase: `training · ${steps} steps × batch ${cfg.batch}` });
    const tr = await this.trainer.call<{ weights: Float32Array; lp: number; lv: number; size: number }>({
      type: "train",
      steps,
      batch: cfg.batch,
      lr: cfg.lr,
      wd: 1e-4,
    });
    const NW = tr.weights;

    // 3. evaluate: vs previous gen, vs older gen, vs pure-MCTS reference, vs random
    this.set({ phase: `evaluating gen ${gen}` });
    const hist = this.snap.history;
    const prev = hist[hist.length - 1];
    const older = hist.length >= 5 ? hist[hist.length - 5] : null;
    const olderW = older ? this.checkpoints.get(older.gen) : undefined;

    const split = async (a: AgentSpec, b: AgentSpec, games: number, simsB: number) => {
      const parts = Math.min(games, this.pool.size * 2);
      const per = Math.ceil(games / parts);
      const ps: Promise<{ w: number; d: number; l: number }>[] = [];
      for (let g = 0; g < games; g += per) {
        const n = Math.min(per, games - g);
        ps.push(this.pool.run({ type: "match", a, b, games: n, simsA: cfg.evalSims, simsB, parity: g % 2 }));
      }
      const rs = await Promise.all(ps);
      return rs.reduce((x, r) => ({ w: x.w + r.w, d: x.d + r.d, l: x.l + r.l }), { w: 0, d: 0, l: 0 });
    };
    const [rPrev, rOld, rRef, rRand] = await Promise.all([
      split(NW, W, cfg.evalGames, cfg.evalSims),
      olderW ? split(NW, olderW, Math.round(cfg.evalGames / 2), cfg.evalSims) : Promise.resolve(null),
      split(NW, "rollout", 10, cfg.refSims),
      split(NW, "random", 10, 0),
    ]);
    const score = (r: { w: number; d: number; l: number }) => (r.w + 0.5 * r.d) / (r.w + r.d + r.l);
    const results = [{ opp: prev.elo, ...rPrev }];
    if (rOld && older) results.push({ opp: older.elo, ...rOld });
    const elo = fitElo(results, prev.elo);

    this.current = NW;
    this.checkpoints.set(gen, NW);
    const secs = (performance.now() - t0) / 1000;
    const rec: GenRecord = {
      gen,
      elo,
      vsPrev: score(rPrev),
      vsRef: score(rRef),
      vsRandom: score(rRand),
      lp: tr.lp,
      lv: tr.lv,
      buffer: tr.size,
      games: this.snap.totalGames + cfg.gamesPerGen,
      plies: plies / cfg.gamesPerGen,
      firstWin: w1 / cfg.gamesPerGen,
      secs,
    };
    this.set({
      gen,
      history: [...hist, rec],
      gens: [...this.checkpoints.keys()],
      totalGames: rec.games,
      totalPositions: this.snap.totalPositions + positions,
      posPerSec: positions / spSecs,
    });
    this.log(
      `gen ${gen}: Elo ${elo.toFixed(0)} · vs prev ${rPrev.w}-${rPrev.d}-${rPrev.l}` +
        (rOld ? ` · vs g${older!.gen} ${rOld.w}-${rOld.d}-${rOld.l}` : "") +
        ` · vs mcts${cfg.refSims} ${rRef.w}-${rRef.d}-${rRef.l} · loss π ${tr.lp.toFixed(3)} v ${tr.lv.toFixed(3)} · ${secs.toFixed(1)}s`,
    );
    this.persist();
  }

  private persist() {
    try {
      localStorage.setItem(
        LS_KEY,
        JSON.stringify({
          gen: this.snap.gen,
          history: this.snap.history,
          totalGames: this.snap.totalGames,
          totalPositions: this.snap.totalPositions,
          weights: f32ToB64(this.current),
        }),
      );
    } catch {
      /* quota: ignore */
    }
  }

  private restore(): boolean {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) return false;
      const d = JSON.parse(raw);
      const w = b64ToF32(d.weights);
      this.current = w;
      this.checkpoints.set(d.gen, w);
      const history = (d.history as GenRecord[]).map((h) => {
        const o = { ...h } as any;
        for (const k of Object.keys(o)) if (o[k] === null) o[k] = NaN;
        return o as GenRecord;
      });
      this.snap = { ...this.snap, gen: d.gen, history, gens: [d.gen], totalGames: d.totalGames, totalPositions: d.totalPositions };
      this.snap.log = [`restored gen ${d.gen} from localStorage (replay buffer empty)`];
      return true;
    } catch {
      return false;
    }
  }

  exportModel(): string {
    return JSON.stringify({ format: "az4-mlp-84-128-128", gen: this.snap.gen, weights: f32ToB64(this.current) });
  }
}
