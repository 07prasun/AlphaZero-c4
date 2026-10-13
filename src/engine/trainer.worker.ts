/// <reference lib="webworker" />
import { COLS, INPUT } from "./game";
import { Trainer } from "./nn";

const CAP = 60000;
const X = new Float32Array(CAP * INPUT);
const PI = new Float32Array(CAP * COLS);
const Z = new Float32Array(CAP);
let size = 0;
let head = 0;
let trainer = new Trainer();

type Msg =
  | { id: number; type: "init"; weights?: Float32Array }
  | { id: number; type: "add"; x: Float32Array; pi: Float32Array; z: Float32Array; n: number }
  | { id: number; type: "train"; steps: number; batch: number; lr: number; wd: number };

self.onmessage = (e: MessageEvent<Msg>) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      trainer = new Trainer(m.weights);
      size = 0;
      head = 0;
      self.postMessage({ id: m.id, ok: true, result: { weights: trainer.net.w.slice() } });
    } else if (m.type === "add") {
      for (let i = 0; i < m.n; i++) {
        X.set(m.x.subarray(i * INPUT, (i + 1) * INPUT), head * INPUT);
        PI.set(m.pi.subarray(i * COLS, (i + 1) * COLS), head * COLS);
        Z[head] = m.z[i];
        head = (head + 1) % CAP;
        if (size < CAP) size++;
      }
      self.postMessage({ id: m.id, ok: true, result: { size } });
    } else if (m.type === "train") {
      const t0 = performance.now();
      let lp = 0;
      let lv = 0;
      const idx = new Int32Array(m.batch);
      for (let s = 0; s < m.steps; s++) {
        for (let b = 0; b < m.batch; b++) idx[b] = (Math.random() * size) | 0;
        const r = trainer.step(X, PI, Z, idx, m.lr, m.wd);
        lp += r.lp;
        lv += r.lv;
      }
      self.postMessage({
        id: m.id,
        ok: true,
        result: {
          weights: trainer.net.w.slice(),
          lp: lp / m.steps,
          lv: lv / m.steps,
          size,
          ms: performance.now() - t0,
        },
      });
    }
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: String(err) });
  }
};
