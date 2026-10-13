// Tiny dependency-free policy/value MLP with manual backprop + AdamW.
//   x(84) -> 128 relu -> 128 relu -> { policy logits(7), value tanh(1) }
import { COLS, INPUT } from "./game";

export const I = INPUT;
export const H = 128;
export const P = COLS;

const oW1 = 0;
const oB1 = oW1 + H * I;
const oW2 = oB1 + H;
const oB2 = oW2 + H * H;
const oWp = oB2 + H;
const oBp = oWp + P * H;
const oWv = oBp + P;
const oBv = oWv + H;
export const NPARAMS = oBv + 1;

function randn(): number {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

export function initParams(): Float32Array {
  const w = new Float32Array(NPARAMS);
  const he = (o: number, n: number, fanIn: number, gain = 1) => {
    const s = gain * Math.sqrt(2 / fanIn);
    for (let i = 0; i < n; i++) w[o + i] = randn() * s;
  };
  he(oW1, H * I, 42); // ~42 active inputs at most
  he(oW2, H * H, H);
  he(oWp, P * H, H, 0.1);
  he(oWv, H, H, 0.1);
  return w;
}

export class Net {
  w: Float32Array;
  a1 = new Float32Array(H);
  a2 = new Float32Array(H);
  logits = new Float32Array(P);
  nz = new Int32Array(I);
  nnz = 0;

  constructor(w?: Float32Array) {
    this.w = w ? new Float32Array(w) : initParams();
  }

  /** Forward pass. Leaves activations in scratch buffers. Returns value; probs written to `probs`. */
  forward(x: Float32Array, xo: number, probs: Float32Array): number {
    const { w, a1, a2, logits, nz } = this;
    let nnz = 0;
    for (let i = 0; i < I; i++) if (x[xo + i] !== 0) nz[nnz++] = i;
    this.nnz = nnz;
    for (let j = 0; j < H; j++) {
      let s = w[oB1 + j];
      const row = oW1 + j * I;
      for (let k = 0; k < nnz; k++) s += w[row + nz[k]];
      a1[j] = s > 0 ? s : 0;
    }
    for (let j = 0; j < H; j++) {
      let s = w[oB2 + j];
      const row = oW2 + j * H;
      for (let i = 0; i < H; i++) s += w[row + i] * a1[i];
      a2[j] = s > 0 ? s : 0;
    }
    let mx = -Infinity;
    for (let k = 0; k < P; k++) {
      let s = w[oBp + k];
      const row = oWp + k * H;
      for (let i = 0; i < H; i++) s += w[row + i] * a2[i];
      logits[k] = s;
      if (s > mx) mx = s;
    }
    let sum = 0;
    for (let k = 0; k < P; k++) {
      const e = Math.exp(logits[k] - mx);
      probs[k] = e;
      sum += e;
    }
    for (let k = 0; k < P; k++) probs[k] /= sum;
    let v = w[oBv];
    for (let i = 0; i < H; i++) v += w[oWv + i] * a2[i];
    return Math.tanh(v);
  }
}

export class Trainer {
  net: Net;
  g = new Float32Array(NPARAMS);
  m = new Float32Array(NPARAMS);
  v = new Float32Array(NPARAMS);
  t = 0;
  private probs = new Float32Array(P);
  private da1 = new Float32Array(H);
  private dz2 = new Float32Array(H);

  constructor(w?: Float32Array) {
    this.net = new Net(w);
  }

  /** One minibatch SGD step. Returns mean policy CE and value MSE. */
  step(X: Float32Array, PI: Float32Array, Z: Float32Array, idx: ArrayLike<number>, lr: number, wd: number) {
    const { net, g, probs, da1, dz2 } = this;
    const w = net.w;
    g.fill(0);
    let lp = 0;
    let lv = 0;
    const B = idx.length;
    const dl = new Float32Array(P);
    for (let b = 0; b < B; b++) {
      const s = idx[b];
      const v = net.forward(X, s * I, probs);
      const { a1, a2, nz, nnz } = net;
      const z = Z[s];
      for (let k = 0; k < P; k++) {
        const pi = PI[s * P + k];
        if (pi > 0) lp -= pi * Math.log(probs[k] + 1e-9);
        dl[k] = probs[k] - pi;
      }
      lv += (v - z) * (v - z);
      const dv = 2 * (v - z) * (1 - v * v);
      // heads
      for (let k = 0; k < P; k++) {
        const d = dl[k];
        const row = oWp + k * H;
        for (let j = 0; j < H; j++) g[row + j] += d * a2[j];
        g[oBp + k] += d;
      }
      for (let j = 0; j < H; j++) g[oWv + j] += dv * a2[j];
      g[oBv] += dv;
      for (let j = 0; j < H; j++) {
        if (a2[j] <= 0) {
          dz2[j] = 0;
          continue;
        }
        let d = w[oWv + j] * dv;
        for (let k = 0; k < P; k++) d += w[oWp + k * H + j] * dl[k];
        dz2[j] = d;
      }
      // layer 2
      da1.fill(0);
      for (let j = 0; j < H; j++) {
        const d = dz2[j];
        if (d === 0) continue;
        const row = oW2 + j * H;
        for (let i = 0; i < H; i++) {
          g[row + i] += d * a1[i];
          da1[i] += w[row + i] * d;
        }
        g[oB2 + j] += d;
      }
      // layer 1 (sparse input)
      for (let j = 0; j < H; j++) {
        if (a1[j] <= 0) continue;
        const d = da1[j];
        const row = oW1 + j * I;
        for (let k = 0; k < nnz; k++) g[row + nz[k]] += d;
        g[oB1 + j] += d;
      }
    }
    // AdamW
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);
    const { m, v } = this;
    const inv = 1 / B;
    for (let i = 0; i < NPARAMS; i++) {
      const gi = g[i] * inv;
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      w[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8) + wd * w[i]);
    }
    return { lp: lp / B, lv: lv / B };
  }
}
