// Connect Four, 6 rows x 7 cols. Index = r*COLS + c, r=0 is the bottom row.
export const ROWS = 6;
export const COLS = 7;
export const CELLS = ROWS * COLS;
export const INPUT = CELLS * 2;

export type Player = 1 | -1;

export interface State {
  b: Int8Array; // cells: 0 empty, 1 first player, -1 second player
  h: Int8Array; // column heights
  p: Player; // side to move
  n: number; // plies played
  winner: number; // 0 none / draw, 1 or -1
  done: boolean;
  last: number; // last placed cell index
}

export function newState(): State {
  return { b: new Int8Array(CELLS), h: new Int8Array(COLS), p: 1, n: 0, winner: 0, done: false, last: -1 };
}

export function clone(s: State): State {
  return { b: s.b.slice(), h: s.h.slice(), p: s.p, n: s.n, winner: s.winner, done: s.done, last: s.last };
}

export function legal(s: State): number[] {
  const r: number[] = [];
  if (s.done) return r;
  for (let c = 0; c < COLS; c++) if (s.h[c] < ROWS) r.push(c);
  return r;
}

const DIRS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

function lineThrough(b: Int8Array, r: number, c: number, p: number): number[] | null {
  for (const [dr, dc] of DIRS) {
    const cells = [r * COLS + c];
    for (const sgn of [1, -1]) {
      let rr = r + dr * sgn;
      let cc = c + dc * sgn;
      while (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && b[rr * COLS + cc] === p) {
        cells.push(rr * COLS + cc);
        rr += dr * sgn;
        cc += dc * sgn;
      }
    }
    if (cells.length >= 4) return cells;
  }
  return null;
}

function wins(b: Int8Array, r: number, c: number, p: number): boolean {
  for (const [dr, dc] of DIRS) {
    let k = 1;
    for (const sgn of [1, -1]) {
      let rr = r + dr * sgn;
      let cc = c + dc * sgn;
      while (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && b[rr * COLS + cc] === p) {
        k++;
        rr += dr * sgn;
        cc += dc * sgn;
      }
    }
    if (k >= 4) return true;
  }
  return false;
}

/** Mutating move. */
export function play(s: State, c: number): void {
  const r = s.h[c];
  const i = r * COLS + c;
  s.b[i] = s.p;
  s.h[c] = r + 1;
  s.n++;
  s.last = i;
  if (wins(s.b, r, c, s.p)) {
    s.winner = s.p;
    s.done = true;
  } else if (s.n === CELLS) {
    s.done = true;
  }
  s.p = -s.p as Player;
}

export function winLine(s: State): number[] | null {
  if (!s.winner || s.last < 0) return null;
  const r = Math.floor(s.last / COLS);
  const c = s.last % COLS;
  return lineThrough(s.b, r, c, s.winner);
}

/** Canonical encoding from the perspective of the side to move: [own stones | opponent stones]. */
export function encode(s: State, out: Float32Array, off = 0, mirror = false): void {
  for (let i = 0; i < INPUT; i++) out[off + i] = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const v = s.b[r * COLS + c];
      if (!v) continue;
      const j = r * COLS + (mirror ? COLS - 1 - c : c);
      out[off + (v === s.p ? j : CELLS + j)] = 1;
    }
  }
}
