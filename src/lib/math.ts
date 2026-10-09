// Small, dependency-free maths used by the widgets. Everything here is exact
// arithmetic on tiny toy tensors; tests/math.test.ts checks the identities the
// page claims (e.g. linear attention's two routes agree, RoPE is relative).

export type Vec = number[];
export type Mat = number[][];

export const dot = (a: Vec, b: Vec) => a.reduce((s, x, i) => s + x * b[i], 0);
export const add = (a: Vec, b: Vec) => a.map((x, i) => x + b[i]);
export const scale = (a: Vec, s: number) => a.map((x) => x * s);
export const norm = (a: Vec) => Math.sqrt(dot(a, a));
export const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array(c).fill(0));
export const matVec = (m: Mat, v: Vec): Vec => m.map((row) => dot(row, v));
export const outer = (a: Vec, b: Vec): Mat => a.map((x) => b.map((y) => x * y));
export const matAdd = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((x, j) => x + b[i][j]));
export const matScale = (a: Mat, s: number): Mat => a.map((r) => r.map((x) => x * s));

/** Softmax with temperature; -Infinity entries get exactly zero weight. */
export function softmax(x: Vec, temperature = 1): Vec {
  const t = Math.max(temperature, 1e-6);
  const m = Math.max(...x.filter((v) => v !== -Infinity));
  if (!isFinite(m)) return x.map(() => 0);
  const e = x.map((v) => (v === -Infinity ? 0 : Math.exp((v - m) / t)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Weighted sum of value vectors. */
export const mix = (w: Vec, values: Vec[]): Vec =>
  values.reduce((acc, v, j) => add(acc, scale(v, w[j])), new Array(values[0].length).fill(0));

/** Causal scaled dot-product attention for one query position i. */
export function attendRow(q: Vec, keys: Vec[], i: number, causal = true): { scores: Vec; weights: Vec } {
  const d = q.length;
  const scores = keys.map((k, j) => (causal && j > i ? -Infinity : dot(q, k) / Math.sqrt(d)));
  return { scores, weights: softmax(scores) };
}

// ---------- RoPE family ----------

/** Rotate a 2-vector by angle a. */
export const rot2 = ([x, y]: Vec, a: number): Vec => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

/** Per-pair rotation frequencies theta_i = base^(-2i/d), i = 0..d/2-1. */
export const ropeFreqs = (d: number, base = 10000): Vec => Array.from({ length: d / 2 }, (_, i) => Math.pow(base, (-2 * i) / d));

/** Apply RoPE to a d-vector at position p with the given per-pair frequencies. */
export function rope(v: Vec, p: number, freqs: Vec): Vec {
  const out: Vec = [];
  freqs.forEach((f, i) => out.push(...rot2([v[2 * i], v[2 * i + 1]], p * f)));
  return out;
}

export type Scaling = 'none' | 'pi' | 'ntk' | 'yarn';

/**
 * Effective per-pair frequencies after a context-extension method with factor s.
 * PI (Chen et al. 2023): divide every frequency by s.
 * NTK-aware (bloc97 2023): raise the base to base * s^(d/(d-2)).
 * YaRN "NTK-by-parts" (Peng et al. 2023): interpolate only pairs whose wavelength
 * is long relative to the training context; ramp between alpha and beta rotations.
 */
export function scaledFreqs(d: number, base: number, s: number, method: Scaling, trainLen: number, alpha = 1, beta = 32): Vec {
  const f = ropeFreqs(d, base);
  if (method === 'none' || s <= 1) return f;
  if (method === 'pi') return f.map((x) => x / s);
  if (method === 'ntk') return ropeFreqs(d, base * Math.pow(s, d / (d - 2)));
  return f.map((x) => {
    const r = trainLen / ((2 * Math.PI) / x); // full rotations inside the training window
    const g = r < alpha ? 0 : r > beta ? 1 : (r - alpha) / (beta - alpha);
    return (1 - g) * (x / s) + g * x;
  });
}

/** YaRN attention temperature: sqrt(1/t) = 0.1 ln(s) + 1, returned as the logit multiplier 1/t. */
export const yarnLogitScale = (s: number) => (s <= 1 ? 1 : Math.pow(0.1 * Math.log(s) + 1, 2));

// ---------- ALiBi ----------

/** ALiBi head slopes for n heads (n a power of two): 2^(-8i/n), i = 1..n. */
export const alibiSlopes = (n: number): Vec => Array.from({ length: n }, (_, i) => Math.pow(2, (-8 * (i + 1)) / n));

// ---------- KV cache ----------

export interface CacheCfg { layers: number; kvHeads: number; headDim: number; tokens: number; batch: number; bytes: number }
/** 2 (K and V) x layers x kv_heads x head_dim x T x batch x bytes. */
export const kvCacheBytes = (c: CacheCfg) => 2 * c.layers * c.kvHeads * c.headDim * c.tokens * c.batch * c.bytes;

export function fmtBytes(b: number): string {
  const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b < 10 ? b.toFixed(2) : b < 100 ? b.toFixed(1) : Math.round(b)} ${u[i]}`;
}

export function fmtCount(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(n >= 1e13 ? 0 : 1) + 'T';
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'K';
  return String(Math.round(n));
}

// ---------- linear attention / delta rule ----------

/** Unnormalised linear attention, direct route: y = sum_j (q.k_j) v_j. */
export const linearDirect = (q: Vec, keys: Vec[], values: Vec[]): Vec => mix(keys.map((k) => dot(q, k)), values);

/** Same thing via the running state S = sum_j v_j k_j^T, y = S q. */
export function linearState(keys: Vec[], values: Vec[]): Mat {
  return keys.reduce((S, k, j) => matAdd(S, outer(values[j], k)), zeros(values[0].length, keys[0].length));
}

/** Additive (Hebbian) write: S <- S + v k^T. */
export const writeAdd = (S: Mat, k: Vec, v: Vec): Mat => matAdd(S, outer(v, k));

/**
 * Delta-rule write (Schlag et al. 2021): read the old value S k, then write only
 * beta * (v - S k) along k. With unit k and beta = 1, S k becomes exactly v.
 * Gated DeltaNet (Yang et al. 2024) first decays the whole state by alpha.
 */
export function writeDelta(S: Mat, k: Vec, v: Vec, beta = 1, alpha = 1): Mat {
  const decayed = matScale(S, alpha);
  const old = matVec(decayed, k);
  const delta = scale(add(v, scale(old, -1)), beta);
  return matAdd(decayed, outer(delta, k));
}

/** Deterministic pseudo-random generator so every visitor sees the same toy numbers. */
export function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
