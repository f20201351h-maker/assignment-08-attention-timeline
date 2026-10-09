import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  softmax, attendRow, rope, ropeFreqs, dot, linearDirect, linearState, matVec, writeAdd, writeDelta, zeros,
  kvCacheBytes, alibiSlopes, scaledFreqs, yarnLogitScale,
} from '../src/lib/math.ts';

const close = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('softmax sums to one and masks -Infinity to exactly zero', () => {
  const w = softmax([1, 2, 3, -Infinity]);
  close(w.reduce((a, b) => a + b, 0), 1);
  assert.equal(w[3], 0);
  // worked example: [1,2,3] -> about [0.09, 0.24, 0.67]
  assert.deepEqual(w.slice(0, 3).map((x) => +x.toFixed(2)), [0.09, 0.24, 0.67]);
});

test('causal row never puts weight on the future', () => {
  const keys = [[1, 0], [0, 1], [1, 1], [2, 2]];
  const { weights } = attendRow([1, 1], keys, 1);
  assert.equal(weights[2], 0);
  assert.equal(weights[3], 0);
});

test('RoPE score depends only on relative offset', () => {
  const f = ropeFreqs(8);
  const q = [0.3, -1.2, 0.8, 0.1, -0.5, 0.9, 1.1, -0.2];
  const k = [1.0, 0.4, -0.7, 0.6, 0.2, -1.1, 0.5, 0.3];
  const s1 = dot(rope(q, 8, f), rope(k, 2, f));
  const s2 = dot(rope(q, 18, f), rope(k, 12, f));
  const s3 = dot(rope(q, 1008, f), rope(k, 1002, f));
  close(s1, s2, 1e-9);
  close(s1, s3, 1e-7);
});

test('linear attention: direct route equals running-state route (worked example)', () => {
  // scalar example from the notes: q=2, k=[0.5,1,1.5], v=[10,20,30] -> 140
  const y = linearDirect([2], [[0.5], [1], [1.5]], [[10], [20], [30]]);
  close(y[0], 140);
  const S = linearState([[0.5], [1], [1.5]], [[10], [20], [30]]);
  close(S[0][0], 70);
  close(matVec(S, [2])[0], 140);
  // vector case
  const ks = [[1, 0, 2], [0.5, -1, 0], [2, 1, 1]];
  const vs = [[1, 2], [3, -1], [0, 4]];
  const q = [0.2, 0.7, -0.3];
  const a = linearDirect(q, ks, vs);
  const b = matVec(linearState(ks, vs), q);
  a.forEach((x, i) => close(x, b[i]));
});

test('add-only write gives 95, delta write gives 55 (worked example)', () => {
  const k = [1, 0];
  let S = writeAdd(zeros(1, 2), k, [40]);
  close(matVec(writeAdd(S, k, [55]), k)[0], 95);
  close(matVec(writeDelta(S, k, [55]), k)[0], 55);
  // gated: alpha = 0 wipes the state before writing
  S = writeAdd(S, [0, 1], [7]);
  close(matVec(writeDelta(S, k, [55], 1, 0), [0, 1])[0], 0);
});

test('KV cache formula reproduces the 6.44 GB yardstick', () => {
  const one = kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, tokens: 32768, batch: 1, bytes: 2 });
  close(one / 1e9, 6.442450944);
  close(kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, tokens: 32768, batch: 8, bytes: 2 }) / 1e9, 51.539607552);
});

test('ALiBi slopes for 8 heads are 1/2 ... 1/256', () => {
  assert.deepEqual(alibiSlopes(8), [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]);
});

test('context-extension scalings behave as described', () => {
  const d = 64, base = 10000, s = 4, L = 4096;
  const f = ropeFreqs(d, base);
  const pi = scaledFreqs(d, base, s, 'pi', L);
  const ntk = scaledFreqs(d, base, s, 'ntk', L);
  const yarn = scaledFreqs(d, base, s, 'yarn', L);
  close(pi[5], f[5] / s);
  close(ntk[0], f[0]); // NTK leaves the fastest pair untouched
  close(ntk[d / 2 - 1], f[d / 2 - 1] / s, 1e-12); // ...and fully interpolates the slowest
  close(yarn[0], f[0]); // high-frequency pairs are not interpolated
  close(yarn[d / 2 - 1], f[d / 2 - 1] / s); // low-frequency pairs are
  close(yarnLogitScale(1), 1);
  assert.ok(yarnLogitScale(16) > 1);
});
