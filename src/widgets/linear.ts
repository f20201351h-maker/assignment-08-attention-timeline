import { s, svgRoot, frame, seg, slider, sweepButton, fmt } from '../lib/dom.ts';
import { dot, softmax, rng, linearState, matVec, mix, add } from '../lib/math.ts';

type Mode = 'raw' | 'phi' | 'softmax';
const elu1 = (x: number) => (x > 0 ? x + 1 : Math.exp(x)); // elu(x) + 1, always positive

/** Two routes to the same output: revisit every key, or read one running state. */
export function linearWidget() {
  const f = frame('Two routes to one answer — until softmax comes back',
    'Toy 3-number keys, values and query. The left route keeps every key/value and revisits them when the query arrives (that list is the KV cache). The right route folds each pair into one 3×3 state as it arrives. Without softmax both routes give identical outputs; with exact softmax there is no fixed-size state that reproduces the answer.');
  let mode: Mode = 'raw';
  let n = 6;
  f.controls.append(
    seg<Mode>({ label: 'Score', value: mode, options: [['raw', 'q·k (no softmax)'], ['phi', 'φ(q)·φ(k), normalised (2020)'], ['softmax', 'exact softmax']], onChange: (v) => { mode = v; draw(); } }),
  );
  const nS = slider({ label: 'Tokens so far', min: 1, max: 14, value: n, onInput: (v) => { n = v; draw(); } });
  f.controls.append(nS, sweepButton('▶ Stream', nS, 1, 14, (v) => { n = v; draw(); }, 380));
  const svg = svgRoot(760, 300, 'Growing key-value list versus fixed-size state');
  f.stage.append(svg);
  const r = rng(11);
  const K = Array.from({ length: 14 }, () => [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1]);
  const V = Array.from({ length: 14 }, () => [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1]);
  const q = [0.9, -0.4, 0.6];

  function draw() {
    svg.replaceChildren();
    const ks = K.slice(0, n), vs = V.slice(0, n);
    let direct: number[], viaState: number[] | null, S: number[][] | null;
    if (mode === 'raw') {
      direct = mix(ks.map((k) => dot(q, k)), vs);
      S = linearState(ks, vs);
      viaState = matVec(S, q);
    } else if (mode === 'phi') {
      const pq = q.map(elu1), pk = ks.map((k) => k.map(elu1));
      const w = pk.map((k) => dot(pq, k));
      const z = w.reduce((a, b) => a + b, 0);
      direct = mix(w.map((x) => x / z), vs);
      S = linearState(pk, vs);
      const zs = pk.reduce((acc, k) => add(acc, k), [0, 0, 0]);
      viaState = matVec(S, pq).map((x) => x / dot(zs, pq));
    } else {
      direct = mix(softmax(ks.map((k) => dot(q, k) / Math.sqrt(3))), vs);
      S = null; viaState = null;
    }
    // left: list
    svg.append(s('text', { x: 30, y: 22, class: 'lbl strong' }, 'Route A: keep every (k, v)'));
    ks.forEach((_, j) => {
      const y = 34 + j * 17;
      svg.append(s('rect', { x: 30, y, width: 70, height: 13, rx: 2, class: 'fill-exact', style: 'fill-opacity:.55' }));
      svg.append(s('rect', { x: 104, y, width: 70, height: 13, rx: 2, class: 'fill-memory', style: 'fill-opacity:.55' }));
      svg.append(s('text', { x: 65, y: y + 10, class: 'cell-t small' }, `k${j + 1}`));
      svg.append(s('text', { x: 139, y: y + 10, class: 'cell-t small' }, `v${j + 1}`));
    });
    svg.append(s('text', { x: 190, y: 44, class: 'lbl small muted' }, `${n} pairs × 6 numbers`));
    svg.append(s('text', { x: 190, y: 60, class: 'lbl small muted' }, 'grows with every token'));
    // right: state
    const sx = 330;
    svg.append(s('text', { x: sx, y: 22, class: 'lbl strong' }, 'Route B: one running state S = Σ v kᵀ'));
    if (S) {
      const mx = Math.max(...S.flat().map(Math.abs), 1e-6);
      S.forEach((row, i) => row.forEach((v, j) => {
        svg.append(s('rect', { x: sx + j * 46, y: 36 + i * 46, width: 42, height: 42, rx: 4, class: v >= 0 ? 'cell' : 'fill-exact', style: `fill-opacity:${0.15 + 0.8 * Math.abs(v) / mx}` }));
        svg.append(s('text', { x: sx + j * 46 + 21, y: 36 + i * 46 + 25, class: 'cell-t small' }, fmt(v, 1)));
      }));
      svg.append(s('text', { x: sx + 150, y: 60, class: 'lbl small muted' }, '3 × 3 = 9 numbers'));
      svg.append(s('text', { x: sx + 150, y: 76, class: 'lbl small muted' }, 'after 1 token or 1 million'));
    } else {
      svg.append(s('rect', { x: sx, y: 36, width: 134, height: 134, rx: 6, class: 'cell-masked' }));
      svg.append(s('text', { x: sx + 67, y: 100, class: 'lbl-c' }, 'no fixed state'));
      svg.append(s('text', { x: sx + 67, y: 116, class: 'lbl-c small muted' }, 'exp(q·kⱼ) / Σ exp(q·k)'));
      svg.append(s('text', { x: sx + 150, y: 60, class: 'lbl small muted' }, 'the denominator depends on'));
      svg.append(s('text', { x: sx + 150, y: 76, class: 'lbl small muted' }, 'q, so keys must be revisited'));
    }
    // outputs
    const oy = 215;
    const row = (label: string, v: number[] | null, y: number) => {
      svg.append(s('text', { x: 330, y, class: 'lbl' }, label));
      svg.append(s('text', { x: 470, y, class: 'lbl num' }, v ? `[${v.map((x) => fmt(x, 4)).join(', ')}]` : '—'));
    };
    svg.append(s('text', { x: 330, y: oy - 22, class: 'lbl strong' }, 'Output for the current query'));
    row('Route A (revisit all):', direct, oy);
    row('Route B (read state):', viaState, oy + 22);
    const match = viaState && direct.every((x, i) => Math.abs(x - viaState![i]) < 1e-9);
    svg.append(s('text', { x: 330, y: oy + 52, class: 'lbl strong ' + (match ? 'text-good' : 'text-bad') }, match ? '✓ identical — the past was summarised before the query arrived' : '✗ softmax ties every weight to every other score'));
    f.readout.innerHTML = mode === 'softmax'
      ? 'Exact softmax needs the individual keys after the query arrives. That is why the KV cache exists — and why it grows.'
      : mode === 'phi'
        ? 'Katharopoulos et al. keep weights positive with φ(x) = elu(x) + 1 and divide by Σ φ(q)·φ(k): a weighted average again, computable from a fixed state. It is still not softmax — the weights are flatter, and every memory shares one 3×3 matrix.'
        : 'Raw scores: the distributive law lets the query be factored out. Cost per token is constant; the price is that old pairs are blended into one matrix and can interfere.';
  }
  draw();
  return f.root;
}
