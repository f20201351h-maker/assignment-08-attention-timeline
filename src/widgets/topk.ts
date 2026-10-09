import { s, svgRoot, frame, slider, seg } from '../lib/dom.ts';
import { softmax, rng } from '../lib/math.ts';

/** Top-k: keep the k largest scores, renormalise, compare with full softmax. */
export function topkWidget() {
  const f = frame('Keep the best k keys — and notice what selection still costs',
    'Twelve keys, one query. Scores are toy numbers. Top-k masks every score outside the k largest to −∞ before softmax, so only k values are mixed. The bottom bars separate the two costs: mixing values falls with k, but naïve selection still had to score all twelve keys to know which were best.');
  let k = 3;
  let shape: 'peaked' | 'spread' = 'peaked';
  f.controls.append(
    slider({ label: 'k', min: 1, max: 12, value: k, onInput: (v) => { k = v; draw(); } }),
    seg<'peaked' | 'spread'>({ label: 'This query needs', value: shape, options: [['peaked', 'one fact'], ['spread', 'many clues']], onChange: (v) => { shape = v; draw(); } }),
  );
  const svg = svgRoot(760, 300, 'Top-k selection over twelve attention scores');
  f.stage.append(svg);

  const r = rng(7);
  const vals = Array.from({ length: 12 }, () => r() * 2 - 1); // 1-d values
  const peaked = [0.2, 3.1, -0.4, 0.5, 2.0, -1.0, 0.1, 0.4, 1.2, -0.3, 0.0, 0.6];
  const spread = [0.9, 1.1, 0.7, 1.0, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0, 0.9];

  function draw() {
    svg.replaceChildren();
    const sc = shape === 'peaked' ? peaked : spread;
    const order = sc.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).map((p) => p[1]);
    const keep = new Set(order.slice(0, k));
    const full = softmax(sc);
    const sparse = softmax(sc.map((x, i) => (keep.has(i) ? x : -Infinity)));
    const bw = 44, x0 = 40, base = 170;
    svg.append(s('text', { x: x0, y: 20, class: 'lbl strong' }, 'Attention weight per key: full softmax (outline) vs top-k (filled)'));
    for (let i = 0; i < 12; i++) {
      const x = x0 + i * (bw + 12);
      const hf = full[i] * 220, hs = sparse[i] * 220;
      svg.append(s('rect', { x, y: base - hf, width: bw, height: Math.max(0.5, hf), class: 'fill-card stroke-muted', 'stroke-width': 1.2, 'stroke-dasharray': '3 2' }));
      if (keep.has(i)) svg.append(s('rect', { x: x + 6, y: base - hs, width: bw - 12, height: Math.max(0.5, hs), rx: 3, class: 'bar' }));
      svg.append(s('text', { x: x + bw / 2, y: base + 16, class: 'lbl-c num' + (keep.has(i) ? '' : ' muted') }, `k${i + 1}`));
    }
    svg.append(s('line', { x1: x0 - 6, x2: x0 + 12 * (bw + 12), y1: base, y2: base, class: 'axis' }));
    const yFull = full.reduce((a, w, i) => a + w * vals[i], 0);
    const ySparse = sparse.reduce((a, w, i) => a + w * vals[i], 0);
    const dropped = order.slice(k).reduce((a, i) => a + full[i], 0);
    // cost bars
    const cy = 215;
    svg.append(s('text', { x: x0, y: cy, class: 'lbl small' }, 'keys scored'));
    svg.append(s('rect', { x: x0 + 90, y: cy - 12, width: 12 * 40, height: 14, rx: 3, class: 'bar compute' }));
    svg.append(s('text', { x: x0 + 96 + 480, y: cy, class: 'lbl small num' }, '12 (unchanged)'));
    svg.append(s('text', { x: x0, y: cy + 26, class: 'lbl small' }, 'values mixed'));
    svg.append(s('rect', { x: x0 + 90, y: cy + 14, width: k * 40, height: 14, rx: 3, class: 'bar memory' }));
    svg.append(s('text', { x: x0 + 96 + k * 40, y: cy + 26, class: 'lbl small num' }, String(k)));
    svg.append(s('text', { x: x0, y: cy + 56, class: 'lbl small muted' }, 'Real systems avoid scoring everything with a cheaper proposal step: a window, a router, or a light indexer (DeepSeek, 2025).'));
    f.readout.innerHTML = `Top-${k} throws away <b>${(dropped * 100).toFixed(0)}%</b> of the attention mass full softmax would have used. Output ${ySparse.toFixed(3)} vs exact ${yFull.toFixed(3)} (error ${Math.abs(ySparse - yFull).toFixed(3)}). ` +
      (shape === 'peaked' ? 'When one key dominates, small k is nearly free.' : 'When evidence is spread out, small k changes the answer.');
  }
  draw();
  return f.root;
}
