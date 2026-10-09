import { s, svgRoot, frame, seg, slider } from '../lib/dom.ts';

type Pattern = 'dense' | 'strided' | 'fixed' | 'window';

/** Which (query i, key j) pairs are computed under each causal pattern. */
function allowed(p: Pattern, i: number, j: number, w: number): boolean {
  if (j > i) return false;
  if (p === 'dense') return true;
  if (p === 'window') return i - j < w;
  if (p === 'strided') return i - j < w || (i - j) % w === 0; // head A: previous w; head B: every w-th
  // fixed: attend within own block, plus the last position ("summary") of every earlier block
  return Math.floor(i / w) === Math.floor(j / w) || j % w === w - 1;
}

/** Positions reachable from token i after L layers (information can hop). */
function reach(p: Pattern, i: number, w: number, layers: number, n: number): Set<number> {
  let set = new Set([i]);
  for (let l = 0; l < layers; l++) {
    const next = new Set(set);
    for (const q of set) for (let j = 0; j <= q; j++) if (allowed(p, q, j, w)) next.add(j);
    set = next;
  }
  return new Set([...set].filter((x) => x < n));
}

export function patternWidget(initial: Pattern = 'strided') {
  const f = frame('Which query–key pairs get computed?',
    'Each square is one score. Dense causal attention fills the whole lower triangle; sparse patterns keep a structured subset decided by <em>position</em>, not content. The orange tokens show what the last token can reach after stacking layers: information hops, so a narrow window still reaches far in a deep network — indirectly.');
  let p: Pattern = initial;
  let w = 4;
  let layers = 1;
  const N = 24;
  f.controls.append(
    seg<Pattern>({ label: 'Pattern', value: p, options: [['dense', 'Dense'], ['strided', 'Strided (2019)'], ['fixed', 'Fixed (2019)'], ['window', 'Sliding window']], onChange: (v) => { p = v; draw(); } }),
    slider({ label: 'Window / stride w', min: 2, max: 8, value: w, onInput: (v) => { w = v; draw(); } }),
    slider({ label: 'Layers stacked', min: 1, max: 6, value: layers, onInput: (v) => { layers = v; draw(); } }),
  );
  const svg = svgRoot(760, 330, 'Attention pattern matrix and receptive field');
  f.stage.append(svg);

  function draw() {
    svg.replaceChildren();
    const cs = 11.5, ox = 30, oy = 20;
    let count = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
      const on = allowed(p, i, j, w);
      if (on) count++;
      svg.append(s('rect', { x: ox + j * cs, y: oy + i * cs, width: cs - 1.5, height: cs - 1.5, rx: 1.5, class: on ? 'cell' : 'cell-off', style: on ? 'fill-opacity:.85' : '' }));
    }
    svg.append(s('text', { x: ox, y: oy + N * cs + 16, class: 'lbl small' }, 'keys →  (rows = queries, 24 tokens)'));
    const dense = (N * (N + 1)) / 2;
    // receptive field strip
    const rx = 340, ry = 40;
    svg.append(s('text', { x: rx, y: 24, class: 'lbl strong' }, `What token ${N} can reach after ${layers} layer${layers > 1 ? 's' : ''}`));
    const r = reach(p, N - 1, w, layers, N);
    const bw = 16.5;
    for (let j = 0; j < N; j++) {
      svg.append(s('rect', { x: rx + j * bw, y: ry, width: bw - 2, height: 26, rx: 3, class: r.has(j) ? 'fill-compute' : 'cell-off', style: r.has(j) ? 'fill-opacity:.9' : '' }));
    }
    svg.append(s('text', { x: rx, y: ry + 44, class: 'lbl small muted' }, 'token 1'));
    svg.append(s('text', { x: rx + N * bw - 2, y: ry + 44, class: 'lbl small muted', 'text-anchor': 'end' }, `token ${N}`));
    // cost bars: per-query scores at large T
    const T = 131072;
    const perQ = p === 'dense' ? T : p === 'window' ? w * 1024 : p === 'strided' ? Math.round(2 * Math.sqrt(T)) : Math.round(2 * Math.sqrt(T));
    const by = 150;
    svg.append(s('text', { x: rx, y: by, class: 'lbl strong' }, 'Scores per query at T = 131,072'));
    const scaleW = 400;
    const lg = (x: number) => Math.log10(x) / Math.log10(T);
    svg.append(s('rect', { x: rx, y: by + 12, width: scaleW, height: 20, rx: 3, class: 'bar ghost' }));
    svg.append(s('text', { x: rx + scaleW - 6, y: by + 26, class: 'lbl small', 'text-anchor': 'end' }, 'dense: 131,072'));
    svg.append(s('rect', { x: rx, y: by + 40, width: Math.max(3, scaleW * lg(perQ)), height: 20, rx: 3, class: 'bar compute' }));
    svg.append(s('text', { x: rx + 6, y: by + 75, class: 'lbl small' }, p === 'dense' ? 'every earlier key' : p === 'window' ? `window W = ${w * 1024} (scaled-up version of this toy) — constant in T` : `≈ 2√T = ${perQ.toLocaleString()} (stride ≈ √T)`));
    svg.append(s('text', { x: rx, y: by + 95, class: 'lbl small muted' }, 'log scale; sparse patterns turn T² total work into T√T (strided/fixed) or T·W (window)'));
    f.readout.textContent = `${count} of ${dense} scores computed in this toy (${Math.round((100 * count) / dense)}%). ` +
      (p === 'window' ? `The KV cache can be a rolling buffer of the last ${w} tokens; anything older is reachable only by hopping through layers (${layers} × ${w - 1} positions here).` :
        p === 'dense' ? 'Every score computed: one hop reaches everything, at T² cost.' :
          'Two kinds of heads/positions together give every token a path to every earlier token within two hops.');
  }
  draw();
  return f.root;
}
