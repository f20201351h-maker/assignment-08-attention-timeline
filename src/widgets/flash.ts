import { s, svgRoot, frame, slider, button, reducedMotion, fmt } from '../lib/dom.ts';
import { rng, softmax, fmtBytes } from '../lib/math.ts';

/**
 * FlashAttention as a tile walk: the T×T score matrix is never stored.
 * For one query row we keep a running max m, running sum l and running output,
 * rescaling when a later tile raises the max (the "online softmax" trick).
 */
export function flashWidget() {
  const f = frame('Exact attention without ever writing the score matrix',
    'The big square is the causal score matrix, cut into 8×8 tiles. Standard attention writes every score to GPU main memory (HBM), reads it back for softmax, writes the probabilities, reads them again. FlashAttention loads one block of queries into fast on-chip memory, streams key/value tiles past it, and keeps only three running numbers per row. Tiles above the diagonal are skipped. The final output is mathematically identical (up to floating-point rounding) — not an approximation.');
  const B = 8;
  let step = 0;
  let logN = 13;
  const total = (B * (B + 1)) / 2;
  const stepSl = slider({ label: 'Tiles processed', min: 0, max: total, value: step, onInput: (v) => { step = v; draw(); } });
  let timer = 0;
  f.controls.append(stepSl,
    button('Play', () => {
      clearInterval(timer);
      if (reducedMotion()) { step = total; stepSl.set(step); draw(); return; }
      step = 0;
      timer = window.setInterval(() => { step++; stepSl.set(step); draw(); if (step >= total) clearInterval(timer); }, 160);
    }, 'ghost'),
    slider({ label: 'Sequence T', min: 10, max: 17, value: logN, fmt: (v) => (2 ** v).toLocaleString(), onInput: (v) => { logN = v; draw(); } }));
  const svg = svgRoot(760, 300, 'Tiled attention computation with running softmax statistics');
  f.stage.append(svg);

  // one query row (last row of block 5) against 6 key tiles of 4 keys each
  const r = rng(42);
  const ROWBLOCK = 5, KPT = 4;
  const scores = Array.from({ length: (ROWBLOCK + 1) * KPT }, () => r() * 4 - 1);
  const vals = Array.from({ length: scores.length }, () => r() * 2 - 1);
  const order: [number, number][] = [];
  for (let i = 0; i < B; i++) for (let j = 0; j <= i; j++) order.push([i, j]);

  function draw() {
    svg.replaceChildren();
    const cs = 30, x0 = 20, y0 = 20;
    order.forEach(([i, j], n) => {
      const done = n < step, cur = n === step - 1;
      svg.append(s('rect', { x: x0 + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, rx: 3, class: cur ? 'fill-compute' : done ? 'cell' : 'cell-off', style: done && !cur ? 'fill-opacity:.35' : '' }));
    });
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) svg.append(s('rect', { x: x0 + j * cs, y: y0 + i * cs, width: cs - 2, height: cs - 2, class: 'cell-masked' }));
    svg.append(s('rect', { x: x0 - 3, y: y0 + ROWBLOCK * cs - 3, width: (ROWBLOCK + 1) * cs + 4, height: cs + 4, rx: 4, class: 'stroke-accent', fill: 'none', 'stroke-width': 1.5, 'stroke-dasharray': '3 2' }));
    svg.append(s('text', { x: x0, y: y0 + B * cs + 18, class: 'lbl small muted' }, 'skipped (future) tiles are dashed'));

    // online softmax for the highlighted row
    const doneTiles = order.slice(0, step).filter(([i]) => i === ROWBLOCK).map(([, j]) => j);
    let m = -Infinity, l = 0, acc = 0;
    for (const j of doneTiles) {
      for (let k = j * KPT; k < (j + 1) * KPT; k++) {
        const mNew = Math.max(m, scores[k]);
        const c = Math.exp(m - mNew);
        l = l * c + Math.exp(scores[k] - mNew);
        acc = acc * c + Math.exp(scores[k] - mNew) * vals[k];
        m = mNew;
      }
    }
    const exact = softmax(scores).reduce((a, w, k) => a + w * vals[k], 0);
    const px = 300;
    svg.append(s('text', { x: px, y: 30, class: 'lbl strong' }, 'Kept on-chip for the dashed row'));
    const rows: [string, string][] = [
      ['running max m', doneTiles.length ? fmt(m, 3) : '—'],
      ['running sum l', doneTiles.length ? fmt(l, 3) : '—'],
      ['running output acc / l', doneTiles.length ? fmt(acc / l, 4) : '—'],
      ['exact softmax output', fmt(exact, 4)],
    ];
    rows.forEach(([k, v], n) => {
      svg.append(s('text', { x: px, y: 58 + n * 22, class: 'lbl' }, k));
      svg.append(s('text', { x: px + 190, y: 58 + n * 22, class: 'lbl num' }, v));
    });
    const finished = doneTiles.length === ROWBLOCK + 1;
    svg.append(s('text', { x: px, y: 152, class: 'lbl strong ' + (finished ? 'text-good' : 'muted') }, finished ? '✓ same as the exact answer (up to rounding)' : `${doneTiles.length} of ${ROWBLOCK + 1} key tiles seen for this row`));
    // memory bars
    const T = 2 ** logN;
    const stdMem = T * T * 2; // bf16 score/probability matrix, one head
    const flashMem = T * 2 * 4; // m and l per row, fp32
    const by = 190;
    svg.append(s('text', { x: px, y: by, class: 'lbl strong' }, `Extra memory per head, T = ${T.toLocaleString()}`));
    const lg = (x: number) => Math.max(4, (Math.log10(x) / Math.log10(stdMem)) * 240);
    svg.append(s('text', { x: px, y: by + 24, class: 'lbl small' }, 'standard: T×T matrix'));
    svg.append(s('rect', { x: px + 130, y: by + 12, width: lg(stdMem), height: 16, rx: 3, class: 'bar compute' }));
    svg.append(s('text', { x: px + 136 + lg(stdMem), y: by + 25, class: 'lbl small num' }, fmtBytes(stdMem)));
    svg.append(s('text', { x: px, y: by + 66, class: 'lbl small' }, 'flash: m, l per row'));
    svg.append(s('rect', { x: px + 130, y: by + 54, width: lg(flashMem), height: 16, rx: 3, class: 'bar memory' }));
    svg.append(s('text', { x: px + 136 + lg(flashMem), y: by + 67, class: 'lbl small num' }, fmtBytes(flashMem)));
    svg.append(s('text', { x: px, y: by + 92, class: 'lbl small muted' }, 'log scale. FLOPs are unchanged (still ~T²); the win is memory traffic and footprint.'));
    f.readout.textContent = step === 0
      ? 'Press Play. Watch the running max: when a later tile holds a bigger score, earlier partial sums are rescaled by exp(m_old − m_new) instead of being recomputed.'
      : `${step} of ${total} tiles done. Nothing T×T is ever written out; the KV cache at inference is exactly as large as before.`;
  }
  draw();
  return f.root;
}
