import { h, s, svgRoot, frame, seg, slider, toggle, fmt } from '../lib/dom.ts';
import { dot, softmax, mix } from '../lib/math.ts';

// Hand-made 4-d toy vectors (not a trained model) chosen so the pattern is
// readable: "sat" asks "who?", "mat" asks "where?", nouns look for a determiner.
const TOK = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const Q = [[0, 0, 0, 0.6], [0.2, 0, 0, 1.6], [2.2, 0.3, 0, 0], [0, 1.8, 0.4, 0], [0, 0, 0, 0.6], [0, 0.3, 2.0, 0.5]];
const K = [[0, 0, 0, 1.2], [2.0, 0, 0, 0.3], [0, 2.0, 0, 0], [0, 0.5, 1.6, 0], [0, 0, 0, 1.2], [0.3, 0, 1.8, 0.2]];
const V = [[0.1, 0, 0, 1], [1, 0.1, 0, 0], [0, 1, 0.1, 0], [0, 0.2, 1, 0], [0.1, 0, 0, 1], [0.2, 0, 0.9, 0.1]];
const DIMS = ['who', 'action', 'place', 'det.'];

type Step = 'score' | 'scale' | 'mask' | 'softmax' | 'mix';

export function attentionWidget() {
  const f = frame('One attention head, every number visible',
    'Six tokens, four numbers each, hand-picked so the pattern is readable (this is a toy, not a trained model). Click a row to choose the query. Turn the causal mask off and watch weight leak onto tokens that have not been written yet; drag temperature to sharpen or flatten the softmax.');
  let step: Step = 'softmax';
  let qi = 2;
  let masked = true;
  let temp = 1;

  f.controls.append(
    seg<Step>({ label: 'Step', value: step, options: [['score', '1 · QKᵀ'], ['scale', '2 · ÷√d'], ['mask', '3 · mask'], ['softmax', '4 · softmax'], ['mix', '5 · Σ w·V']], onChange: (v) => { step = v; draw(); } }),
    toggle('Causal mask', masked, (v) => { masked = v; draw(); }),
    slider({ label: 'Temperature', min: 0.25, max: 3, step: 0.05, value: temp, fmt: (v) => v.toFixed(2), onInput: (v) => { temp = v; draw(); } }),
  );

  const svg = svgRoot(780, 330, 'Attention score matrix and the selected query row');
  f.stage.append(svg);
  const eq = h('div', { class: 'eq' });
  f.stage.append(eq);

  function rowValues(i: number) {
    const raw = K.map((k) => dot(Q[i], k));
    const scaled = raw.map((x) => x / Math.sqrt(4));
    const m = scaled.map((x, j) => (masked && j > i ? -Infinity : x));
    const w = softmax(m, temp);
    return { raw, scaled, m, w };
  }

  function cellValue(i: number, j: number): number {
    const r = rowValues(i);
    if (step === 'score') return r.raw[j];
    if (step === 'scale') return r.scaled[j];
    if (step === 'mask') return r.m[j];
    return r.w[j];
  }

  function draw() {
    svg.replaceChildren();
    const cs = 42, ox = 78, oy = 46;
    const maxAbs = step === 'score' ? 4.5 : step === 'scale' || step === 'mask' ? 2.25 : 1;
    svg.append(s('text', { x: ox + 3 * cs, y: 18, class: 'lbl-c' }, 'keys →'));
    svg.append(s('text', { x: 14, y: oy + 3 * cs, class: 'lbl', transform: `rotate(-90 14 ${oy + 3 * cs})`, 'text-anchor': 'middle' }, 'queries →'));
    TOK.forEach((t, j) => svg.append(s('text', { x: ox + j * cs + cs / 2, y: oy - 8, class: 'lbl-c tok' }, t)));
    TOK.forEach((t, i) => {
      const row = s('g', { class: 'm-row' + (i === qi ? ' sel' : ''), tabindex: 0, role: 'button', 'aria-label': `Use ${t} as the query` });
      row.addEventListener('click', () => { qi = i; draw(); });
      row.addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter' || (e as KeyboardEvent).key === ' ') { e.preventDefault(); qi = i; draw(); } });
      row.append(s('rect', { x: 26, y: oy + i * cs, width: ox - 28 + 6 * cs, height: cs, class: 'row-hit' }));
      row.append(s('text', { x: ox - 8, y: oy + i * cs + cs / 2 + 4, class: 'lbl tok', 'text-anchor': 'end' }, t));
      TOK.forEach((_, j) => {
        const v = cellValue(i, j);
        const future = j > i;
        const x = ox + j * cs, y = oy + i * cs;
        if (v === -Infinity) {
          row.append(s('rect', { x: x + 1, y: y + 1, width: cs - 2, height: cs - 2, class: 'cell-masked' }));
          row.append(s('text', { x: x + cs / 2, y: y + cs / 2 + 4, class: 'cell-t muted' }, '−∞'));
        } else {
          const a = Math.min(1, Math.abs(v) / maxAbs);
          row.append(s('rect', { x: x + 1, y: y + 1, width: cs - 2, height: cs - 2, rx: 3, class: 'cell' + (future && !masked && step !== 'score' && step !== 'scale' ? ' leak' : ''), style: `fill-opacity:${0.08 + 0.85 * a}` }));
          row.append(s('text', { x: x + cs / 2, y: y + cs / 2 + 4, class: 'cell-t' + (a > 0.55 ? ' on' : '') }, fmt(v, step === 'softmax' || step === 'mix' ? 2 : 1)));
        }
      });
      svg.append(row);
    });

    // Right panel: the selected query row in detail
    const px = 370, pw = 390;
    const r = rowValues(qi);
    svg.append(s('text', { x: px, y: 18, class: 'lbl strong' }, `Query: “${TOK[qi]}”`));
    if (step !== 'mix') {
      const vals = step === 'score' ? r.raw : step === 'scale' ? r.scaled : step === 'mask' ? r.m : r.w;
      const bw = pw - 70;
      vals.forEach((v, j) => {
        const y = oy + j * cs + 8;
        svg.append(s('text', { x: px + 52, y: y + 16, class: 'lbl tok', 'text-anchor': 'end' }, TOK[j]));
        if (v === -Infinity) {
          svg.append(s('text', { x: px + 62, y: y + 16, class: 'lbl muted' }, 'future → −∞ → weight 0'));
          return;
        }
        const wv = Math.max(0, (v / maxAbs) * bw);
        svg.append(s('rect', { x: px + 60, y, width: Math.max(1, wv), height: 22, rx: 3, class: 'bar' + (j > qi && !masked && step === 'softmax' ? ' leak' : '') }));
        svg.append(s('text', { x: px + 66 + wv, y: y + 16, class: 'lbl num' }, fmt(v, step === 'softmax' ? 3 : 2)));
      });
      if (step === 'softmax') {
        const leak = r.w.slice(qi + 1).reduce((a, b) => a + b, 0);
        f.readout.textContent = masked
          ? `Weights sum to ${fmt(r.w.reduce((a, b) => a + b, 0), 3)}. Biggest share goes to “${TOK[r.w.indexOf(Math.max(...r.w))]}”.`
          : `Mask off: ${(leak * 100).toFixed(0)}% of “${TOK[qi]}”'s attention now lands on tokens from the future. In training that is cheating.`;
      } else if (step === 'mask') {
        f.readout.textContent = masked ? 'Future positions get −∞ before softmax, so they receive exactly zero weight.' : 'Mask is off: nothing stops this query from reading ahead.';
      } else if (step === 'scale') {
        f.readout.textContent = 'Divide by √dₖ = √4 = 2. With wide heads raw dot products grow, and an unscaled softmax saturates.';
      } else {
        f.readout.textContent = `${TOK.length} × ${TOK.length} = ${TOK.length ** 2} comparisons. Every query meets every key — that is the T² bill.`;
      }
    } else {
      // value mixing: stacked contribution per output dim
      const out = mix(r.w, V);
      const bw = 70, gap = 20, by = oy + 210;
      DIMS.forEach((d, k) => {
        const x = px + 20 + k * (bw + gap);
        let acc = 0;
        V.forEach((v, j) => {
          const c = r.w[j] * v[k];
          if (c <= 0) return;
          const hgt = c * 180;
          svg.append(s('rect', { x, y: by - (acc + c) * 180, width: bw, height: Math.max(0.5, hgt), class: 'stack', style: `fill-opacity:${0.25 + 0.7 * r.w[j]}` }));
          if (hgt > 14) svg.append(s('text', { x: x + bw / 2, y: by - (acc + c / 2) * 180 + 4, class: 'cell-t on' }, TOK[j]));
          acc += c;
        });
        svg.append(s('text', { x: x + bw / 2, y: by + 16, class: 'lbl-c' }, d));
        svg.append(s('text', { x: x + bw / 2, y: by - acc * 180 - 6, class: 'lbl-c num' }, fmt(out[k], 2)));
      });
      svg.append(s('line', { x1: px + 10, x2: px + pw - 10, y1: by, y2: by, class: 'axis' }));
      f.readout.textContent = `New vector for “${TOK[qi]}” = Σ weight × value. Each column shows which tokens' values it is built from.`;
    }

    const eqs: Record<Step, string> = {
      score: 'score<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub>',
      scale: 'score<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub> / √d<sub>k</sub>',
      mask: 'score<sub>ij</sub> + M<sub>ij</sub>,  M = 0 (past) or −∞ (future)',
      softmax: 'w<sub>ij</sub> = exp(score<sub>ij</sub>/τ) / Σ<sub>j′</sub> exp(score<sub>ij′</sub>/τ)',
      mix: 'out<sub>i</sub> = Σ<sub>j</sub> w<sub>ij</sub> v<sub>j</sub>   ⇒   Attention(Q,K,V) = softmax(QKᵀ/√d<sub>k</sub> + M) V',
    };
    eq.innerHTML = eqs[step];
  }
  draw();
  return f.root;
}
