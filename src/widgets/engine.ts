import { h, seg, slider, toggle, reducedMotion } from '../lib/dom.ts';
import { dot, softmax } from '../lib/math.ts';

/*
 * The attention engine: one causal head, computed for real on tiny hand-made
 * 8-number embeddings, animated in five stages:
 *   1 project   x -> q, k, v (three real matrix products)
 *   2 compare   the query pulse visits every visible key; q.k/sqrt(d) appears
 *   3 softmax   scores compete; arcs thicken by weight; future keys get -inf
 *   4 gather    value particles stream back in proportion to weight
 *   5 update    x + attention output: the word's meaning moves on the map
 * Embedding dims are labelled so the arithmetic is readable; it is a teaching
 * toy, not a trained model.
 */

const DIMS = ['money', 'nature', 'living', 'action', 'place', 'function', 'seek-context', 'seek-agent'];
const CONTENT = 6; // first six dims are content; last two only shape queries
type V8 = number[];
const V = (o: Partial<Record<string, number>>): V8 => DIMS.map((d) => o[d] ?? 0);
const FN = V({ function: 1 });

const VOCAB: Record<string, V8> = {
  bank: V({ money: 0.6, nature: 0.6, place: 0.6, 'seek-context': 1 }),
  river: V({ nature: 1.6, place: 0.4 }), water: V({ nature: 1.4 }), muddy: V({ nature: 1.0 }), shore: V({ nature: 1.2, place: 0.6 }), fish: V({ nature: 0.8, living: 0.8 }),
  cash: V({ money: 1.6 }), money: V({ money: 1.6 }), loan: V({ money: 1.5 }), deposited: V({ money: 1.0, action: 0.8, 'seek-agent': 0.3 }), paid: V({ money: 1.0, action: 0.8 }), account: V({ money: 1.3 }),
  walked: V({ action: 1, 'seek-agent': 0.8 }), sat: V({ action: 1, 'seek-agent': 0.8 }), cross: V({ action: 1, place: 0.2, 'seek-agent': 0.8 }), swam: V({ action: 1, nature: 0.4, 'seek-agent': 0.8 }),
  animal: V({ nature: 0.3, living: 1.6 }), cat: V({ living: 1.6 }), dog: V({ living: 1.6 }), man: V({ living: 1.4 }), woman: V({ living: 1.4 }),
  street: V({ place: 1.4 }), mat: V({ place: 1.0 }), road: V({ place: 1.4 }), city: V({ place: 1.2, money: 0.3 }),
  it: V({ function: 0.6, 'seek-agent': 1.2 }), tired: V({ living: 0.4, action: 0.3, 'seek-agent': 1.0 }), wide: V({ place: 0.5, 'seek-context': 0.6 }),
  i: V({ living: 0.6, function: 0.6 }), he: V({ living: 0.6, function: 0.6 }), she: V({ living: 0.6, function: 0.6 }),
};
for (const w of ['the', 'a', 'an', 'of', 'to', 'at', 'on', 'my', 'was', 'by', 'along', 'because', 'and', 'in', "didn't", 'is', 'with', 'from', 'too']) VOCAB[w] = FN;

function embed(w: string): V8 {
  if (VOCAB[w]) return VOCAB[w];
  // unknown word: small deterministic content vector from a hash, so typing anything works
  let hsh = 2166136261;
  for (const c of w) hsh = Math.imul(hsh ^ c.charCodeAt(0), 16777619);
  return DIMS.map((_, i) => (i < CONTENT ? (((hsh >>> (i * 4)) & 15) / 15) * 0.5 : 0));
}

// projections (rows = output dim). Real matrix products, hand-designed.
const WQ: number[][] = DIMS.map((_, r) => DIMS.map((_, c) =>
  (r === 0 && c === 6) || (r === 1 && c === 6) ? 2.8 : r === 2 && c === 7 ? 3.4 : r === 3 && c === 3 ? 0.3 : r === 5 && c === 5 ? 0.2 : 0));
const WK: number[][] = DIMS.map((_, r) => DIMS.map((_, c) => (r !== c || r >= CONTENT ? 0 : [1, 1, 1, 0.5, 0.6, 0.4][r])));
const WV: number[][] = DIMS.map((_, r) => DIMS.map((_, c) => (r !== c || r >= CONTENT ? 0 : r === 5 ? 0.3 : 1)));
const mv = (m: number[][], x: V8) => m.map((row) => dot(row, x));

const PRESETS: [string, string][] = [
  ['I walked along the river to the bank', 'River bank'],
  ['I deposited my cash at the bank', 'Money bank'],
  ["The animal didn't cross the street because it was tired", 'What is “it”?'],
  ['The bank of the river was muddy', 'Mask demo'],
];
const STAGES = ['Project', 'Compare', 'Softmax', 'Gather', 'Update'];
const COL = { q: '#7aa2ff', k: '#fb923c', v: '#2dd4bf', out: '#f472b6', ink: '#efe8da', dim: '#8a8172', bad: '#f87171' };
const ease = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));

export function attentionEngine() {
  let words: string[] = [];
  let qi = 0;
  let masked = true;
  let temp = 1;
  let head: 'meaning' | 'prev' = 'meaning';
  let p = 0;             // stage progress 0..5
  let playing = false;
  let hover = -1;
  let visible = false;
  let clock = 0;

  const root = h('figure', { class: 'widget engine' });
  const title = h('div', { class: 'w-title' }, h('span', { class: 'w-dot' }), 'The attention engine — one head, computed live');
  const input = h('input', { type: 'text', class: 'engine-input', 'aria-label': 'Sentence (up to 11 words)', maxlength: 90 });
  const presetBox = h('div', { class: 'engine-presets' });
  PRESETS.forEach(([s, label]) => presetBox.append(h('button', { type: 'button', class: 'chip-btn', onclick: () => load(s) }, label)));
  const form = h('form', { class: 'engine-form', onsubmit: (e: Event) => { e.preventDefault(); load(input.value); } }, input, h('button', { type: 'submit', class: 'btn' }, 'Run'));
  const headSeg = seg<'meaning' | 'prev'>({ label: 'Head', value: head, options: [['meaning', 'meaning head'], ['prev', 'previous-token head']], onChange: (v) => { head = v; compute(); replay(); } });
  const maskT = toggle('Causal mask', masked, (v) => { masked = v; compute(); replay(); });
  const tempS = slider({ label: 'Temperature', min: 0.3, max: 3, step: 0.05, value: temp, fmt: (v) => v.toFixed(2), onInput: (v) => { temp = v; compute(); if (p < 2) p = 2.999; draw(); } });
  const playBtn = h('button', { type: 'button', class: 'btn' }, '▶ Replay');
  playBtn.addEventListener('click', () => replay());
  const scrub = h('input', { type: 'range', min: 0, max: 500, value: 0, 'aria-label': 'Scrub through the five stages', class: 'engine-scrub' });
  scrub.addEventListener('input', () => { playing = false; p = Number(scrub.value) / 100; draw(); });
  const stageNav = h('div', { class: 'engine-stages', role: 'tablist', 'aria-label': 'Stages' });
  const stageBtns = STAGES.map((sname, i) => {
    const b = h('button', { type: 'button', role: 'tab', class: 'stage-btn' }, h('b', null, String(i + 1)), sname);
    b.addEventListener('click', () => { playing = false; p = i + 0.999; draw(); });
    stageNav.append(b);
    return b;
  });

  const stage = h('div', { class: 'engine-stage' });
  const canvas = h('canvas', { class: 'engine-canvas', 'aria-hidden': 'true' });
  const tokRow = h('div', { class: 'engine-tokens', role: 'group', 'aria-label': 'Tokens — choose the query' });
  const stageTag = h('div', { class: 'engine-tag' });
  stage.append(canvas, stageTag, tokRow);

  const lower = h('div', { class: 'engine-lower' });
  const mapCanvas = h('canvas', { class: 'engine-map', 'aria-hidden': 'true' });
  const vecPanel = h('div', { class: 'engine-vecs' });
  lower.append(h('div', { class: 'engine-mapwrap' }, mapCanvas), vecPanel);
  const readout = h('div', { class: 'w-readout', 'aria-live': 'polite' });

  root.append(title,
    h('div', { class: 'engine-top' }, presetBox, form),
    h('div', { class: 'w-controls' }, headSeg, maskT, tempS),
    h('div', { class: 'engine-transport' }, playBtn, stageNav, scrub),
    stage, lower, readout,
    h('figcaption', { class: 'w-caption', html: 'Click any word to make it the query (or press Replay). Eight labelled numbers per word, three hand-designed projection matrices, real dot products, real softmax — a readable toy, not a trained model. The <em>meaning head</em> compares content; the <em>previous-token head</em> ignores content and just looks one step back, the way real heads specialise. Try the mask demo: with the causal mask on, “bank” cannot see the “river” that comes after it.' }));

  // ---------- state ----------
  let X: V8[] = [], Q: V8[] = [], K: V8[] = [], VV: V8[] = [];
  let scores: number[] = [], weights: number[] = [], out: V8 = [];

  function compute() {
    X = words.map(embed);
    Q = X.map((x) => mv(WQ, x));
    K = X.map((x) => mv(WK, x));
    VV = X.map((x) => mv(WV, x));
    const d = Math.sqrt(DIMS.length);
    scores = words.map((_, j) => {
      if (masked && j > qi) return -Infinity;
      return head === 'meaning' ? dot(Q[qi], K[j]) / d : j === qi - 1 ? 3 : 0;
    });
    weights = softmax(scores, temp);
    out = VV[0].map((_, c) => weights.reduce((a, w, j) => a + w * VV[j][c], 0));
    renderTokens();
    renderVecs();
  }

  function renderTokens() {
    tokRow.replaceChildren(...words.map((w, j) => {
      const b = h('button', { type: 'button', class: 'etok' + (j === qi ? ' q' : '') + (masked && j > qi ? ' fut' : ''), 'aria-pressed': j === qi ? 'true' : 'false', 'aria-label': `${w}${j === qi ? ' (query)' : ''}` }, w);
      b.addEventListener('click', () => { qi = j; compute(); replay(); });
      const hv = (k: number) => { hover = k; updateReadout(); renderVecs(); draw(); };
      b.addEventListener('mouseenter', () => hv(j));
      b.addEventListener('mouseleave', () => hv(-1));
      b.addEventListener('focus', () => hv(j));
      return b;
    }));
  }

  function renderVecs() {
    const kj = hover >= 0 && hover !== qi && !(masked && hover > qi) ? hover : weights.map((w, j) => [w, j]).filter(([, j]) => j !== qi).sort((a, b) => b[0] - a[0])[0]?.[1] ?? qi;
    const prod = Q[qi].map((x, i) => x * K[kj][i]);
    const row = (label: string, v: V8, col: string, n = CONTENT) => h('div', { class: 'vrow' },
      h('span', { class: 'vlab', style: `color:${col}` }, label),
      ...v.slice(0, n).map((x) => h('span', { class: 'vcell', title: x.toFixed(2), style: `--a:${Math.min(1, Math.abs(x) / 1.6)};--c:${col}` }, x === 0 ? '·' : x.toFixed(1))));
    const after = X[qi].map((x, i) => (i < CONTENT ? x + out[i] : x));
    vecPanel.replaceChildren(
      h('div', { class: 'vrow vhead' }, h('span', { class: 'vlab' }, ''), ...DIMS.slice(0, CONTENT).map((d) => h('span', { class: 'vcell vdim' }, d))),
      ...(head === 'meaning' ? [
        row(`query of “${words[qi]}”`, Q[qi], 'var(--exact)'),
        row(`key of “${words[kj]}”`, K[kj], 'var(--compute)'),
        h('div', { class: 'vrow vsum' }, h('span', { class: 'vlab' }, 'q × k, per dimension'), ...prod.slice(0, CONTENT).map((x) => h('span', { class: 'vcell', style: `--a:${Math.min(1, Math.abs(x) / 4)};--c:var(--accent)` }, x === 0 ? '·' : x.toFixed(1))),
          h('span', { class: 'vtot' }, `Σ = ${prod.reduce((a, b) => a + b, 0).toFixed(2)} → ÷√8 = ${(prod.reduce((a, b) => a + b, 0) / Math.sqrt(8)).toFixed(2)}`)),
        h('div', { class: 'vgap' })] : []),
      row(`“${words[qi]}” before`, X[qi], 'var(--ink)'),
      row('attention output', out, '#db2777'),
      row('after (x + output)', after, 'var(--memory)'),
      h('p', { class: 'vnote' }, `${head === 'meaning' ? 'Hover a word to see its key matched against the query, dimension by dimension. ' : ''}The output is added back to the word’s own vector (the residual stream). The query was built from the word’s two hidden “seek” dimensions.`));
  }

  function load(sentence: string) {
    const ws = sentence.toLowerCase().replace(/[^a-z' ]/g, ' ').split(/\s+/).filter(Boolean).slice(0, 11);
    if (!ws.length) return;
    words = ws;
    input.value = ws.join(' ');
    // default query: the most "seeking" word, else the last
    const seekers = ws.map((w, j) => [embed(w)[6] + embed(w)[7], j]).sort((a, b) => b[0] - a[0]);
    qi = seekers[0][0] > 0.9 ? seekers[0][1] : ws.length - 1;
    compute();
    layout();
    replay();
  }

  function replay() {
    if (reducedMotion()) { p = 5; draw(); updateReadout(); return; }
    p = 0; playing = true; last = performance.now();
    loop();
  }

  // ---------- geometry ----------
  let W = 800, H = 380, dpr = 1;
  const xs: number[] = [];
  function layout() {
    const r = stage.getBoundingClientRect();
    W = Math.max(280, r.width); H = W < 560 ? 330 : 390;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    stage.style.height = `${H}px`;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    const mw = Math.min(300, Math.max(240, (lower.getBoundingClientRect().width || W) * 0.34));
    mapCanvas.width = mw * dpr; mapCanvas.height = 230 * dpr;
    mapCanvas.style.width = `${mw}px`; mapCanvas.style.height = '230px';
    xs.length = 0;
    const btns = [...tokRow.children] as HTMLElement[];
    btns.forEach((b) => { const br = b.getBoundingClientRect(); xs.push(br.left - r.left + br.width / 2); });
    draw();
  }
  new ResizeObserver(() => layout()).observe(stage);

  // ---------- drawing ----------
  const ctx = canvas.getContext('2d')!;
  const mctx = mapCanvas.getContext('2d')!;
  const cellOf = () => Math.max(3, Math.min(7, (W / Math.max(words.length, 1)) / 6.5));

  function glyph(x: number, yBottom: number, v: V8, col: string, fill: number, cell: number, n = DIMS.length) {
    for (let i = 0; i < n; i++) {
      const y = yBottom - (i + 1) * (cell + 1);
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      ctx.fillRect(x, y, cell, cell);
      const a = Math.min(1, Math.abs(v[i]) / 1.8) * clamp01(fill * n - i);
      if (a <= 0) continue;
      ctx.globalAlpha = 0.25 + 0.75 * a;
      ctx.fillStyle = col;
      ctx.fillRect(x, y, cell, cell);
      ctx.globalAlpha = 1;
    }
  }

  function arcPoint(x1: number, x2: number, y: number, t: number) {
    const hgt = Math.min(H * 0.34, 22 + Math.abs(x1 - x2) * 0.3);
    const cx = (x1 + x2) / 2, cy = y - hgt * 2;
    const u = 1 - t;
    return [u * u * x1 + 2 * u * t * cx + t * t * x2, u * u * y + 2 * u * t * cy + t * t * y, cx, cy] as const;
  }

  function draw() {
    if (!words.length || !xs.length) return;
    const pp = p;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // background: deep stage with a faint grid
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#0d0c10'); g.addColorStop(1, '#17141a');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.035)'; ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 28) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }

    const cell = cellOf();
    const gH = DIMS.length * (cell + 1);
    const yTok = H - 46;               // top of token chips
    const yGl = yTok - 14;             // glyph bottom
    const yArc = yGl - gH - 8;         // arcs start here
    const n = words.length;
    const fill = ease(clamp01(pp));    // stage 1

    // beams from token to its glyphs (stage 1)
    if (pp < 1.4) {
      words.forEach((_, j) => {
        const a = (1 - clamp01((pp - 1) / 0.4)) * 0.5;
        const grd = ctx.createLinearGradient(0, yTok, 0, yGl - gH);
        grd.addColorStop(0, `rgba(239,232,218,${a})`); grd.addColorStop(1, 'rgba(239,232,218,0)');
        ctx.fillStyle = grd;
        ctx.fillRect(xs[j] - cell * 1.6, yGl - gH * fill, cell * 3.2, yTok - yGl + gH * fill);
      });
    }
    // K and V glyphs for every token; Q glyph for the query
    words.forEach((_, j) => {
      const fut = masked && j > qi;
      ctx.globalAlpha = fut ? 0.35 : 1;
      glyph(xs[j] - cell - 1, yGl, K[j], COL.k, ease(clamp01(pp * 1.2 - j * 0.03)), cell);
      glyph(xs[j] + 1, yGl, VV[j], COL.v, ease(clamp01(pp * 1.2 - j * 0.03)), cell);
      ctx.globalAlpha = 1;
    });
    const qx = xs[qi];
    glyph(qx - cell * 2 - 3, yGl, Q[qi], COL.q, fill, cell);
    // labels under glyph row
    ctx.font = `600 ${Math.max(8, cell + 2)}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    if (n <= 8 || W > 700) {
      words.forEach((_, j) => {
        ctx.fillStyle = COL.k; ctx.fillText('k', xs[j] - cell / 2 - 1, yGl + 11);
        ctx.fillStyle = COL.v; ctx.fillText('v', xs[j] + cell / 2 + 1, yGl + 11);
      });
      ctx.fillStyle = COL.q; ctx.fillText('q', qx - cell * 1.5 - 3, yGl + 11);
    }

    // arcs
    const cmp = clamp01(pp - 1);        // stage 2
    const sm = ease(clamp01(pp - 2));   // stage 3
    const gat = clamp01(pp - 3);        // stage 4
    const maxW = Math.max(...weights);
    words.forEach((_, j) => {
      const fut = masked && j > qi;
      const x2 = xs[j];
      const order = Math.abs(j - qi) / Math.max(1, n);
      const u = clamp01((cmp - order * 0.45) / 0.5);
      if (pp < 1) return;
      if (j === qi) {
        // self-attention loop
        const r = 12 + 10 * (sm * weights[j] / maxW);
        ctx.strokeStyle = `rgba(122,162,255,${0.25 + 0.6 * u})`;
        ctx.lineWidth = 1 + sm * 8 * weights[j];
        ctx.beginPath(); ctx.arc(qx, yArc - r, r, 0.15 * Math.PI, 2.85 * Math.PI); ctx.stroke();
        labelScore(j, qx, yArc - 2 * r - 8, u, sm, fut);
        return;
      }
      const [, , cx, cy] = arcPoint(qx, x2, yArc, 0);
      const wv = weights[j] / Math.max(maxW, 1e-9);
      const thick = 1 + (1 - sm) * 1 + sm * 10 * weights[j];
      ctx.lineWidth = fut ? 1 : thick;
      ctx.setLineDash(fut ? [3, 5] : []);
      ctx.strokeStyle = fut ? `rgba(248,113,113,${0.35 * u + 0.1})` : `rgba(${sm > 0 ? '244,114,182' : '122,162,255'},${0.18 + 0.7 * u * (sm > 0 ? 0.3 + 0.7 * wv : 1)})`;
      ctx.beginPath(); ctx.moveTo(qx, yArc); ctx.quadraticCurveTo(cx, cy, x2, yArc); ctx.stroke();
      ctx.setLineDash([]);
      // travelling query pulse (stage 2)
      if (!fut && u > 0 && u < 1) {
        const [px, py] = arcPoint(qx, x2, yArc, u);
        glow(px, py, 5, COL.q);
      }
      // label just above the key end of the arc, plus a flash when the pulse lands
      labelScore(j, x2, yArc - 10, u, sm, fut);
      if (!fut && u >= 1 && cmp < 1.2 && pp < 2.4) {
        const fl = clamp01(1 - (cmp - order * 0.45 - 0.5) * 2.2);
        if (fl > 0) { ctx.globalAlpha = fl * (0.3 + 0.7 * clamp01(scores[j] / 2.5)); glow(x2, yArc + 2, 4 + 10 * fl, COL.k); ctx.globalAlpha = 1; }
      }
      // value particles flowing back (stages 4-5)
      if (!fut && pp >= 3 && weights[j] > 0.004) {
        const count = Math.round(4 + 36 * weights[j]);
        for (let i = 0; i < count; i++) {
          let t: number;
          if (pp < 4) { const launch = (i / count) * 0.55; t = clamp01((gat - launch) / 0.45); if (t <= 0 || t >= 1) continue; }
          else t = (clock * 0.22 + i / count) % 1;
          const [px, py] = arcPoint(x2, qx, yArc, t);
          ctx.globalAlpha = 0.35 + 0.6 * Math.sin(Math.PI * t);
          ctx.fillStyle = COL.v;
          ctx.beginPath(); ctx.arc(px, py, 1.6 + 1.8 * wv, 0, 2 * Math.PI); ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
    });

    // competition bar: raw scores (compare) morphing into softmax shares
    if (pp >= 1.2) {
      const bx = 16, bw = W - 32, by = 52, bh = 18;
      const vis = words.map((_, j) => j).filter((j) => !(masked && j > qi));
      const raw = vis.map((j) => Math.max(0.05, scores[j] + 1));
      const rs = raw.reduce((a, b) => a + b, 0);
      const share = vis.map((j, n) => (1 - sm) * (raw[n] / rs) + sm * weights[j]);
      ctx.font = '600 10px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.fillStyle = '#a79f90';
      ctx.fillText(sm < 0.5 ? 'raw scores (shifted, before softmax)' : 'softmax: shares of attention (sum = 100%)', bx, by - 7);
      let x = bx;
      const best = vis.reduce((a, j) => (weights[j] > weights[a] ? j : a), vis[0]);
      vis.forEach((j, n) => {
        const w = share[n] * bw * clamp01((pp - 1.2) * 2);
        ctx.fillStyle = j === best ? COL.out : j === qi ? COL.q : `rgba(244,114,182,${0.18 + 0.5 * (weights[j] / Math.max(...weights))})`;
        ctx.fillRect(x, by, Math.max(0, w - 2), bh);
        if (w > 44) { ctx.fillStyle = j === best || j === qi ? '#0d0c10' : '#efe8da'; ctx.fillText(words[j], x + 5, by + 12.5); }
        x += w;
      });
    }

    // output glyph near the query (fills during gather)
    if (pp >= 3) {
      const oy = yArc - 4;
      const f = pp < 4 ? ease(gat) : 1;
      const o = out.map((x) => x * f);
      const ox = qx + 26 + cell < W - 4 ? qx + 26 : qx - 26 - cell;
      ctx.save();
      ctx.shadowColor = COL.out; ctx.shadowBlur = 12 * f;
      glyph(ox, oy, o, COL.out, 1, cell, CONTENT);
      ctx.restore();
    }
    // highlight hovered key
    if (hover >= 0 && hover !== qi && xs[hover] !== undefined) {
      ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
      ctx.strokeRect(xs[hover] - cell - 4, yGl - gH - 3, cell * 2 + 6, gH + 6);
    }

    // stage tag
    const si = Math.min(4, Math.floor(pp));
    stageTag.textContent = `${si + 1} · ${STAGES[si]}`;
    stageBtns.forEach((b, i) => b.classList.toggle('on', i === si));
    scrub.value = String(Math.round(pp * 100));
    drawMap();
  }

  function labelScore(j: number, x: number, y: number, u: number, sm: number, fut: boolean) {
    if (u < 1 && !fut) return;
    ctx.textAlign = 'center';
    ctx.font = `600 ${W < 560 ? 10 : 11.5}px "JetBrains Mono", monospace`;
    if (fut) { ctx.fillStyle = COL.bad; ctx.fillText('−∞', x, y); return; }
    const s = scores[j];
    const txt = sm < 0.5 ? (head === 'meaning' ? s.toFixed(2) : s.toFixed(0)) : `${Math.round(weights[j] * 100)}%`;
    ctx.globalAlpha = sm < 0.5 ? 1 - sm * 1.2 : (sm - 0.5) * 2;
    ctx.fillStyle = sm < 0.5 ? COL.q : COL.out;
    if (words.length <= 9 || W > 700 || weights[j] > 0.08) ctx.fillText(txt, x, y);
    ctx.globalAlpha = 1;
  }

  function glow(x: number, y: number, r: number, col: string) {
    ctx.save();
    ctx.shadowColor = col; ctx.shadowBlur = 16;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }

  // meaning map: the two content dims the attention output moves most
  function drawMap() {
    const mw = mapCanvas.width / dpr, mh = mapCanvas.height / dpr;
    mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mctx.fillStyle = '#121016'; mctx.fillRect(0, 0, mw, mh);
    const dimsBy = [0, 1, 2, 4, 3].sort((a, b) => Math.abs(out[b]) - Math.abs(out[a]));
    const [ax, ay] = [dimsBy[0], dimsBy[1]].sort((a, b) => a - b);
    const pad = 30, sc = (mw - pad * 2) / 2.4, scy = (mh - pad * 2) / 2.4;
    const P = (v: V8) => [pad + v[ax] * sc, mh - pad - v[ay] * scy] as const;
    mctx.strokeStyle = 'rgba(255,255,255,0.15)'; mctx.lineWidth = 1;
    mctx.beginPath(); mctx.moveTo(pad, mh - pad); mctx.lineTo(mw - 8, mh - pad); mctx.moveTo(pad, mh - pad); mctx.lineTo(pad, 10); mctx.stroke();
    mctx.font = '600 11px Inter, sans-serif'; mctx.fillStyle = '#a79f90'; mctx.textAlign = 'right';
    mctx.fillText(`${DIMS[ax]} →`, mw - 8, mh - 12);
    mctx.save(); mctx.translate(14, pad); mctx.rotate(-Math.PI / 2); mctx.textAlign = 'right'; mctx.fillText(`${DIMS[ay]} →`, 0, 0); mctx.restore();
    // reference words
    mctx.textAlign = 'left'; mctx.font = '500 10.5px Inter, sans-serif';
    const refs = new Set(['river', 'cash', 'animal', 'street', 'money', 'water', 'cat', 'road', 'loan']);
    const placed: [number, number][] = [];
    for (const w of refs) {
      const v = VOCAB[w]; if (!v) continue;
      if (Math.abs(v[ax]) + Math.abs(v[ay]) < 0.3) continue;
      const [x, y] = P(v);
      if (x > mw - 4 || y < 4 || placed.some(([a, b]) => Math.abs(a - x) < 34 && Math.abs(b - y) < 12)) continue;
      placed.push([x, y]);
      mctx.fillStyle = 'rgba(239,232,218,0.25)'; mctx.beginPath(); mctx.arc(x, y, 3, 0, 7); mctx.fill();
      mctx.fillText(w, x + 5, y + 3);
    }
    const before = X[qi];
    const f = ease(clamp01(p - 4));
    const after = before.map((x, i) => (i < CONTENT ? x + out[i] * f : x));
    const [bx, by] = P(before), [axp, ayp] = P(after);
    mctx.strokeStyle = COL.out; mctx.lineWidth = 2;
    if (f > 0) { mctx.beginPath(); mctx.moveTo(bx, by); mctx.lineTo(axp, ayp); mctx.stroke(); }
    mctx.strokeStyle = COL.ink; mctx.lineWidth = 1.5; mctx.beginPath(); mctx.arc(bx, by, 5, 0, 7); mctx.stroke();
    mctx.fillStyle = COL.ink; mctx.fillText(`“${words[qi]}”`, bx + 7, by - 6);
    if (f > 0) {
      mctx.save(); mctx.shadowColor = COL.out; mctx.shadowBlur = 14;
      mctx.fillStyle = COL.out; mctx.beginPath(); mctx.arc(axp, ayp, 6, 0, 7); mctx.fill(); mctx.restore();
      mctx.fillStyle = COL.out; mctx.fillText('after attention', axp + 8, ayp + 14);
    }
  }

  function updateReadout() {
    if (!words.length) return;
    const si = Math.min(4, Math.floor(p));
    const vis = words.map((w, j) => [weights[j], w, j] as const).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]);
    const top = vis[0];
    if (hover >= 0 && hover !== qi) {
      const fut = masked && hover > qi;
      readout.innerHTML = fut
        ? `“${words[hover]}” comes after “${words[qi]}”: the causal mask sets its score to −∞, so it gets exactly 0%.`
        : head === 'meaning'
          ? `q(“${words[qi]}”) · k(“${words[hover]}”) / √8 = <b>${scores[hover].toFixed(2)}</b> → after softmax <b>${(weights[hover] * 100).toFixed(1)}%</b> of the attention.`
          : `The previous-token head ignores content: “${words[hover]}” gets ${(weights[hover] * 100).toFixed(1)}% only because of where it sits.`;
      return;
    }
    const msgs = [
      `Each word is multiplied by three different matrices: a query (blue — what am I looking for?), a key (orange — what do I contain?) and a value (teal — what do I hand over?).`,
      `The query of “${words[qi]}” is dotted with every visible key. Big number = good match. ${masked ? `Words after “${words[qi]}” are in the future and are skipped.` : 'Mask off: future words are compared too.'}`,
      `Softmax turns scores into shares that sum to 100%. Raising one score lowers every other share — keys compete. Temperature ${temp.toFixed(2)}: ${temp < 0.8 ? 'sharp, nearly winner-takes-all' : temp > 1.6 ? 'flat, attention spread thin' : 'the standard setting'}.`,
      `Values flow back in proportion to the shares. ${top ? `“${top[1]}” sends ${(top[0] * 100).toFixed(0)}% of the mix.` : ''}`,
      `The mix is added to “${words[qi]}”’s own vector. ${head === 'meaning' && top && top[2] !== qi ? `“${words[qi]}” now carries some of “${top[1]}”: watch it move on the map.` : 'Same word, new context, new vector — that is the whole job of attention.'}`,
    ];
    readout.textContent = msgs[si];
  }

  // ---------- loop ----------
  let last = performance.now();
  let raf = 0;
  function loop() {
    cancelAnimationFrame(raf);
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      clock += dt;
      const prevStage = Math.floor(p);
      if (playing) { p = Math.min(5, p + dt * 0.62); if (p >= 5) playing = false; }
      if (Math.floor(p) !== prevStage) updateReadout();
      draw();
      // keep the gentle value stream alive only while on screen
      if (visible && (playing || p >= 4) && !reducedMotion()) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }
  new IntersectionObserver((es) => {
    const was = visible;
    visible = es[0].isIntersecting;
    if (visible && !was) { if (p === 0 && !playing) replay(); else loop(); }
  }, { threshold: 0.2 }).observe(root);

  // first load once the DOM exists
  requestAnimationFrame(() => { load(PRESETS[0][0]); updateReadout(); });
  return root;
}
