import { s, svgRoot, frame, seg, slider, toggle, button, animate, fmt } from '../lib/dom.ts';
import { rng, rot2, dot, softmax, alibiSlopes } from '../lib/math.ts';

/** Learned absolute table: rows exist only up to max_position, and only trained rows mean anything. */
export function learnedWallWidget() {
  const f = frame('A lookup table has a last row',
    'Each column is one row of a learned position table (8 of its numbers shown as colour). Rows the model saw during training have learned structure; rows past the training length were never updated, and rows past the table size do not exist at all.');
  let train = 16, ask = 24;
  const SIZE = 20;
  f.controls.append(
    slider({ label: 'Trained up to', min: 4, max: SIZE, value: train, onInput: (v) => { train = v; draw(); } }),
    slider({ label: 'Ask for position', min: 1, max: 32, value: ask, onInput: (v) => { ask = v; draw(); } }),
  );
  const svg = svgRoot(760, 200, 'Learned position table with untrained and missing rows');
  f.stage.append(svg);
  const r = rng(3);
  const noise = Array.from({ length: 32 * 8 }, () => r());
  function draw() {
    svg.replaceChildren();
    const cw = 21, x0 = 30, y0 = 30, ch = 14;
    for (let p = 0; p < 32; p++) {
      const x = x0 + p * cw;
      for (let d = 0; d < 8; d++) {
        const y = y0 + d * ch;
        if (p >= SIZE) { svg.append(s('rect', { x, y, width: cw - 2, height: ch - 2, class: 'cell-masked' })); continue; }
        const v = p < train ? 0.5 + 0.5 * Math.sin(p / (1.5 + d) + d) : noise[p * 8 + d];
        svg.append(s('rect', { x, y, width: cw - 2, height: ch - 2, rx: 2, class: p < train ? 'cell' : 'fill-bad', style: `fill-opacity:${p < train ? 0.15 + 0.8 * v : 0.12 + 0.3 * v}` }));
      }
      if (p % 4 === 0) svg.append(s('text', { x: x + cw / 2, y: y0 + 8 * ch + 14, class: 'lbl-c num' }, String(p)));
    }
    const ax = x0 + (ask - 1) * cw + cw / 2;
    svg.append(s('path', { d: `M${ax},${y0 - 18} l-6,-8 h12 z`, class: 'fill-ink' }));
    svg.append(s('text', { x: x0 + SIZE * cw + 6, y: y0 - 6, class: 'lbl small muted' }, 'no row exists →'));
    svg.append(s('text', { x: x0, y: y0 + 8 * ch + 34, class: 'lbl small' }, `trained rows 0–${train - 1}`));
    const status = ask - 1 >= SIZE ? 'does not exist — the lookup fails (the hard wall)'
      : ask - 1 >= train ? 'exists, but was never trained — it injects noise exactly where you hoped to extend'
        : 'was trained — the model knows what this position means';
    f.readout.innerHTML = `Position ${ask - 1}: row ${status}.`;
  }
  draw();
  return f.root;
}

/** Sinusoids: each pair of dimensions is a clock hand turning at its own speed. */
export function sinusoidWidget() {
  const f = frame('Position as a bank of clocks',
    'Each pair of dimensions is a hand turning at its own frequency (fast at the top, slow at the bottom). A position is the reading of all the hands at once. The right panel shows the dot product between position p and position p + k: it depends almost only on the offset k, which is why the authors hoped relative offsets would be easy to learn.');
  let p = 5;
  const D = 16;
  f.controls.append(slider({ label: 'Position p', min: 0, max: 63, value: p, onInput: (v) => { p = v; draw(); } }));
  const svg = svgRoot(760, 250, 'Sinusoidal position encoding as rotating clock hands');
  f.stage.append(svg);
  const pe = (pos: number) => Array.from({ length: D }, (_, i) => {
    const w = 1 / Math.pow(100, (2 * Math.floor(i / 2)) / D); // base 100 for a 64-token toy (paper uses 10000)
    return i % 2 === 0 ? Math.sin(pos * w) : Math.cos(pos * w);
  });
  function draw() {
    svg.replaceChildren();
    for (let i = 0; i < D / 2; i++) {
      const cx = 40 + (i % 4) * 82, cy = 50 + Math.floor(i / 4) * 92, R = 32;
      const w = 1 / Math.pow(100, (2 * i) / D);
      svg.append(s('circle', { cx, cy, r: R, class: 'fill-card stroke-muted' }));
      const a = p * w;
      svg.append(s('line', { x1: cx, y1: cy, x2: cx + R * Math.sin(a), y2: cy - R * Math.cos(a), class: 'stroke-accent', 'stroke-width': 3, 'stroke-linecap': 'round' }));
      svg.append(s('text', { x: cx, y: cy + R + 14, class: 'lbl-c small muted' }, `pair ${i + 1}`));
    }
    // similarity vs offset
    const x0 = 400, y0 = 200, W = 330, H = 150;
    svg.append(s('text', { x: x0, y: 30, class: 'lbl strong' }, `PE(p) · PE(p + k), for k = −20 … 20`));
    svg.append(s('line', { x1: x0, x2: x0 + W, y1: y0, y2: y0, class: 'axis' }));
    const base = pe(p);
    let d = '';
    for (let k = -20; k <= 20; k++) {
      const q = p + k;
      const v = q < 0 ? NaN : dot(base, pe(q)) / (D / 2);
      if (isNaN(v)) continue;
      d += `${d ? 'L' : 'M'}${(x0 + ((k + 20) / 40) * W).toFixed(1)},${(y0 - ((v + 0.3) / 1.3) * H).toFixed(1)}`;
    }
    svg.append(s('path', { d, class: 'curve stroke-length' }));
    svg.append(s('line', { x1: x0 + W / 2, x2: x0 + W / 2, y1: y0 - H, y2: y0, class: 'grid' }));
    svg.append(s('text', { x: x0 + W / 2, y: y0 + 16, class: 'lbl-c small' }, 'k = 0'));
    f.readout.textContent = `Move p: the curve barely changes. The encoding is a fixed function, so position ${p + 1000} has a value too — but the model has only ever been trained to read the positions it saw.`;
  }
  draw();
  return f.root;
}

/** RoPE: rotate q and k by position; their dot product only sees the gap. */
export function ropeWidget() {
  const f = frame('Rotate by position, compare by angle',
    'One 2-d pair of a query and a key, with the same content vector. RoPE rotates the query by i·θ and the key by j·θ. The score depends on the angle between them — i.e. on i − j. Press “shift both” to move the pair ten places along the sequence: both arrows turn, the gap and the score do not.');
  let on = true, i = 8, j = 2;
  const theta = 0.35;
  const base = [0.95, 0.3];
  const ci = slider({ label: 'Query position i', min: 0, max: 40, value: i, onInput: (v) => { i = v; draw(); } });
  const cj = slider({ label: 'Key position j', min: 0, max: 40, value: j, onInput: (v) => { j = v; draw(); } });
  f.controls.append(toggle('RoPE on', on, (v) => { on = v; draw(); }), ci, cj,
    button('Shift both +10', () => {
      const i0 = i, j0 = j, di = Math.min(10, 40 - Math.max(i, j));
      animate(700, (t) => { i = i0 + di * t; j = j0 + di * t; draw(); }, () => { i = i0 + di; j = j0 + di; ci.set(i); cj.set(j); draw(); });
    }, 'ghost'));
  const svg = svgRoot(760, 260, 'Rotary position embedding on one pair of dimensions');
  f.stage.append(svg);
  const hist: number[] = [];
  function draw() {
    svg.replaceChildren();
    const cx = 150, cy = 130, R = 100;
    svg.append(s('circle', { cx, cy, r: R, class: 'fill-card stroke-muted', 'stroke-dasharray': '2 4' }));
    const qv = on ? rot2(base, i * theta) : base;
    const kv = on ? rot2(base, j * theta) : base;
    const arr = (v: number[], cls: string, lab: string) => {
      svg.append(s('line', { x1: cx, y1: cy, x2: cx + v[0] * R, y2: cy - v[1] * R, class: cls, 'stroke-width': 4, 'stroke-linecap': 'round' }));
      svg.append(s('text', { x: cx + v[0] * (R + 16), y: cy - v[1] * (R + 16) + 4, class: 'lbl-c strong' }, lab));
    };
    arr(kv, 'stroke-memory', `k @ ${Math.round(j)}`);
    arr(qv, 'stroke-accent', `q @ ${Math.round(i)}`);
    const score = dot(qv, kv);
    // score panel
    const x0 = 330;
    svg.append(s('text', { x: x0, y: 30, class: 'lbl strong' }, 'What attention sees'));
    svg.append(s('text', { x: x0, y: 60, class: 'lbl' }, `positions i = ${Math.round(i)}, j = ${Math.round(j)}`));
    svg.append(s('text', { x: x0, y: 82, class: 'lbl' }, `gap i − j = ${Math.round(i - j)}`));
    svg.append(s('text', { x: x0, y: 104, class: 'lbl' }, `angle between = ${on ? fmt(((i - j) * theta * 180) / Math.PI % 360, 0) + '°' : '0° (no position)'}`));
    svg.append(s('text', { x: x0, y: 134, class: 'lbl strong' }, `score q·k = ${fmt(score, 3)}`));
    hist.push(score); if (hist.length > 60) hist.shift();
    svg.append(s('text', { x: x0, y: 168, class: 'lbl small muted' }, 'score over your last moves'));
    let d = '';
    hist.forEach((v, n) => { d += `${n ? 'L' : 'M'}${x0 + n * 6},${230 - v * 40}`; });
    svg.append(s('line', { x1: x0, x2: x0 + 360, y1: 230, y2: 230, class: 'grid' }));
    svg.append(s('path', { d, class: 'curve stroke-accent', style: 'stroke-width:2' }));
    f.readout.textContent = on
      ? 'R(iθ)q · R(jθ)k = q · R((j − i)θ)k: absolute rotations cancel, the gap survives. Real heads use many pairs at different θ, from fast (local detail) to very slow (long range).'
      : 'RoPE off: identical content gives an identical score whether the tokens are adjacent or 30 apart. Content alone cannot see distance.';
  }
  draw();
  return f.root;
}

/** ALiBi: subtract slope × distance from every score; each head gets its own slope. */
export function alibiWidget() {
  const f = frame('A distance penalty, one slope per head',
    'ALiBi adds no position vectors at all. Each head subtracts m·(i − j) from the score before softmax, with fixed slopes 1/2, 1/4 … 1/256 for 8 heads. Steep heads look locally; gentle heads can still reach far. Because the penalty is defined for any distance, the same rule applies at lengths never seen in training.');
  let head = 3, len = 32;
  const slopes = alibiSlopes(8);
  f.controls.append(
    seg<string>({ label: 'Head', value: String(head), options: slopes.map((m, i) => [String(i), `m=1/${Math.round(1 / m)}`] as [string, string]), onChange: (v) => { head = Number(v); draw(); } }),
    slider({ label: 'Sequence length', min: 8, max: 64, value: len, onInput: (v) => { len = v; draw(); } }),
  );
  const svg = svgRoot(760, 230, 'ALiBi attention weights by distance for one head');
  f.stage.append(svg);
  const r = rng(5);
  const content = Array.from({ length: 64 }, () => r() * 1.5);
  function draw() {
    svg.replaceChildren();
    const m = slopes[head];
    const i = len - 1;
    const scores = Array.from({ length: len }, (_, j) => content[j] - m * (i - j));
    const w = softmax(scores);
    const x0 = 30, W = 700, base = 170, bw = W / len;
    svg.append(s('text', { x: x0, y: 22, class: 'lbl strong' }, `Last token's attention over ${len} earlier tokens (same toy content scores, bias = −${fmt(m, 4)} × distance)`));
    w.forEach((v, j) => svg.append(s('rect', { x: x0 + j * bw + 1, y: base - v * 400, width: Math.max(1, bw - 2), height: Math.max(0.5, v * 400), rx: 1.5, class: j > 23 ? 'bar' : 'bar', style: j >= 24 ? '' : 'opacity:.55' })));
    svg.append(s('line', { x1: x0, x2: x0 + W, y1: base, y2: base, class: 'axis' }));
    if (len > 24) svg.append(s('line', { x1: x0 + 24 * bw, x2: x0 + 24 * bw, y1: 40, y2: base + 8, class: 'stroke-muted', 'stroke-dasharray': '3 3' }));
    svg.append(s('text', { x: x0, y: base + 18, class: 'lbl small muted' }, 'oldest'));
    svg.append(s('text', { x: x0 + W, y: base + 18, class: 'lbl small muted', 'text-anchor': 'end' }, 'newest'));
    const far = w.slice(0, Math.max(0, len - 8)).reduce((a, b) => a + b, 0);
    f.readout.innerHTML = `This head puts <b>${(far * 100).toFixed(0)}%</b> of its weight more than 8 tokens back. ${m >= 0.125 ? 'Steep slope: effectively a soft sliding window.' : 'Gentle slope: long-range reading is still possible, just discounted.'} The recency bias is built in — useful for language modelling, a handicap when the answer sits far back.`;
  }
  draw();
  return f.root;
}

/** NoPE: a causal mask alone lets the model infer position. */
export function nopeWidget() {
  const f = frame('Position, smuggled in by the causal mask',
    'No position encoding anywhere. Suppose one head attends uniformly to everything it can see, and only the first token carries a marker value of 1. Token t can see t tokens, so it reads back exactly 1/t. Its position is now written in its own activations — the mask did it.');
  const svg = svgRoot(760, 180, 'Uniform causal attention recovering position as 1 over t');
  f.stage.append(svg);
  const N = 24, x0 = 40, W = 680, base = 150, bw = W / N;
  for (let t = 1; t <= N; t++) {
    const v = 1 / t;
    svg.append(s('rect', { x: x0 + (t - 1) * bw + 2, y: base - v * 120, width: bw - 4, height: v * 120, rx: 2, class: 'bar length' }));
    if (t % 4 === 1) svg.append(s('text', { x: x0 + (t - 1) * bw + bw / 2, y: base + 16, class: 'lbl-c num' }, `t=${t}`));
  }
  svg.append(s('text', { x: x0, y: 20, class: 'lbl strong' }, 'value read by token t = 1 / t  (distinct for every position)'));
  f.readout.textContent = 'This is the intuition behind Haviv et al. (2022): causal language models without positional encodings still learn absolute position. A bidirectional encoder has no such signal.';
  return f.root;
}
