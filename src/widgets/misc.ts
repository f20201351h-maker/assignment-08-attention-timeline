import { h, s, svgRoot, frame, seg, slider } from '../lib/dom.ts';

/** DroPE: what is done when, and what is shown vs merely defined. */
export function dropeWidget() {
  const f = frame('DroPE: use RoPE to learn, then take it away',
    'The procedure as written in Gelberg et al. (2025): pretrain normally with RoPE, delete RoPE from every layer, then continue training briefly at the original context length — no long-context fine-tuning. The claim it supports is empirical (benchmarks at longer lengths), not a guarantee that any length will work.');
  type View = 'rope' | 'drope';
  let view: View = 'drope';
  f.controls.append(seg<View>({ label: 'Path', value: view, options: [['rope', 'Plain RoPE, ask for 8× length'], ['drope', 'DroPE procedure']], onChange: (v) => { view = v; draw(); } }));
  const svg = svgRoot(760, 200, 'Training phases for plain RoPE versus DroPE');
  f.stage.append(svg);
  function draw() {
    svg.replaceChildren();
    const x0 = 30, y = 50, W = 700;
    const phase = (x: number, w: number, label: string, sub: string, cls: string) => {
      svg.append(s('rect', { x, y, width: w - 6, height: 56, rx: 8, class: cls, style: 'fill-opacity:.85' }));
      svg.append(s('text', { x: x + 12, y: y + 24, class: 'lbl strong', style: 'fill:#fff' }, label));
      svg.append(s('text', { x: x + 12, y: y + 42, class: 'lbl small', style: 'fill:#fff' }, sub));
    };
    svg.append(s('text', { x: x0, y: 28, class: 'lbl strong' }, 'training budget →'));
    if (view === 'rope') {
      phase(x0, W * 0.95, 'Pretrain with RoPE at length L', 'positions 0 … L−1 are the only ones ever used', 'fill-length');
      svg.append(s('text', { x: x0, y: 140, class: 'lbl' }, 'Then ask for 8L tokens at inference:'));
      svg.append(s('text', { x: x0, y: 162, class: 'lbl small muted' }, 'the rotation at 8L is defined (it is a formula) — but slow pairs reach angles never seen in training.'));
      f.readout.textContent = 'Defined is not demonstrated: the positional rule exists at 256K; the model’s competence there is a separate claim.';
    } else {
      phase(x0, W * 0.8, 'Pretrain with RoPE at length L', 'RoPE speeds up learning', 'fill-length');
      phase(x0 + W * 0.8, W * 0.2, 'Drop RoPE', 'recalibrate at L', 'fill-compute');
      svg.append(s('text', { x: x0, y: 140, class: 'lbl' }, 'Then evaluate at lengths far beyond L — nothing positional is left to go out of range.'));
      svg.append(s('text', { x: x0, y: 162, class: 'lbl small muted' }, 'Recalibration in the paper: e.g. the last 2B of 16B tokens for 0.5B models; 20B tokens for Llama-2-7B (≈1% of its 2T pretraining tokens).'));
      f.readout.textContent = 'Why it can work at all: a causal decoder can infer order from the mask (NoPE). RoPE is treated as a training aid, not a permanent part of the model.';
    }
  }
  draw();
  return f.root;
}

/** Selective state (Mamba / gated linear attention): the gate decides per token what to keep. */
export function selectiveWidget() {
  const f = frame('A state that chooses what to remember',
    'A stream of tokens writes into one fixed-size memory. With a fixed decay every token fades at the same rate, important or not. A selective gate is computed from the token itself: filler is let go quickly, the key fact is held. Bars show how much of each token is still in the state at the end.');
  type G = 'none' | 'fixed' | 'selective';
  let g: G = 'selective';
  let decay = 0.85;
  f.controls.append(
    seg<G>({ label: 'Forgetting', value: g, options: [['none', 'None (linear attn)'], ['fixed', 'Fixed decay (RetNet-style)'], ['selective', 'Selective (Mamba / GLA)']], onChange: (v) => { g = v; draw(); } }),
    slider({ label: 'Fixed decay', min: 0.5, max: 0.99, step: 0.01, value: decay, fmt: (v) => v.toFixed(2), onInput: (v) => { decay = v; draw(); } }),
  );
  const words = ['so', 'um', 'the', 'code', 'is', '4417', 'and', 'uh', 'the', 'weather', 'is', 'nice', 'so', 'yeah', 'what', 'is', 'the', 'code', '?'];
  const important = new Set([5]);
  const svg = svgRoot(760, 190, 'How much of each past token survives in a recurrent state');
  f.stage.append(svg);
  function draw() {
    svg.replaceChildren();
    const n = words.length, x0 = 30, cw = 700 / n, base = 130;
    const keep = (i: number) => (g === 'none' ? 1 : g === 'fixed' ? decay : important.has(i) ? 0.995 : 0.6);
    for (let i = 0; i < n; i++) {
      let w = 1;
      for (let t = i + 1; t < n; t++) w *= keep(t === i ? i : i); // gate set by the token's own content
      if (g === 'none') w = 1;
      const hgt = 90 * w;
      svg.append(s('rect', { x: x0 + i * cw + 3, y: base - hgt, width: cw - 6, height: Math.max(1, hgt), rx: 2, class: important.has(i) ? 'bar compute' : 'bar memory', style: 'fill-opacity:.85' }));
      svg.append(s('text', { x: x0 + i * cw + cw / 2, y: base + 16, class: 'lbl-c small' + (important.has(i) ? ' strong' : '') }, words[i]));
    }
    svg.append(s('line', { x1: x0, x2: x0 + 700, y1: base, y2: base, class: 'axis' }));
    f.readout.textContent = g === 'none'
      ? 'No forgetting: everything piles into the same matrix forever; old and new interfere and the state’s magnitude keeps growing.'
      : g === 'fixed'
        ? 'Fixed decay: recent tokens dominate whether or not they matter; the code fades like everything else.'
        : 'Selective: the gate depends on the input, so the state can hold “4417” across the filler. Still lossy — it cannot hold everything, and what it dropped is gone.';
  }
  draw();
  return f.root;
}

/** Bahdanau: one fixed vector vs looking back at every encoder state. */
export function bottleneckWidget() {
  const f = frame('The bottleneck attention was invented to remove',
    'An RNN encoder–decoder squeezes the whole source sentence into one fixed-size vector. Attention lets every output step look back at every encoder state and take a weighted mix. Later work reverses this on purpose: linear attention and DeltaNet squeeze the past into a fixed state again — for speed.');
  type M = 'fixed' | 'attn';
  let m: M = 'attn';
  let step = 3;
  f.controls.append(
    seg<M>({ label: 'Decoder reads', value: m, options: [['fixed', 'one fixed vector (2014 baseline)'], ['attn', 'all encoder states (attention)']], onChange: (v) => { m = v; draw(); } }),
    slider({ label: 'Output word', min: 0, max: 4, value: step, onInput: (v) => { step = v; draw(); } }),
  );
  const src = ['the', 'agreement', 'on', 'the', 'zone'];
  const tgt = ['l’', 'accord', 'sur', 'la', 'zone'];
  const align = [[0.8, 0.1, 0.05, 0.03, 0.02], [0.05, 0.85, 0.05, 0.03, 0.02], [0.02, 0.08, 0.8, 0.06, 0.04], [0.02, 0.03, 0.1, 0.8, 0.05], [0.01, 0.02, 0.05, 0.1, 0.82]];
  const svg = svgRoot(760, 200, 'Encoder-decoder with and without attention');
  f.stage.append(svg);
  function draw() {
    svg.replaceChildren();
    const ex = (i: number) => 80 + i * 130, dx = (i: number) => 80 + i * 130;
    src.forEach((w, i) => {
      svg.append(s('rect', { x: ex(i) - 40, y: 20, width: 80, height: 30, rx: 6, class: 'fill-bg2' }));
      svg.append(s('text', { x: ex(i), y: 40, class: 'lbl-c tok' }, w));
    });
    tgt.forEach((w, i) => {
      svg.append(s('rect', { x: dx(i) - 40, y: 150, width: 80, height: 30, rx: 6, class: i === step ? 'fill-accent' : 'fill-bg2', style: i === step ? 'fill-opacity:.9' : '' }));
      svg.append(s('text', { x: dx(i), y: 170, class: 'lbl-c tok', style: i === step ? 'fill:#fff' : '' }, w));
    });
    if (m === 'fixed') {
      svg.append(s('circle', { cx: 380, cy: 100, r: 16, class: 'fill-ink' }));
      svg.append(s('text', { x: 404, y: 104, class: 'lbl small' }, 'one vector for the whole sentence'));
      src.forEach((_, i) => svg.append(s('line', { x1: ex(i), y1: 50, x2: 380, y2: 86, class: 'stroke-muted', 'stroke-width': 1 })));
      svg.append(s('line', { x1: 380, y1: 116, x2: dx(step), y2: 150, class: 'stroke-ink', 'stroke-width': 2 }));
    } else {
      src.forEach((_, i) => svg.append(s('line', { x1: ex(i), y1: 50, x2: dx(step), y2: 150, class: 'stroke-accent', 'stroke-width': 1 + 9 * align[step][i], style: `opacity:${0.15 + 0.85 * align[step][i]}` })));
    }
    f.readout.textContent = m === 'fixed'
      ? 'Every output word must be produced from the same compressed summary — long sentences lose detail.'
      : `“${tgt[step]}” draws mostly on “${src[align[step].indexOf(Math.max(...align[step]))]}”. Illustrative weights; the paper visualises real alignments like these.`;
  }
  draw();
  return f.root;
}

export const note = (html: string) => h('div', { class: 'callout', html });
