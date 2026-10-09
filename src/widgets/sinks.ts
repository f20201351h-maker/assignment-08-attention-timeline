import { s, svgRoot, frame, seg, slider, sweepButton } from '../lib/dom.ts';
import { softmax, rng } from '../lib/math.ts';

type Policy = 'full' | 'window' | 'sink';

/**
 * Streaming with a bounded cache. A head that has learned to park spare
 * attention on token 0 (high logit) loses its parking spot when token 0 is evicted.
 */
export function sinkWidget() {
  const f = frame('Where does attention go when it has nowhere to go?',
    'Toy logits for one head that, like heads in real LLMs, gives the very first token a large score and uses it as a parking spot for spare attention (softmax must hand out exactly 1). The cache holds 8 recent tokens. Evict the first token and its share is forced onto the recent tokens — the head now mixes in far more content than it was trained to.');
  let policy: Policy = 'window';
  let t = 30;
  f.controls.append(
    seg<Policy>({ label: 'KV cache policy', value: policy, options: [['full', 'Keep everything'], ['window', 'Sliding window (8)'], ['sink', '4 sinks + window (8)']], onChange: (v) => { policy = v; draw(); } }),
  );
  const tS = slider({ label: 'Tokens streamed', min: 9, max: 60, value: t, onInput: (v) => { t = v; draw(); } });
  f.controls.append(tS, sweepButton('▶ Stream', tS, 9, 60, (v) => { t = v; draw(); }, 110));
  const svg = svgRoot(760, 250, 'Attention weights under different cache eviction policies');
  f.stage.append(svg);
  const r = rng(21);
  const logit = Array.from({ length: 60 }, (_, j) => (j === 0 ? 4.2 : r() * 1.2));

  function draw() {
    svg.replaceChildren();
    const cached = [...Array(t).keys()].filter((j) => policy === 'full' || j >= t - 8 || (policy === 'sink' && j < 4));
    const fullW = softmax(logit.slice(0, t));
    const w = softmax(cached.map((j) => logit[j]));
    const contentShare = cached.reduce((a, j, n) => a + (j === 0 ? 0 : w[n]), 0);
    const trainedShare = 1 - fullW[0];
    const x0 = 30, W = 700, cw = W / 60, base = 170;
    svg.append(s('text', { x: x0, y: 22, class: 'lbl strong' }, `Token ${t}'s attention over the stream (grey = evicted from the cache)`));
    for (let j = 0; j < t; j++) {
      const idx = cached.indexOf(j);
      const x = x0 + j * cw;
      if (idx < 0) { svg.append(s('rect', { x: x + 1, y: base - 4, width: cw - 2, height: 4, class: 'bar ghost' })); continue; }
      const hgt = Math.min(130, w[idx] * 320);
      svg.append(s('rect', { x: x + 1, y: base - hgt, width: cw - 2, height: Math.max(1, hgt), rx: 1.5, class: j < 4 && policy !== 'window' ? 'bar length' : j === 0 ? 'bar length' : 'bar' }));
    }
    svg.append(s('line', { x1: x0, x2: x0 + W, y1: base, y2: base, class: 'axis' }));
    svg.append(s('text', { x: x0, y: base + 16, class: 'lbl small muted' }, 'token 0'));
    // share meter
    const my = 205;
    svg.append(s('text', { x: x0, y: my, class: 'lbl small' }, 'share of attention on real content'));
    svg.append(s('rect', { x: x0 + 210, y: my - 12, width: 400, height: 14, rx: 3, class: 'bar ghost' }));
    svg.append(s('rect', { x: x0 + 210, y: my - 12, width: 400 * contentShare, height: 14, rx: 3, class: contentShare > trainedShare * 1.5 ? 'fill-bad' : 'fill-memory' }));
    svg.append(s('line', { x1: x0 + 210 + 400 * trainedShare, x2: x0 + 210 + 400 * trainedShare, y1: my - 18, y2: my + 6, class: 'stroke-ink', 'stroke-width': 2 }));
    svg.append(s('text', { x: x0 + 210 + 400 * trainedShare, y: my + 20, class: 'lbl-c small' }, 'what training taught it'));
    svg.append(s('text', { x: x0 + 620, y: my, class: 'lbl small num' }, `${(contentShare * 100).toFixed(0)}%`));
    f.readout.textContent = policy === 'full'
      ? `Full cache: ${cached.length} entries and growing. The head parks ${(fullW[0] * 100).toFixed(0)}% on token 0 — exactly the regime it was trained in.`
      : policy === 'window'
        ? `Window only: token 0 is gone, so ${(contentShare * 100).toFixed(0)}% of the attention lands on content instead of ~${(trainedShare * 100).toFixed(0)}%. StreamingLLM reports this is when perplexity explodes.`
        : `Keeping the first 4 tokens restores the parking spot: content share ${(contentShare * 100).toFixed(0)}%, cache fixed at 12 entries forever. But tokens in the middle are gone for good — this streams, it does not remember.`;
  }
  draw();
  return f.root;
}
