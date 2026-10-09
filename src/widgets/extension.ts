import { s, svgRoot, frame, seg, slider } from '../lib/dom.ts';
import { ropeFreqs, scaledFreqs, yarnLogitScale, type Scaling } from '../lib/math.ts';

/**
 * For every RoPE dimension pair: the largest angle seen in training vs the
 * largest angle needed at the target length, after each scaling method.
 * Red = angles the model never saw (extrapolation). Also shows the angle
 * between neighbouring tokens (resolution) shrinking under interpolation.
 */
export function extensionWidget(initial: Scaling = 'pi') {
  const f = frame('Stretching RoPE past its training length',
    'Each column is one of 32 rotary pairs (head dim 64, base 10,000, trained on 4K tokens), fastest on the left. Grey shows the range of angles that pair reached during training; the coloured bar shows the range needed at the target length. Anything beyond the grey (red) is an angle the model has never seen. The lower strip shows the angle between neighbouring tokens: interpolation squeezes it, making nearby tokens harder to tell apart.');
  let method: Scaling = initial;
  let s2 = 3; // log2 of scale factor
  f.controls.append(
    seg<Scaling>({ label: 'Method', value: method, options: [['none', 'Plain RoPE'], ['pi', 'PI (Jun 2023)'], ['ntk', 'NTK-aware (Jun 2023)'], ['yarn', 'YaRN (Aug 2023)']], onChange: (v) => { method = v; draw(); } }),
    slider({ label: 'Target length', min: 1, max: 5, value: s2, fmt: (v) => `${4 * 2 ** v}K (×${2 ** v})`, onInput: (v) => { s2 = v; draw(); } }),
  );
  const svg = svgRoot(760, 320, 'Rotation angles per RoPE frequency under different scaling methods');
  f.stage.append(svg);

  function draw() {
    svg.replaceChildren();
    const d = 64, base = 10000, L = 4096, scale = 2 ** s2, Lt = L * scale;
    const f0 = ropeFreqs(d, base);
    const fs = scaledFreqs(d, base, scale, method, L);
    const x0 = 40, W = 690, bw = W / 32, top = 30, H = 170;
    // y: log of max angle (radians), from 1 to 1e5
    const ly = (a: number) => top + H - (Math.log10(Math.max(a, 1e-3)) + 1) / 5.5 * H;
    svg.append(s('text', { x: x0, y: 18, class: 'lbl strong' }, 'largest rotation angle reached (log scale)'));
    let unseen = 0;
    for (let i = 0; i < 32; i++) {
      const seen = L * f0[i];
      const need = Lt * fs[i];
      const x = x0 + i * bw;
      const yS = ly(seen), yN = ly(need);
      svg.append(s('rect', { x: x + 2, y: yS, width: bw - 4, height: top + H - yS, rx: 2, class: 'bar ghost' }));
      // full turn line: pairs that complete many turns in training have seen every angle
      const wrapsInTraining = seen >= 2 * Math.PI;
      const over = need > seen && !wrapsInTraining;
      if (over) unseen++;
      svg.append(s('rect', { x: x + bw / 2 - 2, y: Math.min(yN, yS), width: 4, height: Math.abs(yS - yN) + 0.5, class: over ? 'fill-bad' : 'fill-length' }));
      svg.append(s('circle', { cx: x + bw / 2, cy: yN, r: 3.5, class: over ? 'fill-bad' : 'fill-length' }));
    }
    svg.append(s('line', { x1: x0, x2: x0 + W, y1: ly(2 * Math.PI), y2: ly(2 * Math.PI), class: 'stroke-muted', 'stroke-dasharray': '4 4' }));
    svg.append(s('text', { x: x0 + 4, y: ly(2 * Math.PI) + 14, class: 'lbl small muted halo' }, 'one full turn (2π) — pairs above this line already saw every angle in training'));
    svg.append(s('text', { x: x0, y: top + H + 14, class: 'lbl small muted' }, 'fast pairs (local detail)'));
    svg.append(s('text', { x: x0 + W, y: top + H + 14, class: 'lbl small muted', 'text-anchor': 'end' }, 'slow pairs (long range)'));
    // resolution strip: neighbour angle relative to original
    const ry = 250;
    svg.append(s('text', { x: x0, y: ry - 8, class: 'lbl strong' }, 'angle between neighbouring tokens, relative to training (resolution)'));
    for (let i = 0; i < 32; i++) {
      const rel = fs[i] / f0[i];
      svg.append(s('rect', { x: x0 + i * bw + 2, y: ry, width: bw - 4, height: 26, rx: 2, class: 'fill-compute', style: `fill-opacity:${0.08 + 0.85 * (1 - rel)}` }));
    }
    svg.append(s('text', { x: x0, y: ry + 44, class: 'lbl small muted' }, 'darker = more squeezed'));
    const msgs: Record<Scaling, string> = {
      none: `${unseen} slow pairs must rotate further than anything seen in training. The formula is defined there; the model’s competence is not.`,
      pi: `Position Interpolation divides every position by ${scale}: no unseen angles, but every pair — including the fast ones that separate neighbouring tokens — is squeezed ${scale}×. Needs some fine-tuning to recover.`,
      ntk: `NTK-aware changes the base instead (b′ = b·s^(d/(d−2))): fast pairs keep their full resolution, slow pairs are interpolated. Works without fine-tuning to a degree, but a few middle pairs still drift slightly out of range${unseen ? ` (${unseen} here)` : ''}.`,
      yarn: `YaRN interpolates only pairs that never completed a full turn in training, leaves fast pairs alone, ramps in between, and multiplies logits by ${yarnLogitScale(scale).toFixed(2)} (temperature) to keep attention from flattening.`,
    };
    f.readout.textContent = msgs[method];
  }
  draw();
  return f.root;
}
