import { s, svgRoot, frame, slider } from '../lib/dom.ts';
import { fmtCount, fmtBytes, kvCacheBytes } from '../lib/math.ts';

/** The two bills: pairwise compute ~T², cache memory ~T. Log-log axes so both fit. */
export function billsWidget() {
  const f = frame('The two bills attention sends',
    'Both axes are logarithmic so a million tokens fits on screen: a straight line twice as steep means quadratic. Cache numbers use a fixed yardstick (48 layers, 8 KV heads, head dim 128, bf16) for one conversation.');
  let logT = Math.log10(32768);
  f.controls.append(slider({ label: 'Context T', min: 2, max: 6.3, step: 0.01, value: logT, fmt: (v) => fmtCount(Math.round(10 ** v)) + ' tokens', onInput: (v) => { logT = v; draw(); } }));
  const svg = svgRoot(760, 300, 'Quadratic compute versus linear KV cache growth');
  f.stage.append(svg);

  const X0 = 70, X1 = 720, Y0 = 260, Y1 = 20;
  const xs = (lt: number) => X0 + ((lt - 2) / 4.3) * (X1 - X0);
  // y axis: log10 of "units", from 2 (T=100 linear) to 12.6 (T² at 2M)
  const ys = (lv: number) => Y0 - ((lv - 2) / 10.6) * (Y0 - Y1);

  function draw() {
    svg.replaceChildren();
    for (let e = 2; e <= 6; e++) {
      svg.append(s('line', { x1: xs(e), x2: xs(e), y1: Y1, y2: Y0, class: 'grid' }));
      svg.append(s('text', { x: xs(e), y: Y0 + 18, class: 'lbl-c num' }, fmtCount(10 ** e)));
    }
    svg.append(s('text', { x: (X0 + X1) / 2, y: Y0 + 36, class: 'lbl-c' }, 'context length T (log scale)'));
    svg.append(s('line', { x1: X0, x2: X1, y1: Y0, y2: Y0, class: 'axis' }));
    const path = (fn: (lt: number) => number) => {
      let d = '';
      for (let lt = 2; lt <= 6.3; lt += 0.05) d += `${d ? 'L' : 'M'}${xs(lt).toFixed(1)},${ys(fn(lt)).toFixed(1)}`;
      return d;
    };
    svg.append(s('path', { d: path((lt) => 2 * lt), class: 'curve stroke-compute' }));
    svg.append(s('path', { d: path((lt) => lt), class: 'curve stroke-memory' }));
    svg.append(s('text', { x: xs(5.2), y: ys(10.4) - 10, class: 'lbl strong', style: 'fill:var(--compute)' }, 'compute: T² query–key scores'));
    svg.append(s('text', { x: xs(5.0), y: ys(5) + 22, class: 'lbl strong', style: 'fill:var(--memory)' }, 'memory: one K and V per token'));
    const x = xs(logT);
    svg.append(s('line', { x1: x, x2: x, y1: Y1, y2: Y0, class: 'stroke-ink', 'stroke-width': 1, 'stroke-dasharray': '4 4' }));
    svg.append(s('circle', { cx: x, cy: ys(2 * logT), r: 6, class: 'fill-card stroke-compute', 'stroke-width': 3 }));
    svg.append(s('circle', { cx: x, cy: ys(logT), r: 6, class: 'fill-card stroke-memory', 'stroke-width': 3 }));
    const T = Math.round(10 ** logT);
    const cache = kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, tokens: T, batch: 1, bytes: 2 });
    f.readout.innerHTML = `At <b>${fmtCount(T)}</b> tokens: <b style="color:var(--compute)">${fmtCount(T * T)}</b> scores per head per layer, and a cache of <b style="color:var(--memory)">${fmtBytes(cache)}</b> for one user. Double T: scores ×4, cache ×2.`;
  }
  draw();
  return f.root;
}
