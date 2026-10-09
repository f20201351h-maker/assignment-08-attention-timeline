import { s, svgRoot, frame, seg, slider, fmt } from '../lib/dom.ts';
import { matVec, writeAdd, writeDelta, zeros, type Mat } from '../lib/math.ts';

type Rule = 'add' | 'delta' | 'gated';

/**
 * A 2-d key space with a scalar value: S is 1×2, read(k) = S·k.
 * Writes: A→40, then B→10, then A→55 (the correction).
 */
export function deltaWidget(initial: Rule = 'delta') {
  const gatedFirst = initial === 'gated';
  const f = frame(gatedFirst ? 'Correct one memory, fade the rest' : 'Write only the difference',
    'One fixed-size memory, three writes: key A → 40, key B → 10, then A’s answer changes to 55. Keys A and B are unit vectors; when they are not perpendicular, writing one disturbs the other. Add-only writes pile up; the delta rule reads first and writes only the error; the gate (α) shrinks the whole state before each write so stale associations can fade.');
  let rule: Rule = initial;
  let angle = 60;
  let beta = 1;
  let alpha = 0.8;
  f.controls.append(
    seg<Rule>({ label: 'Write rule', value: rule, options: [['add', 'Add-only (2020)'], ['delta', 'Delta (2021)'], ['gated', 'Gated delta (2024)']], onChange: (v) => { rule = v; draw(); } }),
    slider({ label: 'Angle A↔B', min: 20, max: 90, value: angle, fmt: (v) => `${v}°`, onInput: (v) => { angle = v; draw(); } }),
    slider({ label: 'β (write strength)', min: 0, max: 1, step: 0.05, value: beta, fmt: (v) => v.toFixed(2), onInput: (v) => { beta = v; draw(); } }),
    slider({ label: 'α (keep)', min: 0.3, max: 1, step: 0.05, value: alpha, fmt: (v) => v.toFixed(2), onInput: (v) => { alpha = v; draw(); } }),
  );
  const svg = svgRoot(760, 290, 'Memory readouts for keys A and B after each write');
  f.stage.append(svg);

  function draw() {
    svg.replaceChildren();
    const th = (angle * Math.PI) / 180;
    const A = [1, 0], B = [Math.cos(th), Math.sin(th)];
    const writes: [number[], number, string][] = [[A, 40, 'A → 40'], [B, 10, 'B → 10'], [A, 55, 'A → 55']];
    let S: Mat = zeros(1, 2);
    const states: Mat[] = [];
    for (const [k, v] of writes) {
      S = rule === 'add' ? writeAdd(S, k, [v]) : writeDelta(S, k, [v], beta, rule === 'gated' ? alpha : 1);
      states.push(S);
    }
    // key plane
    const cx = 120, cy = 160, R = 100;
    svg.append(s('circle', { cx, cy, r: R, class: 'fill-card stroke-muted', 'stroke-dasharray': '2 4' }));
    const arrow = (v: number[], cls: string, label: string, len = R) => {
      const x = cx + v[0] * len, y = cy - v[1] * len;
      svg.append(s('line', { x1: cx, y1: cy, x2: x, y2: y, class: cls, 'stroke-width': 3, 'stroke-linecap': 'round' }));
      svg.append(s('circle', { cx: x, cy: y, r: 4, class: cls.replace('stroke', 'fill') }));
      svg.append(s('text', { x: x + 8, y: y - 6, class: 'lbl strong' }, label));
    };
    arrow(A, 'stroke-exact', 'key A');
    arrow(B, 'stroke-memory', 'key B');
    svg.append(s('text', { x: cx, y: 286, class: 'lbl-c small muted' }, 'Keys live in the same small space,'));
    svg.append(s('text', { x: cx, y: 274, class: 'lbl-c small muted' }, `so A and B overlap by cos ${angle}° = ${Math.cos(th).toFixed(2)}`));
    // readouts after each write
    const x0 = 290, colW = 150, base = 220, sc = 1.6;
    writes.forEach(([, , label], t) => {
      const x = x0 + t * colW;
      svg.append(s('text', { x: x + 50, y: 22, class: 'lbl-c strong' }, `after ${label}`));
      const st = states[t];
      const want = t === 0 ? [40, 0] : t === 1 ? [40, 10] : [55, 10];
      [A, B].forEach((k, i) => {
        const val = matVec(st, k)[0];
        const bx = x + i * 52 + 10;
        const hgt = Math.max(0, Math.min(190, val * sc));
        svg.append(s('rect', { x: bx, y: base - hgt, width: 38, height: Math.max(0.5, hgt), rx: 3, class: i === 0 ? 'fill-exact' : 'fill-memory', style: 'fill-opacity:.85' }));
        if (t > 0 || i === 0) svg.append(s('line', { x1: bx - 4, x2: bx + 42, y1: base - want[i] * sc, y2: base - want[i] * sc, class: 'stroke-ink', 'stroke-width': 2, 'stroke-dasharray': '4 3' }));
        svg.append(s('text', { x: bx + 19, y: base - hgt - 6, class: 'lbl-c num' }, fmt(val, 1)));
        svg.append(s('text', { x: bx + 19, y: base + 16, class: 'lbl-c small' }, i === 0 ? 'read A' : 'read B'));
      });
    });
    svg.append(s('line', { x1: x0, x2: x0 + 3 * colW, y1: base, y2: base, class: 'axis' }));
    svg.append(s('text', { x: x0, y: 262, class: 'lbl small muted' }, 'dashed line = the answer we wanted at that moment'));
    const fin = states[2];
    const a = matVec(fin, A)[0], b = matVec(fin, B)[0];
    const msg = rule === 'add'
      ? `Add-only: A reads ${fmt(a, 1)} (wanted 55) — the “95” problem${angle < 90 ? ', and B is polluted too' : ''}. Nothing ever subtracts the old answer.`
      : rule === 'delta'
        ? `Delta rule: A reads ${fmt(a, 1)}, B reads ${fmt(b, 1)}. ${beta === 1 ? 'With β = 1 the latest write is exact for its key' : 'Smaller β writes only part of the correction'}${angle < 90 ? '; overlapping keys still nudge each other — a fixed state has finite room' : ''}.`
        : `Gated delta: A ${fmt(a, 1)}, B ${fmt(b, 1)}. α < 1 lets the model wipe stale associations fast (useful at a topic change) at the price of forgetting things it still needed.`;
    f.readout.textContent = msg;
  }
  draw();
  return f.root;
}
