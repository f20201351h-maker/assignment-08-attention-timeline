import { s, svgRoot, frame, seg, slider } from '../lib/dom.ts';
import { kvCacheBytes, fmtBytes, fmtCount } from '../lib/math.ts';

type Layout = '8' | '4' | '2' | '1';

/** MHA → GQA → MQA: 8 query heads share fewer stored K/V heads. */
export function headsWidget(initial: Layout = '1') {
  const f = frame('How many K/V heads does each token leave behind?',
    'Eight query heads in every layout — the questions stay different. Only the number of stored key/value heads changes. The cache uses a fixed yardstick (48 layers, head dim 128, bf16). The memory saving is exact arithmetic; the quality cost of sharing is not modelled here.');
  let kv: Layout = initial;
  let logT = Math.log2(32768);
  let users = 8;
  f.controls.append(
    seg<Layout>({ label: 'K/V heads', value: kv, options: [['8', 'MHA · 8'], ['4', 'GQA · 4'], ['2', 'GQA · 2'], ['1', 'MQA · 1']], onChange: (v) => { kv = v; draw(); } }),
    slider({ label: 'Context', min: 10, max: 20, step: 1, value: logT, fmt: (v) => fmtCount(2 ** v), onInput: (v) => { logT = v; draw(); } }),
    slider({ label: 'Active users', min: 1, max: 64, value: users, onInput: (v) => { users = v; draw(); } }),
  );
  const svg = svgRoot(760, 300, 'Query heads mapped to shared key-value heads, with cache size');
  f.stage.append(svg);

  function draw() {
    svg.replaceChildren();
    const n = Number(kv);
    const qx = 40, kx = 40, qy = 40, ky = 170, hw = 30, gap = 8;
    svg.append(s('text', { x: qx, y: 24, class: 'lbl strong' }, 'query heads (recomputed each step)'));
    svg.append(s('text', { x: kx, y: ky + 62, class: 'lbl strong' }, 'stored K/V heads (cached for every token)'));
    const kvW = (8 * (hw + gap) - gap - (n - 1) * gap) / n;
    for (let g = 0; g < n; g++) {
      const x = kx + g * (kvW + gap);
      for (let q = 0; q < 8 / n; q++) {
        const qi = g * (8 / n) + q;
        const cx = qx + qi * (hw + gap) + hw / 2;
        svg.append(s('path', { d: `M${cx},${qy + 46} C${cx},${(qy + ky) / 2 + 20} ${x + kvW / 2},${(qy + ky) / 2 - 10} ${x + kvW / 2},${ky}`, class: 'curve stroke-memory', style: 'stroke-width:1.5;opacity:.6' }));
      }
      svg.append(s('rect', { x, y: ky, width: kvW, height: 40, rx: 5, class: 'fill-memory', style: 'fill-opacity:.85' }));
      svg.append(s('text', { x: x + kvW / 2, y: ky + 25, class: 'cell-t on' }, n === 1 ? 'K,V shared by all' : `K,V ${g + 1}`));
    }
    for (let q = 0; q < 8; q++) {
      const x = qx + q * (hw + gap);
      svg.append(s('rect', { x, y: qy, width: hw, height: 46, rx: 5, class: 'fill-exact', style: 'fill-opacity:.85' }));
      svg.append(s('text', { x: x + hw / 2, y: qy + 28, class: 'cell-t on' }, `Q${q + 1}`));
    }
    // cache bars for all layouts
    const T = 2 ** logT;
    const bx = 400, by = 40, bw = 300;
    svg.append(s('text', { x: bx, y: 24, class: 'lbl strong' }, `KV cache, ${fmtCount(T)} tokens × ${users} user${users > 1 ? 's' : ''}`));
    const max = kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, tokens: T, batch: users, bytes: 2 });
    (['8', '4', '2', '1'] as Layout[]).forEach((l, i) => {
      const b = kvCacheBytes({ layers: 48, kvHeads: Number(l), headDim: 128, tokens: T, batch: users, bytes: 2 });
      const y = by + i * 46;
      const sel = l === kv;
      svg.append(s('text', { x: bx, y: y + 14, class: 'lbl' + (sel ? ' strong' : '') }, l === '8' ? 'MHA (8)' : l === '1' ? 'MQA (1)' : `GQA (${l})`));
      svg.append(s('rect', { x: bx + 70, y, width: bw * (b / max), height: 22, rx: 4, class: sel ? 'bar memory' : 'bar ghost' }));
      svg.append(s('text', { x: bx + 76 + bw * (b / max), y: y + 15, class: 'lbl num' }, fmtBytes(b)));
    });
    svg.append(s('text', { x: bx, y: by + 196, class: 'lbl small muted' }, 'An 80 GB accelerator holds weights too; the cache competes for what is left.'));
    const b = kvCacheBytes({ layers: 48, kvHeads: n, headDim: 128, tokens: T, batch: users, bytes: 2 });
    const b2 = kvCacheBytes({ layers: 48, kvHeads: n, headDim: 128, tokens: T * 2, batch: users, bytes: 2 });
    f.readout.innerHTML = `${n === 8 ? 'Multi-head' : n === 1 ? 'Multi-query' : 'Grouped-query'}: <b>${fmtBytes(b)}</b> of cache — ${8 / n}× smaller than MHA. But double the context and it becomes ${fmtBytes(b2)}: sharing lowers the slope, it does not stop the line.`;
  }
  draw();
  return f.root;
}

/** MLA: cache a small latent per token instead of full K and V. DeepSeek-V2 dims. */
export function mlaWidget() {
  const f = frame('What one token leaves in the cache, per layer (DeepSeek-V2 sizes)',
    'Numbers per token per layer for DeepSeek-V2’s attention shape (128 heads × 128 dims). MLA caches a 512-wide latent plus one shared 64-wide RoPE key; the paper notes this equals GQA with 2.25 groups. Keys and values are re-expanded from the latent by matrices that can be folded into the query and output projections at inference.');
  type L = 'mha' | 'gqa' | 'mqa' | 'mla';
  let sel: L = 'mla';
  f.controls.append(seg<L>({ label: 'Layout', value: sel, options: [['mha', 'MHA'], ['gqa', 'GQA · 8 groups'], ['mqa', 'MQA'], ['mla', 'MLA']], onChange: (v) => { sel = v; draw(); } }));
  const svg = svgRoot(760, 280, 'Per-token cache size for MHA, GQA, MQA and MLA');
  f.stage.append(svg);
  const sizes: Record<L, number> = { mha: 2 * 128 * 128, gqa: 2 * 8 * 128, mqa: 2 * 128, mla: 512 + 64 };
  const labels: Record<L, string> = { mha: 'MHA: K and V for 128 heads', gqa: 'GQA: 8 shared K/V heads', mqa: 'MQA: one K/V head', mla: 'MLA: latent c (512) + RoPE key (64)' };

  function draw() {
    svg.replaceChildren();
    const x0 = 40, w = 680;
    (['mha', 'gqa', 'mqa', 'mla'] as L[]).forEach((l, i) => {
      const y = 30 + i * 52;
      const frac = Math.sqrt(sizes[l] / sizes.mha); // sqrt so small ones stay visible; labelled
      svg.append(s('text', { x: x0, y: y - 6, class: 'lbl' + (l === sel ? ' strong' : '') }, labels[l]));
      if (l === 'mla') {
        const a = w * Math.sqrt(512 / sizes.mha), b = w * Math.sqrt(576 / sizes.mha) - a;
        svg.append(s('rect', { x: x0, y, width: a, height: 22, rx: 4, class: l === sel ? 'bar memory' : 'bar ghost' }));
        svg.append(s('rect', { x: x0 + a, y, width: b, height: 22, rx: 0, class: l === sel ? 'bar length' : 'bar ghost' }));
      } else {
        svg.append(s('rect', { x: x0, y, width: w * frac, height: 22, rx: 4, class: l === sel ? 'bar memory' : 'bar ghost' }));
      }
      svg.append(s('text', { x: x0 + w * frac + 8, y: y + 15, class: 'lbl num' }, `${sizes[l].toLocaleString()} numbers`));
    });
    svg.append(s('text', { x: x0, y: 250, class: 'lbl small muted' }, 'Bar length ∝ √size so the small layouts stay visible; the printed numbers are exact. Violet = the decoupled RoPE part.'));
    const r = sizes.mha / sizes[sel];
    f.readout.innerHTML = sel === 'mla'
      ? `MLA stores <b>576</b> numbers per token per layer — ${Math.round(r)}× less than full MHA at this shape, about the size of GQA with 2.25 groups (2 × 2.25 × 128 = 576). Position needs its own small uncompressed key, because RoPE’s rotation would sit between the matrices MLA wants to fold together.`
      : `${labels[sel]}: ${sizes[sel].toLocaleString()} numbers — ${r === 1 ? 'the baseline' : r + '× less than MHA'}.`;
  }
  draw();
  return f.root;
}
