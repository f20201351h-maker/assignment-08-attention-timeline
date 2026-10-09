import { s, svgRoot, frame, seg, slider } from '../lib/dom.ts';
import { kvCacheBytes, fmtBytes, fmtCount } from '../lib/math.ts';

type Ratio = '0' | '1' | '3' | '7' | 'all';

/** Depth schedule: how many fixed-state layers per exact-attention layer. */
export function scheduleWidget() {
  const f = frame('A memory system per layer, not per model',
    'A 48-layer toy model. Orange layers are softmax attention (GQA, 8 KV heads × 128, bf16) and keep a cache that grows with context; teal layers keep a fixed-size recurrent state (16 heads × 128 × 128, bf16). Published hybrids for reference: Jamba 1 attention : 7 Mamba; MiniMax-01 1 : 7 linear; Qwen3-Next and Kimi Linear 1 : 3 (state : attention = 3 : 1). A DDDGDDDG schedule is also 3 : 1.');
  let ratio: Ratio = '3';
  let logT = 17;
  f.controls.append(
    seg<Ratio>({ label: 'State layers per attention layer', value: ratio, options: [['0', 'all attention'], ['1', '1 : 1'], ['3', '3 : 1'], ['7', '7 : 1'], ['all', 'all state']], onChange: (v) => { ratio = v; draw(); } }),
    slider({ label: 'Context', min: 12, max: 20, value: logT, fmt: (v) => fmtCount(2 ** v), onInput: (v) => { logT = v; draw(); } }),
  );
  const svg = svgRoot(760, 230, 'Layer stack with attention and state layers and resulting memory');
  f.stage.append(svg);

  function draw() {
    svg.replaceChildren();
    const L = 48;
    const isAttn = (l: number) => ratio === 'all' ? false : ratio === '0' ? true : (l + 1) % (Number(ratio) + 1) === 0;
    const nA = [...Array(L).keys()].filter(isAttn).length;
    const x0 = 30, cw = 14.2;
    svg.append(s('text', { x: x0, y: 20, class: 'lbl strong' }, `${L} layers, bottom → top: ${nA} attention, ${L - nA} state`));
    for (let l = 0; l < L; l++) {
      svg.append(s('rect', { x: x0 + l * cw, y: 32, width: cw - 2.5, height: 44, rx: 2.5, class: isAttn(l) ? 'fill-compute' : 'fill-memory', style: 'fill-opacity:.85' }));
      svg.append(s('text', { x: x0 + l * cw + (cw - 2.5) / 2, y: 58, class: 'cell-t on small' }, isAttn(l) ? 'G' : 'D'));
    }
    const T = 2 ** logT;
    const kv = kvCacheBytes({ layers: nA, kvHeads: 8, headDim: 128, tokens: T, batch: 1, bytes: 2 });
    const state = (L - nA) * 16 * 128 * 128 * 2;
    const allAttn = kvCacheBytes({ layers: L, kvHeads: 8, headDim: 128, tokens: T, batch: 1, bytes: 2 });
    const by = 110, W = 520;
    const bar = (label: string, b: number, cls: string, y: number) => {
      svg.append(s('text', { x: x0, y: y + 13, class: 'lbl small' }, label));
      svg.append(s('rect', { x: x0 + 150, y, width: Math.max(2, W * b / allAttn), height: 18, rx: 3, class: cls }));
      svg.append(s('text', { x: x0 + 156 + Math.max(2, W * b / allAttn), y: y + 13, class: 'lbl small num' }, fmtBytes(b)));
    };
    bar('attention KV cache', kv, 'bar compute', by);
    bar('recurrent state', state, 'bar memory', by + 28);
    bar('all-attention baseline', allAttn, 'bar ghost', by + 56);
    svg.append(s('text', { x: x0, y: by + 104, class: 'lbl small muted' }, `per sequence at ${fmtCount(T)} tokens; state size does not depend on context length`));
    f.readout.textContent = ratio === 'all'
      ? 'All state: tiny, constant memory — and no layer can ever look back at an exact earlier token. Recall-heavy tasks suffer.'
      : ratio === '0'
        ? 'All attention: exact access everywhere, and the whole cache scales with context.'
        : `Every attention layer you remove cuts the growing part of memory; every one you keep is another chance to re-read exact tokens. The ratio is a choice backed by ablations at specific scales, not a law.`;
  }
  draw();
  return f.root;
}
