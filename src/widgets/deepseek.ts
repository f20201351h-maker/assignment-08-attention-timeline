import { s, svgRoot, frame, seg, slider } from '../lib/dom.ts';
import { rng, fmtCount } from '../lib/math.ts';

type Mode = 'nsa' | 'dsa' | 'csa' | 'hca';

const INFO: Record<Mode, { title: string; stored: (T: number) => number; read: (T: number) => number; scored: (T: number) => number; note: string }> = {
  nsa: {
    title: 'NSA (Feb 2025): compressed + selected blocks + window',
    stored: (T) => T + T / 16,
    read: (T) => T / 16 + 16 * 64 + 512,
    scored: (T) => T / 16,
    note: 'Paper settings: compression block 32 / stride 16, 16 selected blocks of 64, window 512. The compressed tokens double as the cheap scorer for choosing blocks. Token-level K/V are all still stored.',
  },
  dsa: {
    title: 'DSA (Sep 2025): lightning indexer picks top-k tokens',
    stored: (T) => T,
    read: () => 2048,
    scored: (T) => T,
    note: 'A small FP8 indexer with a few heads scores every earlier token (still O(T²) in total, but cheap per score); the main MLA attention reads only the top 2,048. The cache is not smaller — the compute is.',
  },
  csa: {
    title: 'V4 CSA (Apr 2026): compress 4 → 1, then top-k',
    stored: (T) => T / 4 + 128,
    read: () => 1024 + 128,
    scored: (T) => T / 4,
    note: 'Every 4 tokens become one KV entry; a lightning indexer picks top-k compressed entries (1,024 for V4-Pro, 512 for Flash), plus 128 uncompressed recent tokens. Both the cache and the reads shrink.',
  },
  hca: {
    title: 'V4 HCA (Apr 2026): compress 128 → 1, read all',
    stored: (T) => T / 128,
    read: (T) => T / 128,
    scored: () => 0,
    note: 'Heavily compressed attention: 128 tokens per entry and no selection — dense attention over a very short summary. Interleaved with CSA layers so the model has both a cheap global view and a selective detailed one.',
  },
};

/** Toy strip of 128 tokens + real-scale arithmetic from the papers' settings. */
export function deepseekWidget(initial: Mode = 'nsa') {
  const f = frame('Three DeepSeek answers to “which keys should this query read?”',
    'The strip is a toy 128-token history (query at the right end) drawn to the same shape as each design; block sizes are shrunk to fit. The numbers below use the settings reported in each paper. Quality is not modelled: compression can blur detail and a selector can miss the key that mattered.');
  let mode: Mode = initial;
  let logT = 20;
  f.controls.append(
    seg<Mode>({ label: 'Design', value: mode, options: [['nsa', 'NSA'], ['dsa', 'DSA'], ['csa', 'V4 · CSA'], ['hca', 'V4 · HCA']], onChange: (v) => { mode = v; draw(); } }),
    slider({ label: 'Context', min: 15, max: 20, value: logT, fmt: (v) => fmtCount(2 ** v), onInput: (v) => { logT = v; draw(); } }),
  );
  const svg = svgRoot(760, 270, 'Which parts of a long history each DeepSeek design stores and reads');
  f.stage.append(svg);
  const r = rng(9);
  const rel = Array.from({ length: 128 }, (_, i) => (i === 37 || i === 38 || i === 81 ? 0.95 : r() * 0.6));

  function draw() {
    svg.replaceChildren();
    const x0 = 30, cw = 5.4, y = 40;
    svg.append(s('text', { x: x0, y: 22, class: 'lbl strong' }, INFO[mode].title));
    const row = (label: string, yy: number, cell: (i: number) => [string, number] | null) => {
      svg.append(s('text', { x: x0, y: yy - 4, class: 'lbl small muted' }, label));
      for (let i = 0; i < 128; i++) {
        const c = cell(i);
        svg.append(s('rect', { x: x0 + i * cw, y: yy, width: cw - 1, height: 14, class: c ? c[0] : 'cell-off', style: c ? `fill-opacity:${c[1]}` : '' }));
      }
    };
    const topBlocks = (size: number, n: number) => {
      const blocks = Array.from({ length: 128 / size }, (_, b) => [Math.max(...rel.slice(b * size, (b + 1) * size)), b]).sort((a, b) => b[0] - a[0]);
      return new Set(blocks.slice(0, n).map((x) => x[1]));
    };
    if (mode === 'nsa') {
      row('compressed branch: one coarse entry per block (all read)', y + 10, (i) => ['fill-length', i % 8 === 7 ? 0.9 : 0.25]);
      const sel = topBlocks(8, 3);
      row('selected branch: top blocks read in full (+ first & local)', y + 50, (i) => (sel.has(Math.floor(i / 8)) || i < 8 ? ['cell', 0.9] : null));
      row('window branch: last 16 tokens', y + 90, (i) => (i >= 112 ? ['fill-memory', 0.9] : null));
    } else if (mode === 'dsa') {
      row('indexer: scores every token (cheap)', y + 10, (i) => ['fill-length', 0.15 + 0.7 * rel[i]]);
      const top = new Set(rel.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 16).map((x) => x[1]));
      row('main attention: only the top-k tokens', y + 50, (i) => (top.has(i) ? ['cell', 0.95] : null));
      row('KV cache: every token still stored', y + 90, () => ['fill-memory', 0.5]);
    } else if (mode === 'csa') {
      row('cache: one entry per 4 tokens', y + 10, (i) => ['fill-memory', i % 4 === 3 ? 0.9 : 0.15]);
      const sel = topBlocks(4, 6);
      row('indexer picks top-k compressed entries', y + 50, (i) => (sel.has(Math.floor(i / 4)) ? ['cell', 0.9] : null));
      row('plus a short uncompressed window', y + 90, (i) => (i >= 120 ? ['fill-compute', 0.9] : null));
    } else {
      row('cache: one entry per 32 tokens in this toy (128 in V4)', y + 10, (i) => ['fill-memory', i % 32 === 31 ? 0.9 : 0.12]);
      row('read: every compressed entry, densely', y + 50, (i) => (i % 32 === 31 ? ['cell', 0.9] : null));
    }
    const T = 2 ** logT;
    const I = INFO[mode];
    const ny = 190;
    const stat = (label: string, v: number, x: number) => {
      svg.append(s('text', { x, y: ny, class: 'lbl small muted' }, label));
      svg.append(s('text', { x, y: ny + 22, class: 'lbl strong num', style: 'font-size:15px' }, v ? fmtCount(v) : '—'));
      svg.append(s('text', { x, y: ny + 40, class: 'lbl small muted' }, v ? `${((100 * v) / T).toFixed(v / T < 0.01 ? 2 : 1)}% of dense` : ''));
    };
    stat(`KV entries stored / layer`, I.stored(T), x0);
    stat('entries the main attention reads / query', I.read(T), x0 + 240);
    stat('entries a light selector scores / query', I.scored(T), x0 + 500);
    f.readout.textContent = I.note;
  }
  draw();
  return f.root;
}
