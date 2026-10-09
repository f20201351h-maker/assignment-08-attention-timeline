import './styles.css';
import { h, s } from './lib/dom.ts';
import { ENTRIES, LEDGER, PRESSURES, byId, fmtDate, monthYear, year, fyear, yearsBetween, type Entry, type Pressure } from './lib/ledger.ts';
import { STORIES } from './content/entries.ts';
import { ERAS } from './content/eras.ts';
import { attentionWidget } from './widgets/attention.ts';
import { attentionEngine } from './widgets/engine.ts';
import { billsWidget } from './widgets/bills.ts';
import { buildSections } from './sections.ts';

const app = document.getElementById('app')!;
const P_COLOR: Record<Pressure, string> = { compute: 'var(--compute)', memory: 'var(--memory)', position: 'var(--length)', fidelity: 'var(--exact)' };

// ---------- theme ----------
function themeButton() {
  const btn = h('button', { class: 'theme-btn', type: 'button', 'aria-label': 'Toggle colour theme' }, '◐');
  const saved = (() => { try { return localStorage.getItem('theme'); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  btn.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch { /* private mode */ }
  });
  return btn;
}

// ---------- hero ----------
function heroArt() {
  const words = ['The', 'animal', 'didn’t', 'cross', 'the', 'street', 'because', 'it', 'was', 'tired'];
  // illustrative weights: what “it” might attend to (hand-written, not a model output)
  const W: Record<number, number[]> = {
    7: [0.02, 0.55, 0.02, 0.04, 0.02, 0.2, 0.05, 0.1],
    9: [0.01, 0.35, 0.02, 0.03, 0.01, 0.05, 0.03, 0.4, 0.08, 0.02],
    5: [0.05, 0.1, 0.05, 0.4, 0.3, 0.1],
    3: [0.1, 0.5, 0.3, 0.1],
  };
  const svg = s('svg', { viewBox: '0 0 560 260', role: 'img', 'aria-label': 'A sentence with attention arcs from one word to earlier words' }) as SVGSVGElement;
  const xs = words.map((_, i) => 34 + i * 54.5);
  let q = 7;
  function draw() {
    svg.replaceChildren();
    const w = W[q] ?? [];
    w.forEach((v, j) => {
      if (j === q) return;
      const x1 = xs[q], x2 = xs[j], mid = (x1 + x2) / 2, hgt = 30 + Math.abs(x1 - x2) * 0.42;
      svg.append(s('path', { d: `M${x1},190 Q${mid},${190 - hgt * 1.6} ${x2},190`, class: 'arc', 'stroke-width': 1 + 12 * v, style: `opacity:${0.18 + 0.82 * v}` }));
    });
    words.forEach((t, i) => {
      const g = s('text', { x: xs[i], y: 216, class: 'htok' + (i === q ? ' q' : ''), tabindex: W[i] ? 0 : -1, role: W[i] ? 'button' : null });
      g.textContent = t;
      if (W[i]) {
        g.addEventListener('click', () => { q = i; draw(); });
        g.addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter') { q = i; draw(); } });
      }
      svg.append(g);
    });
    svg.append(s('text', { x: 280, y: 250, class: 'hint' }, 'tap “it”, “tired”, “street” or “cross” — illustrative weights'));
  }
  draw();
  // gently cycle the query until the visitor takes over
  const cycle = [7, 9, 5, 3];
  let n = 0;
  const timer = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : window.setInterval(() => { n = (n + 1) % cycle.length; q = cycle[n]; draw(); }, 3200);
  svg.addEventListener('pointerdown', () => clearInterval(timer), { once: true });
  svg.addEventListener('keydown', () => clearInterval(timer), { once: true });
  return h('div', { class: 'hero-art' }, svg);
}

function hero() {
  return h('header', { class: 'hero', id: 'top' },
    h('div', null,
      h('div', { class: 'kicker' }, 'A chronological field guide'),
      h('h1', { html: 'Attention, <em>in the order it happened</em>' }),
      h('p', { class: 'lede' }, 'Vanilla attention was not wrong. It was expensive. Everything after it is somebody looking at that bill and trying to pay less of it — and the order in which they tried shows the field changing its mind.'),
      h('div', { class: 'moods' },
        ...PRESSURES.map((p) => h('span', { class: 'mood' }, h('i', { style: `background:${P_COLOR[p.key]}` }), `${p.label}: ${p.blurb}`))),
      h('div', { class: 'hero-meta' },
        h('span', null, `${ENTRIES.length} mechanisms, ${year(ENTRIES[0].date)}–${year(ENTRIES[ENTRIES.length - 1].date)}`),
        h('span', null, 'every date checked against its primary source'),
        h('a', { href: '#foundation' }, 'Start with one attention head ↓'))),
    heroArt());
}

// ---------- foundation ----------
function foundation() {
  const sec = h('section', { class: 'section', id: 'foundation' });
  sec.append(
    h('div', { class: 'kicker' }, 'Start here'),
    h('h2', null, 'One attention head, before anything changes it'),
    attentionEngine(),
    h('p', null, 'Every token is a vector. Attention is the first place tokens look at each other. Each token makes three projections of itself: a ', h('strong', null, 'query'), ' (what am I looking for?), a ', h('strong', null, 'key'), ' (what do I contain?) and a ', h('strong', null, 'value'), ' (what do I hand over if chosen?). Why three: the query–key pair decides who attends to whom, and the value carries what is transported — tie them together and the model loses the ability to point asymmetrically at another token.'),
    h('ol', { class: 'steps' },
      h('li', null, h('strong', null, 'Score. '), 'Dot every query with every key: one number per pair.'),
      h('li', null, h('strong', null, 'Scale. '), 'Divide by √dₖ so wide heads do not push softmax into saturation.'),
      h('li', null, h('strong', null, 'Mask. '), 'In a decoder, set every future position to −∞ so it gets exactly zero weight.'),
      h('li', null, h('strong', null, 'Softmax. '), 'Turn each query’s row of scores into positive weights that sum to one. This is where keys compete.'),
      h('li', null, h('strong', null, 'Weighted sum. '), 'Average the values with those weights. That is the token’s new vector.')),
    h('div', { class: 'eq-block', html: 'Attention(Q, K, V) = softmax( QKᵀ / √d<sub>k</sub> + M ) V' }),
    h('p', null, 'The engine above shows one query at a time. The same computation as a matrix — every query against every key at once, the way a GPU actually does it during training:'),
    attentionWidget(),
    h('h3', null, 'Two facts that create the rest of the timeline'),
    h('p', null, h('strong', null, 'Every token is compared with every other token. '), 'Six tokens give 36 scores; a hundred thousand give ten billion, per head, per layer. That is the ', h('strong', { style: 'color:var(--compute)' }, 'compute bill'), '. And to generate the next token without recomputing the past, the model keeps every earlier key and value — the KV cache — one copy per active conversation. That is the ', h('strong', { style: 'color:var(--memory)' }, 'memory bill'), '. They grow differently, and much of the history is the field switching which one it is fighting.'),
    billsWidget(),
    h('p', null, h('strong', null, 'Attention has no built-in sense of order. '), 'The scores come from content alone, so something must tell the model where each token sits — and whatever does must keep working at lengths the model never trained on. That is the ', h('strong', { style: 'color:var(--length)' }, 'position bill'), '. Every shortcut against the first two bills risks the fourth thing we care about: ', h('strong', { style: 'color:var(--exact)' }, 'fidelity'), ', exact access to any earlier token.'),
  );
  return sec;
}

// ---------- overview map ----------
function overviewMap() {
  const sec = h('section', { class: 'section', id: 'map' });
  sec.append(
    h('div', { class: 'kicker' }, 'The whole history on one axis'),
    h('h2', null, 'Which bill was the field paying, and when?'),
    h('p', null, 'Each dot is a mechanism, placed at its verified date and in the row of the bill it mainly attacks. A dashed tail runs to the first notable released model that used it. Long tails are ideas that waited for their bill to become the one that hurt. Click any dot to jump to it.'),
  );
  const x0 = 2013.4, x1 = 2026.7, W = 1100, L = 120, R = 20;
  const lanes: Pressure[] = ['fidelity', 'compute', 'memory', 'position'];
  const laneH = 116, top = 30;
  const X = (d: string) => L + ((fyear(d) - x0) / (x1 - x0)) * (W - L - R);
  const svg = s('svg', { viewBox: `0 0 ${W} ${top + lanes.length * laneH + 40}`, role: 'img', 'aria-label': 'Timeline of all mechanisms by date and by the bill they target' });
  for (let y = 2014; y <= 2026; y++) {
    const x = X(`${y}-01-01`);
    svg.append(s('line', { x1: x, x2: x, y1: top - 6, y2: top + lanes.length * laneH, class: 'grid' }));
    svg.append(s('text', { x: x + 2, y: top + lanes.length * laneH + 18, class: 'yr', 'text-anchor': 'start' }, String(y)));
  }
  lanes.forEach((p, li) => {
    const y = top + li * laneH;
    svg.append(s('rect', { x: L - 6, y: y + 4, width: W - L - R + 12, height: laneH - 8, rx: 10, class: 'fill-bg2', style: 'opacity:.55' }));
    svg.append(s('text', { x: 8, y: y + laneH / 2 + 4, class: 'lane', style: `fill:${P_COLOR[p]}` }, PRESSURES.find((q) => q.key === p)!.label));
    const items = ENTRIES.filter((e) => e.primary === p);
    const rowEnd: number[] = [-1e9, -1e9, -1e9, -1e9];
    items.forEach((e) => {
      const x = X(e.date);
      let row = rowEnd.findIndex((end) => x - end > 6);
      if (row < 0) row = rowEnd.indexOf(Math.min(...rowEnd));
      const label = e.short;
      const flip = x + label.length * 6.3 + 12 > W - 4; // label to the left near the right edge
      rowEnd[row] = flip ? x + 10 : x + label.length * 6.3 + 14;
      const cy = y + 20 + row * 25;
      if (e.firstUse && yearsBetween(e.date, e.firstUse.date) > 0.08) {
        svg.append(s('line', { x1: x, x2: X(e.firstUse.date), y1: cy, y2: cy, class: 'lag', style: `stroke:${P_COLOR[p]};opacity:.55` }));
        svg.append(s('circle', { cx: X(e.firstUse.date), cy, r: 3, style: `fill:${P_COLOR[p]};opacity:.7` }));
      }
      const g = s('a', { href: `#${e.id}`, class: 'dot', 'aria-label': `${e.name}, ${fmtDate(e.date)}` });
      g.append(s('circle', { cx: x, cy, r: e.tier === 'milestone' ? 6.5 : 4.5, style: `fill:${P_COLOR[p]}` }));
      g.append(s('text', { x: flip ? x - 9 : x + 9, y: cy + 4, class: 'mlab', 'text-anchor': flip ? 'end' : 'start' }, label));
      g.append(s('title', null, `${e.name} — ${fmtDate(e.date)}${e.firstUse ? `\nfirst notable use: ${e.firstUse.model}, ${monthYear(e.firstUse.date)}` : ''}`));
      svg.append(g);
    });
  });
  sec.append(h('div', { class: 'map' }, h('div', { class: 'map-scroll' }, svg)),
    h('div', { class: 'legend' },
      ...PRESSURES.map((p) => h('span', null, h('i', { style: `background:${P_COLOR[p.key]}` }), p.label)),
      h('span', null, h('i', { class: 'lagkey' }), 'wait until first notable use'),
      h('span', null, 'big dot = milestone · small = connective step')));
  return sec;
}

// ---------- timeline ----------
function gauge() {
  const rows = PRESSURES.map((p) => {
    const fill = h('div', { class: 'gauge-fill', style: `background:${P_COLOR[p.key]}` });
    const row = h('div', { class: 'gauge-row' }, h('span', null, p.label), h('div', { class: 'gauge-track' }, fill));
    return { p: p.key, row, fill };
  });
  const el = h('div', { class: 'gauge', 'aria-hidden': 'true' }, ...rows.map((r) => r.row));
  return {
    el,
    set(e: Entry) {
      rows.forEach(({ p, row, fill }) => {
        const v = e.effects[p];
        // centre-anchored diverging bar: right = relieves the bill, left = pays into it
        const w = Math.abs(v) * 25;
        fill.style.left = v >= 0 ? '50%' : `${50 - w}%`;
        fill.style.width = `${w}%`;
        fill.style.opacity = v < 0 ? '0.45' : '1';
        row.classList.toggle('hit', v !== 0);
      });
    },
  };
}

function entryCard(e: Entry) {
  const st = STORIES[e.id];
  if (!st) throw new Error(`missing story for ${e.id}`);
  const art = h('article', { class: 'entry', id: e.id, 'data-id': e.id, style: `--dotc:${P_COLOR[e.primary]}` });
  const meta = h('div', { class: 'entry-meta' },
    h('time', { datetime: e.date }, fmtDate(e.date)),
    h('span', { class: 'chip' }, e.kind),
    e.tier === 'connective' ? h('span', { class: 'chip minor' }, 'connective step') : null,
    e.inClassList ? null : h('span', { class: 'chip added', title: 'Not in the core list I started from — added for the story' }, 'added for the story'));
  const roots = e.precursors.filter((p) => p.date < e.date);
  art.append(
    h('div', { class: 'entry-head' }, meta,
      h('h3', null, e.name),
      h('p', { class: 'oneliner' }, st.oneliner),
      roots.length ? h('p', { class: 'muted', style: 'font:500 .82rem/1.5 var(--sans)' }, `Earlier roots: ${roots.map((r) => `${monthYear(r.date)} — ${r.title.split(' — ')[0]}`).join('; ')}.`) : null),
    h('h4', null, 'The moment'),
    h('div', { class: 'prose', html: st.moment }),
    h('h4', null, 'What changed'),
    h('div', { class: 'prose', html: st.change }));
  if (st.widget) {
    const slot = h('div', { class: 'lazy-ph widget' }, 'Loading interactive…');
    art.append(slot);
    lazy(slot, st.widget);
  }
  art.append(
    h('div', { class: 'tradeoffs' },
      h('div', { class: 'buys' }, h('h5', null, 'What it buys'), h('ul', null, ...st.buys.map((b) => h('li', { html: b })))),
      h('div', { class: 'costs' }, h('h5', null, 'What it costs'), h('ul', null, ...st.costs.map((c) => h('li', { html: c }))))),
    h('div', { class: 'choose', html: `<strong>When would I pick it?</strong> ${st.choose}` }),
    st.next.id ? h('a', { class: 'next-link', href: `#${st.next.id}` }, `${st.next.text} → ${byId(st.next.id).short}, ${monthYear(byId(st.next.id).date)}`) : h('p', { class: 'next-link' }, st.next.text),
    provenance(e));
  return art;
}

function provenance(e: Entry) {
  const list = (items: { title?: string; what?: string; date: string; url: string; note?: string }[]) =>
    items.length ? h('ul', null, ...items.map((p) => h('li', null, p.url ? h('a', { href: p.url, target: '_blank', rel: 'noopener' }, p.title ?? p.what ?? '') : (p.title ?? p.what ?? ''), ` (${fmtDate(p.date)})`, p.note ? h('span', { class: 'muted' }, ` — ${p.note}`) : null))) : h('span', { class: 'muted' }, 'none recorded');
  return h('details', { class: 'prov' },
    h('summary', null, h('span', null, `Date & sources: ${fmtDate(e.date)}`), h('span', { class: 'muted mono' }, e.source.id)),
    h('div', { class: 'prov-body' },
      h('dl', null,
        h('dt', null, 'What the date is'), h('dd', null, e.dateEvent),
        h('dt', null, 'Primary source'), h('dd', null, h('a', { href: e.source.url, target: '_blank', rel: 'noopener' }, e.source.title), ` — ${e.source.authors}. ${e.source.venue}.`),
        h('dt', null, 'Earlier roots'), h('dd', null, list(e.precursors)),
        h('dt', null, 'Later / related'), h('dd', null, list(e.adoption)),
        e.firstUse ? h('dt', null, 'First notable use') : null,
        e.firstUse ? h('dd', null, h('a', { href: e.firstUse.url, target: '_blank', rel: 'noopener' }, e.firstUse.model), ` (${fmtDate(e.firstUse.date)})`) : null,
        e.notes ? h('dt', null, 'Notes') : null, e.notes ? h('dd', null, e.notes) : null,
        h('dt', null, 'Verification'), h('dd', null, e.verification))));
}

function lazy(slot: HTMLElement, make: () => HTMLElement) {
  const mount = () => {
    try { slot.replaceWith(make()); } catch (err) {
      slot.textContent = 'This interactive failed to load; the explanation above stands on its own.';
      console.error(err);
    }
  };
  if (!('IntersectionObserver' in window)) return mount();
  const io = new IntersectionObserver((es) => { if (es.some((x) => x.isIntersecting)) { io.disconnect(); mount(); } }, { rootMargin: '600px 0px' });
  io.observe(slot);
}

function timeline() {
  const wrap = h('section', { id: 'timeline' });
  const intro = h('div', { class: 'section' },
    h('div', { class: 'kicker' }, 'The timeline'),
    h('h2', null, `${ENTRIES.length} mechanisms, in the order they appeared`),
    h('p', null, 'Strict date order — not the order they were taught and not grouped by family. Each card says what existed at that moment, what hurt, what changed, what it bought, what it cost, and when you would pick it. Cards marked ', h('span', { class: 'chip added' }, 'added for the story'), ' are mechanisms outside the core list I started from that the story needs. Every date opens to its primary source.'));
  const grid = h('div', { class: 'timeline' });
  const g = gauge();
  const ryear = h('div', { class: 'rail-year' }, year(ENTRIES[0].date));
  const rmonth = h('div', { class: 'rail-month' }, '');
  const rname = h('div', { class: 'rail-name' }, '');
  const ticks = h('nav', { class: 'rail-ticks', 'aria-label': 'Timeline entries' });
  const rail = h('aside', { class: 'rail' }, rmonth, ryear, rname, g.el,
    h('div', { class: 'rail-note' }, 'Bars right of centre: relieves that bill. Left: pays into it. Judged against the default of the moment.'), ticks);
  const stream = h('div', { class: 'stream' });
  const mini = h('div', { class: 'minirail', 'aria-hidden': 'true' }, h('span', { class: 'y' }), h('span', { class: 'n' }), h('span', { class: 'pd' }, ...PRESSURES.map(() => h('i'))));
  const links = new Map<string, HTMLAnchorElement>();
  ENTRIES.forEach((e) => {
    const era = ERAS.find((x) => x.first === e.id);
    if (era) {
      const members = ENTRIES.slice(ENTRIES.indexOf(e), ENTRIES.findIndex((x, i) => i > ENTRIES.indexOf(e) && ERAS.some((r) => r.first === x.id)) >>> 0);
      const span = `${year(members[0].date)}${year(members[members.length - 1].date) !== year(members[0].date) ? '–' + year(members[members.length - 1].date) : ''}`;
      stream.append(h('div', { class: 'era', id: era.id }, h('div', { class: 'span' }, span), h('h2', null, era.title), h('p', null, era.body)));
      ticks.append(h('a', { href: `#${era.id}`, class: 'era-link' }, `${span} · ${era.title}`));
    }
    stream.append(entryCard(e));
    const a = h('a', { href: `#${e.id}` }, `${e.short}`);
    links.set(e.id, a);
    ticks.append(a);
  });
  grid.append(rail, stream);
  wrap.append(intro, mini, grid);

  let current = '';
  const setCurrent = (id: string) => {
    if (id === current) return;
    current = id;
    const e = byId(id);
    ryear.textContent = year(e.date);
    rmonth.textContent = monthYear(e.date).split(' ')[0];
    rname.textContent = e.short;
    g.set(e);
    links.forEach((a, k) => a.classList.toggle('on', k === id));
    links.get(id)?.scrollIntoView({ block: 'nearest' });
    (mini.children[0] as HTMLElement).textContent = monthYear(e.date);
    (mini.children[1] as HTMLElement).textContent = e.short;
    PRESSURES.forEach((p, i) => { ((mini.children[2] as HTMLElement).children[i] as HTMLElement).style.background = e.effects[p.key] > 0 ? P_COLOR[p.key] : 'var(--line)'; });
  };
  const io = new IntersectionObserver((es) => {
    const vis = es.filter((x) => x.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (vis[0]) setCurrent((vis[0].target as HTMLElement).dataset.id!);
  }, { rootMargin: '-35% 0px -55% 0px' });
  requestAnimationFrame(() => wrap.querySelectorAll('.entry').forEach((el) => io.observe(el)));
  setCurrent(ENTRIES[0].id);
  return wrap;
}

// ---------- assemble ----------
const progress = h('div', { class: 'progress' });
app.append(
  h('a', { class: 'skip', href: '#timeline' }, 'Skip to the timeline'),
  h('div', { class: 'topbar' },
    h('a', { class: 'brand', href: '#top' }, 'Attention, in order'),
    h('nav', { 'aria-label': 'Sections' },
      h('a', { href: '#foundation' }, 'Foundation'), h('a', { href: '#map' }, 'Map'), h('a', { href: '#timeline' }, 'Timeline'),
      h('a', { href: '#what-it-shows' }, 'What it shows'), h('a', { href: '#sources' }, 'Sources')),
    themeButton(), progress),
  h('main', null, hero(), foundation(), h('hr', { class: 'section-rule' }), overviewMap(), timeline(), ...buildSections()),
);
window.addEventListener('scroll', () => {
  const max = document.documentElement.scrollHeight - innerHeight;
  progress.style.width = `${(100 * scrollY) / Math.max(1, max)}%`;
}, { passive: true });

export { LEDGER };
