// Tiny DOM helpers. No framework: each widget is a function that builds its own
// DOM once and re-renders its SVG on input.

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | EventListener | undefined | null>;

function apply(el: Element, attrs: Attrs | null) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'html') (el as HTMLElement).innerHTML = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
}

function append(el: Element, children: Child[]) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs | null = null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  apply(el, attrs);
  append(el, children);
  return el;
}

const NS = 'http://www.w3.org/2000/svg';
export function s(tag: string, attrs: Attrs | null = null, ...children: Child[]): SVGElement {
  const el = document.createElementNS(NS, tag);
  apply(el, attrs);
  append(el, children);
  return el as SVGElement;
}

export const svgRoot = (w: number, hgt: number, label: string) =>
  s('svg', { viewBox: `0 0 ${w} ${hgt}`, role: 'img', 'aria-label': label, class: 'stage-svg' }) as SVGSVGElement;

let uid = 0;
const nextId = (p: string) => `${p}-${++uid}`;

export interface SliderOpts { label: string; min: number; max: number; step?: number; value: number; fmt?: (v: number) => string; onInput: (v: number) => void }
export function slider(o: SliderOpts) {
  const id = nextId('sl');
  const out = h('output', { for: id, class: 'ctl-val' }, (o.fmt ?? String)(o.value));
  const input = h('input', { id, type: 'range', min: o.min, max: o.max, step: o.step ?? 1, value: o.value });
  input.addEventListener('input', () => {
    const v = Number(input.value);
    out.textContent = (o.fmt ?? String)(v);
    o.onInput(v);
  });
  const wrap = h('div', { class: 'ctl ctl-slider' }, h('label', { for: id }, o.label), input, out);
  return Object.assign(wrap, {
    set(v: number) { input.value = String(v); out.textContent = (o.fmt ?? String)(v); },
  });
}

export interface SegOpts<T extends string> { label: string; options: [T, string][]; value: T; onChange: (v: T) => void }
export function seg<T extends string>(o: SegOpts<T>) {
  const name = nextId('seg');
  const group = h('div', { class: 'ctl ctl-seg', role: 'radiogroup', 'aria-label': o.label });
  group.append(h('span', { class: 'ctl-label' }, o.label));
  const box = h('div', { class: 'seg' });
  const inputs: HTMLInputElement[] = [];
  for (const [val, text] of o.options) {
    const id = nextId('opt');
    const input = h('input', { type: 'radio', name, id, value: val, checked: val === o.value });
    input.addEventListener('change', () => input.checked && o.onChange(val));
    inputs.push(input);
    box.append(input, h('label', { for: id }, text));
  }
  group.append(box);
  return Object.assign(group, {
    set(v: T) { inputs.forEach((i) => (i.checked = i.value === v)); },
  });
}

export function toggle(label: string, checked: boolean, onChange: (v: boolean) => void) {
  const id = nextId('tg');
  const input = h('input', { type: 'checkbox', id, role: 'switch', checked });
  input.addEventListener('change', () => onChange(input.checked));
  return h('div', { class: 'ctl ctl-toggle' }, input, h('label', { for: id }, label));
}

export function button(label: string, onClick: () => void, cls = '') {
  return h('button', { type: 'button', class: `btn ${cls}`, onclick: () => onClick() }, label);
}

/** Standard widget frame: title, controls row, stage, live readout, caption. */
export function frame(title: string, caption: string) {
  const controls = h('div', { class: 'w-controls' });
  const stage = h('div', { class: 'w-stage' });
  const readout = h('div', { class: 'w-readout', 'aria-live': 'polite' });
  const root = h('figure', { class: 'widget' },
    h('div', { class: 'w-title' }, h('span', { class: 'w-dot', 'aria-hidden': 'true' }), title),
    controls, stage, readout,
    h('figcaption', { class: 'w-caption', html: caption }));
  return { root, controls, stage, readout };
}

export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Cheap animation loop that stops itself; jumps to the end under reduced motion. */
export function animate(duration: number, step: (t: number) => void, done?: () => void) {
  if (reducedMotion()) { step(1); done?.(); return () => {}; }
  let raf = 0;
  const t0 = performance.now();
  const tick = (now: number) => {
    const t = Math.min(1, (now - t0) / duration);
    step(t);
    if (t < 1) raf = requestAnimationFrame(tick); else done?.();
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

export const fmt = (x: number, d = 2) => (Math.abs(x) < 1e-9 ? '0' : x.toFixed(d));

/** A ▶ button that sweeps a slider from min to max, calling the same update the slider would. */
export function sweepButton(label: string, ctl: { set(v: number): void }, min: number, max: number, update: (v: number) => void, msPerStep = 120) {
  let timer = 0;
  const btn = button(label, () => {
    clearInterval(timer);
    if (reducedMotion()) { ctl.set(max); update(max); return; }
    let v = min;
    ctl.set(v); update(v);
    timer = window.setInterval(() => { v++; ctl.set(v); update(v); if (v >= max) clearInterval(timer); }, msPerStep);
  }, 'ghost');
  return btn;
}
