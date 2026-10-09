// Build-time audit of the date ledger and of every hand-written text that could
// contradict it. Fails the build on: unsorted/invalid dates, missing sources,
// stories without ledger entries (or vice versa), and any year mentioned in
// content/README that the ledger does not know about.
import { readFileSync, readdirSync } from 'node:fs';

const ledger = JSON.parse(readFileSync('src/data/ledger.json', 'utf8'));
const errors = [];
const DATE = /^\d{4}(-\d{2}(-\d{2})?)?$/;
const known = new Set();
const ids = new Set();

const knownMonths = new Set();
const ym = (d) => d.slice(0, 7);
for (const e of ledger.entries) {
  knownMonths.add(ym(e.date));
  for (const r of [...e.precursors, ...e.adoption, ...(e.firstUse ? [e.firstUse] : [])]) knownMonths.add(ym(r.date));
}
const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

for (const e of ledger.entries) {
  if (ids.has(e.id)) errors.push(`duplicate id ${e.id}`);
  ids.add(e.id);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(e.date)) errors.push(`${e.id}: entry date must be YYYY-MM-DD, got ${e.date}`);
  if (!e.source?.url?.startsWith('https://')) errors.push(`${e.id}: primary source URL missing`);
  if (!e.dateEvent) errors.push(`${e.id}: dateEvent missing`);
  for (const k of ['compute', 'memory', 'position', 'fidelity']) if (Math.abs(e.effects?.[k] ?? 9) > 2) errors.push(`${e.id}: bad effect ${k}`);
  known.add(e.date.slice(0, 4));
  for (const r of [...e.precursors, ...e.adoption, ...(e.firstUse ? [e.firstUse] : [])]) {
    if (!DATE.test(r.date)) errors.push(`${e.id}: bad related date ${r.date}`);
    known.add(r.date.slice(0, 4));
  }
  if (e.firstUse && e.firstUse.date < e.date) errors.push(`${e.id}: first use before publication`);
  for (const p of e.precursors) if (p.date > e.date) errors.push(`${e.id}: precursor "${p.title}" is dated after the entry`);
}

// Stories must match ledger ids one-to-one.
const stories = readFileSync('src/content/entries.ts', 'utf8');
const storyIds = [...stories.matchAll(/^  '?([a-z-]+)'?: \{$/gm)].map((m) => m[1]);
for (const id of storyIds) if (!ids.has(id)) errors.push(`story ${id} has no ledger entry`);
for (const id of ids) if (!storyIds.includes(id)) errors.push(`ledger entry ${id} has no story`);

// Any 4-digit year in hand-written text must be a year the ledger knows.
const widgetFiles = readdirSync('src/widgets').map((f) => `src/widgets/${f}`);
for (const f of ['src/content/entries.ts', 'src/content/eras.ts', 'src/sections.ts', 'src/main.ts', 'README.md', ...widgetFiles]) {
  let text;
  try { text = readFileSync(f, 'utf8'); } catch { continue; }
  for (const m of text.matchAll(/(?<![\d.:/-])\b(19[5-9]\d|20[0-3]\d)\b(?!\.\d)/g)) {
    if (!known.has(m[1])) errors.push(`${f}: mentions year ${m[1]}, which no ledger date supports`);
  }
  // "Mar 2021", "26 April 2026": the month must match some ledger date too
  for (const m of text.matchAll(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (19\d\d|20\d\d)\b/g)) {
    const key = `${m[2]}-${String(MONTHS.indexOf(m[1].toLowerCase()) + 1).padStart(2, '0')}`;
    if (!knownMonths.has(key)) errors.push(`${f}: mentions "${m[0]}", which matches no ledger date`);
  }
}

if (errors.length) {
  console.error('Ledger check FAILED:\n  ' + errors.join('\n  '));
  process.exit(1);
}
console.log(`Ledger check OK: ${ledger.entries.length} entries, ${storyIds.length} stories, years ${[...known].sort().join(' ')}`);
