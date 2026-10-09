// Visual + behavioural QA with the locally installed Chrome (no browser download).
// Usage: node scripts/qa.mjs [url] [outDir]
// Checks: console errors, horizontal overflow, timeline date order, every widget
// mounted, broken in-page anchors; saves screenshots at several viewports.
import puppeteer from 'puppeteer-core';
import { mkdirSync, readFileSync } from 'node:fs';

const LEDGER = JSON.parse(readFileSync('src/data/ledger.json', 'utf8')).entries;

const url = process.argv[2] ?? 'http://localhost:5173/';
const out = process.argv[3] ?? 'qa-shots';
mkdirSync(out, { recursive: true });
const CHROME = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';

const viewports = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'laptop', width: 1180, height: 760 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'mobile', width: 375, height: 760, isMobile: true, hasTouch: true },
];

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
let failures = 0;
const fail = (m) => { failures++; console.log('  FAIL', m); };

for (const vp of viewports) {
  for (const scheme of vp.name === 'desktop' || vp.name === 'mobile' ? ['light', 'dark'] : ['light']) {
    const page = await browser.newPage();
    const errors = [];
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.setViewport({ width: vp.width, height: vp.height, isMobile: !!vp.isMobile, hasTouch: !!vp.hasTouch, deviceScaleFactor: 1 });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
    await page.goto(url, { waitUntil: 'networkidle0' });
    // scroll through so every lazy widget mounts
    const total = await page.evaluate(() => document.documentElement.scrollHeight);
    for (let y = 0; y < total; y += vp.height * 0.8) { await page.evaluate((yy) => window.scrollTo(0, yy), y); await new Promise((r) => setTimeout(r, 40)); }
    await new Promise((r) => setTimeout(r, 300));
    const report = await page.evaluate(() => {
      const doc = document.documentElement;
      const overflowX = doc.scrollWidth - doc.clientWidth;
      const wide = [...document.querySelectorAll('body *')].filter((el) => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && (r.right > doc.clientWidth + 1 || r.left < -1) && !el.closest('.map-scroll, .tbl-wrap, .eq, .eq-block, .seg');
      }).slice(0, 8).map((el) => `${el.tagName.toLowerCase()}.${el.className?.baseVal ?? el.className}`);
      const dates = [...document.querySelectorAll('article.entry time')].map((t) => t.getAttribute('datetime'));
      const rendered = [...document.querySelectorAll('article.entry')].map((a) => [a.id, a.querySelector('time').getAttribute('datetime'), a.querySelector('.prov summary span').textContent]);
      const tableDates = [...document.querySelectorAll('#sources tbody tr')].map((tr) => [tr.querySelector('a').getAttribute('href').slice(1), tr.querySelector('td.date').textContent]);
      const sorted = dates.every((d, i) => i === 0 || dates[i - 1] <= d);
      const placeholders = document.querySelectorAll('.lazy-ph').length;
      const widgets = document.querySelectorAll('figure.widget').length;
      const badAnchors = [...document.querySelectorAll('a[href^="#"]')].map((a) => a.getAttribute('href')).filter((h) => h.length > 1 && !document.getElementById(h.slice(1)));
      const tiny = [...document.querySelectorAll('svg text')].filter((t) => {
        const r = t.getBoundingClientRect();
        return r.height > 0 && r.height < 7.5;
      }).length;
      return { overflowX, wide, dates: dates.length, rendered, tableDates, sorted, placeholders, widgets, badAnchors, tiny, height: doc.scrollHeight };
    });
    console.log(`${vp.name}/${scheme}: height ${report.height}px, ${report.dates} entries, ${report.widgets} widgets, tiny svg labels ${report.tiny}`);
    if (report.overflowX > 0) fail(`horizontal overflow ${report.overflowX}px: ${report.wide.join(', ')}`);
    if (!report.sorted) fail('timeline not in date order');
    // rendered page vs ledger: every card's date, its provenance summary, and the sources table
    const M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const human = (d) => { const [y, m, day] = d.split('-'); return `${Number(day)} ${M[m - 1]} ${y}`; };
    for (const e of LEDGER) {
      const r = report.rendered.find((x) => x[0] === e.id);
      if (!r) { fail(`card ${e.id} missing from page`); continue; }
      if (r[1] !== e.date) fail(`card ${e.id} shows ${r[1]}, ledger says ${e.date}`);
      if (!r[2].includes(human(e.date))) fail(`provenance of ${e.id} shows "${r[2]}"`);
      const t = report.tableDates.find((x) => x[0] === e.id);
      if (!t || t[1] !== human(e.date)) fail(`sources table row for ${e.id} shows ${t?.[1]}`);
    }
    if (report.placeholders) fail(`${report.placeholders} widgets never mounted`);
    if (report.badAnchors.length) fail(`broken anchors ${report.badAnchors.join(' ')}`);
    if (errors.length) fail(`console errors: ${errors.slice(0, 3).join(' | ')}`);
    // screenshots: top, a few sections
    for (const [tag, sel] of [['hero', '#top'], ['foundation', '#foundation'], ['map', '#map'], ['rope', '#rope'], ['yarn', '#yarn'], ['csa', '#csa'], ['shows', '#what-it-shows']]) {
      await page.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: 'start', behavior: 'instant' }), sel);
      await new Promise((r) => setTimeout(r, 150));
      await page.screenshot({ path: `${out}/${vp.name}-${scheme}-${tag}.png` });
    }
    await page.close();
  }
}
// reduced motion: the engine must render its final state without animating
{
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.querySelector('.engine')?.scrollIntoView({ behavior: 'instant' }));
  await new Promise((r) => setTimeout(r, 800));
  const tag = await page.$eval('.engine-tag', (el) => el.textContent);
  const readout = await page.$eval('.engine .w-readout', (el) => el.textContent);
  console.log(`reduced-motion: engine stage "${tag}"`);
  if (!tag?.includes('Update') || !readout) fail('engine does not reach its final state under reduced motion');
  // clicking a token re-runs instantly
  await page.evaluate(() => [...document.querySelectorAll('.etok')].find((b) => b.textContent === 'river')?.click());
  await new Promise((r) => setTimeout(r, 300));
  const q = await page.$eval('.etok.q', (el) => el.textContent);
  if (q !== 'river') fail('clicking a token did not make it the query');
  await page.close();
}
await browser.close();
console.log(failures ? `QA: ${failures} failure(s)` : 'QA: all checks passed');
process.exit(failures ? 1 : 0);
