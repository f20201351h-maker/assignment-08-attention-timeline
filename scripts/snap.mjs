// Quick element screenshot for visual review: node scripts/snap.mjs <selector> <width> <out.png> [waitMs] [scheme]
import puppeteer from 'puppeteer-core';
const [sel, width = '1280', out = 'snap.png', wait = '4000', scheme = 'light'] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const page = await browser.newPage();
const errs = [];
page.on('pageerror', (e) => errs.push(String(e)));
page.on('console', (m) => m.type() === 'error' && errs.push(m.text()));
await page.setViewport({ width: Number(width), height: 900, deviceScaleFactor: 1, isMobile: Number(width) < 600 });
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
await page.goto(process.env.URL ?? 'http://localhost:5173/', { waitUntil: 'networkidle0' });
await page.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: 'start', behavior: 'instant' }), sel);
await new Promise((r) => setTimeout(r, Number(wait)));
const el = await page.$(sel);
await el.screenshot({ path: out });
console.log(errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no errors');
await browser.close();
