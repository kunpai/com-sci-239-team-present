// Scripted browser regression for viewer/index.html. Needs `npm i puppeteer-core` and Google Chrome.
// Usage: node viewer/browser_check.js <project-root> <screenshot-dir> <hostile-page.html>

const puppeteer = require('puppeteer-core');
const path = require('path');
const ROOT = process.argv[2], SHOTS = process.argv[3], EVIL = process.argv[4];
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
let failures = 0;
const check = (name, ok, extra = '') => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  — ' + extra : '')); if (!ok) failures++; };
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1100 });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const url = (h, q = '') => `file://${ROOT}/viewer/index.html${q}${h}`;
  const nav = async (h, q) => { await page.goto(url(h, q), { waitUntil: 'load' }); await page.evaluate(() => document.fonts && document.fonts.ready); };
  const count = (sel) => page.$$eval(sel, (els) => els.length);
  const text = (sel) => page.$eval(sel, (e) => e.textContent);

  await nav('#/matxfer_full2');
  check('overview renders 3 chains', (await count('.chain')) === 3);
  check('dev runs hidden by default (3 pills)', (await count('.run-pill')) === 3);
  await page.click('#opt-dev');
  check('dev runs checkbox reveals all 5 pills', (await count('.run-pill')) === 5);
  await page.click('#opt-dev');

  await nav('#/matxfer_full2/1/timeline/C/0');
  check('deep link opens C trial 1', (await page.$eval('.card[data-key="C-0"]', (e) => e.dataset.open)) === 'true');
  check('failing step shows the immintrin.h compiler error', (await text('.card[data-key="C-0"] .stderr')).includes('immintrin.h'));
  check('C memory block is verbatim incl. double bullet', (await text('.col.C .index-card pre')).startsWith('- - **Baseline'));

  await nav('#/matxfer_full2/1/timeline');
  const headA0 = '.card[data-key="A-0"] .card-head';
  await page.click(headA0);
  check('click expands a card and builds code lazily', (await count('.card[data-key="A-0"] .codeblock')) === 1);
  await page.click(headA0);
  check('click again collapses it', (await page.$eval('.card[data-key="A-1"]', (e) => e.dataset.open)) === 'false');
  await page.click('.card[data-key="A-3"] .card-head');
  await page.click('.card[data-key="A-3"] .seg button:nth-child(2)');
  check('diff view shows +/- summary vs previous attempt', /lines vs\. previous|Identical/.test(await text('.card[data-key="A-3"] .code')));
  const inspect = await page.$('.card[data-key="A-3"] .row-actions button');
  await inspect.click();
  check('inspector drawer opens', (await page.$('#drawer:not([hidden])')) !== null);
  const tabs = await page.$$eval('#drawer-tabs button', (b) => b.map((x) => x.textContent));
  check('inspector has prompt/reply/code/result/critique/json tabs', ['Generation prompt', 'Generation reply', 'Code', 'Result', 'Critique reply', 'JSON'].every((t) => tabs.includes(t)), tabs.join(', '));
  check('prompt tab shows the full prompt', (await text('#drawer-body pre')).length > 1000 && (await text('#drawer-body pre')).includes('Attempt'));
  await page.screenshot({ path: path.join(SHOTS, 'drawer.png') });
  await page.keyboard.press('Escape');
  check('Esc closes the drawer', (await page.$('#drawer[hidden]')) !== null);

  await page.click('#opt-fail');
  check('failures-only hides passing attempts in A', (await count('.col.A .card')) === 0 && (await text('.col.A .rail')).includes('No failing attempts'));
  check('failures-only keeps the failed C trial', (await count('.col.C .card')) === 1);
  await page.click('#opt-fail');

  await page.keyboard.press('m');
  check("'m' opens the memory panel", page.url().includes('/memory'));
  await page.keyboard.press('m');
  check("'m' again returns to the timeline", page.url().includes('/timeline'));
  await page.keyboard.press(']');
  check("']' goes to the next chain", /matxfer_full2\/2\//.test(decodeURIComponent(page.url())));
  await page.keyboard.press('j');
  check("'j' focuses a step", await page.evaluate(() => document.activeElement && document.activeElement.classList.contains('card-head')));
  const before = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.keyboard.press('t');
  check("'t' toggles the theme", (await page.evaluate(() => document.documentElement.dataset.theme)) !== before);

  await nav('#/does-not-exist/9/memory');
  check('bad deep link falls back to a valid page (no error panel)', (await count('.error-panel')) === 0 && (await count('.run-head')) === 1);
  await nav('#/matxfer_generic/3/timeline');
  check('aborted repeat renders its error + empty arms', (await count('.banner')) >= 1 && (await count('.empty-note')) >= 3);
  await nav('#/matxfer_generic/2/memory');
  check('hard-clipped memory is flagged in the memory panel', (await text('.col.A')).includes('hard-clipped'));

  // --- final-review fixes ---
  await nav('#/matxfer_generic');
  check('aborted repeat is labelled in the A summary row', (await text('.sheet tbody tr:first-child')).includes('aborted'));
  const tilesA = await page.$$eval('.tile.A', (els) => els.map((e) => e.textContent));
  check('aborted chain A tile says it aborted before memory was written', tilesA[0].includes('aborted before memory') && !tilesA[0].includes('no memory written'), tilesA[0].slice(0, 120));
  await nav('#/matxfer_generic/1/memory');
  const memA = await text('.col.A');
  check('memory panel blames the abort, not a missing correct candidate', memA.includes('aborted') && !memA.includes('no correct candidate'));
  await nav('#/matxfer_generic');
  const tilesC = await page.$$eval('.tile.C', (els) => els.map((e) => e.textContent));
  check('C tile word count excludes the harness bullet (200, not 201)', tilesC[1].includes('memory shown: 200 words'), tilesC[1].slice(0, 160));
  await nav('#/matxfer_generic/2/memory');
  check('memory panel chip excludes the harness bullet too', (await text('.col.C .mem-meta')).includes('200 words shown'));
  await nav('#/matxfer_full2/1/memory');
  check('carried-in A memory is not styled as a lesson; the new lesson is', (await count('.col.C .index-card.lesson')) === 1 && (await text('.col.C')).includes('carried in from arm A'));
  await nav('#/does-not-exist/1/timeline/C/0');
  check('unknown run in a deep link shows a notice and no stray chain', (await text('#app')).includes('does-not-exist') && (await text('#app')).includes('not found') && (await count('.crumbs')) === 0);
  await nav('#/matxfer_full');
  check('pre-fix harness run is marked', (await text('.run-head')).includes('older harness'));
  await nav('#/matxfer_full2');
  check('post-fix run is not marked', !(await text('.run-head')).includes('older harness'));

  await nav('#/matxfer_full2', '?theme=light');
  await page.screenshot({ path: path.join(SHOTS, 'light_overview.png') });
  await nav('#/matxfer_full2/1/memory', '?theme=light');
  await page.screenshot({ path: path.join(SHOTS, 'light_memory.png') });
  await page.setViewport({ width: 390, height: 900 });
  await nav('#/matxfer_full2/1/timeline', '?theme=light');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2);
  check('no horizontal overflow at phone width', !overflow);
  await page.screenshot({ path: path.join(SHOTS, 'phone.png') });
  await page.setViewport({ width: 1500, height: 1100 });

  // hostile text
  const evil = await browser.newPage();
  const evilErrors = [];
  evil.on('pageerror', (e) => evilErrors.push(e.message));
  await evil.goto('file://' + EVIL + '#/matxfer_evil/1/timeline/B/0', { waitUntil: 'load' });
  check('hostile </script><img onerror> text did not execute', (await evil.evaluate(() => window.__pwned)) === undefined);
  check('hostile text is not parsed into the DOM as an element', (await evil.$$eval('img[src="x"]', (e) => e.length)) === 0);
  check('hostile text is shown literally', (await evil.$eval('.card[data-key="B-0"] .stderr', (e) => e.textContent)).includes('</script><img'));
  check('hostile page has no script errors', evilErrors.length === 0, evilErrors.join(' | '));

  check('no console/page errors on the real-data page', errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
  console.log(failures ? `\n${failures} CHECK(S) FAILED` : '\nALL CHECKS PASSED');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('SCRIPT ERROR', e); process.exit(2); });
