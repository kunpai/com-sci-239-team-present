// Scripted browser regression for viewer/index.html. Needs `npm i puppeteer-core` and Google Chrome.
// Usage: node viewer/browser_check.js <project-root> <screenshot-dir> <hostile-page.html>

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
const ROOT = process.argv[2], SHOTS = process.argv[3], EVIL = process.argv[4];
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
let failures = 0;
const check = (name, ok, extra = '') => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  — ' + extra : '')); if (!ok) failures++; };
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--allow-file-access-from-files'] });
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'deny' });  // the download tests click real links; do not write files or block close()
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1100 });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  const url = (h, q = '') => `file://${ROOT}/viewer/index.html${q}${h}`;
  const nav = async (h, q) => { await page.goto(url(h, q), { waitUntil: 'load' }); await page.evaluate(() => document.fonts && document.fonts.ready); };
  const count = (sel) => page.$$eval(sel, (els) => els.length);
  const clickText = (label) => page.evaluate((l) => [...document.querySelectorAll('.tx-bar button')].find((b) => b.textContent === l).click(), label);
  const text = (sel) => page.$eval(sel, (e) => e.textContent);

  const runDirs = fs.readdirSync(path.join(ROOT, 'traces')).filter((d) => ['A', 'B', 'C'].some((a) => fs.existsSync(path.join(ROOT, 'traces', d, a))));
  const nonDev = runDirs.filter((d) => !/calib|quick/.test(d)).length;
  const stepFiles = (run, arm, rep = 1) => { const d = path.join(ROOT, 'traces', run, arm, 'r' + rep); return fs.existsSync(d) ? fs.readdirSync(d).filter((f) => /^\d+-[A-Za-z_]+\.json$/.test(f)).length : 0; };
  await nav('#/matxfer_full2');
  check('overview renders 3 chains', (await count('.chain')) === 3);
  check(`dev runs hidden by default (${nonDev} pills)`, (await count('.run-pill')) === nonDev);
  await page.click('#opt-dev');
  check(`dev runs checkbox reveals all ${runDirs.length} pills`, (await count('.run-pill')) === runDirs.length);
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


  // --- transcript view: every trace step of a chain, in order ---
  const settle = () => new Promise((r) => setTimeout(r, 450));  // search is debounced
  await nav('#/matxfer_full2/1/transcript');
  for (const arm of ['A', 'B', 'C']) {
    check(`transcript shows every trace file for arm ${arm}`, (await count(`.tx-arm.${arm} .card`)) === stepFiles('matxfer_full2', arm), `${await count(`.tx-arm.${arm} .card`)} vs ${stepFiles('matxfer_full2', arm)} files`);
  }
  check('transcript tab is pressed and the failures-only toggle is hidden', (await page.$eval('.tabs button:last-child', (b) => b.getAttribute('aria-pressed'))) === 'true' && (await page.$eval('#opt-fail', (e) => e.parentElement.hidden)) === true);
  check('no turn header prints the literal word null or undefined', !/\bnull\b|undefined/.test(await page.$$eval('.card-head', (els) => els.map((e) => e.textContent).join(' '))));
  const titles = await page.$$eval('.tx-arm.C .card-head .ttl', (els) => els.map((e) => e.textContent));
  check('transcript has the raw lesson call and the memory update in arm C', titles.some((t) => t.includes('lesson (reflection)')) && titles.some((t) => t.includes('memory update')), titles.join(' | ').slice(0, 200));
  await page.click('.card[data-key="T-A-1"] .card-head');
  check('expanding a Claude call shows the full prompt and the reply', (await count('.card[data-key="T-A-1"] .bubble')) === 2 && (await text('.card[data-key="T-A-1"] .bubble pre')).includes('matmul'));
  await clickText('Expand all');
  check('Expand all opens every turn', (await page.$$eval('.card', (els) => els.every((e) => e.dataset.open === 'true'))));
  await clickText('Collapse all');
  check('Collapse all closes every turn', (await page.$$eval('.card', (els) => els.every((e) => e.dataset.open === 'false'))));
  const total = await count('.card');
  await page.evaluate(() => { const b = [...document.querySelectorAll('.tx-bar label input')][0]; b.click(); });
  check('unticking "Claude calls" hides every llm turn', (await count('.kind.llm')) === 0 && (await count('.card')) < total);
  await page.evaluate(() => { const b = [...document.querySelectorAll('.tx-bar label input')][0]; b.click(); });
  await page.type('.tx-bar input[type="search"]', 'immintrin');
  await settle();
  const hits = await count('.card');
  check('search narrows the transcript to turns containing the text', hits >= 1 && hits < total, `${hits} of ${total}`);
  check('search leaves the shortcut keys alone while typing', page.url().includes('/transcript'));
  await page.$eval('.tx-bar input[type="search"]', (e) => { e.value = ''; e.dispatchEvent(new Event('input', { bubbles: true })); });
  await settle();
  check('clearing the search restores every turn', (await count('.card')) === total);
  await page.evaluate(() => { window.__blobs = []; const orig = URL.createObjectURL.bind(URL); URL.createObjectURL = (b) => { window.__blobs.push(b); return orig(b); }; });
  await clickText('Download .md');
  const md = await page.evaluate(() => window.__blobs[0].text());
  check('Download .md contains every arm, numbered steps and prompts', md.startsWith('# matxfer_full2 · chain 1 · transcript') && md.includes('## Arm C') && md.includes('### 01 · Claude call · generate') && md.includes('**Prompt**'), md.slice(0, 80));
  await clickText('Download .json');
  const js = JSON.parse(await page.evaluate(() => window.__blobs[1].text()));
  check('Download .json is parseable and complete', ['A', 'B', 'C'].every((k) => js[k].length === stepFiles('matxfer_full2', k)));
  await page.click('.tx-bar .tabs button:nth-child(3)');
  await page.waitForFunction(() => document.querySelectorAll('.tx-arm').length === 1);
  check('arm filter shows only arm B and puts it in the URL', (await count('.tx-arm')) === 1 && (await count('.tx-arm.B')) === 1 && decodeURIComponent(page.url()).includes('/transcript/B'));
  await nav('#/matxfer_full2/1/transcript/C/2');
  check('deep link opens one transcript step', (await page.$eval('.card[data-key="T-C-2"]', (e) => e.dataset.open)) === 'true');
  await nav('#/matxfer_full2/1/timeline');
  await page.keyboard.press('a');
  check("'a' opens the transcript", page.url().includes('/transcript'));
  await page.keyboard.press('a');
  check("'a' again returns to the timeline", page.url().includes('/timeline'));
  await page.screenshot({ path: path.join(SHOTS, 'timeline_after.png') });
  await nav('#/matxfer_full2/1/transcript', '?theme=light');
  await page.click('.card[data-key="T-C-1"] .card-head');
  await page.click('.card[data-key="T-C-3"] .card-head').catch(() => {});
  await new Promise((r) => setTimeout(r, 900));
  await page.screenshot({ path: path.join(SHOTS, 'transcript_light.png') });
  await nav('#/matxfer_full2/1/transcript', '?theme=dark');
  await new Promise((r) => setTimeout(r, 900));
  await page.screenshot({ path: path.join(SHOTS, 'transcript_dark.png') });

  // --- review fixes ---
  await nav('#/matxfer_full2/1/transcript');
  const allCards = await count('.card');
  await page.type('.tx-bar input[type="search"]', 'cost_usd');
  await settle();
  check('search matches the text of a turn, not JSON key names', (await count('.card')) === 0, `${await count('.card')} cards for a key-only word`);
  await page.$eval('.tx-bar input[type="search"]', (e) => { e.value = ''; e.dispatchEvent(new Event('input', { bubbles: true })); });
  await settle();
  check('clearing the search restores all turns again', (await count('.card')) === allCards);
  await page.evaluate(() => document.querySelectorAll('.tx-bar label input')[0].click());
  await nav('#/matxfer_full2/1/transcript/C/1');
  check('a deep link opens its step even when a filter would have hidden it', (await count('.card[data-key="T-C-1"]')) === 1 && (await page.$eval('.card[data-key="T-C-1"]', (e) => e.dataset.open)) === 'true');
  check('the filters are reset by that deep link', (await page.$$eval('.tx-bar label input', (els) => els.every((e) => e.checked))));
  await nav('#/matxfer_full2/1/transcript');
  await page.click('.card[data-key="T-A-1"] .card-head');
  await page.keyboard.press('f');
  check("'f' does nothing on the transcript (its toggle is hidden), so open turns stay open", (await page.$eval('.card[data-key="T-A-1"]', (e) => e.dataset.open)) === 'true');
  await nav('#/matxfer_scalar_full2/1/memory');
  check('memory tab says arm A was not re-run when its memory was reused', (await text('.col.A')).includes('reused'));

  // a run whose arm A was not re-run (memory reused from an earlier run)
  await nav('#/matxfer_scalar_full2');
  check('reused-memory run: arm A tile says it was not re-run', (await page.$$eval('.tile.A', (els) => els.every((e) => e.textContent.includes('not re-run') && e.textContent.includes('traces/matxfer_full2')))));
  await nav('#/matxfer_scalar_full2/1/transcript');
  check('reused-memory run: arm A transcript is the single reuse event', (await count('.tx-arm.A .card')) === 1 && (await text('.tx-arm.A .card-head')).includes('memory reused'));
  await nav('#/matxfer_scalar_full2/1/timeline');
  check('reused-memory run: timeline shows the reused memory, no empty attempts', (await text('.col.A')).includes('memory reused') && (await count('.col.A .card')) === 0);
  // arm D: x86 + A's memory, a run with no B or C
  await nav('#/matxfer_d_full2');
  check('arm D run: tiles are A (reused) and D only', (await count('.tile')) === 2 && (await count('.tile.D')) === 1 && (await count('.tile.B, .tile.C')) === 0);
  check('arm D run: summary lists D on amd64 and no B or C row', (await text('table.sheet')).includes('amd64 · with A') && !(await text('table.sheet')).includes('arm/v7'));
  await page.click('.tile.D');
  await page.waitForSelector('.col.D');
  check('arm D run: timeline has two columns, A then D, with the memory shown', (await count('.col')) === 2 && (await text('.col.D')).includes('Memory shown to the model') && (await count('.col.D .card')) >= 1);
  check('arm D run: column header says amd64, not arm/v7', (await text('.col.D .col-head')).includes('amd64') && !(await text('.col.D .col-head')).includes('arm/v7'));
  await nav('#/matxfer_d_full2/1/memory');
  check('arm D run: memory tab has a D column that matches memory.md', (await text('.col.D')).includes('matches memory.md'));
  await nav('#/matxfer_d_full2/1/transcript');
  check('arm D run: transcript tabs are All, A and D', (await page.$$eval('.tx-bar .tabs button', (b) => b.map((x) => x.textContent).join(','))) === 'All arms,A,D');
  check('arm D run: transcript has D steps', (await count('.tx-arm.D .card')) >= 2);
  await nav('#/matxfer_d_full2/1/transcript/D');
  check('arm D deep link filters to arm D', (await count('.tx-arm')) === 1 && (await count('.tx-arm.D')) === 1);
  await nav('#/matxfer_generic');
  check('a run with no D keeps its three tiles and has no D row', (await count('.tile.D')) === 0 && (await count('.tile.B')) > 0 && (await count('.tile.C')) > 0 && !(await text('table.sheet')).includes('same platform as A'));
  // arm D attached to the run its memory came from
  await nav('#/matxfer_full2');
  check('source run shows D: summary row and a D tile on each of its three chains', (await text('table.sheet')).includes('same platform as A') && (await count('.tile.D:not(.empty)')) === 3);
  check('attached D says which run it was made in', (await page.$$eval('.tile.D', (els) => els.every((e) => /run separately as matxfer_d_full2(_r23)?/.test(e.textContent)))));
  check('every chain of the source run has a finished D tile, none empty', (await count('.tile.D.empty')) === 0);
  check('attached D leaves the B and C tiles of all three chains in place', (await count('.tile.B')) === 3 && (await count('.tile.C')) === 3);
  await nav('#/matxfer_full2/1/timeline/A/0');
  check('timeline/A/0 shows four columns A B C D', (await count('.col')) === 4 && (await count('.col.D')) === 1);
  check('attached D column carries its provenance and the memory it saw', (await text('.col.D')).includes('run separately as matxfer_d_full2') && (await text('.col.D')).includes('Memory shown to the model'));
  await nav('#/matxfer_full2/1/timeline/D/0');
  check('deep link opens the first D trial', (await count('.col.D .card')) >= 1);
  await nav('#/matxfer_full2/1/memory');
  check('memory tab shows D beside C with provenance', (await count('.col.D')) === 1 && (await text('.col.D')).includes('run separately as matxfer_d_full2'));
  await nav('#/matxfer_full2/1/transcript');
  check('source run transcript has a D tab and D steps', (await page.$$eval('.tx-bar .tabs button', (b) => b.map((x) => x.textContent).join(','))) === 'All arms,A,B,C,D' && (await count('.tx-arm.D .card')) >= 2);
  await nav('#/matxfer_full2/2/timeline');
  check('chain 2 of the source run renders A B C D with its own D trial', (await count('.col.A')) === 1 && (await count('.col.B')) === 1 && (await count('.col.C')) === 1 && (await count('.col.D .card')) >= 1 && (await text('.col.D')).includes('run separately as matxfer_d_full2_r23'));
  await nav('#/matxfer_generic/3/transcript');
  check('aborted repeat transcript renders its partial steps or a note without errors', (await count('.error-panel')) === 0 && (await count('.tx-arm')) === 3);

  // hostile text
  const evil = await browser.newPage();
  const evilErrors = [];
  evil.on('pageerror', (e) => evilErrors.push(e.message));
  await evil.goto('file://' + EVIL + '#/matxfer_evil/1/timeline/B/0', { waitUntil: 'load' });
  check('hostile </script><img onerror> text did not execute', (await evil.evaluate(() => window.__pwned)) === undefined);
  check('hostile text is not parsed into the DOM as an element', (await evil.$$eval('img[src="x"]', (e) => e.length)) === 0);
  check('hostile text is shown literally', (await evil.$eval('.card[data-key="B-0"] .stderr', (e) => e.textContent)).includes('</script><img'));
  await evil.goto('file://' + EVIL + '#/matxfer_evil/1/transcript/B/3', { waitUntil: 'load' });
  await evil.waitForSelector('.card[data-key="T-B-3"] .bubble.reply pre');
  check('hostile transcript text did not execute and renders literally', (await evil.evaluate(() => window.__pwned)) === undefined && (await evil.$$eval('img[src="x"]', (e) => e.length)) === 0 && (await evil.$eval('.card[data-key="T-B-3"] .bubble.reply pre', (e) => e.textContent)).includes('</script><img'));
  await evil.goto('file://' + EVIL + '#/matxfer_evil/1/transcript/B', { waitUntil: 'load' });
  await evil.waitForSelector('.tx-arm.B .card');
  await evil.evaluate(() => { window.__blobs = []; const o = URL.createObjectURL.bind(URL); URL.createObjectURL = (b) => { window.__blobs.push(b); return o(b); }; });
  await evil.evaluate(() => [...document.querySelectorAll('.tx-bar button')].find((b) => b.textContent === 'Download .md').click());
  const evilMd = await evil.evaluate(() => window.__blobs[0].text());
  check('downloaded markdown keeps model-chosen fields on one heading line', !/^# injected/m.test(evilMd) && !/^<script>/m.test(evilMd) && evilMd.includes('### 07'));
  check('hostile page has no script errors', evilErrors.length === 0, evilErrors.join(' | '));

  check('no console/page errors on the real-data page', errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
  console.log(failures ? `\n${failures} CHECK(S) FAILED` : '\nALL CHECKS PASSED');
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error('SCRIPT ERROR', e); process.exit(2); });
