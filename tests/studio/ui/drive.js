// Drives the Studio chat like a user, in jsdom, against a running server. Prints one JSON object with what it saw.
// usage: node drive.js <base url> <scenario>
const { JSDOM } = require('jsdom');
const base = process.argv[2], scenario = process.argv[3];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ORDER = ['product', 'agents', 'access', 'tools', 'model', 'steps', 'files', 'compare'];

// a stand-in for the browser's speech recognition: when started it "hears" a sentence
class FakeSpeech {
  constructor() { FakeSpeech.last = this; }
  start() { this.started = true; setTimeout(() => this.onresult && this.onresult({ results: [[{ transcript: 'build me a hotel page' }]] }), 30); }
  stop() { this.started = false; setTimeout(() => this.onend && this.onend(), 10); }
}

async function open(slow = null) {
  const dom = await JSDOM.fromURL(base + '/studio', {
    runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = async (u, o = {}) => {
        const { signal, ...rest } = o, ac = new AbortController();      // jsdom's AbortSignal is not Node's: hand Node's fetch its own and forward the abort
        if (signal) signal.addEventListener('abort', () => ac.abort());
        const r = await fetch(new URL(u, base), { ...rest, signal: ac.signal });
        if (slow && slow.test(String(u))) await sleep(900);                 // `slow`: make some answers arrive late
        return r;
      };
      if (scenario === 'mic') w.webkitSpeechRecognition = FakeSpeech;
    },
  });
  const w = dom.window, d = w.document, $ = (id) => d.getElementById(id);
  async function until(cond, what, ms = 15000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { try { if (cond()) return; } catch (e) { /* not yet */ } await sleep(25); }
    throw new Error('timed out waiting for: ' + what);
  }
  const type = (el, v) => { el.value = v; el.dispatchEvent(new w.Event('input', { bubbles: true })); el.dispatchEvent(new w.Event('change', { bubbles: true })); };
  const tick = (el, on) => { el.checked = on; el.dispatchEvent(new w.Event('change', { bubbles: true })); };
  await until(() => $('localchip').textContent && !$('localchip').textContent.startsWith('Checking'), 'the page to load');
  return { w, d, $, until, type, tick };
}

const ask = (p, text) => { p.type(p.$('cmd'), text); p.$('composer').requestSubmit(); };
const card = (p, id) => p.d.querySelector(`.qcard[data-q="${id}"]`);
const waitCard = (p, id, ms) => p.until(() => card(p, id), 'the ' + id + ' question', ms);
const q = (p, id, sel) => [...card(p, id).querySelectorAll(sel)];
const nextBtn = (p, id) => card(p, id).querySelector('[data-next]');
// answer the questions in order (optionally acting on a card first); stops after `upto`, or at the confirm card
async function through(p, upto = 'confirm', acts = {}, seen = {}) {
  for (const id of ORDER) {
    await waitCard(p, id);
    if (seen[id]) seen[id] = seen[id](p);
    if (acts[id]) await acts[id](p);
    nextBtn(p, id).click();
    if (id === upto) return;
  }
  await waitCard(p, 'confirm');
}
const chat = (p) => [...p.d.querySelectorAll('#thread > .msg')].map((m) => [m.classList.contains('user') ? 'user' : 'bot', (m.querySelector('.txt') || m).textContent, [...m.querySelectorAll('.pill')].map((x) => x.textContent)]);
const planNow = (p) => JSON.parse(JSON.stringify(p.w.studio.state.plan));
const runDone = (p, ms = 15000) => p.until(() => p.w.studio.state.summary && ['done', 'error', 'cancelled'].includes(p.w.studio.state.summary.status), 'the build to finish', ms);
const run = (p) => p.d.querySelector('.run');
const rq = (p, sel) => run(p).querySelector(sel);

const scenarios = {
  async plan(p) {
    const empty = { visible: !p.$('empty').hidden, suggestions: p.d.querySelectorAll('#suggest button').length, micDisabled: p.$('mic').disabled };
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    const out = { empty, chip: p.$('localchip').textContent, chipClass: p.$('localchip').className, head: card(p, 'product').querySelector('.qhead').textContent,
                  name: q(p, 'product', 'input')[0].value, kind: q(p, 'product', 'select')[0].value, source: p.w.studio.state.source, openCards: p.d.querySelectorAll('.qcard').length };
    const seen = {
      agents: (pp) => { out.roles = q(pp, 'agents', '[data-role]').filter((c) => c.checked).map((c) => c.dataset.role); out.coderDisabled = q(pp, 'agents', '[data-role="coder"]')[0].disabled; },
      access: (pp) => {
        out.perms = Object.fromEntries(q(pp, 'access', '[data-perm]').map((c) => [c.dataset.perm, { on: c.checked, disabled: c.disabled, risky: c.closest('label').classList.contains('risky') }]));
        out.riskyBadges = q(pp, 'access', '.badge').map((b) => b.textContent);
      },
      tools: (pp) => { out.tools = q(pp, 'tools', '[data-tool]').filter((c) => c.checked).map((c) => c.dataset.tool); out.toolsDisabled = q(pp, 'tools', '[data-tool]').filter((c) => c.disabled).map((c) => c.dataset.tool); },
      model: (pp) => { out.modelMode = q(pp, 'model', 'input[name=mmode]').find((r) => r.checked).value; const sel = q(pp, 'model', 'select')[0]; out.localOptions = [...sel.options].map((o) => o.textContent); out.localValue = sel.value; },
      steps: (pp) => { out.steps = q(pp, 'steps', '.steprow').map((r) => [r.querySelector('select').value, r.querySelector('input').value]); },
    };
    await through(p, 'confirm', {}, seen);
    out.userSet = [...p.w.studio.state.userSet];
    out.answeredLines = p.d.querySelectorAll('.answered').length;
    out.confirmRows = [...card(p, 'confirm').querySelectorAll('tbody tr')].map((r) => [...r.children].map((c) => c.textContent));
    out.chat = chat(p);
    return out;
  },
  async onebyone(p) {
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    const counts = [p.d.querySelectorAll('.qcard').length];
    nextBtn(p, 'product').click();
    await waitCard(p, 'agents');
    counts.push(p.d.querySelectorAll('.qcard').length);
    const heads = [card(p, 'agents').querySelector('.qhead').textContent];
    const back = card(p, 'agents').querySelector('.btnrow .btn:not(.primary)');
    back.click();                                                    // Back: the product question is asked again
    await waitCard(p, 'product');
    return { counts, heads, afterBack: p.d.querySelectorAll('.qcard').length, answered: p.d.querySelectorAll('.answered').length };
  },
  async thinking(p) {
    await ask(p, 'a to-do list web page');
    await p.until(() => p.d.querySelector('#thread .thinking'), 'the thinking message');
    const during = { text: p.d.querySelector('#thread .thinking .txt').textContent, send: p.$('send').disabled, botmodel: p.$('botmodel').disabled, stopVisible: !!p.$('stopwait') };
    await waitCard(p, 'product');
    return { during, after: { thinking: p.d.querySelectorAll('#thread .thinking').length, send: p.$('send').disabled, stop: !!p.$('stopwait') } };
  },
  async stopwait(p) {
    await ask(p, 'a to-do list web page');
    await p.until(() => p.$('stopwait'), 'the stop button');
    p.$('stopwait').click();
    await p.until(() => !p.d.querySelector('#thread .thinking') && p.d.querySelector('#thread').textContent.includes('Stopped waiting'), 'the page to come back');
    await sleep(2500);                                                         // long enough for the (ignored) late answer
    return { cards: p.d.querySelectorAll('.qcard').length, chat: chat(p).map((m) => m[1]), thinking: p.d.querySelectorAll('#thread .thinking').length, planSet: !!p.w.studio.state.plan };
  },
  async botmodel(p) {
    const opts = [...p.$('botmodel').options].map((o) => [o.value, o.textContent]);
    const first = p.$('botmodel').value;
    p.type(p.$('botmodel'), 'llama3.2:1b');
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    return { opts, first, chosen: p.$('botmodel').value };
  },
  async hand(p) {
    p.$('byhand').click();
    const out = {};
    await through(p, 'confirm', {}, {
      product: (pp) => { out.name = q(pp, 'product', 'input')[0].value; },
      agents: (pp) => { out.roles = q(pp, 'agents', '[data-role]').filter((c) => c.checked).map((c) => c.dataset.role); },
    });
    return { ...out, plan: planNow(p), chat: chat(p), source: p.w.studio.state.source };
  },
  async edit(p) {
    await ask(p, 'a to-do list web page');
    await through(p, 'steps', {
      agents: (pp) => pp.tick(q(pp, 'agents', '[data-role="docs"]')[0], false),                  // the person removes the docs writer
      access: (pp) => pp.tick(q(pp, 'access', '[data-perm="network_access"]')[0], true),         // and switches going online on
    });
    await waitCard(p, 'steps');
    const afterDocs = { roles: planNow(p).roles, steps: q(p, 'steps', '.steprow').map((r) => r.querySelector('select').value), perm: planNow(p).permissions.network_access, userSet: [...p.w.studio.state.userSet] };
    nextBtn(p, 'steps').click();
    await waitCard(p, 'files'); nextBtn(p, 'files').click();
    await waitCard(p, 'compare'); nextBtn(p, 'compare').click();
    await waitCard(p, 'confirm');
    // change the access again: switch going online off; the person's own switch is forgotten
    const rows = [...card(p, 'confirm').querySelectorAll('.btnrow .btn')].find((b) => b.textContent === 'Change access');
    rows.click();
    await waitCard(p, 'access');
    p.tick(q(p, 'access', '[data-perm="network_access"]')[0], false);
    nextBtn(p, 'access').click();
    await waitCard(p, 'confirm');
    const netOff = { userSet: [...p.w.studio.state.userSet], perm: planNow(p).permissions.network_access };
    // add a tester and a step, then remove the first step
    [...card(p, 'confirm').querySelectorAll('.btnrow .btn')].find((b) => b.textContent === 'Change agents').click();
    await waitCard(p, 'agents');
    p.tick(q(p, 'agents', '[data-role="tester"]')[0], true);
    nextBtn(p, 'agents').click();
    await waitCard(p, 'confirm');
    const withTester = planNow(p).roles;
    [...card(p, 'confirm').querySelectorAll('.btnrow .btn')].find((b) => b.textContent === 'Change steps').click();
    await waitCard(p, 'steps');
    const before = q(p, 'steps', '.steprow').length;
    card(p, 'steps').querySelector('.btnrow .btn').click();                                        // Add a step
    const added = q(p, 'steps', '.steprow').length;
    q(p, 'steps', '[data-del]')[0].click();
    const afterDelete = q(p, 'steps', '.steprow').length;
    nextBtn(p, 'steps').click();
    await waitCard(p, 'confirm');
    const last = planNow(p).steps;
    return { afterDocs, netOff, withTester, before, added, afterDelete, stepsNow: last.length, lastTask: last[last.length - 1].task };
  },
  async revise(p) {
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    await p.until(() => !p.$('send').disabled || p.$('cmd').value === '', 'the bot to be free');
    await ask(p, 'add a tester');
    await p.until(() => chat(p).some((m) => m[1].startsWith('Done:')), 'the bot to answer');
    await waitCard(p, 'product');
    return { roles: planNow(p).roles, last: chat(p).filter((m) => m[0] === 'bot' && m[1].startsWith('Done:'))[0], cards: p.d.querySelectorAll('.qcard').length };
  },
  async build(p) {
    await ask(p, 'a to-do list web page');
    await through(p);
    p.$('build').click();
    await runDone(p);
    await p.until(() => rq(p, '.files').children.length > 0, 'the files list');
    const filesList = [...rq(p, '.files').querySelectorAll('button')].map((b) => b.dataset.file);
    rq(p, '.files').querySelector('button[data-file="app.js"]').click();
    await p.until(() => !rq(p, '.preview').hidden && rq(p, '.preview').textContent.length > 0, 'the preview');
    await p.until(() => card(p, 'confirm'), 'the plan to be offered again');
    return {
      status: rq(p, '.runstatus').textContent, events: [...run(p).querySelectorAll('.events .ev')].map((e) => [e.className, e.textContent]), counters: rq(p, '.counters').textContent,
      tabs: [...rq(p, '.tabs').children].map((b) => b.textContent), files: filesList, preview: rq(p, '.preview').textContent, download: rq(p, '.linkbtn').getAttribute('href'),
      downloadHidden: rq(p, '.linkbtn').hidden, stopHidden: rq(p, '#stop').hidden, buildDisabled: p.$('build').disabled, sidebar: [...p.d.querySelectorAll('#runs .runitem')].map((b) => b.textContent),
      dot: p.d.querySelector('#runs .dot').className,
    };
  },
  async compare(p) {
    await ask(p, 'a to-do list web page');
    await through(p, 'confirm', { compare: (pp) => pp.tick(q(pp, 'compare', 'input[value=yes]')[0], true) });
    p.$('build').click();
    await runDone(p, 30000);
    await p.until(() => run(p).querySelector('.headline'), 'the comparison', 30000);
    const cmp = run(p).querySelector('.headline').parentElement;
    const out = { headline: cmp.querySelector('.headline').textContent, rows: [...cmp.querySelectorAll('tbody tr')].map((r) => [...r.children].map((c) => c.textContent)),
      notes: [...cmp.querySelectorAll('li')].map((l) => l.textContent), tabs: [...rq(p, '.tabs').children].map((b) => b.textContent), activeTab: rq(p, '.tabs .active').textContent };
    rq(p, '.tabs button[data-tab="baseline"]').click();
    await p.until(() => cmp.hidden, 'the tab to switch');
    await p.until(() => (rq(p, '.linkbtn').getAttribute('href') || '').endsWith('arm=baseline'), 'the files of the plain build');
    return { ...out, plainEvents: [...run(p).querySelectorAll('.events .ev')].map((e) => e.textContent).length, plainFiles: [...rq(p, '.files').querySelectorAll('button')].map((b) => b.dataset.file), plainDownload: rq(p, '.linkbtn').getAttribute('href') };
  },
  async key(p) {
    await ask(p, 'a to-do list web page');
    await through(p, 'tools');
    await waitCard(p, 'model');
    p.tick(q(p, 'model', 'input[value=api]')[0], true);
    const sels = q(p, 'model', 'select');                               // [local model, how, provider]
    const modes = [...sels[1].options].map((o) => o.value);
    p.type(sels[1], 'own_key');
    const provSel = q(p, 'model', 'select')[2];
    const provs = [...provSel.options].map((o) => o.textContent);
    p.type(provSel, 'anthropic');
    p.type(q(p, 'model', 'input[type=text]')[0], 'claude-x');
    nextBtn(p, 'model').click();
    const warnEl = card(p, 'model').querySelector('.warn');
    await p.until(() => !warnEl.hidden, 'the key prompt');
    const useLocal = [...card(p, 'model').querySelectorAll('button')].find((b) => b.textContent === 'Use the local model instead');
    const warn = { text: warnEl.textContent, useLocalVisible: !useLocal.hidden, stillOnModel: !!card(p, 'model'), runStarted: p.w.studio.state.run !== null, modes, provs };
    useLocal.click();
    await p.until(() => q(p, 'model', 'input[value=local]')[0].checked, 'the switch back to local');
    return { ...warn, afterLocal: q(p, 'model', 'input[value=local]')[0].checked, warnHidden: warnEl.hidden, apiBoxHidden: card(p, 'model').querySelector('.apibox').hidden };
  },
  async stop(p) {
    await ask(p, 'a to-do list web page');
    await through(p);
    p.$('build').click();
    await p.until(() => rq(p, '#stop') && !rq(p, '#stop').hidden, 'the stop button');
    const before = { confirmGone: !card(p, 'confirm') };
    rq(p, '#stop').click();
    await p.until(() => p.w.studio.state.summary && p.w.studio.state.summary.status === 'cancelled', 'the build to stop');
    await p.until(() => rq(p, '#stop').hidden && card(p, 'confirm') && !p.$('build').disabled, 'the page to come back');
    return { before, status: rq(p, '.runstatus').textContent, buildDisabled: p.$('build').disabled };
  },
  async references(p) {
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    const f1 = new p.w.File(['# Brief\nuse blue'], 'brief.md', { type: 'text/markdown' });
    const big = new p.w.File(['x'.repeat(200000)], 'huge.txt', { type: 'text/plain' });
    Object.defineProperty(p.$('refs'), 'files', { value: [f1, big], configurable: true });
    p.$('refs').dispatchEvent(new p.w.Event('change', { bubbles: true }));
    await p.until(() => p.w.studio.state.refs.length === 1, 'the reference to be read');
    const names = [...p.$('refchips').querySelectorAll('.refitem')].map((r) => r.firstChild.textContent);
    const msg = p.$('voicehint').textContent;
    await through(p, 'files');
    const filesAnswer = q(p, 'files', '.refitem').length;
    p.$('refchips').querySelector('[data-ref="0"]').click();
    return { names, msg, filesAnswer, afterRemove: p.w.studio.state.refs.length, chips: p.$('refchips').hidden };
  },
  async hostile(p) {
    await ask(p, 'a page that says hello');
    await waitCard(p, 'product');
    const name = q(p, 'product', 'input')[0].value;
    await through(p);
    p.$('build').click();
    await p.until(() => p.w.studio.state.summary && p.w.studio.state.summary.status === 'done', 'the build');
    await p.until(() => rq(p, '.files').children.length > 0, 'the files');
    rq(p, '.files button').click();
    await p.until(() => !rq(p, '.preview').hidden, 'the preview');
    return { name, threadHtml: p.$('thread').innerHTML, preview: rq(p, '.preview').textContent, pwned: p.w.__pwned === undefined ? null : p.w.__pwned,
             injectedNodes: p.d.querySelectorAll('#thread img:not(.av), #thread script, .preview *, #thread img.av:not([src="/static/pal-logo-64.png"])').length, scripts: p.d.querySelectorAll('script').length };
  },
  async race(p) {
    // the files answer for the Pal build is held back, so it arrives AFTER the answer for the plain build the person clicked next
    await ask(p, 'a to-do list web page');
    await through(p, 'confirm', { compare: (pp) => pp.tick(q(pp, 'compare', 'input[value=yes]')[0], true) });
    p.$('build').click();
    await p.until(() => p.w.studio.state.summary && p.w.studio.state.summary.status === 'done', 'both builds to finish', 30000);
    await p.until(() => rq(p, '.tabs button[data-tab="baseline"]'), 'the tabs');
    rq(p, '.tabs button[data-tab="baseline"]').click();
    await p.until(() => (rq(p, '.linkbtn').getAttribute('href') || '').endsWith('arm=baseline'), 'the plain build files');
    await sleep(1400);                                                      // long enough for the held-back (stale) answer to arrive
    return { download: rq(p, '.linkbtn').getAttribute('href'), files: [...rq(p, '.files').querySelectorAll('button')].map((b) => b.dataset.file) };
  },
  async mic(p) {
    const b = p.$('mic');
    const before = { disabled: b.disabled, pressed: b.getAttribute('aria-pressed'), sendDisabled: p.$('send').disabled };
    b.click();
    await p.until(() => p.$('cmd').value === 'build me a hotel page', 'the words to appear');
    const during = { pressed: b.getAttribute('aria-pressed'), live: b.classList.contains('live'), text: p.$('cmd').value, sendDisabled: p.$('send').disabled };
    b.click();
    await p.until(() => b.getAttribute('aria-pressed') === 'false', 'listening to stop');
    return { before, during, after: { pressed: b.getAttribute('aria-pressed'), live: b.classList.contains('live') } };
  },
  async micoff(p) {
    return { disabled: p.$('mic').disabled, title: p.$('mic').title };
  },
  async newbuild(p) {
    await ask(p, 'a to-do list web page');
    await waitCard(p, 'product');
    p.$('newbuild').click();
    return { emptyVisible: !p.$('empty').hidden, threadHidden: p.$('thread').hidden, threadChildren: p.$('thread').children.length, plan: p.w.studio.state.plan };
  },
};

(async () => {
  const p = await open(scenario === 'race' ? /files\?arm=pal/ : null);
  try { console.log(JSON.stringify(await scenarios[scenario](p))); process.exit(0); }
  catch (e) { console.log(JSON.stringify({ error: String(e && e.message || e), html: p.d.body.textContent.slice(0, 600) })); process.exit(1); }
})();
