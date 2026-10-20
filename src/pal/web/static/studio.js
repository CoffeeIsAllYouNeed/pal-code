/* Pal Studio: a chat. Describe a product; the bot proposes a plan, then asks a few questions one at a time (agents, access, tools, model,
   steps, files, comparison); then it builds, and you watch every step. Type or speak a change at any time.
   Everything a model wrote (replies, file names, file contents) is shown with textContent, never as HTML. */
(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const h = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
};
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
  return data;
};
const post = (path, body, signal) => api(path, { method: 'POST', body: JSON.stringify(body), signal });
const KINDS = { web_app: 'Web page or app', cli_tool: 'Command-line tool', library: 'Library', script: 'Script', document: 'Document', other: 'Other' };
const FINAL = ['done', 'error', 'cancelled'];
const n = (x) => Number(x || 0).toLocaleString('en-US');
const SUGGESTIONS = ['A landing page for a hotel', 'A to-do web app', 'A script that cleans a CSV file', 'A page where I paste meeting notes and get action items'];

const S = { cat: null, plan: null, notes: [], source: '', userSet: new Set(), refs: [], apiKey: '', open: null, q: 0, ret: false, stage: null,
            busy: false, abort: null, run: null, runs: [], summary: null };

// ---------------------------------------------------------------- the thread
const scrollDown = () => { const m = $('main'); m.scrollTop = m.scrollHeight; };
function showThread() { $('empty').hidden = true; $('below').hidden = true; $('thread').hidden = false; document.querySelector('.mainwrap').classList.remove('is-empty'); }
function append(el) { showThread(); $('thread').append(el); scrollDown(); return el; }
function addUser(text) { return append(h('div', { class: 'msg user', text })); }
function addBot(text, pills = []) {
  const body = h('div', { class: 'txt' }, text);
  if (pills.length) body.append(h('div', { class: 'pills' }, ...pills.map((c) => h('span', { class: 'pill', text: c }))));
  return append(h('div', { class: 'msg bot' }, h('img', { class: 'av', src: '/static/pal-logo-64.png', alt: '' }), body));
}
// a local model can take minutes (the first request loads it into memory), so say that the bot is working and for how long
function thinking() {
  const label = h('span', { text: 'Thinking... 0s' });
  const stopBtn = h('button', { type: 'button', id: 'stopwait', text: 'Stop waiting', onclick: () => { if (S.abort) S.abort.abort(); } });
  const m = append(h('div', { class: 'msg bot thinking' }, h('img', { class: 'av', src: '/static/pal-logo-64.png', alt: '' }), h('div', { class: 'txt' }, label, ' ', stopBtn)));
  const t0 = Date.now();
  const iv = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    label.textContent = `Thinking... ${s}s` + (s >= 15 ? '. The first request loads the model into memory, which can take a few minutes on a laptop. A smaller bot model is much faster.' : '');
  }, 1000);
  return { stop: () => { clearInterval(iv); m.remove(); } };
}
function setBusy(b) { S.busy = b; updateSend(); $('botmodel').disabled = b || !S.cat.local.ok; }
function updateSend() { $('send').disabled = S.busy || !$('cmd').value.trim(); }
function sizeInput() { $('cmd').rows = Math.min(6, Math.max(1, $('cmd').value.split('\n').length)); }

// ---------------------------------------------------------------- the local model chip and the bot's model
function renderChip() {
  const loc = S.cat.local, chip = $('localchip');
  chip.className = 'chip ' + (loc.ok ? 'ok' : 'bad');
  chip.textContent = loc.ok ? `Local model: ${loc.recommended} (the best fit for this computer)` + (loc.hardware ? ` \u00B7 ${loc.hardware}` : '') : (loc.error || 'Ollama is not reachable');
  $('startmodel').textContent = loc.ok ? loc.recommended : 'not reachable';
}
function fillBotModels() {
  const loc = S.cat.local, models = loc.models.filter((x) => !x.embedding);
  const opts = models.length ? models.map((x) => h('option', { value: x.name, text: x.label + (x.name === loc.recommended ? '  (strongest, slowest)' : '') })) : [h('option', { value: '', text: 'No local model' })];
  $('botmodel').replaceChildren(...opts);
  if (loc.recommended) $('botmodel').value = loc.recommended;
  $('botmodel').disabled = !loc.ok;
}

// ---------------------------------------------------------------- reference files (the + button)
function refNote(t) { const e = $('voicehint'); e.hidden = !t; e.textContent = t || ''; }
function renderRefs() {
  const chips = S.refs.map((r, i) => h('span', { class: 'refitem' }, r.name, h('button', { type: 'button', 'data-ref': i, title: 'Remove ' + r.name, 'aria-label': 'Remove ' + r.name, text: '✕' })));
  $('refchips').hidden = !chips.length;
  $('refchips').replaceChildren(...chips);
  document.querySelectorAll('.reflist').forEach((box) => box.replaceChildren(...S.refs.map((r, i) => h('span', { class: 'refitem' }, r.name, h('button', { type: 'button', 'data-ref': i, title: 'Remove ' + r.name, 'aria-label': 'Remove ' + r.name, text: '✕' })))));
}
function readRefs(files) {
  refNote('');
  for (const f of files) {
    if (S.refs.length >= S.cat.limits.max_references) { refNote(`At most ${S.cat.limits.max_references} reference files.`); break; }
    if (f.size > S.cat.limits.max_reference_kb * 1000) { refNote(`${f.name} is too big (the limit is ${S.cat.limits.max_reference_kb} KB).`); continue; }
    const rd = new FileReader();
    rd.onload = () => { S.refs.push({ name: f.name, text: String(rd.result) }); renderRefs(); };
    rd.readAsText(f);
  }
}
function dropRef(ev) { const b = ev.target.closest('button[data-ref]'); if (b) { S.refs.splice(Number(b.dataset.ref), 1); renderRefs(); } }

// ---------------------------------------------------------------- the plan, and keeping it valid (every change is checked by the server)
const roleName = (id) => (S.cat.roles.find((r) => r.id === id) || {}).name || id;
async function validate() {
  const data = await post('/api/studio/validate', { plan: S.plan, user_set: [...S.userSet] });
  S.plan = { ...data.plan };
  S.notes = data.notes || [];
  if (S.notes.length) addBot('I changed a few things to keep it safe or valid:', S.notes);
  return S.plan;
}
const withPlan = (fn) => { const p = JSON.parse(JSON.stringify(S.plan)); fn(p); S.plan = p; };

function checkItem(attrs, title, desc, why, cls = '') {
  const box = h('input', { type: 'checkbox', ...attrs });
  const body = h('div', {}, h('div', {}, h('b', { text: title }), ' ', ...(cls.includes('risky') ? [h('span', { class: 'badge', text: 'only you can switch this on' })] : [])), h('div', { class: 'why', text: desc }), why ? h('div', { class: 'why', text: why }) : null);
  return h('label', { class: 'item ' + cls }, box, body);
}

// ---------------------------------------------------------------- the questions: each builds its controls and returns what it will change
const Q = [
  { id: 'product', title: 'What is it?', help: 'This is what I understood. Correct anything that is off.',
    build(body) {
      const p = S.plan.product;
      const name = h('input', { type: 'text', maxlength: 60, value: p.name, 'aria-label': 'Name' });
      const kind = h('select', { 'aria-label': 'Kind' }, ...Object.entries(KINDS).map(([k, v]) => h('option', { value: k, text: v, selected: k === p.kind })));
      const sum = h('textarea', { rows: 3, maxlength: 400, 'aria-label': 'What it does' }, p.summary);
      body.append(h('label', {}, 'Name', name), h('label', {}, 'Kind', kind), h('label', {}, 'What it does', sum));
      return { mutate: (pl) => { pl.product = { name: name.value.trim(), kind: kind.value, summary: sum.value.trim() }; } };
    },
    say: (pl) => `${pl.product.name} (${KINDS[pl.product.kind] || pl.product.kind})`, label: 'Product' },

  { id: 'agents', title: 'Which agents should work on it?', help: 'Each agent does one job. The coder is always on.',
    build(body) {
      const list = S.cat.roles.map((r) => checkItem({ 'data-role': r.id, checked: S.plan.roles.includes(r.id), disabled: r.id === 'coder' }, r.name, r.description,
        S.plan.reasons['role:' + r.id] ? 'Why: ' + S.plan.reasons['role:' + r.id] : '', S.plan.roles.includes(r.id) ? '' : 'off'));
      body.append(...list);
      return { mutate: (pl) => { pl.roles = [...body.querySelectorAll('[data-role]')].filter((c) => c.checked).map((c) => c.dataset.role); } };
    },
    say: (pl) => pl.roles.map(roleName).join(', '), label: 'Agents' },

  { id: 'access', title: 'What may they access?', help: 'Anything that sends data off this machine is off unless you switch it on.',
    build(body) {
      body.append(...S.cat.permissions.map((p) => checkItem({ 'data-perm': p.id, checked: !!S.plan.permissions[p.id], disabled: p.locked }, p.title, p.description,
        p.locked ? 'Locked: ' + p.why_locked : '', (p.risky ? 'risky ' : '') + (p.locked ? 'locked' : ''))));
      return { mutate: (pl) => {
        body.querySelectorAll('[data-perm]').forEach((c) => {
          pl.permissions[c.dataset.perm] = c.checked;
          const meta = S.cat.permissions.find((p) => p.id === c.dataset.perm);
          if (meta && meta.risky) { if (c.checked) S.userSet.add(c.dataset.perm); else S.userSet.delete(c.dataset.perm); }
        });
      } };
    },
    say: (pl) => { const on = S.cat.permissions.filter((p) => pl.permissions[p.id]).map((p) => p.title); return on.length ? on.join('; ') : 'Nothing extra'; }, label: 'Access' },

  { id: 'tools', title: 'Which tools may they use?', help: 'Tools give exact answers (a calculator, a document lookup). Some need access you switched on above.',
    build(body) {
      const perms = S.plan.permissions, groups = {};
      for (const t of S.cat.tools) (groups[t.group] = groups[t.group] || []).push(t);
      for (const [g, tools] of Object.entries(groups)) {
        body.append(h('h4', { text: g }));
        for (const t of tools) {
          const blocked = t.needs && !perms[t.needs];
          const permTitle = t.needs ? (S.cat.permissions.find((p) => p.id === t.needs) || {}).title : '';
          const why = !t.ready ? t.why_not : (blocked ? `needs "${permTitle}" (switch it on in the access question)` : (S.plan.reasons['tool:' + t.name] ? 'Why: ' + S.plan.reasons['tool:' + t.name] : ''));
          body.append(checkItem({ 'data-tool': t.name, checked: S.plan.tools.includes(t.name), disabled: !t.ready || blocked }, `${t.title} (${t.name})`, t.summary, why, (!t.ready || blocked) ? 'off' : ''));
        }
      }
      return { mutate: (pl) => { pl.tools = [...body.querySelectorAll('[data-tool]')].filter((c) => c.checked).map((c) => c.dataset.tool); } };
    },
    say: (pl) => (pl.tools.length ? pl.tools.join(', ') : 'None'), label: 'Tools' },

  { id: 'model', title: 'Which model builds it?', help: 'A local model stays on this machine. An API model needs your key (kept in this tab only).',
    build(body) {
      const m = S.plan.model, loc = S.cat.local, models = loc.models.filter((x) => !x.embedding);
      const rl = h('input', { type: 'radio', name: 'mmode', value: 'local', 'aria-label': 'Local model', checked: m.mode !== 'api' });
      const ra = h('input', { type: 'radio', name: 'mmode', value: 'api', 'aria-label': 'API model', checked: m.mode === 'api' });
      const localSel = h('select', { 'aria-label': 'Local model', disabled: !loc.ok }, ...models.map((x) => h('option', { value: x.name, text: x.label + (x.name === loc.recommended ? '  (recommended: the strongest)' : '') })));
      if (m.mode !== 'api' && m.model) localSel.value = m.model;
      const localBox = h('div', { class: 'localbox' }, h('label', {}, 'Model', localSel), h('p', { class: 'hint', text: loc.ok ? `Runs on this machine. Nothing leaves it. Context window: ${n(S.cat.limits.num_ctx)} tokens.` : (loc.error || 'Ollama is not reachable.') }));
      const modeSel = h('select', { 'aria-label': 'How' }, ...S.cat.llm_modes.map((x) => h('option', { value: x.key, text: x.title })));
      modeSel.value = m.mode === 'api' && m.api_mode ? m.api_mode : S.cat.llm_mode;
      const modeNote = h('p', { class: 'hint' });
      const prov = h('select', { 'aria-label': 'Provider' });
      const apiModel = h('input', { type: 'text', list: 'apimodels', placeholder: 'model name', 'aria-label': 'API model name' });
      const key = h('input', { type: 'password', autocomplete: 'off', placeholder: 'paste your key (kept in this tab only)', 'aria-label': 'API key' });
      key.value = S.apiKey;
      const keyWarn = h('p', { class: 'warn', hidden: true });
      const useLocal = h('button', { type: 'button', class: 'btn', hidden: true, text: 'Use the local model instead', onclick: () => { rl.checked = true; toggle(); keyWarn.hidden = true; useLocal.hidden = true; } });
      const apiBox = h('div', { class: 'apibox' }, h('label', {}, 'How', modeSel), modeNote, h('label', {}, 'Provider', prov), h('label', {}, 'Model', apiModel), h('datalist', { id: 'apimodels' }), h('label', {}, 'API key', key), keyWarn, useLocal);
      const fillProv = (want) => {
        const mode = S.cat.llm_modes.find((x) => x.key === modeSel.value) || { providers: [], summary: '' };
        modeNote.textContent = mode.summary;
        prov.replaceChildren(...mode.providers.map((p) => h('option', { value: p.name, text: p.name + (p.key_set ? ' (key set)' : '') })));
        if (want && mode.providers.some((p) => p.name === want)) prov.value = want;
        if (modeSel.value === 'explore') loadOpenRouterModels(apiModel); else apiBox.querySelector('#apimodels').replaceChildren();
      };
      const toggle = () => { localBox.hidden = ra.checked; apiBox.hidden = !ra.checked; };
      modeSel.addEventListener('change', () => fillProv(''));
      rl.addEventListener('change', toggle); ra.addEventListener('change', toggle);
      fillProv(m.mode === 'api' ? m.provider : '');
      if (m.mode === 'api') apiModel.value = m.model || '';
      toggle();
      body.append(h('div', { class: 'choice' }, h('label', {}, rl, ' Local (Ollama, on this machine)'), h('label', {}, ra, ' API model')), localBox, apiBox);
      return {
        check() {
          keyWarn.hidden = true; useLocal.hidden = true;
          if (ra.checked && !apiModel.value.trim()) { keyWarn.hidden = false; keyWarn.textContent = 'Choose a model name for the API provider.'; return false; }
          if (ra.checked && !key.value && !S.cat.providers[prov.value]) { keyWarn.hidden = false; keyWarn.textContent = `There is no API key for ${prov.value}. Paste one above, or use the local model.`; useLocal.hidden = false; return false; }
          if (rl.checked && !localSel.value) { keyWarn.hidden = false; keyWarn.textContent = 'There is no local model: start Ollama and pull one (for example: ollama pull llama3.1:8b), or switch to an API model.'; return false; }
          return true;
        },
        mutate: (pl) => {
          S.apiKey = ra.checked ? key.value : '';
          pl.model = ra.checked ? { mode: 'api', provider: prov.value, model: apiModel.value.trim() } : { mode: 'local', model: localSel.value };
        },
      };
    },
    say: (pl) => (pl.model.mode === 'api' ? `${pl.model.provider}: ${pl.model.model}` : `${pl.model.model} (local)`), label: 'Model' },

  { id: 'steps', title: 'What are the steps?', help: 'The agents work through these in order. Change the wording, the agent, or the order.',
    build(body) {
      const roles = S.plan.roles, list = h('div', { class: 'steps' });
      const row = (s) => {
        const r = h('div', { class: 'steprow' },
          h('select', { 'aria-label': 'Agent for this step' }, ...roles.map((x) => h('option', { value: x, text: roleName(x), selected: x === s.role }))),
          h('input', { type: 'text', value: s.task, maxlength: 300, 'aria-label': 'Task for this step' }),
          h('span', {}, h('button', { type: 'button', 'data-move': -1, title: 'Move up', text: '↑' }), h('button', { type: 'button', 'data-move': 1, title: 'Move down', text: '↓' }), h('button', { type: 'button', 'data-del': 1, title: 'Remove this step', text: '✕' })));
        return r;
      };
      list.append(...S.plan.steps.map(row));
      list.addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        const r = b.closest('.steprow');
        if (b.dataset.del) r.remove();
        else if (b.dataset.move) { const to = Number(b.dataset.move) < 0 ? r.previousElementSibling : r.nextElementSibling; if (to) { if (Number(b.dataset.move) < 0) to.before(r); else to.after(r); } }
      });
      const add = h('button', { type: 'button', class: 'btn', text: 'Add a step', onclick: () => {
        const role = roles.includes('reviewer') ? 'reviewer' : roles[roles.length - 1] || 'coder';
        list.append(row({ role, task: 'Describe what this step should do' }));
      } });
      body.append(list, h('div', { class: 'btnrow' }, add));
      return { mutate: (pl) => { pl.steps = [...list.querySelectorAll('.steprow')].map((r) => ({ role: r.querySelector('select').value, task: r.querySelector('input').value.trim() })).filter((s) => s.task); } };
    },
    say: (pl) => `${pl.steps.length} step${pl.steps.length === 1 ? '' : 's'}`, label: 'Steps' },

  { id: 'files', title: 'Any reference files?', help: 'A brief or a design document. Pal sends only the parts that match each step; a plain client pastes them in full. Text files, up to 150 KB each.',
    build(body) {
      body.append(h('div', { class: 'reflist' }), h('div', { class: 'btnrow' }, h('button', { type: 'button', class: 'btn', id: 'attachq', text: 'Attach files', onclick: () => $('refs').click() })));
      renderRefs();
      return { mutate: (pl) => { if (S.refs.length) pl.permissions.read_references = true; } };
    },
    say: () => (S.refs.length ? S.refs.map((r) => r.name).join(', ') : 'None'), label: 'Files', cont: 'Continue' },

  { id: 'compare', title: 'Show the difference?', help: 'Build it twice, once with a plain client and once with Pal, and compare tokens and time. It takes about twice as long. Small builds can show little or no saving.',
    build(body) {
      const yes = h('input', { type: 'radio', name: 'cmp', value: 'yes', checked: !!S.plan.metrics, 'aria-label': 'Build twice and compare' });
      const no = h('input', { type: 'radio', name: 'cmp', value: 'no', checked: !S.plan.metrics, 'aria-label': 'Build once' });
      body.append(h('div', { class: 'choice' }, h('label', {}, yes, ' Yes, build it twice and compare'), h('label', {}, no, ' No, build it once')));
      return { mutate: (pl) => { pl.metrics = yes.checked; } };
    },
    say: (pl) => (pl.metrics ? 'Build twice and compare' : 'Build once'), label: 'Compare' },
];

function closeOpen() { if (S.open) { S.open.remove(); S.open = null; } }
function answered(i) {
  const q = Q[i];
  return h('div', { class: 'answered', 'data-q': q.id }, h('span', { class: 'k', text: q.label }), h('span', { class: 'v', text: q.say(S.plan) }),
    h('button', { type: 'button', text: 'Change', onclick: () => askQ(i, true) }));
}
function askQ(i, ret = false) {
  closeOpen();
  S.q = i; S.ret = ret; S.stage = 'q';
  const q = Q[i], body = h('div', { class: 'qbody' });
  const got = q.build(body);
  const err = h('p', { class: 'warn', hidden: true });
  const next = h('button', { type: 'button', class: 'btn primary', 'data-next': q.id, text: ret ? 'Done' : (i === Q.length - 1 ? 'Review' : 'Continue') });
  const back = h('button', { type: 'button', class: 'btn', text: 'Back', hidden: ret || i === 0 });
  const card = h('div', { class: 'qcard', 'data-q': q.id, role: 'group', 'aria-label': q.title },
    h('div', { class: 'qhead', text: `Question ${i + 1} of ${Q.length}` }), h('div', { class: 'qtitle', text: q.title }), h('p', { class: 'qhelp', text: q.help }), body, err,
    h('div', { class: 'btnrow' }, next, back));
  next.addEventListener('click', async () => {
    if (got.check && !got.check()) return;
    next.disabled = true;
    try {
      withPlan((pl) => got.mutate(pl));
      await validate();
      card.replaceWith(answered(i));
      S.open = null;
      if (ret || i === Q.length - 1) showConfirm(); else askQ(i + 1);
    } catch (e) { err.hidden = false; err.textContent = e.message; next.disabled = false; }
  });
  back.addEventListener('click', () => { const prev = card.previousElementSibling; if (prev && prev.classList.contains('answered')) prev.remove(); askQ(i - 1); });
  S.open = card;
  append(card);
  renderRefs();
  const first = card.querySelector('input:not([disabled]),select,textarea');
  if (first && !('ontouchstart' in window)) first.focus({ preventScroll: true });
}
function showConfirm() {
  closeOpen();
  S.stage = 'confirm';
  const p = S.plan, running = S.run && !FINAL.includes((S.run.summary || {}).status || 'queued');
  const rows = Q.map((q) => h('tr', {}, h('td', { text: q.label }), h('td', { text: q.say(p) })));
  const changes = h('div', { class: 'btnrow' }, ...Q.map((q, i) => h('button', { type: 'button', class: 'btn', text: 'Change ' + q.label.toLowerCase(), onclick: () => askQ(i, true) })));
  const build = h('button', { type: 'button', class: 'btn primary', id: 'build', disabled: !!running, text: running ? 'Building...' : 'Build it' });
  const msg = h('p', { class: 'warn', id: 'buildmsg', hidden: true });
  build.addEventListener('click', () => startBuild(build, msg));
  const card = h('div', { class: 'qcard', 'data-q': 'confirm', role: 'group', 'aria-label': 'Ready to build' },
    h('div', { class: 'qhead', text: 'Ready' }), h('div', { class: 'qtitle', text: 'Here is the plan. Shall I build it?' }),
    h('table', { class: 'summary' }, h('tbody', {}, ...rows)), h('p', { class: 'hint', text: 'You can also type or say a change below, for example "add a tester".' }),
    msg, h('div', { class: 'btnrow' }, build), h('h4', { text: 'Or change something' }), changes);
  S.open = card;
  append(card);
}

// ---------------------------------------------------------------- talking to the bot
async function sendChat(text) {
  if (!text || S.busy) return;
  addUser(text);
  setBusy(true);
  const wait = thinking();
  S.abort = new AbortController();
  const model = $('botmodel').value || undefined;
  try {
    let r;
    const first = !S.plan;
    if (first) r = await post('/api/studio/plan', { command: text, user_set: [...S.userSet], model }, S.abort.signal);
    else r = await post('/api/studio/revise', { plan: S.plan, instruction: text, user_set: [...S.userSet], model }, S.abort.signal);
    wait.stop();
    S.plan = r.plan; S.notes = r.notes || []; S.source = r.source;
    if (first) { document.querySelectorAll('.answered').forEach((e) => e.remove()); }
    addBot(r.message, r.changes || []);
    if (r.local && !r.local.ok) addBot(r.local.error);
    if (S.source === 'fallback') addBot('The bot could not plan this, so I made a plan by simple rules. Please check each answer.');
    if (first) { addBot(`I have a plan. I will ask ${Q.length} short questions, one at a time. You can type or say a change at any time.`); askQ(0); }
    else if (S.stage === 'confirm') { const gone = S.open; showConfirm(); if (gone && gone.isConnected) gone.remove(); }
    else askQ(S.q, S.ret);
  } catch (e) {
    wait.stop();
    addBot(e.name === 'AbortError' ? 'Stopped waiting. The model may still be working in the background. Try a smaller bot model, or choose "Set up by hand".' : 'That did not work: ' + e.message);
  } finally { setBusy(false); }
}
async function manualStart() {
  if (S.busy) return;
  S.plan = { product: { name: 'My project', kind: 'web_app', summary: '' }, roles: ['coder'], tools: [], permissions: {}, model: { mode: 'local', model: S.cat.local.recommended || '' }, steps: [], reasons: {}, metrics: false };
  S.source = '';
  showThread();
  closeOpen();
  const data = await post('/api/studio/validate', { plan: S.plan, user_set: [] });
  S.plan = data.plan; S.notes = [];
  addBot('You are setting this up by hand. I will ask a few questions, one at a time. You can still type a change at any time.');
  askQ(0);
}
function newBuild() {
  if (S.abort && S.busy) S.abort.abort();
  closeOpen();
  Object.assign(S, { plan: null, notes: [], source: '', stage: null, q: 0, ret: false, apiKey: '' });
  S.userSet = new Set();
  $('thread').replaceChildren();
  $('thread').hidden = true; $('empty').hidden = false; $('below').hidden = false; document.querySelector('.mainwrap').classList.add('is-empty');
  $('cmd').value = ''; sizeInput(); updateSend();
  $('side').classList.remove('open');
}

// ---------------------------------------------------------------- building and watching
async function startBuild(btn, msg) {
  msg.hidden = true;
  btn.disabled = true;
  try {
    if (S.refs.length && !S.plan.permissions.read_references) withPlan((pl) => { pl.permissions.read_references = true; });
    const plan = await validate();
    const body = { plan, user_set: [...S.userSet], references: S.refs };
    if (S.apiKey && plan.model.mode === 'api') body.api_key = S.apiKey;
    const r = await post('/api/studio/run', body);
    btn.textContent = 'Building...';
    startWork(r.run_id, plan.product.name);
  } catch (e) { msg.hidden = false; msg.textContent = e.message; btn.disabled = false; }
}
function startWork(runId, product) {
  const el = {
    status: h('div', { class: 'runstatus', role: 'status', text: 'Starting...' }), tabs: h('div', { class: 'tabs', role: 'tablist' }), counters: h('div', { class: 'counters' }),
    events: h('div', { class: 'events', 'aria-live': 'polite' }), files: h('ul', { class: 'files' }), preview: h('pre', { class: 'preview', hidden: true }),
    filesbox: h('div', { hidden: true }), download: h('a', { class: 'linkbtn', href: '#', hidden: true, text: 'Download this build (.zip)' }), compare: h('div', { hidden: true }),
    stop: h('button', { type: 'button', class: 'btn', id: 'stop', text: 'Stop' }),
  };
  el.filesbox.append(h('h3', { text: 'Files' }), h('div', { class: 'filesrow' }, el.files, el.preview), h('p', {}, el.download));
  const card = h('div', { class: 'run', 'data-run': runId }, h('div', { class: 'qhead', text: 'Building ' + product }), el.status, el.tabs, el.counters, el.events, el.filesbox, el.compare, h('div', { class: 'btnrow' }, el.stop));
  const R = { id: runId, product, since: 0, events: [], summary: null, tab: null, timer: null, polling: false, filesSeq: 0, fileSeq: 0, el, card };
  S.run = R; S.runs.push(R); S.summary = null;
  closeOpen();
  append(card);
  addRunItem(R);
  el.stop.addEventListener('click', async () => { el.stop.disabled = true; try { await post(`/api/studio/run/${R.id}/cancel`, {}); } catch (e) { addBot(e.message); } });
  el.tabs.addEventListener('click', async (e) => { const b = e.target.closest('button[data-tab]'); if (!b) return; R.tab = b.dataset.tab; renderWork(R); if (R.tab !== 'compare' && FINAL.includes(R.summary.status)) await showFilesFor(R, R.tab); });
  el.files.addEventListener('click', (e) => { const b = e.target.closest('button[data-file]'); if (b) openFile(R, b.dataset.file); });
  R.timer = setInterval(() => poll(R), 700);
  poll(R);
}
function addRunItem(R) {
  const box = $('runs');
  if (box.querySelector('.navnote')) box.replaceChildren();
  R.item = h('button', { type: 'button', class: 'runitem', 'data-run': R.id, onclick: () => { R.card.scrollIntoView({ block: 'start' }); $('side').classList.remove('open'); } }, h('span', { class: 'dot live' }), h('span', { text: R.product }));
  box.prepend(R.item);
}
async function poll(R) {
  if (R.polling) return;
  R.polling = true;
  try {
    const d = await api(`/api/studio/run/${R.id}?since=${R.since}`);
    R.events.push(...d.events);
    R.since += d.events.length;
    R.summary = d; S.summary = d;
    renderWork(R);
    if (FINAL.includes(d.status)) await finishRun(R);
  } catch (e) { R.el.status.textContent = 'Lost contact with the server: ' + e.message; clearInterval(R.timer); }
  finally { R.polling = false; }
}
async function finishRun(R) {
  clearInterval(R.timer);
  R.el.stop.hidden = true;
  R.item.querySelector('.dot').className = 'dot done';
  const arms = Object.keys(R.summary.arms);
  if (R.summary.comparison) R.tab = 'compare';
  else if (!R.tab) R.tab = arms[0] || null;
  renderWork(R);
  await showFilesFor(R, R.tab === 'compare' ? arms[arms.length - 1] : R.tab);
  addBot(R.summary.status === 'done' ? 'Done. Preview or download the files above, or open the project in the editor. Type a change below if you want something different, or build it again.' : 'The build stopped. Check the steps above, change something, and build again.');
  if (Object.values(R.summary.arms).some((a) => a.files > 0)) {
    $('vieweditor').disabled = false;
    append(h('div', { class: 'btnrow' }, h('button', { type: 'button', class: 'btn primary', id: 'openeditor', text: 'Open in editor', onclick: () => window.studio.openEditor && window.studio.openEditor(R) })));
  }
  showConfirm();
}
const activeArm = (R) => { const arms = Object.keys((R.summary || {}).arms || {}); return R.tab && R.tab !== 'compare' ? R.tab : arms[arms.length - 1]; };
function renderWork(R) {
  const d = R.summary, el = R.el;
  if (!d) return;
  const arms = Object.entries(d.arms);
  const labels = { done: 'Finished', running: 'Building...', queued: 'Starting...', error: 'Stopped by an error', cancelled: 'Stopped' };
  el.status.textContent = `${d.product}: ${labels[d.status] || d.status} (${d.seconds}s)`;
  if (!R.tab) R.tab = arms.length ? arms[arms.length - 1][0] : null;
  const tabs = arms.map(([k, a]) => h('button', { type: 'button', role: 'tab', class: R.tab === k ? 'active' : '', 'data-tab': k, text: a.label }));
  if (d.comparison) tabs.push(h('button', { type: 'button', role: 'tab', class: R.tab === 'compare' ? 'active' : '', 'data-tab': 'compare', text: 'Compare' }));
  el.tabs.replaceChildren(...(arms.length > 1 || d.comparison ? tabs : []));
  const arm = activeArm(R);
  const t = arm && d.arms[arm] ? d.arms[arm].totals : null;
  el.counters.replaceChildren(...(t && t.steps ? [
    h('span', {}, 'Step ', h('b', { text: `${t.steps_done} of ${t.steps}` })), h('span', {}, 'Tokens read ', h('b', { text: n(t.prompt_tokens) })), h('span', {}, 'written ', h('b', { text: n(t.completion_tokens) })),
    ...(t.helper_tokens ? [h('span', {}, 'helper ', h('b', { text: n(t.helper_tokens) }))] : []), h('span', {}, 'Time ', h('b', { text: t.seconds + 's' })), h('span', {}, 'Files ', h('b', { text: String(t.files) })),
    ...(t.blocked ? [h('span', { class: 'bad' }, 'Refused ', h('b', { text: String(t.blocked) }))] : [])] : []));
  const show = R.tab === 'compare';
  el.events.hidden = show; el.compare.hidden = !show; el.filesbox.hidden = show || !R.events.some((e) => e.kind === 'file');
  if (show) renderCompare(R, d.comparison); else renderEvents(R, arm);
}
function renderEvents(R, arm) {
  const box = R.el.events;
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
  box.replaceChildren(...R.events.filter((e) => !e.arm || e.arm === arm).map((e) => h('div', { class: 'ev ev-' + e.kind }, h('span', { class: 't', text: e.t + 's' }), e.text)));
  if (stick) box.scrollTop = box.scrollHeight;
}
function renderCompare(R, c) {
  const b = c.baseline, m = c.pal, saved = c.main_tokens_saved;
  const head = saved > 0 ? `Without Pal: ${n(b.main_tokens)} tokens. With Pal: ${n(m.main_tokens)}. Saved ${n(saved)} tokens (${c.main_tokens_saved_pct}%).`
    : `Without Pal: ${n(b.main_tokens)} tokens. With Pal: ${n(m.main_tokens)}. Pal used ${n(-saved)} more tokens (${Math.abs(c.main_tokens_saved_pct)}% more).`;
  const row = (label, a, z, pc, cls = '') => h('tr', { class: cls }, h('td', { text: label }), h('td', { text: a }), h('td', { text: z }), h('td', { class: pc != null && pc < 0 ? 'neg' : '', text: pc == null ? '' : (pc >= 0 ? pc + '% saved' : Math.abs(pc) + '% more') }));
  const table = h('table', { class: 'cmp' },
    h('thead', {}, h('tr', {}, h('th', { text: '' }), h('th', { text: 'Plain chat client' }), h('th', { text: 'Pal' }), h('th', { text: 'Difference' }))),
    h('tbody', {},
      row('Main-model tokens', n(b.main_tokens), n(m.main_tokens), c.main_tokens_saved_pct, 'saved'),
      row('Tokens read (prompts)', n(b.prompt_tokens), n(m.prompt_tokens), c.prompt_tokens_saved_pct),
      row('Tokens written', n(b.completion_tokens), n(m.completion_tokens), null),
      row('Helper tokens (its own small jobs)', '0', n(m.helper_tokens), null),
      row('All tokens, helper included', n(b.main_tokens), n(m.main_tokens + m.helper_tokens), c.all_tokens_saved_pct),
      row('Time (seconds)', String(b.seconds), String(m.seconds), c.seconds_saved_pct),
      row('Files written', String(b.files), String(m.files), null), row('Problems left in the files', String(b.problems_left), String(m.problems_left), null)));
  R.el.compare.replaceChildren(h('div', { class: 'headline', text: head }), table, h('ul', {}, ...c.notes.map((t) => h('li', { class: 'hint', text: t }))));
}
async function showFilesFor(R, arm) {
  if (!arm) return;
  const mine = ++R.filesSeq;                       // only the latest request may draw: an older answer arriving late must not replace a newer tab's list
  const d = await api(`/api/studio/run/${R.id}/files?arm=${encodeURIComponent(arm)}`);
  if (mine !== R.filesSeq) return;
  R.el.files.replaceChildren(...d.files.map((f) => h('li', {}, h('button', { type: 'button', 'data-file': f.path, text: `${f.path} (${n(f.bytes)} B)` }))));
  R.el.filesbox.hidden = !d.files.length;
  const saved = R.summary && R.summary.arms[arm] && R.summary.arms[arm].totals.files > 0;
  R.el.download.hidden = !(d.files.length && saved);
  R.el.download.href = `/api/studio/run/${R.id}/download?arm=${encodeURIComponent(arm)}`;
  R.el.preview.hidden = true;
}
async function openFile(R, path) {
  const mine = ++R.fileSeq;
  const d = await api(`/api/studio/run/${R.id}/file?arm=${encodeURIComponent(activeArm(R))}&path=${encodeURIComponent(path)}`);
  if (mine !== R.fileSeq) return;
  R.el.preview.hidden = false;
  R.el.preview.textContent = d.content + (d.truncated ? '\n... (shortened)' : '');
  R.el.files.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.file === path));
}

// ---------------------------------------------------------------- the microphone (voice.js: Pal's speech-to-text, else the browser's)
function setupMic() {
  if (window.palVoice) window.palVoice($('mic'), $('cmd'), refNote, () => { sizeInput(); updateSend(); });
  else $('mic').disabled = true;
}

// ---------------------------------------------------------------- start
async function init() {
  try { S.cat = await api('/api/studio/catalog'); }
  catch (e) { $('localchip').className = 'chip bad'; $('localchip').textContent = 'Could not load the Studio: ' + e.message; return; }
  renderChip();
  fillBotModels();
  $('suggest').replaceChildren(...SUGGESTIONS.map((t) => h('button', { type: 'button', text: t, onclick: () => sendChat(t) })));
  $('composer').addEventListener('submit', (e) => { e.preventDefault(); const t = $('cmd').value.trim(); if (!t || S.busy) return; $('cmd').value = ''; sizeInput(); updateSend(); sendChat(t); });
  $('cmd').addEventListener('input', () => { sizeInput(); updateSend(); });
  $('cmd').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); } });
  $('attach').addEventListener('click', () => $('refs').click());
  $('refs').addEventListener('change', (e) => { readRefs([...e.target.files]); e.target.value = ''; });
  $('refchips').addEventListener('click', dropRef);
  $('thread').addEventListener('click', dropRef);
  $('byhand').addEventListener('click', manualStart);
  $('newbuild').addEventListener('click', newBuild);
  $('menu').addEventListener('click', () => { const o = $('side').classList.toggle('open'); $('menu').setAttribute('aria-expanded', String(o)); });
  setupMic();
  updateSend();
  window.studio = { state: S, validate, poll: () => (S.run ? poll(S.run) : null), util: { h, api, post, n } };      // for tests and for curious people at the console
}
let orLoaded = false;
async function loadOpenRouterModels(input) {
  const list = document.getElementById('apimodels');
  if (orLoaded) return;
  try {
    const d = await api('/api/llm/models?limit=500');
    orLoaded = true;
    if (list) list.replaceChildren(...d.models.map((m) => h('option', { value: m.id, label: m.label })));
    input.placeholder = 'pick one of ' + d.models.length + ' models, or type a name';
  } catch (e) { input.placeholder = 'the model list could not be read: type a model name'; }
}
init();
})();
