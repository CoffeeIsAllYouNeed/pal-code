/* Pal editor: open a finished build as a project. File list, tabs, a code pane with line numbers, and "Ask Pal" on the right: Pal proposes a change as a diff,
   you accept or reject it. Every request goes through the token-saving client on the server, and the panel shows what it cost.
   Everything a model or a file contains is shown with textContent (or as a textarea's value), never as HTML. */
(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const wait = () => new Promise((r) => { const t = setInterval(() => { if (window.studio && window.studio.util) { clearInterval(t); r(window.studio); } }, 20); });

wait().then((studio) => {
const { h, api, post, n } = studio.util;
const E = { run: null, arm: '', files: [], tabs: [], cur: '', buf: {}, proposal: null, busy: false, ctl: null, seq: 0 };

const has = () => !!E.run;
const dirty = (p) => E.buf[p] && E.buf[p].text !== E.buf[p].orig;
const say = (text, cls = '') => { const m = h('div', { class: 'edmsg ' + cls, text }); $('edmsgs').append(m); $('edmsgs').scrollTop = $('edmsgs').scrollHeight; return m; };
const status = (t, bad = false) => { const s = $('edstatus'); s.textContent = t; s.classList.toggle('bad', bad); };
const armQ = () => `arm=${encodeURIComponent(E.arm)}`;

function renderFiles() {
  $('edfiles').replaceChildren(...E.files.map((f) => h('li', {}, h('button', { type: 'button', class: f.path === E.cur ? 'active' : '', 'data-file': f.path },
    h('span', { class: 'fname', text: f.path }), dirty(f.path) ? h('span', { class: 'dotmark', text: '●', title: 'Unsaved changes' }) : null))));
}
function renderTabs() {
  $('edtabs').replaceChildren(...E.tabs.map((p) => h('div', { class: 'edtab' + (p === E.cur ? ' active' : '') },
    h('button', { type: 'button', class: 'edtabname', 'data-tab': p, text: p.split('/').pop() + (dirty(p) ? ' ●' : ''), title: p }),
    h('button', { type: 'button', class: 'edtabx', 'data-close': p, 'aria-label': 'Close ' + p, text: '×' }))));
}
function gutter() {
  const code = $('edcode'), lines = code.value.split('\n').length;
  $('edgutter').textContent = Array.from({ length: lines }, (_, i) => i + 1).join('\n');
  $('edgutter').scrollTop = code.scrollTop;
}
function caret() {
  const code = $('edcode'), before = code.value.slice(0, code.selectionStart).split('\n');
  $('edpos').textContent = `Ln ${before.length}, Col ${before[before.length - 1].length + 1}`;
}
function show() {
  const p = E.cur, b = E.buf[p];
  $('edcode').hidden = !p; $('edgutter').hidden = !p; $('edempty').hidden = !!p;
  if (!p) { $('edpath').textContent = ''; $('edsave').disabled = true; renderFiles(); renderTabs(); return; }
  $('edcode').value = b.text;
  $('edcode').readOnly = !!b.readonly || E.busy;
  $('edpath').textContent = p + (b.readonly ? ' (too large to edit here)' : '');
  $('edsave').disabled = !dirty(p) || !!b.readonly;
  gutter(); caret(); renderFiles(); renderTabs();
  status(dirty(p) ? 'Unsaved changes' : 'Saved');
}
async function openFile(p) {
  if (!E.buf[p]) {
    const d = await api(`/api/studio/run/${E.run}/file?${armQ()}&path=${encodeURIComponent(p)}&full=1`);
    E.buf[p] = { text: d.content, orig: d.content, readonly: d.truncated };
  }
  if (!E.tabs.includes(p)) E.tabs.push(p);
  E.cur = p;
  show();
}
function closeTab(p) {
  if (dirty(p) && !window.confirm(`Close ${p} and lose the unsaved changes?`)) return;
  const i = E.tabs.indexOf(p);
  E.tabs.splice(i, 1);
  delete E.buf[p];
  if (E.cur === p) E.cur = E.tabs[Math.min(i, E.tabs.length - 1)] || '';
  if (E.cur) openFile(E.cur); else show();
}
async function refreshFiles() {
  const d = await api(`/api/studio/run/${E.run}/files?${armQ()}`);
  E.files = d.files;
  renderFiles();
}
async function save(p = E.cur) {
  const b = E.buf[p];
  if (!b || !dirty(p) || b.readonly) return true;
  try {
    const d = await api(`/api/studio/run/${E.run}/file`, { method: 'PUT', body: JSON.stringify({ arm: E.arm, path: p, content: b.text }) });
    b.orig = b.text;
    await refreshFiles();
    status(d.saved_to_disk ? 'Saved' : 'Saved (this build was a dry run, so it lives in this session only)');
    if (p === E.cur) { $('edsave').disabled = true; renderTabs(); }
    return true;
  } catch (e) { status('Could not save: ' + e.message, true); return false; }
}
async function saveAll() { for (const p of Object.keys(E.buf)) if (!(await save(p))) return false; return true; }

// ---------------------------------------------------------------- Ask Pal
function meter(session) {
  const m = $('edmeter');
  if (!session || !session.asks) { m.textContent = 'Each request goes through the token-saving layer.'; return; }
  m.replaceChildren(h('b', { text: n(session.pal_tokens) }), ` tokens used over ${session.asks} request${session.asks > 1 ? 's' : ''}. A plain chat that resends the project would use about `,
    h('b', { text: n(session.plain_estimate) }), session.saved_pct > 0 ? ` (${session.saved_pct}% saved, estimated).` : '. Pal has not saved tokens yet: it pays off as the project and the conversation grow.');
}
function renderProposal(pr) {
  E.proposal = pr;
  const box = $('eddiff');
  box.hidden = !pr;
  $('edcode').hidden = !!pr || !E.cur; $('edgutter').hidden = !!pr || !E.cur; $('edempty').hidden = !!pr || !!E.cur;
  if (!pr) { box.replaceChildren(); return; }
  const parts = pr.files.map((f) => h('section', { class: 'dfile' },
    h('div', { class: 'dhead' }, h('b', { text: f.path }), f.new ? ' (new file)' : ''),
    h('pre', { class: 'dlines' }, ...f.diff.map(([t, x]) => h('span', { class: 'dl d-' + t, text: (t === 'add' ? '+ ' : t === 'del' ? '- ' : t === 'hunk' ? '' : '  ') + x + '\n' })))));
  box.replaceChildren(h('div', { class: 'dbar' }, h('span', { text: `Pal suggests changing ${pr.files.length} file${pr.files.length > 1 ? 's' : ''}` }),
    h('span', { class: 'sp' }), h('button', { type: 'button', class: 'btn', id: 'edreject', text: 'Reject' }), h('button', { type: 'button', class: 'btn primary', id: 'edaccept', text: 'Accept' })), ...parts);
}
async function accept() {
  const pr = E.proposal;
  if (!pr) return;
  try {
    for (const f of pr.files) {
      await api(`/api/studio/run/${E.run}/file`, { method: 'PUT', body: JSON.stringify({ arm: E.arm, path: f.path, content: f.content }) });
      if (E.buf[f.path]) E.buf[f.path] = { text: f.content, orig: f.content, readonly: false };
    }
    await refreshFiles();
    say(`Applied to ${pr.files.map((f) => f.path).join(', ')}.`, 'ok');
    renderProposal(null);
    const first = pr.files[0].path;
    if (!E.tabs.includes(first)) await openFile(first); else show();
  } catch (e) { say('Could not apply the change: ' + e.message, 'bad'); }
}
function reject() { renderProposal(null); say('Rejected. Nothing changed.'); show(); }
function busy(b) { E.busy = b; $('edsend').disabled = b || !$('edask').value.trim(); $('edask').disabled = b; $('edcode').readOnly = b || !!(E.cur && E.buf[E.cur].readonly); $('edstop').hidden = !b; }
async function ask(text) {
  if (E.busy || !text) return;
  if (E.proposal) renderProposal(null);
  say(text, 'me');
  busy(true);
  if (!(await saveAll())) { say('Fix the save problem first, then ask again.', 'bad'); busy(false); return; }
  const pending = say('Pal is reading the project...', 'wait');
  const mine = ++E.seq;
  E.ctl = new AbortController();
  try {
    const d = await post(`/api/studio/run/${E.run}/ask`, { arm: E.arm, path: E.cur, instruction: text, api_key: studio.state.apiKey || undefined }, E.ctl.signal);
    if (mine !== E.seq) return;
    pending.remove();
    if (d.message) say(d.message);
    d.refused.forEach((r) => say(`Refused ${r.path}: ${r.reason}`, 'bad'));
    if (d.note) say(d.note);
    const u = d.usage;
    say(`Read ${n(u.prompt_tokens)} tokens, wrote ${n(u.completion_tokens)}${u.helper_tokens ? `, helper ${n(u.helper_tokens)}` : ''} in ${u.seconds}s. A plain chat would have read about ${n(u.plain_estimate - u.completion_tokens)}.`, 'meta');
    meter(d.session);
    if (d.files.length) renderProposal(d); else if (!d.message) say('Pal did not suggest any change.');
  } catch (e) {
    if (mine !== E.seq) return;
    pending.remove();
    say(e.name === 'AbortError' ? 'Stopped.' : 'Something went wrong: ' + e.message, 'bad');
  } finally { if (mine === E.seq) busy(false); }
}

// ---------------------------------------------------------------- opening and closing the editor
function pickArm(R) {
  const arms = Object.entries((R.summary || {}).arms || {}).filter(([, a]) => a.files > 0).map(([k]) => k);
  return arms.includes('pal') ? 'pal' : arms[arms.length - 1] || '';
}
studio.editorArm = pickArm;
async function open(R) {
  const arm = pickArm(R);
  if (!arm) return;
  if (E.run !== R.id || E.arm !== arm) {
    Object.assign(E, { run: R.id, arm, files: [], tabs: [], cur: '', buf: {}, proposal: null });
    $('edmsgs').replaceChildren();
    say('Ask me to change the open file, add a feature, or explain the code. I will show you the change first, and nothing is saved until you accept it.');
    meter(null);
    renderProposal(null);
    $('eddownload').href = `/api/studio/run/${R.id}/download?arm=${encodeURIComponent(arm)}`;
  }
  document.querySelector('.mainwrap').hidden = true;
  $('editor').hidden = false;
  $('viewchat').classList.remove('active'); $('vieweditor').classList.add('active');
  $('side').classList.remove('open');
  await refreshFiles();
  const first = E.cur || (['index.html', 'main.py', 'readme.md'].map((w) => E.files.find((f) => f.path.toLowerCase() === w)).find(Boolean) || E.files[0] || {}).path || '';
  if (first) await openFile(first); else show();
}
function close() {
  $('editor').hidden = true;
  document.querySelector('.mainwrap').hidden = false;
  $('vieweditor').classList.remove('active'); $('viewchat').classList.add('active');
}
studio.openEditor = open;

$('viewchat').addEventListener('click', close);
$('vieweditor').addEventListener('click', () => { const R = studio.state.runs.slice().reverse().find((r) => pickArm(r)); if (R) open(R); });
$('edfiles').addEventListener('click', (e) => { const b = e.target.closest('button[data-file]'); if (b && !E.busy) openFile(b.dataset.file); });
$('edtabs').addEventListener('click', (e) => {
  const c = e.target.closest('button[data-close]'), t = e.target.closest('button[data-tab]');
  if (c) closeTab(c.dataset.close); else if (t) openFile(t.dataset.tab);
});
$('edcode').addEventListener('input', () => { const b = E.buf[E.cur]; if (!b) return; b.text = $('edcode').value; $('edsave').disabled = !dirty(E.cur); status(dirty(E.cur) ? 'Unsaved changes' : 'Saved'); gutter(); renderTabs(); renderFiles(); });
$('edcode').addEventListener('scroll', () => { $('edgutter').scrollTop = $('edcode').scrollTop; });
['keyup', 'click'].forEach((ev) => $('edcode').addEventListener(ev, caret));
$('edcode').addEventListener('keydown', (e) => {
  if (e.key === 'Tab' && !e.shiftKey) {                       // a code editor keeps the focus and inserts two spaces
    e.preventDefault();
    const c = $('edcode'), a = c.selectionStart;
    c.setRangeText('  ', a, c.selectionEnd, 'end');
    c.dispatchEvent(new Event('input'));
  }
});
$('editor').addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); save(); } });
$('edsave').addEventListener('click', () => save());
$('eddiff').addEventListener('click', (e) => { if (e.target.closest('#edaccept')) accept(); else if (e.target.closest('#edreject')) reject(); });
$('edform').addEventListener('submit', (e) => { e.preventDefault(); const t = $('edask').value.trim(); $('edask').value = ''; ask(t); });
$('edask').addEventListener('input', () => { $('edsend').disabled = E.busy || !$('edask').value.trim(); });
$('edask').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('edform').requestSubmit(); } });
$('edstop').addEventListener('click', () => { if (E.ctl) E.ctl.abort(); });
$('ednew').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = $('ednewname').value.trim();
  if (!name || !has()) return;
  try {
    await api(`/api/studio/run/${E.run}/file`, { method: 'PUT', body: JSON.stringify({ arm: E.arm, path: name, content: '' }) });
    $('ednewname').value = '';
    await refreshFiles();
    const f = E.files.find((x) => x.path.toLowerCase() === name.toLowerCase().replace(/\\/g, '/'));
    await openFile(f ? f.path : name);
  } catch (err) { status('Could not create the file: ' + err.message, true); }
});
window.addEventListener('beforeunload', (e) => { if (Object.keys(E.buf).some(dirty)) { e.preventDefault(); e.returnValue = ''; } });
studio.editor = E;
});
})();
