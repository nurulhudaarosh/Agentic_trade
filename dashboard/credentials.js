/* Settings → Connections: paste API keys for the broker and LLM providers, test and save them. */
const CredentialsApp = (() => {
  const state = { status: null, results: {}, busy: {}, drafts: {} };
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const toast = (m, err) => App.toast(m, err);
  const t = App.t;

  async function call(path, body) {
    const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    const data = await res.json();
    if (data.status) state.status = data.status;
    if (!res.ok && !data.test) throw new Error(data.error || res.statusText);
    return data;
  }

  async function load() {
    try { state.status = await call('/api/credentials'); }
    catch (err) { $('cred-list').innerHTML = `<div class="ra-muted">${esc(err.message)}</div>`; return; }
    render();
  }

  function badge(p) {
    const r = state.results[p.id] || p.last_test;
    if (r && r.ok === false && (state.results[p.id] || p.source === 'app')) return `<span class="cred-badge bad">${t('cred.failed')}</span>`;
    if (!p.configured) return `<span class="cred-badge off">${t('cred.off')}</span>`;
    if (r && r.ok) return `<span class="cred-badge ok">${t('cred.ok')}${p.source === '.env' ? ' · .env' : ''}</span>`;
    return p.source === 'app' ? `<span class="cred-badge warn">${t('cred.untested')}</span>` : `<span class="cred-badge warn">${t('cred.env')}</span>`;
  }

  function inputRow(pid, f, value, idx) {
    const id = `cred-${pid}-${f.id}${idx == null ? '' : `-${idx}`}`;
    return `<div class="cred-input">
      <input type="password" id="${id}" data-field="${f.id}" value="${esc(value)}" placeholder="${esc(t('cred.paste'))}"
        autocomplete="off" spellcheck="false" onfocus="if(this.value.startsWith('••••'))this.select()">
      <button type="button" class="cred-icon" title="${esc(t('cred.show'))}" onclick="CredentialsApp.reveal('${id}')">👁</button>
      ${idx != null ? `<button type="button" class="cred-icon cred-drop" title="${esc(t('cred.drop'))}" onclick="CredentialsApp.dropRow('${pid}', ${idx})">✕</button>` : ''}
    </div>`;
  }

  function render() {
    const s = state.status; if (!s) return;
    $('cred-list').innerHTML = s.providers.map(cardHtml).join('');
    $('cred-store').textContent = t('cred.store', { f: s.store });
  }

  // Re-render one card only, so half-typed keys in the other cards survive.
  function renderCard(pid) {
    const p = state.status.providers.find((x) => x.id === pid), el = $(`cred-card-${pid}`);
    if (p && el) el.outerHTML = cardHtml(p);
  }

  function cardHtml(p) {
    // Values typed but not saved yet (e.g. a key the provider rejected) stay in the form.
    const shown = state.drafts[p.id] || p.masked;
    let fields;
    if (p.multi) {
      const f = p.fields[0];
      const keys = shown.keys && shown.keys.length ? shown.keys : [''];
      fields = `<label class="field-label">${t(`cred.f.${p.id}.${f.id}`)} <span class="dim">${t('cred.multi')}</span></label>
        <div id="cred-rows-${p.id}" class="cred-rows">${keys.map((k, i) => inputRow(p.id, f, k, i)).join('')}</div>
        <button type="button" class="btn btn-secondary btn-sm cred-add" onclick="CredentialsApp.addRow('${p.id}')">+ ${t('cred.add')}</button>
        ${p.extra_env_keys ? `<div class="dim">${t('cred.extra', { n: p.extra_env_keys })}</div>` : ''}`;
    } else {
      fields = p.fields.map((f) => `<label class="field-label" for="cred-${p.id}-${f.id}">${t(`cred.f.${p.id}.${f.id}`)}</label>${inputRow(p.id, f, shown[f.id] || '')}`).join('');
    }
    const r = state.results[p.id] || p.last_test;
    const busy = state.busy[p.id];
    return `<form class="cred-card" id="cred-card-${p.id}" onsubmit="return false" autocomplete="off">
      <div class="cred-head">
        <div><div class="cred-name">${t(`cred.${p.id}.name`)}</div><div class="dim">${t(`cred.${p.id}.desc`)}</div></div>
        ${badge(p)}
      </div>
      <div class="cred-fields">${fields}</div>
      <div class="cred-actions">
        <button class="btn btn-success btn-sm" ${busy ? 'disabled' : ''} onclick="CredentialsApp.save('${p.id}')">${busy === 'save' ? t('cred.checking') : t('cred.save')}</button>
        <button class="btn btn-secondary btn-sm" ${busy || !p.configured ? 'disabled' : ''} onclick="CredentialsApp.test('${p.id}')">${busy === 'test' ? t('cred.testing') : t('cred.test')}</button>
        ${p.saved_in_app ? `<button class="btn btn-secondary btn-sm" ${busy ? 'disabled' : ''} onclick="CredentialsApp.remove('${p.id}')">${t('cred.remove')}</button>` : ''}
        <a class="cred-link" href="${esc(p.signup)}" target="_blank" rel="noopener">${t('cred.get')} ↗</a>
      </div>
      ${r ? `<div class="cred-result ${r.ok ? 'ok' : 'bad'}">${r.ok ? '✓' : '✕'} ${esc(r.message)}</div>` : ''}
    </form>`;
  }

  function collect(pid) {
    const p = state.status.providers.find((x) => x.id === pid);
    if (p.multi) {
      return { keys: [...$(`cred-rows-${pid}`).querySelectorAll('input')].map((i) => i.value.trim()).filter(Boolean) };
    }
    return Object.fromEntries(p.fields.map((f) => [f.id, $(`cred-${pid}-${f.id}`).value.trim()]));
  }

  async function run(pid, kind, fn) {
    state.busy[pid] = kind; renderCard(pid);
    try { await fn(); }
    catch (err) { state.results[pid] = { ok: false, message: err.message }; }
    finally { delete state.busy[pid]; renderCard(pid); }
  }

  const save = (pid) => {
    state.drafts[pid] = collect(pid);
    return run(pid, 'save', async () => {
      const data = await call('/api/credentials/save', { provider: pid, values: state.drafts[pid] });
      if (data.test) state.results[pid] = data.test;
      if (data.saved) {
        delete state.drafts[pid];
        toast(t('cred.saved'));
        if (pid === 'alpaca' && typeof fetchPortfolio === 'function') fetchPortfolio();
      } else toast(t('cred.rejected'), true);
    });
  };

  const test = (pid) => run(pid, 'test', async () => {
    const data = await call('/api/credentials/test', { provider: pid });
    state.results[pid] = data.test;
  });

  const remove = (pid) => {
    if (!confirm(t('cred.remove.confirm'))) return;
    run(pid, 'remove', async () => { await call('/api/credentials/remove', { provider: pid }); delete state.results[pid]; delete state.drafts[pid]; toast(t('cred.removed')); });
  };

  function addRow(pid) {
    const box = $(`cred-rows-${pid}`), p = state.status.providers.find((x) => x.id === pid);
    const idx = box.children.length;
    box.insertAdjacentHTML('beforeend', inputRow(pid, p.fields[0], '', idx));
    box.lastElementChild.querySelector('input').focus();
  }

  function dropRow(pid, idx) {
    const box = $(`cred-rows-${pid}`);
    if (box.children.length === 1) { box.querySelector('input').value = ''; return; }
    box.children[idx].remove();
    [...box.children].forEach((row, i) => row.querySelectorAll('.cred-drop').forEach((b) => b.setAttribute('onclick', `CredentialsApp.dropRow('${pid}', ${i})`)));
  }

  function reveal(id) { const el = $(id); el.type = el.type === 'password' ? 'text' : 'password'; }

  window.addEventListener('DOMContentLoaded', load);
  App.on('lang', () => { if (state.status) render(); });
  return { load, save, test, remove, addRow, dropRow, reveal, statusData: () => state.status };
})();
