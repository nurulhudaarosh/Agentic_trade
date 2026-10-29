/* App shell: language, theme, the one global ticker, navigation and shared helpers. */
const App = (() => {
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  const PAGES = ['home', 'trade', 'portfolio', 'autopilot', 'insights', 'scanner', 'agent', 'research', 'settings'];
  const LOADERS = {};
  const state = {
    lang: store.get('lang', 'en') === 'bn' ? 'bn' : 'en',
    theme: store.get('theme', 'system'),
    ticker: store.get('ticker', 'TSLA'),
    page: 'home',
    tickers: ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'COIN', 'NVDA', 'AAPL', 'GOOGL', 'META', 'AMD'],
    prices: {},
    companies: {},
  };

  // ------------------------------------------------------------------ i18n
  function t(key, vars) {
    const entry = (window.I18N || {})[key];
    let s = entry ? (entry[state.lang === 'bn' ? 1 : 0] ?? entry[0]) : key;
    if (!entry && window.console) console.warn('i18n missing', key);
    if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] ?? m));
    return s;
  }

  function translate(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll('[data-i18n-html]').forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
    root.querySelectorAll('[data-i18n-ph]').forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
    root.querySelectorAll('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); el.setAttribute('aria-label', el.title); });
  }

  function setLang(lang) {
    state.lang = lang === 'bn' ? 'bn' : 'en';
    store.set('lang', state.lang);
    document.documentElement.lang = state.lang;
    document.querySelectorAll('[data-lang]').forEach((b) => b.classList.toggle('active', b.dataset.lang === state.lang));
    translate();
    setTitle();
    emit('lang');
  }

  // ------------------------------------------------------------------ theme
  function setTheme(theme) {
    state.theme = ['light', 'dark'].includes(theme) ? theme : 'system';
    store.set('theme', state.theme);
    if (state.theme === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.dataset.theme = state.theme;
    document.querySelectorAll('[data-theme-opt]').forEach((b) => b.classList.toggle('active', b.dataset.themeOpt === state.theme));
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = isDark() ? '☾' : '☀';
    emit('theme');
  }
  const isDark = () => state.theme === 'dark' || (state.theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  function toggleTheme() { setTheme(isDark() ? 'light' : 'dark'); }

  // ------------------------------------------------------------------ events
  const listeners = {};
  function on(evt, fn) { (listeners[evt] = listeners[evt] || []).push(fn); }
  function emit(evt, data) { (listeners[evt] || []).forEach((fn) => { try { fn(data); } catch (e) { console.error(e); } }); }

  // ------------------------------------------------------------------ ticker
  function setTicker(tk) {
    tk = String(tk || '').toUpperCase();
    if (!tk || tk === state.ticker) { closePicker(); return; }
    state.ticker = tk;
    store.set('ticker', tk);
    renderPicker();
    closePicker();
    emit('ticker', tk);
    reloadPage();
  }

  function setPrices(prices) {
    Object.assign(state.prices, prices || {});
    renderPicker();
  }

  function setTickers(list, companies) {
    if (list && list.length) state.tickers = [...new Set(list.map((x) => String(x).toUpperCase()))];
    Object.assign(state.companies, companies || {});
    if (!state.tickers.includes(state.ticker)) state.ticker = state.tickers[0];
    renderPicker();
  }

  function renderPicker() {
    const tk = state.ticker, p = state.prices[tk];
    const label = document.getElementById('picker-ticker');
    if (!label) return;
    label.textContent = tk;
    document.getElementById('picker-price').textContent = p ? money(p) : '';
    const list = document.getElementById('picker-list');
    const q = (document.getElementById('picker-search').value || '').trim().toUpperCase();
    list.innerHTML = state.tickers.filter((x) => !q || x.includes(q) || (state.companies[x] || '').toUpperCase().includes(q)).map((x) =>
      `<button class="picker-item ${x === tk ? 'active' : ''}" onclick="App.setTicker('${x}')" role="option" aria-selected="${x === tk}">
        <span><b>${x}</b><small>${esc(state.companies[x] || '')}</small></span><span class="num">${state.prices[x] ? money(state.prices[x]) : ''}</span></button>`).join('')
      || `<div class="muted pad">${t('picker.none')}</div>`;
  }
  function togglePicker(ev) {
    ev && ev.stopPropagation();
    const pop = document.getElementById('picker-pop');
    pop.hidden = !pop.hidden;
    if (!pop.hidden) { renderPicker(); const s = document.getElementById('picker-search'); s.value = ''; renderPicker(); s.focus(); }
  }
  function closePicker() { const pop = document.getElementById('picker-pop'); if (pop) pop.hidden = true; }

  // ------------------------------------------------------------------ navigation
  function register(page, loader) { LOADERS[page] = loader; }

  function go(page, opts = {}) {
    if (!PAGES.includes(page)) page = 'home';
    state.page = page;
    document.querySelectorAll('.page').forEach((el) => el.classList.toggle('active', el.id === `page-${page}`));
    document.querySelectorAll('[data-page]').forEach((el) => el.classList.toggle('active', el.dataset.page === page));
    document.body.classList.remove('nav-open');
    if (location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
    setTitle();
    if (!opts.noScroll) window.scrollTo({ top: 0 });
    reloadPage();
  }
  function reloadPage() { const fn = LOADERS[state.page]; if (fn) fn(); }
  function setTitle() {
    const el = document.getElementById('page-title');
    if (el) el.textContent = t(`nav.${state.page}`);
    document.title = `${t(`nav.${state.page}`)} · Agentic Trade`;
  }

  // ------------------------------------------------------------------ helpers
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const money = (v, d = 2) => (v == null || Number.isNaN(Number(v))) ? '–'
    : `${Number(v) < 0 ? '−' : ''}$${Math.abs(Number(v)).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })}`;
  const smoney = (v, d = 2) => (v == null) ? '–' : `${Number(v) >= 0 ? '+' : '−'}$${Math.abs(Number(v)).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })}`;
  const pct = (v, d = 1) => (v == null || Number.isNaN(v)) ? '–' : `${(v * 100).toFixed(d)}%`;
  const spct = (v, d = 1) => (v == null || Number.isNaN(v)) ? '–' : `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(d)}%`;
  const num = (v, d = 2) => (v == null || Number.isNaN(Number(v))) ? '–' : Number(v).toFixed(d);
  const signCls = (v) => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
  const action = (a) => t(`act.${String(a || 'Hold').toLowerCase()}`);
  const badge = (a) => `<span class="chip ${String(a || 'hold').toLowerCase()}">${esc(action(a))}</span>`;
  const time = (iso) => iso ? new Date(iso).toLocaleTimeString(state.lang === 'bn' ? 'bn-BD' : [], { hour: '2-digit', minute: '2-digit' }) : '–';
  const dateTime = (iso) => iso ? new Date(iso).toLocaleString(state.lang === 'bn' ? 'bn-BD' : [], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '–';

  async function api(path, body) {
    const res = await fetch(path, body !== undefined ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    let data;
    try { data = await res.json(); } catch (e) { throw new Error(res.statusText || 'bad response'); }
    if (!res.ok || data.error) { const err = new Error(data.error || res.statusText); err.data = data; throw err; }
    return data;
  }

  let toastTimer;
  function toast(msg, isErr = false) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.className = `toast show ${isErr ? 'err' : 'ok'}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 4000);
  }

  function setAlert(id, html, kind = 'warn') {
    const box = document.getElementById('alerts');
    if (!box) return;
    let el = box.querySelector(`[data-alert="${id}"]`);
    if (!html) { if (el) el.remove(); return; }
    if (!el) { el = document.createElement('div'); el.dataset.alert = id; box.appendChild(el); }
    el.className = `alert ${kind}`;
    el.innerHTML = html;
  }

  // ------------------------------------------------------------------ boot
  function init() {
    // ?lang=bn / ?theme=dark in the URL override (and remember) the saved choice.
    const q = new URLSearchParams(location.search);
    if (q.get('lang')) state.lang = q.get('lang') === 'bn' ? 'bn' : 'en';
    if (q.get('theme')) state.theme = q.get('theme');
    if (q.get('ticker')) { state.ticker = q.get('ticker').toUpperCase(); store.set('ticker', state.ticker); }
    setTheme(state.theme);
    setLang(state.lang);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.theme === 'system') setTheme('system'); });
    document.addEventListener('click', (e) => { if (!e.target.closest('.picker')) closePicker(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closePicker(); document.body.classList.remove('nav-open'); } });
    document.getElementById('picker-search').addEventListener('input', renderPicker);
    document.getElementById('picker-search').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { const first = document.querySelector('#picker-list .picker-item'); if (first) first.click(); }
    });
    window.addEventListener('hashchange', () => { const p = location.hash.slice(1); if (p && p !== state.page) go(p); });
    renderPicker();
    // Old deep links (#analytics, #agentic, #terminal) map onto the new pages.
    const legacy = { analytics: 'insights', agentic: 'autopilot', terminal: 'trade' };
    const h = location.hash.slice(1);
    go(legacy[h] || h || 'home');
  }
  window.addEventListener('DOMContentLoaded', init);

  return {
    state, t, translate, setLang, setTheme, toggleTheme, isDark, on, emit, setTicker, setPrices, setTickers,
    togglePicker, register, go, reloadPage, esc, money, smoney, pct, spct, num, signCls, badge, action, time, dateTime,
    api, toast, setAlert, store,
    get ticker() { return state.ticker; },
  };
})();

// Globals kept for older modules and inline handlers.
const t = App.t;
function showToast(msg, isErr) { App.toast(msg, isErr); }
function switchTab(page) { App.go(page); }
