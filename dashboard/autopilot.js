/* AI Autopilot page, Budget & limits settings, and the autopilot widgets on Home. */
const Autopilot = (() => {
  // formsDrawn / budgetDrawn: the editable forms are drawn once per visit, so polling never wipes what you are typing.
  const state = { s: null, draft: null, budgetDraft: null, settingsPanel: 'budget', formsDrawn: false, budgetDrawn: false };
  const { t, esc, money, pct, num, badge } = App;
  const $ = (id) => document.getElementById(id);

  const PERM = [
    { v: 'advisory', icon: '👀', lvl: 'safest' },
    { v: 'approve', icon: '✋', lvl: 'safe' },
    { v: 'guarded', icon: '🛡️', lvl: 'balanced' },
    { v: 'full', icon: '🚀', lvl: 'auto' },
  ];
  const DECISION = [{ v: 'consensus', icon: '📏' }, { v: 'llm', icon: '🤖' }, { v: 'llm_and_consensus', icon: '🤝' }];

  async function refresh() {
    try { state.s = await App.api('/api/autopilot/status'); }
    catch (e) { return; }
    if (!state.draft) state.draft = { ...state.s.config };
    if (!state.budgetDraft) state.budgetDraft = { ...state.s.budget, allowed_tickers: [...state.s.budget.allowed_tickers] };
    renderGlobal();
    if (App.state.page === 'autopilot') renderPage(!state.formsDrawn);
    if (App.state.page === 'home') { renderHomeWidgets(); Trade.renderSetup(); }
    if (App.state.page === 'settings') renderBudget(!state.budgetDrawn);
  }

  // ------------------------------------------------------------------ global bits (sidebar, alerts)
  function renderGlobal() {
    const s = state.s; if (!s) return;
    const n = s.pending.length;
    $('nav-pending').hidden = !n; $('nav-pending').textContent = n;
    const dot = $('pilot-dot'), txt = $('pilot-text');
    if (s.kill_switch) { dot.className = 'dot bad'; txt.textContent = t('pilot.stopped'); }
    else if (s.config.enabled) { dot.className = 'dot'; txt.textContent = `${t('pilot.on')} · ${t(`perm.${s.config.permission_mode}.short`)}`; }
    else { dot.className = 'dot off'; txt.textContent = t('pilot.off'); }
    App.setAlert('kill', s.kill_switch ? `🛑 <b>${t('alert.kill')}</b> ${esc(s.kill_reason || '')}
      <button class="btn btn-sm btn-secondary" onclick="App.go('autopilot')">${t('common.open')}</button>` : '', 'bad');
    App.setAlert('pending', n && App.state.page !== 'autopilot' ? `✋ ${t('alert.pending', { n })}
      <button class="btn btn-sm" onclick="App.go('autopilot')">${t('alert.review')}</button>` : '', 'warn');
  }

  // ------------------------------------------------------------------ page
  function renderPage(forms = true) {
    const s = state.s; if (!s) return;
    const c = s.config, L = s.limits;
    const orb = $('ap-orb');
    if (s.kill_switch) { orb.className = 'pilot-orb stop'; orb.textContent = '🛑'; $('ap-state').textContent = t('ap.state.stopped'); }
    else if (c.enabled) { orb.className = 'pilot-orb run'; orb.textContent = '🤖'; $('ap-state').textContent = t('ap.state.running'); }
    else { orb.className = 'pilot-orb'; orb.textContent = '⏸'; $('ap-state').textContent = t('ap.state.paused'); }
    const bits = [];
    if (s.kill_switch) bits.push(esc(s.kill_reason));
    else if (c.enabled) {
      if (c.market_hours_only && !s.market_open) bits.push(t('ap.waitmarket'));
      if (s.next_run) bits.push(`${t('ap.next')} ${App.time(s.next_run)}`);
    } else bits.push(t('ap.paused.sub'));
    if (s.last_run) bits.push(`${t('ap.last')} ${App.dateTime(s.last_run)}`);
    bits.push(`${t(`perm.${c.permission_mode}.t`)} · ${t(`dec.${c.decision_mode}.t`)}`);
    $('ap-state-sub').innerHTML = bits.join(' · ');
    const tg = $('ap-toggle');
    tg.textContent = c.enabled ? `⏸ ${t('ap.pause')}` : `▶ ${t('ap.start')}`;
    tg.className = `btn ${c.enabled ? 'btn-secondary' : 'btn-success'}`;
    tg.disabled = s.kill_switch;
    $('ap-once').disabled = s.kill_switch || s.running_cycle;
    $('ap-kill').hidden = s.kill_switch;
    $('ap-resume').hidden = !s.kill_switch;

    // limits
    if (L) {
      const meter = (label, used, cap, fmtUsed, fmtCap, sub, inverse) => {
        const r = cap > 0 ? Math.min(1, used / cap) : 0;
        const cls = r >= 1 ? 'neg' : r >= 0.75 ? 'warn' : inverse ? 'pos' : '';
        return `<div class="gauge"><div class="row"><b>${label}</b><span class="num">${fmtUsed} / ${fmtCap}</span></div>
          <div class="meter ${cls}"><b style="width:${(r * 100).toFixed(1)}%"></b></div><span class="dim">${sub}</span></div>`;
      };
      $('ap-limits').innerHTML = [
        meter(t('lim.invested'), L.invested, L.managed - L.reserve, money(L.invested, 0), money(L.managed - L.reserve, 0), t('lim.invested.sub', { r: money(L.reserve, 0) })),
        meter(t('lim.daily'), Math.max(0, -L.daily_pnl), L.daily_loss_limit, money(Math.max(0, -L.daily_pnl), 0), money(L.daily_loss_limit, 0), L.daily_loss_hit ? t('lim.hit.buys') : t('lim.daily.sub')),
        meter(t('lim.dd'), L.drawdown, L.drawdown_limit, money(L.drawdown, 0), money(L.drawdown_limit, 0), L.drawdown_hit ? t('lim.hit.buys') : t('lim.dd.sub')),
        meter(t('lim.trades'), L.trades_today, L.max_trades_per_day, L.trades_today, L.max_trades_per_day, t('lim.trades.sub')),
      ].join('') + (L.budget_exceeds_equity ? `<div class="alert warn">${t('lim.overbudget')}</div>` : '');
    } else $('ap-limits').innerHTML = `<div class="muted">${esc(s.limits_error || '')}</div>`;

    // approval queue
    $('ap-queue-card').classList.toggle('accent', s.pending.length > 0);
    $('ap-queue-bulk').innerHTML = s.pending.length > 1
      ? `<button class="btn btn-success btn-sm" onclick="Autopilot.resolveAll(true)">${t('ap.approveall')}</button><button class="btn btn-secondary btn-sm" onclick="Autopilot.resolveAll(false)">${t('ap.rejectall')}</button>` : '';
    $('ap-queue').innerHTML = s.pending.map((p) => `<div class="proposal">
        <div><div class="title">${badge(p.action)} ${esc(p.ticker)} <span class="num">${money(p.notional)}</span>
          <span class="dim">≈ ${num(p.shares, 3)} ${t('trade.shares').toLowerCase()} · ${t('ap.expires')} ${App.time(p.expires_at)}</span></div>
          <div class="why">${esc((p.decision || {}).reasoning || '')}</div>
          <div class="why">⚠ ${p.why_approval.map(esc).join(' · ')}${(p.decision || {}).confidence != null ? ` · ${t('ap.conf')} ${pct(p.decision.confidence, 0)}` : ''}</div></div>
        <div class="row"><button class="btn btn-success btn-sm" onclick="Autopilot.resolve('${p.id}', true)">✓ ${t('ap.approve')}</button>
          <button class="btn btn-secondary btn-sm" onclick="Autopilot.resolve('${p.id}', false)">✕ ${t('ap.reject')}</button></div></div>`).join('')
      || `<div class="empty-state">${t('ap.queue.empty')}</div>`;

    // log
    $('ap-log').innerHTML = s.log.map((e) => `<div class="log-row"><span class="dim">${App.time(e.time)}</span>
        <span><span class="chip ${esc(e.status)}">${esc(statusLabel(e.status))}</span></span>
        <span>${e.ticker ? `<b>${esc(e.ticker)}</b>` : ''} ${e.action && e.action !== 'Hold' ? esc(App.action(e.action)) : ''}</span>
        <span class="why">${e.notional ? `<b>${money(e.notional)}</b> · ` : ''}${esc(e.reason || '')}${e.via ? ` <span class="dim">· ${esc(e.via)}</span>` : ''}</span></div>`).join('')
      || `<div class="empty-state">${t('ap.log.empty')}</div>`;

    if (forms) renderForms();
  }
  const statusLabel = (st) => { const k = `status.${st}`; const v = t(k); return v === k ? st : v; };

  function renderForms() {
    const d = state.draft; if (!d) return;
    state.formsDrawn = true;
    $('ap-perm').innerHTML = PERM.map((p) => `<button class="option-card ${d.permission_mode === p.v ? 'active' : ''}" onclick="Autopilot.set('permission_mode','${p.v}')">
        <span class="lvl" style="color:var(--${p.lvl === 'auto' ? 'warn' : p.lvl === 'balanced' ? 'primary' : 'pos'})">${t(`lvl.${p.lvl}`)}</span>
        <span class="t">${p.icon} ${t(`perm.${p.v}.t`)}</span><span class="d">${t(`perm.${p.v}.d`)}</span></button>`).join('');
    const guarded = d.permission_mode === 'guarded', approve = d.permission_mode === 'approve';
    $('ap-guarded').innerHTML = (guarded ? `
      <div class="field"><label>${t('ap.minconf')}: <b id="ap-conf-v">${pct(d.min_confidence, 0)}</b></label>
        <input type="range" min="0" max="1" step="0.05" value="${d.min_confidence}" oninput="Autopilot.set('min_confidence', +this.value, true);document.getElementById('ap-conf-v').textContent=Math.round(this.value*100)+'%'">
        <span class="hint">${t('ap.minconf.hint')}</span></div>
      <div class="field"><label>${t('ap.automax')}</label><div class="input-affix pre"><span>$</span>
        <input type="number" min="0" step="50" value="${d.auto_max_order_usd}" onchange="Autopilot.set('auto_max_order_usd', +this.value, true)"></div>
        <span class="hint">${t('ap.automax.hint')}</span></div>` : '') + (approve ? `
      <div class="field"><label class="check"><input type="checkbox" ${d.auto_exits ? 'checked' : ''} onchange="Autopilot.set('auto_exits', this.checked, true)"> ${t('ap.autoexits')}</label>
        <span class="hint">${t('ap.autoexits.hint')}</span></div>` : '');
    $('ap-decision').innerHTML = DECISION.map((o) => `<button class="option-card ${d.decision_mode === o.v ? 'active' : ''}" onclick="Autopilot.set('decision_mode','${o.v}')">
        <span class="t">${o.icon} ${t(`dec.${o.v}.t`)}</span><span class="d">${t(`dec.${o.v}.d`)}</span></button>`).join('');
    const opt = (v, label, cur) => `<option value="${v}" ${String(cur) === String(v) ? 'selected' : ''}>${label}</option>`;
    $('ap-schedule').innerHTML = `
      <div class="field"><label>${t('ap.every')}</label><select onchange="Autopilot.set('interval_min', +this.value, true)">
        ${[5, 15, 30, 60, 240, 1440].map((m) => opt(m, m < 60 ? t('ap.min', { n: m }) : m < 1440 ? t('ap.hours', { n: m / 60 }) : t('ap.daily'), d.interval_min)).join('')}</select></div>
      <div class="field"><label>${t('ap.percycle')}</label><select onchange="Autopilot.set('tickers_per_cycle', +this.value, true)">
        ${[1, 2, 3, 5, 10].map((n) => opt(n, n, d.tickers_per_cycle)).join('')}</select><span class="hint">${t('ap.percycle.hint')}</span></div>
      <div class="field"><label>${t('ap.sizing')}</label><select onchange="Autopilot.set('sizing', this.value)">
        ${opt('max_position', t('ap.sizing.max'), d.sizing)}${opt('allocation', t('ap.sizing.alloc'), d.sizing)}</select></div>
      ${d.sizing === 'allocation' ? `<div class="field"><label>${t('ap.method')}</label><select onchange="Autopilot.set('allocation_method', this.value, true)">
        ${['inverse_vol', 'min_variance', 'equal', 'signal_tilt'].map((m) => opt(m, t(`split.m.${m}`), d.allocation_method)).join('')}</select></div>` : ''}
      <div class="field"><label>&nbsp;</label><label class="check"><input type="checkbox" ${d.market_hours_only ? 'checked' : ''} onchange="Autopilot.set('market_hours_only', this.checked, true)"> ${t('ap.markethours')}</label></div>`;
    $('ap-save-note').textContent = dirty() ? t('ap.unsaved') : '';
  }

  const CONFIG_KEYS = ['permission_mode', 'decision_mode', 'interval_min', 'market_hours_only', 'tickers_per_cycle', 'sizing',
    'allocation_method', 'min_confidence', 'auto_max_order_usd', 'auto_exits'];
  const dirty = () => state.s && CONFIG_KEYS.some((k) => String(state.draft[k]) !== String(state.s.config[k]));

  function set(key, value, quiet) {
    state.draft[key] = value;
    if (quiet) $('ap-save-note').textContent = dirty() ? t('ap.unsaved') : '';
    else renderForms();
  }

  async function saveConfig(extra = {}) {
    const cfg = Object.fromEntries(CONFIG_KEYS.map((k) => [k, state.draft[k]]));
    if (cfg.permission_mode === 'full' && state.s.config.permission_mode !== 'full' && !confirm(t('ap.full.confirm'))) return false;
    try {
      state.s = await App.api('/api/autopilot/config', { config: { ...cfg, ...extra } });
      state.draft = { ...state.s.config };
      App.toast(t('common.saved'));
      renderGlobal(); renderPage();
      return true;
    } catch (e) { App.toast(e.message, true); return false; }
  }

  async function toggle() {
    const s = state.s; if (!s) return;
    if (!s.config.enabled) {
      if (!confirm(t('ap.start.confirm', { mode: t(`perm.${state.draft.permission_mode}.t`) }))) return;
      await saveConfig({ enabled: true });
    } else {
      try { state.s = await App.api('/api/autopilot/config', { config: { enabled: false } }); App.toast(t('ap.paused.toast')); renderGlobal(); renderPage(false); }
      catch (e) { App.toast(e.message, true); }
    }
  }

  async function runOnce() {
    const btn = $('ap-once');
    btn.disabled = true;
    App.toast(t('ap.once.running'));
    try {
      const r = await App.api('/api/autopilot/run_once', {});
      state.s = r.status;
      const counts = {};
      r.results.forEach((x) => { counts[x.status] = (counts[x.status] || 0) + 1; });
      App.toast(t('ap.once.done', { s: Object.entries(counts).map(([k, v]) => `${v} ${statusLabel(k)}`).join(', ') }));
      renderGlobal(); renderPage(false);
      if (window.Trade) Trade.fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
    finally { btn.disabled = false; }
  }

  async function kill() {
    if (!confirm(t('ap.kill.confirm'))) return;
    const closeAll = confirm(t('ap.kill.close'));
    try {
      const r = await App.api('/api/autopilot/kill', { close_all: closeAll, reason: t('ap.kill.reason') });
      state.s = r.status;
      App.toast(t('ap.kill.done', { n: r.cancelled }) + (closeAll ? ` ${t('ap.kill.closed', { n: r.closed.filter((x) => x.success).length })}` : ''), false);
      renderGlobal(); renderPage(false);
      if (window.Trade) Trade.fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  async function resume() {
    if (!confirm(t('ap.resume.confirm'))) return;
    try { state.s = await App.api('/api/autopilot/resume', {}); App.toast(t('ap.resumed')); renderGlobal(); renderPage(false); }
    catch (e) { App.toast(e.message, true); }
  }

  async function resolve(id, approve) {
    try {
      const r = await App.api('/api/autopilot/resolve', { id, approve });
      state.s = r.status;
      const p = r.results[0].proposal;
      App.toast(approve ? (p.status === 'approved' ? t('ap.approved', { tk: p.ticker }) : `${t('ap.notexec')}: ${p.note || p.status}`) : t('ap.rejected'), approve && p.status !== 'approved');
      renderGlobal(); renderPage(false);
      if (window.Trade) Trade.fetchPortfolio();
    } catch (e) { App.toast(e.message, true); refresh(); }
  }

  async function resolveAll(approve) {
    const ids = state.s.pending.map((p) => p.id);
    if (!confirm(t(approve ? 'ap.approveall.confirm' : 'ap.rejectall.confirm', { n: ids.length }))) return;
    try {
      const r = await App.api('/api/autopilot/resolve', { ids, approve });
      state.s = r.status;
      const ok = r.results.filter((x) => x.proposal.status === 'approved').length;
      App.toast(approve ? t('ap.approvedn', { ok, n: ids.length }) : t('ap.rejected'));
      renderGlobal(); renderPage(false);
      if (window.Trade) Trade.fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  // ------------------------------------------------------------------ settings: budget
  const BUDGET_FIELDS = [
    { k: 'total_budget', pre: '$', step: 500, min: 100 },
    { k: 'cash_reserve_pct', post: '%', step: 1, min: 0, max: 95 },
    { k: 'max_position_pct', post: '%', step: 1, min: 1, max: 100 },
    { k: 'max_order_usd', pre: '$', step: 100, min: 1 },
    { k: 'max_daily_loss_pct', post: '%', step: 0.5, min: 0.1, max: 50 },
    { k: 'max_drawdown_pct', post: '%', step: 1, min: 0.5, max: 90 },
    { k: 'max_trades_per_day', step: 1, min: 0, max: 500 },
  ];

  function renderBudget(forms = true) {
    const b = state.budgetDraft; if (!b) return;
    if (forms) {
      state.budgetDrawn = true;
      $('bud-form').innerHTML = BUDGET_FIELDS.map((f) => {
        const input = `<input type="number" id="bud-${f.k}" value="${b[f.k]}" step="${f.step}" min="${f.min}" ${f.max != null ? `max="${f.max}"` : ''} oninput="Autopilot.budgetInput('${f.k}', this.value)">`;
        return `<div class="field"><label for="bud-${f.k}">${t(`bud.${f.k}`)}</label>
          ${f.pre ? `<div class="input-affix pre"><span>${f.pre}</span>${input}</div>` : f.post ? `<div class="input-affix">${input}<span>${f.post}</span></div>` : input}
          <span class="hint">${t(`bud.${f.k}.hint`)}</span></div>`;
      }).join('');
      $('bud-tickers').innerHTML = state.s.options.universe.map((tk) => `<button class="chip chip-toggle ${b.allowed_tickers.includes(tk) ? 'on' : ''}"
        onclick="Autopilot.toggleTicker('${tk}', this)">${tk}</button>`).join('') +
        `<button class="btn btn-ghost btn-sm" onclick="Autopilot.allTickers(true)">${t('bud.all')}</button><button class="btn btn-ghost btn-sm" onclick="Autopilot.allTickers(false)">${t('bud.none')}</button>`;
    }
    const inv = b.total_budget * (1 - b.cash_reserve_pct / 100);
    const p = (Trade.portfolio() || {});
    $('bud-summary').innerHTML = `<span>💡 ${t('bud.summary', {
      budget: money(b.total_budget, 0), inv: money(inv, 0), reserve: money(b.total_budget - inv, 0),
      per: money(b.total_budget * b.max_position_pct / 100, 0), order: money(b.max_order_usd, 0),
      loss: money(b.total_budget * b.max_daily_loss_pct / 100, 0), dd: money(b.total_budget * b.max_drawdown_pct / 100, 0),
      n: b.max_trades_per_day, k: b.allowed_tickers.length })}` +
      (p.equity && b.total_budget > p.equity ? `<br>⚠️ ${t('bud.over', { eq: money(p.equity, 0) })}` : '') + '</span>';
    const changed = state.s && JSON.stringify(b) !== JSON.stringify(state.s.budget);
    $('bud-note').textContent = changed ? t('ap.unsaved') : (state.s && state.s.budget_saved ? '' : t('bud.defaults'));
  }
  function budgetInput(k, v) { state.budgetDraft[k] = k === 'max_trades_per_day' ? parseInt(v || 0, 10) : Number(v); renderBudget(false); }
  function toggleTicker(tk, el) {
    const list = state.budgetDraft.allowed_tickers;
    const i = list.indexOf(tk);
    if (i >= 0) list.splice(i, 1); else list.push(tk);
    el.classList.toggle('on', i < 0);
    renderBudget(false);
  }
  function allTickers(on) { state.budgetDraft.allowed_tickers = on ? [...state.s.options.universe] : []; renderBudget(); }

  async function saveBudget() {
    const b = state.budgetDraft;
    if (!b.allowed_tickers.length) { App.toast(t('bud.err.tickers'), true); return; }
    try {
      state.s = await App.api('/api/autopilot/config', { budget: b });
      state.budgetDraft = { ...state.s.budget, allowed_tickers: [...state.s.budget.allowed_tickers] };
      App.toast(t('bud.saved'));
      renderBudget();
      renderGlobal();
    } catch (e) { App.toast(e.message, true); }
  }

  function openSettings(panel) {
    state.settingsPanel = panel;
    App.go('settings');
  }
  function showSettingsPanel(panel) {
    state.settingsPanel = panel;
    document.querySelectorAll('#set-tabs button').forEach((b) => b.classList.toggle('active', b.dataset.panel === panel));
    document.querySelectorAll('#page-settings .panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${panel}`));
    if (panel === 'keys' && window.CredentialsApp) CredentialsApp.load();
    if (panel === 'budget') renderBudget();
  }

  // ------------------------------------------------------------------ home widgets
  function renderHomeWidgets() {
    const s = state.s; if (!s) return;
    const c = s.config, L = s.limits;
    $('home-pilot-sub').textContent = s.kill_switch ? t('ap.state.stopped') : c.enabled ? t('ap.state.running') : t('ap.state.paused');
    $('home-pilot-body').innerHTML = `
      <div class="stats">
        <div class="stat"><span class="k">${t('home.perm')}</span><span class="v">${t(`perm.${c.permission_mode}.t`)}</span></div>
        <div class="stat"><span class="k">${t('home.decides')}</span><span class="v">${t(`dec.${c.decision_mode}.t`)}</span></div>
        <div class="stat"><span class="k">${t('home.pending')}</span><span class="v ${s.pending.length ? 'neg' : ''}">${s.pending.length}</span></div>
        <div class="stat"><span class="k">${t('home.tradestoday')}</span><span class="v">${L ? `${L.trades_today} / ${L.max_trades_per_day}` : '–'}</span></div>
      </div>
      ${s.pending.length ? `<button class="btn btn-block" onclick="App.go('autopilot')">✋ ${t('alert.pending', { n: s.pending.length })}</button>` : ''}`;
    if (!L) { $('home-limits').innerHTML = ''; return; }
    const bar = (label, used, cap, txt) => {
      const r = cap > 0 ? Math.min(1, used / cap) : 0;
      return `<div class="gauge"><div class="row"><span>${label}</span><b class="num">${txt}</b></div><div class="meter ${r >= 1 ? 'neg' : r >= 0.75 ? 'warn' : ''}"><b style="width:${(r * 100).toFixed(1)}%"></b></div></div>`;
    };
    $('home-limits').innerHTML = [
      bar(t('lim.invested'), L.invested, L.managed - L.reserve, `${money(L.invested, 0)} / ${money(L.managed - L.reserve, 0)}`),
      bar(t('lim.daily'), Math.max(0, -L.daily_pnl), L.daily_loss_limit, `${money(Math.max(0, -L.daily_pnl), 0)} / ${money(L.daily_loss_limit, 0)}`),
      bar(t('lim.dd'), L.drawdown, L.drawdown_limit, `${money(L.drawdown, 0)} / ${money(L.drawdown_limit, 0)}`),
    ].join('') + `<div class="dim">${t('home.budgetline', { b: money(s.budget.total_budget, 0), r: s.budget.cash_reserve_pct, p: s.budget.max_position_pct })}</div>`;
  }

  // ------------------------------------------------------------------ wiring
  App.register('autopilot', () => { state.formsDrawn = false; renderPage(); refresh(); });
  App.register('settings', () => { state.budgetDrawn = false; showSettingsPanel(state.settingsPanel); refresh(); });
  App.on('lang', () => { renderGlobal(); if (App.state.page === 'autopilot') renderPage(); if (App.state.page === 'settings') renderBudget(); });
  window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('#set-tabs button').forEach((b) => b.addEventListener('click', () => showSettingsPanel(b.dataset.panel)));
    refresh();
    setInterval(() => { if (!document.hidden) refresh(); }, 8000);
  });

  return { refresh, toggle, runOnce, kill, resume, resolve, resolveAll, set, saveConfig, saveBudget, budgetInput, toggleTicker,
    allTickers, openSettings, renderHomeWidgets, data: () => state.s, recentLog: (n) => (state.s ? state.s.log.slice(0, n) : []) };
})();
