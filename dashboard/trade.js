/* Home, Trade and Portfolio pages + general preferences. */
const Trade = (() => {
  const COMPANIES = { TSLA: 'Tesla', NFLX: 'Netflix', AMZN: 'Amazon', MSFT: 'Microsoft', COIN: 'Coinbase', NVDA: 'NVIDIA',
    AAPL: 'Apple', GOOGL: 'Alphabet (Google)', META: 'Meta Platforms', AMD: 'AMD' };
  const state = { portfolio: null, market: null, tech: null, side: 'Buy', unit: 'usd', brief: null, briefTicker: null, settings: null };
  const { t, esc, money, smoney, pct, spct, num, signCls, badge } = App;
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ data
  async function bootstrap() {
    App.setTickers(Object.keys(COMPANIES), COMPANIES);
    try {
      const s = await App.api('/api/summary');
      const extra = (s.tickers || []).filter((x) => !COMPANIES[x]);
      App.setTickers([...Object.keys(COMPANIES), ...extra], COMPANIES);
      App.setPrices(s.latest_prices || {});
    } catch (e) { /* prices fill in from the market endpoint */ }
    fetchPortfolio();
    setInterval(() => { if (!document.hidden) fetchPortfolio(); }, 10000);
    setInterval(() => { if (!document.hidden && App.state.page === 'trade') load(); }, 15000);
  }

  async function fetchPortfolio() {
    try {
      state.portfolio = await App.api('/api/portfolio');
    } catch (e) { return; }
    const p = state.portfolio;
    const alpaca = p.execution_mode === 'alpaca_paper';
    $('mode-dot').className = `dot ${alpaca ? '' : 'warn'}`;
    $('mode-text').textContent = alpaca ? t('mode.alpaca') : t('mode.local');
    $('btn-reset').hidden = alpaca;
    $('prefs-local').hidden = alpaca;
    const prices = {};
    (p.positions || []).forEach((x) => { prices[x.ticker] = x.current_price; });
    App.setPrices(prices);
    if (App.state.page === 'home') renderHome();
    if (App.state.page === 'portfolio') renderPortfolio();
    if (App.state.page === 'trade') { renderPosition(); updateTicket(); }
  }

  const positions = () => {
    const p = state.portfolio || {};
    return p.execution_mode === 'alpaca_paper' ? (p.positions || []) : [...(p.positions || []), ...(p.alpaca_positions || [])];
  };
  const held = (tk) => positions().find((x) => x.ticker === tk);
  const signed = (x) => (x.side === 'Short' ? -1 : 1) * Number(x.current_value || 0);

  // ------------------------------------------------------------------ home
  async function loadHome() {
    document.querySelectorAll('.tk-label').forEach((el) => { el.textContent = App.ticker; });
    renderHome();
    if (state.briefTicker !== App.ticker) {
      state.brief = null;
      $('home-view').innerHTML = `<div class="muted">${t('common.loading')}</div>`;
      try {
        state.brief = await App.api(`/api/agent/brief?ticker=${App.ticker}`);
        state.briefTicker = App.ticker;
      } catch (e) {
        $('home-view').innerHTML = `<div class="muted">${esc(t('home.view.err'))}</div>`;
        return;
      }
    }
    renderHomeView();
  }

  function renderHome() {
    const p = state.portfolio;
    if (p) {
      const invested = positions().reduce((a, x) => a + Math.abs(signed(x)), 0);
      $('home-equity').textContent = money(p.equity);
      const d = (p.risk || {}).daily_pnl;
      $('home-today').innerHTML = d == null ? '' : `<span class="${signCls(d)}">${smoney(d)}</span> ${t('home.today')}`;
      $('home-cash').textContent = money(p.cash, 0);
      $('home-invested').textContent = money(invested, 0);
      $('home-upnl').innerHTML = `<span class="${signCls(p.unrealized_pnl)}">${smoney(p.unrealized_pnl, 0)}</span>`;
      $('home-npos').textContent = positions().length;
      $('home-mode').textContent = p.execution_mode === 'alpaca_paper' ? t('mode.alpaca.short') : t('mode.local.short');
    }
    if (window.Autopilot) Autopilot.renderHomeWidgets();
    renderActivity();
    renderSetup();
  }

  function renderHomeView() {
    const b = state.brief;
    if (!b) return;
    const c = b.consensus;
    const conf = c.confidence;
    const meaning = t(`home.meaning.${c.action.toLowerCase()}`);
    $('home-view').innerHTML = `
      <div class="verdict"><div><div class="big ${c.action.toLowerCase()}">${esc(App.action(c.action))}</div>
        <div class="dim">${t('home.conf')} ${pct(conf, 0)} · ${t('home.asof')} ${esc(b.date)}</div></div></div>
      <div class="muted">${esc(meaning)}</div>
      <div class="stats">
        <div class="stat"><span class="k">${t('home.price')}</span><span class="v">${money(b.price)}</span><span class="s ${signCls(b.change_1d)}">${spct(b.change_1d, 2)} ${t('home.1d')}</span></div>
        <div class="stat"><span class="k">${t('home.20d')}</span><span class="v ${signCls(b.change_20d)}">${spct(b.change_20d)}</span></div>
        <div class="stat"><span class="k">${t('home.pup')}</span><span class="v">${b.price_model.p_up == null ? '–' : pct(b.price_model.p_up, 0)}</span><span class="s">${t('home.pup.sub')}</span></div>
        <div class="stat"><span class="k">${t('home.news')}</span><span class="v">${b.n_today ?? 0}</span><span class="s">${t('home.news.sub')}</span></div>
      </div>
      <div class="dim">${t('home.caveat')}</div>`;
  }

  function renderActivity() {
    const box = $('home-activity');
    const orders = ((state.portfolio || {}).recent_orders || []).slice(0, 6).map((o) => ({
      time: o.timestamp, ticker: o.ticker, action: o.action || o.decision, status: o.status === 'filled' || o.order_id ? 'executed' : o.status,
      why: o.reasoning, src: o.is_ai ? 'AI' : t('common.manual') }));
    const log = (window.Autopilot ? Autopilot.recentLog(6) : []).filter((e) => e.status !== 'executed').map((e) => ({
      time: e.time, ticker: e.ticker || '', action: e.action, status: e.status, why: e.reason, src: t('nav.autopilot.short') }));
    const rows = [...orders, ...log].filter((r) => r.time).sort((a, b) => String(b.time).localeCompare(String(a.time))).slice(0, 8);
    box.innerHTML = rows.map((r) => `<div class="activity-row">
        <span class="dim when">${App.dateTime(r.time)}</span>
        <span>${r.action ? badge(r.action) : ''} <b>${esc(r.ticker)}</b></span>
        <span class="why" title="${esc(r.why || '')}">${esc(r.why || '')}</span>
        <span class="chip ${esc(r.status || '')}">${esc(statusLabel(r.status))}</span></div>`).join('')
      || `<div class="empty-state">${t('home.noactivity')}</div>`;
  }
  const statusLabel = (s) => { const k = `status.${s || 'recorded'}`; const v = t(k); return v === k ? (s || '') : v; };

  // First-run guide: drawn only once every status it depends on has loaded (so it never flashes up),
  // and once all three steps are done (or the user hides it) it stays gone for good.
  async function renderSetup() {
    const card = $('home-setup');
    if (App.store.get('setupDone') === '1') { card.hidden = true; return; }
    let cred = window.CredentialsApp && CredentialsApp.statusData();
    if (!cred) { try { cred = await App.api('/api/credentials'); } catch (e) { return; } }
    const pilot = window.Autopilot && Autopilot.data();
    if (!pilot) { card.hidden = true; return; }
    const steps = [
      { done: cred.ready.broker, title: t('setup.s1'), sub: t('setup.s1.sub'), go: () => Autopilot.openSettings('keys') },
      { done: cred.ready.llm, title: t('setup.s2'), sub: t('setup.s2.sub'), go: () => Autopilot.openSettings('keys') },
      { done: !!pilot.budget_saved, title: t('setup.s3'), sub: t('setup.s3.sub'), go: () => Autopilot.openSettings('budget') },
    ];
    const all = steps.every((s) => s.done);
    if (all) App.store.set('setupDone', '1');
    card.hidden = all;
    $('home-steps').innerHTML = steps.map((s, i) => `<div class="step ${s.done ? 'done' : ''}"><span class="n">${s.done ? '✓' : i + 1}</span>
      <div class="grow"><b>${esc(s.title)}</b><span>${esc(s.sub)}</span></div>
      ${s.done ? '' : `<button class="btn btn-sm" data-step="${i}">${t('setup.go')}</button>`}</div>`).join('');
    $('home-steps').querySelectorAll('[data-step]').forEach((b) => b.addEventListener('click', () => steps[+b.dataset.step].go()));
    App.setAlert('setup', !cred.ready.broker && !cred.ready.llm
      ? `🔑 ${esc(t('alert.nokeys'))} <button class="btn btn-sm" onclick="Autopilot.openSettings('keys')">${t('alert.addkeys')}</button>` : '', 'info');
  }

  // ------------------------------------------------------------------ trade
  async function load() {
    const tk = App.ticker;
    document.querySelectorAll('.tk-label').forEach((el) => { el.textContent = tk; });
    $('trade-company').textContent = COMPANIES[tk] || '';
    try {
      const [m, tech] = await Promise.all([App.api(`/api/market_data?ticker=${tk}`), App.api(`/api/technical_analysis?ticker=${tk}`)]);
      if (tk !== App.ticker) return;
      state.market = m; state.tech = tech;
      App.setPrices({ [tk]: m.latest_price });
    } catch (e) { App.toast(`${t('err.market')}: ${e.message}`, true); return; }
    renderQuote(); draw(); renderSignals(); renderPosition(); renderPresets(); updateTicket();
  }

  function renderQuote() {
    const m = state.market;
    $('trade-price').textContent = money(m.latest_price);
    const ch = $('trade-change');
    ch.textContent = `${m.change_pct >= 0 ? '▲' : '▼'} ${Math.abs(m.change_pct).toFixed(2)}%`;
    ch.className = `chg num ${signCls(m.change_pct)}`;
    $('trade-source').textContent = `${m.price_source === 'alpaca_paper_live' ? t('trade.live') : t('trade.hist')} · ${String(m.as_of || '').slice(0, 16).replace('T', ' ')}`;
  }

  function draw() {
    const m = state.market;
    if (!m || !m.dates || !m.dates.length) return;
    const on = (id) => $(id).checked;
    const ind = m.indicators || {};
    Charts.price($('trade-chart'), { dates: m.dates, candles: m.candles || [], prices: m.prices, ind }, {
      height: 380, candles: on('chk-candles'), sma20: on('chk-sma20'), sma50: on('chk-sma50'), bands: on('chk-bb'),
      labels: { open: t('trade.open'), close: t('trade.close'), sma20: t('trade.sma20'), sma50: t('trade.sma50') } });
    $('rsi-panel').hidden = !on('chk-rsi');
    if (on('chk-rsi') && ind.rsi_14) Charts.oscillator($('trade-rsi'), m.dates.length, { lo: 0, hi: 100, height: 110,
      lines: [{ values: ind.rsi_14, color: 'var(--yellow)' }], levels: [{ v: 70, color: 'var(--neg)' }, { v: 30, color: 'var(--pos)' }] });
    $('macd-panel').hidden = !on('chk-macd');
    if (on('chk-macd') && ind.macd) {
      const lim = Math.max(0.5, ...ind.macd.map((v) => Math.abs(v || 0)), ...(ind.macd_hist || []).map((v) => Math.abs(v || 0)));
      Charts.oscillator($('trade-macd'), m.dates.length, { lo: -lim, hi: lim, height: 110, hist: ind.macd_hist,
        lines: [{ values: ind.macd, color: 'var(--primary)' }, { values: ind.macd_signal, color: 'var(--orange)' }], levels: [{ v: 0, color: 'var(--grid)' }] });
    }
  }

  function renderSignals() {
    const tech = state.tech;
    if (!tech || !tech.signals) return;
    const s = tech.signals, score = tech.score || 0;
    const rec = String(tech.recommendation || 'NEUTRAL').toLowerCase().replace(' ', '_');
    const cls = score >= 1 ? 'buy' : score <= -1 ? 'sell' : 'hold';
    $('trade-rec').innerHTML = `<span class="chip ${cls}">${t(`rec.${rec}`)}</span>`;
    $('trade-score-dot').style.left = `${((score + 6) / 12) * 100}%`;
    $('trade-score-text').textContent = `${t('trade.score')} ${score > 0 ? '+' : ''}${score} / ±6`;
    const rsi = s.rsi ? s.rsi.value : null;
    const rsiTxt = rsi == null ? '' : rsi > 70 ? t('sig.rsi.hot') : rsi < 30 ? t('sig.rsi.cold') : rsi >= 50 ? t('sig.rsi.up') : t('sig.rsi.down');
    const macdUp = s.macd && s.macd.signal === 'Bullish';
    const ma = s.moving_averages ? s.moving_averages.signal : '';
    const maKey = { 'Strong Bullish': 'sig.ma.strongup', Bullish: 'sig.ma.up', Bearish: 'sig.ma.down', 'Strong Bearish': 'sig.ma.strongdown' }[ma] || 'sig.ma.down';
    const pb = s.bollinger ? s.bollinger.percent_b : null;
    const bbTxt = pb == null ? '' : pb > 80 ? t('sig.bb.high') : pb < 20 ? t('sig.bb.low') : t('sig.bb.mid');
    const item = (name, val, desc, help) => `<div class="stat"><span class="k">${name} <span class="help" title="${esc(help)}">?</span></span><span class="v">${val}</span><span class="s">${esc(desc)}</span></div>`;
    $('trade-sigs').innerHTML = [
      item(t('sig.rsi'), num(rsi, 1), rsiTxt, t('sig.rsi.help')),
      item(t('sig.macd'), s.macd ? num(s.macd.macd) : '–', macdUp ? t('sig.macd.up') : t('sig.macd.down'), t('sig.macd.help')),
      item(t('sig.ma'), `<span class="${maKey.includes('up') ? 'pos' : 'neg'}">${maKey.includes('up') ? '↗' : '↘'}</span>`, t(maKey), t('sig.ma.help')),
      item(t('sig.bb'), pb == null ? '–' : `${num(pb, 0)}%`, bbTxt, t('sig.bb.help')),
    ].join('');
  }

  function renderPosition() {
    const tk = App.ticker, pos = held(tk);
    if (!pos) {
      $('pos-status').innerHTML = `<span class="chip">${t('pos.none')}</span>`;
      $('pos-body').innerHTML = `<div class="muted">${t('pos.none.sub', { tk: esc(tk) })}</div>`;
      return;
    }
    $('pos-status').innerHTML = `<span class="chip ${pos.side === 'Short' ? 'short' : 'long'}">${t(pos.side === 'Short' ? 'pos.short' : 'pos.long')}</span>`;
    $('pos-body').innerHTML = `
      <div class="row between"><div><div class="dim">${t('pos.pnl')}</div>
        <div style="font-size:1.4rem;font-weight:800" class="${signCls(pos.unrealized_pnl)}">${smoney(pos.unrealized_pnl)} (${pos.unrealized_pnl_pct >= 0 ? '+' : ''}${num(pos.unrealized_pnl_pct)}%)</div></div>
        <div style="text-align:right"><div class="dim">${t('pos.value')}</div><div style="font-size:1.1rem;font-weight:750">${money(pos.current_value)}</div></div></div>
      <div class="stats">
        <div class="stat"><span class="k">${t('pos.shares')}</span><span class="v">${num(pos.shares, 4).replace(/\.?0+$/, '')}</span></div>
        <div class="stat"><span class="k">${t('pos.avg')}</span><span class="v">${money(pos.avg_price)}</span></div>
        <div class="stat"><span class="k">${t('pos.cost')}</span><span class="v">${money(pos.invested_cost)}</span></div>
      </div>
      <div class="row"><button class="btn btn-danger" style="flex:1" onclick="Trade.closeTicker(1)">${t('pos.sellall')}</button>
        <button class="btn btn-secondary" onclick="Trade.closeTicker(0.5)">${t('pos.sellhalf')}</button></div>`;
  }

  function setSide(side) {
    state.side = side;
    $('ord-buy').classList.toggle('active', side === 'Buy');
    $('ord-sell').classList.toggle('active', side === 'Sell');
    renderPresets(); updateTicket();
  }
  function setUnit(unit) {
    const px = price();
    const v = Number($('ord-amount').value) || 0;
    if (unit !== state.unit && px) $('ord-amount').value = unit === 'usd' ? (v * px).toFixed(2) : (v / px).toFixed(4);
    state.unit = unit;
    $('unit-usd').classList.toggle('active', unit === 'usd');
    $('unit-sh').classList.toggle('active', unit === 'shares');
    $('ord-affix').textContent = unit === 'usd' ? '$' : '#';
    updateTicket();
  }
  const price = () => (state.market && state.market.ticker === App.ticker ? state.market.latest_price : App.state.prices[App.ticker]) || 0;

  function renderPresets() {
    const fr = state.side === 'Buy' ? [0.05, 0.1, 0.25] : [0.25, 0.5, 1];
    $('ord-presets').innerHTML = fr.map((f) => `<button class="btn btn-secondary btn-sm" onclick="Trade.preset(${f})">${f * 100}% ${state.side === 'Buy' ? t('trade.ofcash') : t('trade.ofheld')}</button>`).join('');
  }
  function preset(f) {
    const px = price();
    if (!px) return;
    let sh;
    if (state.side === 'Buy') sh = ((state.portfolio || {}).cash || 0) * f / px;
    else { const pos = held(App.ticker); sh = pos ? pos.shares * f : 0; }
    $('ord-amount').value = state.unit === 'usd' ? (sh * px).toFixed(2) : sh.toFixed(4);
    updateTicket();
  }
  function ticketShares() {
    const v = Number($('ord-amount').value) || 0, px = price();
    return state.unit === 'usd' ? (px ? v / px : 0) : v;
  }
  function updateTicket() {
    const sh = ticketShares(), px = price();
    $('ord-summary').textContent = state.unit === 'usd' ? `≈ ${num(sh, 4)} ${t('trade.sharesof', { tk: App.ticker })}` : `≈ ${money(sh * px)}`;
    const btn = $('ord-submit');
    btn.textContent = `${App.action(state.side)} ${App.ticker}`;
    btn.className = `btn btn-lg ${state.side === 'Buy' ? 'btn-success' : 'btn-danger'}`;
  }

  async function submit() {
    const sh = ticketShares(), tk = App.ticker;
    if (!(sh > 0)) { App.toast(t('trade.err.amount'), true); return; }
    const pos = held(tk);
    let msg = t('trade.confirm', { side: App.action(state.side), sh: num(sh, 4), tk, usd: money(sh * price()) });
    if (state.side === 'Sell' && (!pos || pos.shares < sh - 1e-9)) msg += `\n\n${t('trade.shortwarn')}`;
    if (!confirm(msg)) return;
    try {
      const res = await App.api('/api/manual_trade', { ticker: tk, action: state.side, shares: sh, order_type: 'Market', notes: 'Manual order from Trade page' });
      App.toast(res.success === false ? `${t('trade.blocked')}: ${res.error}` : t('trade.sent', { side: App.action(state.side), tk }), res.success === false);
      fetchPortfolio();
    } catch (e) { App.toast(`${t('trade.blocked')}: ${e.message}`, true); }
  }

  async function closeTicker(fraction) {
    const tk = App.ticker, pos = held(tk);
    if (!pos) return;
    const sh = fraction >= 1 ? null : pos.shares * fraction;
    if (!confirm(t('pos.confirm', { n: fraction >= 1 ? t('pos.all') : '50%', tk }))) return;
    try {
      const res = await App.api('/api/close_position', { ticker: tk, shares: sh });
      App.toast(res.success === false ? res.error : t('pos.closed', { tk }), res.success === false);
      fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  async function runAI() {
    const btn = $('btn-run-ai'), tk = App.ticker;
    if (!confirm(t('trade.ai.confirm', { tk }))) return;
    btn.disabled = true;
    const old = btn.innerHTML;
    btn.textContent = t('trade.ai.thinking', { tk });
    try {
      const d = await App.api('/api/ai_trade', { ticker: tk });
      const box = $('ai-result');
      box.hidden = false;
      const executed = d.order && d.order.status === 'executed';
      box.innerHTML = `<div class="row">${badge(d.action)} <span class="dim">${esc(d.provider)} · ${esc(d.model)}</span></div>
        <div style="font-size:.88rem">${esc(d.reasoning)}</div>
        <div class="chip ${executed ? 'executed' : d.action === 'Hold' ? '' : 'blocked'}">${executed ? t('trade.ai.done') : d.action === 'Hold' ? t('trade.ai.hold') : esc(d.order_error || t('trade.ai.notrade'))}</div>
        ${d.llm_error ? `<div class="dim">${t('trade.ai.fallback')}: ${esc(d.llm_error)}</div>` : ''}`;
      fetchPortfolio();
    } catch (e) { App.toast(`${t('err.ai')}: ${e.message}`, true); }
    finally { btn.disabled = false; btn.innerHTML = old; }
  }

  // ------------------------------------------------------------------ portfolio
  function renderPortfolio() {
    const p = state.portfolio;
    if (!p) return;
    const tiles = [
      ['blue', t('port.equity'), money(p.equity), `${t('port.start')}: ${money(p.initial_capital, 0)}`],
      ['', t('home.cash'), money(p.cash), t('port.cash.sub')],
      [p.unrealized_pnl >= 0 ? 'green' : 'red', t('home.profit'), `<span class="${signCls(p.unrealized_pnl)}">${smoney(p.unrealized_pnl)}</span>`, t('port.unreal.sub')],
      [p.total_pnl >= 0 ? 'green' : 'red', t('port.total'), `<span class="${signCls(p.total_pnl)}">${smoney(p.total_pnl)}</span>`, `${p.total_pnl_pct >= 0 ? '+' : ''}${num(p.total_pnl_pct)}%`],
    ];
    $('port-kpis').innerHTML = tiles.map(([c, k, v, s]) => `<div class="kpi-card ${c}"><span class="kpi-title">${k}</span><span class="kpi-val">${v}</span><span class="kpi-sub">${esc(s)}</span></div>`).join('');
    const pos = positions();
    $('positions-table').innerHTML = `<thead><tr><th>${t('col.stock')}</th><th>${t('col.side')}</th><th class="num">${t('pos.shares')}</th><th class="num">${t('pos.avg')}</th>
      <th class="num">${t('col.price')}</th><th class="num">${t('pos.value')}</th><th class="num">${t('pos.pnl')}</th><th></th></tr></thead><tbody>` +
      (pos.map((x) => `<tr><td><b>${esc(x.ticker)}</b> <span class="dim">${esc(COMPANIES[x.ticker] || '')}</span></td>
        <td><span class="chip ${x.side === 'Short' ? 'short' : 'long'}">${t(x.side === 'Short' ? 'pos.short' : 'pos.long')}</span></td>
        <td class="num">${num(x.shares, 4).replace(/\.?0+$/, '')}</td><td class="num">${money(x.avg_price)}</td><td class="num">${money(x.current_price)}</td>
        <td class="num">${money(x.current_value)}</td><td class="num ${signCls(x.unrealized_pnl)}"><b>${smoney(x.unrealized_pnl)}</b> <span class="dim">${x.unrealized_pnl_pct >= 0 ? '+' : ''}${num(x.unrealized_pnl_pct)}%</span></td>
        <td class="num"><button class="btn btn-secondary btn-sm" onclick="App.setTicker('${esc(x.ticker)}');App.go('trade')">${t('port.trade')}</button>
          <button class="btn btn-danger btn-sm" onclick="Trade.closeFromTable('${esc(x.ticker)}')">${t('port.close')}</button></td></tr>`).join('')
      || `<tr class="empty"><td colspan="8">${t('port.nopos')}</td></tr>`) + '</tbody>';
    const orders = p.recent_orders || [];
    $('orders-table').innerHTML = `<thead><tr><th>${t('col.time')}</th><th>${t('col.stock')}</th><th>${t('col.action')}</th><th class="num">${t('pos.shares')}</th>
      <th class="num">${t('col.price')}</th><th>${t('col.source')}</th><th>${t('col.status')}</th><th>${t('col.why')}</th></tr></thead><tbody>` +
      (orders.map((o) => `<tr><td class="dim">${App.dateTime(o.timestamp)}</td><td><b>${esc(o.ticker)}</b></td><td>${badge(o.action || o.decision)}</td>
        <td class="num">${o.shares != null ? num(o.shares, 4).replace(/\.?0+$/, '') : '–'}</td><td class="num">${o.price ? money(o.price) : '–'}</td>
        <td>${o.is_ai ? `🤖 ${esc(o.provider || 'AI')}` : t('common.manual')}</td><td><span class="chip">${esc(o.status || 'filled')}</span></td>
        <td class="wrap" title="${esc(o.reasoning || '')}">${esc(String(o.reasoning || '').slice(0, 140))}</td></tr>`).join('')
      || `<tr class="empty"><td colspan="8">${t('port.noorders')}</td></tr>`) + '</tbody>';
  }

  async function closeFromTable(tk) {
    if (!confirm(t('pos.confirm', { n: t('pos.all'), tk }))) return;
    try {
      const res = await App.api('/api/close_position', { ticker: tk });
      App.toast(res.success === false ? res.error : t('pos.closed', { tk }), res.success === false);
      fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  function exportCSV() {
    const orders = (state.portfolio || {}).recent_orders || [];
    let csv = 'timestamp,ticker,action,shares,price,notional,source,status,reasoning\n';
    orders.forEach((o) => {
      csv += [o.timestamp, o.ticker, o.action || o.decision, o.shares, o.price, o.notional, o.is_ai ? 'AI' : 'Manual', o.status, o.reasoning]
        .map((v) => `"${String(v ?? '').replace(/"/g, '""')}"`).join(',') + '\n';
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `orders_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  }

  async function resetAccount() {
    const cap = Number(($('set-capital') || {}).value) || 100000;
    if (!confirm(t('port.reset.confirm', { cap: money(cap, 0) }))) return;
    try {
      const res = await App.api('/api/reset_portfolio', { initial_capital: cap });
      App.toast(res.success === false ? res.error : t('port.reset.done'), res.success === false);
      fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  // ------------------------------------------------------------------ preferences
  async function loadPrefs() {
    try {
      state.settings = (await App.api('/api/settings')).settings || {};
      $('set-risk').value = state.settings.risk_profile || 'standard';
      $('set-provider').value = state.settings.provider || 'auto';
    } catch (e) { /* defaults stay */ }
  }
  async function savePrefs() {
    try {
      await App.api('/api/update_settings', { risk_profile: $('set-risk').value, provider: $('set-provider').value });
      App.toast(t('common.saved'));
    } catch (e) { App.toast(e.message, true); }
  }

  // ------------------------------------------------------------------ wiring
  App.register('home', loadHome);
  App.register('trade', load);
  App.register('portfolio', () => { renderPortfolio(); fetchPortfolio(); });
  App.on('lang', () => {
    if (App.state.page === 'home') { renderHome(); renderHomeView(); }
    if (App.state.page === 'trade' && state.market) { renderQuote(); renderSignals(); renderPosition(); renderPresets(); updateTicket(); draw(); }
    if (App.state.page === 'portfolio') renderPortfolio();
    const p = state.portfolio;
    if (p) $('mode-text').textContent = p.execution_mode === 'alpaca_paper' ? t('mode.alpaca') : t('mode.local');
  });
  App.on('ticker', () => { state.market = null; });
  window.addEventListener('resize', () => { if (App.state.page === 'trade') draw(); });
  window.addEventListener('DOMContentLoaded', () => {
    ['chk-candles', 'chk-sma20', 'chk-sma50', 'chk-bb', 'chk-rsi', 'chk-macd'].forEach((id) => $(id).addEventListener('change', draw));
    bootstrap();
    loadPrefs();
  });

  function hideSetup() { App.store.set('setupDone', '1'); $('home-setup').hidden = true; }

  return { renderSetup, hideSetup, load, fetchPortfolio, setSide, setUnit, preset, updateTicket, submit, closeTicker, closeFromTable, runAI, exportCSV,
    resetAccount, savePrefs, COMPANIES, portfolio: () => state.portfolio, positions };
})();
