/* Insights page: budget split, pattern finder, what-if (equity distribution), risk, correlation. */
const AnalyticsApp = (() => {
  const state = { panel: 'split', method: 'inverse_vol', lookback: 120, plan: null, analogs: null, backtest: null, sweep: null,
    portfolio: null, risk: null, corr: null, scope: 'ticker', corrLookback: 120, days: 21, loadedFor: {},
    algo: null, exp: null, expSystem: 'Ours', rl: null };
  const { t, esc, money, pct, spct, num, signCls, badge } = App;
  const $ = (id) => document.getElementById(id);
  const METHODS = ['inverse_vol', 'min_variance', 'equal', 'signal_tilt'];
  const usd0 = (v) => money(v, 0);

  // ------------------------------------------------------------------ navigation inside Insights
  function showPanel(panel) {
    state.panel = panel;
    document.querySelectorAll('#in-tabs button').forEach((b) => b.classList.toggle('active', b.dataset.panel === panel));
    document.querySelectorAll('#page-insights .panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${panel}`));
    load();
  }

  function load() {
    const fresh = state.loadedFor[state.panel] === `${state.panel}:${App.ticker}`;
    if (state.panel === 'split') { if (fresh && state.plan) renderPlan(); else loadPlan(); }
    else if (state.panel === 'pattern') { if (fresh && state.analogs) { renderAnalogs(); renderBacktest(); renderSweep(); } else runPattern(); }
    else if (state.panel === 'whatif') { if (fresh && state.portfolio) renderPortfolio(); else loadPortfolio(); }
    else if (state.panel === 'risk') { if (fresh && state.risk) renderRisk(); else loadRisk(); }
    else if (state.panel === 'corr') { if (state.corr) renderCorr(); else loadCorr(); }
    else if (state.panel === 'algo') { if (fresh && state.algo) renderAlgoAll(); else loadAlgo(); }
  }
  const mark = (panel) => { state.loadedFor[panel] = `${panel}:${App.ticker}`; };

  // ------------------------------------------------------------------ budget split
  async function loadPlan() {
    $('split-sub').textContent = t('common.loading');
    renderMethods();
    try {
      state.plan = await App.api(`/api/analytics/allocation?method=${state.method}&lookback=${state.lookback}`);
      mark('split');
      renderPlan();
    } catch (e) { $('split-sub').textContent = e.message; App.toast(e.message, true); }
  }

  function renderMethods() {
    $('split-methods').innerHTML = METHODS.map((m) => `<button class="option-card ${state.method === m ? 'active' : ''}" onclick="AnalyticsApp.setMethod('${m}')">
      <span class="t">${t(`split.m.${m}`)}</span><span class="d">${t(`split.m.${m}.d`)}</span></button>`).join('');
  }

  function renderPlan() {
    const p = state.plan; if (!p) return;
    renderMethods();
    $('split-sub').textContent = t('split.sub', { b: usd0(p.budget), r: p.reserve_pct, c: p.max_position_pct, d: p.as_of });
    const rows = p.rows.filter((r) => r.weight > 0.0005 || Math.abs(r.current_usd) > 1);
    const slices = rows.map((r, i) => ({ label: `${r.ticker}: ${pct(r.weight)}`, value: r.weight, color: Charts.PALETTE[i % Charts.PALETTE.length] }));
    Charts.donut($('split-donut'), [...slices, { label: `${t('split.cash')}: ${pct(p.cash.weight)}`, value: p.cash.weight, color: 'var(--s-bh)' }]);
    $('split-center').innerHTML = `<b>${usd0(p.budget)}</b><span>${t('split.budget')}</span>`;
    const tradeCell = (r) => r.trade === 'Hold' ? `<span class="dim">${t('split.ontarget')}</span>` : `${badge(r.trade)} <span class="num">${usd0(Math.abs(r.delta_usd))}</span>`;
    $('split-table').innerHTML = `<thead><tr><th>${t('col.stock')}</th><th class="num">${t('split.share')}</th><th class="num">${t('split.amount')}</th>
      <th class="num">${t('split.approx')}</th><th class="num">${t('split.now')}</th><th>${t('split.todo')}</th></tr></thead><tbody>` +
      rows.map((r, i) => `<tr><td><i class="swatch" style="background:${slices[i].color}"></i> <b>${esc(r.ticker)}</b>${r.signal ? ` ${badge(r.signal)}` : ''}</td>
        <td class="num">${pct(r.weight)}</td><td class="num">${usd0(r.target_usd)}</td><td class="num">${num(r.shares, 2)}</td>
        <td class="num">${usd0(r.current_usd)}</td><td>${tradeCell(r)}</td></tr>`).join('') +
      `<tr><td><i class="swatch" style="background:var(--s-bh)"></i> <b>${t('split.cash')}</b></td><td class="num">${pct(p.cash.weight)}</td>
        <td class="num">${usd0(p.cash.usd)}</td><td></td><td></td><td class="dim">${t(p.cash.note === 'reserve' ? 'split.cash.reserve' : 'split.cash.caps')}</td></tr></tbody>`;
    const st = p.stats;
    $('split-stats').innerHTML = [
      [t('split.vol'), pct(st.ann_vol), t('split.vol.sub', { v: usd0(p.budget * st.ann_vol) })],
      [t('split.effn'), num(st.effective_n, 1), t('split.effn.sub')],
      [t('split.invested'), pct(st.invested, 0), t('split.invested.sub')],
      [t('split.lookback'), `${p.lookback} ${t('common.days')}`, t('split.lookback.sub')],
    ].map(([k, v, s]) => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span><span class="s">${esc(s)}</span></div>`).join('') +
      (p.outside_plan.length ? `<div class="stat" style="grid-column:1/-1"><span class="k">${t('split.outside')}</span><span class="s">${p.outside_plan.map((o) => `${esc(o.ticker)} ${usd0(o.current_usd)}`).join(', ')}</span></div>` : '');
    $('split-explain').innerHTML = METHODS.map((m) => `<p style="margin-bottom:.5rem"><b>${t(`split.m.${m}`)}:</b> ${t(`split.m.${m}.long`)}</p>`).join('');
    const bt = p.backtest;
    if (bt && bt.dates && bt.dates.length) {
      const names = Object.keys(bt.equity);
      const colors = ['var(--c1)', 'var(--c3)', 'var(--c2)', 'var(--s-bh)'];
      Charts.line($('split-bt-chart'), bt.dates, { lines: names.map((n, i) => ({ name: label(n), values: bt.equity[n], color: colors[i % colors.length], width: i === 3 ? 1.4 : 2.1, dash: i === 3 ? '4 3' : undefined })),
        fmt: (v) => `${((v - 1) * 100).toFixed(0)}%`, baseline: 1, height: 260, xfmt: (d) => String(d).slice(0, 7) });
      $('split-bt-sub').textContent = t('split.bt.sub', { d: bt.start, r: bt.rebalance_days });
      $('split-bt-table').innerHTML = `<thead><tr><th>${t('split.method')}</th><th class="num">${t('m.total')}</th><th class="num">${t('m.annvol')}</th>
        <th class="num">${t('m.sharpe')}</th><th class="num">${t('m.maxdd')}</th></tr></thead><tbody>` +
        names.map((n, i) => { const m = bt.metrics[n]; return `<tr><td><i class="swatch" style="background:${colors[i % colors.length]}"></i> ${esc(label(n))}</td>
          <td class="num ${signCls(m.total_return)}">${spct(m.total_return)}</td><td class="num">${pct(m.ann_vol)}</td><td class="num">${num(m.sharpe)}</td>
          <td class="num neg">${spct(m.max_drawdown)}</td></tr>`; }).join('') + '</tbody>';
    }
  }
  // The backend returns English method names; show translated ones.
  const label = (n) => ({ 'Risk parity (inverse volatility)': t('split.m.inverse_vol'), 'Minimum variance': t('split.m.min_variance'),
    'Equal weight': t('split.m.equal'), 'SPY (same exposure)': t('split.spy') }[n] || n);

  function setMethod(m) { state.method = m; loadPlan(); }

  async function rebalance() {
    const p = state.plan; if (!p) return;
    const n = p.rows.filter((r) => Math.abs(r.delta_usd) >= 25).length;
    if (!n) { App.toast(t('split.nothing')); return; }
    if (!confirm(t('split.rebalance.confirm', { n }))) return;
    try {
      const r = await App.api('/api/autopilot/rebalance', { method: state.method });
      const ok = r.queued.filter((q) => q.queued).length;
      const skipped = r.queued.filter((q) => !q.queued);
      App.toast(t('split.queued', { ok, n: r.queued.length }) + (skipped.length ? ` · ${skipped.map((s) => `${s.ticker}: ${s.reason}`).join('; ')}` : ''), ok === 0);
      if (window.Autopilot) Autopilot.refresh();
      if (ok) App.go('autopilot');
    } catch (e) { App.toast(e.message, true); }
  }

  async function useForAutopilot() {
    try {
      await App.api('/api/autopilot/config', { config: { sizing: 'allocation', allocation_method: state.method } });
      App.toast(t('split.used', { m: t(`split.m.${state.method}`) }));
      if (window.Autopilot) Autopilot.refresh();
    } catch (e) { App.toast(e.message, true); }
  }

  // ------------------------------------------------------------------ pattern finder
  function params() {
    return new URLSearchParams({
      ticker: App.ticker, window: $('an-window').value, horizon: $('an-horizon').value, k: $('an-k').value, scope: state.scope,
      threshold: String(Number($('an-threshold').value) / 100), long_only: $('an-longonly').checked ? '1' : '0',
    });
  }

  async function runPattern() {
    const btn = $('an-run');
    btn.disabled = true;
    $('an-status').textContent = t('pat.matching');
    try {
      const qs = params();
      [state.analogs, state.backtest] = await Promise.all([App.api(`/api/analytics/analogs?${qs}`), App.api(`/api/analytics/backtest?${qs}`)]);
      state.sweep = null;
      mark('pattern');
      renderAnalogs(); renderBacktest(); renderSweep();
      $('an-status').textContent = t('pat.asof', { d: state.analogs.as_of });
      runSweep();
    } catch (e) { App.toast(e.message, true); $('an-status').textContent = e.message; }
    finally { btn.disabled = false; }
  }

  function renderAnalogs() {
    const a = state.analogs; if (!a) return;
    const f = a.forecast, H = a.horizon, Wn = a.window;
    const tiles = [
      [f.action === 'Buy' ? 'green' : f.action === 'Sell' ? 'red' : '', t('pat.verdict'), badge(f.action), t('pat.verdict.sub', { th: pct(f.threshold, 1) })],
      ['purple', t('pat.expected', { h: H }), `<span class="${signCls(f.expected_return)}">${spct(f.expected_return, 2)}</span>`, t('pat.expected.sub', { m: spct(f.median_return, 2) })],
      ['', t('pat.pup'), pct(f.p_up, 0), t('pat.pup.sub', { n: a.matches.filter((m) => m.forward_return > 0).length, k: a.k })],
      ['', t('pat.quality'), num(f.mean_similarity, 2), t('pat.quality.sub')],
      ['', t('pat.range'), `${spct(f.worst, 1)} … ${spct(f.best, 1)}`, t('pat.range.sub', { h: H })],
    ];
    $('an-kpis').innerHTML = tiles.map(([c, k, v, s]) => `<div class="kpi-card ${c}"><span class="kpi-title">${esc(k)}</span><span class="kpi-val">${v}</span><span class="kpi-sub">${esc(s)}</span></div>`).join('');
    const labels = Array.from({ length: Wn + H }, (_, i) => i - (Wn - 1));
    const lead = Array(Wn - 1).fill(null).concat([0]);
    const lines = a.matches.map((m) => ({ faint: true, width: 1.2, color: m.forward_return >= 0 ? 'var(--pos)' : 'var(--neg)',
      title: `${m.ticker} ${m.start} → ${m.end} · ${t('pat.sim')} ${num(m.similarity)} · ${spct(m.forward_return)}`,
      values: [...m.window_path, ...m.forward_path] }));
    lines.push({ name: t('pat.mean'), color: 'var(--yellow)', width: 2.4, dash: '5 3', values: [...lead, ...f.mean_path] });
    lines.push({ name: `${a.ticker} ${t('pat.now')}`, color: 'var(--text)', width: 2.8, values: [...a.current_path, ...Array(H).fill(null)] });
    Charts.line($('an-overlay'), labels, { lines, bands: [{ name: t('pat.band'), color: 'color-mix(in srgb, var(--yellow) 18%, transparent)', lo: [...lead, ...f.p25_path], hi: [...lead, ...f.p75_path] }],
      fmt: (v) => spct(v, 0), height: 320, vline: Wn - 1, vlineLabel: t('pat.today'), baseline: 0,
      xfmt: (d) => d === 0 ? t('pat.today') : (d > 0 ? `+${d}${t('pat.d')}` : `${d}${t('pat.d')}`) });
    $('an-matches').innerHTML = `<thead><tr><th>#</th><th>${t('col.stock')}</th><th>${t('pat.when')}</th><th class="num">${t('pat.sim')}</th>
      <th class="num">${t('pat.after', { h: H })}</th><th class="num">${t('pat.best')}</th><th class="num">${t('pat.worst')}</th></tr></thead><tbody>` +
      a.matches.map((m, i) => `<tr><td>${i + 1}</td><td><b>${esc(m.ticker)}</b></td><td>${esc(m.start)} → ${esc(m.end)}</td><td class="num">${num(m.similarity)}</td>
        <td class="num ${signCls(m.forward_return)}"><b>${spct(m.forward_return)}</b></td><td class="num pos">${spct(m.max_up)}</td><td class="num neg">${spct(m.max_down)}</td></tr>`).join('') + '</tbody>';
    const ex = $('an-exec');
    ex.disabled = f.action === 'Hold';
    ex.textContent = f.action === 'Hold' ? t('pat.exec.hold') : t('pat.exec', { a: App.action(f.action), tk: a.ticker });
  }

  async function executeVerdict() {
    const a = state.analogs; if (!a || a.forecast.action === 'Hold') return;
    const f = a.forecast;
    if (!confirm(t('pat.exec.confirm', { a: App.action(f.action), tk: a.ticker, k: a.k, e: spct(f.expected_return, 2), h: a.horizon }))) return;
    try {
      const d = await App.api('/api/agent/execute', { ticker: a.ticker, action: f.action, provider: 'pattern-analog',
        model: `${a.window}d/${a.horizon}d/k=${a.k}`, reasoning: `Chart-analog verdict: ${a.k} similar past ${a.window}-day windows imply ${spct(f.expected_return, 2)} over ${a.horizon} days, P(up) ${pct(f.p_up, 0)}.` });
      const err = d.result && d.result.error;
      App.toast(err ? `${t('trade.blocked')}: ${err}` : t('trade.sent', { side: App.action(f.action), tk: a.ticker }), !!err);
      if (window.Trade) Trade.fetchPortfolio();
    } catch (e) { App.toast(e.message, true); }
  }

  const BT_METRICS = [['total_return', 'm.total', spct], ['ann_vol', 'm.annvol', pct], ['sharpe', 'm.sharpe', num], ['max_drawdown', 'm.maxdd', spct],
    ['hit_rate', 'm.hit', pct], ['rank_ic', 'm.ic', (v) => num(v, 3)], ['exposure', 'm.exposure', pct], ['trades', 'm.trades', (v) => v ?? '–'], ['win_rate', 'm.win', pct]];

  function renderBacktest() {
    const b = state.backtest; if (!b) return;
    const names = ['Analog strategy', 'Buy & Hold'];
    const nm = { 'Analog strategy': t('pat.strategy'), 'Buy & Hold': t('pat.bh') };
    const colors = { 'Analog strategy': 'var(--primary)', 'Buy & Hold': 'var(--s-bh)' };
    Charts.line($('an-bt-chart'), b.dates, { lines: names.map((n) => ({ name: nm[n], values: b.equity[n], color: colors[n], width: n === 'Buy & Hold' ? 1.6 : 2.4 })),
      fmt: (v) => `${((v - 1) * 100).toFixed(0)}%`, baseline: 1, height: 260, xfmt: (d) => String(d).slice(0, 7) });
    $('an-bt-table').innerHTML = `<thead><tr><th></th>${names.map((n) => `<th class="num">${nm[n]}</th>`).join('')}</tr></thead><tbody>` +
      BT_METRICS.map(([k, lab, fmt]) => `<tr><td>${t(lab)}</td>${names.map((n) => { const v = b.metrics[n][k];
        return `<td class="num ${k.includes('return') ? signCls(v) : ''}">${v === undefined ? '' : fmt(v)}</td>`; }).join('')}</tr>`).join('') + '</tbody>';
    const s = b.metrics['Analog strategy'], bh = b.metrics['Buy & Hold'];
    const beat = (s.sharpe ?? -9) > (bh.sharpe ?? -9);
    $('an-bt-note').innerHTML = `${esc(b.dates[0])} → ${esc(b.dates[b.dates.length - 1])} · ${t('pat.bt.rules')} <b class="${beat ? 'pos' : 'neg'}">${beat ? t('pat.beat') : t('pat.notbeat')}</b>` +
      (s.rank_ic != null && s.rank_ic <= 0 ? ` ${t('pat.noise')}` : '');
  }

  async function runSweep() {
    const btn = $('an-sweep-btn');
    btn.disabled = true;
    try { state.sweep = await App.api(`/api/analytics/sweep?${params()}`); renderSweep(); }
    catch (e) { App.toast(e.message, true); }
    finally { btn.disabled = false; }
  }

  function renderSweep() {
    const s = state.sweep, box = $('an-sweep');
    if (!s) { box.innerHTML = `<div class="muted">${t('pat.grid.running')}</div>`; return; }
    const cell = (w, h) => s.cells.find((c) => c.window === w && c.horizon === h);
    const curW = Number($('an-window').value), curH = Number($('an-horizon').value);
    const better = s.cells.filter((c) => c.metrics && c.metrics.sharpe > (s.buy_and_hold.sharpe ?? 0)).length;
    box.innerHTML = `<table class="tbl heat"><thead><tr><th>${t('pat.grid.head')}</th>${s.horizons.map((h) => `<th class="num">${h}${t('pat.d')}</th>`).join('')}</tr></thead><tbody>` +
      s.windows.map((w) => `<tr><td>${t('pat.grid.row', { w })}</td>${s.horizons.map((h) => { const c = cell(w, h), m = c && c.metrics;
        if (!m) return '<td class="num">–</td>';
        return `<td class="num${w === curW && h === curH ? ' an-cell-focus' : ''}" style="background:${Charts.heat(m.sharpe, 2)}" title="${spct(m.total_return)} · ${t('m.hit')} ${pct(m.hit_rate)} · IC ${num(m.rank_ic, 3)}">${num(m.sharpe)}<div class="an-cell-sub">${spct(m.total_return, 0)}</div></td>`; }).join('')}</tr>`).join('') +
      `</tbody></table><div class="dim" style="margin-top:.5rem">${t('pat.grid.note', { d: s.start, bh: num(s.buy_and_hold.sharpe), n: better, k: s.cells.length })}</div>`;
  }

  // ------------------------------------------------------------------ what-if
  async function loadPortfolio() {
    try { state.portfolio = await App.api(`/api/analytics/portfolio?ticker=${App.ticker}&days=${state.days}`); mark('whatif'); renderPortfolio(); }
    catch (e) { App.toast(e.message, true); }
  }

  function renderPortfolio() {
    const p = state.portfolio; if (!p) return;
    const alloc = p.allocation;
    const slices = alloc.map((a, i) => ({ label: `${a.ticker}: ${pct(a.weight)}`, value: a.weight, short: a.weight < 0,
      color: a.ticker === 'Cash' ? 'var(--s-bh)' : Charts.PALETTE[i % Charts.PALETTE.length] }));
    Charts.donut($('an-donut'), slices);
    $('an-donut-center').innerHTML = `<b>${usd0(p.equity)}</b><span>${p.hypothetical ? t('wi.whatif') : t('wi.equity')}</span>`;
    $('an-alloc-basis').innerHTML = p.hypothetical ? t('wi.hypo', { tk: esc(alloc[0].ticker) }) : t('wi.real', { n: p.lookback });
    $('an-alloc-table').innerHTML = `<thead><tr><th>${t('wi.holding')}</th><th class="num">${t('split.amount')}</th><th class="num">${t('split.share')}</th><th class="num">${t('wi.riskshare')}</th></tr></thead><tbody>` +
      alloc.map((a, i) => `<tr><td><i class="swatch" style="background:${slices[i].color}"></i> ${a.ticker === 'Cash' ? t('split.cash') : esc(a.ticker)}${a.weight < 0 ? ` (${t('pos.short')})` : ''}</td>
        <td class="num">${usd0(a.exposure)}</td><td class="num">${pct(a.weight)}</td>
        <td class="num"><span class="an-bar"><b style="width:${Math.max(0, Math.min(100, (a.risk_share || 0) * 100))}%"></b></span>${pct(a.risk_share, 0)}</td></tr>`).join('') + '</tbody>';
    const r = p.risk, mc = p.monte_carlo;
    const tile = (k, v, c = '') => `<div class="signal-item"><span class="signal-item-name">${k}</span><span class="signal-item-val ${c}">${v}</span></div>`;
    $('an-risk-tiles').innerHTML = [tile(t('wi.vol'), pct(r.ann_vol)), tile(t('wi.beta'), num(r.beta_spy)), tile(t('wi.exposure'), `${pct(r.net_exposure, 0)} / ${pct(r.gross_exposure, 0)}`),
      tile(t('wi.var'), usd0(r.var_95_1d)), tile(t('wi.cvar'), usd0(r.cvar_95_1d))].join('');
    Charts.line($('an-fan'), mc.fan.p50.map((_, i) => i), {
      lines: [{ name: t('wi.median'), values: mc.fan.p50, color: 'var(--primary)', width: 2.4 }],
      bands: [{ name: '5–95%', lo: mc.fan.p5, hi: mc.fan.p95, color: 'color-mix(in srgb, var(--primary) 12%, transparent)' },
        { name: '25–75%', lo: mc.fan.p25, hi: mc.fan.p75, color: 'color-mix(in srgb, var(--primary) 25%, transparent)' }],
      fmt: usd0, baseline: p.equity, height: 230, xfmt: (d) => d === 0 ? t('wi.now') : `+${d}${t('pat.d')}`,
    });
    Charts.histogram($('an-mc-hist'), mc.histogram.edges, mc.histogram.counts, {
      fmt: (v) => `$${(v / 1000).toFixed(0)}k`, height: 190, colorOf: (v) => v < p.equity ? 'var(--neg)' : 'var(--pos)',
      markers: [{ x: p.equity, label: t('wi.today'), color: 'var(--text-muted)' }],
    });
    $('an-mc-kpis').innerHTML = [tile(t('wi.expected'), usd0(mc.expected), signCls(mc.expected - p.equity)), tile(t('wi.ploss'), pct(mc.p_loss, 0)),
      tile(t('wi.ploss5'), pct(mc.p_loss_5pct, 0), mc.p_loss_5pct > 0.2 ? 'neg' : ''), tile(t('wi.range'), `${usd0(mc.p5)} – ${usd0(mc.p95)}`)].join('');
    $('an-mc-note').textContent = t('wi.note', { n: p.paths.toLocaleString(), d: p.days });
  }

  // ------------------------------------------------------------------ risk
  async function loadRisk() {
    try { state.risk = await App.api(`/api/analytics/risk?ticker=${App.ticker}`); mark('risk'); renderRisk(); }
    catch (e) { App.toast(e.message, true); }
  }

  function renderRisk() {
    const r = state.risk; if (!r) return;
    const s = r.stats;
    $('an-risk-title').textContent = t('risk.title', { tk: r.ticker });
    Charts.histogram($('an-ret-hist'), r.histogram.edges, r.histogram.counts, {
      overlay: r.histogram.normal, fmt: (v) => `${(v * 100).toFixed(0)}%`, height: 230, colorOf: (v) => v < 0 ? 'var(--neg)' : 'var(--pos)',
      markers: [{ x: -s.var_95, label: `${t('risk.var')} ${pct(s.var_95)}`, color: 'var(--orange)' }],
    });
    const tile = (k, v, help) => `<div class="signal-item" ${help ? `title="${esc(help)}"` : ''}><span class="signal-item-name">${k}</span><span class="signal-item-val">${v}</span></div>`;
    $('an-ret-stats').innerHTML = [
      tile(t('risk.vol'), pct(s.ann_vol), t('risk.vol.help')), tile(t('m.sharpe'), num(s.sharpe), t('risk.sharpe.help')),
      tile(t('m.maxdd'), spct(s.max_drawdown)), tile(t('risk.nowdd'), spct(s.current_drawdown)),
      tile(t('risk.beta'), `${num(s.beta_spy)} / ${num(s.corr_spy)}`, t('risk.beta.help')), tile(t('risk.updays'), pct(s.up_days, 0)),
      tile(t('risk.best'), `${spct(s.best_day)} · ${s.best_date}`), tile(t('risk.worst'), `${spct(s.worst_day)} · ${s.worst_date}`),
      tile(t('risk.varcvar'), `${pct(s.var_95)} / ${pct(s.cvar_95)}`, t('risk.var.help')), tile(t('risk.vol20'), pct(s.vol_20d)),
    ].join('');
    Charts.line($('an-dd'), r.dates, {
      lines: [{ name: t('risk.dd.line'), values: r.drawdown, color: 'var(--neg)', width: 1.8 }],
      bands: [{ lo: r.drawdown.map(() => 0), hi: r.drawdown, color: 'color-mix(in srgb, var(--neg) 14%, transparent)' }],
      fmt: (v) => `${(v * 100).toFixed(0)}%`, height: 300, baseline: 0, xfmt: (d) => String(d).slice(0, 7),
    });
  }

  // ------------------------------------------------------------------ correlation
  async function loadCorr() {
    try { state.corr = await App.api(`/api/analytics/correlation?lookback=${state.corrLookback}`); renderCorr(); }
    catch (e) { App.toast(e.message, true); }
  }

  function renderCorr() {
    const c = state.corr; if (!c) return;
    const tk = c.tickers, focus = App.ticker;
    $('an-corr').innerHTML = `<table class="tbl heat an-corr"><thead><tr><th></th>${tk.map((x) => `<th class="num">${x}</th>`).join('')}</tr></thead><tbody>` +
      c.matrix.map((row, i) => `<tr class="${tk[i] === focus ? 'ra-focus' : ''}"><td><b>${tk[i]}</b></td>${row.map((v, j) =>
        `<td class="num" style="background:${i === j ? 'transparent' : Charts.heat(v, 1)}" title="${tk[i]} / ${tk[j]}: ${num(v)}">${i === j ? '' : num(v)}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
    const pairs = [];
    tk.forEach((a, i) => tk.forEach((b, j) => { if (j > i && !['SPY', 'QQQ'].includes(a) && !['SPY', 'QQQ'].includes(b)) pairs.push([a, b, c.matrix[i][j]]); }));
    pairs.sort((x, y) => x[2] - y[2]);
    const fp = ([a, b, v]) => `${a}–${b} <b>${num(v)}</b>`;
    $('an-corr-note').innerHTML = t('corr.note', { n: c.lookback, d: c.as_of, low: pairs.slice(0, 3).map(fp).join(', '), high: pairs.slice(-3).reverse().map(fp).join(', ') });
  }

  // ------------------------------------------------------------------ algorithms + RL
  async function loadAlgo() {
    $('algo-title').textContent = t('algo.title', { tk: App.ticker });
    $('algo-table').innerHTML = `<tbody><tr class="empty"><td>${t('common.loading')}</td></tr></tbody>`;
    try {
      [state.algo, state.rl] = await Promise.all([App.api(`/api/analytics/algorithms?ticker=${App.ticker}`),
        App.api(`/api/analytics/rl?ticker=${App.ticker}&epochs=${$('qrl-epochs').value}`)]);
      mark('algo');
      renderAlgo(); renderRL();
      loadExperience(state.expSystem);
    } catch (e) { $('algo-table').innerHTML = `<tbody><tr class="empty"><td>${esc(e.message)}</td></tr></tbody>`; App.toast(e.message, true); }
  }
  function renderAlgoAll() { renderAlgo(); renderExperience(); renderRL(); }

  const spark = (vals) => {
    if (!vals || vals.length < 2) return '';
    const W = 90, H = 26, lo = Math.min(...vals, 1), hi = Math.max(...vals, 1);
    const x = (i) => (i / (vals.length - 1)) * W, y = (v) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
    const end = vals[vals.length - 1];
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="display:block"><line x1="0" x2="${W}" y1="${y(1)}" y2="${y(1)}" stroke="var(--grid)"/>
      <polyline fill="none" stroke="${end >= 1 ? 'var(--pos)' : 'var(--neg)'}" stroke-width="1.6" points="${vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}"/></svg>`;
  };

  function renderAlgo() {
    const b = state.algo; if (!b) return;
    $('algo-title').textContent = t('algo.title', { tk: b.ticker });
    $('algo-sub').textContent = t('algo.sub', { d: b.decision_date });
    const v = b.votes, total = v.Buy + v.Hold + v.Sell || 1;
    $('algo-votes').innerHTML = `<div style="min-width:220px"><div class="row between dim"><span>${t('algo.votes')}</span>
      <span>${badge('Buy')} ${v.Buy} · ${badge('Hold')} ${v.Hold} · ${badge('Sell')} ${v.Sell}</span></div>
      <div class="meter" style="display:flex;height:10px;margin-top:6px"><b style="position:static;width:${v.Buy / total * 100}%;background:var(--pos);border-radius:0"></b>
      <b style="position:static;width:${v.Hold / total * 100}%;background:var(--text-dim);border-radius:0"></b><b style="position:static;width:${v.Sell / total * 100}%;background:var(--neg);border-radius:0"></b></div></div>`;
    $('algo-table').innerHTML = `<thead><tr><th>${t('algo.name')}</th><th>${t('algo.family')}</th><th>${t('algo.today')}</th>
      <th class="num">${t('m.hit')}</th><th class="num">${t('algo.hit30')}</th>
      <th class="num">${t('m.total')}</th><th class="num">${t('m.sharpe')}</th><th>${t('algo.curve')}</th></tr></thead><tbody>` +
      b.rows.map((r) => { const rec = r.record || {};
        return `<tr data-sys="${esc(r.name)}" class="${r.name === state.expSystem ? 'ra-focus' : ''}" title="${esc(t(`algo.d.${r.id}`))}">
          <td style="white-space:normal;min-width:240px"><b>${esc(r.name)}</b><div class="dim">${esc(t(`algo.d.${r.id}`))}</div></td>
          <td><span class="chip ${r.family === 'memory' ? 'info' : ''}">${t(`fam.${r.family}`)}</span></td>
          <td>${badge(r.today.action)}${r.today.abstain ? ` <span class="dim">${t('ag.abstain')}</span>` : ''}</td>
          <td class="num">${pct(rec.hit_rate, 0)}</td><td class="num">${pct(rec.hit_rate_30, 0)}</td>
          <td class="num ${signCls(rec.total_return)}"><b>${spct(rec.total_return)}</b></td><td class="num">${num(rec.sharpe)}</td><td>${spark(r.spark)}</td></tr>`; }).join('') + '</tbody>';
    $('algo-table').querySelectorAll('tr[data-sys]').forEach((tr) => tr.addEventListener('click', () => {
      state.expSystem = tr.dataset.sys; renderAlgo(); loadExperience(state.expSystem);
    }));
    const src = b.source;
    $('algo-source').innerHTML = `<span>📂 ${t('algo.source', { f: esc(src.file), n: src.rows.toLocaleString(), a: src.algorithms, k: src.tickers, d1: src.first, d2: src.last,
      t1: b.test_range ? b.test_range[0] : '–', t2: b.test_range ? b.test_range[1] : '–', per: (b.rows[0] && b.rows[0].record ? b.rows[0].record.decisions : '–') })}</span>`;
  }

  async function loadExperience(system) {
    $('exp-title').textContent = t('exp.title', { a: system, tk: App.ticker });
    try { state.exp = await App.api(`/api/analytics/experience?ticker=${App.ticker}&system=${encodeURIComponent(system)}&limit=40`); renderExperience(); }
    catch (e) { App.toast(e.message, true); }
  }

  function renderExperience() {
    const e = state.exp; if (!e) return;
    $('exp-title').textContent = t('exp.title', { a: e.system, tk: e.ticker });
    Charts.line($('exp-chart'), e.curve.dates, { lines: [
      { name: e.system, values: e.curve.algo, color: 'var(--primary)', width: 2.3 },
      { name: t('pat.bh'), values: e.curve.buy_hold, color: 'var(--s-bh)', width: 1.5, dash: '4 3' }],
      fmt: (v) => `${((v - 1) * 100).toFixed(0)}%`, baseline: 1, height: 240, xfmt: (d) => String(d).slice(0, 7) });
    const r = e.record || {};
    const st = (k, v, s = '') => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span>${s ? `<span class="s">${s}</span>` : ''}</div>`;
    $('exp-stats').innerHTML = [st(t('m.total'), `<span class="${signCls(r.total_return)}">${spct(r.total_return)}</span>`),
      st(t('m.hit'), pct(r.hit_rate, 0), t('exp.hit.sub', { n: r.active_days })), st(t('m.sharpe'), num(r.sharpe)), st(t('m.maxdd'), spct(r.max_drawdown)),
      st(t('exp.mix'), `${pct(r.buy_share, 0)} / ${pct(r.sell_share, 0)}`, t('exp.mix.sub'))].join('');
    $('exp-log').innerHTML = `<thead><tr><th>${t('exp.date')}</th><th>${t('exp.call')}</th><th>${t('exp.held')}</th><th class="num">${t('ag.th.signal')}</th>
      <th class="num">${t('exp.move')}</th><th class="num">${t('exp.result')}</th><th>${t('exp.right')}</th><th>${t('exp.split')}</th></tr></thead><tbody>` +
      e.log.map((x) => `<tr><td>${esc(x.date)} → ${esc(x.next_date)}</td><td>${badge(x.action)}</td><td>${t(`held.${x.held.toLowerCase()}`)}</td>
        <td class="num">${num(x.signal, 3)}</td><td class="num ${signCls(x.next_return)}">${spct(x.next_return, 2)}</td>
        <td class="num ${signCls(x.net)}"><b>${spct(x.net, 2)}</b></td>
        <td>${x.right == null ? '<span class="dim">–</span>' : x.right ? '<span class="pos">✓</span>' : '<span class="neg">✗</span>'}</td>
        <td><span class="chip">${x.split === 'test' ? t('exp.test') : t('exp.warmup')}</span></td></tr>`).join('') + '</tbody>';
  }

  async function trainRL() {
    try {
      App.toast(t('rl.training'));
      state.rl = await App.api(`/api/analytics/rl?ticker=${App.ticker}&epochs=${$('qrl-epochs').value}`);
      renderRL();
    } catch (e) { App.toast(e.message, true); }
  }

  const stateText = (s) => `${t(`rl.s.ret5.${s.ret5}`)} · ${t(`rl.s.trend.${s.trend}`)} · ${t(`rl.s.rsi.${s.rsi}`)} · ${t(`rl.s.pos.${s.position.toLowerCase()}`)}`;
  const rlAct = (a) => t(`rl.a.${a.toLowerCase()}`);

  function renderRL() {
    const r = state.rl; if (!r) return;
    $('qrl-steps').innerHTML = ['see', 'act', 'reward', 'learn'].map((k, i) => `<div class="option-card" style="cursor:default">
      <span class="t">${['👀', '🎯', '💰', '🧠'][i]} ${i + 1}. ${t(`rl.step.${k}`)}</span><span class="d">${t(`rl.step.${k}.d`)}</span></div>`).join('');
    const a = r.metrics.agent, bh = r.metrics.buy_hold;
    const tiles = [
      [r.today.action === 'Long' ? 'green' : '', t('rl.k.today'), `<span class="chip ${r.today.action === 'Long' ? 'buy' : ''}">${rlAct(r.today.action)}</span>`,
        `Q: ${t('rl.a.flat')} ${num(r.today.q[0], 3)} · ${t('rl.a.long')} ${num(r.today.q[1], 3)}`],
      ['blue', t('rl.k.state'), `<span style="font-size:.95rem">${stateText(r.today.state)}</span>`, t('rl.k.state.sub', { n: r.today.visits })],
      [a.total_return >= bh.total_return ? 'green' : 'red', t('rl.k.return'), `<span class="${signCls(a.total_return)}">${spct(a.total_return)}</span>`,
        t('rl.k.return.sub', { b: spct(bh.total_return), s: num(a.sharpe), bs: num(bh.sharpe) })],
      ['purple', t('rl.k.exp'), r.experience.transitions.toLocaleString(), t('rl.k.exp.sub', { s: r.experience.states_seen, n: r.experience.states })],
    ];
    $('qrl-kpis').innerHTML = tiles.map(([c, k, v, s]) => `<div class="kpi-card ${c}"><span class="kpi-title">${k}</span><span class="kpi-val">${v}</span><span class="kpi-sub">${esc(s)}</span></div>`).join('');
    Charts.line($('qrl-curve'), r.learning_curve.map((_, i) => i + 1), { lines: [{ name: t('rl.avgreward'), values: r.learning_curve, color: 'var(--c4)', width: 2.2 }],
      fmt: (v) => `${Number(v).toFixed(3)}%`, height: 200, xfmt: (d) => `#${d}` });
    Charts.line($('qrl-equity'), r.equity.dates, { lines: [
      { name: t('rl.agent'), values: r.equity.agent, color: 'var(--c4)', width: 2.3 },
      { name: t('pat.bh'), values: r.equity.buy_hold, color: 'var(--s-bh)', width: 1.5, dash: '4 3' }],
      fmt: (v) => `${((v - 1) * 100).toFixed(0)}%`, baseline: 1, height: 200, xfmt: (d) => String(d).slice(0, 7) });
    $('qrl-test-note').textContent = t('rl.test.note', { d: r.test_start, m: pct(a.time_in_market, 0), w: a.switches, c: r.cost_bps });
    const cur = JSON.stringify(r.today.state);
    $('qrl-qtable').innerHTML = `<thead><tr><th>${t('rl.situation')}</th><th class="num">Q ${t('rl.a.flat')}</th><th class="num">Q ${t('rl.a.long')}</th><th>${t('rl.choice')}</th><th class="num">${t('rl.seen')}</th></tr></thead><tbody>` +
      r.q_table.filter((x) => x.visits > 0).sort((x, y) => y.visits - x.visits).map((x) => {
        const isNow = JSON.stringify({ ret5: x.ret5, trend: x.trend, rsi: x.rsi, position: x.position }) === cur;
        return `<tr class="${isNow ? 'ra-focus' : ''}"><td style="white-space:normal">${isNow ? '★ ' : ''}${stateText(x)}</td>
          <td class="num" style="${x.best === 'Flat' ? 'background:var(--primary-soft);font-weight:700' : ''}">${num(x.q_flat, 3)}</td>
          <td class="num" style="${x.best === 'Long' ? 'background:var(--pos-soft);font-weight:700' : ''}">${num(x.q_long, 3)}</td>
          <td><span class="chip ${x.best === 'Long' ? 'buy' : ''}">${rlAct(x.best)}</span>${x.changed ? ` <span class="dim" title="${esc(t('rl.changed'))}">↻</span>` : ''}</td>
          <td class="num">${x.visits.toLocaleString()}</td></tr>`; }).join('') + '</tbody>';
    $('qrl-exp-note').textContent = t('rl.exp.note', { p: r.experience.pretrain_days.toLocaleString(), e: r.epochs });
    $('qrl-replay').innerHTML = `<thead><tr><th>${t('exp.date')}</th><th>${t('rl.situation')}</th><th>${t('rl.action')}</th><th class="num">${t('exp.move')}</th><th class="num">${t('rl.reward')}</th></tr></thead><tbody>` +
      r.experience.recent.map((x) => `<tr><td>${esc(x.date)}</td><td style="white-space:normal">${stateText(x.state)}</td>
        <td><span class="chip ${x.action === 'Long' ? 'buy' : ''}">${rlAct(x.action)}</span></td>
        <td class="num ${signCls(x.next_return)}">${spct(x.next_return, 2)}</td><td class="num ${signCls(x.reward)}"><b>${spct(x.reward, 2)}</b></td></tr>`).join('') + '</tbody>';
  }

  // ------------------------------------------------------------------ wiring
  function bindSeg(id, onChange) {
    $(id).querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
      onChange(b.dataset.v);
    }));
  }

  const RENDER = { split: renderPlan, pattern: () => { renderAnalogs(); renderBacktest(); renderSweep(); }, whatif: renderPortfolio, risk: renderRisk, corr: renderCorr,
    algo: renderAlgoAll };
  App.register('insights', load);
  App.on('lang', () => { if (App.state.page === 'insights') RENDER[state.panel](); });
  window.addEventListener('resize', () => { if (App.state.page === 'insights' && state.panel !== 'corr') RENDER[state.panel](); });
  window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('#in-tabs button').forEach((b) => b.addEventListener('click', () => showPanel(b.dataset.panel)));
    const want = new URLSearchParams(location.search).get('panel');
    if (want && document.getElementById(`panel-${want}`)) showPanel(want);
    bindSeg('an-scope', (v) => { state.scope = v; });
    bindSeg('an-corr-lookback', (v) => { state.corrLookback = Number(v); loadCorr(); });
    bindSeg('an-days', (v) => { state.days = Number(v); loadPortfolio(); });
    bindSeg('split-lookback', (v) => { state.lookback = Number(v); loadPlan(); });
  });

  return { runPattern, runSweep, executeVerdict, setMethod, rebalance, useForAutopilot, showPanel, trainRL };
})();
