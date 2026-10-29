/* Research-grounded agent pages: AI memory (Memory Agent), Stock scanner and Research lab. */
const ResearchApp = (() => {
  const SERIES_COLORS = {
    'Ours': 'var(--s-ours)', 'FinMem': 'var(--s-finmem)', 'Plain RAG': 'var(--s-plain)',
    'Buy & Hold': 'var(--s-bh)', 'LGBM (price)': 'var(--s-lgbm)', 'LGBM (price+Ours)': 'var(--s-lgbm-ours)',
  };
  const DEFAULT_ON = ['Buy & Hold', 'LGBM (price)', 'Plain RAG', 'FinMem', 'Ours'];
  const STRATEGY_ORDER = ['Buy & Hold', 'Momentum (1d)', 'TS-Momentum (20d)', 'Mean-Reversion (5d)', 'MA-Crossover',
    'Daily News Sentiment', 'LogReg (price)', 'LGBM (price)', 'LGBM (price+FinMem)', 'LGBM (price+Ours)',
    'Plain RAG', 'FinMem', 'Ours', 'Ours (signed N1)'];
  const state = { status: null, brief: null, llm: null, research: {}, protocol: 'rolling', scope: 'all',
    hidden: new Set(), jobTimer: null };

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : `${(v * 100).toFixed(d)}%`;
  const spct = (v, d = 1) => (v === null || v === undefined) ? '–' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
  const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);
  const snum = (v, d = 2) => (v === null || v === undefined) ? '–' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}`;
  const signCls = (v) => v > 0 ? 'ra-pos' : v < 0 ? 'ra-neg' : '';
  const badge = (a) => `<span class="ra-badge ${String(a || 'hold').toLowerCase()}">${esc(App.action(a))}</span>`;
  const color = (name) => SERIES_COLORS[name] || 'var(--s-other)';
  const toast = (m, err) => App.toast(m, err);
  const t = App.t;

  async function api(path, body) {
    const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.statusText);
    return data;
  }

  async function ensureStatus(force = false) {
    if (state.status && !force) return state.status;
    state.status = await api('/api/agent/status');
    const sel = $('ag-backbone');
    if (sel && !sel.options.length) sel.innerHTML = state.status.backbones.map((b) => `<option>${b}</option>`).join('');
    renderStatus();
    return state.status;
  }

  function renderStatus() {
    const s = state.status;
    if (!s) return;
    $('ag-status').textContent = t('ag.status', { d: s.latest_decision_date, p: s.last_price_date });
    const job = s.job || {};
    const box = $('ag-job');
    if (job.running || (job.log && job.log.length)) {
      box.hidden = false;
      box.innerHTML = (job.log || []).map(esc).join('<br>') + (job.error ? `<br><span class="ra-neg">${esc(job.error)}</span>` : '');
      box.scrollTop = box.scrollHeight;
    }
    $('ag-refresh').disabled = !!job.running;
    $('ag-refresh').textContent = job.running ? `⟳ ${t('ag.refreshing')}` : `⟳ ${t('ag.refresh')}`;
  }

  // ------------------------------------------------------------- SVG helpers
  function svgEl(width, height) {
    return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img">`;
  }
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const step0 = span / n;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 1);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  /** Multi-series line chart with crosshair + tooltip (one y-axis). */
  function lineChart(container, dates, series, opts = {}) {
    const W = container.clientWidth || 900, H = opts.height || 340;
    const m = { l: 52, r: 18, t: 12, b: 28 };
    const names = Object.keys(series);
    let lo = Infinity, hi = -Infinity;
    names.forEach((n) => series[n].forEach((v) => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (opts.baseline !== undefined) { lo = Math.min(lo, opts.baseline); hi = Math.max(hi, opts.baseline); }
    const pad = (hi - lo) * 0.06 || 0.01; lo -= pad; hi += pad;
    const x = (i) => m.l + (i / Math.max(dates.length - 1, 1)) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    let out = svgEl(W, H);
    niceTicks(lo, hi).forEach((t) => {
      out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)"/>`;
      out += `<text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${opts.fmt ? opts.fmt(t) : t}</text>`;
    });
    if (opts.baseline !== undefined) out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(opts.baseline)}" y2="${y(opts.baseline)}" stroke="var(--text-dim)" stroke-dasharray="3 3"/>`;
    const step = Math.max(1, Math.floor(dates.length / 7));
    const monthly = dates.length > 150;
    let prev = '';
    for (let i = 0; i < dates.length; i += step) {
      const lab = monthly ? String(dates[i]).slice(0, 7) : String(dates[i]).slice(5, 10);
      if (lab !== prev) out += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${lab}</text>`;
      prev = lab;
    }
    names.forEach((n) => {
      const pts = series[n].map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).join(' ');
      const w = n === 'Ours' ? 2.6 : 1.6;
      out += `<polyline fill="none" stroke="${color(n)}" stroke-width="${w}" stroke-linejoin="round" points="${pts}"/>`;
    });
    out += `<line class="ra-cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-dim)" visibility="hidden"/>`;
    out += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" class="ra-hit"/></svg>`;
    container.innerHTML = out + '<div class="ra-tip" hidden></div>';
    const svg = container.querySelector('svg'), tip = container.querySelector('.ra-tip'), cross = container.querySelector('.ra-cross');
    svg.querySelector('.ra-hit').addEventListener('mousemove', (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const i = Math.max(0, Math.min(dates.length - 1, Math.round((px - m.l) / (W - m.l - m.r) * (dates.length - 1))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      const rows = names.map((n) => [n, series[n][i]]).filter((r) => r[1] != null).sort((a, b) => b[1] - a[1]);
      tip.innerHTML = `<div style="margin-bottom:4px;font-weight:700">${esc(dates[i])}</div>` + rows.map(([n, v]) =>
        `<div class="ra-tip-row"><span><i class="ra-dot" style="background:${color(n)}"></i>${esc(n)}</span><b>${opts.fmt ? opts.fmt(v) : v}</b></div>`).join('');
      tip.hidden = false;
      const left = (x(i) / W) * r.width;
      tip.style.left = `${left > r.width * 0.6 ? left - tip.offsetWidth - 12 : left + 12}px`;
      tip.style.top = '8px';
    });
    svg.querySelector('.ra-hit').addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  }

  /** Horizontal bars, diverging around zero (or around `center`). */
  function hbars(container, rows, opts = {}) {
    const W = container.clientWidth || 520, rowH = 26, m = { l: opts.labelWidth || 170, r: 70, t: 6, b: 6 };
    const H = m.t + m.b + rows.length * rowH;
    const center = opts.center ?? 0;
    const vals = rows.map((r) => r.value);
    const lo = Math.min(center, ...vals), hi = Math.max(center, ...vals);
    const span = (hi - lo) || 1;
    const x = (v) => m.l + ((v - lo) / span) * (W - m.l - m.r);
    let out = svgEl(W, H);
    out += `<line x1="${x(center)}" x2="${x(center)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-dim)"/>`;
    rows.forEach((r, k) => {
      const yy = m.t + k * rowH + 5, h = rowH - 10;
      const x0 = Math.min(x(center), x(r.value)), w = Math.max(2, Math.abs(x(r.value) - x(center)));
      out += `<text x="${m.l - 8}" y="${yy + h - 2}" text-anchor="end">${esc(r.label)}</text>`;
      out += `<rect x="${x0}" y="${yy}" width="${w}" height="${h}" rx="3" fill="${r.color || 'var(--s-ours)'}"><title>${esc(r.label)}: ${esc(r.text || r.value)}</title></rect>`;
      out += `<text x="${Math.max(x(r.value), x(center)) + 6}" y="${yy + h - 2}">${esc(r.text ?? num(r.value, 3))}</text>`;
    });
    container.innerHTML = out + '</svg>';
  }

  // ------------------------------------------------------------ Memory Agent
  async function loadAgent() {
    try {
      await ensureStatus();
      state.brief = await api(`/api/agent/brief?ticker=${App.ticker}`);
      state.llm = null;
      renderBrief();
    } catch (err) { toast(`Agent: ${err.message}`, true); }
  }

  function renderBrief() {
    const b = state.brief, o = b.systems.Ours || {}, c = b.consensus;
    const ep = b.episodic || {}, peer = b.peer || {};
    const tiles = [
      [`${b.ticker} · ${b.date}`, `$${num(b.price)}`, `<span class="${signCls(b.change_1d)}">${spct(b.change_1d, 2)}</span> ${t('home.1d')} · ${spct(b.change_20d)} 20d`, 'blue'],
      [t('ag.k.signal'), snum(o.signal), t('ag.k.signal.sub', { th: num(o.threshold), a: o.abstain ? t('ag.abstain') : App.action(o.action) }), 'purple'],
      [t('ag.k.consensus'), `${badge(c.action)}`, `${t('home.conf')} ${pct(c.confidence, 0)}${c.agreement ? ` · ${t('ag.agree')}` : ''}`, 'green'],
      [t('home.pup'), b.price_model.p_up == null ? '–' : num(b.price_model.p_up), t('ag.k.pup.sub'), 'orange'],
      [t('ag.k.episodic'), ep.edge == null ? '–' : spct(ep.edge, 2), ep.n ? t('ag.k.episodic.sub', { n: ep.n }) : t('ag.k.episodic.none'), ''],
      [t('ag.k.peer'), peer.n ? snum(peer.signal) : '–', peer.n ? t('ag.k.peer.sub', { n: peer.n }) : t('ag.k.peer.none'), ''],
    ];
    $('ag-kpis').innerHTML = tiles.map(([t, v, s, cls]) =>
      `<div class="kpi-card ${cls}"><div class="kpi-title">${esc(t)}</div><div class="kpi-val">${v}</div><div class="kpi-sub">${s}</div></div>`).join('');
    $('ag-consensus-badge').innerHTML = badge(c.action);
    $('ag-reasons').innerHTML = c.reasons.map((r) => `<li>${esc(r)}</li>`).join('');
    renderMemories();
    renderSystems();
    renderHistory();
    renderLLM();
  }

  function meter(value, colorVar, centered) {
    if (value == null) return '<i></i>';
    const v = Math.max(-1, Math.min(1, value));
    const style = centered
      ? (v >= 0 ? `left:50%;width:${v * 50}%` : `left:${50 + v * 50}%;width:${-v * 50}%`)
      : `left:0;width:${Math.max(0, Math.min(1, value)) * 100}%`;
    return `<i><b style="${style};background:${colorVar}"></b></i>`;
  }

  function renderMemories() {
    const b = state.brief;
    const cited = new Set((state.llm && state.llm.decision && state.llm.decision.cited_ids) || []);
    const groups = [['news', t('ag.layer.news')], ['10-Q', t('ag.layer.10q')], ['10-K', t('ag.layer.10k')]];
    $('ag-memories').innerHTML = groups.map(([layer, title]) => {
      const items = b.memories.filter((m) => m.layer === layer);
      if (!items.length) return '';
      return `<div class="ra-layer-title"><span>${title}</span><span>${items.length}</span></div>` + items.map((m) => {
        const s = m.sentiment;
        const text = m.url ? `<a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.text)}</a>` : esc(m.text);
        return `<div class="ra-mem ${cited.has(String(m.id)) ? 'cited' : ''}">
          <div class="ra-mem-text">${text}</div>
          <div class="ra-mem-meta">
            <span>${esc(m.date)} · ${t('ag.ago', { n: m.age_days })}</span>
            <span class="ra-meter">${t('ag.tone')} ${meter(s, s >= 0 ? 'var(--pos)' : 'var(--neg)', true)} <span class="${signCls(s)}">${snum(s)}</span></span>
            <span class="ra-meter">${t('ag.cred')} ${meter(m.credibility, 'var(--s-ours)')} ${num(m.credibility)}</span>
            ${m.corroboration > 1 ? `<span>×${Math.round(m.corroboration)} ${t('ag.reports')}</span>` : ''}
            <span>${esc(m.category)}${m.author && m.author !== 'sec' ? ' · ' + esc(m.author) : ''}</span>
            <span>id ${esc(m.id)}</span>
          </div></div>`;
      }).join('');
    }).join('') || `<div class="ra-muted">${t('ag.nomem')}</div>`;
  }

  function renderSystems() {
    const sys = state.brief.systems;
    const rows = ['Plain RAG', 'FinMem', 'Ours'].filter((n) => sys[n]).map((n) => {
      const s = sys[n];
      return `<tr class="${n === 'Ours' ? 'ra-focus' : ''}"><td><i class="ra-dot" style="background:${color(n)}"></i> ${n}</td>
        <td class="num ${signCls(s.signal)}">${snum(s.signal)}</td><td class="num">±${num(s.threshold)}</td>
        <td>${badge(s.abstain ? 'Hold' : s.action)}${s.abstain ? ` <span class="ra-muted">${t('ag.abstain')}</span>` : ''}</td>
        <td class="num">${s.layers.join(' / ')}</td><td class="num">${num(s.mean_age, 0)}d</td>
        <td class="num">${num(s.redundancy)}</td><td class="num">${num(s.conflict)}</td></tr>`;
    }).join('');
    $('ag-systems').innerHTML = `<thead><tr><th>${t('ag.th.system')}</th><th class="num">${t('ag.th.signal')}</th><th class="num">${t('ag.th.thr')}</th><th>${t('col.action')}</th>
      <th class="num">news/10-Q/10-K</th><th class="num">${t('ag.th.age')}</th><th class="num">${t('ag.th.redund')}</th><th class="num">${t('ag.th.conflict')}</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function renderHistory() {
    const h = state.brief.history, box = $('ag-history');
    box.innerHTML = '<div id="ag-h-price"></div><div id="ag-h-signal" style="margin-top:6px"></div>';
    SERIES_COLORS[state.brief.ticker] = 'var(--s-bh)';
    lineChart($('ag-h-price'), h.price_date, { [state.brief.ticker]: h.close }, { height: 170, fmt: (v) => `$${Number(v).toFixed(0)}` });
    const sigBox = $('ag-h-signal');
    const W = sigBox.clientWidth || 600, H = 110, m = { l: 52, r: 18, t: 6, b: 18 };
    const vals = h.signal, n = vals.length;
    const lim = Math.max(0.3, ...vals.map((v) => Math.abs(v || 0)));
    const x = (i) => m.l + (i + 0.5) / n * (W - m.l - m.r), bw = Math.max(1.5, (W - m.l - m.r) / n - 2);
    const y = (v) => m.t + (1 - (v + lim) / (2 * lim)) * (H - m.t - m.b);
    const th = state.brief.systems.Ours ? state.brief.systems.Ours.threshold : 0;
    let out = svgEl(W, H);
    [-th, th].forEach((t) => { out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--text-dim)" stroke-dasharray="3 3"/>`; });
    out += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--grid)"/>`;
    out += `<text x="${m.l - 8}" y="${y(lim) + 10}" text-anchor="end">+${lim.toFixed(1)}</text><text x="${m.l - 8}" y="${y(-lim)}" text-anchor="end">−${lim.toFixed(1)}</text>`;
    vals.forEach((v, i) => {
      const y0 = Math.min(y(0), y(v)), hh = Math.max(1, Math.abs(y(v) - y(0)));
      out += `<rect x="${x(i) - bw / 2}" y="${y0}" width="${bw}" height="${hh}" rx="1.5" fill="${v >= 0 ? 'var(--pos)' : 'var(--neg)'}" opacity="0.85"><title>${h.date[i]}: ${snum(v)}</title></rect>`;
    });
    out += `<text x="${m.l}" y="${H - 2}">${t('ag.sigchart')}</text>`;
    sigBox.innerHTML = out + '</svg>';
  }

  function renderLLM() {
    const box = $('ag-llm');
    if (!state.llm) { box.hidden = true; return; }
    const d = state.llm.decision || {};
    box.hidden = false;
    const ok = String(d.status || '').startsWith('ok');
    box.innerHTML = `
      <div class="ra-row"><b>${t('ag.llm')}</b> ${badge(d.action)} <span class="ra-muted">${esc(state.llm.model)} · ${esc(state.llm.provider)} · ${num(state.llm.latency_s, 1)}s · ${t('home.conf')} ${num(d.confidence)}</span></div>
      ${ok ? `<div>${esc(d.reasoning)}</div>` : `<div class="ra-neg">${t('ag.llm.down', { s: esc(d.status) })}</div>`}
      ${(d.cited_ids || []).length ? `<div class="ra-muted">${t('ag.cited')}: ${d.cited_ids.map(esc).join(', ')}</div>` : ''}
      <div class="ra-row">
        <button class="btn btn-success" onclick="ResearchApp.execute('${esc(d.action)}', 'llm')">${t('ag.approve.llm')}</button>
        <button class="btn btn-secondary" onclick="ResearchApp.execute('${esc(state.brief.consensus.action)}', 'consensus')">${t('ag.use.cons')}</button>
      </div>
      <details><summary>${t('ag.prompt')}</summary><pre>${esc(state.llm.prompt)}</pre></details>`;
  }

  async function askLLM() {
    const btn = $('ag-ask');
    btn.disabled = true; btn.textContent = `🧠 ${t('ag.thinking')}`;
    try {
      state.llm = await api('/api/agent/llm', { ticker: App.ticker, backbone: $('ag-backbone').value });
      state.brief = state.llm.brief;
      renderBrief();
    } catch (err) { toast(`LLM: ${err.message}`, true); }
    finally { btn.disabled = false; btn.textContent = `🧠 ${t('ag.ask')}`; }
  }

  async function execute(action, source) {
    const b = state.brief;
    if (action === 'Hold') { toast(t('ag.hold')); return; }
    if (!confirm(t('ag.exec.confirm', { a: App.action(action), tk: b.ticker }))) return;
    try {
      const reasoning = source === 'llm' && state.llm ? state.llm.decision.reasoning : b.consensus.reasons.join('; ');
      const res = await api('/api/agent/execute', { ticker: b.ticker, action, reasoning, price: b.price,
        provider: source === 'llm' ? state.llm.provider : 'agent', model: source === 'llm' ? state.llm.model : 'memory+price consensus' });
      toast(res.result && res.result.error ? `${t('trade.blocked')}: ${res.result.error}` : t('trade.sent', { side: App.action(action), tk: b.ticker }), !!(res.result && res.result.error));
      Trade.fetchPortfolio();
    } catch (err) { toast(`Order: ${err.message}`, true); }
  }

  async function runAgent() {
    const tk = App.ticker;
    if (!confirm(t('trade.ai.confirm', { tk }))) return;
    const btn = $('ag-run'); btn.disabled = true;
    try {
      const res = await api('/api/agent/run', { ticker: tk, backbone: $('ag-backbone').value });
      toast(`${tk}: ${App.action(res.action)} · ${res.model}${res.order_error ? ' (' + res.order_error + ')' : ''}`, !!res.order_error);
      if (res.brief) { state.brief = res.brief; state.llm = null; renderBrief(); }
      Trade.fetchPortfolio();
    } catch (err) { toast(`Agent: ${err.message}`, true); }
    finally { btn.disabled = false; }
  }

  async function refreshData() {
    try {
      const r = await api('/api/agent/refresh', {});
      if (!r.started) { toast(r.reason, true); return; }
      toast(t('ag.refresh.started'));
      clearInterval(state.jobTimer);
      state.jobTimer = setInterval(async () => {
        await ensureStatus(true);
        if (!state.status.job.running) { clearInterval(state.jobTimer); toast(state.status.job.error ? t('ag.refresh.fail') : t('ag.refresh.done'), !!state.status.job.error); loadAgent(); }
      }, 3000);
      await ensureStatus(true);
    } catch (err) { toast(err.message, true); }
  }

  // ------------------------------------------------------------------ Scanner
  async function loadScanner() {
    $('sc-table').innerHTML = `<tbody><tr><td class="ra-muted">${t('sc.scanning')}</td></tr></tbody>`;
    try {
      const data = await api('/api/agent/scan');
      state.scan = data.rows;
      $('sc-date').textContent = data.rows.length ? t('sc.date', { d: data.rows[0].date }) : '';
      const lim = Math.max(0.3, ...data.rows.map((r) => Math.abs(r.ours_signal || 0)));
      $('sc-table').innerHTML = `<thead><tr><th>${t('col.stock')}</th><th class="num">${t('col.price')}</th><th class="num">1d</th><th class="num">20d</th>
        <th>${t('ag.k.signal')}</th><th>Ours</th><th>FinMem</th><th class="num">P(up)</th><th>${t('sc.rec')}</th><th class="num">${t('home.conf')}</th>
        <th class="num">${t('home.news')}</th><th>${t('sc.top')}</th></tr></thead><tbody>` + data.rows.map((r) => {
          const v = r.ours_signal || 0, w = Math.abs(v) / lim * 50;
          const bar = `<span class="ra-meter"><i style="width:90px"><b style="${v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`};background:${v >= 0 ? 'var(--pos)' : 'var(--neg)'}"></b></i> <span class="${signCls(v)}">${snum(v)}</span></span>`;
          return `<tr onclick="ResearchApp.openTicker('${r.ticker}')"><td><b>${r.ticker}</b> <span class="ra-muted">${esc(r.company)}</span></td>
            <td class="num">$${num(r.price)}</td><td class="num ${signCls(r.change_1d)}">${spct(r.change_1d, 2)}</td><td class="num ${signCls(r.change_20d)}">${spct(r.change_20d)}</td>
            <td>${bar}</td><td>${badge(r.abstain ? 'Hold' : r.ours_action)}${r.abstain ? ` <span class="ra-muted">${t('ag.abstain')}</span>` : ''}</td><td>${badge(r.finmem_action)}</td>
            <td class="num">${num(r.p_up)}</td><td>${badge(r.consensus)}</td><td class="num">${pct(r.confidence, 0)}</td>
            <td class="num">${r.n_today}</td><td class="wrap">${esc(String(r.top_memory || "").slice(0, 110))}</td></tr>`;
        }).join('') + '</tbody>';
    } catch (err) { toast(`Scanner: ${err.message}`, true); }
  }

  function openTicker(tk) {
    App.go('agent');
    App.setTicker(tk);
  }

  async function executeAll() {
    const rows = (state.scan || []).filter((r) => r.consensus !== 'Hold');
    if (!rows.length) { toast(t('sc.allhold')); return; }
    if (!confirm(t('sc.exec.confirm', { n: rows.length, list: rows.map((r) => `${App.action(r.consensus)} ${r.ticker}`).join(', ') }))) return;
    let ok = 0;
    for (const r of rows) {
      try {
        const res = await api('/api/agent/execute', { ticker: r.ticker, action: r.consensus, price: r.price, reasoning: 'scanner consensus' });
        if (res.result && !res.result.error) ok += 1;
      } catch (err) { /* reported below */ }
    }
    toast(t('sc.exec.done', { ok, n: rows.length }));
    Trade.fetchPortfolio();
  }

  // ------------------------------------------------------------- Research Lab
  function bindSeg(id, key) {
    const el = $(id);
    if (el.dataset.bound) return;
    el.dataset.bound = '1';
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      el.querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
      state[key] = b.dataset.v;
      loadResearch();
    }));
  }

  async function loadResearch() {
    bindSeg('rl-protocol', 'protocol'); bindSeg('rl-scope', 'scope');
    try {
      if (!state.research[state.protocol]) state.research[state.protocol] = await api(`/api/research?protocol=${state.protocol}`);
      const r = state.research[state.protocol];
      if (!r.available) { $('rl-kpis').innerHTML = `<div class="ra-muted">${t('rl.none')}</div>`; return; }
      renderResearch(r);
    } catch (err) { toast(`Research: ${err.message}`, true); }
  }

  function renderResearch(r) {
    const P = state.scope === 'all' ? r.portfolio : r.finmem5;
    $('rl-range').textContent = t('rl.range', { a: r.test_range[0], b: r.test_range[1], n: r.test_days });
    const o = P.Ours || {}, f = P.FinMem || {}, bh = P['Buy & Hold'] || {}, tf = r.tests.FinMem || {};
    const tiles = [
      [t('rl.k.sharpe'), num(o.sharpe), `FinMem ${num(f.sharpe)} · Buy&Hold ${num(bh.sharpe)}`, 'purple'],
      [t('rl.k.cr'), spct(o.cumulative_return), `FinMem ${spct(f.cumulative_return)}`, 'green'],
      [t('rl.k.mdd'), pct(o.max_drawdown), `FinMem ${pct(f.max_drawdown)}`, 'red'],
      [t('rl.k.dsharpe'), snum(tf.sharpe_diff), `95% CI [${num(tf.ci_low)}, ${num(tf.ci_high)}]`, ''],
      [t('rl.k.wilcoxon'), `p = ${num(tf.wilcoxon_p, 3)}`, `${num(tf.mean_diff_bps, 1)} bps/day · ${tf.wilcoxon_p < 0.05 ? t('rl.sig') : t('rl.notsig')}`, 'orange'],
    ];
    $('rl-kpis').innerHTML = tiles.map(([a, v, s, cls]) =>
      `<div class="kpi-card ${cls}"><div class="kpi-title">${esc(a)}</div><div class="kpi-val">${v}</div><div class="kpi-sub">${esc(s)}</div></div>`).join('');

    // equity
    const curves = r.curves[state.scope];
    const names = Object.keys(curves.series).filter((n) => STRATEGY_ORDER.includes(n));
    if (!state.hiddenInit) { names.forEach((n) => { if (!DEFAULT_ON.includes(n)) state.hidden.add(n); }); state.hiddenInit = true; }
    $('rl-legend').innerHTML = names.map((n) => `<button class="${state.hidden.has(n) ? 'off' : ''}" data-n="${esc(n)}"><i class="ra-dot" style="background:${color(n)}"></i>${esc(n)}</button>`).join('');
    $('rl-legend').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      const n = b.dataset.n; state.hidden.has(n) ? state.hidden.delete(n) : state.hidden.add(n); renderResearch(r);
    }));
    const shown = {}; names.filter((n) => !state.hidden.has(n)).forEach((n) => { shown[n] = curves.series[n]; });
    lineChart($('rl-equity'), curves.date, shown, { height: 360, baseline: 1, fmt: (v) => `${((v - 1) * 100).toFixed(1)}%` });

    // main table
    const order = STRATEGY_ORDER.filter((n) => P[n]);
    $('rl-table').innerHTML = `<thead><tr><th>${t('rl.strategy')}</th><th class="num">CR</th><th class="num">Sharpe</th><th class="num">Sortino</th><th class="num">MDD</th><th class="num">Vol</th><th class="num">Turnover</th></tr></thead><tbody>` +
      order.map((n) => { const m = P[n]; return `<tr class="${n === 'Ours' ? 'ra-focus' : ''}"><td><i class="ra-dot" style="background:${color(n)}"></i> ${esc(n)}</td>
        <td class="num ${signCls(m.cumulative_return)}">${spct(m.cumulative_return)}</td><td class="num">${num(m.sharpe)}</td><td class="num">${num(m.sortino)}</td>
        <td class="num">${pct(m.max_drawdown)}</td><td class="num">${pct(m.annualized_volatility)}</td><td class="num">${num(m.turnover)}</td></tr>`; }).join('') + '</tbody>';

    // tests
    const tests = Object.entries(r.tests).sort((a, b) => (a[1].wilcoxon_p ?? 1) - (b[1].wilcoxon_p ?? 1));
    $('rl-tests').innerHTML = `<thead><tr><th>${t('rl.baseline')}</th><th class="num">Δ bps/day</th><th class="num">ΔSharpe [95% CI]</th><th class="num">Wilcoxon p</th><th class="num">Holm p</th></tr></thead><tbody>` +
      tests.map(([n, v]) => `<tr><td>${esc(n)}</td><td class="num ${signCls(v.mean_diff_bps)}">${snum(v.mean_diff_bps, 1)}</td>
        <td class="num">${snum(v.sharpe_diff)} [${num(v.ci_low)}, ${num(v.ci_high)}]</td><td class="num">${num(v.wilcoxon_p, 3)}</td>
        <td class="num">${v.wilcoxon_p_holm < 0.05 ? '<b>' + num(v.wilcoxon_p_holm, 3) + '</b>' : num(v.wilcoxon_p_holm, 3)}</td></tr>`).join('') + '</tbody>';

    // heatmap
    const tickers = Object.keys(r.per_ticker.Ours || {});
    const heat = (v) => Charts.heat(v, 2.5);
    $('rl-heatmap').innerHTML = `<table class="ra-table ra-heat"><thead><tr><th>${t('rl.strategy')}</th>${tickers.map((t) => `<th class="num">${t}</th>`).join('')}</tr></thead><tbody>` +
      order.map((n) => `<tr class="${n === 'Ours' ? 'ra-focus' : ''}"><td>${esc(n)}</td>${tickers.map((t) => { const v = (r.per_ticker[n] || {})[t]; const s = v ? v.sharpe : null;
        return `<td class="num" style="background:${s == null ? 'transparent' : heat(s)}">${num(s)}</td>`; }).join('')}</tr>`).join('') + '</tbody></table>';

    // model zoo (extension experiments)
    const z = r.extensions;
    $('rl-zoo-card').hidden = !z;
    if (z) {
      const zooOrder = ['Buy & Hold', 'LGBM (price)', 'GRU (price)', 'Transformer (price)', 'Transformer (price+memory)', 'RL Q-learning',
        'Pattern analogs (kNN)', 'Ours', 'FinMem', 'LGBM + memory risk overlay', 'Model vote (LGBM, Transformer, Ours)'].filter((n) => z.portfolio[n]);
      const explore = (n) => n.startsWith('LGBM +') || n.startsWith('Model vote');
      $('rl-zoo').innerHTML = `<thead><tr><th>${t('rl.strategy')}</th><th class="num">CR</th><th class="num">Sharpe</th><th class="num">MDD</th>
        <th class="num">Beta</th><th class="num">Alpha/yr</th><th class="num">ΔSharpe vs Ours [95% CI]</th></tr></thead><tbody>` +
        zooOrder.map((n) => { const m = z.portfolio[n], tt = z.tests[`${n} vs Ours`];
          return `<tr class="${n === 'Ours' ? 'ra-focus' : ''}"><td>${esc(n)}${explore(n) ? ' *' : ''}</td><td class="num ${signCls(m.cumulative_return)}">${spct(m.cumulative_return)}</td>
          <td class="num">${num(m.sharpe)}</td><td class="num">${pct(m.max_drawdown)}</td><td class="num">${num(m.beta)}</td>
          <td class="num ${signCls(m.alpha_annual)}">${spct(m.alpha_annual)}</td>
          <td class="num">${tt ? `${snum(tt.sharpe_diff)} [${num(tt.ci_low, 1)}, ${num(tt.ci_high, 1)}]` : '–'}</td></tr>`; }).join('') + '</tbody>';
    }

    // IC
    const icRows = ['Plain RAG', 'FinMem', 'Ours', 'Ours (signed N1)'].filter((n) => r.ic[n]).map((n) => ({
      label: n, value: r.ic[n].ic_pooled, color: color(n), text: `${snum(r.ic[n].ic_pooled, 3)} (p=${num(r.ic[n].ic_pooled_p)})` }));
    hbars($('rl-ic'), icRows, { labelWidth: 130 });

    // LLM
    if (r.llm && r.llm.length) {
      const bbs = [...new Set(r.llm.map((x) => x.backbone))];
      $('rl-llm').innerHTML = `<thead><tr><th>Backbone</th><th>System</th><th class="num">CR</th><th class="num">Sharpe</th><th class="num">Buy/Sell/Hold</th><th class="num">Decisions</th></tr></thead><tbody>` +
        bbs.flatMap((bb) => ['No-Memory', 'FinMem', 'Ours'].map((s) => r.llm.find((x) => x.backbone === bb && x.system === s)).filter(Boolean).map((x) =>
          `<tr class="${x.system === 'Ours' ? 'ra-focus' : ''}"><td>${esc(x.backbone)}</td><td>${esc(x.system)}</td><td class="num ${signCls(x.cumulative_return)}">${spct(x.cumulative_return)}</td>
          <td class="num">${num(x.sharpe)}</td><td class="num">${pct(x.buy, 0)} / ${pct(x.sell, 0)} / ${pct(x.hold, 0)}</td><td class="num">${x.n}</td></tr>`)).join('') + '</tbody>';
    } else {
      $('rl-llm').innerHTML = `<tbody><tr><td class="ra-muted">${t('rl.llm.none')}</td></tr></tbody>`;
    }

    // memory ops
    const ops = r.memory_stats;
    const opNames = ['items', 'merged', 'retired', 'promoted', 'purged'];
    $('rl-ops').innerHTML = `<thead><tr><th>${t('ag.th.system')}</th>${opNames.map((k) => `<th class="num">${k}</th>`).join('')}</tr></thead><tbody>` +
      ['Plain RAG', 'FinMem', 'Ours'].filter((n) => ops[n]).map((n) => `<tr class="${n === 'Ours' ? 'ra-focus' : ''}"><td>${n}</td>${opNames.map((k) => `<td class="num">${(ops[n][k] ?? 0).toLocaleString()}</td>`).join('')}</tr>`).join('') + '</tbody>';

    // credibility
    const cred = (r.credibility || []).filter((c) => c.level === 'c').sort((a, b) => b.posterior - a.posterior);
    hbars($('rl-cred'), cred.map((c) => ({ label: c.name, value: c.posterior, color: c.posterior >= 0.5 ? 'var(--s-ours)' : 'var(--s-finmem)',
      text: `${num(c.posterior, 3)} (n=${Math.round(c.n).toLocaleString()})` })), { center: 0.5, labelWidth: 110 });

    // findings (computed from the numbers shown above)
    const ff = r.finmem5;
    const ic = r.ic;
    const tF = r.tests.FinMem || {};
    $('rl-findings').innerHTML = [
      t('rl.f1', { o: num(r.portfolio.Ours.sharpe), f: num(r.portfolio.FinMem.sharpe), p: num(r.portfolio['Plain RAG'].sharpe), o5: spct(ff.Ours.cumulative_return), f5: spct(ff.FinMem.cumulative_return) }),
      t(tF.wilcoxon_p < 0.05 ? 'rl.f2.sig' : 'rl.f2.notsig', { p: num(tF.wilcoxon_p, 3), lo: num(tF.ci_low), hi: num(tF.ci_high) }),
      t('rl.f3', { o: snum(ic.Ours.ic_pooled, 3), f: snum(ic.FinMem.ic_pooled, 3), p: snum(ic['Plain RAG'].ic_pooled, 3), l: num(r.portfolio['LGBM (price)'].sharpe), b: num(r.portfolio['Buy & Hold'].sharpe) }),
      t('rl.f4', { p: num((r.tests['Daily News Sentiment'] || {}).wilcoxon_p_holm, 3) }),
].map((x) => `<li>${x}</li>`).join('');

    // figures
    $('rl-figures').innerHTML = (r.figures || []).map((f) => `<figure data-f="${esc(f)}"><img loading="lazy" src="/figures/${esc(f)}.png" alt="${esc(f)}"><figcaption>${esc(f.replace(/^fig\d+_/, '').replace(/_/g, ' '))}</figcaption></figure>`).join('');
    $('rl-figures').querySelectorAll('figure').forEach((fig) => fig.addEventListener('click', () => {
      const lb = $('ra-lightbox'); lb.querySelector('img').src = `/figures/${fig.dataset.f}.png`; lb.hidden = false;
    }));
  }

  App.register('agent', loadAgent);
  App.register('scanner', loadScanner);
  App.register('research', loadResearch);
  App.on('lang', () => {
    const pg = App.state.page;
    if (pg === 'agent' && state.brief) { renderStatus(); renderBrief(); }
    if (pg === 'scanner') loadScanner();
    if (pg === 'research' && state.research[state.protocol]) renderResearch(state.research[state.protocol]);
  });
  window.addEventListener('resize', () => {
    if (App.state.page === 'research' && state.research[state.protocol]) renderResearch(state.research[state.protocol]);
    if (App.state.page === 'agent' && state.brief) renderHistory();
  });

  return { loadAgent, askLLM, execute, runAgent, refreshData, loadScanner, openTicker, executeAll, loadResearch };
})();
