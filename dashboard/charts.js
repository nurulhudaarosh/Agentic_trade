/* Theme-aware SVG charts shared by the Trade and Insights pages (colours come from CSS variables). */
const Charts = (() => {
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const mag = 10 ** Math.floor(Math.log10(span / n));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 1);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function attachCrosshair(container, W, n, m, x, rowsAt, labelAt) {
    const svg = container.querySelector('svg'), tip = container.querySelector('.ra-tip'), cross = container.querySelector('.ch-cross');
    const hit = svg.querySelector('.ch-hit');
    const move = (clientX) => {
      const r = svg.getBoundingClientRect();
      const px = (clientX - r.left) * (W / r.width);
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (W - m.l - m.r) * (n - 1))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
      const rows = rowsAt(i);
      if (!rows.length) { tip.hidden = true; return; }
      tip.innerHTML = `<div style="margin-bottom:4px;font-weight:700">${esc(labelAt(i))}</div>` + rows.map(([nm, c, v]) =>
        `<div class="ra-tip-row"><span><i class="ra-dot" style="background:${c}"></i>${esc(nm)}</span><b>${esc(v)}</b></div>`).join('');
      tip.hidden = false;
      const left = (x(i) / W) * r.width;
      tip.style.left = `${left > r.width * 0.6 ? left - tip.offsetWidth - 12 : left + 12}px`;
      tip.style.top = '8px';
    };
    hit.addEventListener('mousemove', (ev) => move(ev.clientX));
    hit.addEventListener('touchmove', (ev) => { move(ev.touches[0].clientX); }, { passive: true });
    const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); };
    hit.addEventListener('mouseleave', hide);
    hit.addEventListener('touchend', hide);
  }

  /**
   * Line chart with shaded bands, faint background lines and a crosshair tooltip.
   * lines: [{name, values, color, width, dash, faint, title}], bands: [{name, lo, hi, color}]
   */
  function line(container, labels, { lines = [], bands = [], fmt = (v) => Number(v).toFixed(2), height = 300, vline, vlineLabel = '', baseline, xfmt } = {}) {
    const W = container.clientWidth || 900, H = height, m = { l: 58, r: 14, t: 10, b: 26 };
    let lo = Infinity, hi = -Infinity;
    const see = (v) => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
    lines.forEach((l) => l.values.forEach(see));
    bands.forEach((b) => { b.lo.forEach(see); b.hi.forEach(see); });
    if (baseline != null) see(baseline);
    if (!Number.isFinite(lo)) { container.innerHTML = ''; return; }
    const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.05 || 0.01; lo -= pad; hi += pad;
    const n = labels.length;
    const x = (i) => m.l + (i / Math.max(n - 1, 1)) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const lab = (i) => xfmt ? xfmt(labels[i], i) : labels[i];
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    niceTicks(lo, hi).forEach((tk) => {
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(tk)}" y2="${y(tk)}" stroke="var(--grid)"/>`;
      s += `<text x="${m.l - 8}" y="${y(tk) + 4}" text-anchor="end">${fmt(tk)}</text>`;
    });
    if (baseline != null) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(baseline)}" y2="${y(baseline)}" stroke="var(--text-dim)" stroke-dasharray="3 3"/>`;
    const step = Math.max(1, Math.ceil(n / Math.max(3, Math.min(8, Math.floor(W / 80)))));
    for (let i = 0; i < n; i += step) s += `<text x="${x(i)}" y="${H - 7}" text-anchor="middle">${esc(lab(i))}</text>`;
    if (vline != null) s += `<line x1="${x(vline)}" x2="${x(vline)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-muted)" stroke-dasharray="4 3"/><text x="${x(vline) + 4}" y="${m.t + 10}">${esc(vlineLabel)}</text>`;
    const pts = (vals) => vals.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean);
    bands.forEach((b) => {
      const top = pts(b.hi), bot = pts(b.lo).reverse();
      if (top.length) s += `<polygon points="${top.concat(bot).join(' ')}" fill="${b.color}" stroke="none"/>`;
    });
    lines.forEach((l) => {
      s += `<polyline fill="none" stroke="${l.color}" stroke-width="${l.width || 1.8}" stroke-linejoin="round"${l.dash ? ` stroke-dasharray="${l.dash}"` : ''}${l.faint ? ' opacity="0.4"' : ''} points="${pts(l.values).join(' ')}">${l.title ? `<title>${esc(l.title)}</title>` : ''}</polyline>`;
    });
    s += `<line class="ch-cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-dim)" visibility="hidden"/>`;
    s += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" class="ch-hit"/></svg>`;
    container.innerHTML = s + '<div class="ra-tip" hidden></div>';
    const named = lines.filter((l) => !l.faint && l.name);
    attachCrosshair(container, W, n, m, x, (i) => named.filter((l) => l.values[i] != null).map((l) => [l.name, l.color, fmt(l.values[i])])
      .concat(bands.filter((b) => b.name && b.lo[i] != null).map((b) => [b.name, b.color, `${fmt(b.lo[i])} – ${fmt(b.hi[i])}`])), lab);
  }

  /** Candlestick / line price chart with moving averages and bands. */
  function price(container, data, opts) {
    const { dates, candles, prices, ind = {} } = data;
    const W = container.clientWidth || 800, H = opts.height || 380, m = { l: 58, r: 14, t: 12, b: 26 };
    const n = dates.length;
    if (!n) { container.innerHTML = ''; return; }
    const vals = [...prices];
    if (opts.candles) candles.forEach((c) => vals.push(c.high, c.low));
    if (opts.bands) ['bb_upper', 'bb_lower'].forEach((k) => (ind[k] || []).forEach((v) => v != null && vals.push(v)));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
    const x = (i) => m.l + ((i + 0.5) / n) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    niceTicks(lo, hi).forEach((tk) => {
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(tk)}" y2="${y(tk)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${y(tk) + 4}" text-anchor="end">$${tk}</text>`;
    });
    const path = (arr) => arr.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).join(' ');
    if (opts.bands && ind.bb_upper) {
      const top = ind.bb_upper.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean);
      const bot = ind.bb_lower.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).reverse();
      s += `<polygon points="${top.concat(bot).join(' ')}" fill="var(--primary-soft)"/>`;
    }
    if (opts.candles) {
      const bw = Math.max(1.5, Math.min(9, (W - m.l - m.r) / n * 0.6));
      candles.forEach((c, i) => {
        const col = c.close >= c.open ? 'var(--candle-up)' : 'var(--candle-down)';
        s += `<line x1="${x(i)}" x2="${x(i)}" y1="${y(c.high)}" y2="${y(c.low)}" stroke="${col}"/>`;
        s += `<rect x="${x(i) - bw / 2}" y="${Math.min(y(c.open), y(c.close))}" width="${bw}" height="${Math.max(1, Math.abs(y(c.close) - y(c.open)))}" fill="${col}" rx="1"/>`;
      });
    } else {
      s += `<polyline fill="none" stroke="var(--price-line)" stroke-width="2" points="${path(prices)}"/>`;
    }
    if (opts.sma20 && ind.sma_20) s += `<polyline fill="none" stroke="var(--cyan)" stroke-width="1.6" points="${path(ind.sma_20)}"/>`;
    if (opts.sma50 && ind.sma_50) s += `<polyline fill="none" stroke="var(--purple)" stroke-width="1.6" points="${path(ind.sma_50)}"/>`;
    const step = Math.max(1, Math.ceil(n / Math.max(3, Math.min(7, Math.floor(W / 95)))));
    for (let i = 0; i < n; i += step) s += `<text x="${x(i)}" y="${H - 7}" text-anchor="middle">${esc(String(dates[i]).slice(0, 10))}</text>`;
    s += `<line class="ch-cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-dim)" visibility="hidden"/>`;
    s += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" class="ch-hit"/></svg>`;
    container.innerHTML = s + '<div class="ra-tip" hidden></div>';
    const money = (v) => `$${Number(v).toFixed(2)}`;
    attachCrosshair(container, W, n, { l: m.l, r: m.r }, x, (i) => {
      const c = candles[i] || {};
      const rows = opts.candles && c.open != null
        ? [[opts.labels.open, 'var(--text-dim)', money(c.open)], [opts.labels.close, 'var(--price-line)', money(c.close)]]
        : [[opts.labels.close, 'var(--price-line)', money(prices[i])]];
      if (opts.sma20 && ind.sma_20 && ind.sma_20[i] != null) rows.push([opts.labels.sma20, 'var(--cyan)', money(ind.sma_20[i])]);
      if (opts.sma50 && ind.sma_50 && ind.sma_50[i] != null) rows.push([opts.labels.sma50, 'var(--purple)', money(ind.sma_50[i])]);
      return rows;
    }, (i) => String(dates[i]).slice(0, 16).replace('T', ' '));
  }

  /** Small oscillator panel: lines + optional histogram + reference levels. */
  function oscillator(container, n, { lines = [], hist, levels = [], lo, hi, height = 120 }) {
    const W = container.clientWidth || 800, H = height, m = { l: 58, r: 14, t: 6, b: 6 };
    const x = (i) => m.l + ((i + 0.5) / n) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    levels.forEach((lv) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(lv.v)}" y2="${y(lv.v)}" stroke="${lv.color}" stroke-dasharray="3 3"/><text x="${m.l - 8}" y="${y(lv.v) + 4}" text-anchor="end">${lv.v}</text>`; });
    if (hist) {
      const bw = Math.max(1, (W - m.l - m.r) / n * 0.6);
      hist.forEach((v, i) => { if (v == null) return; s += `<rect x="${x(i) - bw / 2}" y="${Math.min(y(0), y(v))}" width="${bw}" height="${Math.abs(y(v) - y(0))}" fill="${v >= 0 ? 'var(--candle-up)' : 'var(--candle-down)'}" opacity="0.55"/>`; });
    }
    lines.forEach((l) => { s += `<polyline fill="none" stroke="${l.color}" stroke-width="1.5" points="${l.values.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).join(' ')}"/>`; });
    container.innerHTML = s + '</svg>';
  }

  /** Histogram from bin edges; optional overlay curve and vertical markers. */
  function histogram(container, edges, counts, { overlay, markers = [], fmt = (v) => Number(v).toFixed(2), height = 240, colorOf } = {}) {
    const W = container.clientWidth || 600, H = height, m = { l: 40, r: 12, t: 16, b: 26 };
    const lo = edges[0], hi = edges[edges.length - 1];
    const top = Math.max(...counts, ...(overlay || [0])) * 1.08 || 1;
    const x = (v) => m.l + ((v - lo) / (hi - lo || 1)) * (W - m.l - m.r);
    const y = (c) => m.t + (1 - c / top) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    niceTicks(0, top, 4).forEach((tk) => {
      s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(tk)}" y2="${y(tk)}" stroke="var(--grid)"/><text x="${m.l - 6}" y="${y(tk) + 4}" text-anchor="end">${tk}</text>`;
    });
    counts.forEach((c, i) => {
      const x0 = x(edges[i]), x1 = x(edges[i + 1]);
      const mid = (edges[i] + edges[i + 1]) / 2;
      s += `<rect x="${x0 + 0.5}" y="${y(c)}" width="${Math.max(1, x1 - x0 - 1)}" height="${y(0) - y(c)}" fill="${colorOf ? colorOf(mid) : 'var(--primary)'}" opacity="0.8"><title>${fmt(edges[i])} – ${fmt(edges[i + 1])}: ${c}</title></rect>`;
    });
    if (overlay) {
      const pts = overlay.map((v, i) => `${x((edges[i] + edges[i + 1]) / 2).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
      s += `<polyline fill="none" stroke="var(--yellow)" stroke-width="1.6" stroke-dasharray="4 3" points="${pts}"/>`;
    }
    niceTicks(lo, hi, 6).forEach((tk) => { s += `<text x="${x(tk)}" y="${H - 7}" text-anchor="middle">${fmt(tk)}</text>`; });
    markers.forEach((mk, k) => {
      s += `<line x1="${x(mk.x)}" x2="${x(mk.x)}" y1="${m.t}" y2="${y(0)}" stroke="${mk.color}" stroke-width="1.5" stroke-dasharray="3 2"/>`;
      s += `<text x="${x(mk.x) + 4}" y="${m.t + 4 + k * 12}" style="fill:${mk.color}">${esc(mk.label)}</text>`;
    });
    container.innerHTML = s + '</svg>';
  }

  function donut(container, slices) {
    const S = 200, r = 80, w = 26, c = S / 2;
    const total = slices.reduce((a, b) => a + Math.abs(b.value), 0) || 1;
    let a0 = -Math.PI / 2, s = `<svg viewBox="0 0 ${S} ${S}" role="img" class="an-donut"><circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="${w}"/>`;
    slices.forEach((sl) => {
      const frac = Math.abs(sl.value) / total;
      if (frac <= 0.0005) return;
      if (frac >= 0.9999) { s += `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${sl.color}" stroke-width="${w}"><title>${esc(sl.label)}</title></circle>`; return; }
      const a1 = a0 + frac * 2 * Math.PI;
      const p = (a) => `${(c + r * Math.cos(a)).toFixed(2)},${(c + r * Math.sin(a)).toFixed(2)}`;
      s += `<path d="M${p(a0)} A${r},${r} 0 ${frac > 0.5 ? 1 : 0} 1 ${p(a1)}" fill="none" stroke="${sl.color}" stroke-width="${w}"${sl.short ? ' stroke-dasharray="6 3"' : ''}><title>${esc(sl.label)}</title></path>`;
      a0 = a1;
    });
    container.innerHTML = s + '</svg>';
  }

  // Diverging cell colour for heatmaps: blue = positive, red = negative.
  const heat = (v, scale = 1) => {
    const a = Math.max(-1, Math.min(1, v / scale));
    return a >= 0 ? `color-mix(in srgb, var(--primary) ${Math.round(8 + a * 55)}%, transparent)`
      : `color-mix(in srgb, var(--neg) ${Math.round(8 - a * 55)}%, transparent)`;
  };
  const PALETTE = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)', 'var(--c7)', 'var(--c9)', 'var(--c10)', 'var(--c8)'];

  return { niceTicks, line, price, oscillator, histogram, donut, heat, PALETTE };
})();
