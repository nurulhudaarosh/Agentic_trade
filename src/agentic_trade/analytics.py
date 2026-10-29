"""Market analytics for the dashboard's Analytics tab.

* Pattern analogs: find past windows whose price *shape* matches the current one
  and show what followed them (the "has this chart happened before?" view).
* Analog backtest: trade on that forecast walk-forward and compare to buy & hold.
* Risk report: return distribution, VaR / CVaR, drawdown, beta.
* Correlation matrix across the study universe.
* Portfolio: equity allocation, risk contributions and a Monte Carlo
  distribution of future equity.

Everything uses the study's daily panel (``data/study/prices.parquet``). The
analog engine is strictly walk-forward: a past window is eligible only once its
forward outcome is fully observed on the decision day.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .study.data import BENCHMARKS, UNIVERSE

ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT / "data" / "study" / "prices.parquet"
TRADING_DAYS = 252

_lock = threading.Lock()
_cache: dict[str, Any] = {"stamp": None, "close": None}


def _f(v: Any, d: int = 6) -> float | None:
    v = float(v)
    return round(v, d) if np.isfinite(v) else None


def _fl(values: Any, d: int = 6) -> list[float | None]:
    return [_f(v, d) for v in values]


def close_panel() -> pd.DataFrame:
    """Daily closes, one column per ticker, cached until the parquet changes."""
    stamp = PRICES_PATH.stat().st_mtime
    with _lock:
        if _cache["stamp"] != stamp:
            df = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "close"])
            close = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
            close.index = pd.to_datetime(close.index)
            _cache.update(stamp=stamp, close=close.ffill().dropna(how="all"))
        return _cache["close"]


def _check_ticker(ticker: str, close: pd.DataFrame) -> str:
    ticker = ticker.upper()
    if ticker not in close.columns:
        raise ValueError(f"unknown ticker {ticker}; available: {', '.join(UNIVERSE)}")
    return ticker


def _dates(index: pd.Index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def _perf(daily: np.ndarray) -> dict[str, float | None]:
    """Summary statistics of a daily simple-return series."""
    daily = np.asarray(daily, dtype=float)
    if daily.size == 0:
        return {"total_return": None, "ann_return": None, "ann_vol": None, "sharpe": None, "max_drawdown": None}
    equity = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    vol = daily.std(ddof=1) if daily.size > 1 else 0.0
    return {
        "total_return": _f(equity[-1] - 1.0),
        "ann_return": _f(equity[-1] ** (TRADING_DAYS / daily.size) - 1.0),
        "ann_vol": _f(vol * np.sqrt(TRADING_DAYS)),
        "sharpe": _f(daily.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else None,
        "max_drawdown": _f((equity / peak - 1.0).min()),
    }


# --------------------------------------------------------------------------- pattern analogs
class _AnalogIndex:
    """Z-normalised log-price windows of every candidate ticker, flattened."""

    def __init__(self, close: pd.DataFrame, tickers: list[str], window: int, horizon: int) -> None:
        self.window, self.horizon = window, horizon
        self.close = close[tickers].to_numpy(dtype=float)
        logp = np.log(self.close)
        shapes, owner, end = [], [], []
        for j in range(len(tickers)):
            wins = np.lib.stride_tricks.sliding_window_view(logp[:, j], window)
            shapes.append(wins)
            owner.append(np.full(len(wins), j))
            end.append(np.arange(window - 1, logp.shape[0]))
        z = np.concatenate(shapes)
        z = z - z.mean(axis=1, keepdims=True)
        sd = z.std(axis=1, keepdims=True)
        self.z = np.divide(z, sd, out=np.zeros_like(z), where=sd > 0)
        self.owner = np.concatenate(owner)
        self.end = np.concatenate(end)
        self.valid = np.isfinite(self.z).all(axis=1) & (sd[:, 0] > 0)

    def query(self, j: int, t: int) -> np.ndarray:
        return self.z[self.owner_offset(j) + t - (self.window - 1)]

    def owner_offset(self, j: int) -> int:
        return j * (self.close.shape[0] - self.window + 1)

    def match(self, j: int, t: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k analogs for ticker ``j`` on day ``t`` using only information known at ``t``."""
        q = self.query(j, t)
        eligible = self.valid & (self.end + self.horizon <= t)
        # Same ticker: the analog must not overlap the window being matched.
        eligible &= ~((self.owner == j) & (self.end > t - self.window))
        idx = np.flatnonzero(eligible)
        if idx.size == 0:
            return idx, np.array([])
        sim = self.z[idx] @ q / self.window  # Pearson correlation of the two shapes
        order = idx[np.argsort(-sim)]
        sim_sorted = np.sort(sim)[::-1]
        chosen, chosen_sim = [], []
        min_sep = max(1, self.window // 2)
        for c, s in zip(order, sim_sorted):
            # One analog per episode: skip near-duplicates of an already chosen window.
            if any(self.owner[c] == self.owner[p] and abs(self.end[c] - self.end[p]) < min_sep for p in chosen):
                continue
            chosen.append(c)
            chosen_sim.append(s)
            if len(chosen) == k:
                break
        return np.array(chosen, dtype=int), np.array(chosen_sim)

    def outcomes(self, chosen: np.ndarray) -> np.ndarray:
        """Forward paths (k x horizon) as simple return from each analog's last day."""
        h = np.arange(1, self.horizon + 1)
        e, j = self.end[chosen], self.owner[chosen]
        return self.close[e[:, None] + h[None, :], j[:, None]] / self.close[e, j][:, None] - 1.0


def _forecast(paths: np.ndarray, sims: np.ndarray, threshold: float) -> dict[str, Any]:
    w = np.clip(sims, 1e-6, None)
    w = w / w.sum()
    final = paths[:, -1]
    expected = float(final @ w)
    p_up = float(w[final > 0].sum())
    if expected > threshold and p_up >= 0.55:
        action = "Buy"
    elif expected < -threshold and p_up <= 0.45:
        action = "Sell"
    else:
        action = "Hold"
    return {"expected": expected, "p_up": p_up, "action": action}


def _scope_tickers(close: pd.DataFrame, ticker: str, scope: str) -> list[str]:
    if scope == "universe":
        return [t for t in UNIVERSE if t in close.columns]
    return [ticker]


def pattern_analogs(ticker: str = "TSLA", window: int = 20, horizon: int = 10, k: int = 8,
                    scope: str = "ticker", threshold: float = 0.01) -> dict[str, Any]:
    """Past windows that look like the latest ``window`` days and what happened next."""
    close = close_panel()
    ticker = _check_ticker(ticker, close)
    if not (5 <= window <= 120 and 1 <= horizon <= 60 and 1 <= k <= 30):
        raise ValueError("window must be 5-120, horizon 1-60, k 1-30")
    tickers = _scope_tickers(close, ticker, scope)
    if ticker not in tickers:
        tickers = [ticker] + tickers
    index = _AnalogIndex(close, tickers, window, horizon)
    j, t = tickers.index(ticker), len(close) - 1
    chosen, sims = index.match(j, t, k)
    if chosen.size == 0:
        raise ValueError("not enough history for this window / horizon")
    paths = index.outcomes(chosen)
    fc = _forecast(paths, sims, threshold)
    px = index.close[:, j]
    current = px[t - window + 1: t + 1] / px[t] - 1.0
    dates = close.index
    matches = []
    for c, s, path in zip(chosen, sims, paths):
        e, o = int(index.end[c]), int(index.owner[c])
        series = index.close[e - window + 1: e + 1, o] / index.close[e, o] - 1.0
        matches.append({
            "ticker": tickers[o], "start": dates[e - window + 1].strftime("%Y-%m-%d"),
            "end": dates[e].strftime("%Y-%m-%d"), "similarity": _f(s, 4),
            "window_path": _fl(series), "forward_path": _fl(path),
            "forward_return": _f(path[-1]), "max_up": _f(path.max()), "max_down": _f(path.min()),
        })
    finals = paths[:, -1]
    return {
        "ticker": ticker, "scope": scope, "window": window, "horizon": horizon, "k": int(chosen.size),
        "as_of": dates[t].strftime("%Y-%m-%d"), "price": _f(px[t], 2),
        "current_path": _fl(current),
        "forecast": {
            "expected_return": _f(fc["expected"]), "p_up": _f(fc["p_up"], 4), "action": fc["action"],
            "threshold": threshold, "median_return": _f(np.median(finals)),
            "best": _f(finals.max()), "worst": _f(finals.min()),
            "mean_similarity": _f(sims.mean(), 4),
            "mean_path": _fl(paths.mean(axis=0)),
            "p25_path": _fl(np.percentile(paths, 25, axis=0)),
            "p75_path": _fl(np.percentile(paths, 75, axis=0)),
        },
        "matches": matches,
    }


def min_start(window: int, horizon: int) -> int:
    """First backtest day: leave enough fully observed analogs to match against."""
    return window + horizon + max(40, window)


def analog_backtest(ticker: str = "TSLA", window: int = 20, horizon: int = 10, k: int = 8,
                    scope: str = "ticker", threshold: float = 0.01, long_only: bool = False,
                    cost_bps: float = 5.0, start: int | None = None) -> dict[str, Any]:
    """Walk-forward: every day, act on the analog forecast and hold for one day."""
    close = close_panel()
    ticker = _check_ticker(ticker, close)
    tickers = _scope_tickers(close, ticker, scope)
    if ticker not in tickers:
        tickers = [ticker] + tickers
    index = _AnalogIndex(close, tickers, window, horizon)
    j, n = tickers.index(ticker), len(close)
    px = index.close[:, j]
    if start is None:
        start = min_start(window, horizon)
    if start >= n - 2:
        raise ValueError("not enough history to backtest this window / horizon")

    positions, expected, realized = [], [], []
    for t in range(start, n - 1):
        chosen, sims = index.match(j, t, k)
        if chosen.size == 0:
            positions.append(0.0)
            expected.append(np.nan)
        else:
            fc = _forecast(index.outcomes(chosen), sims, threshold)
            pos = {"Buy": 1.0, "Sell": -1.0, "Hold": 0.0}[fc["action"]]
            positions.append(max(pos, 0.0) if long_only else pos)
            expected.append(fc["expected"])
        realized.append(px[t + horizon] / px[t] - 1.0 if t + horizon < n else np.nan)

    pos = np.array(positions)
    next_ret = px[start + 1:] / px[start:-1] - 1.0
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * next_ret - turnover * cost_bps / 1e4
    expected_a, realized_a = np.array(expected), np.array(realized)

    active = (pos != 0) & np.isfinite(realized_a)
    hit = float((np.sign(pos[active]) == np.sign(realized_a[active])).mean()) if active.any() else None
    both = np.isfinite(expected_a) & np.isfinite(realized_a)
    ic = float(pd.Series(expected_a[both]).corr(pd.Series(realized_a[both]), method="spearman")) if both.sum() > 10 else None

    trades, side, entry = [], 0.0, None
    for i, p in enumerate(np.append(pos, 0.0)):
        if p != side:
            if side != 0 and entry is not None:
                trades.append(side * (px[start + i] / px[start + entry] - 1.0))
            side, entry = p, i
    trades_a = np.array(trades)
    dates = _dates(close.index[start: n])
    return {
        "ticker": ticker, "scope": scope, "window": window, "horizon": horizon, "k": k,
        "threshold": threshold, "long_only": long_only, "cost_bps": cost_bps,
        "dates": dates,
        "equity": {
            "Analog strategy": _fl(np.concatenate([[1.0], np.cumprod(1.0 + strat)])),
            "Buy & Hold": _fl(np.concatenate([[1.0], np.cumprod(1.0 + next_ret)])),
        },
        "positions": _fl(np.append(pos, pos[-1] if pos.size else 0.0), 1),
        "metrics": {
            "Analog strategy": {**_perf(strat), "hit_rate": _f(hit, 4) if hit is not None else None,
                                "rank_ic": _f(ic, 4) if ic is not None else None,
                                "exposure": _f((pos != 0).mean(), 4), "trades": int(trades_a.size),
                                "win_rate": _f((trades_a > 0).mean(), 4) if trades_a.size else None,
                                "avg_trade": _f(trades_a.mean()) if trades_a.size else None},
            "Buy & Hold": _perf(next_ret),
        },
    }


# --------------------------------------------------------------------------- risk
def risk_report(ticker: str = "TSLA", bins: int = 40) -> dict[str, Any]:
    close = close_panel()
    ticker = _check_ticker(ticker, close)
    px = close[ticker].dropna()
    r = px.pct_change().dropna()
    counts, edges = np.histogram(r, bins=bins)
    q05 = float(r.quantile(0.05))
    equity = px / px.iloc[0]
    drawdown = equity / equity.cummax() - 1.0
    beta = corr = None
    if "SPY" in close.columns:
        spy = close["SPY"].pct_change().reindex(r.index)
        both = pd.concat([r, spy], axis=1).dropna()
        if len(both) > 20:
            cov = np.cov(both.iloc[:, 0], both.iloc[:, 1])
            beta, corr = cov[0, 1] / cov[1, 1], both.iloc[:, 0].corr(both.iloc[:, 1])
    vol20 = r.rolling(20).std() * np.sqrt(TRADING_DAYS)
    mu, sd = float(r.mean()), float(r.std())
    normal = len(r) * np.diff(edges) * np.exp(-0.5 * ((edges[:-1] + np.diff(edges) / 2 - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return {
        "ticker": ticker, "as_of": px.index[-1].strftime("%Y-%m-%d"), "n": int(len(r)),
        "histogram": {"edges": _fl(edges), "counts": [int(c) for c in counts], "normal": _fl(normal, 3)},
        "stats": {
            **_perf(r.to_numpy()),
            "mean_daily": _f(mu), "std_daily": _f(sd),
            "skew": _f(r.skew(), 3), "excess_kurtosis": _f(r.kurt(), 3),
            "var_95": _f(-q05), "cvar_95": _f(-r[r <= q05].mean()),
            "best_day": _f(r.max()), "best_date": r.idxmax().strftime("%Y-%m-%d"),
            "worst_day": _f(r.min()), "worst_date": r.idxmin().strftime("%Y-%m-%d"),
            "up_days": _f((r > 0).mean(), 4),
            "current_drawdown": _f(drawdown.iloc[-1]),
            "beta_spy": _f(beta, 3) if beta is not None else None,
            "corr_spy": _f(corr, 3) if corr is not None else None,
            "vol_20d": _f(vol20.iloc[-1]),
        },
        "dates": _dates(px.index),
        "drawdown": _fl(drawdown),
        "vol_20d": _fl(vol20.reindex(px.index)),
    }


def correlation_matrix(lookback: int = 120) -> dict[str, Any]:
    close = close_panel()
    tickers = [t for t in UNIVERSE + BENCHMARKS if t in close.columns]
    r = close[tickers].pct_change().dropna(how="all")
    if lookback > 0:
        r = r.tail(lookback)
    c = r.corr()
    return {"tickers": tickers, "lookback": int(len(r)), "as_of": r.index[-1].strftime("%Y-%m-%d"),
            "matrix": [_fl(row, 3) for row in c.to_numpy()]}


# --------------------------------------------------------------------------- portfolio
def portfolio_distribution(state: dict[str, Any], ticker: str = "TSLA", days: int = 21,
                           paths: int = 2000, lookback: int = 250, seed: int = 7) -> dict[str, Any]:
    """Allocation, risk contributions and a bootstrap Monte Carlo of future equity.

    Monte Carlo resamples whole historical days (keeping the cross-asset
    correlation of that day) and applies them to today's exposures. With no open
    positions it shows the hypothetical of putting all equity in ``ticker``.
    """
    if not 1 <= days <= TRADING_DAYS:
        raise ValueError(f"days must be 1-{TRADING_DAYS}")
    close = close_panel()
    equity = float(state.get("equity") or 0.0) or 100000.0
    cash = float(state.get("cash") or 0.0)
    exposures: dict[str, float] = {}
    for p in state.get("positions") or []:
        t = str(p.get("ticker", "")).upper()
        if t not in close.columns:
            continue
        sign = -1.0 if p.get("side") == "Short" else 1.0
        exposures[t] = exposures.get(t, 0.0) + sign * float(p.get("shares", 0.0)) * float(p.get("current_price", 0.0))

    hypothetical = not exposures
    if hypothetical:
        exposures = {_check_ticker(ticker, close): equity}
    tickers = list(exposures)
    w = np.array([exposures[t] / equity for t in tickers])
    rets = close[tickers].pct_change().dropna().tail(lookback)
    cov = rets.cov().to_numpy() * TRADING_DAYS
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
    contrib = (w * (cov @ w)) / port_var if port_var > 0 else np.zeros_like(w)
    port_daily = rets.to_numpy() @ w
    beta = None
    if "SPY" in close.columns:
        spy = close["SPY"].pct_change().reindex(rets.index).to_numpy()
        ok = np.isfinite(spy)
        if ok.sum() > 20:
            beta = float(np.cov(port_daily[ok], spy[ok])[0, 1] / np.var(spy[ok], ddof=1))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(port_daily), size=(paths, days))
    sim = equity * np.cumprod(1.0 + port_daily[draws], axis=1)
    sim = np.concatenate([np.full((paths, 1), equity), sim], axis=1)
    final = sim[:, -1]
    counts, edges = np.histogram(final, bins=36)
    pct_levels = [5, 25, 50, 75, 95]
    fan = {f"p{p}": _fl(np.percentile(sim, p, axis=0), 2) for p in pct_levels}
    q05 = np.percentile(port_daily, 5)

    allocation = [{"ticker": t, "exposure": _f(exposures[t], 2), "weight": _f(wi, 4), "risk_share": _f(ci, 4)}
                  for t, wi, ci in zip(tickers, w, contrib)]
    if not hypothetical:
        allocation.append({"ticker": "Cash", "exposure": _f(cash, 2), "weight": _f(cash / equity, 4), "risk_share": 0.0})
    return {
        "hypothetical": hypothetical, "basis": f"100% {tickers[0]}" if hypothetical else "current portfolio",
        "equity": _f(equity, 2), "days": days, "paths": paths, "lookback": int(len(rets)),
        "allocation": allocation,
        "risk": {
            "ann_vol": _f(port_vol), "beta_spy": _f(beta, 3) if beta is not None else None,
            "gross_exposure": _f(np.abs(w).sum(), 4), "net_exposure": _f(w.sum(), 4),
            "var_95_1d": _f(-q05 * equity, 2), "cvar_95_1d": _f(-port_daily[port_daily <= q05].mean() * equity, 2),
        },
        "monte_carlo": {
            "fan": fan,
            "histogram": {"edges": _fl(edges, 2), "counts": [int(c) for c in counts]},
            "expected": _f(final.mean(), 2), "median": _f(np.median(final), 2),
            "p_loss": _f((final < equity).mean(), 4),
            "p_loss_5pct": _f((final < equity * 0.95).mean(), 4),
            "p5": _f(np.percentile(final, 5), 2), "p95": _f(np.percentile(final, 95), 2),
        },
    }


def analog_sweep(ticker: str = "TSLA", scope: str = "universe", k: int = 8, threshold: float = 0.01,
                 long_only: bool = False, windows: tuple[int, ...] = (10, 20, 40),
                 horizons: tuple[int, ...] = (5, 10, 20)) -> dict[str, Any]:
    """Backtest a window x horizon grid so a single lucky setting is not over-read."""
    # Every cell is scored over the same days as the slowest-to-warm-up setting.
    start = max(min_start(w, h) for w in windows for h in horizons)
    cells = []
    for w in windows:
        for h in horizons:
            try:
                bt = analog_backtest(ticker, w, h, k, scope, threshold, long_only, start=start)
            except ValueError:
                cells.append({"window": w, "horizon": h, "metrics": None})
                continue
            m = bt["metrics"]["Analog strategy"]
            cells.append({"window": w, "horizon": h, "metrics": {
                key: m[key] for key in ("total_return", "sharpe", "max_drawdown", "hit_rate", "rank_ic", "trades")}})
    bh = analog_backtest(ticker, windows[0], horizons[0], k, scope, threshold, long_only, start=start)["metrics"]["Buy & Hold"]
    return {"ticker": ticker.upper(), "scope": scope, "windows": list(windows), "horizons": list(horizons),
            "start": close_panel().index[start].strftime("%Y-%m-%d"),
            "cells": cells, "buy_and_hold": bh}


# --------------------------------------------------------------------------- budget allocation
ALLOCATION_LABELS = {
    "inverse_vol": "Risk parity (inverse volatility)",
    "min_variance": "Minimum variance",
    "equal": "Equal weight",
    "signal_tilt": "Risk parity tilted by agent signals",
}
SIGNAL_TILT = {"Buy": 1.5, "Hold": 1.0, "Sell": 0.0}


def _cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Scale to sum 1 then clip at ``cap``, handing the excess to the unclipped names.
    If ``cap * n < 1`` the result sums to less than 1 (the rest stays in cash)."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(len(w) + 1):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = (w < cap - 1e-12) & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return np.minimum(w, cap)


def _min_variance(cov: np.ndarray, cap: float) -> np.ndarray:
    from scipy.optimize import minimize

    n = len(cov)
    if cap * n <= 1.0 + 1e-9:
        return np.full(n, cap)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1.0 / n), jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(0.0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-12})
    w = res.x if res.success else np.full(n, 1.0 / n)
    return _cap_weights(np.clip(w, 0.0, None), cap)


def allocation_weights(returns: pd.DataFrame, method: str, cap: float,
                       signals: dict[str, str] | None = None) -> np.ndarray:
    """Weights of the *investable* money (sum <= 1) for ``returns.columns``."""
    vol = returns.std().to_numpy() * np.sqrt(TRADING_DAYS)
    vol = np.where(vol > 0, vol, np.nan)
    if method == "equal":
        raw = np.ones(len(vol))
    elif method in ("inverse_vol", "signal_tilt"):
        raw = np.nan_to_num(1.0 / vol)
        if method == "signal_tilt":
            raw = raw * np.array([SIGNAL_TILT.get((signals or {}).get(t, "Hold"), 1.0) for t in returns.columns])
    elif method == "min_variance":
        return _min_variance(returns.cov().to_numpy() * TRADING_DAYS, cap)
    else:
        raise ValueError(f"method must be one of {', '.join(ALLOCATION_LABELS)}")
    return _cap_weights(raw, cap)


def _allocation_backtest(close: pd.DataFrame, tickers: list[str], cap: float, invest: float,
                         lookback: int, rebalance: int = 21) -> dict[str, Any]:
    """Walk-forward: every ``rebalance`` days set weights from the trailing ``lookback``
    days only, hold them (drifting) until the next rebalance. Cash earns nothing."""
    rets = close[tickers].pct_change().dropna()
    methods = [m for m in ALLOCATION_LABELS if m != "signal_tilt"]
    if len(rets) <= lookback + rebalance:
        return {"dates": [], "equity": {}, "metrics": {}}
    curves: dict[str, list[float]] = {}
    daily: dict[str, list[float]] = {}
    for m in methods:
        value, held, eq, dr = 1.0, None, [1.0], []
        for i in range(lookback, len(rets)):
            if held is None or (i - lookback) % rebalance == 0:
                w = allocation_weights(rets.iloc[i - lookback:i], m, cap) * invest
                held = np.append(w, 1.0 - w.sum()) * value  # last slot = cash
            r = np.append(rets.iloc[i].to_numpy(), 0.0)
            held = held * (1.0 + r)
            new_value = held.sum()
            dr.append(new_value / value - 1.0)
            value = new_value
            eq.append(value)
        curves[m], daily[m] = eq, dr
    # Same exposure to an equal mix of SPY for reference.
    if "SPY" in close.columns:
        spy = close["SPY"].pct_change().reindex(rets.index).fillna(0.0).to_numpy()[lookback:] * invest
        curves["SPY (same exposure)"] = list(np.concatenate([[1.0], np.cumprod(1.0 + spy)]))
        daily["SPY (same exposure)"] = list(spy)
    dates = _dates(rets.index[lookback - 1:])
    return {"dates": dates, "start": dates[0], "rebalance_days": rebalance,
            "equity": {ALLOCATION_LABELS.get(k, k): _fl(v) for k, v in curves.items()},
            "metrics": {ALLOCATION_LABELS.get(k, k): _perf(np.array(v)) for k, v in daily.items()}}


def allocation_plan(budget: float, reserve_pct: float = 20.0, max_position_pct: float = 25.0,
                    tickers: list[str] | None = None, method: str = "inverse_vol", lookback: int = 120,
                    holdings: dict[str, float] | None = None, prices: dict[str, float] | None = None,
                    signals: dict[str, str] | None = None) -> dict[str, Any]:
    """How much of ``budget`` to put in each stock, and the trades to get there."""
    close = close_panel()
    tickers = [t for t in (tickers or UNIVERSE) if t in close.columns]
    if not tickers:
        raise ValueError("no tickers selected")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if method not in ALLOCATION_LABELS:
        raise ValueError(f"method must be one of {', '.join(ALLOCATION_LABELS)}")
    invest = 1.0 - reserve_pct / 100.0
    cap = min(1.0, (max_position_pct / 100.0) / invest) if invest > 0 else 0.0
    rets = close[tickers].pct_change().dropna().tail(lookback)
    w = allocation_weights(rets, method, cap, signals) * invest  # share of the whole budget
    cov = rets.cov().to_numpy() * TRADING_DAYS
    vol = np.sqrt(np.diag(cov))
    port_vol = float(np.sqrt(w @ cov @ w))
    contrib = (w * (cov @ w)) / (w @ cov @ w) if port_vol > 0 else np.zeros_like(w)
    eff_n = float(1.0 / np.sum((w / w.sum()) ** 2)) if w.sum() > 0 else 0.0
    holdings, prices = holdings or {}, prices or {}
    rows = []
    for t, wi, vi, ci in zip(tickers, w, vol, contrib):
        px = float(prices.get(t) or close[t].iloc[-1])
        target, current = budget * wi, float(holdings.get(t, 0.0))
        delta = target - current
        rows.append({"ticker": t, "weight": _f(wi, 4), "target_usd": _f(target, 2), "price": _f(px, 2),
                     "shares": _f(target / px, 4), "vol": _f(vi, 4), "risk_share": _f(ci, 4),
                     "signal": (signals or {}).get(t), "current_usd": _f(current, 2), "delta_usd": _f(delta, 2),
                     "trade": "Buy" if delta > 1 else "Sell" if delta < -1 else "Hold"})
    others = {t: v for t, v in holdings.items() if t not in tickers and abs(v) > 1}
    cash_w = 1.0 - float(w.sum())
    return {
        "method": method, "method_label": ALLOCATION_LABELS[method], "budget": _f(budget, 2),
        "reserve_pct": reserve_pct, "max_position_pct": max_position_pct, "lookback": int(len(rets)),
        "as_of": rets.index[-1].strftime("%Y-%m-%d"), "rows": rows,
        "cash": {"weight": _f(cash_w, 4), "usd": _f(budget * cash_w, 2),
                 "note": "reserve" if cash_w <= reserve_pct / 100.0 + 1e-6 else "reserve + caps leave the rest in cash"},
        "outside_plan": [{"ticker": t, "current_usd": _f(v, 2)} for t, v in others.items()],
        "stats": {"ann_vol": _f(port_vol), "effective_n": _f(eff_n, 2), "invested": _f(float(w.sum()), 4)},
        "backtest": _allocation_backtest(close, tickers, cap, invest, lookback),
    }
