"""Performance metrics and statistical tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ANNUAL = 252


def metrics(returns: pd.Series, positions: pd.Series | None = None, benchmark: pd.Series | None = None) -> dict:
    r = pd.Series(returns, dtype=float).fillna(0.0)
    n = len(r)
    if n == 0:
        return {}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    sd = r.std(ddof=1)
    downside = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else np.nan
    ann_ret = float(equity.iloc[-1] ** (ANNUAL / n) - 1)
    out = {
        "cumulative_return": float(equity.iloc[-1] - 1),
        "annualized_return": ann_ret,
        "sharpe": float(r.mean() / sd * math.sqrt(ANNUAL)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * math.sqrt(ANNUAL)) if downside and downside > 0 else np.nan,
        "annualized_volatility": float(sd * math.sqrt(ANNUAL)),
        "daily_volatility": float(sd),
        "max_drawdown": float(dd.min()),
        "calmar": float(ann_ret / abs(dd.min())) if dd.min() < 0 else np.nan,
        "win_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else np.nan,
        "n_days": n,
    }
    if positions is not None:
        p = pd.Series(positions, dtype=float)
        out["exposure"] = float((p != 0).mean())
        out["long_share"] = float((p > 0).mean())
        out["turnover"] = float(p.diff().abs().fillna(p.abs()).mean())
    if benchmark is not None:
        b = pd.Series(benchmark, dtype=float).fillna(0.0)
        if b.std() > 0:
            beta = float(np.cov(r, b)[0, 1] / b.var())
            alpha = float((r.mean() - beta * b.mean()) * ANNUAL)
            out.update({"beta": beta, "alpha_annual": alpha})
    return out


def directional_accuracy(signal: pd.Series, realized: pd.Series) -> float:
    mask = (signal != 0) & (realized != 0)
    if not mask.any():
        return np.nan
    return float((np.sign(signal[mask]) == np.sign(realized[mask])).mean())


def information_coefficient(frame: pd.DataFrame, signal: str = "signal", target: str = "next_return") -> dict:
    """Mean daily cross-sectional Spearman IC plus the pooled time-series IC."""
    daily = []
    for _, g in frame.groupby("date"):
        if g[signal].nunique() > 2 and g[target].nunique() > 2:
            daily.append(stats.spearmanr(g[signal], g[target]).statistic)
    daily = np.array([d for d in daily if np.isfinite(d)])
    pooled = stats.spearmanr(frame[signal], frame[target])
    t_ic = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 2 and daily.std() > 0 else np.nan
    return {"ic_cross_sectional": float(daily.mean()) if len(daily) else np.nan, "ic_t": float(t_ic),
            "ic_pooled": float(pooled.statistic), "ic_pooled_p": float(pooled.pvalue)}


def paired_tests(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon signed-rank and paired t-test on aligned daily returns (a - b)."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = joined["a"] - joined["b"]
    nz = diff[diff != 0]
    out = {"n": int(len(joined)), "mean_diff_bps": float(diff.mean() * 1e4)}
    if len(nz) > 10:
        w = stats.wilcoxon(joined["a"], joined["b"], zero_method="wilcox", alternative="two-sided")
        out.update({"wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue)})
        w1 = stats.wilcoxon(joined["a"], joined["b"], zero_method="wilcox", alternative="greater")
        out["wilcoxon_p_greater"] = float(w1.pvalue)
        t = stats.ttest_rel(joined["a"], joined["b"])
        out.update({"t_stat": float(t.statistic), "t_p": float(t.pvalue)})
    return out


def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, n_boot: int = 5000, block: int = 10,
                                seed: int = 0) -> dict:
    """Stationary-style moving-block bootstrap CI for Sharpe(a) - Sharpe(b) on a common index."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().to_numpy()
    n = len(joined)
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return x.mean() / sd * math.sqrt(ANNUAL) if sd > 0 else 0.0

    observed = sharpe(joined[:, 0]) - sharpe(joined[:, 1])
    n_blocks = math.ceil(n / block)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([joined[s:s + block] for s in starts])[:n]
        diffs[i] = sharpe(sample[:, 0]) - sharpe(sample[:, 1])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"sharpe_diff": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((diffs <= 0).mean()), "samples": diffs}


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1) -> dict:
    d = (pd.Series(loss_a) - pd.Series(loss_b)).dropna().to_numpy()
    n = len(d)
    if n < 10 or d.std() == 0:
        return {"dm_stat": np.nan, "dm_p": np.nan}
    gamma0 = np.var(d, ddof=0)
    var = gamma0 + 2 * sum(np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h)) if h > 1 else gamma0
    stat = d.mean() / math.sqrt(var / n)
    return {"dm_stat": float(stat), "dm_p": float(2 * (1 - stats.norm.cdf(abs(stat))))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in pvalues.items() if p is not None and np.isfinite(p))
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted
