#!/usr/bin/env python3
"""Figures fig21-fig32 for the report: extension models, RL, analogs, allocation and robustness.

Reads artifacts/study/rolling/{daily,extensions_daily}.parquet, extensions_logs.json and the
study price panel; writes PDF + PNG to paper/figures with the study's palette.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from agentic_trade import analytics  # noqa: E402
from agentic_trade.study.evaluation import metrics  # noqa: E402
from agentic_trade.study.figures import (AQUA, BLUE, DIV, GREEN, INK, INK2, MAGENTA, MUTED, ORANGE, RED,  # noqa: E402
                                         SEQ, VIOLET, YELLOW, _save)

OUT = ROOT / "paper" / "figures"
RD = ROOT / "artifacts" / "study" / "rolling"
TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NVDA", "AAPL", "GOOGL", "META", "AMD"]
COL = {"Buy & Hold": INK2, "LGBM (price)": MAGENTA, "Ours": BLUE, "FinMem": ORANGE, "Plain RAG": AQUA,
       "GRU (price)": YELLOW, "Transformer (price)": VIOLET, "Transformer (price+memory)": "#9b8cf0",
       "RL Q-learning": GREEN, "Pattern analogs (kNN)": RED, "LGBM + memory risk overlay": "#c43d74",
       "Model vote (LGBM, Transformer, Ours)": MUTED, "LogReg (price)": "#b0a8a0"}
ZOO = ["Buy & Hold", "LGBM (price)", "GRU (price)", "Transformer (price)", "Transformer (price+memory)",
       "RL Q-learning", "Pattern analogs (kNN)", "Ours", "FinMem"]


def load() -> pd.DataFrame:
    d = pd.read_parquet(RD / "daily.parquet")
    e = pd.read_parquet(RD / "extensions_daily.parquet")
    df = pd.concat([d, e], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["split"] == "test"]


def port(df: pd.DataFrame, s: str, tickers: list[str] | None = None) -> pd.Series:
    g = df[df["system"] == s]
    if tickers:
        g = g[g["ticker"].isin(tickers)]
    return g.groupby("date")["net"].mean().sort_index()


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return x.mean() / sd * math.sqrt(252) if sd > 0 else 0.0


def boot_sharpe(r: pd.Series, n: int = 3000, block: int = 10, seed: int = 0) -> tuple[float, float]:
    x = r.to_numpy()
    rng = np.random.default_rng(seed)
    nb = math.ceil(len(x) / block)
    vals = [sharpe(np.concatenate([x[s:s + block] for s in rng.integers(0, len(x) - block + 1, nb)])[:len(x)]) for _ in range(n)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


# --------------------------------------------------------------------------- figures
def fig21_zoo_equity(df):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s in ZOO:
        r = port(df, s)
        lw = 2.4 if s in ("Ours", "Transformer (price)", "RL Q-learning") else 1.5
        ax.plot(r.index, (1 + r).cumprod(), label=s, color=COL[s], lw=lw, ls="--" if s == "Buy & Hold" else "-")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_title("Model zoo: equal-weight portfolio of 10 stocks, test period (after 5 bps costs)")
    ax.set_ylabel("Growth of $1")
    ax.legend(ncol=3, fontsize=7.5, loc="upper left")
    _save(fig, OUT, "fig21_model_zoo_equity")


def fig22_zoo_sharpe(df):
    names = ZOO + ["LGBM + memory risk overlay", "Model vote (LGBM, Transformer, Ours)"]
    vals, los, his = [], [], []
    for s in names:
        r = port(df, s)
        vals.append(sharpe(r.to_numpy()))
        lo, hi = boot_sharpe(r)
        los.append(vals[-1] - lo)
        his.append(hi - vals[-1])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(names))
    ax.barh(y, np.array(vals)[order], xerr=[np.array(los)[order], np.array(his)[order]],
            color=[COL[names[i]] for i in order], alpha=0.9, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    ax.set_yticks(y, [names[i] + (" *" if names[i].startswith(("LGBM +", "Model vote")) else "") for i in order], fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Annualised Sharpe ratio, 95% block-bootstrap interval")
    ax.set_title("All models on the same protocol (* = exploratory, defined after the main results)")
    _save(fig, OUT, "fig22_model_zoo_sharpe")


def fig23_transformer(logs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.subplots_adjust(wspace=0.38)
    for name, color in (("GRU (price)", YELLOW), ("Transformer (price)", VIOLET), ("Transformer (price+memory)", "#9b8cf0")):
        tr, va = [], []
        for m in logs[name]:
            for s in m["seeds"]:
                tr.append(s["train"]); va.append(s["val"])
        L = max(len(x) for x in tr)
        pad = lambda arr: np.array([x + [np.nan] * (L - len(x)) for x in arr])
        axes[0].plot(np.arange(1, L + 1), np.nanmean(pad(tr), 0), color=color, lw=1.6, label=f"{name} train")
        axes[0].plot(np.arange(1, L + 1), np.nanmean(pad(va), 0), color=color, lw=1.2, ls="--", label=f"{name} val")
    axes[0].axhline(math.log(2), color=MUTED, lw=0.8, ls=":")
    axes[0].text(1, math.log(2) + 0.002, "coin flip (ln 2)", fontsize=7, color=MUTED)
    axes[0].set_title("(a) Loss per epoch (mean of refits)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].legend(fontsize=6)
    for name, color in (("Transformer (price)", VIOLET), ("Transformer (price+memory)", "#9b8cf0")):
        att = np.array([s["attention"] for m in logs[name] for s in m["seeds"] if "attention" in s])
        axes[1].plot(np.arange(-19, 1), att.mean(0), marker="o", ms=3, color=color, label=name)
    axes[1].axhline(1 / 20, color=MUTED, lw=0.8, ls=":")
    axes[1].set_title("(b) Attention over the 20-day window")
    axes[1].set_xlabel("day relative to decision (0 = today)"); axes[1].set_ylabel("mean attention weight")
    axes[1].legend(fontsize=7)
    months = [m["month"][:7] for m in logs["Transformer (price)"]]
    axes[2].bar(months, [m["n_train"] for m in logs["Transformer (price)"]], color=VIOLET, alpha=0.8)
    ax2 = axes[2].twinx()
    ax2.plot(months, [np.mean([s["best_epoch"] for s in m["seeds"]]) for m in logs["Transformer (price)"]], color=INK, marker="o", ms=3)
    ax2.set_ylabel("best epoch (early stopping)")
    ax2.grid(False)
    axes[2].set_title("(c) Training rows per monthly refit")
    axes[2].set_ylabel("training rows")
    axes[2].tick_params(axis="x", rotation=60, labelsize=7)
    _save(fig, OUT, "fig23_deep_training")


def fig24_rl(logs, df):
    rl = logs["RL Q-learning"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 1.1, 1.25]})
    fig.subplots_adjust(wspace=0.55)
    # Pre-training is shared by all ten agents (same data, same seed), so the curve is drawn once.
    lc = rl[TICKERS[0]]["learning_curve"]
    axes[0].plot(np.arange(1, len(lc) + 1), lc, color=GREEN, lw=1.6, marker="o", ms=2.5)
    axes[0].set_title("(a) Pre-training (shared by all stocks)")
    axes[0].set_xlabel("pass"); axes[0].set_ylabel("avg. reward per decision (%)")
    rl_s = [metrics(port(df, "RL Q-learning", [t]))["sharpe"] for t in TICKERS]
    bh_s = [metrics(port(df, "Buy & Hold", [t]))["sharpe"] for t in TICKERS]
    x = np.arange(len(TICKERS))
    axes[1].bar(x - 0.2, bh_s, 0.4, color=INK2, label="Buy & Hold")
    axes[1].bar(x + 0.2, rl_s, 0.4, color=GREEN, label="RL Q-learning")
    axes[1].set_xticks(x, TICKERS, rotation=60, fontsize=7)
    axes[1].axhline(0, color=INK, lw=0.6)
    axes[1].set_title("(b) Sharpe per stock, test period")
    axes[1].legend(fontsize=7)
    # Mean advantage of Long over Flat in each state (averaged over the ten agents).
    adv = np.zeros(36)
    for t in TICKERS:
        q = rl[t]["q_table"]
        adv += np.array([(r["q_long"] or 0) - (r["q_flat"] or 0) for r in q])
    adv /= len(TICKERS)
    grid = adv.reshape(3, 2, 3, 2)  # ret5, trend, rsi, position
    mat = grid.transpose(0, 2, 1, 3).reshape(9, 4)  # rows: ret5 x rsi, cols: trend x position
    lim = np.abs(mat).max()
    im = axes[2].imshow(mat, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    axes[2].set_yticks(range(9), [f"5d {a} · RSI {b}" for a in ("down", "flat", "up") for b in ("low", "mid", "high")], fontsize=6.5)
    axes[2].set_xticks(range(4), ["below MA\ncash", "below MA\nholding", "above MA\ncash", "above MA\nholding"], fontsize=6.5)
    axes[2].grid(False)
    axes[2].set_title("(c) Learned policy: Q(Long) − Q(Flat)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    _save(fig, OUT, "fig24_rl_agent")


def fig25_analog_sweep():
    windows, horizons = (10, 20, 40), (5, 10, 20)
    grid = np.zeros((3, 3))
    beats = np.zeros((3, 3))
    start = max(analytics.min_start(w, h) for w in windows for h in horizons)
    bh = []
    for t in TICKERS:
        for i, w in enumerate(windows):
            for j, h in enumerate(horizons):
                bt = analytics.analog_backtest(t, w, h, 8, "ticker", 0.01, False, start=start)
                s = bt["metrics"]["Analog strategy"]["sharpe"] or 0.0
                b = bt["metrics"]["Buy & Hold"]["sharpe"] or 0.0
                grid[i, j] += s / len(TICKERS)
                beats[i, j] += (s > b)
                if i == 0 and j == 0:
                    bh.append(b)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    lim = max(0.5, np.abs(grid).max())
    im = axes[0].imshow(grid, cmap=DIV, vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    axes[0].set_xticks(range(3), [f"{h}d" for h in horizons]); axes[0].set_yticks(range(3), [f"{w}-day" for w in windows])
    axes[0].set_xlabel("look-ahead"); axes[0].set_ylabel("pattern length"); axes[0].grid(False)
    axes[0].set_title(f"(a) Mean Sharpe over 10 stocks (B&H {np.mean(bh):+.2f})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im2 = axes[1].imshow(beats, cmap=SEQ, vmin=0, vmax=10)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{int(beats[i, j])}/10", ha="center", va="center", fontsize=9)
    axes[1].set_xticks(range(3), [f"{h}d" for h in horizons]); axes[1].set_yticks(range(3), [f"{w}-day" for w in windows])
    axes[1].grid(False)
    axes[1].set_title("(b) Stocks where it beat buy & hold")
    _save(fig, OUT, "fig25_pattern_analogs")
    return {"grid": grid.tolist(), "beats": beats.tolist(), "bh_mean": float(np.mean(bh)), "start": str(analytics.close_panel().index[start].date())}


def fig26_allocation():
    plan = analytics.allocation_plan(10000, 20, 25, TICKERS, "inverse_vol", 120)
    bt = plan["backtest"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = [BLUE, ORANGE, AQUA, INK2]
    for (name, eq), c in zip(bt["equity"].items(), colors):
        axes[0].plot(pd.to_datetime(bt["dates"]), eq, color=c, lw=1.8, ls="--" if "SPY" in name else "-", label=name)
    axes[0].axhline(1, color=MUTED, lw=0.8)
    axes[0].set_title("(a) Budget-split methods, walk-forward (80% invested, max 25%/stock)")
    axes[0].legend(fontsize=7)
    import matplotlib.dates as mdates
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    weights = {}
    for m in ("inverse_vol", "min_variance", "equal"):
        p = analytics.allocation_plan(10000, 20, 25, TICKERS, m, 120)
        weights[analytics.ALLOCATION_LABELS[m]] = [r["weight"] for r in p["rows"]]
    x = np.arange(len(TICKERS))
    for k, ((name, w), c) in enumerate(zip(weights.items(), colors)):
        axes[1].bar(x + (k - 1) * 0.27, w, 0.27, color=c, label=name)
    axes[1].set_xticks(x, TICKERS, rotation=60, fontsize=7)
    axes[1].set_title("(b) Current suggested weights")
    axes[1].legend(fontsize=6.5)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _save(fig, OUT, "fig26_allocation")
    return bt["metrics"]


def fig27_capm(df):
    spy = df[df["system"] == "Buy & Hold"].groupby("date")["next_spy"].mean().sort_index()
    names = ["Buy & Hold", "LGBM (price)", "LogReg (price)", "Ours", "FinMem", "Plain RAG", "GRU (price)", "Transformer (price)",
             "Transformer (price+memory)", "RL Q-learning", "Pattern analogs (kNN)", "Momentum (1d)", "MA-Crossover", "Mean-Reversion (5d)"]
    rows = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for s in names:
        r = port(df, s)
        m = metrics(r, benchmark=spy.reindex(r.index))
        x = sm_alpha_t(r, spy.reindex(r.index))
        rows[s] = {"beta": m["beta"], "alpha_annual": m["alpha_annual"], "alpha_t": x}
        ax.scatter(m["beta"], m["alpha_annual"] * 100, s=60, color=COL.get(s, MUTED), edgecolor="white", zorder=3)
        ax.annotate(s.replace(" (price)", "").replace("Pattern analogs (kNN)", "Analogs"), (m["beta"], m["alpha_annual"] * 100),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color=INK, lw=0.7); ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("market beta (vs SPY)"); ax.set_ylabel("annualised alpha (%)")
    ax.set_title("Risk-adjusted view: CAPM alpha and beta of each strategy")
    _save(fig, OUT, "fig27_capm_alpha_beta")
    return rows


def sm_alpha_t(r: pd.Series, b: pd.Series) -> float:
    x = np.column_stack([np.ones(len(b)), b.to_numpy()])
    y = r.to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = math.sqrt(s2 * np.linalg.inv(x.T @ x)[0, 0])
    return float(coef[0] / se)


def fig28_robustness(df):
    sys_ = ["Buy & Hold", "LGBM (price)", "Ours", "FinMem", "Plain RAG", "Transformer (price)", "RL Q-learning"]
    sets = {"10 stocks": TICKERS, "without TSLA": [t for t in TICKERS if t != "TSLA"],
            "FinMem's 5": ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(sys_))
    out = {}
    for k, (lab, tk) in enumerate(sets.items()):
        vals = [metrics(port(df, s, tk))["sharpe"] for s in sys_]
        out[lab] = dict(zip(sys_, vals))
        axes[0].bar(x + (k - 1) * 0.27, vals, 0.27, label=lab, color=[INK2, BLUE, ORANGE][k], alpha=0.85)
    axes[0].set_xticks(x, [s.replace(" (price)", "") for s in sys_], rotation=30, fontsize=7.5)
    axes[0].axhline(0, color=INK, lw=0.6)
    axes[0].set_title("(a) Sharpe on different stock sets")
    axes[0].legend(fontsize=7)
    loo = []
    for t in TICKERS:
        rest = [u for u in TICKERS if u != t]
        loo.append(metrics(port(df, "Ours", rest))["sharpe"] - metrics(port(df, "FinMem", rest))["sharpe"])
    full = metrics(port(df, "Ours"))["sharpe"] - metrics(port(df, "FinMem"))["sharpe"]
    axes[1].bar(TICKERS, loo, color=[BLUE if v > 0 else RED for v in loo])
    axes[1].axhline(full, color=INK, ls="--", lw=1)
    axes[1].text(9.4, full, " all 10", va="center", fontsize=7)
    axes[1].set_title("(b) Ours − FinMem Sharpe when one stock is left out")
    axes[1].tick_params(axis="x", rotation=60, labelsize=7)
    _save(fig, OUT, "fig28_robustness")
    out["leave_one_out"] = dict(zip(TICKERS, loo))
    out["full_diff"] = full
    return out


def fig29_power(df):
    d = (port(df, "Ours") - port(df, "FinMem")).dropna()
    mu, sd, n = d.mean(), d.std(ddof=1), len(d)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    need = (z * sd / mu) ** 2 if mu != 0 else np.inf
    ns = np.arange(50, 3001, 10)
    mde = z * sd / np.sqrt(ns) * 1e4
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ns, mde, color=BLUE, lw=2, label="smallest detectable difference (80% power, α=5%)")
    ax.axhline(mu * 1e4, color=ORANGE, ls="--", lw=1.2, label=f"observed Ours − FinMem: {mu * 1e4:+.1f} bps/day")
    ax.axvline(n, color=MUTED, ls=":", lw=1)
    ax.text(n + 20, mde.max() * 0.85, f"this study: {n} days", fontsize=7.5, color=INK2)
    if np.isfinite(need) and need < 3000:
        ax.axvline(need, color=RED, ls=":", lw=1)
        ax.text(need + 20, mde.max() * 0.65, f"needed: ≈{need:.0f} days (≈{need / 252:.1f} years)", fontsize=7.5, color=RED)
    elif np.isfinite(need):
        ax.text(ns[-1], mu * 1e4 + 2, f"to detect this edge: ≈{need:,.0f} days (≈{need / 252:.0f} years)", fontsize=8, color=RED, ha="right")
    ax.set_xlabel("number of test days"); ax.set_ylabel("basis points per day")
    ax.set_title("Statistical power: how much data a daily edge of this size needs")
    ax.legend(fontsize=7.5)
    _save(fig, OUT, "fig29_power")
    return {"mean_bps": float(mu * 1e4), "sd_bps": float(sd * 1e4), "n": int(n), "days_needed": float(need), "mde_now_bps": float(z * sd / np.sqrt(n) * 1e4)}


def fig30_regimes(df):
    spy = df[df["system"] == "Buy & Hold"].groupby("date")["next_spy"].mean().sort_index()
    up = spy > 0
    sys_ = ["Buy & Hold", "LGBM (price)", "Ours", "FinMem", "Plain RAG", "Transformer (price)", "RL Q-learning", "Pattern analogs (kNN)"]
    res = {}
    for s in sys_:
        r = port(df, s).reindex(spy.index)
        res[s] = {"up_bps": float(r[up].mean() * 1e4), "down_bps": float(r[~up].mean() * 1e4)}
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(sys_))
    ax.bar(x - 0.2, [res[s]["up_bps"] for s in sys_], 0.4, color=GREEN, label=f"market up days ({up.sum()})")
    ax.bar(x + 0.2, [res[s]["down_bps"] for s in sys_], 0.4, color=RED, label=f"market down days ({(~up).sum()})")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, [s.replace(" (price)", "").replace("Pattern analogs (kNN)", "Analogs") for s in sys_], rotation=25, fontsize=7.5)
    ax.set_ylabel("mean return, bps/day")
    ax.set_title("Market regimes: average daily result on S&P 500 up vs. down days")
    ax.legend(fontsize=7.5)
    _save(fig, OUT, "fig30_regimes")
    return res


def fig31_market():
    corr = analytics.correlation_matrix(0)
    tk = corr["tickers"]
    mat = np.array(corr["matrix"], dtype=float)
    risk = {t: analytics.risk_report(t)["stats"] for t in TICKERS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    im = axes[0].imshow(mat, cmap=DIV, vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(tk)), tk, rotation=60, fontsize=7); axes[0].set_yticks(range(len(tk)), tk, fontsize=7)
    for i in range(len(tk)):
        for j in range(len(tk)):
            if i != j:
                axes[0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    axes[0].grid(False)
    axes[0].set_title(f"(a) Correlation of daily returns ({corr['lookback']} days)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    x = np.arange(len(TICKERS))
    axes[1].bar(x - 0.2, [risk[t]["ann_vol"] for t in TICKERS], 0.4, color=BLUE, label="annualised volatility")
    axes[1].bar(x + 0.2, [-risk[t]["max_drawdown"] for t in TICKERS], 0.4, color=RED, label="max drawdown (abs.)")
    axes[1].plot(x, [risk[t]["var_95"] * 10 for t in TICKERS], "o", color=INK, ms=4, label="1-day VaR 95% (×10)")
    axes[1].set_xticks(x, TICKERS, rotation=60, fontsize=7)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_title("(b) Risk of each stock, Aug 2025 – Oct 2026")
    axes[1].legend(fontsize=7)
    _save(fig, OUT, "fig31_market_risk")
    return {t: {k: risk[t][k] for k in ("ann_vol", "max_drawdown", "var_95", "cvar_95", "beta_spy", "sharpe", "skew", "excess_kurtosis")} for t in TICKERS}


def fig32_monthly_zoo(df):
    sys_ = ["Buy & Hold", "LGBM (price)", "Transformer (price)", "RL Q-learning", "Ours", "FinMem"]
    mon = {}
    for s in sys_:
        r = port(df, s)
        mon[s] = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    mat = pd.DataFrame(mon).T
    fig, ax = plt.subplots(figsize=(10, 3.2))
    lim = np.nanmax(np.abs(mat.to_numpy()))
    im = ax.imshow(mat.to_numpy(), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iat[i, j] * 100:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(sys_)), sys_, fontsize=8)
    ax.set_xticks(range(mat.shape[1]), [str(c) for c in mat.columns], rotation=45, fontsize=7)
    ax.grid(False)
    ax.set_title("Monthly return (%) of the main models, test period")
    fig.colorbar(im, ax=ax, fraction=0.025)
    _save(fig, OUT, "fig32_monthly_models")


if __name__ == "__main__":
    df = load()
    logs = json.loads((RD / "extensions_logs.json").read_text())
    extra = {}
    fig21_zoo_equity(df); print("fig21")
    fig22_zoo_sharpe(df); print("fig22")
    fig23_transformer(logs); print("fig23")
    fig24_rl(logs, df); print("fig24")
    extra["analog_sweep"] = fig25_analog_sweep(); print("fig25")
    extra["allocation"] = fig26_allocation(); print("fig26")
    extra["capm"] = fig27_capm(df); print("fig27")
    extra["robustness"] = fig28_robustness(df); print("fig28")
    extra["power"] = fig29_power(df); print("fig29")
    extra["regimes"] = fig30_regimes(df); print("fig30")
    extra["stock_risk"] = fig31_market(); print("fig31")
    fig32_monthly_zoo(df); print("fig32")
    (RD / "report_extra.json").write_text(json.dumps(extra, indent=2, default=float))
    print(json.dumps({k: v for k, v in extra.items() if k in ("power", "robustness")}, indent=1, default=float))
