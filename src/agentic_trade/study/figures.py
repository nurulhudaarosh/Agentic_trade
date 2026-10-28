"""Publication figures for the study (PDF + PNG, one consistent palette)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .backtest import SPLITS
from .evaluation import block_bootstrap_sharpe_diff, metrics
from .memory import categorize

# Reference categorical palette (validated order): blue, orange, aqua, yellow, magenta, green, violet, red.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
COLORS = {"Ours": BLUE, "FinMem": ORANGE, "Plain RAG": AQUA, "Buy & Hold": INK2, "SPY": MUTED,
          "LGBM (price+Ours)": VIOLET, "LGBM (price+FinMem)": YELLOW, "LGBM (price)": MAGENTA}
MAIN3 = ["Plain RAG", "FinMem", "Ours"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc3ee", BLUE, "#123f78"])
DIV = LinearSegmentedColormap.from_list("div", [RED, "#f7d6d4", "#f2f1ee", "#cfe0f6", BLUE])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 110,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def _color(label: str) -> str:
    return COLORS.get(label, "#c3c2b7")


def _equity(daily: pd.DataFrame, label: str, tickers: list[str] | None = None) -> pd.Series:
    g = daily[(daily["system"] == label) & (daily["split"] == "test")]
    if tickers:
        g = g[g["ticker"].isin(tickers)]
    port = g.groupby("date")["net"].mean().sort_index()
    return (1 + port).cumprod()


def fig_data_overview(news: pd.DataFrame, prices: pd.DataFrame, tickers: list[str], out: Path) -> None:
    news = news.copy()
    news["month"] = pd.to_datetime(news["trade_date"]).dt.to_period("M")
    counts = pd.DataFrame({t: news[news["symbols"].str.split(",").map(lambda s: t in s)].groupby("month").size()
                           for t in tickers}).fillna(0).T
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 1.2, 1.2]})
    ax = axes[0]
    im = ax.imshow(counts.values, aspect="auto", cmap=SEQ)
    ax.set_yticks(range(len(tickers)), counts.index)
    ax.set_xticks(range(len(counts.columns)), [p.strftime("%b\n%y") for p in counts.columns], fontsize=7)
    ax.grid(False)
    ax.set_title("(a) Articles per ticker per month")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax = axes[1]
    cats = news["headline"].map(categorize).value_counts()
    ax.barh(cats.index[::-1], cats.values[::-1], color=BLUE, height=0.65)
    ax.set_title("(b) Headline category"); ax.set_xlabel("articles")
    ax = axes[2]
    ax.hist(news["sentiment"], bins=41, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_title("(c) FinBERT polarity (p+ − p−)"); ax.set_xlabel("polarity"); ax.set_ylabel("articles")
    fig.tight_layout()
    _save(fig, out, "fig01_data_overview")


def fig_timeline(prices: pd.DataFrame, filings: pd.DataFrame, out: Path) -> None:
    close = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    close.index = pd.to_datetime(close.index)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    norm = close / close.iloc[0] * 100
    for t in norm.columns:
        if t in ("SPY", "QQQ"):
            continue
        ax.plot(norm.index, norm[t], color="#c3c2b7", linewidth=0.8)
    ax.plot(norm.index, norm["SPY"], color=INK, linewidth=2, label="SPY")
    shades = {"warmup": "#f2f1ee", "train": "#e3eefb", "validation": "#fdebd9", "test": "#ddf3ea"}
    for name, (a, b) in SPLITS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=shades[name], zorder=0)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, ax.get_ylim()[1] * 0.97, name,
                ha="center", va="top", color=INK2, fontsize=8)
    for d in pd.to_datetime(filings["filed"]):
        if d >= norm.index[0]:
            ax.axvline(d, color=ORANGE, alpha=0.25, linewidth=0.8)
    ax.set_ylabel("price (start = 100)")
    ax.set_title("Study timeline: walk-forward splits, ten stocks (grey), SPY (black), SEC filings (orange ticks)")
    ax.set_xlim(norm.index[0], norm.index[-1])
    _save(fig, out, "fig02_timeline")


def fig_equity(daily: pd.DataFrame, out: Path, labels: list[str], name: str, title: str,
               tickers: list[str] | None = None) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    ends = []
    for label in labels:
        eq = _equity(daily, label, tickers)
        if eq.empty:
            continue
        lw = 2.4 if label == "Ours" else 1.5
        idx = pd.to_datetime(eq.index)
        ax.plot(idx, eq.values, color=_color(label), linewidth=lw, label=label, zorder=3 if label == "Ours" else 2)
        ends.append([float(eq.iloc[-1]), label, idx[-1]])
        dd = eq / eq.cummax() - 1
        ax2.plot(idx, dd.values * 100, color=_color(label), linewidth=1.2 if label != "Ours" else 2)
    # Direct end labels, nudged apart so close finishes stay legible.
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = []
    for y, label, x in ends:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(f"{label} {100 * (y - 1):+.1f}%", (x, y), xytext=(x + pd.Timedelta(days=3), yy), textcoords="data",
                    va="center", fontsize=7.5, color=INK2)
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_ylabel("growth of $1 (net of 5 bps)")
    ax.set_title(title)
    ax.legend(ncol=min(len(labels), 6), loc="upper left", fontsize=8)
    ax2.set_ylabel("drawdown (%)")
    _save(fig, out, name)


def fig_per_ticker(daily: pd.DataFrame, out: Path, tickers: list[str]) -> None:
    labels = ["Buy & Hold", "Plain RAG", "FinMem", "Ours"]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), sharex=True)
    for ax, t in zip(axes.flat, tickers):
        for label in labels:
            g = daily[(daily["system"] == label) & (daily["ticker"] == t) & (daily["split"] == "test")].sort_values("date")
            eq = (1 + g["net"]).cumprod()
            ax.plot(pd.to_datetime(g["date"]), eq.values, color=_color(label), linewidth=2 if label == "Ours" else 1.1, label=label)
        ax.set_title(t); ax.axhline(1, color=MUTED, linewidth=0.6)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Per-ticker cumulative return on the test window (growth of $1, net of costs)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig05_per_ticker")


def fig_heatmap(summary: dict, out: Path, labels: list[str], metric: str = "sharpe") -> None:
    tickers = list(next(iter(summary["per_ticker"].values())).keys())
    mat = np.array([[summary["per_ticker"][l][t][metric] for t in tickers] for l in labels])
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(labels) + 1.2))
    lim = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(tickers)), tickers); ax.set_yticks(range(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.grid(False)
    ax.set_title(f"Per-ticker {metric.replace('_', ' ')} on the test window")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    _save(fig, out, f"fig06_heatmap_{metric}")


def fig_bars(summary: dict, out: Path, labels: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (key, title, scale) in zip(axes, [("sharpe", "Sharpe ratio", 1), ("cumulative_return", "Cumulative return (%)", 100),
                                             ("max_drawdown", "Max drawdown (%)", 100)]):
        vals = [summary["portfolio"][l][key] * scale for l in labels]
        colors = [_color(l) if l in COLORS else "#c3c2b7" for l in labels]
        ax.barh(labels, vals, color=colors, height=0.66)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title)
    axes[0].invert_yaxis()
    fig.suptitle("Equal-weight ten-stock portfolio, test window (net of 5 bps)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig07_strategy_bars")


def fig_ablation(summary: dict, out: Path) -> None:
    full = summary["portfolio"]["Ours"]
    rows = [(l, summary["portfolio"][l]) for l in summary["portfolio"] if l.startswith("w/o")]
    rows.append(("FinMem (all removed)", summary["portfolio"]["FinMem"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, key, title, scale in [(axes[0], "sharpe", "Δ Sharpe vs. full model", 1),
                                  (axes[1], "cumulative_return", "Δ cumulative return (pp)", 100)]:
        deltas = [(m[key] - full[key]) * scale for _, m in rows]
        ax.barh([l for l, _ in rows], deltas, color=[RED if d < 0 else BLUE for d in deltas], height=0.6)
        for y, d in enumerate(deltas):
            ax.text(d, y, f" {d:+.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.8); ax.set_title(title)
        span = max(abs(min(deltas)), abs(max(deltas)))
        ax.set_xlim(min(deltas) - 0.3 * span, max(deltas) + 0.3 * span)
    axes[0].invert_yaxis()
    fig.suptitle(f"Component ablation (full model: Sharpe {full['sharpe']:.2f}, CR {100 * full['cumulative_return']:.1f}%)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig08_ablation")


def fig_topk(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for base, color in (("FinMem", ORANGE), ("Ours", BLUE)):
        ks = [1, 3, 5, 10]
        vals = [summary["portfolio"][base if k == 5 else f"{base} K={k}"]["sharpe"] for k in ks]
        ax.plot(ks, vals, marker="o", color=color, label=base, markersize=6)
    ax.set_xticks([1, 3, 5, 10]); ax.set_xlabel("top-K memories per layer"); ax.set_ylabel("test Sharpe")
    ax.set_title("Working-memory capacity"); ax.legend()
    _save(fig, out, "fig09_topk")


def fig_costs(daily: pd.DataFrame, out: Path, labels: list[str]) -> None:
    from .backtest import apply_costs

    costs = [0, 2, 5, 10, 15, 20, 30]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for label in labels:
        g = daily[(daily["system"] == label) & (daily["split"] == "test")].copy()
        vals = []
        for c in costs:
            g["net_c"] = apply_costs(g, cost_bps=c)
            vals.append(metrics(g.groupby("date")["net_c"].mean())["sharpe"])
        ax.plot(costs, vals, marker="o", markersize=5, color=_color(label), label=label, linewidth=2.2 if label == "Ours" else 1.4)
    ax.set_xlabel("one-way transaction cost (bps)"); ax.set_ylabel("test Sharpe"); ax.legend(fontsize=7)
    ax.set_title("Transaction-cost sensitivity")
    _save(fig, out, "fig10_costs")


def fig_rolling(daily: pd.DataFrame, out: Path, labels: list[str], window: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for label in labels:
        g = daily[(daily["system"] == label) & (daily["split"] == "test")]
        port = g.groupby("date")["net"].mean().sort_index()
        roll = port.rolling(window).mean() / port.rolling(window).std() * np.sqrt(252)
        ax.plot(pd.to_datetime(roll.index), roll.values, color=_color(label), label=label, linewidth=2.2 if label == "Ours" else 1.3)
    ax.axhline(0, color=MUTED, linewidth=0.8); ax.legend(ncol=len(labels), fontsize=8)
    ax.set_ylabel(f"{window}-day rolling Sharpe"); ax.set_title("Stability of risk-adjusted performance")
    _save(fig, out, "fig11_rolling_sharpe")


def fig_ic(summary: dict, out: Path) -> None:
    labels = [l for l in ["Plain RAG", "FinMem", "Ours"] + [l for l in summary["ic"] if l.startswith("w/o")] if l in summary["ic"]]
    vals = [summary["ic"][l]["ic_pooled"] for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(labels, vals, color=[_color(l) if l in MAIN3 else "#c3c2b7" for l in labels], height=0.6)
    for y, (l, v) in enumerate(zip(labels, vals)):
        p = summary["ic"][l]["ic_pooled_p"]
        ax.text(v, y, f" {v:+.3f} (p={p:.2f})", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8); ax.invert_yaxis()
    ax.set_xlabel("Spearman IC: memory signal vs. next-day market-adjusted return")
    ax.set_title("Predictive content of the retrieved memory (test window)")
    _save(fig, out, "fig12_information_coefficient")


def fig_retrieval(signals: pd.DataFrame, out: Path) -> None:
    s = signals[(signals["split"] == "test") & signals["system"].isin(MAIN3)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (col, title) in zip(axes[:3], [("redundancy", "Redundancy (mean pairwise cosine)"),
                                          ("mean_age", "Mean memory age (days)"),
                                          ("conflict", "Polarity conflict in context")]):
        data = [s.loc[s["system"] == l, col].dropna().to_numpy() for l in MAIN3]
        bp = ax.boxplot(data, tick_labels=MAIN3, showfliers=False, patch_artist=True, widths=0.55)
        for patch, l in zip(bp["boxes"], MAIN3):
            patch.set_facecolor(_color(l)); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_title(title)
    ax = axes[3]
    comp = s.groupby("system")[["n_shallow", "n_intermediate", "n_deep"]].mean().reindex(MAIN3)
    left = np.zeros(len(comp))
    for col, color, lab in [("n_shallow", BLUE, "news"), ("n_intermediate", ORANGE, "10-Q"), ("n_deep", AQUA, "10-K")]:
        ax.barh(MAIN3, comp[col], left=left, color=color, label=lab, height=0.6, edgecolor="white", linewidth=1)
        left += comp[col].to_numpy()
    ax.legend(fontsize=7, loc="lower right"); ax.set_title("Context composition (items)")
    fig.suptitle("Retrieval diagnostics on the test window", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig13_retrieval_diagnostics")


def fig_credibility(summary: dict, out: Path) -> None:
    table = pd.DataFrame(summary["memory_stats"]["_credibility"].get("Ours", []))
    if table.empty:
        return
    cats = table[table["level"] == "c"].sort_values("posterior")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(cats["name"], cats["posterior"], color=BLUE, height=0.6)
    for y, (_, r) in enumerate(cats.iterrows()):
        ax.text(r["posterior"], y, f"  {r['posterior']:.3f}  (n={int(r['n'])})", va="center", fontsize=7, color=INK2)
    g = table[table["level"] == "g"]["posterior"].iloc[0]
    ax.axvline(g, color=INK2, linestyle="--", linewidth=1, label=f"global {g:.3f}")
    ax.axvline(0.5, color=MUTED, linewidth=0.8)
    ax.set_xlim(min(0.45, cats["posterior"].min() - 0.01), max(0.56, cats["posterior"].max() + 0.04))
    ax.set_xlabel("posterior directional hit-rate (market-adjusted)")
    ax.set_title("Learned source credibility by category (end of test)"); ax.legend(fontsize=7)
    _save(fig, out, "fig14_credibility")


def fig_bootstrap(daily: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    ours = daily[(daily["system"] == "Ours") & (daily["split"] == "test")].groupby("date")["net"].mean()
    for ax, other in zip(axes, ["FinMem", "Plain RAG"]):
        o = daily[(daily["system"] == other) & (daily["split"] == "test")].groupby("date")["net"].mean()
        boot = block_bootstrap_sharpe_diff(ours, o)
        ax.hist(boot["samples"], bins=60, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.axvline(boot["sharpe_diff"], color=RED, linewidth=1.6, label=f"observed {boot['sharpe_diff']:+.2f}")
        ax.axvspan(boot["ci_low"], boot["ci_high"], color=BLUE, alpha=0.08, label=f"95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]")
        ax.set_title(f"Sharpe(Ours) − Sharpe({other})"); ax.legend(fontsize=7); ax.set_xlabel("difference")
    fig.suptitle("Moving-block bootstrap (block = 10 days, 5,000 resamples)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig15_bootstrap")


def fig_monthly(daily: pd.DataFrame, out: Path, labels: list[str]) -> None:
    rows = []
    for label in labels:
        g = daily[(daily["system"] == label) & (daily["split"] == "test")]
        port = g.groupby("date")["net"].mean()
        port.index = pd.to_datetime(port.index)
        rows.append(((1 + port).resample("ME").prod() - 1).rename(label))
    mat = pd.concat(rows, axis=1).T * 100
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.3))
    lim = np.nanmax(np.abs(mat.values))
    im = ax.imshow(mat.values, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [c.strftime("%b %y") for c in mat.columns]); ax.set_yticks(range(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.grid(False); ax.set_title("Monthly portfolio return (%) on the test window")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    _save(fig, out, "fig16_monthly")


def fig_positions(daily: pd.DataFrame, out: Path, labels: list[str]) -> None:
    t = daily[(daily["split"] == "test") & daily["system"].isin(labels)]
    share = t.groupby("system")["position"].agg(long=lambda p: (p > 0).mean(), short=lambda p: (p < 0).mean(),
                                                flat=lambda p: (p == 0).mean()).reindex(labels)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(labels) + 1))
    left = np.zeros(len(share))
    for col, color in (("long", BLUE), ("flat", "#d6d5d0"), ("short", ORANGE)):
        ax.barh(labels, share[col] * 100, left=left, color=color, label=col, height=0.62, edgecolor="white", linewidth=1)
        left += share[col].to_numpy() * 100
    ax.invert_yaxis(); ax.set_xlabel("share of ticker-days (%)"); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Position mix on the test window", pad=22)
    _save(fig, out, "fig17_positions")


def fig_memory_dynamics(signals: pd.DataFrame, summary: dict, out: Path) -> None:
    s = signals[signals["system"].isin(MAIN3)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for l in MAIN3:
        g = s[s["system"] == l].groupby("date")["active_items"].sum()
        ax.plot(pd.to_datetime(g.index), g.values, color=_color(l), label=l, linewidth=2 if l == "Ours" else 1.4)
    ax.set_yscale("log"); ax.set_ylabel("active memories (all tickers, log)"); ax.legend(fontsize=8)
    ax.set_title("Memory size over time")
    ax = axes[1]
    stats = summary["memory_stats"]
    keys = ["merged", "retired", "promoted", "purged"]
    x = np.arange(len(keys))
    for i, l in enumerate(["FinMem", "Ours"]):
        ax.bar(x + (i - 0.5) * 0.38, [stats[l][k] for k in keys], width=0.36, color=_color(l), label=l)
    ax.set_xticks(x, keys); ax.set_yscale("symlog"); ax.legend(fontsize=8)
    ax.set_title("Memory operations (count)")
    fig.tight_layout()
    _save(fig, out, "fig18_memory_dynamics")


def fig_peer_graph(news: pd.DataFrame, tickers: list[str], out: Path) -> None:
    counts: dict[tuple, int] = {}
    for syms in news["symbols"].fillna(""):
        s = sorted({x for x in syms.split(",") if x in tickers})
        if 1 < len(syms.split(",")) <= 6:
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    counts[(s[i], s[j])] = counts.get((s[i], s[j]), 0) + 1
    angles = np.linspace(0, 2 * np.pi, len(tickers), endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(tickers, angles)}
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    top = max(counts.values()) if counts else 1
    for (a, b), c in sorted(counts.items(), key=lambda kv: kv[1]):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=BLUE, alpha=0.15 + 0.8 * c / top,
                linewidth=0.5 + 6 * c / top, zorder=1)
    for t, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color="white", edgecolor=INK2, linewidth=1.2, zorder=2)
        ax.text(x, y, t, ha="center", va="center", fontsize=8, fontweight="bold", color=INK, zorder=3)
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_title("Co-mention graph learned from article symbols\n(edge width = co-mentions, ≤6-symbol articles)")
    _save(fig, out, "fig19_peer_graph")


def fig_llm(llm: pd.DataFrame, out: Path) -> None:
    if llm is None or llm.empty:
        return
    llm = llm.copy()
    llm["ret"] = llm["position"] * llm["next_return"]
    rows = []
    for (bb, sysname), g in llm.groupby(["backbone", "system"]):
        m = metrics(g.groupby("date")["ret"].mean().sort_index())
        rows.append({"backbone": bb, "system": sysname, "sharpe": m["sharpe"], "cr": m["cumulative_return"],
                     "mdd": m["max_drawdown"]})
    tab = pd.DataFrame(rows)
    bh = metrics(llm[(llm["backbone"] == llm["backbone"].iloc[0]) & (llm["system"] == "FinMem")]
                 .groupby("date")["next_return"].mean().sort_index())
    systems = ["No-Memory", "FinMem", "Ours"]
    colors = {"No-Memory": "#c3c2b7", "FinMem": ORANGE, "Ours": BLUE}
    bbs = sorted(tab["backbone"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    panels = [(axes[0], "sharpe", "Sharpe ratio", 1, bh["sharpe"]), (axes[1], "cr", "Cumulative return (%)", 100, 100 * bh["cumulative_return"]),
              (axes[2], "mdd", "Max drawdown (%)", 100, 100 * bh["max_drawdown"])]
    for ax, key, title, scale, ref in panels:
        x = np.arange(len(bbs))
        for i, s in enumerate(systems):
            vals = [tab[(tab.backbone == b) & (tab.system == s)][key].mean() * scale for b in bbs]
            ax.bar(x + (i - 1) * 0.27, vals, width=0.25, color=colors[s], label=s)
        ax.axhline(ref, color=INK2, linestyle="--", linewidth=1, label="Buy & Hold")
        ax.set_xticks(x, bbs); ax.axhline(0, color=INK2, linewidth=0.8); ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("LLM backbones deciding from the same retrieved memory (FinMem's 5 stocks, last 50 sessions)", fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig20_llm_backbones")


def render_all(data_dir: Path, study_dir: Path, out: Path, protocol: str = "rolling") -> None:
    from .pipeline import ROLLING_START

    news = pd.read_parquet(data_dir / "news_annotated.parquet")
    prices = pd.read_parquet(data_dir / "prices.parquet")
    filings = pd.read_parquet(data_dir / "filings.parquet")
    daily = pd.read_parquet(study_dir / protocol / "daily.parquet")
    signals = pd.read_parquet(study_dir / "signals.parquet")
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    if protocol == "rolling":
        signals["split"] = np.where(signals["date"] >= ROLLING_START, "test", "pre")
    summary = json.loads((study_dir / protocol / "summary.json").read_text())
    tickers = sorted(daily["ticker"].unique(), key=["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NVDA", "AAPL", "GOOGL", "META", "AMD"].index)
    span = f"{summary['test_range'][0]} to {summary['test_range'][1]}, {summary['test_days']} days"
    headline = ["Buy & Hold", "LGBM (price)", "Plain RAG", "FinMem", "Ours"]
    fig_data_overview(news, prices, tickers, out)
    fig_timeline(prices, filings, out)
    fig_equity(daily, out, headline, "fig03_equity_portfolio", f"Ten-stock equal-weight portfolio ({protocol} protocol, {span})")
    fig_equity(daily, out, ["Buy & Hold", "Plain RAG", "FinMem", "Ours"], "fig04_equity_finmem5",
               "FinMem's five stocks (TSLA, NFLX, AMZN, MSFT, COIN)", ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"])
    fig_per_ticker(daily, out, tickers)
    order = [l for l in ["Buy & Hold", "Momentum (1d)", "TS-Momentum (20d)", "Mean-Reversion (5d)", "MA-Crossover",
                         "Daily News Sentiment", "LogReg (price)", "LGBM (price)", "LGBM (price+FinMem)",
                         "LGBM (price+Ours)", "Plain RAG", "FinMem", "Ours"] if l in summary["portfolio"]]
    fig_heatmap(summary, out, order, "sharpe")
    fig_heatmap(summary, out, order, "cumulative_return")
    fig_bars(summary, out, order)
    fig_ablation(summary, out)
    fig_topk(summary, out)
    fig_costs(daily, out, ["Buy & Hold", "Plain RAG", "FinMem", "Ours", "LGBM (price+Ours)"])
    fig_rolling(daily, out, ["Buy & Hold", "FinMem", "Ours"])
    fig_ic(summary, out)
    fig_retrieval(signals, out)
    fig_credibility(summary, out)
    fig_bootstrap(daily, out)
    fig_monthly(daily, out, ["Buy & Hold", "Plain RAG", "FinMem", "Ours"])
    fig_positions(daily, out, ["Plain RAG", "FinMem", "Ours", "LGBM (price+Ours)"])
    fig_memory_dynamics(signals, summary, out)
    fig_peer_graph(news, tickers, out)
    llm_path = study_dir / "llm" / "llm_decisions.parquet"
    if llm_path.exists():
        fig_llm(pd.read_parquet(llm_path), out)
