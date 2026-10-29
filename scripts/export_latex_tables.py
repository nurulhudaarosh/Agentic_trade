#!/usr/bin/env python3
"""Generate LaTeX tables for the report directly from study artifacts (no hand-copied numbers)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ORDER = ["Buy & Hold", "Momentum (1d)", "TS-Momentum (20d)", "Mean-Reversion (5d)", "MA-Crossover",
         "Daily News Sentiment", "LogReg (price)", "LGBM (price)", "LGBM (price+FinMem)", "LGBM (price+Ours)",
         "Plain RAG", "FinMem", "Ours"]
MEMORY = ["Plain RAG", "FinMem", "Ours"]


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def f(x, pct=False, d=2, sign=False):
    if x is None or x != x:
        return "--"
    v = 100 * x if pct else x
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def bold_best(rows: list[list[str]], col: int, higher: bool = True) -> None:
    vals = [(float(r[col]), i) for i, r in enumerate(rows) if r[col] != "--"]
    best = (max if higher else min)(vals)[1]
    rows[best][col] = r"\textbf{" + rows[best][col] + "}"


def main_table(s: dict, caption: str, label: str) -> str:
    rows = []
    for name in [n for n in ORDER if n in s["portfolio"]]:
        m = s["portfolio"][name]
        rows.append([tex(name), f(m["cumulative_return"], True), f(m["sharpe"]), f(m["sortino"]),
                     f(m["annualized_volatility"], True), f(m["max_drawdown"], True), f(m["win_rate"], True, 1),
                     f(m["turnover"])])
    for col, hi in ((1, True), (2, True), (3, True), (5, True)):
        bold_best(rows, col, hi)
    body = "\n".join(" & ".join(r) + r" \\" + (r" \hline" if r[0] in ("Daily News Sentiment", "LGBM (price+Ours)") else "") for r in rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\begin{{tabular}}{{lrrrrrrr}}
\hline
Strategy & CR (\%) & Sharpe & Sortino & Ann. Vol (\%) & MDD (\%) & Win (\%) & Turnover \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def finmem5_table(s: dict) -> str:
    rows = []
    for name in ["Buy & Hold", "Plain RAG", "FinMem", "Ours"]:
        m = s["finmem5"][name]
        rows.append([tex(name), f(m["cumulative_return"], True), f(m["sharpe"]), f(m["max_drawdown"], True),
                     f(m["annualized_volatility"], True)])
    for col in (1, 2, 3):
        bold_best(rows, col)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Equal-weight portfolio of FinMem's five stocks (TSLA, NFLX, AMZN, MSFT, COIN), rolling protocol, net of 5\,bps.}}
\label{{tab:finmem5}}
\begin{{tabular}}{{lrrrr}}
\hline
System & CR (\%) & Sharpe & MDD (\%) & Ann. Vol (\%) \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def per_ticker_table(s: dict) -> str:
    tickers = list(s["per_ticker"]["Ours"])
    names = ["Buy & Hold", "LGBM (price)", "Plain RAG", "FinMem", "Ours"]
    lines = []
    for n in names:
        lines.append(tex(n) + " & " + " & ".join(f(s["per_ticker"][n][t]["sharpe"]) for t in tickers) + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Per-ticker Sharpe ratio, rolling protocol (net of costs).}}
\label{{tab:perticker}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l{'r' * len(tickers)}}}
\hline
Strategy & {' & '.join(tickers)} \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def tests_table(s: dict) -> str:
    rows = sorted(s["tests"]["Ours"].items(), key=lambda kv: kv[1].get("wilcoxon_p", 1))
    lines = []
    for name, v in rows:
        lines.append(f"{tex(name)} & {f(v['mean_diff_bps'], d=1, sign=True)} & {f(v['sharpe_diff'], sign=True)} & "
                     f"[{f(v['ci_low'])}, {f(v['ci_high'])}] & {f(v['wilcoxon_p'], d=3)} & {f(v.get('wilcoxon_p_holm'), d=3)} & {f(v['dm_p'], d=3)}" + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Ours versus every baseline on pooled (ticker, date) daily returns, rolling protocol. $\Delta$ = Ours $-$ baseline; CI from a 10-day moving-block bootstrap (5{{,}}000 resamples); Holm-adjusted Wilcoxon $p$; DM = Diebold--Mariano.}}
\label{{tab:tests}}
\small
\begin{{tabular}}{{lrrcrrr}}
\hline
Baseline & $\Delta$ bps/day & $\Delta$Sharpe & 95\% CI & Wilcoxon $p$ & Holm $p$ & DM $p$ \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def ablation_table(s: dict) -> str:
    full = s["portfolio"]["Ours"]
    names = ["Ours"] + [n for n in s["portfolio"] if n.startswith("w/o")] + ["FinMem"]
    lines = []
    for n in names:
        m = s["portfolio"][n]
        d = "" if n == "Ours" else f(m["sharpe"] - full["sharpe"], sign=True)
        lines.append(f"{tex(n)} & {f(m['cumulative_return'], True)} & {f(m['sharpe'])} & {d} & {f(m['max_drawdown'], True)} & {f(s['ic'][n]['ic_pooled'], d=3, sign=True)}" + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Component ablation of our memory (rolling protocol). Each row removes one component; ``FinMem'' removes all of them.}}
\label{{tab:ablation}}
\begin{{tabular}}{{lrrrrr}}
\hline
Variant & CR (\%) & Sharpe & $\Delta$Sharpe & MDD (\%) & IC \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def topk_table(s: dict) -> str:
    lines = []
    for base in ("FinMem", "Ours"):
        vals = [s["portfolio"][base if k == 5 else f"{base} K={k}"]["sharpe"] for k in (1, 3, 5, 10)]
        lines.append(base + " & " + " & ".join(f(v) for v in vals) + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Sensitivity to working-memory capacity (top-$K$ per layer), Sharpe ratio, rolling protocol.}}
\label{{tab:topk}}
\begin{{tabular}}{{lrrrr}}
\hline
System & $K=1$ & $K=3$ & $K=5$ & $K=10$ \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def ic_ops_table(s: dict) -> str:
    lines = []
    for n in MEMORY:
        ic, ops = s["ic"][n], s["memory_stats"][n]
        lines.append(f"{n} & {f(ic['ic_pooled'], d=3, sign=True)} & {f(ic['ic_pooled_p'], d=2)} & {f(ic['ic_cross_sectional'], d=3, sign=True)} & "
                     f"{ops['items']:,} & {ops['merged']:,} & {ops['retired']:,} & {ops['promoted']:,} & {ops['purged']:,}" + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Predictive content of each memory signal (Spearman IC vs.\ next-day market-adjusted return, rolling test period) and memory operations over the full run (ten tickers).}}
\label{{tab:icops}}
\small
\begin{{tabular}}{{lrrrrrrrr}}
\hline
System & IC & $p$ & IC (cross-sect.) & Items & Merged & Retired & Promoted & Purged \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def data_table(data_dir: Path) -> str:
    news = pd.read_parquet(data_dir / "news_annotated.parquet", columns=["symbols", "trade_date"])
    filings = pd.read_parquet(data_dir / "filings_annotated.parquet", columns=["ticker", "form"])
    tickers = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NVDA", "AAPL", "GOOGL", "META", "AMD"]
    lines = []
    for t in tickers:
        n = int(news["symbols"].fillna("").str.split(",").map(lambda s: t in s).sum())
        q = int(((filings["ticker"] == t) & (filings["form"] == "10-Q")).sum())
        k = int(((filings["ticker"] == t) & (filings["form"] == "10-K")).sum())
        lines.append(f"{t} & {n:,} & {q} & {k}" + r" \\")
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Dataset composition: articles tagged with each ticker (an article can mention several), and SEC filings since July 2024.}}
\label{{tab:data}}
\begin{{tabular}}{{lrrr}}
\hline
Ticker & News articles & 10-Q & 10-K \\
\hline
{chr(10).join(lines)}
\hline
Unique total & {len(news):,} & {int((filings['form'] == '10-Q').sum())} & {int((filings['form'] == '10-K').sum())} \\
\hline
\end{{tabular}}
\end{{table}}
"""


def llm_table(path: Path) -> str:
    if not path.exists():
        return r"\begin{table}[ht]\centering\caption{LLM backbone comparison (pending).}\label{tab:llm}\begin{tabular}{l}\hline Experiment still running.\\\hline\end{tabular}\end{table}" + "\n"
    frame = pd.read_parquet(path)
    frame["ret"] = frame["position"] * frame["next_return"]
    lines, counts = [], set()
    for bb in sorted(frame["backbone"].unique()):
        for sysname in ("No-Memory", "FinMem", "Ours"):
            g = frame[(frame["backbone"] == bb) & (frame["system"] == sysname)]
            if g.empty:
                continue
            port = g.groupby("date")["ret"].mean().sort_index()
            sd = port.std(ddof=1)
            sharpe = port.mean() / sd * 252 ** 0.5 if sd > 0 else 0.0
            cr = (1 + port).prod() - 1
            mix = "/".join(f"{100 * (g['action'] == a).mean():.0f}" for a in ("Buy", "Sell", "Hold"))
            eq = (1 + port).cumprod()
            mdd = float((eq / eq.cummax() - 1).min())
            counts.add(len(g))
            lines.append(f"{bb} & {sysname} & {f(cr, True)} & {f(sharpe)} & {f(mdd, True)} & {mix}" + r" \\")
        lines.append(r"\hline")
    from scipy.stats import wilcoxon

    piv = frame.pivot_table(index=["backbone", "ticker", "date"], columns="system", values="ret")
    tests = []
    for other in ("FinMem", "No-Memory"):
        res = wilcoxon(piv["Ours"], piv[other])
        tests.append(f"Ours vs {other}: {1e4 * (piv['Ours'] - piv[other]).mean():+.1f} bps/day, Wilcoxon $p={res.pvalue:.3f}$")
    bh = frame[(frame["backbone"] == frame["backbone"].iloc[0]) & (frame["system"] == "FinMem")].groupby("date")["next_return"].mean()
    bh_cr = (1 + bh).prod() - 1
    note = r"\multicolumn{6}{p{0.9\textwidth}}{\footnotesize "
    lines.append(note + "Pooled over backbones: " + "; ".join(tests) + r".} \\")
    lines.append(note + r"Buy \& Hold over the same window: CR " + f(bh_cr, True) + r"\%.} \\")
    per_row = f", {counts.pop()} decisions per row" if len(counts) == 1 else ""
    return rf"""\begin{{table}}[ht]
\centering
\caption{{LLM backbones deciding from the same retrieved memory (FinMem-style prompt, temperature 0, single trial), FinMem's five stocks, last 50 test sessions{per_row}.}}
\label{{tab:llm}}
\small
\begin{{tabular}}{{llrrrc}}
\hline
Backbone & Memory & CR (\%) & Sharpe & MDD (\%) & Buy/Sell/Hold (\%) \\
\hline
{chr(10).join(lines)}
\hline
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("artifacts/study"))
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--out", type=Path, default=Path("paper/uiu_template/tables"))
    parser.add_argument("--no-llm", action="store_true", help="emit a placeholder while the LLM run is in progress")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rolling = json.loads((args.study / "rolling" / "summary.json").read_text())
    fixed = json.loads((args.study / "fixed" / "summary.json").read_text())
    outputs = {
        "main_rolling": main_table(rolling, f"Ten-stock equal-weight portfolio, rolling walk-forward protocol ({rolling['test_range'][0]} to {rolling['test_range'][1]}, {rolling['test_days']} sessions), net of 5\\,bps. Best value per column in bold.", "tab:main"),
        "main_fixed": main_table(fixed, f"Ten-stock portfolio, fixed-split protocol (test {fixed['test_range'][0]} to {fixed['test_range'][1]}, {fixed['test_days']} sessions).", "tab:fixed"),
        "finmem5": finmem5_table(rolling),
        "per_ticker": per_ticker_table(rolling),
        "tests": tests_table(rolling),
        "ablation": ablation_table(rolling),
        "topk": topk_table(rolling),
        "ic_ops": ic_ops_table(rolling),
        "data": data_table(args.data),
        "llm": llm_table(Path("/nonexistent") if args.no_llm else args.study / "llm" / "llm_decisions.parquet"),
    }
    for name, body in outputs.items():
        (args.out / f"{name}.tex").write_text(body)
    print(f"wrote {len(outputs)} tables to {args.out}")


if __name__ == "__main__":
    main()
