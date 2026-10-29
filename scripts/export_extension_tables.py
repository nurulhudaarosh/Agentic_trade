#!/usr/bin/env python3
"""LaTeX tables for the extension chapter, generated from saved results only.

Inputs: artifacts/study/rolling/{extensions_summary,report_extra}.json.
Outputs: paper/uiu_template/tables/ext_*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "artifacts" / "study" / "rolling"
OUT = ROOT / "paper" / "uiu_template" / "tables"
TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NVDA", "AAPL", "GOOGL", "META", "AMD"]
ZOO = ["Buy & Hold", "LGBM (price)", "LogReg (price)", "GRU (price)", "Transformer (price)", "Transformer (price+memory)",
       "RL Q-learning", "Pattern analogs (kNN)", "Ours", "FinMem"]
EXPL = ["LGBM + memory risk overlay", "Model vote (LGBM, Transformer, Ours)"]


def pct(v, d=1):
    return "--" if v is None else f"{v * 100:+.{d}f}\\%"


def vol(v):
    return "--" if v is None else f"{v * 100:.0f}\\%"


def num(v, d=2):
    return "--" if v is None else f"{v:+.{d}f}"


def esc(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def table(name: str, caption: str, label: str, cols: str, header: str, rows: list[str], small: bool = True) -> None:
    body = "\n".join(rows)
    # Wide tables are scaled to the text width so they never run into the margin.
    wide = len(cols) > 6
    open_, close_ = ("\\resizebox{\\textwidth}{!}{%\n", "}\n") if wide else ("", "")
    tex = (f"\\begin{{table}}[H]\n\\centering\n{'\\small' if small else ''}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"{open_}\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n{close_}\\end{{table}}\n")
    (OUT / f"{name}.tex").write_text(tex)


def main() -> None:
    s = json.loads((RD / "extensions_summary.json").read_text())
    x = json.loads((RD / "report_extra.json").read_text())
    P = s["portfolio"]

    rows = []
    for k in ZOO + EXPL:
        m = P[k]
        ic = s["ic"].get(k, {}).get("ic_pooled")
        mark = "$^{*}$" if k in EXPL else ""
        bold = k in ("Ours", "Transformer (price)", "RL Q-learning")
        name = f"\\textbf{{{esc(k)}}}" if bold else esc(k)
        rows.append(f"{name}{mark} & {pct(m['cumulative_return'])} & {num(m['sharpe'])} & {num(m.get('sortino'))} & "
                    f"{pct(m['max_drawdown'])} & {vol(m['annualized_volatility'])} & {num(m.get('beta'))} & "
                    f"{pct(m.get('alpha_annual'))} & {'--' if ic is None else f'{ic:+.3f}'} \\\\")
        if k == "FinMem":
            rows.append("\\midrule")
    table("ext_main", "Extension models on the primary rolling protocol (10 stocks, 227 test days, 5 bps costs). "
          "Beta and alpha are against SPY; IC is the pooled Spearman correlation of the model's signal with the next-day return. "
          "$^{*}$Exploratory combinations defined after the main results were known.",
          "tab:ext-main", "lrrrrrrrr", "Model & CR & Sharpe & Sortino & MDD & Vol & $\\beta$ & $\\alpha$/yr & IC", rows)

    rows = []
    for k in ["GRU (price)", "Transformer (price)", "Transformer (price+memory)", "RL Q-learning", "Pattern analogs (kNN)"] + EXPL:
        cells = []
        for ref in ("Buy & Hold", "LGBM (price)", "Ours"):
            t = s["tests"][f"{k} vs {ref}"]
            cells.append(f"{t['sharpe_diff']:+.2f} [{t['ci_low']:+.1f}, {t['ci_high']:+.1f}] & {t.get('wilcoxon_p', float('nan')):.2f}")
        rows.append(f"{esc(k)}{'$^{*}$' if k in EXPL else ''} & " + " & ".join(cells) + " \\\\")
    table("ext_tests", "Pairwise comparisons of the extension models: $\\Delta$Sharpe with 95\\% block-bootstrap interval and "
          "Wilcoxon signed-rank $p$ on daily portfolio returns, against buy-and-hold, the price-only LightGBM and our memory.",
          "tab:ext-tests", "lrrrrrr",
          "Model & \\multicolumn{2}{c}{vs Buy \\& Hold} & \\multicolumn{2}{c}{vs LGBM (price)} & \\multicolumn{2}{c}{vs Ours} \\\\\n"
          " & $\\Delta$Sharpe [CI] & $p$ & $\\Delta$Sharpe [CI] & $p$ & $\\Delta$Sharpe [CI] & $p$", rows)

    rows = []
    for k in ["Buy & Hold", "LGBM (price)", "GRU (price)", "Transformer (price)", "Transformer (price+memory)", "RL Q-learning",
              "Pattern analogs (kNN)", "Ours", "FinMem"]:
        pt = s["per_ticker"][k]
        rows.append(f"{esc(k.replace(' (price)', '').replace('Pattern analogs (kNN)', 'Analogs').replace('Transformer (price+memory)', 'Transf.+mem').replace('Transformer', 'Transf.'))} & "
                    + " & ".join(f"{pt[t]:+.2f}" for t in TICKERS) + " \\\\")
    table("ext_per_ticker", "Sharpe ratio per stock in the test period.", "tab:ext-per-ticker", "l" + "r" * 10,
          "Model & " + " & ".join(TICKERS), rows)

    r = x["robustness"]
    rows = [f"{esc(k)} & " + " & ".join(f"{r[c][k]:+.2f}" for c in ("10 stocks", "without TSLA", "FinMem's 5")) + " \\\\"
            for k in r["10 stocks"]]
    table("ext_robust", "Robustness to the stock set: Sharpe with all ten stocks, without Tesla, and on FinMem's five stocks.",
          "tab:ext-robust", "lrrr", "Model & 10 stocks & without TSLA & FinMem's 5", rows)

    c = x["capm"]
    rows = [f"{esc(k)} & {v['beta']:+.2f} & {v['alpha_annual'] * 100:+.1f}\\% & {v['alpha_t']:+.2f} \\\\" for k, v in c.items()]
    table("ext_capm", "CAPM decomposition of daily portfolio returns against SPY: market beta, annualised alpha and its $t$-statistic.",
          "tab:ext-capm", "lrrr", "Strategy & $\\beta$ & $\\alpha$ per year & $t(\\alpha)$", rows)

    g = x["regimes"]
    rows = [f"{esc(k)} & {v['up_bps']:+.1f} & {v['down_bps']:+.1f} \\\\" for k, v in g.items()]
    table("ext_regimes", "Average daily result (basis points) on days when the S\\&P 500 rose versus fell.", "tab:ext-regimes",
          "lrr", "Strategy & up days & down days", rows)

    a = x["allocation"]
    rows = [f"{esc(k)} & {pct(v['total_return'])} & {vol(v['ann_vol'])} & {num(v['sharpe'])} & {pct(v['max_drawdown'])} \\\\" for k, v in a.items()]
    table("ext_alloc", "Budget-split (portfolio construction) methods, walk-forward with monthly rebalancing, 80\\% invested, "
          "at most 25\\% of the budget per stock. SPY row: the same 80\\% invested in the S\\&P 500.",
          "tab:ext-alloc", "lrrrr", "Method & Return & Volatility & Sharpe & MDD", rows)

    k = x["stock_risk"]
    rows = [f"{t} & {k[t]['ann_vol'] * 100:.0f}\\% & {k[t]['max_drawdown'] * 100:.1f}\\% & {k[t]['var_95'] * 100:.1f}\\% & "
            f"{k[t]['cvar_95'] * 100:.1f}\\% & {k[t]['beta_spy']:.2f} & {k[t]['skew']:+.2f} & {k[t]['excess_kurtosis']:.2f} \\\\" for t in TICKERS]
    table("ext_stock_risk", "Risk profile of each stock over the full price history (Aug 2025 -- Oct 2026).", "tab:ext-stock-risk",
          "lrrrrrrr", "Stock & Vol/yr & MDD & VaR$_{95}$ & CVaR$_{95}$ & $\\beta$ & Skew & Ex.\\ kurt.", rows)

    pw = x["power"]
    (OUT / "ext_power_numbers.tex").write_text(
        f"% generated\n\\newcommand{{\\powerMean}}{{{pw['mean_bps']:+.1f}}}\n\\newcommand{{\\powerSd}}{{{pw['sd_bps']:.1f}}}\n"
        f"\\newcommand{{\\powerN}}{{{pw['n']}}}\n\\newcommand{{\\powerNeed}}{{{pw['days_needed']:,.0f}}}\n"
        f"\\newcommand{{\\powerYears}}{{{pw['days_needed'] / 252:.1f}}}\n\\newcommand{{\\powerMde}}{{{pw['mde_now_bps']:.1f}}}\n")
    print("tables written:", sorted(p.name for p in OUT.glob("ext_*.tex")))


if __name__ == "__main__":
    sys.exit(main())
