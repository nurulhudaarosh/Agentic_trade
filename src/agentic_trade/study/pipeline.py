"""End-to-end study: run memory systems once, then evaluate under two protocols.

``fixed``    train Oct-Feb, validation Mar-Apr (threshold / margin selection), test May-Oct.
``rolling``  walk-forward over Dec 2025 - Oct 2026: at the start of every month the
             decision threshold is re-selected on the trailing 42 trading days and the
             ML models are re-fitted on all earlier data. Primary protocol (2x the test
             days and no dependence on a single regime).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .annotate import annotate_dataset
from .backtest import StudyData, apply_costs, positions_from_signal, run_systems
from .data import FINMEM_TICKERS, UNIVERSE
from .evaluation import (block_bootstrap_sharpe_diff, diebold_mariano, directional_accuracy, holm,
                         information_coefficient, metrics, paired_tests)
from .memory import MemoryConfig
from .policies import (MEMORY_FEATURES, OURS_EXTRA, PRICE_FEATURES, fit_predict, positions_from_prob,
                       price_features, rule_baselines)

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
MARGINS = [0.0, 0.01, 0.02, 0.03, 0.05]
ROLLING_MARGIN = 0.02
ROLLING_START = date(2025, 12, 1)
TUNE_DAYS = 42
COST_BPS = 5.0
ABLATIONS = {
    "w/o credibility (N1)": "credibility", "w/o dedup (N2)": "dedup", "w/o contradiction (N3)": "contradiction",
    "w/o situational query (N4)": "situational", "w/o data importance (N5)": "importance",
    "w/o peer graph (N6)": "peers", "w/o episodic recall (N7)": "episodic",
}
MAIN = {"Plain RAG": MemoryConfig.plain_rag(), "FinMem": MemoryConfig.fin_mem(), "Ours": MemoryConfig.ours(),
        "Ours (signed N1)": MemoryConfig.ours(signed=True)}
DISPLAY_ORDER = ["Buy & Hold", "Momentum (1d)", "TS-Momentum (20d)", "Mean-Reversion (5d)", "MA-Crossover",
                 "Daily News Sentiment", "LogReg (price)", "LGBM (price)", "LGBM (price+FinMem)",
                 "LGBM (price+Ours)", "Plain RAG", "FinMem", "Ours", "Ours (signed N1)"]
EXCLUDED_FROM_TESTS = ("w/o", "FinMem K", "Ours K", "Ours (signed")


def system_configs() -> dict[str, MemoryConfig]:
    configs = dict(MAIN)
    for label, flag in ABLATIONS.items():
        configs[label] = MemoryConfig.ours(**{flag: False})
    for k in (1, 3, 10):
        configs[f"FinMem K={k}"] = MemoryConfig.fin_mem(top_k=k)
        configs[f"Ours K={k}"] = MemoryConfig.ours(top_k=k)
    return configs


def portfolio(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("date")[col].mean().sort_index()


def _sharpe_for(frame: pd.DataFrame, threshold: float) -> float:
    f = frame.copy()
    f["position"] = positions_from_signal(f, threshold)
    f["net"] = apply_costs(f, cost_bps=COST_BPS)
    return metrics(portfolio(f, "net"))["sharpe"]


def _best_threshold(frame: pd.DataFrame) -> tuple[float, dict]:
    scores = {th: _sharpe_for(frame, th) for th in THRESHOLDS}
    # Ties go to the larger threshold (the more conservative rule).
    return max(scores, key=lambda k: (round(scores[k], 6), k)), scores


def _month_starts(days: list[date]) -> list[date]:
    months = sorted({(d.year, d.month) for d in days if d >= ROLLING_START})
    return [min(d for d in days if (d.year, d.month) == m) for m in months]


def rolling_thresholds(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    days = sorted(frame["date"].unique())
    th = pd.Series(np.nan, index=frame.index)
    chosen = {}
    starts = _month_starts(days)
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else date.max
        hist_days = [d for d in days if d < start][-TUNE_DAYS:]
        best, _ = _best_threshold(frame[frame["date"].isin(hist_days)])
        chosen[str(start)] = best
        th[(frame["date"] >= start) & (frame["date"] < end)] = best
    first = chosen[str(starts[0])]
    return th.fillna(first), chosen


def _finish(f: pd.DataFrame, label: str) -> pd.DataFrame:
    f = f.copy()
    f["system"] = label
    f["net"] = apply_costs(f, cost_bps=COST_BPS)
    f["gross"] = f["position"] * f["next_return"]
    return f


def evaluate(signals: pd.DataFrame, base: pd.DataFrame, labels: list[str], protocol: str) -> tuple[pd.DataFrame, dict]:
    """Return per-(system, ticker, date) daily results with a protocol-specific ``split``."""
    rows, choices = [], {}
    for label in labels:
        frame = signals[signals["system"] == label].copy()
        if protocol == "fixed":
            th, scores = _best_threshold(frame[frame["split"] == "validation"])
            choices[label] = {"threshold": th, "validation_sharpe": scores}
        else:
            th, chosen = rolling_thresholds(frame)
            choices[label] = {"monthly_threshold": chosen}
        frame["position"] = positions_from_signal(frame, th)
        rows.append(_finish(frame, label))
    cols = ["ticker", "date", "split", "next_return", "next_spy"]
    for label, pos in rule_baselines(base).items():
        f = base[cols].copy()
        f["position"] = pos.to_numpy()
        f["signal"] = pos.to_numpy()
        rows.append(_finish(f, label))
    learned = {
        "LogReg (price)": (PRICE_FEATURES, "logreg"),
        "LGBM (price)": (PRICE_FEATURES, "lgbm"),
        "LGBM (price+FinMem)": (PRICE_FEATURES + [f"fm_{c}" for c in MEMORY_FEATURES], "lgbm"),
        "LGBM (price+Ours)": (PRICE_FEATURES + MEMORY_FEATURES + OURS_EXTRA, "lgbm"),
    }
    for label, (feats, kind) in learned.items():
        if protocol == "fixed":
            prob = pd.Series(np.mean([fit_predict(base, feats, kind, seed=s).to_numpy() for s in range(5)], axis=0), index=base.index)
            val = base["split"] == "validation"
            margins = {}
            for m in MARGINS:
                v = base[val].copy()
                v["position"] = positions_from_prob(prob[val], v, m)
                v["net"] = apply_costs(v, cost_bps=COST_BPS)
                margins[m] = metrics(portfolio(v, "net"))["sharpe"]
            margin = max(margins, key=margins.get)
            choices[label] = {"margin": margin, "validation_sharpe": margins}
            pos = positions_from_prob(prob, base, margin)
        else:
            prob = pd.Series(0.5, index=base.index)
            days = sorted(base["date"].unique())
            starts = _month_starts(days)
            for k, start in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else date.max
                fit_frame = base.copy()
                fit_frame["split"] = np.where((base["date"] < start) & (base["split"] != "warmup"), "train", "other")
                # Labels of the last pre-month day resolve on the first day of the month: drop it.
                last = max(d for d in days if d < start)
                fit_frame.loc[fit_frame["date"] == last, "split"] = "other"
                month = (base["date"] >= start) & (base["date"] < end)
                p = np.mean([fit_predict(fit_frame, feats, kind, seed=s).to_numpy() for s in range(3)], axis=0)
                prob[month] = p[month.to_numpy()]
            choices[label] = {"margin": ROLLING_MARGIN}
            pos = positions_from_prob(prob, base, ROLLING_MARGIN)
        f = base[cols].copy()
        f["position"] = pos.to_numpy()
        f["signal"] = prob.to_numpy() - 0.5
        rows.append(_finish(f, label))
    daily = pd.concat(rows, ignore_index=True)
    if protocol == "rolling":
        daily["split"] = np.where(daily["date"] >= ROLLING_START, "test", "pre")
    return daily, choices


def run(data_dir: Path, out_dir: Path, tickers: list[str] = UNIVERSE, reuse: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    annotate_dataset(data_dir)
    configs = system_configs()
    sig_path = out_dir / "signals.parquet"
    if reuse and sig_path.exists():
        signals = pd.read_parquet(sig_path)
        stats = json.loads((out_dir / "memory_stats.json").read_text())
    else:
        data = StudyData.load(data_dir, tickers)
        signals, evidence, stats = run_systems(data, list(configs.values()), list(configs), log_evidence=True)
        signals.to_parquet(sig_path, index=False)
        evidence[evidence["system"].isin(MAIN)].to_parquet(out_dir / "evidence.parquet", index=False)
        (out_dir / "memory_stats.json").write_text(json.dumps(stats, indent=2, default=float))
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    signals = signals[signals["split"] != "warmup"]

    bars = pd.read_parquet(data_dir / "prices.parquet")
    feats = price_features(bars, tickers)
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    base = signals[signals["system"] == "Ours"].merge(feats, on=["ticker", "date"], how="left").reset_index(drop=True)
    fin = signals[signals["system"] == "FinMem"].set_index(["ticker", "date"])
    base = base.join(fin[MEMORY_FEATURES].add_prefix("fm_"), on=["ticker", "date"])

    summaries = {}
    for protocol in ("rolling", "fixed"):
        pdir = out_dir / protocol
        pdir.mkdir(exist_ok=True)
        daily, choices = evaluate(signals, base, list(configs), protocol)
        daily.to_parquet(pdir / "daily.parquet", index=False)
        sig = signals.copy()
        if protocol == "rolling":
            sig["split"] = np.where(sig["date"] >= ROLLING_START, "test", "pre")
        summary = summarize(daily, sig)
        summary.update({"protocol": protocol, "choices": choices, "memory_stats": stats})
        (pdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        write_tables(summary, pdir)
        summaries[protocol] = summary
    return summaries


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return None
    return str(value)


def summarize(daily: pd.DataFrame, signals: pd.DataFrame) -> dict:
    test = daily[daily["split"] == "test"]
    out: dict = {"portfolio": {}, "per_ticker": {}, "ic": {}, "accuracy": {}, "tests": {}, "finmem5": {},
                 "test_days": int(test["date"].nunique()),
                 "test_range": [str(test["date"].min()), str(test["date"].max())]}
    spy = test.groupby("date")["next_spy"].first().sort_index()
    out["portfolio"]["SPY"] = metrics(spy)
    for label, g in test.groupby("system"):
        port = portfolio(g, "net")
        out["portfolio"][label] = metrics(port, portfolio(g, "position"), spy.loc[port.index])
        out["portfolio"][label]["gross_sharpe"] = metrics(portfolio(g, "gross"))["sharpe"]
        out["finmem5"][label] = metrics(portfolio(g[g["ticker"].isin(FINMEM_TICKERS)], "net"))
        out["per_ticker"][label] = {t: metrics(tg.sort_values("date")["net"], tg["position"])
                                    for t, tg in g.groupby("ticker")}
        out["accuracy"][label] = directional_accuracy(g["position"], g["next_return"])
    sig_test = signals[signals["split"] == "test"]
    for label, g in sig_test.groupby("system"):
        out["ic"][label] = information_coefficient(g.assign(ar=g["next_return"] - g["next_spy"]), target="ar")
    for focus in ("Ours", "Ours (signed N1)"):
        if focus not in test["system"].unique():
            continue
        mine = test[test["system"] == focus].set_index(["ticker", "date"])["net"]
        mine_port = portfolio(test[test["system"] == focus], "net")
        raw_p, res_all = {}, {}
        for label, g in test.groupby("system"):
            if label == focus or label.startswith(EXCLUDED_FROM_TESTS):
                continue
            other = g.set_index(["ticker", "date"])["net"]
            res = paired_tests(mine, other)
            boot = block_bootstrap_sharpe_diff(mine_port, portfolio(g, "net"))
            res.update({k: v for k, v in boot.items() if k != "samples"})
            res.update(diebold_mariano(-mine, -other.reindex(mine.index)))
            res_all[label] = res
            raw_p[label] = res.get("wilcoxon_p")
        for label, adj in holm(raw_p).items():
            res_all[label]["wilcoxon_p_holm"] = adj
        out["tests"][focus] = res_all
    return out


def _fmt(x: float, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.2f}" if pct else f"{x:.3f}"


def write_tables(summary: dict, out_dir: Path) -> None:
    head = ["| Strategy | CR (%) | Ann. Ret (%) | Sharpe | Sortino | Ann. Vol (%) | MDD (%) | Calmar | Win (%) | Turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def table(labels: list[str]) -> list[str]:
        lines = list(head)
        for label in labels:
            m = summary["portfolio"][label]
            lines.append(f"| {label} | {_fmt(m['cumulative_return'], True)} | {_fmt(m['annualized_return'], True)} | "
                         f"{_fmt(m['sharpe'])} | {_fmt(m['sortino'])} | {_fmt(m['annualized_volatility'], True)} | "
                         f"{_fmt(m['max_drawdown'], True)} | {_fmt(m['calmar'])} | {_fmt(m['win_rate'], True)} | "
                         f"{_fmt(m.get('turnover'))} |")
        return lines

    order = [s for s in DISPLAY_ORDER if s in summary["portfolio"]]
    (out_dir / "table_main.md").write_text("\n".join(table(order)) + "\n")
    extra = [s for s in summary["portfolio"] if s.startswith(("w/o", "FinMem K", "Ours K"))]
    (out_dir / "table_ablation.md").write_text("\n".join(table(["Ours"] + extra)) + "\n")
    rows = []
    for label in order:
        for t, m in summary["per_ticker"][label].items():
            rows.append({"strategy": label, "ticker": t, **{k: m[k] for k in ("cumulative_return", "sharpe", "max_drawdown", "annualized_volatility")}})
    pd.DataFrame(rows).to_csv(out_dir / "table_per_ticker.csv", index=False)
