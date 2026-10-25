from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .engine import ExperimentConfig, ExperimentEngine
from .case_studies import build_case_cards

SYSTEMS = ["no_memory", "plain_rag", "fin_mem", "ours"]
TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]


def run_one(base: ExperimentConfig, ticker: str, trial: int = 0, **flags: object) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    output = base.output_dir / f"{ticker.lower()}_trial_{trial}"
    config = replace(base, ticker=ticker, trial_id=trial, output_dir=output, **flags)
    engine = ExperimentEngine(config)
    metrics = engine.run()
    engine.write_artifacts(metrics)
    rows = {name: [dict(row, ticker=ticker, system=name) for row in values] for name, values in engine.results.items()}
    daily = pd.DataFrame([row for values in rows.values() for row in values])
    evidence = pd.DataFrame(engine.evidence)
    return metrics, daily, evidence


def paired_wilcoxon(results: pd.DataFrame, metric: str = "daily_return") -> dict:
    pivot = results.pivot_table(index=["ticker", "trial", "date"], columns="system", values=metric, aggfunc="first")
    if "ours" not in pivot or len(pivot) < 2:
        return {"p_value": None, "statistic": None, "significant": False}
    alternatives = [name for name in SYSTEMS if name != "ours" and name in pivot]
    second = max(alternatives, key=lambda name: float(pivot[name].mean())) if alternatives else None
    if second is None:
        return {"p_value": None, "statistic": None, "significant": False}
    paired = pivot[["ours", second]].dropna()
    if len(paired) < 2 or np.allclose(paired["ours"], paired[second]):
        return {"comparison": second, "p_value": 1.0, "statistic": 0.0, "significant": False}
    statistic, p_value = wilcoxon(paired["ours"], paired[second])
    return {"comparison": second, "p_value": float(p_value), "statistic": float(statistic), "significant": bool(p_value < 0.05)}


def markdown_table(summary: pd.DataFrame, output: Path) -> None:
    rows = ["| Ticker | System | Return | Sharpe | Volatility | Drawdown | Hold |", "|---|---|---:|---:|---:|---:|---:|"]
    best = summary.groupby("ticker")["cumulative_return"].transform("max")
    for _, row in summary.iterrows():
        star = "*" if row.get("significant", False) else ""
        bold = row["cumulative_return"] == best.loc[row.name]
        value = f"**{row['cumulative_return']:.4f}**" if bold else f"{row['cumulative_return']:.4f}"
        rows.append(f"| {row['ticker']} | {row['system']} | {value}{star} | {row['sharpe_ratio']:.2f} | {row['annualized_volatility']:.4f} | {row['max_drawdown']:.4f} | {row['hold_rate']:.2f} |")
    output.write_text("\n".join(rows) + "\n")


def plot_curves(daily: pd.DataFrame, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for system, frame in daily.groupby("system"):
        frame = frame.sort_values("date")
        ax.plot((1 + frame["daily_return"].astype(float)).cumprod().values, label=system)
    ax.set(title=title, xlabel="Test observation", ylabel="Growth of $1")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def run_matrix(base: ExperimentConfig, tickers: list[str] = TICKERS, trials: int = 3) -> dict[str, object]:
    records: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        for trial in range(trials):
            metrics, daily, evidence = run_one(base, ticker, trial, temperature=0.7 + trial * 0.01)
            for system, values in metrics.items():
                records.append({"ticker": ticker, "trial": trial, "system": system, **values})
            daily["trial"] = trial; daily_frames.append(daily)
            evidence["trial"] = trial; evidence_frames.append(evidence)
    output = base.output_dir / "matrix"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "metrics_long.csv", index=False)
    daily_all = pd.concat(daily_frames, ignore_index=True); daily_all.to_csv(output / "daily_long.csv", index=False)
    evidence_all = pd.concat(evidence_frames, ignore_index=True); evidence_all.to_csv(output / "evidence_log.csv", index=False)
    build_case_cards(daily_all[daily_all["ticker"].eq("TSLA")], evidence_all[evidence_all["ticker"].eq("TSLA")], output)
    significance = paired_wilcoxon(daily_all)
    summary["significant"] = (summary["system"].eq("ours") & significance.get("significant", False))
    summary.to_csv(output / "table2_metrics.csv", index=False)
    markdown_table(summary, output / "table2.md")
    (output / "wilcoxon.json").write_text(json.dumps(significance, indent=2))
    plot_curves(daily_all[daily_all["ticker"].eq("TSLA")], output / "fig5_tsla_curves.png", "TSLA cumulative return")
    plot_curves(daily_all, output / "fig6_all_tickers_curves.png", "All-ticker cumulative return")
    return {"summary": summary, "daily": daily_all, "evidence": evidence_all, "wilcoxon": significance}


def run_sweep(base: ExperimentConfig, name: str, values: list[object], flag: str) -> pd.DataFrame:
    records = []
    for value in values:
        metrics, daily, _ = run_one(base, base.ticker, 0, **{flag: value})
        for system, result in metrics.items():
            records.append({"setting": value, "system": system, **result})
    frame = pd.DataFrame(records)
    out = base.output_dir / "sweeps"; out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.csv", index=False)
    return frame


def run_risk_profiles(base: ExperimentConfig, profiles: list[str] | None = None) -> pd.DataFrame:
    """Run every risk profile on identical inputs and export Table 4 artifacts."""
    profiles = profiles or ["conservative", "standard", "aggressive"]
    records: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    for profile in profiles:
        metrics, daily, _ = run_one(base, base.ticker, base.trial_id, risk_mode=profile)
        for system, values in metrics.items():
            records.append({"ticker": base.ticker, "trial": base.trial_id, "risk_mode": profile, "system": system, **values})
        daily["risk_mode"] = profile
        daily_frames.append(daily)
    output = base.output_dir / "risk_profiles"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "table4_risk_profiles.csv", index=False)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    daily_all.to_csv(output / "risk_profiles_daily.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ours = summary[summary["system"].eq("ours")]
    axes[0].bar(ours["risk_mode"], ours["cumulative_return"])
    axes[0].set_title("Ours Return by Risk Profile")
    axes[0].set_ylabel("Cumulative return")
    for profile, frame in daily_all[daily_all["system"].eq("ours")].groupby("risk_mode"):
        frame = frame.sort_values("date")
        axes[1].plot((1 + frame["daily_return"].astype(float)).cumprod().values, label=profile)
    axes[1].set_title("Ours Cumulative Curve by Risk Profile")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "fig8_risk_profiles.png", dpi=180)
    fig.savefig(Path("paper/figures/fig8_risk_profiles.png"), dpi=180)
    plt.close(fig)
    return summary
