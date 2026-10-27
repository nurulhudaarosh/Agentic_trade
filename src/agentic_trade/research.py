from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .engine import ExperimentConfig, ExperimentEngine, load_all_events
from .case_studies import build_case_cards
from .probes import run_probe_benchmark

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
    # Average repeated trials per (ticker, date) so LLM sampling variance does not
    # turn the same market day into multiple pseudo-independent observations.
    pivot = results.pivot_table(index=["ticker", "date"], columns="system", values=metric, aggfunc="mean")
    if "ours" not in pivot or len(pivot) < 2:
        return {"p_value": None, "statistic": None, "significant": False}
    alternatives = [name for name in SYSTEMS if name != "ours" and name in pivot]
    second = max(alternatives, key=lambda name: float(pivot[name].mean())) if alternatives else None
    if second is None:
        return {"p_value": None, "statistic": None, "significant": False}
    paired = pivot[["ours", second]].dropna()
    if len(paired) < 2 or np.allclose(paired["ours"], paired[second]):
        return {"comparison": second, "p_value": 1.0, "statistic": 0.0, "significant": False}
    try:
        statistic, p_value = wilcoxon(paired["ours"], paired[second])
        return {"comparison": second, "p_value": float(p_value), "statistic": float(statistic), "significant": bool(p_value < 0.05)}
    except Exception:
        return {"comparison": second, "p_value": 1.0, "statistic": 0.0, "significant": False}


def markdown_table(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
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


def run_matrix(base: ExperimentConfig, tickers: list[str] = TICKERS, trials: int = 1) -> dict[str, object]:
    records: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    # A deterministic backbone ignores temperature, so repeating it would create
    # byte-identical pseudo-trials and invalidate the paired significance test.
    deterministic = base.backbone == "deterministic"
    effective_trials = 1 if deterministic else max(1, trials)
    if deterministic and trials != 1:
        print(f"Deterministic backbone: collapsing {trials} requested trials to 1.")
    for ticker in tickers:
        for trial in range(effective_trials):
            temperature = base.temperature if deterministic else min(1.0, base.temperature + trial * 0.05)
            metrics, daily, evidence = run_one(base, ticker, trial, temperature=temperature)
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


def run_probes(base: ExperimentConfig, tickers: list[str] = TICKERS) -> pd.DataFrame:
    """Run the retrieval probe benchmark over the full ticker universe."""
    events = load_all_events(base.data_dir, tickers)
    frame, _ = run_probe_benchmark(events, base.output_dir / "matrix")
    return frame


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


def summarize_ablations(sweeps_dir: Path, output_dir: Path | None = None) -> pd.DataFrame:
    """Consolidate ablation sweep files into Table 6 CSV and Markdown."""
    target_dir = output_dir or sweeps_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    feature_names = {
        "use_provenance": "w/o Learned Provenance",
        "use_reconciliation": "w/o Contradiction Retirement",
        "use_outcome_recall": "w/o Decision Recall",
        "use_topic_matching": "w/o Topic Matching",
    }
    rows: list[dict] = []
    base_return: float | None = None

    for flag, label in feature_names.items():
        csv_path = sweeps_dir / f"ablation_{flag}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        ours_df = df[df["system"].eq("ours")]
        true_row = ours_df[ours_df["setting"].astype(str).str.lower().eq("true")]
        false_row = ours_df[ours_df["setting"].astype(str).str.lower().eq("false")]

        if not true_row.empty and base_return is None:
            base_return = float(true_row.iloc[0]["cumulative_return"])
            rows.append({
                "configuration": "Full System (Ours)",
                "cumulative_return": base_return,
                "sharpe_ratio": float(true_row.iloc[0]["sharpe_ratio"]),
                "annualized_volatility": float(true_row.iloc[0]["annualized_volatility"]),
                "max_drawdown": float(true_row.iloc[0]["max_drawdown"]),
                "delta_return": 0.0,
            })

        if not false_row.empty:
            ret = float(false_row.iloc[0]["cumulative_return"])
            delta = (ret - base_return) if base_return is not None else 0.0
            rows.append({
                "configuration": label,
                "cumulative_return": ret,
                "sharpe_ratio": float(false_row.iloc[0]["sharpe_ratio"]),
                "annualized_volatility": float(false_row.iloc[0]["annualized_volatility"]),
                "max_drawdown": float(false_row.iloc[0]["max_drawdown"]),
                "delta_return": delta,
            })

    table6 = pd.DataFrame(rows)
    table6.to_csv(target_dir / "table6_ablation.csv", index=False)

    md_lines = [
        "| Configuration | Return | Sharpe | Volatility | Drawdown | Delta Return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in table6.iterrows():
        delta_str = f"{r['delta_return']:+.4f}" if r["configuration"] != "Full System (Ours)" else "-"
        md_lines.append(
            f"| {r['configuration']} | {r['cumulative_return']:.4f} | {r['sharpe_ratio']:.2f} | "
            f"{r['annualized_volatility']:.4f} | {r['max_drawdown']:.4f} | {delta_str} |"
        )
    (target_dir / "table6_ablation.md").write_text("\n".join(md_lines) + "\n")
    return table6


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables, figures, sweeps, and evidence logs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--tickers", default="TSLA,NFLX,AMZN,MSFT,COIN")
    parser.add_argument("--trials", type=int, default=1, help="LLM sampling trials (deterministic backbone always runs once)")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--sweeps", action="store_true")
    parser.add_argument("--risk-profiles", action="store_true")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run matrix, sweeps, risk profiles, and probes")
    args = parser.parse_args()
    base = ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, ticker=args.ticker, start_date=date(2026, 6, 1), train_end=date(2026, 6, 15), test_start=date(2026, 6, 16), end_date=date(2026, 7, 9))
    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]

    if args.matrix or args.all:
        run_matrix(base, tickers, args.trials)
    if args.sweeps or args.all:
        run_sweep(base, "table4_risk_modes", ["conservative", "standard", "aggressive"], "risk_mode")
        run_sweep(base, "table5_top_k", [1, 3, 5, 10], "top_k")
        for feature in ["use_provenance", "use_reconciliation", "use_outcome_recall", "use_topic_matching"]:
            run_sweep(base, f"ablation_{feature}", [True, False], feature)
        summarize_ablations(base.output_dir / "sweeps", base.output_dir / "matrix")
    if args.risk_profiles or args.all:
        run_risk_profiles(base)
    if args.probes or args.all:
        run_probes(base, tickers)
    if not (args.matrix or args.sweeps or args.risk_profiles or args.probes or args.all):
        print("Choose --matrix, --sweeps, --risk-profiles, --probes, or --all")


if __name__ == "__main__":
    main()
