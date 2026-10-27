#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Ensure src is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_trade.engine import ExperimentConfig
from agentic_trade.research import run_matrix, run_probes, run_risk_profiles, run_sweep, summarize_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables, figures, sweeps, and evidence logs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--tickers", default="TSLA,NFLX,AMZN,MSFT,COIN")
    parser.add_argument("--trials", type=int, default=1, help="LLM sampling trials (deterministic backbone always runs once)")
    parser.add_argument("--matrix", action="store_true", help="Run multi-ticker trials and Wilcoxon tests (Table 2)")
    parser.add_argument("--sweeps", action="store_true", help="Run parameter sweeps and component ablations (Tables 4, 5, 6)")
    parser.add_argument("--risk-profiles", action="store_true", help="Run risk mode comparison (Table 4 & Figure 8)")
    parser.add_argument("--probes", action="store_true", help="Run probe retrieval accuracy benchmark (Table 7)")
    parser.add_argument("--all", action="store_true", help="Run matrix, sweeps, risk-profiles, and probes")
    args = parser.parse_args()

    base = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        ticker=args.ticker,
        start_date=date(2026, 6, 1),
        train_end=date(2026, 6, 15),
        test_start=date(2026, 6, 16),
        end_date=date(2026, 7, 9),
    )

    if args.matrix or args.all:
        tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
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
