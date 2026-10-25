#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from agentic_trade.engine import ExperimentConfig
from agentic_trade.research import run_matrix, run_risk_profiles, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables, figures, sweeps, and evidence logs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--tickers", default="TSLA,NFLX,AMZN,MSFT,COIN")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--sweeps", action="store_true")
    parser.add_argument("--risk-profiles", action="store_true")
    args = parser.parse_args()
    base = ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, ticker=args.ticker, start_date=date(2026, 6, 1), train_end=date(2026, 8, 31), test_start=date(2026, 9, 1), end_date=date(2026, 10, 23))
    if args.matrix:
        run_matrix(base, [item.strip().upper() for item in args.tickers.split(",") if item.strip()], args.trials)
    if args.sweeps:
        run_sweep(base, "table4_risk_modes", ["conservative", "standard", "aggressive"], "risk_mode")
        run_sweep(base, "table5_top_k", [1, 3, 5, 10], "top_k")
        for feature in ["use_provenance", "use_reconciliation", "use_outcome_recall", "use_topic_matching"]:
            run_sweep(base, f"ablation_{feature}", [True, False], feature)
    if args.risk_profiles:
        run_risk_profiles(base)
    if not args.matrix and not args.sweeps and not args.risk_profiles:
        print("Choose --matrix, --sweeps, and/or --risk-profiles")

if __name__ == "__main__":
    main()
