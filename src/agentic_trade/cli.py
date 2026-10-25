from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .engine import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Trade memory experiment")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/offline"))
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 6, 1))
    parser.add_argument("--train-end", type=date.fromisoformat, default=date(2026, 8, 31))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 23))
    parser.add_argument("--backbone", default="deterministic", help="Use deterministic or an OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    metrics = run_experiment(ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, ticker=args.ticker, start_date=args.start, train_end=args.train_end, test_start=args.test_start, end_date=args.end, backbone=args.backbone, temperature=args.temperature))
    print(f"Wrote research artifacts to {args.output_dir}")
    for name, values in metrics.items():
        print(f"{name}: return={values.get('cumulative_return', 0.0):.4f}, sharpe={values.get('sharpe_ratio', 0.0):.2f}")

if __name__ == "__main__":
    main()
