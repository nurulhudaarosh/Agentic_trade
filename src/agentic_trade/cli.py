from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

from .engine import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic Trade memory experiment")
    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--train-end", type=date.fromisoformat, default=None)
    parser.add_argument("--test-start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--backbone", default=None, help="Use deterministic or an OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--risk-mode", choices=["conservative", "standard", "aggressive"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.config and args.config.exists():
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    overrides = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.ticker is not None:
        overrides["ticker"] = args.ticker
    if args.start is not None:
        overrides["start_date"] = args.start
    if args.train_end is not None:
        overrides["train_end"] = args.train_end
    if args.test_start is not None:
        overrides["test_start"] = args.test_start
    if args.end is not None:
        overrides["end_date"] = args.end
    if args.backbone is not None:
        overrides["backbone"] = args.backbone
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.risk_mode is not None:
        overrides["risk_mode"] = args.risk_mode
    if args.top_k is not None:
        overrides["top_k"] = args.top_k

    if overrides:
        config = replace(config, **overrides)

    metrics = run_experiment(config)
    print(f"Wrote research artifacts to {config.output_dir}")
    for name, values in metrics.items():
        print(f"{name}: return={values.get('cumulative_return', 0.0):.4f}, sharpe={values.get('sharpe_ratio', 0.0):.2f}")


if __name__ == "__main__":
    main()
