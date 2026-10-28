#!/usr/bin/env python3
"""LLM-backbone experiment on FinMem's five tickers over the last N test days."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from agentic_trade.study.data import FINMEM_TICKERS
from agentic_trade.study.llm_agent import BACKBONES, run_llm_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--study", type=Path, default=Path("artifacts/study"))
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--backbones", default="gpt-oss-120b,gpt-oss-20b,qwen3.8-27b")
    parser.add_argument("--tickers", default=",".join(FINMEM_TICKERS))
    parser.add_argument("--workers", type=int, default=None, help="concurrent decision chains per backbone")
    parser.add_argument("--out-name", default="llm_decisions.parquet")
    args = parser.parse_args()
    signals = pd.read_parquet(args.study / "signals.parquet")
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    evidence = pd.read_parquet(args.study / "evidence.parquet")
    evidence["date"] = pd.to_datetime(evidence["date"]).dt.date
    test_days = sorted(signals.loc[signals["split"] == "test", "date"].unique())[-args.days:]
    prices = pd.read_parquet(args.data / "prices.parquet")
    news = pd.read_parquet(args.data / "news.parquet")
    filings = pd.read_parquet(args.data / "filings.parquet")
    frame = run_llm_experiment(signals, evidence, news, filings, prices, args.tickers.split(","), test_days,
                               args.backbones.split(","), args.study / "llm",
                               workers={b: args.workers for b in args.backbones.split(",")} if args.workers else None,
                               out_name=args.out_name)
    print(frame.groupby(["backbone", "system"])["action"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
