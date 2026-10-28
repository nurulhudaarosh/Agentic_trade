#!/usr/bin/env python3
"""Download the one-year, ten-ticker study dataset (news, bars, SEC filings)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_trade.study.data import UNIVERSE, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/study"))
    parser.add_argument("--tickers", default=",".join(UNIVERSE))
    args = parser.parse_args()
    paths = build_dataset(args.out, [t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
