#!/usr/bin/env python3
"""Run the full one-year study: memory systems, baselines, ML models, statistics, figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_trade.study.data import UNIVERSE
from agentic_trade.study.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/study"))
    parser.add_argument("--fresh", action="store_true", help="recompute memory signals instead of reusing them")
    parser.add_argument("--figures", action="store_true", help="also render publication figures")
    args = parser.parse_args()
    run(args.data, args.out, UNIVERSE, reuse=not args.fresh)
    for protocol in ("rolling", "fixed"):
        print(f"## {protocol}\n" + (args.out / protocol / "table_main.md").read_text())
    if args.figures:
        from agentic_trade.study.figures import render_all

        render_all(args.data, args.out, Path("paper/figures"))


if __name__ == "__main__":
    main()
