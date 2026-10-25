#!/usr/bin/env python3
"""Compatibility entrypoint for the reproducible research CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_trade.cli import main


if __name__ == "__main__":
    main()
