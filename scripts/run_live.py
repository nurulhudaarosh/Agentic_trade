#!/usr/bin/env python3
"""Run Groq-backed decisions and optional Alpaca execution."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_trade.live_trader import main

if __name__ == "__main__":
    main()
