#!/usr/bin/env python3
"""Run the Agentic Trade Research Dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_trade.dashboard_server import main

if __name__ == "__main__":
    main()

