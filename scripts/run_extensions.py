#!/usr/bin/env python3
"""Run the extension experiments (GRU, Transformer, RL, pattern analogs) on the study's rolling protocol."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_trade.study.extensions import run  # noqa: E402

if __name__ == "__main__":
    summary = run(ROOT / "data" / "study", ROOT / "artifacts" / "study")
    for name, m in summary["portfolio"].items():
        print(f"{name:40s} CR {m['cumulative_return']:+.3f}  Sharpe {m['sharpe']:+.2f}  MDD {m['max_drawdown']:+.3f}  beta {m.get('beta', float('nan')):+.2f}")
    print(json.dumps({k: (round(v['sharpe_diff'], 2), round(v.get('wilcoxon_p', float('nan')), 3)) for k, v in summary["tests"].items()}, indent=1))
    print("runtime", summary["runtime_s"], "s")
