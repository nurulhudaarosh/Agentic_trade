"""Full-scale study pipeline: one-year, ten-ticker evaluation of FinMem vs. our memory.

Modules
-------
- ``data``: Alpaca news/bars + SEC EDGAR filings, point-in-time trading-day alignment.
- ``annotate``: FinBERT sentiment and MiniLM sentence embeddings (cached).
- ``memory``: faithful FinMem layered memory and our extended memory (with ablation flags).
- ``backtest``: walk-forward multi-ticker backtester with transaction costs.
- ``policies``: rule, ML and LLM decision policies.
- ``evaluation``: metrics, significance tests, bootstrap confidence intervals.
- ``figures``: publication figures.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
