# Agentic Trade: Memory for Time-Ordered Trading Decisions

## Abstract

This repository evaluates four memory policies for a time-ordered trading decision workflow: No Memory, Plain RAG, FinMem, and Ours. The default reproducible run uses checked-in TSLA event data and price data downloaded at runtime when available. The offline engine is deterministic and is a pipeline baseline, not evidence of LLM capability.

## Research Question

Does structured memory improve decision quality and risk-adjusted returns without using information that was unavailable at decision time?

## Experimental Design

Events are timestamped and become visible only on or after their event date. The training period ends before the test period. The test engine records the next available close as the realized return and includes a Buy-and-Hold reference. Metrics include cumulative return, annualized volatility, Sharpe ratio, maximum drawdown, and hold rate.

## Reproduction

```bash
python3 -m pip install -e .
python3 run_experiment.py --data-dir data --output-dir artifacts/offline
python3 -m pytest -q
```

Generated metrics and daily decisions are in `artifacts/offline/`. Paper figures are copied to `paper/figures/`.

## Limitations

The checked-in dataset is small, the default signal policy is deterministic, transaction costs and slippage are not modeled, and one ticker cannot establish generality or statistical significance. Live news, SEC ingestion, and LLM adapters must be added as explicitly configured modules and reported separately from the offline pipeline.

## Results

Populate this section from the generated `metrics.json` for a named run. Never hand-edit numerical results or present mock values as empirical findings.
