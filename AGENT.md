# Agentic Trade Research Project

This repository is a reproducible research project for comparing memory systems in an agentic trading workflow.

## Systems

- `no_memory`: price-momentum baseline with no retrieved events
- `plain_rag`: recency-ranked flat event store
- `fin_mem`: layered memory with source and layer-aware retrieval
- `ours`: FinMem plus topic matching, provenance weighting, contradiction retirement, and outcome memory

## Run Offline

```bash
python3 -m pip install -e .
python3 run_experiment.py --data-dir data --output-dir artifacts/offline
```

The run writes daily decisions, `metrics.json`, `REPORT.md`, and paper-ready PNG figures under `artifacts/offline/`.

## Reproducibility Rules

- Input events must use `date,ticker,source_type,text,summary`.
- Prices must use `date,close`; provide `data/prices.csv` for a network-free run.
- Events are visible only on or before their event date.
- The final test-day return is zero because its next close is unavailable.
- No API key is stored in source or documentation. Live providers are optional and must read credentials from environment variables.
- Results are descriptive and do not claim statistical significance.

## Tests

```bash
python3 -m pytest -q
```

The tests include future-event leakage protection, contradiction retirement, and an offline end-to-end engine fixture.

## Layout

```text
src/agentic_trade/  core package and CLI
scripts/             research utilities
configs/             experiment configuration
artifacts/           generated run outputs (ignored)
paper/               manuscript and figure outputs
tests/               unit and integration tests
data/                checked-in input data
```
