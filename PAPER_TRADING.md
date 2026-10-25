# Paper Trading

This project supports Alpaca **paper trading only**. It never uses a live endpoint.

## Setup

Install the optional broker client:

```bash
python3 -m pip install -e '.[paper]'
```

Set paper-account credentials in the shell, never in source files:

```bash
export ALPACA_PAPER_KEY='...'
export ALPACA_PAPER_SECRET='...'
```

## Dry Run

Dry-run is the default and cannot submit orders:

```bash
PYTHONPATH=src python3 -m agentic_trade.paper_trader \
  --tickers TSLA,AAPL,NVDA,GOOGL,MSFT \
  --budget 100
```

## Agent Decisions

Provide a CSV with `ticker,decision` columns. Supported decisions are `Buy`, `Sell`, and `Hold`. The current execution adapter only submits Buy orders; Sell is recorded as a non-submitted decision until positions/reconciliation are implemented.

```csv
ticker,decision
TSLA,Buy
AAPL,Hold
NVDA,Buy
GOOGL,Hold
MSFT,Buy
```

Run a dry-run from that file:

```bash
PYTHONPATH=src python3 -m agentic_trade.paper_trader \
  --decisions decisions.csv \
  --budget 100
```

Each Buy is capped at 20% of the configured budget. Every decision is appended to `artifacts/paper_trading/orders.jsonl`.

To inspect an existing order without submitting anything:

```bash
PYTHONPATH=src python3 -m agentic_trade.paper_trader \
  --status ORDER_ID_1 ORDER_ID_2
```

## Submit To Alpaca Paper

Only use `--execute` after checking the printed dry-run plan:

```bash
PYTHONPATH=src python3 -m agentic_trade.paper_trader \
  --decisions decisions.csv \
  --budget 100 \
  --execute
```

The client is constructed with `paper=True`. Start with a small budget and verify orders in the Alpaca paper dashboard. This is an execution adapter, not investment advice, and it does not yet implement portfolio reconciliation, stop-loss orders, or sell-position management.

## LLM-Backed Research Decisions

The research CLI uses deterministic rules by default. To use the configured OpenRouter model for decisions:

```bash
PYTHONPATH=src python3 -m agentic_trade.cli \
  --ticker TSLA \
  --backbone "$MODEL_NAME" \
  --temperature 0.7 \
  --output-dir artifacts/llm_tsla
```

LLM responses are cached under `artifacts/llm_cache/`; failures fall back to `Hold` and are recorded in `metrics.json`. Never mix deterministic and LLM-backed artifacts in one paper table.
