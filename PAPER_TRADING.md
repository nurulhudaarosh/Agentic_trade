# Paper Trading

This project supports Alpaca **paper trading only**. It never connects to live trading endpoints.

## Setup

Install the package and optional broker client:

```bash
python3 -m pip install -e '.[paper]'
```

Set paper-account credentials in the environment or `.env`, never commit credentials into git:

```bash
export ALPACA_PAPER_KEY='...'
export ALPACA_PAPER_SECRET='...'
```

## Dry Run

Dry-run is the default and cannot submit real orders:

```bash
agentic-paper --tickers TSLA,AAPL,NVDA,GOOGL,MSFT --budget 100
```

## Agent Decisions & Position Management

Provide a CSV with `ticker,decision` columns. Supported decisions are `Buy`, `Sell`, and `Hold`:

```csv
ticker,decision
TSLA,Buy
AAPL,Hold
NVDA,Buy
GOOGL,Sell
MSFT,Hold
```

Execute a dry-run against that file:

```bash
agentic-paper --decisions decisions.csv --budget 100
```

### Order Execution Logic:
- **Buy**: Allocated up to `max_position_pct` (default: 20%) of configured budget.
- **Sell**: If the account holds an open long position in the ticker, it closes the position. If no open position exists and `--allow-short` is omitted, the decision is recorded as `skipped_no_position`. If `--allow-short` is enabled, a market sell order is submitted.
- **Hold**: Recorded as `not_submitted`.
- **Audit Logging**: Every order action is appended to `artifacts/paper_trading/orders.jsonl` with an ISO-8601 UTC timestamp.

### Portfolio Reconciliation

To audit current account positions against the monitored 5-ticker universe:

```bash
agentic-paper --reconcile
```

To inspect status for specific existing order IDs without submitting anything:

```bash
agentic-paper --status ORDER_ID_1 ORDER_ID_2
```

## Submit To Alpaca Paper

Only use `--execute` after reviewing the dry-run plan:

```bash
agentic-paper --decisions decisions.csv --budget 100 --execute
```

The client is constructed strictly with `paper=True`. Start with a small budget and verify orders in the Alpaca paper trading dashboard.

## LLM-Backed Research Decisions

The research engine uses deterministic rules by default. To run with an LLM backbone:

```bash
agentic-trade \
  --ticker TSLA \
  --backbone "$MODEL_NAME" \
  --temperature 0.7 \
  --output-dir artifacts/llm_tsla
```

LLM responses are cached under `artifacts/llm_cache/`; failures fall back to `Hold` and are logged in `metrics.json`.
