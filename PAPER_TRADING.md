# Paper Trading

Alpaca **paper trading is the default and the supported workflow**. Live-money
execution exists only behind an explicit double opt-in (`--live` plus
`ALPACA_LIVE_CONFIRM=YES`) and is not used by any automated script.

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

## Groq-Backed Decisions & Risk Profiles (`agentic-live`)

The `agentic-live` entrypoint connects an LLM (Groq / OpenRouter / OpenAI) to the paper-trading adapter:

```bash
# Groq decides with aggressive alpha-seeking profile; dry-run only
agentic-live --model llama-3.3-70b-versatile --risk-mode aggressive --budget 100

# Groq decides; submit to Alpaca paper
agentic-live --model llama-3.3-70b-versatile --risk-mode aggressive --budget 100 --execute

# Refresh news/prices first
agentic-live --refresh --execute --risk-mode aggressive
```

### Risk Profile Tuning (`--risk-mode`):
- `--risk-mode aggressive`: Injects an alpha-seeking directive into the LLM system prompt, prioritizing active `Buy` and `Sell` trades on directional catalysts and preventing passive all-`Hold` collapse.
- `--risk-mode standard`: Balanced risk/reward decision agent.
- `--risk-mode conservative`: Prioritizes capital preservation, falling back to `Hold` on ambiguity.

It reads `GROQ_API_KEY` (or `GROQ_KEY`), defaults to the Groq endpoint, and writes:

- `artifacts/live/decisions_<timestamp>.csv` — action, reasoning, cited memory IDs, model, status, risk mode.
- `artifacts/live/decision_history.jsonl` — resolved `t+1` outcomes used for episodic recall.
- `artifacts/paper_trading/orders.jsonl` — Alpaca order audit trail.

Decision recall is persisted across runs: unresolved records are resolved from the
next available close before the next decision batch.

## Visual Dashboard Audit

Launch the interactive dashboard to inspect order books, reconciled positions, and decision logs visually:

```bash
./run_dashboard.sh
# or: agentic-dashboard --port 8050
```
Navigate to the **💼 Alpaca Paper** tab in `http://localhost:8050` to inspect target signals (`decisions.csv`) and live order execution events (`orders.jsonl`).

## Live-Money Opt-In (Not Recommended for Experiments)

Paper trading is the default. Live orders are blocked unless **all** of the
following are true:

1. `--execute --live` is passed to `agentic-live` (or `agentic-paper --execute --live`).
2. `ALPACA_LIVE_CONFIRM=YES` is set.
3. Real-money `ALPACA_KEY` / `ALPACA_SECRET` are configured.

Without the confirmation variable the broker raises an error rather than
connecting. Review the dry-run output first and use a small budget.
