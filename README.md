# Agentic Trade Research

Reproducible research code for comparing four time-ordered trading memory policies:

- `no_memory`: price-momentum baseline
- `plain_rag`: flat recency-ranked event memory
- `fin_mem`: layered memory
- `ours`: layered memory with provenance, topic matching, contradiction retirement, and outcome memory

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

Copy `.env.example` to `.env` only when using external APIs. Never commit `.env` or real credentials.

## Reproducible Offline Run

The checked-in `data/` files are small research inputs. The default run uses a deterministic policy and writes generated output under the ignored `artifacts/` directory:

```bash
python run_experiment.py --data-dir data --output-dir artifacts/offline
```

## LLM-Backed Run

The client supports Groq-compatible and OpenRouter endpoints. Credentials are read from environment variables and failures fall back to `Hold`:

```bash
set -a; source .env; set +a
env -u OPEN_ROUTER_KEY -u OPEN_ROUTER_kEY \
  python run_experiment.py \
  --ticker TSLA \
  --backbone openai/gpt-oss-20b \
  --output-dir artifacts/llm_tsla
```

LLM-backed outputs are exploratory backtests, not live-trading advice. They include no transaction costs or slippage and do not establish statistical significance.

## Data Refresh

With Alpaca and yfinance credentials configured, refresh the market inputs with:

```bash
python scripts/fetch_real_data.py
```

The input schema is `date,ticker,source_type,text,summary` for events and `date,ticker,close` for prices. Events are only visible on or after their event date. An optional `related_tickers` column accepts comma-separated peer tickers; those events are loaded for the target and shown to the relationship-aware retriever and LLM.

The Ours memory learns source reliability from the training slice, retires same-layer facts when a later event has an overlapping topic and opposite sentiment, and resolves each decision after the next close. Resolved decisions with overlapping topics are recalled in later prompts. `metrics.json` records `facts_retired` and `decisions_recalled`.

## Paper Trading

The paper-trading adapter is documented in [PAPER_TRADING.md](PAPER_TRADING.md). It defaults to dry-run mode and is separate from the research backtest.