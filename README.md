# Agentic Trade Research

Reproducible research framework for evaluating time-ordered trading memory architectures. This project builds directly upon the base paper **FinMem** (*FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design*, Yu et al., Stevens Institute of Technology, [arXiv:2311.13743](https://arxiv.org/abs/2311.13743), PDF available locally at [`FinMem.pdf`](FinMem.pdf)).

---

## 1. One-year study: faithful FinMem vs. our memory (`src/agentic_trade/study/`)

| | |
|---|---|
| Universe | FinMem's TSLA, NFLX, AMZN, MSFT, COIN + NVDA, AAPL, GOOGL, META, AMD |
| Data | 23,247 Alpaca/Benzinga articles (Sep 2025 – Oct 2026), 93 SEC 10-Q/10-K XBRL filings, SIP daily bars + SPY/QQQ |
| Point-in-time | items visible at the first 16:00 ET close after publication; decide at close t, earn close t → t+1; 5 bps costs |
| Encoders | FinBERT polarity + MiniLM embeddings, shared by every system |
| Protocols | **rolling walk-forward** (Dec 2025 – Oct 2026, 227 sessions, monthly threshold re-selection) and fixed split (test May – Oct 2026) |

Memory systems: `plain_rag` (flat vector memory), **faithful `fin_mem`** (Q = 14/90/365,
importance v·α^δ with FinMem's layer probabilities, purge rules, access-counter promotion,
top-K per layer) and **`ours`** = FinMem + N1 hierarchical online source credibility,
N2 duplicate consolidation, N3 semantic contradiction retirement, N4 situational query,
N5 data-driven importance, N6 point-in-time co-mention peer graph, N7 outcome-aware
episodic recall/abstention. Baselines also include B&H, momentum/mean-reversion rules,
LogReg/LightGBM on price features (± memory features) and four free LLM backbones.

Headline (rolling protocol, `artifacts/study/rolling/`): Ours Sharpe −0.20 vs FinMem −0.72
and Plain RAG −0.65; on FinMem's five stocks Ours +3.5% vs FinMem −28.6%. The gain over
FinMem is **not statistically significant** (Wilcoxon p = 0.39), next-day IC of every
news-memory signal is ≈ 0, and Buy & Hold / price-only LightGBM are stronger in this period.
Full notes: [`paper/RESEARCH_LOG.md`](paper/RESEARCH_LOG.md).

```bash
python scripts/build_dataset.py          # download news, bars, SEC filings (needs ALPACA_KEY/SECRET)
python scripts/run_study.py --fresh      # annotate, replay all memory systems, both protocols, tables
python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from agentic_trade.study.figures import render_all; render_all(Path('data/study'), Path('artifacts/study'), Path('paper/figures'))"
python scripts/run_llm_study.py --days 50 # LLM backbones (Groq / OpenRouter free tiers)
```

The earlier 5-week prototype (`engine.py`, `memory.py`, `research.py`) is kept for the
legacy CLI and tests; its numbers are superseded by the study above.

---

## 2. Quickstart & Installation

```bash
# 1. Create and activate environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install in editable mode
pip install -e .

# 3. Verify all 36 unit and integration tests pass
pytest -v
```

> [!NOTE]
> Copy `.env.example` to `.env` only when connecting to live APIs (Alpaca, Groq, OpenRouter). Never commit credentials to version control. The repository is 100% functional offline with checked-in data.

---

## 3. Interactive Research & Trading Dashboard

Launch the zero-dependency interactive web dashboard (Bloomberg/TradingView terminal dark theme):

```bash
# Quick launcher:
./run_dashboard.sh

# Or using Python directly:
python scripts/run_dashboard.py --port 8050

# Or via the installed CLI:
agentic-dashboard --port 8050
```

Open your browser at **`http://localhost:8050`** (deep links: `/#agent`, `/#scanner`, `/#research`) to access:
- **Memory Agent**: the research memory agent (Ours) live per ticker — retrieved memories by layer with credibility, corroboration and age, peer and episodic evidence, FinMem/Plain-RAG comparison, a memory+price-model consensus, an LLM decision with cited memories, and one-click paper execution. "Refresh market data" pulls new news/bars/filings, annotates them and replays memory.
- **Scanner**: all ten stocks with memory signal, FinMem action, price-model P(up) and consensus; execute consensus on paper.
- **Research Lab**: study results for both protocols and both universes — interactive equity curves, all-strategy table, significance tests, per-ticker heatmap, IC, LLM backbones, memory operations, learned credibility and the paper figure gallery.
- **Trading Terminal / AI Agent Harness / Portfolio**: charting, manual orders with server-side risk limits, and the auto-agent loop — which now runs the research memory agent (LLM with consensus fallback).
- **Alpaca Paper Trading Station**: Reconciled positions, order logs, and portfolio rebalances.

---

## 4. Reproducible Offline Backtest

Run the deterministic TSLA baseline pipeline:

```bash
# Direct runner:
python run_experiment.py --data-dir data --output-dir artifacts/offline

# Or using the installed CLI with declarative YAML config:
agentic-trade --config configs/offline_tsla.yaml
```

Generated outputs under `artifacts/offline/`:
- `metrics.json`: Performance metrics (Cumulative Return, Sharpe Ratio, Volatility, Max Drawdown).
- `REPORT.md`: Markdown summary table.
- `evidence_log.csv`: Complete audit trail of retrieved memory IDs and rankings per day.
- `cumulative_returns.png` & `metric_comparison.png`: Publication-ready figures (also copied to `paper/figures/`).

---

## 5. Full Research Suite (Tables 2, 4, 5, 6, 7 & Wilcoxon)

To regenerate all empirical research tables and statistical tests from the paper:

```bash
python scripts/run_research.py --all
```

Or execute individual benchmarks:
- `python scripts/run_research.py --matrix`: 5-ticker paired Wilcoxon signed-rank test (**Table 2**). Deterministic backbones always run a single trial; use `--backbone <model> --trials N` for genuine LLM sampling trials.
- `python scripts/run_research.py --risk-profiles`: Conservative vs. Standard vs. Aggressive risk profiles (**Table 4 & Figure 8**).
- `python scripts/run_research.py --sweeps`: Working memory capacity $K \in \{1, 3, 5, 10\}$ (**Table 5**) & component ablations (**Table 6**).
- `python scripts/run_research.py --probes`: 17-query ticker-scoped entity retrieval accuracy (**Table 7**).

The same options are available through the installed `agentic-research` entrypoint.

Results are exported to `artifacts/research/matrix/`:
- `table2.md` & `table2_metrics.csv`
- `table4_risk_profiles.csv`
- `table6_ablation.md`
- `table7_probes.md`
- `case_study_cards.md`
- `wilcoxon.json`

---

## 6. Paper Trading Execution & Portfolio Reconciliation

The production-ready execution adapter connects agent decisions to Alpaca Paper Trading (dry-run by default):

```bash
# Dry-run against agent decisions CSV
agentic-paper --decisions decisions.csv --budget 100

# Audit account positions against monitored universe
agentic-paper --reconcile

# Submit to Alpaca Paper account (requires ALPACA_PAPER_KEY)
agentic-paper --decisions decisions.csv --budget 100 --execute
```

See [PAPER_TRADING.md](PAPER_TRADING.md) for execution semantics, short-selling (`--allow-short`), and audit logs.

---

## 7. Groq-Backed Live Decisions & LLM Risk Profiles

`agentic-live` generates per-ticker decisions with an LLM (Groq / OpenRouter / OpenAI), persists decision/outcome history, and optionally routes the resulting orders through Alpaca.

```bash
# 1. Configure credentials in .env (never commit it)
#    GROQ_API_KEY=...
#    ALPACA_PAPER_KEY=... / ALPACA_PAPER_SECRET=...

# 2. Refresh news + prices (Alpaca news API + yfinance)
python scripts/fetch_real_data.py --tickers TSLA,NFLX,AMZN,MSFT,COIN

# 3. Dry-run: Groq decides with Aggressive risk profile (avoids passive Hold)
agentic-live --model llama-3.3-70b-versatile --risk-mode aggressive --budget 100

# 4. Execute on the Alpaca PAPER account
agentic-live --model llama-3.3-70b-versatile --risk-mode aggressive --budget 100 --execute

# Or refresh and execute in one step
python scripts/run_live.py --refresh --execute --risk-mode aggressive
```

### LLM Character & Risk Profiles (`--risk-mode`):
To prevent the classic LLM failure mode where a generic "cautious agent" defaults to **100% `Hold`** on mixed news, the system injects distinct system instructions and decision directives:
- **`aggressive`**: Alpha-seeking directive. Instructs the model to capture market upside and strong directional momentum, favoring active `Buy`/`Sell` over passive `Hold` when catalysts exist.
- **`standard`**: Balanced risk/reward decision agent weighing positive/negative catalysts and price momentum.
- **`conservative`**: Capital preservation directive prioritizing drawdown avoidance and defaulting to `Hold` on ambiguity.

- Decisions are written to `artifacts/live/decisions_<timestamp>.csv`.
- Cache keys explicitly include `risk_mode` and `system_prompt`, guaranteeing clean cache isolation across risk inclinations.
- Resolved `t+1` outcomes are stored in `artifacts/live/decision_history.jsonl` and fed back into future prompts (episodic outcome recall).
- `--execute` uses `TradingClient(paper=True)`. **Live-money orders require both `--live` and `ALPACA_LIVE_CONFIRM=YES`**; without them the broker refuses to connect.
- LLM schema failures or missing keys fall back to `Hold`, and the status is recorded in `orders.jsonl`/decision output.

> Trading is risky. The included strategies are research prototypes, not investment advice. Start with paper trading and a small budget.

---

## 8. Documentation Index for Agent Harnesses & Researchers

- [`AGENT.md`](AGENT.md): **Essential reading for AI agents and harnesses.** Contains system invariants, point-in-time constraints, algorithmic formulas, and data contracts.
- [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md): Comprehensive research paper draft documenting the FinMem comparison, our novelties, mathematical formulations, and empirical tables.
- [`PAPER_TRADING.md`](PAPER_TRADING.md): Execution adapter specification and portfolio reconciliation manual.
- [`paper/EXPERIMENT_STATUS.md`](paper/EXPERIMENT_STATUS.md): Empirical completion checklist and honesty guidelines.
