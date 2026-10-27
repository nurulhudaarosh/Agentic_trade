# Agentic Trade Research

Reproducible research framework for evaluating time-ordered trading memory architectures. This project builds directly upon the base paper **FinMem** (*FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design*, Yu et al., Stevens Institute of Technology, [arXiv:2311.13743](https://arxiv.org/abs/2311.13743), PDF available locally at [`FinMem.pdf`](FinMem.pdf)).

---

## 1. Base Paper (FinMem) vs. Our Novel Architecture

### Evaluated Memory Systems:
- `no_memory`: Price-momentum baseline with zero retrieved text events.
- `plain_rag`: Flat recency-ranked vector memory (Generative Agents baseline).
- `fin_mem`: 3-layer human cognitive memory (Shallow News: 14d decay, Intermediate 10-Q: 90d decay, Deep 10-K: 365d decay).
- `ours`: **Our enhanced architecture** extending FinMem with four critical capabilities:

| Novelty | FinMem (Base Paper) | Ours (Agentic Trade) | Advantage |
| :--- | :--- | :--- | :--- |
| **Source Provenance** | Unweighted (all sources equal within layer) | **Learned Reliability**: $r_s \in [0.5, 1.0]$ optimized from training price hit-rates | Prioritizes credible sources over rumor noise |
| **Fact Reconciliation** | Pure time decay; obsolete facts remain active | **Active Contradiction Retirement**: Same-layer topic overlap + opposite sentiment retires old facts | Deactivates superseded guidance immediately |
| **Decision Memory** | Unindexed general text reflection | **Episodic Outcome Recall**: $t+1$ resolved returns recalled for similar future topics | Agent learns directly from its own trading history |
| **Corporate Graph** | Single-ticker isolation | **Multi-Ticker Graph Memory**: `related_tickers` indexed with relationship relevance bonus | Captures supply-chain and competitor spillover |
| **Retrieval Metric** | Downstream PnL only | **Pre-Registered Probe Benchmark (Table 7)** | Quantifies retrieval fidelity: **2.5x Hit@1, 1.7x precision** on the checked-in 17-query suite |

### What the checked-in offline data actually supports

- Events span `2026-06-01..2026-07-09`; the default split trains through `2026-06-15` and evaluates `2026-06-16..2026-07-09`, so test-time news ingestion is exercised.
- The snapshot has **news only** (no 10-Q/10-K), so the intermediate/deep layers are dormant and `fin_mem` collapses to the flat `plain_rag` baseline.
- Publisher attribution is sparse, so provenance falls back to source genre (`media`, `analyst_note`, `press_release`) when `source_name` is unavailable.
- `related_tickers` is supplied by [`data/relationships.csv`](data/relationships.csv); this activates cross-ticker retrieval.
- Deterministic runs are **single-trial**; temperature is only meaningful for LLM backbones. The paired Wilcoxon test aggregates by `(ticker, date)` to avoid pseudo-replication.
- On the TSLA baseline, `ours` returns `+0.47%` vs `-6.00%` for `plain_rag`/`fin_mem`, but the pooled five-ticker comparison is **not statistically significant** (`p=0.50`) and `ours` is worse on NFLX and MSFT. See [`artifacts/research/matrix/table2.md`](artifacts/research/matrix/table2.md).
- Component ablations: provenance changes returns; contradiction retirement changes `facts_retired` but not PnL in this sample; outcome recall only affects LLM prompts (the deterministic policy ignores it); topic matching is neutral in this run. See Table 6.

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

Open your browser at **`http://localhost:8050`** to access:
- **Interactive Multi-Line Equity Curves**: Real-time crosshair tooltips comparing `Ours` vs `FinMem` vs `Plain RAG` vs `Buy & Hold`.
- **Memory & Provenance Inspector**: Audit retrieved headlines, learned source reliability ($r_s$), active vs retired facts (contradiction reconciliation), and episodic outcome recall.
- **Scientific Benchmark Suite**: Tables 2, 4, 6, 7 and Wilcoxon signed-rank significance ($p < 0.05$).
- **Alpaca Paper Trading Station**: Reconciled positions, dry-run order logs, and portfolio rebalances.
- **Live Experiment Runner**: Run offline TSLA backtests and paper rebalances directly from the UI.

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
