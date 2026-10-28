# Agent Harness Specification & Research Reference: Agentic Trade

This document is the definitive guide and operational contract for AI agent harnesses (and human researchers) working in this repository. It provides architecture specifications, theoretical foundations comparing our system against the baseline **FinMem** paper, strict operational invariants, and exact execution instructions.

---

## 1. Research Foundation & Base Paper Alignment

### Base Paper: FinMem (arXiv:2311.13743)
- **Title**: *FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design* (Yu et al., Stevens Institute of Technology).
- **Core Concept**: Emulates human cognitive processing (Craik & Lockhart, 1972) by stratifying financial data into three decay layers:
  1. **Shallow Layer**: Daily market news ($\tau \approx 14$ days decay).
  2. **Intermediate Layer**: Quarterly 10-Q filings ($\tau \approx 90$ days decay).
  3. **Deep Layer**: Annual 10-K filings and extended reflections ($\tau \approx 365$ days decay).
- **FinMem Profiling & Decision Making**: Character definition with risk inclinations (Risk-Seeking, Risk-Averse, Self-Adaptive) and discrete trading decisions (`Buy`, `Sell`, `Hold`).

### Our System ("Ours"): What Was Missing in FinMem & Our Core Novelties

While FinMem introduced layered decay, it suffers from critical real-world limitations that our architecture explicitly solves:

```
+---------------------------------------------------------------------------------------+
|                                    OURS ARCHITECTURE                                  |
|                                                                                       |
|  +--------------------+   +-----------------------+   +----------------------------+  |
|  | Learned Provenance |   | Dynamic Contradiction |   | Episodic Decision Outcome  |  |
|  |    Reliability     |   |      Retirement       |   |       Memory Recall        |  |
|  | (Hit-rate learning |   | (Deactivates stale/   |   | (Resolves at t+1; recalls  |  |
|  |  from train slice) |   |  contradicted facts)  |   |  prior similar outcomes)   |  |
|  +---------+----------+   +-----------+-----------+   +-------------+--------------+  |
|            |                          |                             |                 |
|            +--------------------------+-----------------------------+                 |
|                                       v                                               |
|                    +------------------------------------+                             |
|                    |   Hierarchical Retrieval Engine    |                             |
|                    |  - Stratified layers (10K/10Q/News)|                             |
|                    |  - Topic keyword alignment         |                             |
|                    |  - Cross-ticker relationship score |                             |
|                    |  - Strict point-in-time filtering  |                             |
|                    +------------------+-----------------+                             |
|                                       |                                               |
|                                       v                                               |
|                    +------------------------------------+                             |
|                    |    Execution & Decision Engine     |                             |
|                    |  - Risk Modes (Std/Cons/Aggr)      |                             |
|                    |  - Position-aware Paper Trader     |                             |
|                    |  - Automated Reconciliation        |                             |
|                    +------------------------------------+                             |
+---------------------------------------------------------------------------------------+
```

1. **Dynamic Source Provenance & Hit-Rate Reliability**:
   - *FinMem Flaw*: Treats all sources within a layer equally. Analyst rumors and SEC audited filings have identical trust.
   - *Our Novelty*: Evaluates source predictive accuracy during the training period ($r_s = \frac{\sum hits}{N} \in [0.5, 1.0]$ where hit = $sentiment \times \Delta close_{t+1} > 0$). Publisher attribution is used when available; otherwise events fall back to a source-genre category (`media`, `analyst_note`, `press_release`). Reliability is smoothed toward a 0.7 prior and requires at least two observations. On the checked-in snapshot only two genres have enough support, so this is a small-sample estimate.
2. **Active Contradiction Retirement & Fact Reconciliation**:
   - *FinMem Flaw*: Pure time decay. When an event is reversed (e.g., an acquisition blocked or a guidance miss announced after optimism), the outdated fact lingers in memory and pollutes decisions.
   - *Our Novelty*: When an incoming event in the same layer shares topic tags with an older event but has opposite sentiment ($sentiment_{old} \times sentiment_{new} < 0$), the older event is marked `active = False` and retired.
3. **Resolved Episodic Decision Outcome Memory**:
   - *FinMem Flaw*: Extended reflections summarize trends, but the model cannot query past decisions indexed by topic and see what *actually happened*.
   - *Our Novelty*: Each trading decision is recorded alongside its topic tags. When next-day price settles, the engine calls `resolve(decision_date, realized_return)`. When future similar market topics emerge, `recall(topics)` feeds the past decision and its realized return directly into prompt context.
4. **Supply Chain & Multi-Ticker Relationship Graph Awareness**:
   - *FinMem Flaw*: Single-ticker isolation. Fails to incorporate supplier, customer, or competitor dynamics.
   - *Our Novelty*: Ingests `related_tickers` from [`data/relationships.csv`](data/relationships.csv) and indexed event rows. Applies the relationship scoring bonus only to events from a *related* ticker, so cross-company evidence can surface without flooding the results with unrelated headlines.
5. **Pre-Registered Probe Retrieval Benchmark (Table 7)**:
   - *FinMem Flaw*: No direct ground-truth retrieval evaluation.
   - *Our Novelty*: Built-in benchmark measuring $Hit@1$, $Hit@3$, $Hit@5$, $MRR$, and $Precision@5$ across 17 pre-registered, ticker-scoped corporate relationship, guidance, and legal risk queries (`agentic_trade.probes`).
6. **Production Execution Adapter & Portfolio Reconciliation**:
   - *FinMem Flaw*: Theoretical action tokens only.
   - *Our Novelty*: Position-aware paper trading (`agentic-paper`), budget capping (20%), position liquidation on `Sell`, short support (`--allow-short`), and portfolio reconciliation (`--reconcile`).

---

## 2. Repository Layout & File Manifest

```text
├── AGENT.md                 <- (This file) Operational specification for agent harnesses
├── README.md                <- Quickstart and user guide
├── RESEARCH_PAPER.md        <- Research paper draft comparing FinMem vs. Ours
├── PAPER_TRADING.md         <- Alpaca paper trading and position reconciliation guide
├── FinMem.pdf               <- The base paper (Yu et al., arXiv:2311.13743)
├── pyproject.toml           <- Package metadata, dependencies, scripts, pytest config
├── requirements.txt         <- Pinned core and optional dependencies
├── run_experiment.py        <- Compatibility CLI runner for baseline offline backtest
├── decisions.csv            <- Sample agent decisions input for paper trading
├── configs/                 <- Declarative YAML experiment configurations
│   ├── offline_tsla.yaml    <- Deterministic baseline offline config
│   ├── multi_ticker.yaml    <- 5-ticker matrix configuration
│   ├── risk_conservative.yaml <- Conservative risk profile config
│   └── risk_aggressive.yaml   <- Aggressive risk profile config
├── data/                    <- Checked-in point-in-time market data
│   ├── multi_ticker_news.csv<- Ingested news events (TSLA, NFLX, AMZN, MSFT, COIN)
│   ├── prices.csv           <- Daily close prices
│   └── relationships.csv    <- Static peer/supply-chain graph (`related_tickers`)
├── src/agentic_trade/       <- Core research package
│   ├── __init__.py          <- Package root (configures MPLCONFIGDIR=/tmp/matplotlib)
│   ├── cli.py               <- CLI entrypoint (`agentic-trade`) with YAML config support
│   ├── engine.py            <- ExperimentEngine, ExperimentConfig, backtest loop
│   ├── memory.py            <- MemorySystem, NoMemory, PlainRAG, FinMem, OursMemory
│   ├── paper_trader.py      <- PaperBroker, PaperTradingAgent (`agentic-paper`)
│   ├── live_trader.py       <- Groq decisions + Alpaca execution (`agentic-live`)
│   ├── dashboard_server.py  <- Zero-dependency dashboard HTTP server (`agentic-dashboard`)
│   ├── research.py          <- Multi-ticker matrix, Wilcoxon, sweeps (`agentic-research`)
│   ├── probes.py            <- Table 7 memory probe retrieval benchmark
│   ├── case_studies.py      <- Deterministic pre-registered case card builder
│   └── llm.py               <- OpenRouter / Groq / OpenAI LLM decision client
├── scripts/                 <- Research automation scripts
│   ├── run_research.py      <- Full paper benchmark suite runner (--all, --matrix, etc.)
│   ├── run_live.py          <- Groq + Alpaca wrapper (`--execute`, `--live`)
│   └── fetch_real_data.py   <- Alpaca News & yfinance live ingestion with pagination
├── tests/                   <- unit and integration tests (100% pass required)
│   ├── test_memory.py       <- Leakage, topics, retirement, recall, source categories
│   ├── test_engine.py       <- Backtest execution, relationships, daily returns
│   ├── test_paper_trader.py <- Order limits, Sell closing, shorting, reconcile
│   ├── test_live_trader.py  <- Groq decision flow, history, live opt-in guard
│   ├── test_dashboard.py    <- Dashboard data loaders and API payloads
│   ├── test_llm.py          <- Parser validation, fallback to Hold on missing credentials, risk profile prompt differentiation
│   ├── test_case_studies.py <- Trial deduplication, case card structure
│   ├── test_config.py       <- YAML configuration parsing and serialization
│   ├── test_probes.py       <- Hit@K/MRR/Precision, ticker isolation
│   └── test_research.py     <- Paired Wilcoxon stats, Table 6 ablation consolidation
└── paper/                   <- Publication artifacts and status
    ├── README.md            <- Overview of generated paper figures
    ├── EXPERIMENT_STATUS.md <- Status tracking of empirical tables and limitations
    └── figures/             <- High-resolution plots for publication
```

---

## 3. Strict Invariants for Agent Harnesses (DO NOT BREAK)

When working or modifying code in this repository, any automated agent MUST observe these rules:

1. **Zero Future Lookahead Leakage**:
   - In `engine.py`, memories available on `current_date` MUST satisfy `event.event_date <= current_date`.
   - Realized returns are strictly $r_{t} = \frac{Close_{t+1}}{Close_{t}} - 1$.
   - The final observation day has $r_{final} = 0.0$ because its $t+1$ close is unavailable.
2. **Point-in-Time Train/Test Integrity**:
   - Provenance learning (`_learn_provenance`) and training memory ingestion MUST only use events where `event_date <= train_end`.
   - Test evaluation MUST only process dates where `test_start <= d <= end_date`.
3. **No Fabricated or Mock Results**:
   - Never hardcode or fake numbers in `metrics.json`, `REPORT.md`, or paper markdown tables.
   - All empirical numbers must be generated via code execution from data.
4. **Scoping and Trial Separation**:
   - In `build_case_cards`, always filter `daily` and `evidence` to a single trial (`trial == 0`) so memory ranks 1, 2, 3 reflect distinct memories rather than duplicate trial records.
   - In `engine.py`, compute `cited` *before* referencing it in `llm_result`.
5. **Matplotlib Headless Safety**:
   - Keep `os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")` in `__init__.py` and modules importing `matplotlib` to ensure non-interactive execution without permission errors.
6. **Test Suite Integrity**:
   - All tests in `tests/` (64 as of the one-year study) MUST pass at all times:
     ```bash
     .venv/bin/pytest -v
     ```
7. **Honest Statistics**:
   - Deterministic backbones ignore `temperature`; `run_matrix` collapses pseudo-trials to one run.
   - `paired_wilcoxon` averages repeated trials per `(ticker, date)`; never treat duplicated deterministic rows as independent observations.
   - Do not describe a significant difference as an improvement unless the sign of the metric supports it.
8. **Ticker-Scoped Retrieval**:
   - Probe and decision retrieval MUST pass `ticker=` so events from unrelated companies cannot satisfy a query.
   - Cross-ticker events are allowed only when the query ticker appears in `related_tickers`.
9. **Execution Safety**:
   - Paper trading is the default. Live orders require `--live` **and** `ALPACA_LIVE_CONFIRM=YES`; never weaken this guard.
   - The offline snapshot is news-only (no 10-Q/10-K); do not claim layer ablations are exercised unless filing data is present.
10. **LLM Character & Risk Mode Profiling**:
   - To prevent passive "all-Hold" collapse, LLM prompts must inject character design per `--risk-mode` (`aggressive`, `standard`, `conservative`).
   - Caches under `artifacts/llm_cache/` MUST key on `(model, temperature, prompt, system_prompt, risk_mode)` so changing risk profile generates distinct decision samples.

### Known Limitations of the Offline Snapshot

- News covers `2026-06-01..2026-07-09`; the default split is train `06-01..06-15`, test `06-16..07-09`.
- All sources are `news`; intermediate/deep layers are dormant, so `fin_mem` equals `plain_rag`.
- Publisher fields are sparse; provenance falls back to `media`/`analyst_note`/`press_release` genre reliability.
- Contradiction retirement changes `facts_retired` but not PnL in the current sample; outcome recall only affects LLM prompts, not the deterministic policy.

---

## 4. Standard Operational Playbook

### A. Run Default Baseline Backtest
```bash
python run_experiment.py --data-dir data --output-dir artifacts/offline
# OR using the installed CLI:
agentic-trade --config configs/offline_tsla.yaml
```
Outputs written to `artifacts/offline/`:
- `no_memory_daily.csv`, `plain_rag_daily.csv`, `fin_mem_daily.csv`, `ours_daily.csv`
- `evidence_log.csv`: Full trace of retrieved memories per observation
- `metrics.json`: Performance metrics (Cumulative Return, Sharpe, Volatility, Max Drawdown)
- `REPORT.md`: Markdown summary table
- `cumulative_returns.png`, `metric_comparison.png`

### B. Run Full Research Suite (Tables 2, 4, 5, 6, 7 & Wilcoxon)
```bash
python scripts/run_research.py --all
```
Generated artifacts in `artifacts/research/`:
- `matrix/table2.md` & `table2_metrics.csv`: 5-ticker paired comparison (single deterministic trial)
- `matrix/wilcoxon.json`: Date-pooled paired significance test
- `matrix/case_study_cards.md`: Pre-registered case study cards
- `matrix/table6_ablation.md`: Consolidated component ablation table
- `matrix/table7_probes.md`: 17-query ticker-scoped memory probe benchmark
- `risk_profiles/table4_risk_profiles.csv`: Risk mode evaluation
- `sweeps/table5_top_k.csv`: Working memory capacity sweep ($K \in \{1, 3, 5, 10\}$)

### C. Run Paper Trading Adapter
```bash
# Dry run from decision file
agentic-paper --decisions decisions.csv --budget 100

# Audit current broker positions vs 5-stock universe
agentic-paper --reconcile

# Submit to Alpaca Paper Account (requires ALPACA_PAPER_KEY in .env)
agentic-paper --decisions decisions.csv --budget 100 --execute
```

### D. Run Groq-Backed Decisions and Paper Execution
```bash
# Dry-run with Groq decisions
agentic-live --model llama-3.3-70b-versatile --budget 100

# Submit Groq decisions to Alpaca paper
agentic-live --model llama-3.3-70b-versatile --budget 100 --execute

# Refresh news/prices first
python scripts/run_live.py --refresh --execute
```
`agentic-live` requires `GROQ_API_KEY`. Live-money trading additionally requires
`--live` and `ALPACA_LIVE_CONFIRM=YES`; paper is the default.

---

## 5. Algorithmic Formulations (Quick Reference)

### Retrieval Scoring in `OursMemory`
$$\text{Score}(E) = \left(r_{\text{source}}(E) + \frac{|\text{Topics}(E) \cap \text{Topics}(Q)|}{\max(|\text{Topics}(Q)|, 1)} + \text{Bonus}_{\text{peer}} + 0.05 \cdot \text{Sentiment}(E), \quad t_E\right)$$
where $\text{Bonus}_{\text{peer}} = 0.10$ when the event belongs to a *different* ticker that lists the query ticker in `related_tickers`, else $0$.

### Contradiction Retirement
$$\text{Retire}(E_{\text{old}}) \iff \left(E_{\text{old}}.\text{layer} == E_{\text{new}}.\text{layer}\right) \land \left(\text{Topics}(E_{\text{old}}) \cap \text{Topics}(E_{\text{new}}) \neq \emptyset\right) \land \left(s_{\text{old}} \cdot s_{\text{new}} < 0\right)$$

### Decision Outcome Memory
$$\text{Recall}(Q_{\text{topics}}) = \text{most recent } D \in \text{DecisionHistory} \quad \text{s.t.} \quad D.\text{realized\_return} \neq \text{None} \land \text{Topics}(D) \cap Q_{\text{topics}} \neq \emptyset$$
When recalled, prompt receives:
`ID=outcome:YYYY-MM-DD text="Prior similar decision: BUY; realized return: +0.021500"`
