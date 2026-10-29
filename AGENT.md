# Agent Harness Specification & Research Reference: Agentic Trade

This document is the definitive guide and operational contract for AI agent harnesses (and human researchers) working in this repository. It provides architecture specifications, theoretical foundations comparing our system against the baseline **FinMem** paper, strict operational invariants, and exact execution instructions.

---

## 1. Research Foundation

### Base paper and main baseline: FinMem (arXiv:2311.13743)
*FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design* (Yu et al.).
Three decay layers (shallow news Q=14 days, intermediate 10-Q Q=90, deep 10-K Q=365), recency + relevancy +
importance scoring, purge rules, access-counter promotion, top-K per layer, a self-adaptive risk character and
Buy/Sell/Hold decisions. We re-implement it faithfully in `study/memory.py` (`fin_mem`) as the main baseline.

### Our system: self-correcting agent memory (Remember, Verify, Act)
`ours` in `src/agentic_trade/study/memory.py` keeps the FinMem core and adds seven switchable components:

| Role | Component | Mechanism |
|---|---|---|
| Remember | N2 near-duplicate consolidation | cosine >= 0.88, same layer and polarity sign -> merge, count corroboration |
| | N4 situational query | query = normalize(task embedding + mean embedding of today's articles) |
| | N5 data-driven importance | v from tone strength and news-attention surprise instead of a random draw |
| | N6 point-in-time peer graph | 120-day co-mention graph, top-3 peers, same-day peer news |
| Verify | N1 hierarchical source credibility | Beta-Bernoulli hit rate shrunk author -> category -> global (prior 8), market-adjusted |
| | N3 contradiction retirement | cosine >= 0.72, opposite polarity, both abs(s) >= 0.3 -> older memory retired |
| | N7 outcome-aware episodic recall | 10 most similar past contexts (>= 5 needed); abstain when their signed edge < 0 |
| Act | policies and guardrails | threshold rule, LogReg/LightGBM, LLM backbones; the app's autopilot guardrail |

Results, limitations and every design decision are in `paper/RESEARCH_LOG.md` and the report
(`paper/uiu_template/`). Headline: best memory design on Sharpe and drawdown, not statistically significant.

---

## 2. Repository Layout

```text
├── README.md, AGENT.md, PAPER_TRADING.md, FinMem.pdf
├── src/agentic_trade/
│   ├── study/                 <- one-year study (the research of the report)
│   │   ├── data.py            <- download, point-in-time alignment, incremental refresh
│   │   ├── annotate.py        <- FinBERT polarity, MiniLM embeddings, headline categories
│   │   ├── memory.py          <- Plain RAG, faithful FinMem, Ours (N1-N7 flags)
│   │   ├── backtest.py        <- day-major replay of all memories
│   │   ├── policies.py        <- threshold rule, rule baselines, LogReg/LightGBM
│   │   ├── evaluation.py      <- metrics, Wilcoxon/Holm, Diebold-Mariano, bootstrap, IC
│   │   ├── pipeline.py        <- rolling and fixed protocols
│   │   ├── llm_agent.py       <- LLM backbone experiment (Groq key pool, prompt cache)
│   │   ├── extensions.py      <- GRU, Transformers, Q-learning, pattern analogues, CAPM, power
│   │   └── figures.py         <- figures 1-20
│   ├── dashboard_server.py    <- HTTP API of the application (`agentic-dashboard`)
│   ├── agent_service.py       <- research agent used by the app (brief, consensus, LLM decision)
│   ├── autopilot.py           <- budget, hard guardrail, permission modes, approval queue, scheduler
│   ├── analytics.py           <- pattern analogues, risk, correlation, allocation, Monte Carlo
│   ├── algorithms.py          <- algorithm board and the app's Q-learning agent
│   ├── credentials.py         <- API keys verified with the provider, stored owner-only
│   └── engine.py, memory.py, research.py, probes.py, case_studies.py, paper_trader.py,
│       live_trader.py, llm.py, cli.py   <- legacy 5-week prototype and its CLIs
├── dashboard/                 <- front end: index.html, app.css, core.js, i18n.js (EN/BN), charts.js,
│                                 trade.js, autopilot.js, analytics.js, app.js, credentials.js
├── scripts/                   <- build_dataset, run_study, run_llm_study, run_extensions,
│                                 make_extension_figures, export_latex_tables, export_extension_tables,
│                                 capture_app_screens, crop_paper_screens, run_dashboard (+ legacy scripts)
├── data/study/                <- one-year dataset (news, filings, bars, annotations)
├── artifacts/study/           <- study outputs (signals, evidence, rolling/, fixed/, llm/)
├── tests/                     <- 93 tests (study, autopilot, analytics, algorithms, credentials, legacy)
└── paper/                     <- report (uiu_template/), figures/, RESEARCH_LOG.md
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
   - All tests in `tests/` (93) MUST pass at all times:
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
10. **Study invariants**:
   - The study replay is day-major: all tickers advance together, so shared state (credibility) only uses outcomes already public.
   - An item is visible only from the first 16:00 New York close at or after its timestamp; outcomes of day-t decisions resolve at close t+1.
   - Every table in `paper/uiu_template/tables/` is generated by a script; never edit them by hand.
11. **Application safety**:
   - Every autopilot order goes through `autopilot.gate`; never add an execution path that bypasses it, for any broker.
   - Risk-reducing sells are only blocked by the emergency stop; the agent never opens short positions.
   - API keys are never returned unmasked to the browser; key endpoints accept only same-origin local requests.
12. **LLM Character & Risk Mode Profiling (legacy `agentic-live`)**:
   - To prevent passive "all-Hold" collapse, LLM prompts must inject character design per `--risk-mode` (`aggressive`, `standard`, `conservative`).
   - Caches under `artifacts/llm_cache/` MUST key on `(model, temperature, prompt, system_prompt, risk_mode)` so changing risk profile generates distinct decision samples.

### Known Limitations of the Legacy Offline Snapshot (prototype only)

- News covers `2026-06-01..2026-07-09`; the default split is train `06-01..06-15`, test `06-16..07-09`.
- All sources are `news`; intermediate/deep layers are dormant, so `fin_mem` equals `plain_rag`.
- Publisher fields are sparse; provenance falls back to `media`/`analyst_note`/`press_release` genre reliability.
- Contradiction retirement changes `facts_retired` but not PnL in the current sample; outcome recall only affects LLM prompts, not the deterministic policy.

---

## 4. Operational Playbook

One-year study and application: see `README.md` sections 1 and 3. The commands below are for the legacy prototype.

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

## 5. Legacy Prototype Formulations (Quick Reference)

The study memory's formulas are in the report, Chapter 3 (`paper/uiu_template/3.design.tex`).

### Retrieval Scoring in `OursMemory`
$$\text{Score}(E) = \left(r_{\text{source}}(E) + \frac{|\text{Topics}(E) \cap \text{Topics}(Q)|}{\max(|\text{Topics}(Q)|, 1)} + \text{Bonus}_{\text{peer}} + 0.05 \cdot \text{Sentiment}(E), \quad t_E\right)$$
where $\text{Bonus}_{\text{peer}} = 0.10$ when the event belongs to a *different* ticker that lists the query ticker in `related_tickers`, else $0$.

### Contradiction Retirement
$$\text{Retire}(E_{\text{old}}) \iff \left(E_{\text{old}}.\text{layer} == E_{\text{new}}.\text{layer}\right) \land \left(\text{Topics}(E_{\text{old}}) \cap \text{Topics}(E_{\text{new}}) \neq \emptyset\right) \land \left(s_{\text{old}} \cdot s_{\text{new}} < 0\right)$$

### Decision Outcome Memory
$$\text{Recall}(Q_{\text{topics}}) = \text{most recent } D \in \text{DecisionHistory} \quad \text{s.t.} \quad D.\text{realized\_return} \neq \text{None} \land \text{Topics}(D) \cap Q_{\text{topics}} \neq \emptyset$$
When recalled, prompt receives:
`ID=outcome:YYYY-MM-DD text="Prior similar decision: BUY; realized return: +0.021500"`
