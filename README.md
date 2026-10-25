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
| **Retrieval Metric** | Downstream PnL only | **Pre-Registered Probe Benchmark (Table 7)** | Quantifies exact retrieval fidelity: **2x Hit@1, 2.4x precision** |

---

## 2. Quickstart & Installation

```bash
# 1. Create and activate environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 2. Install in editable mode
pip install -e .

# 3. Verify all 24 unit and integration tests pass
pytest -v
```

> [!NOTE]
> Copy `.env.example` to `.env` only when connecting to live APIs (Alpaca, Groq, OpenRouter). Never commit credentials to version control. The repository is 100% functional offline with checked-in data.

---

## 3. Reproducible Offline Backtest

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

## 4. Full Research Suite (Tables 2, 4, 5, 6, 7 & Wilcoxon)

To regenerate all empirical research tables and statistical tests from the paper:

```bash
python scripts/run_research.py --all
```

Or execute individual benchmarks:
- `python scripts/run_research.py --matrix`: 5-ticker 3-trial matrix & paired Wilcoxon signed-rank test (**Table 2**).
- `python scripts/run_research.py --risk-profiles`: Conservative vs. Standard vs. Aggressive risk profiles (**Table 4 & Figure 8**).
- `python scripts/run_research.py --sweeps`: Working memory capacity $K \in \{1, 3, 5, 10\}$ (**Table 5**) & component ablations (**Table 6**).
- `python scripts/run_research.py --probes`: 25-query entity and relationship retrieval accuracy (**Table 7**).

Results are exported to `artifacts/research/matrix/`:
- `table2.md` & `table2_metrics.csv`
- `table4_risk_profiles.csv`
- `table6_ablation.md`
- `table7_probes.md`
- `case_study_cards.md`
- `wilcoxon.json`

---

## 5. Paper Trading Execution & Portfolio Reconciliation

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

## 6. Documentation Index for Agent Harnesses & Researchers

- [`AGENT.md`](AGENT.md): **Essential reading for AI agents and harnesses.** Contains system invariants, point-in-time constraints, algorithmic formulas, and data contracts.
- [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md): Comprehensive research paper draft documenting the FinMem comparison, our novelties, mathematical formulations, and empirical tables.
- [`PAPER_TRADING.md`](PAPER_TRADING.md): Execution adapter specification and portfolio reconciliation manual.
- [`paper/EXPERIMENT_STATUS.md`](paper/EXPERIMENT_STATUS.md): Empirical completion checklist and honesty guidelines.
