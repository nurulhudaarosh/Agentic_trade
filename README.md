# Agentic Trade — Remember, Verify, Act

**Self-correcting agent memory for agentic trading and market analysis.**
UIU CSE Final Year Design Project. The base paper and main baseline is FinMem
(Yu et al., [arXiv:2311.13743](https://arxiv.org/abs/2311.13743), PDF in [`FinMem.pdf`](FinMem.pdf)).

LLM trading agents retrieve news and filings from a memory before each Buy/Sell/Hold decision.
Existing layered memories trust every source equally, keep contradicted facts until they decay,
store the same story many times and never learn from their own outcomes. Our memory keeps a
layered core and adds seven components:

| Role | Component | What it fixes |
|---|---|---|
| Remember | N2 near-duplicate consolidation | the same story stored many times |
| | N4 situational query | retrieval ignores what is happening today |
| | N5 data-driven importance | importance drawn at random |
| | N6 point-in-time peer graph | news about related companies ignored |
| Verify | N1 hierarchical source credibility | every source trusted equally |
| | N3 contradiction retirement | contradicted facts kept until they decay |
| | N7 outcome-aware episodic recall | no learning from the agent's own outcomes |
| Act | decision policies, guardrails | rule, ML and LLM decisions; budget limits in the app |

---

## 1. Study and results

| | |
|---|---|
| Universe | FinMem's TSLA, NFLX, AMZN, MSFT, COIN + NVDA, AAPL, GOOGL, META, AMD |
| Data | 23,247 Alpaca/Benzinga articles (Sep 2025 – Oct 2026), 93 SEC 10-Q/10-K XBRL filings, daily bars + SPY/QQQ |
| Point-in-time | items visible at the first 16:00 ET close after publication; decide at close t, earn close t → t+1; 5 bps costs |
| Encoders | FinBERT polarity + MiniLM embeddings, shared by every system |
| Protocols | **rolling walk-forward** (Dec 2025 – Oct 2026, 227 sessions, monthly threshold re-selection) and a fixed split |
| Baselines | flat retrieval (Plain RAG), faithful FinMem, rules, LogReg/LightGBM, 3 free LLM backbones, GRU, 2 Transformers, Q-learning, kNN patterns |

Headline (rolling protocol, `artifacts/study/rolling/`):

- Best risk-adjusted result of the memory designs: Sharpe **−0.20** vs FinMem −0.72 and Plain RAG −0.65;
  maximum drawdown −15.2% vs −19.7%; retrieved context 26 vs 58 days old.
- FinMem's own five stocks: **+3.5%** vs FinMem −28.6%; ours beats FinMem in 9 of 10 leave-one-stock-out cases.
- Three Groq LLM backbones (2,250 decisions): our context beats FinMem's with every backbone and roughly halves the drawdown.
- Honest limits: the gain is **not statistically significant** (Wilcoxon p = 0.39; LLM test p = 0.12), the next-day IC
  of every news-memory signal is ≈ 0, buy-and-hold and price-only LightGBM do better in this period, and about
  47 years of daily data would be needed to confirm the observed edge.

Full notes: [`paper/RESEARCH_LOG.md`](paper/RESEARCH_LOG.md). Report: [`paper/uiu_template/`](paper/uiu_template/README.md).

### Reproduce

```bash
python scripts/build_dataset.py              # news, bars, SEC filings (needs ALPACA_KEY / ALPACA_SECRET)
python scripts/run_study.py --fresh --figures # annotate, replay all memories, both protocols, figures 1-20
python scripts/run_llm_study.py --days 50     # LLM backbones (free Groq keys)
python scripts/run_extensions.py              # GRU, Transformers, Q-learning, pattern analogues
python scripts/make_extension_figures.py      # figures 21-32
python scripts/export_latex_tables.py         # report tables
python scripts/export_extension_tables.py
```

---

## 2. Installation and tests

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
pytest -q                                     # 93 tests
```

Copy `.env.example` to `.env` only when connecting to live APIs (Alpaca, Groq, OpenRouter); keys can also be
added from the app's Settings page. Never commit credentials: `.env` and `artifacts/credentials.json` are git-ignored.

---

## 3. The application

```bash
./run_dashboard.sh                            # or: python scripts/run_dashboard.py --port 8050
```

Open `http://localhost:8050`. A dependency-free Python server and a plain HTML/CSS/JS front end, in English
and Bangla, light and dark themes, desktop and phone. One stock picker drives every page.

| Page | What it does |
|---|---|
| Home (`#home`) | account value, autopilot status, the AI's view of the stock in plain words, budget and safety meters |
| Trade (`#trade`) | real daily candles and indicators, a combined chart signal, the AI decision button, an order ticket |
| Portfolio (`#portfolio`) | positions, orders and account history |
| AI Autopilot (`#autopilot`) | schedule, decision mode (rules / LLM / both must agree), permission mode, approval queue, emergency stop |
| Insights (`#insights`) | budget split, pattern finder, what-if, risk, correlation, algorithm board with the Q-learning agent |
| Stock scanner (`#scanner`) | all ten stocks with memory signal, FinMem action, price-model P(up) and consensus |
| AI memory (`#agent`) | the retrieved memories with tone and credibility, the three memories compared, LLM decision with cited IDs |
| Research lab (`#research`) | the study's results: equity curves, tables, tests, model zoo |
| Settings (`#settings`) | budget and hard limits, broker and LLM connections (keys verified before saving), preferences |

Every autopilot order passes one server-side guardrail (cash reserve, per-stock and per-order caps, daily-loss and
drawdown limits, trades per day, allowed stocks). Paper trading is the default; live trading needs `--live` and
`ALPACA_LIVE_CONFIRM=YES`.

Report screenshots: `python scripts/capture_app_screens.py http://localhost:8050` then
`python scripts/crop_paper_screens.py`.

---

## 4. Legacy prototype (5-week snapshot)

The first prototype (`engine.py`, `memory.py`, `research.py`, `paper_trader.py`, `live_trader.py`) ran on a
June–July 2026 snapshot. It is kept for its CLIs and tests; its numbers are superseded by the study above.

```bash
agentic-trade --config configs/offline_tsla.yaml   # offline backtest -> artifacts/offline/
python scripts/run_research.py --all               # prototype tables -> artifacts/research/
agentic-paper --decisions decisions.csv --budget 100 [--execute]
agentic-live --risk-mode aggressive --budget 100 [--execute]
```

See [`PAPER_TRADING.md`](PAPER_TRADING.md) for the execution adapter.

> Trading is risky. This is a research and decision-support tool, not investment advice.

---

## 5. Documentation

- [`paper/uiu_template/`](paper/uiu_template/README.md): the FYDP report (LaTeX, UIU template).
- [`paper/RESEARCH_LOG.md`](paper/RESEARCH_LOG.md): running record of decisions, problems and results.
- [`AGENT.md`](AGENT.md): repository layout and invariants for AI agents and contributors.
- [`PAPER_TRADING.md`](PAPER_TRADING.md): paper-trading adapter and live-money opt-in.
- [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md), [`paper/EXPERIMENT_STATUS.md`](paper/EXPERIMENT_STATUS.md): prototype-era drafts, superseded.
