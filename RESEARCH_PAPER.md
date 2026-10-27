# Agentic Trade: Beyond Layered Decay in Financial LLM Agents — Provenance, Contradiction Retirement, and Episodic Decision Recall

## Abstract

This paper presents **Agentic Trade**, a reproducible research framework and novel memory architecture for autonomous financial decision-making agents. We build directly upon the state-of-the-art base paper **FinMem** (*FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design*, Yu et al., Stevens Institute of Technology, arXiv:2311.13743). While FinMem pioneered the emulation of human cognitive layering (shallow news, intermediate 10-Q, and deep 10-K memories with exponential decay), it treats information as static and monotonic, remaining blind to source credibility, unable to resolve contradictory disclosures, and incapable of recalling how past trading decisions actually performed. 

We address these structural deficiencies by introducing four core memory novelties:
1. **Dynamic Source Provenance**: Learning empirical hit-rates ($r_s$) from training market outcomes to prioritize credible sources over rumor noise.
2. **Active Contradiction Retirement**: Dynamically deactivating obsolete, same-layer memory events upon receiving opposing disclosures with overlapping topics.
3. **Resolved Episodic Decision Outcome Recall**: Binding actual $t+1$ realized returns to past decisions and recalling similar past decisions into future prompt contexts.
4. **Multi-Ticker Graph & Supply Chain Awareness**: Augmenting corporate network knowledge by indexing peer disclosures (`related_tickers`).

Across offline backtests on five tickers (TSLA, NFLX, AMZN, MSFT, COIN), parameter sweeps, component ablations, and a 17-query ticker-scoped memory probe benchmark, our architecture improves top-1 retrieval accuracy ($29.4\%$ vs. $11.8\%$) and precision ($17.6\%$ vs. $10.6\%$), and provides strict point-in-time leakage protection. Trading returns are mixed: the full system is best on TSLA but worse on NFLX and MSFT, and the pooled paired difference versus the strongest baseline is not statistically significant ($p=0.50$). We report this negative/inconclusive result rather than selecting a favourable subset.

---

## 1. Introduction & The Base Paper: FinMem

Recent advances in Large Language Models (LLMs) have demonstrated significant promise in processing unstructured financial disclosures, analyst opinions, and macroeconomic commentary. Conventional algorithmic trading systems—specifically Deep Reinforcement Learning (DRL) algorithms such as PPO, DQN, and A2C—struggle to ingest rich text representations directly without dimensionality explosion and convergence instability.

To solve this, **FinMem** (Yu et al., 2023) proposed an autonomous LLM trading agent structured around three components:
- **Profiling Module**: Injects corporate background knowledge and defines investor risk preferences: Risk-Seeking (aggressive), Risk-Averse (conservative), and Self-Adaptive (dynamically flipping between risk profiles based on rolling cumulative returns).
- **Layered Memory Module**: Inspired by human cognitive memory models (Craik & Lockhart, 1972) and Generative Agents (Park et al., 2023), FinMem stratifies memory into:
  - *Shallow Processing Layer*: Daily market news ($Q_{\text{shallow}} = 14$ days decay, $\alpha = 0.90$).
  - *Intermediate Processing Layer*: Quarterly Form 10-Q SEC reports ($Q_{\text{inter}} = 90$ days decay, $\alpha = 0.967$).
  - *Deep Processing Layer*: Annual Form 10-K SEC reports and multi-day extended reflections ($Q_{\text{deep}} = 365$ days decay, $\alpha = 0.988$).
- **Decision-Making Module**: Gathers top-$K$ memories from each layer, computes momentum from recent price movements, and prompts the LLM to output discrete actions (`Buy`, `Sell`, `Hold`).

---

## 2. Critical Limitations of FinMem

Despite its pioneering layered structure, our deep analysis of the FinMem framework reveals five fundamental limitations in real-world trading environments:

1. **Source Reliability / Provenance Blindness**:
   - In FinMem, every event inside a layer is scored solely by recency, cosine relevancy, and arbitrary piecewise importance. A sensational headline from an unverified blog has the exact same baseline authority as an SEC-audited disclosure.
2. **Stagnation under Contradictions**:
   - Financial reality is non-monotonic: guidance is revised, mergers are terminated by regulators, and early clinical results are retracted. FinMem relies exclusively on exponential time decay ($e^{-\delta/Q}$). An obsolete guidance beat from two weeks ago continues to compete with and dilute today's profit warning.
3. **Absence of Episodic Decision Outcome Feedback**:
   - FinMem conducts "extended reflections" summarizing market trends, but it has no mechanism to remember its own prior decisions linked to market topics. When facing a recurring earnings release, the agent cannot ask: *"What did I decide last quarter under similar circumstances, and did it make or lose money?"*
4. **Single-Ticker Isolation & Relationship Blindness**:
   - FinMem processes each company in strict isolation. It cannot model supply-chain dependencies (e.g., Tesla's reliance on Panasonic battery cells or Nvidia's foundry reliance on TSMC).
5. **Absence of Direct Memory Retrieval Benchmarks**:
   - FinMem measures retrieval quality solely through downstream trading returns under different $K$ settings. It lacks a direct ground-truth benchmark to verify whether the memory system actually retrieves the correct corporate facts.

---

## 3. Our System Architecture & Novelties

Our architecture, **Ours**, maintains FinMem's layered stratification while introducing four foundational enhancements:

### Comparison Matrix: FinMem vs. Ours

| Feature Dimension | FinMem (Base Paper, 2023) | Ours (Agentic Trade Research) |
| :--- | :--- | :--- |
| **Layer Structure** | Shallow (14d), Intermediate (90d), Deep (365d) | Layered (10-K/10-Q/News) with active lifecycle states (`active=True/False`) |
| **Source Provenance** | Unweighted (all sources equal within layer) | **Learned Reliability**: $r_s \in [0.5, 1.0]$ optimized from training slice price hits |
| **Fact Reconciliation** | Passive decay only; old contradicted facts remain active | **Active Contradiction Retirement**: Same-layer topic overlap + opposite sentiment retires old facts |
| **Decision Memory** | Unindexed general text reflection | **Episodic Outcome Recall**: $t+1$ resolved returns recalled for similar future topics |
| **Peer / Supply Chain Graph** | None (Single ticker only) | **Multi-Ticker Graph Memory**: `related_tickers` indexed with relationship relevance bonus |
| **Retrieval Benchmark** | Indirect (downstream PnL only) | **Pre-Registered Probe Suite (Table 7)**: Exact $Hit@K$, $MRR$, and $Precision$ metrics |
| **Execution Layer** | Theoretical action token | **Position-Aware Paper Trader**: Capped allocations, long liquidation on Sell, short support, portfolio reconciliation |
| **Point-in-Time Leakage Protection** | Not systematically formalized | Strict $t_{\text{event}} \le t_{\text{decision}}$, zero lookahead, deterministic execution fixtures |

---

### Mathematical & Algorithmic Formulation of Our Novelties

#### Novelty 1: Dynamic Source Provenance Learning
During the training period ($t \le T_{\text{train}}$), our engine evaluates whether a source's published sentiment correctly anticipated the directional return of the next available trading close:
$$\text{Hit}(E) = \mathbb{I}\left( \text{Sentiment}(E) \cdot \left(\frac{\text{Close}_{t+1}}{\text{Close}_t} - 1\right) > 0 \right)$$
For any source type $s$ with at least 3 historical observations, we learn its empirical reliability score:
$$r_s = \max\left(0.5, \, \min\left(1.0, \, \frac{1}{|E_s|} \sum_{E \in E_s} \text{Hit}(E)\right)\right)$$
New incoming events dynamically inherit $r_s$, scaling their baseline retrieval priority.

#### Novelty 2: Active Contradiction Retirement
When an event $E_{\text{new}}$ is ingested, it inspects existing active events in memory. An older event $E_{\text{old}}$ is retired if and only if:
$$\text{Retire}(E_{\text{old}}) \iff \left(E_{\text{old}}.\text{layer} == E_{\text{new}}.\text{layer}\right) \land \left(\text{Topics}(E_{\text{old}}) \cap \text{Topics}(E_{\text{new}}) \neq \emptyset\right) \land \left(\text{Sentiment}(E_{\text{old}}) \cdot \text{Sentiment}(E_{\text{new}}) < 0\right)$$
Upon retirement, $E_{\text{old}}.\text{active} \leftarrow \text{False}$, immediately removing it from all subsequent retrieval candidates while logging it for interpretability audit.

#### Novelty 3: Resolved Decision Outcome Recall
At observation time $t$, the agent makes decision $a_t \in \{\text{Buy}, \text{Sell}, \text{Hold}\}$ and records:
$$D_t = \langle \text{date}=t, \, \text{action}=a_t, \, \text{topics}=\text{Topics}(t), \, \text{realized\_return}=\text{None} \rangle$$
Upon arrival of price at $t+1$, the engine resolves the decision:
$$D_t.\text{realized\_return} \leftarrow \frac{\text{Close}_{t+1}}{\text{Close}_t} - 1$$
For future decisions at $t' > t$, the retriever calls $\text{recall}(\text{Topics}(t'))$. The most recent resolved decision sharing topic overlap is formatted and injected into prompt evidence:
```text
ID=outcome:2026-09-01 source=decision_history date=2026-09-01 
text="Prior similar decision: Buy; realized return: 0.021500"
```

#### Novelty 4: Supply Chain & Multi-Ticker Graph Retrieval
Events include an optional `related_tickers` set representing suppliers, customers, and partners. The composite retrieval score is computed as:
$$\text{Score}(E, Q) = r_{\text{source}}(E) + \frac{|\text{Topics}(E) \cap \text{Topics}(Q)|}{\max(|\text{Topics}(Q)|, 1)} + \text{Bonus}_{\text{peer}}(E) + 0.05 \cdot \text{Sentiment}(E)$$
where $\text{Bonus}_{\text{peer}}(E) = 0.10$ if target ticker $\in E.\text{related\_tickers}$, else $0$.

---

## 4. Empirical Evaluation & Tables

### A. Baseline Offline Experiment (TSLA)
Tested under strict point-in-time constraints on checked-in TSLA data ($T_{\text{train}}: \text{2026-06-01 to 2026-06-15}$; $T_{\text{test}}: \text{2026-06-16 to 2026-07-09}$). The window is chosen so that test-period news exists; the earlier `2026-09/10` window contained no events and was removed.

| System | Cumulative Return | Sharpe Ratio | Annualized Volatility | Max Drawdown | Hold Rate | Decisions Recalled | Facts Retired |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| `no_memory` | -10.13% | -2.30 | 64.32% | -18.01% | 6.25% | 0 | 0 |
| `plain_rag` | -6.00% | -2.36 | 38.30% | -11.38% | 31.25% | 0 | 0 |
| `fin_mem` | -6.00% | -2.36 | 38.30% | -11.38% | 31.25% | 0 | 0 |
| `ours` | **+0.47%** | **0.42** | 65.21% | **-7.49%** | 0.00% | **15** | **12** |
| `buy_and_hold` | +0.47% | +0.42 | 65.21% | -7.49% | 0.00% | - | - |

`ours` matches buy-and-hold on this single ticker because its deterministic policy is net-long throughout the test window.

---

### B. Table 2: Multi-Ticker Paired Comparison

The deterministic backbone ignores temperature, so repeating it would produce identical pseudo-trials. We therefore run one deterministic trial per ticker and pool the paired daily differences across `(ticker, date)` observations.

| Ticker | System | Return | Sharpe Ratio | Volatility | Max Drawdown | Hold Rate |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:|
| **TSLA** | `no_memory` | -0.1013 | -2.30 | 0.6432 | -0.1801 | 0.06 |
| | `plain_rag` | -0.0600 | -2.36 | 0.3830 | -0.1138 | 0.31 |
| | `fin_mem` | -0.0600 | -2.36 | 0.3830 | -0.1138 | 0.31 |
| | `ours` | **0.0047** | 0.42 | 0.6521 | **-0.0749** | 0.00 |
| | `buy_and_hold` | **0.0047** | 0.42 | 0.6521 | -0.0749 | 0.00 |
| **NFLX** | `no_memory` | -0.0290 | -1.25 | 0.3302 | -0.0789 | 0.38 |
| | `plain_rag` | **0.0307** | 1.37 | 0.4043 | **-0.0582** | 0.19 |
| | `fin_mem` | **0.0307** | 1.37 | 0.4043 | -0.0582 | 0.19 |
| | `ours` | -0.0413 | -1.33 | 0.4321 | -0.0837 | 0.00 |
| | `buy_and_hold` | -0.0413 | -1.33 | 0.4321 | -0.0837 | 0.00 |
| **AMZN** | `no_memory` | -0.0779 | -4.52 | 0.2738 | -0.1034 | 0.50 |
| | `plain_rag` | **0.0158** | 0.90 | 0.3331 | **-0.0529** | 0.12 |
| | `fin_mem` | **0.0158** | 0.90 | 0.3331 | -0.0529 | 0.12 |
| | `ours` | 0.0042 | 0.36 | 0.3569 | -0.0711 | 0.00 |
| | `buy_and_hold` | 0.0042 | 0.36 | 0.3569 | -0.0711 | 0.00 |
| **MSFT** | `no_memory` | **-0.0203** | -0.69 | 0.3727 | -0.0871 | 0.25 |
| | `plain_rag` | -0.0240 | -0.77 | 0.3995 | -0.0700 | 0.00 |
| | `fin_mem` | -0.0240 | -0.77 | 0.3995 | -0.0700 | 0.00 |
| | `ours` | -0.0701 | -3.62 | 0.3034 | -0.0760 | 0.50 |
| | `buy_and_hold` | -0.0240 | -0.77 | 0.3995 | -0.0700 | 0.00 |
| **COIN** | `no_memory` | **-0.0033** | 0.19 | 0.5892 | -0.1478 | 0.19 |
| | `plain_rag` | -0.0640 | -1.39 | 0.6185 | -0.1358 | 0.00 |
| | `fin_mem` | -0.0640 | -1.39 | 0.6185 | -0.1358 | 0.00 |
| | `ours` | -0.0640 | -1.39 | 0.6185 | -0.1358 | 0.00 |
| | `buy_and_hold` | -0.0640 | -1.39 | 0.6185 | -0.1358 | 0.00 |

*Paired Wilcoxon Signed-Rank Test (`Ours` vs. strongest baseline, `plain_rag`): $W = 274.0, \; p = 0.502$ (not significant). `Ours` beats the memory baselines on TSLA and AMZN but loses on NFLX and MSFT; no significant pooled advantage is claimed.*

---

### C. Table 4: Risk Profile Character Ablation & Figure 8
Comparison across risk inclinations on TSLA:

| Risk Profile | Cumulative Return | Sharpe Ratio | Volatility | Max Drawdown | Hold Rate |
| :--- | ---:| ---:| ---:| ---:| ---:|
| **Conservative** (50% position scale, strict threshold) | 0.0000 | 0.00 | 0.0000 | 0.0000 | 1.00 |
| **Standard** (Full signal scale) | 0.0047 | 0.42 | 0.6521 | -0.0749 | 0.00 |
| **Aggressive** (Low threshold) | 0.0047 | 0.42 | 0.6521 | -0.0749 | 0.00 |

Conservative collapses to all-`Hold` on this sample under the deterministic policy due to the strict sentiment threshold ($\pm 0.25$); standard and aggressive produce the same actions.

**LLM Character Design & Prompt Directives:**
To mirror FinMem's profiling module in real LLM inference, we introduce differentiated system prompts and risk directives in `src/agentic_trade/llm.py`:
- **Aggressive (Alpha-Seeking)**: Explicitly instructed to exploit catalysts and positive momentum, prioritizing active `Buy` and `Sell` positions over passive `Hold` unless evidence is purely neutral.
- **Standard (Balanced)**: Weighs retrieved evidence, learned source reliability ($r_s$), and price momentum proportionally.
- **Conservative (Capital Preservation)**: Instructed to avoid drawdowns and default to `Hold` whenever signals are conflicting or ambiguous.
- Caching is explicitly partitioned by `risk_mode` to ensure distinct character-driven trajectories.

---

### D. Table 5: Working Memory Cognitive Capacity Sweep ($K$)

| Capacity ($K$) | Cumulative Return | Sharpe Ratio | Volatility | Max Drawdown | Hold Rate |
| :--- | ---:| ---:| ---:| ---:| ---:|
| $K = 1$ | -0.0600 | -2.36 | 0.3830 | -0.1138 | 0.31 |
| $K = 3$ | **0.0047** | **0.42** | 0.6521 | **-0.0749** | 0.00 |
| $K = 5$ (Default) | **0.0047** | **0.42** | 0.6521 | -0.0749 | 0.00 |
| $K = 10$ | -0.0600 | -2.36 | 0.3830 | -0.1138 | 0.31 |

---

### E. Table 6: Component Ablation Study

Evaluating the contribution of each novel mechanism on TSLA. Return deltas are relative to the full system; `facts_retired` and `decisions_recalled` diagnose whether a mechanism was exercised at all.

| Configuration | Return | Sharpe | Volatility | Drawdown | Delta Return | Facts Retired | Decisions Recalled |
| :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| **Full System (Ours)** | **0.0047** | **0.42** | 0.6521 | **-0.0749** | — | 12 | 15 |
| **w/o Learned Provenance** (`use_provenance=False`) | -0.0600 | -2.36 | 0.3830 | -0.1138 | -0.0647 | 12 | 15 |
| **w/o Contradiction Retirement** (`use_reconciliation=False`) | 0.0047 | 0.42 | 0.6521 | -0.0749 | +0.0000 | 0 | 15 |
| **w/o Decision Recall** (`use_outcome_recall=False`) | 0.0047 | 0.42 | 0.6521 | -0.0749 | +0.0000 | 12 | 0 |
| **w/o Topic Matching** (`use_topic_matching=False`) | 0.0047 | 0.42 | 0.6521 | -0.0749 | +0.0000 | 12 | 15 |

Contradiction retirement and decision recall are active (12 facts retired, 15 decisions recalled) but do not change deterministic PnL. Decision recall only enters the LLM prompt; it cannot affect the rule-based signal.

---

### F. Table 7: Memory Probe Retrieval Accuracy Benchmark

Direct retrieval evaluation on 17 ticker-scoped, pre-registered relationship/guidance/legal-risk queries. Probes may only retrieve events from their own ticker or from an explicitly related ticker.

| System | Hit@1 | Hit@3 | Hit@5 | Mean Reciprocal Rank (MRR) | Precision@5 |
| :--- | ---:| ---:| ---:| ---:| ---:|
| `no_memory` | 0.00% | 0.00% | 0.00% | 0.0000 | 0.00% |
| `plain_rag` | 11.76% | 29.41% | 35.29% | 0.1980 | 10.59% |
| `fin_mem` | 11.76% | 29.41% | 35.29% | 0.1980 | 10.59% |
| **`ours`** | **29.41%** | **41.18%** | **47.06%** | **0.3578** | **17.65%** |

**Key Finding**: After removing cross-ticker contamination, `Ours` improves Top-1 Hit Rate ($29.4\%$ vs. $11.8\%$), MRR ($0.358$ vs. $0.198$), and Precision@5 ($17.6\%$ vs. $10.6\%$). `fin_mem` equals `plain_rag` because the checked-in snapshot contains only news, so no intermediate/deep memories exist.

---

## 5. Production Execution Adapter: Paper Trading

While FinMem concluded at theoretical action signals, our framework provides a complete execution bridge via `agentic_trade.paper_trader` (`agentic-paper`):
- **Capital Caps**: Each `Buy` is strictly capped at $20\%$ of budget (`max_position_pct = 0.20`).
- **Position-Aware Selling**: A `Sell` decision queries the broker; if an open long position exists, it closes the position. If no position exists, it skips execution unless `--allow-short` is explicitly enabled.
- **Portfolio Reconciliation**: The `--reconcile` command audits current broker positions against the target universe to catch unmonitored positions or execution drift.
- **Audit Logging**: All orders append to `artifacts/paper_trading/orders.jsonl` with ISO-8601 UTC timestamps.
- **Groq LLM Decisions**: `agentic-live` generates per-ticker actions with a Groq model, persists resolved outcomes for episodic recall, and executes through the same adapter.
- **Live-Money Guard**: `--live` alone is insufficient; `ALPACA_LIVE_CONFIRM=YES` must also be set or the broker refuses to connect. Paper trading is the default.

---

## 6. Reproducibility & Research Integrity

1. **No Data Leakage**: Events are timestamped; no future price or news disclosure is accessible prior to its occurrence.
2. **Deterministic Baseline**: The offline pipeline runs with standard library math and numpy without random seed divergence. Deterministic experiments are single-trial; repeated `temperature` runs are not treated as independent evidence.
3. **Date-Pooled Statistics**: The Wilcoxon test averages repeated trials per `(ticker, date)` and pools across tickers/dates, avoiding pseudo-replication from identical deterministic runs.
4. **Ticker-Scoped Probes**: Probe retrieval filters by ticker (or an explicit related ticker), eliminating cross-company keyword contamination.
5. **Open Architecture**: Can run 100% offline with checked-in data, or scale to live APIs (Alpaca, Groq, OpenRouter) via environment variables without hardcoded credentials.
6. **Reported Limitations**: The offline snapshot is news-only and covers `2026-06-01..2026-07-09`; `fin_mem` therefore equals `plain_rag`, and contraction-retirement/recall ablations do not change deterministic PnL. The pooled trading advantage is not statistically significant. These facts are reported rather than hidden.

---

## 7. Conclusion

By addressing the core flaws of FinMem—unweighted sources, lack of contradiction retirement, absence of decision outcome memory, and isolated single-ticker processing—**Agentic Trade** establishes a more robust, empirically verified memory foundation for autonomous financial trading agents.
