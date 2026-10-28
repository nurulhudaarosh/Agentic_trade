"""Walk-forward, point-in-time multi-ticker backtest of memory architectures.

Timeline (inclusive):
    warm-up     2025-09-02 .. 2025-09-30   memory ingestion only
    train       2025-10-01 .. 2026-02-27   online learning, ML-model fitting
    validation  2026-03-02 .. 2026-04-30   decision-threshold selection
    test        2026-05-01 .. 2026-10-26   frozen hyper-parameters, reported results

The loop is day-major (all tickers advance together) so that anything shared across
tickers - e.g. source credibility - only ever sees outcomes that were already public.
A decision taken at the close of day t earns close(t) -> close(t+1); outcomes are
resolved at the close of t+1, before the t+1 decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .memory import Credibility, MemoryConfig, MemoryStore, categorize, evidence_weight

SPLITS = {
    "warmup": (date(2025, 9, 1), date(2025, 9, 30)),
    "train": (date(2025, 10, 1), date(2026, 2, 27)),
    "validation": (date(2026, 3, 1), date(2026, 4, 30)),
    "test": (date(2026, 5, 1), date(2026, 10, 27)),
}
COMPANY = {
    "TSLA": "Tesla", "NFLX": "Netflix", "AMZN": "Amazon", "MSFT": "Microsoft", "COIN": "Coinbase",
    "NVDA": "NVIDIA", "AAPL": "Apple", "GOOGL": "Alphabet", "META": "Meta Platforms", "AMD": "Advanced Micro Devices",
}
HIT_MIN_POLARITY = 0.2


def split_of(day: date) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= day <= end:
            return name
    return "none"


@dataclass
class StudyData:
    news: pd.DataFrame
    news_emb: np.ndarray
    filings: pd.DataFrame
    filings_emb: np.ndarray
    prices: pd.DataFrame                  # wide: index date, columns tickers (+SPY, QQQ)
    tickers: list[str]
    days: list[date] = field(default_factory=list)

    @classmethod
    def load(cls, data_dir: Path, tickers: list[str]) -> "StudyData":
        news = pd.read_parquet(data_dir / "news_annotated.parquet")
        filings = pd.read_parquet(data_dir / "filings_annotated.parquet")
        bars = pd.read_parquet(data_dir / "prices.parquet")
        prices = bars.pivot_table(index="date", columns="ticker", values="close").sort_index()
        news_emb = np.load(data_dir / "news_emb.npy").astype(np.float32)
        filings_emb = np.load(data_dir / "filings_emb.npy").astype(np.float32)
        news["trade_date"] = pd.to_datetime(news["trade_date"]).dt.date
        filings["trade_date"] = pd.to_datetime(filings["trade_date"]).dt.date
        news["symbol_set"] = news["symbols"].fillna("").map(lambda s: frozenset(x for x in s.split(",") if x))
        news["category"] = news["headline"].map(categorize)
        news["row"] = np.arange(len(news))
        filings["row"] = np.arange(len(filings))
        days = [d for d in prices.index if d >= SPLITS["warmup"][0]]
        return cls(news, news_emb, filings, filings_emb, prices, tickers, days)


class _Embedder:
    _model = None

    @classmethod
    def encode(cls, texts: list[str]) -> np.ndarray:
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return cls._model.encode(texts, normalize_embeddings=True).astype(np.float32)


def task_query(ticker: str) -> str:
    # FinMem's retrieval query is the trading inquiry plus the character profile.
    return (f"Can you make an investment decision on {COMPANY.get(ticker, ticker)} ({ticker}) stock? "
            f"As an expert trading agent consider short-term news, quarterly 10-Q and annual 10-K insights.")


class PeerGraph:
    """Point-in-time co-mention graph over article symbol lists (rolling window)."""

    def __init__(self, window_days: int = 120):
        self.window = window_days
        self.log: list[tuple[int, frozenset]] = []

    def add(self, ordinal: int, symbols: frozenset) -> None:
        if 1 < len(symbols) <= 6:            # multi-name wraps carry little relational signal
            self.log.append((ordinal, symbols))

    def peers(self, ticker: str, ordinal: int, universe: set[str], k: int) -> dict[str, float]:
        counts: dict[str, float] = {}
        cutoff = ordinal - self.window
        for when, symbols in self.log:
            if when < cutoff or when >= ordinal or ticker not in symbols:
                continue
            for other in symbols:
                if other != ticker and other in universe:
                    counts[other] = counts.get(other, 0.0) + 1.0
        if not counts:
            return {}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:k]
        total = sum(c for _, c in top)
        return {name: c / total for name, c in top}


def run_systems(data: StudyData, configs: list[MemoryConfig], labels: list[str] | None = None,
                log_evidence: bool = False, include_last: bool = False,
                evidence_since: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every memory configuration over every ticker; returns (signals, evidence, stats).

    ``include_last`` also emits a decision row for the newest session (its next-day
    return is unknown, NaN) - used by the live agent service."""
    labels = labels or [c.name for c in configs]
    tickers = data.tickers
    dim = data.news_emb.shape[1]
    base_q = dict(zip(tickers, _Embedder.encode([task_query(t) for t in tickers])))
    stores = {(lab, t): MemoryStore(t, cfg, dim) for lab, cfg in zip(labels, configs) for t in tickers}
    cred = {lab: Credibility() for lab in labels}
    graph = PeerGraph()
    universe = set(tickers)

    news_by_day = {d: g for d, g in data.news.groupby("trade_date")}
    filings_by_day = {d: g for d, g in data.filings.groupby("trade_date")}
    first_day = data.days[0]
    # Filings published before the window seed intermediate/deep memory.
    early_filings = data.filings[data.filings["trade_date"] <= first_day]

    close = data.prices
    rets = close.pct_change()
    count_hist = {t: [] for t in tickers}
    rows, evidence = [], []
    pending: dict[tuple, dict] = {}     # (label, ticker) -> info about yesterday's decision

    def ingest_filing(row: pd.Series) -> None:
        layer = "deep" if row["form"] == "10-K" else "intermediate"
        t_event = pd.Timestamp(row["filed"]).toordinal()
        for lab, cfg in zip(labels, configs):
            stores[(lab, row["ticker"])].add(row["id"], t_event, layer, data.filings_emb[row["row"]],
                                              float(row["sentiment"]), "sec", "filing", attention=1.0)

    for _, row in early_filings.iterrows():
        if row["ticker"] in universe:
            ingest_filing(row)

    for di, day in enumerate(data.days):
        now = day.toordinal()
        todays = news_by_day.get(day)
        # ---- 1. resolve yesterday's decisions with today's close (now public) -------------
        if di > 0:
            spy_r = float(rets.at[day, "SPY"]) if "SPY" in rets else 0.0
            for (lab, t), info in pending.items():
                r = float(rets.at[day, t])
                ar = r - spy_r
                store = stores[(lab, t)]
                cfg = store.config
                if info["idx"].size:
                    helpful = (np.sign(store.sent[info["idx"]]) == np.sign(r)) & (np.abs(store.sent[info["idx"]]) >= HIT_MIN_POLARITY)
                    top3 = np.zeros(info["idx"].size, dtype=bool); top3[:3] = True
                    store.reinforce(info["idx"], helpful & top3, now)
                if cfg.credibility:
                    for author, category, s in info["observed"]:
                        if abs(s) >= HIT_MIN_POLARITY and ar != 0:
                            cred[lab].update(author, category, bool(np.sign(s) == np.sign(ar)))
                store.remember_episode(info["context"], info["signal"], ar)
            pending = {}
        # ---- 2. ingest today's information ----------------------------------------------
        if todays is not None:
            for symbols in todays["symbol_set"]:
                graph.add(now, symbols)
        for _, row in filings_by_day.get(day, pd.DataFrame()).iterrows():
            if row["ticker"] in universe and row["trade_date"] > first_day:
                ingest_filing(row)
        own_today: dict[str, pd.DataFrame] = {}
        for t in tickers:
            g = todays[todays["symbol_set"].map(lambda s: t in s)] if todays is not None else None
            own_today[t] = g if g is not None else pd.DataFrame()
            n_today = len(own_today[t])
            hist = count_hist[t][-20:]
            attention = (n_today + 1.0) / (np.mean(hist) + 1.0) if hist else 1.0
            count_hist[t].append(n_today)
            if n_today:
                for _, item in own_today[t].iterrows():
                    emb = data.news_emb[item["row"]]
                    for lab in labels:
                        stores[(lab, t)].add(str(item["id"]), now, "shallow", emb, float(item["sentiment"]),
                                             str(item["author"]), item["category"], attention)
            for lab in labels:
                stores[(lab, t)].maintain(now)
        if split_of(day) == "warmup" or (di + 1 >= len(data.days) and not include_last):
            continue
        next_day = data.days[di + 1] if di + 1 < len(data.days) else None
        # ---- 3. retrieve, aggregate, record ----------------------------------------------
        for t in tickers:
            today_items = own_today[t]
            today_emb = data.news_emb[today_items["row"].to_numpy()] if len(today_items) else None
            observed = list(zip(today_items["author"].astype(str), today_items["category"],
                                today_items["sentiment"].astype(float))) if len(today_items) else []
            day_sent = float(today_items["sentiment"].mean()) if len(today_items) else 0.0
            for lab, cfg in zip(labels, configs):
                store = stores[(lab, t)]
                q = base_q[t]
                if cfg.situational and today_emb is not None:
                    q = q + today_emb.mean(axis=0)
                    q = q / (np.linalg.norm(q) + 1e-9)
                ret = store.retrieve(now, q, cred[lab] if cfg.credibility else None)
                sents = store.sent[ret.idx]
                num = float(np.sum(ret.weight * sents))
                den = float(np.sum(np.abs(ret.weight)))
                peer_signal, peer_n = 0.0, 0
                if cfg.peers:
                    peers = graph.peers(t, now, universe, cfg.peer_k)
                    for p, w in peers.items():
                        pg = own_today.get(p)
                        if pg is None or not len(pg):
                            continue
                        pg = pg[~pg["symbol_set"].map(lambda s: t in s)]
                        if not len(pg):
                            continue
                        pe = data.news_emb[pg["row"].to_numpy()] @ q
                        top = np.argsort(-pe)[: cfg.peer_k]
                        ps = pg["sentiment"].to_numpy()[top]
                        pc = np.array([cred[lab].score(a, c) for a, c in zip(pg["author"].astype(str).to_numpy()[top], pg["category"].to_numpy()[top])]) if cfg.credibility else np.full(top.size, 0.5)
                        pw = 0.5 * w * evidence_weight(pc, cfg)
                        num += float(np.sum(pw * ps)); den += float(np.sum(np.abs(pw)))
                        peer_signal += float(np.sum(pw * ps)); peer_n += int(top.size)
                signal = num / den if den > 0 else 0.0
                if ret.idx.size:
                    w = np.abs(ret.weight) / (np.abs(ret.weight).sum() + 1e-9)
                    context = (store.emb[ret.idx] * w[:, None]).sum(axis=0)
                    context = context / (np.linalg.norm(context) + 1e-9)
                else:
                    context = q
                edge, n_ep = store.recall(context)
                abstain = edge is not None and edge < 0
                pending[(lab, t)] = {"idx": ret.idx.copy(), "observed": observed if cfg.credibility else [],
                                     "context": context, "signal": signal}
                layer_counts = np.bincount(ret.layer, minlength=3) if ret.idx.size else np.zeros(3, int)
                rows.append({
                    "system": lab, "ticker": t, "date": day, "next_date": next_day, "split": split_of(day),
                    "signal": signal, "abstain": abstain, "episodic_edge": edge, "episodic_n": n_ep,
                    "peer_signal": peer_signal, "peer_n": peer_n, "day_sentiment": day_sent,
                    "n_today": len(today_items), "n_retrieved": int(ret.idx.size),
                    "n_shallow": int(layer_counts[0]), "n_intermediate": int(layer_counts[1]), "n_deep": int(layer_counts[2]),
                    "mean_age": float(np.mean(now - store.t_event[ret.idx])) if ret.idx.size else np.nan,
                    "redundancy": _redundancy(store.emb[ret.idx[ret.layer == 0]]) if ret.idx.size else np.nan,
                    "conflict": _conflict(sents),
                    "mean_cred": float(np.mean(ret.cred)) if cfg.credibility and ret.idx.size else np.nan,
                    "active_items": int(store.active[: store.n].sum()),
                    "next_return": float(rets.at[next_day, t]) if next_day else np.nan,
                    "next_spy": float(rets.at[next_day, "SPY"]) if next_day else np.nan,
                })
                if log_evidence and ret.idx.size and (evidence_since is None or day >= evidence_since):
                    for rank, (i, lay, sc, wt, cr) in enumerate(zip(ret.idx, ret.layer, ret.score, ret.weight, ret.cred), 1):
                        evidence.append({"system": lab, "ticker": t, "date": day, "rank": rank, "item_id": store.item_id[i],
                                         "layer": int(lay), "score": float(sc), "weight": float(wt), "cred": float(cr),
                                         "sentiment": float(store.sent[i]), "category": store.category[i],
                                         "author": store.author[i], "age_days": int(now - store.t_event[i]),
                                         "corroboration": float(store.corrob[i])})
    stats = {lab: {"retired": sum(stores[(lab, t)].retired for t in tickers),
                   "merged": sum(stores[(lab, t)].merged for t in tickers),
                   "promoted": sum(stores[(lab, t)].promoted for t in tickers),
                   "purged": sum(stores[(lab, t)].purged for t in tickers),
                   "items": sum(stores[(lab, t)].n for t in tickers)} for lab in labels}
    stats["_credibility"] = {lab: _cred_table(cred[lab]) for lab in labels if cred[lab].counts}
    return pd.DataFrame(rows), pd.DataFrame(evidence), stats


def _redundancy(emb: np.ndarray) -> float:
    """Mean pairwise cosine similarity of retrieved shallow memories (higher = more duplicated)."""
    if emb.shape[0] < 2:
        return np.nan
    sims = emb @ emb.T
    iu = np.triu_indices(emb.shape[0], 1)
    return float(sims[iu].mean())


def _conflict(sents: np.ndarray) -> float:
    polar = sents[np.abs(sents) >= HIT_MIN_POLARITY]
    if polar.size < 2:
        return 0.0
    return float(min((polar > 0).mean(), (polar < 0).mean()) * 2)


def _cred_table(c: Credibility) -> list[dict]:
    out = []
    for key, n in c.counts.items():
        if key[0] in ("g", "c"):
            out.append({"level": key[0], "name": key[1] if len(key) > 1 else "global", "n": n,
                        "hit_rate": c.hits[key] / n, "posterior": c._post(key, 0.5)})
    return out


# ------------------------------------------------------------------ trading rules
def positions_from_signal(frame: pd.DataFrame, threshold: float | pd.Series, long_only: bool = False,
                          respect_abstain: bool = True) -> pd.Series:
    """FinMem action semantics: Buy -> long, Sell -> short (or flat), Hold -> keep position.

    ``threshold`` may be a scalar or a per-row Series (rolling re-selection)."""
    out = pd.Series(0.0, index=frame.index)
    th_all = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=frame.index)
    for _, g in frame.sort_values("date").groupby("ticker", sort=False):
        pos = 0.0
        for i, s, ab, th in zip(g.index, g["signal"].to_numpy(), g["abstain"].to_numpy(), th_all.loc[g.index].to_numpy()):
            if respect_abstain and ab:
                action = 0
            else:
                action = 1 if s > th else -1 if s < -th else 0
            if action == 1:
                pos = 1.0
            elif action == -1:
                pos = 0.0 if long_only else -1.0
            out.at[i] = pos
    return out


def apply_costs(frame: pd.DataFrame, position_col: str = "position", cost_bps: float = 5.0) -> pd.Series:
    """Daily strategy return net of a proportional cost on position changes."""
    net = pd.Series(0.0, index=frame.index)
    for _, g in frame.sort_values("date").groupby("ticker", sort=False):
        pos = g[position_col].to_numpy()
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net.loc[g.index] = pos * g["next_return"].to_numpy() - turnover * cost_bps / 1e4
    return net
