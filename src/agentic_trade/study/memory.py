"""Memory architectures for the study.

Three families share one vectorised store so that every difference in behaviour is
caused by an explicit, ablatable design choice:

``plain_rag``  Flat vector memory (Generative-Agents style): one pool, one decay,
               score = recency + relevancy, top-K overall.
``fin_mem``    Faithful FinMem (Yu et al., 2023): shallow/intermediate/deep layers
               with Q = 14/90/365, importance v ~ {40,60,80} sampled with layer-specific
               probabilities and decayed by alpha_l ** delta, purge rules, access-counter
               reinforcement (+5) and promotion to deeper layers, top-K per layer.
``ours``       FinMem plus seven components (each switchable for ablation):

  N1 ``credibility``   hierarchical Beta-Bernoulli source credibility learned online
                       from market-adjusted outcomes (author -> category -> global).
  N2 ``dedup``         near-duplicate consolidation: a re-reported story corroborates
                       the stored memory instead of occupying another retrieval slot.
  N3 ``contradiction`` semantic contradiction retirement: a newer, similar item with
                       opposite polarity retires the stale memory.
  N4 ``situational``   situational query: the retrieval query is conditioned on the
                       day's own news, not only on a fixed task prompt.
  N5 ``importance``    data-driven importance (attention surprise x classifier
                       confidence x corroboration) instead of FinMem's random draw.
  N6 ``peers``         relational memory: evidence from co-mentioned firms, with edge
                       weights learned point-in-time from article co-mentions.
  N7 ``episodic``      outcome-aware episodic memory: k-NN recall of past decision
                       contexts and their realised returns; abstain when similar
                       situations were historically mis-signalled.

``signed`` (post-hoc variant of N1, added after the first test run and reported as
such): adaptive evidence polarity. Credibility below 0.5 means a source type has been
followed by the *opposite* market-adjusted move (over-reaction), so its evidence is
counted with a negative weight instead of being discarded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

LAYERS = ("shallow", "intermediate", "deep")
LAYER_ID = {name: i for i, name in enumerate(LAYERS)}
Q = np.array([14.0, 90.0, 365.0])
ALPHA = np.array([0.9, 0.967, 0.988])
V_LEVELS = np.array([40.0, 60.0, 80.0])
V_PROBS = np.array([[0.8, 0.15, 0.05], [0.05, 0.8, 0.15], [0.05, 0.15, 0.8]])
PROMOTE_AT = np.array([60.0, 80.0, np.inf])  # importance (pre-scaling) needed to move deeper
ACCESS_BONUS = 5.0
PURGE_RECENCY = 0.05
PURGE_IMPORTANCE = 5.0

CATEGORY_PATTERNS = [
    ("analyst", r"price target|upgrade|downgrade|maintains|initiates|reiterat|analyst|overweight|underweight|outperform|neutral rating|buy rating|sell rating"),
    ("earnings", r"earnings|\beps\b|revenue|quarter|\bq[1-4]\b|guidance|outlook|results|beats|misses|forecast"),
    ("options", r"options|unusual|whale|call activity|put activity"),
    ("insider", r"insider|13f|stake|sells? shares|buys? shares|form 4|holdings|position in"),
    ("legal", r"lawsuit|\bsec\b|probe|investigation|antitrust|\bftc\b|\bdoj\b|court|regulator|fine[sd]?\b|recall"),
    ("market_wrap", r"stocks? (moving|to watch)|movers|pre-market|premarket|after-hours|mid-day|midday|futures|nasdaq|s&p|dow |market (update|wrap)|trading (higher|lower)|shares are (trading|down|up)"),
    ("corporate", r"announces|launch|partner|deal|contract|acquir|merger|unveil|deliver|production|ceo|executive"),
]
CATEGORIES = [name for name, _ in CATEGORY_PATTERNS] + ["other", "filing"]


def categorize(headline: str) -> str:
    lowered = headline.lower()
    for name, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return "other"


@dataclass
class MemoryConfig:
    name: str = "fin_mem"
    layered: bool = True
    top_k: int = 5                     # per layer when layered, overall otherwise
    flat_q: float = 14.0
    random_importance: bool = True
    access_counter: bool = True
    purge: bool = True
    credibility: bool = False
    dedup: bool = False
    contradiction: bool = False
    situational: bool = False
    importance: bool = False
    peers: bool = False
    episodic: bool = False
    signed: bool = False
    dedup_sim: float = 0.88
    contradiction_sim: float = 0.72
    polarity_min: float = 0.3
    peer_k: int = 3
    episodic_k: int = 10
    episodic_min: int = 5
    seed: int = 7

    @classmethod
    def plain_rag(cls, top_k: int = 5) -> "MemoryConfig":
        return cls(name="plain_rag", layered=False, top_k=top_k, random_importance=False,
                   access_counter=False, purge=False)

    @classmethod
    def fin_mem(cls, top_k: int = 5) -> "MemoryConfig":
        return cls(name="fin_mem", top_k=top_k)

    @classmethod
    def ours(cls, top_k: int = 5, **off: bool) -> "MemoryConfig":
        flags = dict(credibility=True, dedup=True, contradiction=True, situational=True,
                     importance=True, peers=True, episodic=True, random_importance=False)
        for key, value in off.items():
            flags[key] = value
        if not flags["importance"]:
            flags["random_importance"] = True
        return cls(name="ours", top_k=top_k, **flags)


def evidence_weight(cred: np.ndarray, config: "MemoryConfig") -> np.ndarray:
    """Aggregation weight of evidence given its source credibility."""
    if not config.credibility:
        return np.ones(np.shape(cred))
    if config.signed:
        return 2.0 * (np.asarray(cred) - 0.5)
    return np.clip(2.0 * (np.asarray(cred) - 0.5), 0.0, 1.0) + 0.05


class Credibility:
    """Hierarchical Beta-Bernoulli credibility: author -> category -> global prior."""

    def __init__(self, strength: float = 8.0):
        self.strength = strength
        self.hits: dict[tuple, float] = {}
        self.counts: dict[tuple, float] = {}

    def update(self, author: str, category: str, hit: bool) -> None:
        for key in (("g",), ("c", category), ("a", author)):
            self.hits[key] = self.hits.get(key, 0.0) + float(hit)
            self.counts[key] = self.counts.get(key, 0.0) + 1.0

    def _post(self, key: tuple, prior: float) -> float:
        n = self.counts.get(key, 0.0)
        return (self.hits.get(key, 0.0) + self.strength * prior) / (n + self.strength)

    def score(self, author: str, category: str) -> float:
        g = self._post(("g",), 0.5)
        c = self._post(("c", category), g)
        return self._post(("a", author), c) if author else c


@dataclass
class Retrieval:
    idx: np.ndarray                     # store indices, ordered by score within layer
    layer: np.ndarray
    score: np.ndarray
    weight: np.ndarray                  # evidence weight used for aggregation
    cred: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peer_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    peer_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    abstain: bool = False
    episodic_edge: float | None = None
    episodic_n: int = 0


class MemoryStore:
    """Per-ticker memory. Items are appended in availability order."""

    def __init__(self, ticker: str, config: MemoryConfig, dim: int):
        self.ticker = ticker
        self.config = config
        self.rng = np.random.default_rng(config.seed + sum(map(ord, ticker)))
        cap = 4096
        self.n = 0
        self.emb = np.zeros((cap, dim), dtype=np.float32)
        self.t_event = np.zeros(cap)        # day ordinal used for recency/decay
        self.layer = np.zeros(cap, dtype=int)
        self.v = np.zeros(cap)              # importance base (pre-scaling)
        self.bonus = np.zeros(cap)          # access-counter reinforcement
        self.t_reset = np.zeros(cap)        # recency reset on promotion
        self.sent = np.zeros(cap)
        self.active = np.zeros(cap, dtype=bool)
        self.corrob = np.ones(cap)
        self.item_id = np.empty(cap, dtype=object)
        self.author = np.empty(cap, dtype=object)
        self.category = np.empty(cap, dtype=object)
        self.retired = 0
        self.merged = 0
        self.promoted = 0
        self.purged = 0
        self.episodes: list[tuple[np.ndarray, float, float]] = []   # (context, signal, outcome)

    def _grow(self) -> None:
        for name in ("emb", "t_event", "layer", "v", "bonus", "t_reset", "sent", "active", "corrob",
                     "item_id", "author", "category"):
            arr = getattr(self, name)
            new = np.zeros((arr.shape[0] * 2,) + arr.shape[1:], dtype=arr.dtype) if arr.dtype != object else np.empty(arr.shape[0] * 2, dtype=object)
            if name == "corrob":
                new[:] = 1.0
            new[: arr.shape[0]] = arr
            setattr(self, name, new)

    # ----------------------------------------------------------------- write
    def add(self, item_id: str, t: float, layer: str, emb: np.ndarray, sentiment: float,
            author: str, category: str, attention: float = 1.0) -> None:
        cfg = self.config
        lid = LAYER_ID[layer] if cfg.layered else 0
        live = np.flatnonzero(self.active[: self.n] & (self.layer[: self.n] == lid))
        if live.size and (cfg.dedup or cfg.contradiction):
            sims = self.emb[live] @ emb.astype(np.float32)
            if cfg.dedup:
                dup = live[(sims >= cfg.dedup_sim) & (np.sign(self.sent[live]) == np.sign(sentiment))]
                if dup.size:
                    j = dup[np.argmax(self.emb[dup] @ emb.astype(np.float32))]
                    self.corrob[j] += 1.0
                    self.t_event[j] = t          # the story is still current
                    self.merged += 1
                    return
            if cfg.contradiction and abs(sentiment) >= cfg.polarity_min:
                stale = live[(sims >= cfg.contradiction_sim)
                             & (self.sent[live] * sentiment < 0)
                             & (np.abs(self.sent[live]) >= cfg.polarity_min)
                             & (self.t_event[live] <= t)]
                if stale.size:
                    self.active[stale] = False
                    self.retired += int(stale.size)
        if self.n == self.emb.shape[0]:
            self._grow()
        i = self.n
        self.emb[i] = emb
        self.t_event[i] = t
        self.t_reset[i] = t
        self.layer[i] = lid
        if cfg.importance:
            # Attention surprise x classifier confidence, mapped onto FinMem's 40..80 range.
            self.v[i] = 40.0 + 40.0 * float(np.clip(0.5 * abs(sentiment) + 0.5 * np.tanh(attention - 1.0), 0, 1)) + (20.0 * lid)
        elif cfg.random_importance:
            self.v[i] = self.rng.choice(V_LEVELS, p=V_PROBS[lid])
        else:
            self.v[i] = 0.0
        self.sent[i] = sentiment
        self.active[i] = True
        self.item_id[i] = item_id
        self.author[i] = author
        self.category[i] = category
        self.n += 1

    # ------------------------------------------------------------------ read
    def _importance(self, idx: np.ndarray, now: float) -> np.ndarray:
        delta = np.maximum(now - self.t_reset[idx], 0)
        alpha = ALPHA[self.layer[idx]] if self.config.layered else ALPHA[0]
        corrob = np.log1p(self.corrob[idx] - 1.0) * 10.0 if self.config.dedup else 0.0
        return (self.v[idx] + self.bonus[idx] + corrob) * np.power(alpha, delta)

    def _recency(self, idx: np.ndarray, now: float) -> np.ndarray:
        delta = np.maximum(now - self.t_reset[idx], 0)
        q = Q[self.layer[idx]] if self.config.layered else self.config.flat_q
        return np.exp(-delta / q)

    def maintain(self, now: float) -> None:
        """FinMem purge rule: forget when recency < 0.05 or importance < 5."""
        if not self.config.purge:
            return
        idx = np.flatnonzero(self.active[: self.n])
        if not idx.size:
            return
        dead = (self._recency(idx, now) < PURGE_RECENCY) | (self._importance(idx, now) < PURGE_IMPORTANCE)
        # Filings are the only deep/intermediate source; FinMem keeps them a full cycle.
        dead &= self.layer[idx] == 0
        self.active[idx[dead]] = False
        self.purged += int(dead.sum())

    def retrieve(self, now: float, query: np.ndarray, credibility: Credibility | None = None) -> Retrieval:
        cfg = self.config
        idx = np.flatnonzero(self.active[: self.n] & (self.t_event[: self.n] <= now))
        if not idx.size:
            return Retrieval(np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
        rec = self._recency(idx, now)
        rel = self.emb[idx] @ query
        imp = np.minimum(self._importance(idx, now) / 100.0, 1.0) if (cfg.random_importance or cfg.importance) else 0.0
        score = rec + rel + imp
        cred = np.full(idx.size, 0.5)
        if cfg.credibility and credibility is not None:
            cred = np.array([credibility.score(a, c) for a, c in zip(self.author[idx], self.category[idx])])
            # Signed mode ranks by informativeness |cred - 0.5|; otherwise by reliability.
            score = score + (np.abs(cred - 0.5) * 4.0 if cfg.signed else (cred - 0.5) * 2.0)
        chosen, layers, scores = [], [], []
        groups = range(3) if cfg.layered else [0]
        for lid in groups:
            mask = self.layer[idx] == lid
            if not mask.any():
                continue
            local = np.flatnonzero(mask)
            order = local[np.argsort(-score[local], kind="stable")[: cfg.top_k]]
            chosen.append(idx[order]); layers.append(np.full(order.size, lid)); scores.append(score[order])
        if not chosen:
            return Retrieval(np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
        sel = np.concatenate(chosen)
        cred_sel = cred[np.searchsorted(idx, sel)]
        # Baselines weight retrieved evidence equally (an LLM reading the context);
        # with credibility, evidence counts in proportion to its learned reliability.
        weight = evidence_weight(cred_sel, cfg)
        if cfg.dedup:
            weight = weight * (1.0 + np.log1p(self.corrob[sel] - 1.0))
        return Retrieval(sel, np.concatenate(layers), np.concatenate(scores), weight.astype(float), cred_sel)

    def reinforce(self, idx: np.ndarray, helpful: np.ndarray, now: float) -> None:
        """Access counter: pivotal memories gain +5 importance and may be promoted."""
        if not self.config.access_counter or not self.config.layered:
            return
        for i, good in zip(idx, helpful):
            if not good:
                continue
            self.bonus[i] += ACCESS_BONUS
            lid = self.layer[i]
            if self.v[i] + self.bonus[i] >= PROMOTE_AT[lid] and lid < 2:
                self.layer[i] = lid + 1
                self.t_reset[i] = now        # recency resets to 1.0 on promotion
                self.promoted += 1

    # --------------------------------------------------------------- episodes
    def remember_episode(self, context: np.ndarray, signal: float, outcome: float) -> None:
        if self.config.episodic and signal != 0:
            self.episodes.append((context.astype(np.float32), float(signal), float(outcome)))

    def recall(self, context: np.ndarray) -> tuple[float | None, int]:
        """Similarity-weighted signed edge of the signal in the k most similar past contexts."""
        if not self.config.episodic or len(self.episodes) < self.config.episodic_min:
            return None, 0
        contexts = np.stack([e[0] for e in self.episodes])
        sims = contexts @ context.astype(np.float32)
        top = np.argsort(-sims)[: self.config.episodic_k]
        top = top[sims[top] > 0.3]
        if top.size < self.config.episodic_min:
            return None, int(top.size)
        signals = np.array([np.sign(self.episodes[i][1]) for i in top])
        outcomes = np.array([self.episodes[i][2] for i in top])
        w = sims[top]
        return float(np.sum(w * signals * outcomes) / np.sum(w)), int(top.size)
