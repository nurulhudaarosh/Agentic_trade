from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

TOPICS = {
    "earnings": ("earnings", "eps", "profit", "loss", "quarterly", "revenue"),
    "guidance": ("guidance", "outlook", "forecast", "raised", "lowered"),
    "legal_risk": ("legal", "lawsuit", "regulation", "compliance", "risk"),
    "mergers_acquisitions": ("merger", "acquisition", "buyout", "takeover", "spac"),
    "capital_return": ("dividend", "buyback", "shareholder"),
}
POSITIVE = (
    "beat", "beats", "strong", "growth", "grow", "upgrade", "raised", "raise", "profit",
    "positive", "surge", "approval", "record", "rally", "jump", "soar", "gain", "wins",
    "partnership", "expansion", "buyback", "dividend", "outperform", "bullish", "tops",
)
NEGATIVE = (
    "miss", "misses", "weak", "downgrade", "lowered", "cut", "cuts", "loss", "negative",
    "lawsuit", "sued", "risk", "fall", "falls", "drop", "plunge", "slump", "decline",
    "probe", "investigation", "recall", "halt", "delay", "warning", "fraud", "penalty",
    "bearish", "selloff", "crash",
)

PUBLISHERS = (
    "Bloomberg",
    "Reuters",
    "Wall Street Journal",
    "WSJ",
    "Benzinga",
    "CNBC",
    "Barron's",
    "MarketWatch",
    "Financial Times",
    "The Motley Fool",
    "Seeking Alpha",
    "Insider Monkey",
    "Business Wire",
    "GlobeNewswire",
    "PR Newswire",
    "Investor's Business Daily",
    "Yahoo Finance",
    "TheStreet",
    "TipRanks",
    "Zacks",
    "Dow Jones",
    "TechCrunch",
)


PRESS_RELEASE_MARKERS = (
    "announces",
    "announced",
    "declares",
    "to participate",
    "to present",
    "will host",
    "conference call",
    "webcast",
    "schedules",
    "files for",
    "files s-1",
    "files an s-1",
)
ANALYST_MARKERS = (
    "analyst",
    "price target",
    "upgrade",
    "downgrade",
    "initiates coverage",
    "rating",
    "maintains",
)


def infer_source_name(text: str) -> str:
    """Best-effort publisher attribution from an article headline/summary.

    Alpaca's news feed normally provides a source field; this fallback keeps the
    checked-in offline data usable for source-level provenance learning.
    """
    lowered = text.lower()
    for publisher in sorted(PUBLISHERS, key=len, reverse=True):
        if publisher.lower() in lowered:
            return publisher
    return "unknown"


def infer_source_category(text: str) -> str:
    """Coarse source genre used when publisher attribution is unavailable."""
    lowered = text.lower()
    if any(marker in lowered for marker in PRESS_RELEASE_MARKERS):
        return "press_release"
    if any(marker in lowered for marker in ANALYST_MARKERS):
        return "analyst_note"
    return "media"


def event_matches_ticker(event: "MemoryEvent", ticker: str | None) -> bool:
    if ticker is None:
        return True
    return event.ticker == ticker or ticker in event.related_tickers


@dataclass
class MemoryEvent:
    event_id: str
    event_date: date
    ticker: str
    source_type: str
    text: str
    summary: str = ""
    sentiment: float = 0.0
    topic_tags: set[str] = field(default_factory=set)
    related_tickers: set[str] = field(default_factory=set)
    source_reliability: float = 0.7
    layer: str = "shallow"
    active: bool = True
    source_name: str = ""
    source_category: str = ""

    def __post_init__(self) -> None:
        self.layer = {"10-Q": "intermediate", "10-K": "deep"}.get(self.source_type, "shallow")
        if not self.topic_tags:
            self.topic_tags = extract_topics(f"{self.text} {self.summary}")
        if self.sentiment == 0.0:
            self.sentiment = infer_sentiment(f"{self.text} {self.summary}")
        if not self.source_name:
            self.source_name = infer_source_name(f"{self.text} {self.summary}")
        if not self.source_category:
            self.source_category = infer_source_category(f"{self.text} {self.summary}")
        self.source_reliability = {"news": 0.7, "10-Q": 0.9, "10-K": 1.0}.get(self.source_type, 0.5)


def extract_topics(text: str) -> set[str]:
    lowered = text.lower()
    topics = {topic for topic, words in TOPICS.items() if any(word in lowered for word in words)}
    return topics or {"market_news"}


def infer_sentiment(text: str) -> float:
    lowered = text.lower()
    positive = sum(word in lowered for word in POSITIVE)
    negative = sum(word in lowered for word in NEGATIVE)
    if positive == negative:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 1)))


class MemorySystem:
    name = "base"
    def __init__(self) -> None:
        self.events: list[MemoryEvent] = []

    def add(self, event: MemoryEvent) -> None:
        self.events.append(event)

    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5, ticker: str | None = None) -> list[MemoryEvent]:
        visible = [
            event for event in self.events
            if event.active and event.event_date <= as_of and event_matches_ticker(event, ticker)
        ]
        return sorted(visible, key=lambda event: (event.event_date, event.source_reliability), reverse=True)[:top_k]


class NoMemory(MemorySystem):
    name = "no_memory"
    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5, ticker: str | None = None) -> list[MemoryEvent]:
        return []


class PlainRAG(MemorySystem):
    name = "plain_rag"


class FinMem(MemorySystem):
    name = "fin_mem"
    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5, ticker: str | None = None) -> list[MemoryEvent]:
        visible = [
            event for event in self.events
            if event.active and event.event_date <= as_of and event_matches_ticker(event, ticker)
        ]
        return sorted(
            visible,
            key=lambda event: (event.event_date, event.source_reliability, event.layer == "deep"),
            reverse=True,
        )[:top_k]


class OursMemory(FinMem):
    name = "ours"
    def __init__(self) -> None:
        super().__init__()
        self.retired_events: list[MemoryEvent] = []
        self.decision_history: list[dict] = []
        self.recalled_count = 0
        self.source_reliability: dict[str, float] = {}

    def learn_source_reliability(self, source_type: str, hit_rate: float) -> None:
        self.source_reliability[source_type] = max(0.5, min(1.0, hit_rate))

    def add(self, event: MemoryEvent) -> None:
        learned = (
            self.source_reliability.get(event.source_name)
            or self.source_reliability.get(event.source_category)
            or self.source_reliability.get(event.source_type)
        )
        if learned is not None:
            event.source_reliability = learned
        for old in self.events:
            if old.active and old.event_date <= event.event_date and old.layer == event.layer:
                if old.topic_tags & event.topic_tags and old.sentiment * event.sentiment < 0:
                    old.active = False
                    self.retired_events.append(old)
        super().add(event)

    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5, ticker: str | None = None) -> list[MemoryEvent]:
        allowed = set(topics)
        visible = [
            event for event in self.events
            if event.active and event.event_date <= as_of and event_matches_ticker(event, ticker)
        ]
        def score(event: MemoryEvent) -> tuple[float, date]:
            topic_score = len(event.topic_tags & allowed) / max(len(allowed), 1)
            relationship_score = (
                0.1
                if ticker and event.ticker != ticker and ticker in event.related_tickers
                else 0.0
            )
            return ((event.source_reliability + topic_score + relationship_score + event.sentiment * 0.05), event.event_date)
        return sorted(visible, key=score, reverse=True)[:top_k]

    def record_decision(self, record: dict) -> None:
        self.decision_history.append(record)

    def resolve(self, decision_date: date, realized_return: float) -> None:
        for record in reversed(self.decision_history):
            if record["date"] == decision_date.isoformat() and record.get("realized_return") is None:
                record["realized_return"] = realized_return
                return

    def recall(self, topics: Iterable[str]) -> dict | None:
        allowed = set(topics)
        for record in reversed(self.decision_history):
            if record.get("realized_return") is not None and set(record.get("topics", [])) & allowed:
                self.recalled_count += 1
                return record
        return None
