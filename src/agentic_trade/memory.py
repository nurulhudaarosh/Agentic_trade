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
POSITIVE = ("beat", "strong", "growth", "upgrade", "raised", "profit", "positive", "surge", "approval")
NEGATIVE = ("miss", "weak", "downgrade", "lowered", "loss", "negative", "lawsuit", "risk", "fall")

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

    def __post_init__(self) -> None:
        self.layer = {"10-Q": "intermediate", "10-K": "deep"}.get(self.source_type, "shallow")
        if not self.topic_tags:
            self.topic_tags = extract_topics(f"{self.text} {self.summary}")
        if self.sentiment == 0.0:
            self.sentiment = infer_sentiment(f"{self.text} {self.summary}")
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

    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5) -> list[MemoryEvent]:
        allowed = set(topics)
        visible = [event for event in self.events if event.active and event.event_date <= as_of]
        return sorted(visible, key=lambda event: (event.event_date, event.source_reliability), reverse=True)[:top_k]


class NoMemory(MemorySystem):
    name = "no_memory"
    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5) -> list[MemoryEvent]:
        return []


class PlainRAG(MemorySystem):
    name = "plain_rag"


class FinMem(MemorySystem):
    name = "fin_mem"
    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5) -> list[MemoryEvent]:
        allowed = set(topics)
        visible = [event for event in self.events if event.active and event.event_date <= as_of]
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
        if event.source_type in self.source_reliability:
            event.source_reliability = self.source_reliability[event.source_type]
        for old in self.events:
            if old.active and old.event_date <= event.event_date and old.layer == event.layer:
                if old.topic_tags & event.topic_tags and old.sentiment * event.sentiment < 0:
                    old.active = False
                    self.retired_events.append(old)
        super().add(event)

    def retrieve(self, as_of: date, topics: Iterable[str] = (), top_k: int = 5) -> list[MemoryEvent]:
        allowed = set(topics)
        visible = [event for event in self.events if event.active and event.event_date <= as_of]
        def score(event: MemoryEvent) -> tuple[float, date]:
            topic_score = len(event.topic_tags & allowed) / max(len(allowed), 1)
            relationship_score = 0.1 if event.related_tickers else 0.0
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
