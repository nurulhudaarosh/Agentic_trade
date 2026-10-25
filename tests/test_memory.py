from datetime import date

from agentic_trade.memory import MemoryEvent, OursMemory, extract_topics


def test_future_event_is_not_retrieved_early():
    memory = OursMemory()
    memory.add(MemoryEvent("future", date(2026, 9, 5), "TSLA", "news", "earnings profit"))
    assert memory.retrieve(date(2026, 9, 4), {"earnings"}) == []
    assert memory.retrieve(date(2026, 9, 5), {"earnings"})[0].event_id == "future"


def test_topics_are_deterministic():
    assert "earnings" in extract_topics("quarterly profit beat")


def test_contradiction_retires_prior_event():
    memory = OursMemory()
    memory.add(MemoryEvent("old", date(2026, 9, 1), "TSLA", "news", "earnings profit beat"))
    memory.add(MemoryEvent("new", date(2026, 9, 2), "TSLA", "news", "earnings loss miss"))
    assert [event.event_id for event in memory.retrieve(date(2026, 9, 2), {"earnings"})] == ["new"]
    assert [event.event_id for event in memory.retired_events] == ["old"]


def test_learned_source_reliability_is_applied_to_new_events():
    memory = OursMemory()
    memory.learn_source_reliability("news", 0.9)
    memory.add(MemoryEvent("news", date(2026, 9, 1), "TSLA", "news", "earnings profit"))
    assert memory.events[0].source_reliability == 0.9


def test_previous_decision_is_recalled_after_resolution():
    memory = OursMemory()
    memory.record_decision({"date": "2026-09-01", "action": "Buy", "topics": ["earnings"], "realized_return": None})
    memory.resolve(date(2026, 9, 1), 0.02)
    recalled = memory.recall(["earnings"])
    assert recalled is not None
    assert recalled["realized_return"] == 0.02
    assert memory.recalled_count == 1
