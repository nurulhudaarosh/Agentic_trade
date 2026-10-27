from datetime import date

from agentic_trade.memory import MemoryEvent, OursMemory, extract_topics, infer_source_category


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


def test_source_category_inference():
    assert infer_source_category("Company announces quarterly results") == "press_release"
    assert infer_source_category("Analyst raises price target") == "analyst_note"
    assert infer_source_category("Tesla shares rally on demand") == "media"


def test_relationship_bonus_only_applies_to_related_tickers():
    memory = OursMemory()
    memory.add(MemoryEvent("own", date(2026, 9, 1), "TSLA", "news", "earnings profit beat", related_tickers={"MSFT"}))
    memory.add(MemoryEvent("peer", date(2026, 9, 1), "MSFT", "news", "earnings profit beat", related_tickers={"TSLA"}))
    own = memory.retrieve(date(2026, 9, 1), {"earnings"}, top_k=1, ticker="TSLA")
    assert own[0].ticker == "MSFT"  # peer receives the relationship bonus
    assert memory.retrieve(date(2026, 9, 1), {"earnings"}, top_k=2, ticker="COIN") == []


def test_previous_decision_is_recalled_after_resolution():
    memory = OursMemory()
    memory.record_decision({"date": "2026-09-01", "action": "Buy", "topics": ["earnings"], "realized_return": None})
    memory.resolve(date(2026, 9, 1), 0.02)
    recalled = memory.recall(["earnings"])
    assert recalled is not None
    assert recalled["realized_return"] == 0.02
    assert memory.recalled_count == 1
