import pytest

from agentic_trade.agent_service import STUDY_DIR, AgentService


def ours(action="Buy", signal=0.4, abstain=False, threshold=0.2):
    return {"action": action, "signal": signal, "abstain": abstain, "threshold": threshold}


def test_consensus_agreement_raises_confidence():
    c = AgentService._consensus(ours("Buy"), 0.6, None)
    assert c["action"] == "Buy" and c["agreement"] and c["confidence"] == 0.6


def test_consensus_disagreement_holds():
    c = AgentService._consensus(ours("Buy"), 0.40, None)
    assert c["action"] == "Hold" and not c["agreement"]


def test_episodic_abstention_overrides_price_model():
    c = AgentService._consensus(ours("Sell", -0.5, abstain=True), 0.9, None)
    assert c["action"] == "Hold" and c["confidence"] == 0.3


def test_weak_price_model_does_not_vote():
    c = AgentService._consensus(ours("Hold", 0.0), 0.51, None)
    assert c["action"] == "Hold"
    c = AgentService._consensus(ours("Hold", 0.0), 0.60, None)
    assert c["action"] == "Buy" and c["confidence"] == 0.4


@pytest.mark.skipif(not (STUDY_DIR / "signals.parquet").exists(), reason="study artifacts not generated")
def test_brief_uses_research_memory():
    service = AgentService()
    brief = service.brief("TSLA")
    assert set(brief["systems"]) == {"Plain RAG", "FinMem", "Ours"}
    assert brief["memories"] and all(m["layer"] in ("news", "10-Q", "10-K") for m in brief["memories"])
    assert brief["consensus"]["action"] in ("Buy", "Sell", "Hold")
