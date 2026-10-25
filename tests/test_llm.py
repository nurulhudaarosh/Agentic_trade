import json
from pathlib import Path

from agentic_trade.llm import OpenRouterClient


def test_llm_parser_accepts_structured_action(tmp_path: Path):
    client = OpenRouterClient("test-model", 0.7, tmp_path, api_key="test")
    result = client._parse(json.dumps({"action": "Buy", "reasoning": "positive", "cited_memory_ids": ["m1"]}))
    assert result["action"] == "Buy"
    assert result["cited_memory_ids"] == ["m1"]


def test_missing_key_falls_back_to_hold(tmp_path: Path):
    client = OpenRouterClient("test-model", 0.7, tmp_path, api_key=None)
    client.api_key = None
    result = client.decide("ours", "TSLA", "2026-09-01", 0.01, [])
    assert result["action"] == "Hold"
    assert result["llm_status"] == "missing_key"
    assert client.stats.failures == 1
