import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_risk_profile_prompts_are_differentiated(tmp_path: Path):
    client = OpenRouterClient("test-model", 0.7, tmp_path, api_key="test")
    p_agg = client._prompt("ours", "TSLA", "2026-09-01", 0.02, [], risk_mode="aggressive")
    p_con = client._prompt("ours", "TSLA", "2026-09-01", 0.02, [], risk_mode="conservative")
    p_std = client._prompt("ours", "TSLA", "2026-09-01", 0.02, [], risk_mode="standard")

    assert "AGGRESSIVE" in p_agg
    assert "CONSERVATIVE" in p_con
    assert "STANDARD" in p_std
    assert p_agg != p_con


def test_parser_handles_think_tags_and_markdown_blocks(tmp_path: Path):
    client = OpenRouterClient("test-model", 0.7, tmp_path, api_key="test")
    raw_response = (
        "<think>Evaluating earnings surprise and momentum...</think>\n"
        "```json\n"
        '{"action": "Buy", "reasoning": "Strong EPS beat", "cited_memory_ids": ["ev1", "ev2"]}\n'
        "```"
    )
    result = client._parse(raw_response)
    assert result["action"] == "Buy"
    assert result["reasoning"] == "Strong EPS beat"
    assert result["cited_memory_ids"] == ["ev1", "ev2"]
    assert result["llm_status"] == "success"


def test_failover_from_groq_to_openrouter(tmp_path: Path):
    client = OpenRouterClient(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        cache_dir=tmp_path / "cache",
        api_key="mock_key",
        primary_provider="groq",
    )
    client.groq_key = "mock_groq_key"
    client.openrouter_key = "mock_openrouter_key"

    def mock_post(url, *args, **kwargs):
        mock_resp = MagicMock()
        if "groq.com" in url:
            # Simulate Groq 429 rate limit
            mock_resp.status_code = 429
            mock_resp.text = "Rate limit reached"
            mock_resp.json.return_value = {"error": {"message": "Rate limit reached"}}
            return mock_resp
        elif "openrouter.ai" in url or "agentrouter.org" in url:
            # OpenRouter free model succeeds
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "action": "Buy",
                                "reasoning": "OpenRouter free model decision",
                                "cited_memory_ids": ["m100"],
                            })
                        }
                    }
                ]
            }
            return mock_resp
        mock_resp.status_code = 500
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        result = client.decide("ours", "TSLA", "2026-09-01", 0.03, [{"id": "m100", "source": "news", "date": "2026-09-01", "text": "catalyst"}])

    assert result["action"] == "Buy"
    assert result["provider"] == "openrouter"
    assert result["llm_status"] == "success"
    assert client.stats.fallbacks == 1


def test_failover_from_openrouter_to_groq(tmp_path: Path):
    client = OpenRouterClient(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        cache_dir=tmp_path / "cache",
        api_key="mock_key",
        primary_provider="openrouter",
    )
    client.groq_key = "mock_groq_key"
    client.openrouter_key = "mock_openrouter_key"

    def mock_post(url, *args, **kwargs):
        mock_resp = MagicMock()
        if "openrouter.ai" in url or "agentrouter.org" in url:
            # Simulate OpenRouter free models down / 503
            mock_resp.status_code = 503
            mock_resp.text = "Service unavailable"
            mock_resp.json.return_value = {"error": {"message": "Service unavailable"}}
            return mock_resp
        elif "groq.com" in url:
            # Groq succeeds
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "action": "Sell",
                                "reasoning": "Groq fallback decision",
                                "cited_memory_ids": [],
                            })
                        }
                    }
                ]
            }
            return mock_resp
        mock_resp.status_code = 500
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        result = client.decide("ours", "TSLA", "2026-09-01", -0.04, [])

    assert result["action"] == "Sell"
    assert result["provider"] == "groq"
    assert result["llm_status"] == "success"
    assert client.stats.fallbacks == 1


def test_openrouter_free_model_rotation(tmp_path: Path):
    client = OpenRouterClient(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        cache_dir=tmp_path / "cache",
        api_key="mock_key",
        primary_provider="openrouter",
        openrouter_free_models=["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-exp:free"],
    )
    client.groq_key = None
    client.openrouter_key = "mock_openrouter_key"

    def mock_post(url, *args, **kwargs):
        body = kwargs.get("json", {})
        model = body.get("model", "")
        mock_resp = MagicMock()
        if model == "meta-llama/llama-3.3-70b-instruct:free":
            # First free model is busy (429)
            mock_resp.status_code = 429
            mock_resp.text = "Rate limit reached"
            mock_resp.json.return_value = {"error": {"message": "Rate limit reached"}}
            return mock_resp
        elif model == "google/gemini-2.0-flash-exp:free":
            # Second free model succeeds
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "action": "Hold",
                                "reasoning": "Gemini free model success",
                                "cited_memory_ids": [],
                            })
                        }
                    }
                ]
            }
            return mock_resp
        mock_resp.status_code = 500
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        result = client.decide("ours", "TSLA", "2026-09-01", 0.0, [])

    assert result["action"] == "Hold"
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.0-flash-exp:free"
    assert result["llm_status"] == "success"
