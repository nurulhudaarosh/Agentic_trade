from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".env"))
except ImportError:
    pass


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    parse_failures: int = 0
    last_error: str | None = None


class OpenRouterClient:
    def __init__(self, model: str, temperature: float, cache_dir: Path, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self.cache_dir = cache_dir
        groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY") or os.getenv("GROOQ_KEY")
        openrouter_key = os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPEN_ROUTER_kEY")
        configured_url = base_url or os.getenv("OPEN_ROUTER_BASE_URL") or ("https://api.groq.com/openai/v1" if groq_key and not openrouter_key else "https://openrouter.ai/api/v1")
        self.base_url = configured_url.rstrip("/")
        self.api_key = api_key or (groq_key if self.base_url == "https://api.groq.com/openai/v1" else openrouter_key)
        self.stats = LLMStats()

    def decide(self, system_name: str, ticker: str, decision_date: str, momentum: float, memories: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self._prompt(system_name, ticker, decision_date, momentum, memories)
        cache_key = hashlib.sha256(json.dumps({"model": self.model, "temperature": self.temperature, "prompt": prompt}, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self.stats.cache_hits += 1
            return json.loads(cache_path.read_text())
        if not self.api_key:
            self.stats.failures += 1
            self.stats.last_error = "missing_api_key"
            return {"action": "Hold", "reasoning": "LLM key unavailable", "cited_memory_ids": [], "llm_status": "missing_key"}
        self.stats.calls += 1
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "temperature": self.temperature, "max_tokens": 1024, "messages": [{"role": "system", "content": "You are a cautious financial research decision agent. Output exactly one compact valid JSON object and no analysis or markdown."}, {"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = self._parse(content)
            cache_path.write_text(json.dumps(result))
            return result
        except Exception as exc:
            self.stats.failures += 1
            status = f"http_{response.status_code}" if "response" in locals() else type(exc).__name__
            self.stats.last_error = status
            return {"action": "Hold", "reasoning": f"LLM unavailable: {status}", "cited_memory_ids": [], "llm_status": "failure"}

    def _parse(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            self.stats.parse_failures += 1
            return {"action": "Hold", "reasoning": "Invalid JSON response", "cited_memory_ids": [], "llm_status": "parse_failure"}
        action = str(data.get("action", "Hold")).title()
        if action not in {"Buy", "Sell", "Hold"}:
            self.stats.parse_failures += 1
            action = "Hold"
        cited = data.get("cited_memory_ids", [])
        if not isinstance(cited, list):
            cited = []
        return {"action": action, "reasoning": str(data.get("reasoning", "")), "cited_memory_ids": [str(item) for item in cited], "llm_status": "success"}

    @staticmethod
    def _prompt(system_name: str, ticker: str, decision_date: str, momentum: float, memories: list[dict[str, Any]]) -> str:
        evidence = "\n".join(f"ID={item['id']} source={item['source']} date={item['date']} relationship={item.get('relationship', '')} text={item['text']}" for item in memories) or "No memories retrieved."
        return f"Ticker: {ticker}\nDate: {decision_date}\nSystem: {system_name}\nPrevious return: {momentum:.6f}\nEvidence:\n{evidence}\nOutput exactly one compact JSON object with string action (Buy, Sell, or Hold), string reasoning, and array cited_memory_ids. Do not output analysis, markdown, or any other text."
