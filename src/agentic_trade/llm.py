from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("agentic_trade.llm")


def _load_env_fallback() -> None:
    for candidate in [Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]:
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


_load_env_fallback()

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(".env"))
except ImportError:
    pass


def _get_groq_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY") or os.getenv("GROOQ_KEY")


def _get_openrouter_key() -> str | None:
    return (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER_KEY")
        or os.getenv("OPEN_ROUTER_kEY")
        or os.getenv("AGENT_ROUTER_KEY")
    )


DEFAULT_OPENROUTER_FREE_MODELS = [
    "recursive/flint-v1.0:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    parse_failures: int = 0
    fallbacks: int = 0
    last_error: str | None = None
    last_provider: str | None = None
    last_model: str | None = None
    providers_used: dict[str, int] = field(default_factory=lambda: {"groq": 0, "openrouter": 0})


SYSTEM_PROMPTS = {
    "aggressive": (
        "You are an active, aggressive alpha-seeking financial trading agent. "
        "Your objective is to maximize portfolio return by capturing market upside and strong catalysts. "
        "When retrieved evidence contains positive catalysts, analyst upgrades, or bullish momentum, strongly favor 'Buy' over 'Hold'. "
        "When evidence contains negative catalysts, downgrades, or downward momentum, strongly favor 'Sell'. "
        "Only choose 'Hold' if the market evidence is completely devoid of directional information or purely neutral. "
        "Do not default to passive Hold when an active trade opportunity exists. "
        "Output exactly one compact valid JSON object and no analysis or markdown."
    ),
    "standard": (
        "You are a balanced, objective financial trading decision agent. "
        "Your objective is to balance risk and return systematically. "
        "Weigh retrieved news catalysts, source credibility, and recent price momentum. "
        "Choose 'Buy' when positive evidence outweighs negative risks. "
        "Choose 'Sell' when negative catalysts or downward momentum dominate. "
        "Choose 'Hold' only when evidence is genuinely conflicting or inconclusive. "
        "Output exactly one compact valid JSON object and no analysis or markdown."
    ),
    "conservative": (
        "You are a cautious capital preservation trading agent. "
        "Your highest priority is risk aversion and avoiding drawdowns. "
        "Only choose 'Buy' when high-conviction positive catalysts exist with minimal downside risk. "
        "Choose 'Sell' on material adverse developments. "
        "Default to 'Hold' whenever there are conflicting signals, ambiguity, or elevated uncertainty. "
        "Output exactly one compact valid JSON object and no analysis or markdown."
    ),
}


class OpenRouterClient:
    """Multi-provider LLM client with automatic failover between Groq and OpenRouter.

    If Groq fails (rate limit 429, timeout, connection issue, or quota), it automatically
    fails over to OpenRouter free models (and vice versa).
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        primary_provider: str = "auto",
        openrouter_free_models: list[str] | None = None,
    ) -> None:
        self.temperature = temperature
        self.cache_dir = cache_dir or Path(".cache/llm")
        self.stats = LLMStats()

        env_groq_key = _get_groq_key()
        env_openrouter_key = _get_openrouter_key()

        # Assign credentials
        if base_url and "groq" in base_url:
            self.groq_key = api_key or env_groq_key
            self.openrouter_key = env_openrouter_key
        elif base_url and ("openrouter" in base_url or "agentrouter" in base_url):
            self.openrouter_key = api_key or env_openrouter_key
            self.groq_key = env_groq_key
        else:
            self.groq_key = env_groq_key or api_key
            self.openrouter_key = env_openrouter_key or api_key

        self.api_key = api_key or self.groq_key or self.openrouter_key

        self.groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        if self.openrouter_key and self.openrouter_key.startswith("sk-or-"):
            self.openrouter_base_url = "https://openrouter.ai/api/v1"
        else:
            self.openrouter_base_url = (
                os.getenv("OPENROUTER_BASE_URL")
                or os.getenv("OPEN_ROUTER_BASE_URL")
                or os.getenv("AGENT_ROUTER_URL")
                or "https://openrouter.ai/api/v1"
            ).rstrip("/")

        # Model resolution
        groq_env_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        if (
            model
            and not str(model).endswith(":free")
            and "openrouter" not in (base_url or "")
            and model not in ["default"]
        ):
            self.groq_model = model
        else:
            self.groq_model = groq_env_model

        # OpenRouter free models setup
        configured_free = list(openrouter_free_models or [])
        env_free_models = os.getenv("OPENROUTER_FREE_MODELS")
        if env_free_models:
            configured_free.extend([m.strip() for m in env_free_models.split(",") if m.strip()])

        env_model_name = os.getenv("MODEL_NAME")
        if env_model_name and env_model_name.endswith(":free") and env_model_name not in configured_free:
            configured_free.insert(0, env_model_name)

        if model and str(model).endswith(":free") and model not in configured_free:
            configured_free.insert(0, str(model))

        for m in DEFAULT_OPENROUTER_FREE_MODELS:
            if m not in configured_free:
                configured_free.append(m)

        self.openrouter_free_models = configured_free
        self.openrouter_model = self.openrouter_free_models[0]

        # Primary provider selection
        if primary_provider in {"groq", "openrouter"}:
            self.primary_provider = primary_provider
        elif base_url:
            self.primary_provider = "groq" if "groq" in base_url else "openrouter"
        elif model and str(model).endswith(":free"):
            self.primary_provider = "openrouter"
        elif self.groq_key:
            self.primary_provider = "groq"
        elif self.openrouter_key:
            self.primary_provider = "openrouter"
        else:
            self.primary_provider = "groq"

        # Backwards compatibility attributes
        if self.primary_provider == "openrouter":
            self.base_url = base_url or self.openrouter_base_url
            self.model = model or self.openrouter_model
        else:
            self.base_url = base_url or self.groq_base_url
            self.model = self.groq_model

    def decide(
        self,
        system_name: str,
        ticker: str,
        decision_date: str,
        momentum: float,
        memories: list[dict[str, Any]],
        risk_mode: str = "standard",
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_prompt = SYSTEM_PROMPTS.get(risk_mode.lower(), SYSTEM_PROMPTS["standard"])
        prompt = self._prompt(system_name, ticker, decision_date, momentum, memories, risk_mode=risk_mode)

        # Cache check
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "risk_mode": risk_mode,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        cache_path = self.cache_dir / f"{cache_key}.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self.stats.cache_hits += 1
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Check key availability
        active_groq = self.groq_key if self.api_key is not None else None
        active_openrouter = self.openrouter_key if self.api_key is not None else None

        if not active_groq and not active_openrouter:
            self.stats.failures += 1
            self.stats.last_error = "missing_api_key"
            return {
                "action": "Hold",
                "reasoning": "LLM key unavailable",
                "cited_memory_ids": [],
                "llm_status": "missing_key",
            }

        self.stats.calls += 1

        # Build failover plan
        plan = ["openrouter", "groq"] if self.primary_provider == "openrouter" else ["groq", "openrouter"]

        last_error = "unknown_error"
        successful_result: dict[str, Any] | None = None
        successful_provider: str | None = None
        successful_model: str | None = None

        for provider_idx, provider in enumerate(plan):
            if provider == "groq" and active_groq:
                groq_models = [self.groq_model]
                for gm in DEFAULT_GROQ_MODELS:
                    if gm not in groq_models:
                        groq_models.append(gm)

                for gm in groq_models:
                    try:
                        content = self._call_provider(
                            base_url=self.groq_base_url,
                            api_key=active_groq,
                            model=gm,
                            system_prompt=system_prompt,
                            prompt=prompt,
                            is_openrouter=False,
                        )
                        parsed = self._parse(content)
                        if parsed.get("llm_status") == "success":
                            successful_result = parsed
                            successful_provider = "groq"
                            successful_model = gm
                            break
                        last_error = f"groq_{gm}_parse_failure"
                    except Exception as exc:
                        last_error = f"groq_{gm}: {exc}"
                        continue

                if successful_result:
                    break

                if provider_idx < len(plan) - 1 and active_openrouter:
                    self.stats.fallbacks += 1
                    print(
                        f"⚠️ [LLM Failover] Groq encountered error ({last_error}). "
                        f"Failing over to OpenRouter free models..."
                    )

            elif provider == "openrouter" and active_openrouter:
                for om in self.openrouter_free_models:
                    try:
                        content = self._call_provider(
                            base_url=self.openrouter_base_url,
                            api_key=active_openrouter,
                            model=om,
                            system_prompt=system_prompt,
                            prompt=prompt,
                            is_openrouter=True,
                        )
                        parsed = self._parse(content)
                        if parsed.get("llm_status") == "success":
                            successful_result = parsed
                            successful_provider = "openrouter"
                            successful_model = om
                            break
                        last_error = f"openrouter_{om}_parse_failure"
                    except Exception as exc:
                        last_error = f"openrouter_{om}: {exc}"
                        continue

                if successful_result:
                    break

                if provider_idx < len(plan) - 1 and active_groq:
                    self.stats.fallbacks += 1
                    print(
                        f"⚠️ [LLM Failover] OpenRouter free models encountered error ({last_error}). "
                        f"Failing over to Groq..."
                    )

        if successful_result:
            self.stats.last_provider = successful_provider
            self.stats.last_model = successful_model
            if successful_provider:
                self.stats.providers_used[successful_provider] = (
                    self.stats.providers_used.get(successful_provider, 0) + 1
                )
            successful_result["provider"] = successful_provider
            successful_result["model"] = successful_model
            try:
                cache_path.write_text(json.dumps(successful_result), encoding="utf-8")
            except Exception:
                pass
            time.sleep(0.3)
            return successful_result

        self.stats.failures += 1
        self.stats.last_error = last_error
        return {
            "action": "Hold",
            "reasoning": f"LLM unavailable: {last_error}",
            "cited_memory_ids": [],
            "llm_status": "failure",
        }

    def _call_provider(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        prompt: str,
        is_openrouter: bool = False,
        timeout: int = 30,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if is_openrouter:
            headers["HTTP-Referer"] = "https://github.com/agentic-trade"
            headers["X-Title"] = "AgenticTrade"

        body: dict[str, Any] = {
            "model": model,
            "temperature": self.temperature,
            "max_tokens": 1536,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout,
        )

        # Free models that do not support response_format json_object
        if response.status_code == 400 and "response_format" in response.text.lower():
            del body["response_format"]
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=timeout,
            )

        if response.status_code != 200:
            err_msg = ""
            try:
                err_msg = response.json().get("error", {}).get("message", "")
            except Exception:
                err_msg = response.text[:80]
            raise RuntimeError(f"HTTP {response.status_code}: {err_msg or response.reason}")

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Empty choices returned from API")
        return str(choices[0].get("message", {}).get("content", ""))

    def _parse(self, content: str) -> dict[str, Any]:
        if not content or not isinstance(content, str):
            self.stats.parse_failures += 1
            return {
                "action": "Hold",
                "reasoning": "Empty LLM response",
                "cited_memory_ids": [],
                "llm_status": "parse_failure",
            }

        cleaned = content.strip()
        # Remove reasoning tags like <think>...</think> from models like DeepSeek-R1
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        data: Any = None
        # 1. Direct JSON parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2. Extract markdown code block
        if data is None:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # 3. Outermost curly braces
        if data is None:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last > first:
                try:
                    data = json.loads(cleaned[first : last + 1])
                except json.JSONDecodeError:
                    pass

        # 4. Regex fallback extraction
        if data is None:
            act_match = re.search(r'"?action"?\s*:\s*"?(Buy|Sell|Hold)"?', cleaned, flags=re.IGNORECASE)
            if act_match:
                act = act_match.group(1).title()
                rs_match = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', cleaned)
                rs = rs_match.group(1) if rs_match else cleaned[:100]
                cited_match = re.findall(r'"cited_memory_ids"\s*:\s*\[([^\]]*)\]', cleaned)
                cited_ids: list[str] = []
                if cited_match:
                    cited_ids = [x.strip().strip('"\'') for x in cited_match[0].split(",") if x.strip()]
                data = {"action": act, "reasoning": rs, "cited_memory_ids": cited_ids}

        if not isinstance(data, dict):
            self.stats.parse_failures += 1
            return {
                "action": "Hold",
                "reasoning": f"Invalid JSON response: {cleaned[:100]}",
                "cited_memory_ids": [],
                "llm_status": "parse_failure",
            }

        action = str(data.get("action", "Hold")).strip().title()
        if action not in {"Buy", "Sell", "Hold"}:
            self.stats.parse_failures += 1
            action = "Hold"

        cited = data.get("cited_memory_ids", [])
        if not isinstance(cited, list):
            cited = []

        return {
            "action": action,
            "reasoning": str(data.get("reasoning", "")),
            "cited_memory_ids": [str(item) for item in cited],
            "llm_status": "success",
        }

    @staticmethod
    def _prompt(
        system_name: str,
        ticker: str,
        decision_date: str,
        momentum: float,
        memories: list[dict[str, Any]],
        risk_mode: str = "standard",
    ) -> str:
        risk_label = {
            "aggressive": "RISK PROFILE: AGGRESSIVE (Prioritize alpha generation and active trading; favor Buy/Sell over passive Hold unless evidence is completely neutral).",
            "standard": "RISK PROFILE: STANDARD (Balanced risk/reward; take active positions when net evidence has directional edge).",
            "conservative": "RISK PROFILE: CONSERVATIVE (Prioritize capital preservation; default to Hold on ambiguity or mixed signals).",
        }.get(risk_mode.lower(), "RISK PROFILE: STANDARD")

        evidence = "\n".join(
            f"ID={item['id']} source={item['source']} date={item['date']} relationship={item.get('relationship', '')} text={item['text']}"
            for item in memories
        ) or "No memories retrieved."

        return (
            f"Ticker: {ticker}\n"
            f"Date: {decision_date}\n"
            f"System: {system_name}\n"
            f"{risk_label}\n"
            f"Previous return: {momentum:.6f}\n"
            f"Evidence:\n{evidence}\n"
            "Output exactly one compact JSON object with string action (Buy, Sell, or Hold), string reasoning, and array cited_memory_ids. Do not output analysis, markdown, or any other text."
        )
