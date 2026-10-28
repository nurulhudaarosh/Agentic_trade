"""LLM backbone experiment: the same retrieved memories, rendered into a FinMem-style
prompt, decided by several free LLM backbones (no cross-model failover, so every row
is attributable to exactly one model)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .backtest import COMPANY

BACKBONES = {
    "gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
    "gpt-oss-20b": ("groq", "openai/gpt-oss-20b"),
    "qwen3.8-27b": ("groq", "qwen/qwen3.8-27b"),
    "nemotron-3-ultra": ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    "xing4.0-29b": ("openrouter", "xingchen-agi/xing4.0-29b-a4b:free"),
}
# Concurrent decision chains per provider: Groq's free tier is token-per-minute bound (one
# paced stream per model); OpenRouter models without the daily free cap tolerate several.
DEFAULT_WORKERS = {"groq": 1, "openrouter": 5}
ENDPOINTS = {"groq": "https://api.groq.com/openai/v1/chat/completions",
             "openrouter": "https://openrouter.ai/api/v1/chat/completions"}
LAYER_NAME = {0: "short-term news", 1: "quarterly report (10-Q)", 2: "annual report (10-K)"}

SYSTEM = ("You are an expert equity trading agent. You trade a single stock daily and must answer with "
          "a JSON object only: {\"action\": \"Buy\"|\"Sell\"|\"Hold\", \"confidence\": 0-1, "
          "\"reasoning\": \"<=40 words\", \"cited_ids\": [memory ids]}. Buy = hold a long position, "
          "Sell = hold a short position, Hold = keep the current position unchanged.")


GROQ_KEY_VAR = re.compile(r"^GR(O)?OQ(_API)?_KEY\d*$")


def _keys(provider: str) -> list[str]:
    """All credentials for a provider. Groq keys are every env var named GROQ_KEY,
    GROQ_API_KEY, GROOQ_KEY, GROOQ_KEY2, GROQ_KEY3, ... (one per account), de-duplicated."""
    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(Path(".env"))
        names = list(dotenv_values(Path(".env")))
    except ImportError:
        names = []
    if provider == "groq":
        names = sorted({n for n in list(os.environ) + names if GROQ_KEY_VAR.match(n)},
                       key=lambda n: (re.sub(r"\d+$", "", n), int(re.search(r"(\d*)$", n).group(1) or 0)))
        values = [os.getenv(n) for n in names]
    else:
        values = [os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPEN_ROUTER_kEY")]
    seen, out = set(), []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _key(provider: str) -> str | None:
    keys = _keys(provider)
    return keys[0] if keys else None


@dataclass
class Backbone:
    label: str
    temperature: float = 0.0
    cache_dir: Path = Path("artifacts/study/llm_cache")

    def __post_init__(self) -> None:
        self.provider, self.model = BACKBONES[self.label]
        self.keys = _keys(self.provider)
        self.key = self.keys[0] if self.keys else None
        self.calls = self.cache_hits = self.failures = self.rate_limited = 0
        # Per-key pacing: each account has its own per-minute and per-day token budget.
        self.min_interval = 11.0 if self.provider == "groq" else 1.0
        self._last = {k: 0.0 for k in self.keys}
        self._blocked_until = {k: 0.0 for k in self.keys}
        self._pace = threading.Lock()

    def _acquire_key(self) -> str:
        """Wait for the account that can be used soonest (not rate-limited, pacing respected)."""
        while True:
            with self._pace:
                now = time.time()
                ready = {k: max(self._blocked_until[k], self._last[k] + self.min_interval) for k in self.keys}
                key = min(ready, key=ready.get)
                if ready[key] <= now:
                    self._last[key] = now
                    return key
                delay = ready[key] - now
            time.sleep(min(delay, 30.0))

    def _block(self, key: str, seconds: float) -> None:
        with self._pace:
            self._blocked_until[key] = max(self._blocked_until[key], time.time() + seconds)

    def _payload(self, prompt: str) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 1200,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        if self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        elif self.model.startswith("qwen/"):
            body["reasoning_effort"] = "none"
        return body

    def ask(self, prompt: str) -> dict:
        digest = hashlib.sha256(json.dumps([self.model, self.temperature, SYSTEM, prompt]).encode()).hexdigest()
        path = self.cache_dir / self.label / f"{digest}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        path.parent.mkdir(parents=True, exist_ok=True)
        errors = 0
        # Rate limits (429) are waited out without limit: a quota pause must never turn into
        # a recorded "Hold". Only transport/server errors count towards giving up.
        while errors < 8:
            key = self._acquire_key()
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                resp = requests.post(ENDPOINTS[self.provider], headers=headers, json=self._payload(prompt), timeout=90)
            except requests.RequestException:
                errors += 1
                time.sleep(10)
                continue
            if resp.status_code == 429:
                retry = resp.headers.get("retry-after")
                self.rate_limited += 1
                self._block(key, min(float(retry) if retry and retry.replace(".", "").isdigit() else 30.0, 3600) + 1)
                continue
            if resp.status_code >= 500 or resp.status_code == 408:
                errors += 1
                time.sleep(10)
                continue
            if resp.status_code != 200:
                self.failures += 1
                return {"action": "Hold", "status": f"http_{resp.status_code}", "reasoning": resp.text[:200]}
            self.calls += 1
            content = (resp.json()["choices"][0]["message"].get("content") or "")
            result = parse_decision(content)
            if result["status"] == "ok":
                path.write_text(json.dumps(result))
            return result
        self.failures += 1
        return {"action": "Hold", "status": "retries_exhausted", "reasoning": ""}


def parse_decision(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            action = str(obj.get("action", "")).strip().capitalize()
            if action in ("Buy", "Sell", "Hold"):
                return {"action": action, "confidence": float(obj.get("confidence", 0.5) or 0.5),
                        "reasoning": str(obj.get("reasoning", ""))[:300],
                        "cited_ids": [str(x) for x in obj.get("cited_ids", [])][:10], "status": "ok"}
        except (ValueError, TypeError):
            pass
    found = re.search(r"\b(Buy|Sell|Hold)\b", content, re.I)
    if found:
        return {"action": found.group(1).capitalize(), "confidence": 0.5, "reasoning": "", "cited_ids": [], "status": "ok_regex"}
    return {"action": "Hold", "confidence": 0.0, "reasoning": content[:200], "cited_ids": [], "status": "parse_failure"}


def character(cum3: float) -> str:
    # FinMem's self-adaptive character: risk-seeking after gains, risk-averse after losses.
    if cum3 >= 0:
        return ("You are a risk-seeking investor right now (recent cumulative return is non-negative): "
                "act on clear evidence, accept volatility.")
    return ("You are a risk-averse investor right now (recent cumulative return is negative): "
            "protect capital and act only on strong, consistent evidence.")


def build_prompt(system: str, ticker: str, day: str, momentum: float, cum3: float, position: float,
                 memories: list[dict], extras: dict) -> str:
    lines = [f"Ticker: {ticker} ({COMPANY.get(ticker, ticker)}). Decision date: {day} (after the close).",
             f"Current position: {'long' if position > 0 else 'short' if position < 0 else 'flat'}.",
             f"Momentum: 3-day cumulative log return {momentum:+.2%}.",
             character(cum3)]
    if system != "No-Memory":
        lines.append("Retrieved memory (id | layer | date | text):")
        for m in memories:
            tag = ""
            if system == "Ours":
                tag = f" [source credibility {m['cred']:.2f}; reported by {int(m['corrob'])} article(s)]"
            lines.append(f"- {m['id']} | {LAYER_NAME[m['layer']]} | {m['date']} | {m['text']}{tag}")
        if system == "Ours":
            if extras.get("peer"):
                lines.append(f"Related-company evidence: {extras['peer']}")
            if extras.get("episodic"):
                lines.append(f"Outcome memory: {extras['episodic']}")
    lines.append("Considering short-, mid- and long-term information, decide Buy, Sell or Hold for tomorrow. Reply with JSON only.")
    return "\n".join(lines)


def run_llm_experiment(signals: pd.DataFrame, evidence: pd.DataFrame, news: pd.DataFrame, filings: pd.DataFrame,
                       prices: pd.DataFrame, tickers: list[str], dates: list, backbones: list[str],
                       out_dir: Path, systems: tuple[str, ...] = ("No-Memory", "FinMem", "Ours"),
                       workers: dict[str, int] | None = None, out_name: str = "llm_decisions.parquet") -> pd.DataFrame:
    """Each (backbone, ticker, system) is an independent decision chain (its position and
    recent P&L feed the next prompt), so chains run concurrently; calls to one backbone are
    paced by that backbone's shared rate limiter."""
    from concurrent.futures import ThreadPoolExecutor

    text = {str(i): h for i, h in zip(news["id"], news["headline"])}
    text.update({str(i): t for i, t in zip(filings["id"], filings["text"])})
    closes = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    logret = np.log(closes).diff()
    ev = {k: g.sort_values("rank") for k, g in evidence.groupby(["system", "ticker", "date"])}
    # Plain dict built once: pandas' lazy MultiIndex lookups are not thread-safe.
    ours_sig = {(t, d): row for t, d, row in zip(
        signals.loc[signals["system"] == "Ours", "ticker"], signals.loc[signals["system"] == "Ours", "date"],
        signals.loc[signals["system"] == "Ours", ["peer_n", "peer_signal", "episodic_edge", "episodic_n"]].to_dict("records"))}
    all_days = list(closes.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    lock = threading.Lock()
    clients = {label: Backbone(label) for label in backbones}

    def chain(label: str, t: str, system: str) -> None:
        bb = clients[label]
        position, hist, rows_out = 0.0, [], []
        for day in dates:
            i = all_days.index(day)
            momentum = float(logret[t].iloc[i - 2: i + 1].sum())
            cum3 = float(sum(hist[-3:]))
            mem_sys = "FinMem" if system == "No-Memory" else system
            rows = ev.get((mem_sys, t, day))
            memories = []
            if rows is not None and system != "No-Memory":
                for _, r in rows.iterrows():
                    memories.append({"id": r["item_id"], "layer": int(r["layer"]), "text": text.get(str(r["item_id"]), "")[:220],
                                     "date": str(pd.Timestamp.fromordinal(pd.Timestamp(day).toordinal() - int(r["age_days"])).date()),
                                     "cred": float(r["cred"]),
                                     "corrob": float(r["corroboration"])})
            extras = {}
            if system == "Ours" and (t, day) in ours_sig:
                s = ours_sig[(t, day)]
                if s["peer_n"] > 0:
                    extras["peer"] = f"{int(s['peer_n'])} same-day articles on co-mentioned firms, credibility-weighted sentiment {s['peer_signal']:+.2f}."
                if pd.notna(s["episodic_edge"]):
                    verdict = "news signals in such situations were followed by adverse moves - be cautious" if s["episodic_edge"] < 0 else "news signals in such situations were reliable"
                    extras["episodic"] = f"in {int(s['episodic_n'])} similar past situations the signed market-adjusted return was {s['episodic_edge']:+.2%}; {verdict}."
            prompt = build_prompt(system, t, str(day), momentum, cum3, position, memories, extras)
            decision = bb.ask(prompt)
            action = decision["action"]
            position = 1.0 if action == "Buy" else -1.0 if action == "Sell" else position
            nxt = all_days[i + 1]
            r = float(closes.at[nxt, t] / closes.at[day, t] - 1)
            hist.append(position * r)
            rows_out.append({"backbone": label, "system": system, "ticker": t, "date": day, "action": action,
                             "position": position, "next_return": r, "status": decision["status"],
                             "confidence": decision.get("confidence"), "reasoning": decision.get("reasoning", ""),
                             "n_memories": len(memories)})
        with lock:
            results.extend(rows_out)
            pd.DataFrame(results).to_parquet(out_dir / out_name, index=False)
        print(f"[llm] {label} {t} {system} done (calls={bb.calls}, cache={bb.cache_hits}, fail={bb.failures}, "
              f"rate_limited={bb.rate_limited})", flush=True)

    workers = workers or {}
    pools = []
    for label in backbones:
        n = workers.get(label, DEFAULT_WORKERS[clients[label].provider] * max(1, len(clients[label].keys)))
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix=label)
        pools.append([pool.submit(chain, label, t, system) for t in tickers for system in systems])
    for futures in pools:
        for fut in futures:
            fut.result()
    frame = pd.DataFrame(results)
    frame.to_parquet(out_dir / out_name, index=False)
    return frame
