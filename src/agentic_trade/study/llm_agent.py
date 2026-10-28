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
}
ENDPOINTS = {"groq": "https://api.groq.com/openai/v1/chat/completions",
             "openrouter": "https://openrouter.ai/api/v1/chat/completions"}
LAYER_NAME = {0: "short-term news", 1: "quarterly report (10-Q)", 2: "annual report (10-K)"}

SYSTEM = ("You are an expert equity trading agent. You trade a single stock daily and must answer with "
          "a JSON object only: {\"action\": \"Buy\"|\"Sell\"|\"Hold\", \"confidence\": 0-1, "
          "\"reasoning\": \"<=40 words\", \"cited_ids\": [memory ids]}. Buy = hold a long position, "
          "Sell = hold a short position, Hold = keep the current position unchanged.")


def _key(provider: str) -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass
    if provider == "groq":
        return os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY") or os.getenv("GROOQ_KEY")
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPEN_ROUTER_kEY")


@dataclass
class Backbone:
    label: str
    temperature: float = 0.0
    cache_dir: Path = Path("artifacts/study/llm_cache")

    def __post_init__(self) -> None:
        self.provider, self.model = BACKBONES[self.label]
        self.key = _key(self.provider)
        self.calls = self.cache_hits = self.failures = 0
        self.min_interval = 11.0 if self.provider == "groq" else 4.0
        self._last = 0.0

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
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = requests.post(ENDPOINTS[self.provider], headers=headers, json=self._payload(prompt), timeout=90)
            except requests.RequestException:
                time.sleep(10)
                continue
            if resp.status_code == 429:
                retry = resp.headers.get("retry-after")
                time.sleep(min(float(retry) if retry and retry.replace(".", "").isdigit() else 30.0, 300) + 1)
                continue
            if resp.status_code >= 500 or resp.status_code == 408:
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
                       out_dir: Path, systems: tuple[str, ...] = ("No-Memory", "FinMem", "Ours")) -> pd.DataFrame:
    text = {str(i): h for i, h in zip(news["id"], news["headline"])}
    text.update({str(i): t for i, t in zip(filings["id"], filings["text"])})
    closes = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
    logret = np.log(closes).diff()
    ev = {k: g.sort_values("rank") for k, g in evidence.groupby(["system", "ticker", "date"])}
    sig = signals.set_index(["system", "ticker", "date"])
    all_days = list(closes.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(label: str) -> None:
        bb = Backbone(label)
        for t in tickers:
            for system in systems:
                position, hist = 0.0, []
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
                    if system == "Ours" and (system, t, day) in sig.index:
                        s = sig.loc[(system, t, day)]
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
                    with lock:
                        results.append({"backbone": label, "system": system, "ticker": t, "date": day, "action": action,
                                        "position": position, "next_return": r, "status": decision["status"],
                                        "confidence": decision.get("confidence"), "reasoning": decision.get("reasoning", ""),
                                        "n_memories": len(memories)})
                print(f"[llm] {label} {t} {system} done (calls={bb.calls}, cache={bb.cache_hits}, fail={bb.failures})", flush=True)
                with lock:
                    pd.DataFrame(results).to_parquet(out_dir / "llm_decisions.parquet", index=False)

    threads = [threading.Thread(target=worker, args=(b,)) for b in backbones]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    frame = pd.DataFrame(results)
    frame.to_parquet(out_dir / "llm_decisions.parquet", index=False)
    return frame
