"""Groq-backed decision generation and Alpaca execution.

The default path is dry-run. ``--execute`` submits to the Alpaca *paper*
account; live-money orders require both ``--live`` and ``ALPACA_LIVE_CONFIRM=YES``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import ExperimentConfig, ExperimentEngine
from .llm import OpenRouterClient
from .memory import MemoryEvent, OursMemory, event_matches_ticker, extract_topics
from .paper_trader import PaperBroker, PaperTradingAgent, TradingConfig

DEFAULT_TICKERS = ("TSLA", "NFLX", "AMZN", "MSFT", "COIN")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def make_llm(model: str, temperature: float, cache_dir: Path, provider: str = "auto") -> OpenRouterClient:
    return OpenRouterClient(
        model=model,
        temperature=temperature,
        cache_dir=cache_dir,
        primary_provider=provider,
    )


def load_price_history(data_dir: Path) -> dict[str, pd.Series]:
    path = data_dir / "prices.csv"
    if not path.exists():
        raise RuntimeError(f"Missing price file: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = frame["date"].dt.date
    result: dict[str, pd.Series] = {}
    for ticker, group in frame.groupby("ticker"):
        result[str(ticker).upper()] = group.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()
    return result


def resolve_history(history: list[dict[str, Any]], prices: dict[str, pd.Series]) -> list[dict[str, Any]]:
    """Attach realized t+1 returns to already-generated decisions."""
    for record in history:
        if record.get("realized_return") is not None:
            continue
        series = prices.get(record.get("ticker", ""))
        if series is None:
            continue
        decision_date = date.fromisoformat(record["date"])
        future = [day for day in series.index if day > decision_date]
        if not future or decision_date not in series.index:
            continue
        raw_return = float(series.loc[future[0]] / series.loc[decision_date] - 1)
        position = {"Buy": 1.0, "Sell": -1.0}.get(record.get("action", "Hold"), 0.0)
        record["realized_return"] = position * raw_return
    return history


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_history(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_memory(
    config: ExperimentConfig,
    ticker: str,
    as_of: date,
    history: list[dict[str, Any]],
) -> tuple[OursMemory, pd.Series, list[MemoryEvent]]:
    engine = ExperimentEngine(replace(config, ticker=ticker))
    engine.load_data()
    prices = engine.prices["close"].astype(float)
    if as_of not in prices.index:
        eligible = [day for day in prices.index if day <= as_of]
        if not eligible:
            raise RuntimeError(f"No price available on or before {as_of} for {ticker}")
        as_of = max(eligible)

    engine.config.train_end = as_of
    if config.use_provenance:
        engine._learn_provenance()

    memory = OursMemory()
    memory.source_reliability = dict(engine.systems["ours"].source_reliability)
    history_by_ticker = [record for record in history if record.get("ticker") == ticker]
    memory.decision_history = history_by_ticker

    for event in engine.events:
        if event.event_date > as_of:
            continue
        copy = MemoryEvent(
            event.event_id,
            event.event_date,
            event.ticker,
            event.source_type,
            event.text,
            event.summary,
            event.sentiment,
            set(event.topic_tags),
            set(event.related_tickers),
            source_name=event.source_name,
        )
        if config.use_reconciliation:
            memory.add(copy)
        else:
            memory.events.append(copy)
    return memory, prices, engine.events


def _recent_topics(events: list[MemoryEvent], as_of: date, ticker: str, lookback_days: int) -> set[str]:
    cutoff = as_of - timedelta(days=lookback_days)
    texts = [
        event.text for event in events
        if cutoff <= event.event_date <= as_of and event_matches_ticker(event, ticker)
    ]
    return extract_topics(" ".join(texts)) if texts else {"market_news"}


def _prompt_memories(memories: list[MemoryEvent]) -> list[dict[str, Any]]:
    return [
        {
            "id": event.event_id,
            "source": event.source_name or event.source_type,
            "date": event.event_date.isoformat(),
            "text": event.text,
            "relationship": ",".join(sorted(event.related_tickers)),
        }
        for event in memories
    ]


def generate_decisions(
    config: ExperimentConfig,
    tickers: tuple[str, ...],
    llm: OpenRouterClient,
    history_path: Path,
    top_k: int = 5,
    as_of: date | None = None,
    lookback_days: int = 14,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    prices_by_ticker = load_price_history(config.data_dir)
    history = resolve_history(read_history(history_path), prices_by_ticker)
    decisions: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    new_history: list[dict[str, Any]] = []

    risk_threshold = {"conservative": 0.25, "standard": 0.15, "aggressive": 0.05}.get(config.risk_mode, 0.15)

    for ticker in tickers:
        series = prices_by_ticker.get(ticker)
        if series is None or series.empty:
            raise RuntimeError(f"No prices for {ticker}")
        decision_date = as_of or max(series.index)
        memory, prices, events = build_memory(config, ticker, decision_date, history)
        decision_date = max(day for day in prices.index if day <= decision_date)
        previous_return = 0.0
        prior_days = [day for day in prices.index if day < decision_date]
        if prior_days:
            previous_return = float(prices.loc[decision_date] / prices.loc[prior_days[-1]] - 1)

        topics = _recent_topics(events, decision_date, ticker, lookback_days)
        memories = memory.retrieve(decision_date, topics, top_k, ticker=ticker)
        outcome = memory.recall(topics)
        prompt_memories = _prompt_memories(memories)
        if outcome is not None:
            prompt_memories.append({
                "id": f"outcome:{outcome['date']}",
                "source": "decision_history",
                "date": outcome["date"],
                "text": f"Prior similar decision: {outcome['action']}; realized return: {outcome['realized_return']:.6f}",
            })

        try:
            result = llm.decide("ours", ticker, decision_date.isoformat(), previous_return, prompt_memories, risk_mode=config.risk_mode)
        except TypeError:
            result = llm.decide("ours", ticker, decision_date.isoformat(), previous_return, prompt_memories)
        action = result["action"]

        # Risk profile post-processing (mirrors engine._signal + conservative override)
        if memories:
            avg_sentiment = sum(event.sentiment for event in memories) / len(memories)
            if config.risk_mode == "conservative" and abs(avg_sentiment) < risk_threshold:
                action = "Hold"
            elif config.risk_mode == "aggressive" and abs(avg_sentiment) >= risk_threshold:
                pass  # LLM decision stands; aggressive lowers the buy/sell trigger
            elif config.risk_mode == "standard" and abs(avg_sentiment) < risk_threshold:
                pass  # LLM decision stands

        decisions[ticker] = action
        detail = {
            "ticker": ticker,
            "date": decision_date.isoformat(),
            "decision": action,
            "reasoning": result.get("reasoning", ""),
            "cited_memory_ids": result.get("cited_memory_ids", []),
            "llm_status": result.get("llm_status", ""),
            "previous_return": previous_return,
            "topics": sorted(topics),
            "model": llm.model,
            "memory_count": len(memories),
            "risk_mode": config.risk_mode,
        }
        details.append(detail)
        new_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "date": decision_date.isoformat(),
            "action": action,
            "topics": sorted(topics),
            "realized_return": None,
            "model": llm.model,
            "risk_mode": config.risk_mode,
        })

    append_history(history_path, new_history)
    return decisions, details, new_history


def write_decisions(path: Path, details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "decision", "date", "reasoning", "cited_memory_ids", "llm_status", "model", "risk_mode"])
        writer.writeheader()
        for detail in details:
            row = {key: detail.get(key, "") for key in writer.fieldnames}
            row["cited_memory_ids"] = ";".join(str(item) for item in detail.get("cited_memory_ids", []))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Groq trading decisions and optionally execute via Alpaca")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (Groq or OpenRouter, default: %(default)s)")
    parser.add_argument("--provider", choices=["auto", "groq", "openrouter"], default="auto", help="Primary provider with automatic failover (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live"))
    parser.add_argument("--history", type=Path, default=Path("artifacts/live/decision_history.jsonl"))
    parser.add_argument("--audit-path", type=Path, default=Path("artifacts/paper_trading/orders.jsonl"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--execute", action="store_true", help="Submit orders to Alpaca (paper by default)")
    parser.add_argument("--live", action="store_true", help="Use the live Alpaca account; requires ALPACA_LIVE_CONFIRM=YES")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Refresh news/prices via scripts/fetch_real_data.py before deciding")
    parser.add_argument("--risk-mode", choices=["conservative", "standard", "aggressive"], default="standard", help="Risk profile for decision filtering (default: %(default)s)")
    args = parser.parse_args()

    tickers = tuple(item.strip().upper() for item in args.tickers.split(",") if item.strip())
    if args.live and not args.execute:
        parser.error("--live requires --execute")

    config = ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, ticker=tickers[0], top_k=args.top_k, risk_mode=args.risk_mode)

    if args.refresh:
        script = Path(__file__).resolve().parents[2] / "scripts" / "fetch_real_data.py"
        if not script.exists():
            raise RuntimeError(f"Cannot find {script} for --refresh")
        subprocess.run(
            [sys.executable, str(script), "--tickers", ",".join(tickers), "--data-dir", str(args.data_dir)],
            check=True,
        )

    llm = make_llm(args.model, args.temperature, config.llm_cache_dir, provider=args.provider)
    if not (llm.groq_key or llm.openrouter_key or llm.api_key):
        raise RuntimeError("Set GROQ_API_KEY (or GROQ_KEY) or OPEN_ROUTER_KEY before running decisions")

    decisions, details, _ = generate_decisions(
        config,
        tickers,
        llm,
        history_path=args.history,
        top_k=args.top_k,
        as_of=args.as_of,
        lookback_days=args.lookback_days,
    )

    decisions_path = args.output_dir / f"decisions_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    write_decisions(decisions_path, details)

    broker = PaperBroker(execute=args.execute, paper=not args.live)
    agent = PaperTradingAgent(
        TradingConfig(
            tickers=tickers,
            budget=args.budget,
            audit_path=args.audit_path,
            allow_short=args.allow_short,
        ),
        broker,
    )
    results = agent.execute_decisions(decisions)
    mode = "dry_run" if not args.execute else ("live_execute" if args.live else "paper_execute")
    print(json.dumps({
        "mode": mode,
        "risk_mode": args.risk_mode,
        "decisions_path": str(decisions_path),
        "model": llm.model,
        "llm_status": llm.stats.__dict__,
        "decisions": details,
        "orders": results,
    }, indent=2))


if __name__ == "__main__":
    main()
