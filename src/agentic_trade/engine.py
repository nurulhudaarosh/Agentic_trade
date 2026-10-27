from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from .memory import FinMem, MemoryEvent, NoMemory, OursMemory, PlainRAG, extract_topics
from .llm import OpenRouterClient

@dataclass
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts/offline")
    ticker: str = "TSLA"
    start_date: date = date(2026, 6, 1)
    train_end: date = date(2026, 6, 15)
    test_start: date = date(2026, 6, 16)
    end_date: date = date(2026, 7, 9)
    top_k: int = 5
    trial_id: int = 0
    temperature: float = 0.7
    backbone: str = "deterministic"
    risk_mode: str = "standard"
    use_provenance: bool = True
    use_reconciliation: bool = True
    use_outcome_recall: bool = True
    use_topic_matching: bool = True
    llm_cache_dir: Path = Path("artifacts/llm_cache")

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        converted = {}
        for k, v in data.items():
            if k in ("start_date", "train_end", "test_start", "end_date") and isinstance(v, str):
                converted[k] = date.fromisoformat(v)
            elif k in ("data_dir", "output_dir", "llm_cache_dir") and isinstance(v, str):
                converted[k] = Path(v)
            else:
                converted[k] = v
        return cls(**converted)

    def to_dict(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "end_date": self.end_date.isoformat(),
            "top_k": self.top_k,
            "trial_id": self.trial_id,
            "temperature": self.temperature,
            "backbone": self.backbone,
            "risk_mode": self.risk_mode,
            "use_provenance": self.use_provenance,
            "use_reconciliation": self.use_reconciliation,
            "use_outcome_recall": self.use_outcome_recall,
            "use_topic_matching": self.use_topic_matching,
            "llm_cache_dir": str(self.llm_cache_dir),
        }

class ExperimentEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.systems = {name: cls() for name, cls in [("no_memory", NoMemory), ("plain_rag", PlainRAG), ("fin_mem", FinMem), ("ours", OursMemory)]}
        self.events: list[MemoryEvent] = []
        self.prices = pd.DataFrame()
        self.results: dict[str, list[dict]] = {name: [] for name in self.systems}
        self.evidence: list[dict] = []
        self.llm = None if self.config.backbone == "deterministic" else OpenRouterClient(
            model=self.config.backbone,
            temperature=self.config.temperature,
            cache_dir=self.config.llm_cache_dir,
        )

    def _load_relationship_map(self) -> dict[str, set[str]]:
        path = self.config.data_dir / "relationships.csv"
        if not path.exists():
            return {}
        frame = pd.read_csv(path)
        missing = {"ticker", "related_tickers"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        mapping: dict[str, set[str]] = {}
        for _, row in frame.iterrows():
            ticker = str(row["ticker"]).strip().upper()
            related = {
                item.strip().upper()
                for item in str(row["related_tickers"]).split(",")
                if item.strip()
            }
            if ticker:
                mapping[ticker] = related
        return mapping

    def load_data(self) -> None:
        frames = []
        for path in sorted(self.config.data_dir.glob("*.csv")):
            if path.name in {"prices.csv", "relationships.csv"}:
                continue
            frame = pd.read_csv(path)
            required = {"date", "ticker", "source_type", "text"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
            frames.append(frame)
        if frames:
            raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "ticker", "source_type", "text"])
            relationship_map = self._load_relationship_map()
            if "related_tickers" in raw:
                existing = raw["related_tickers"].fillna("").map(
                    lambda value: {item.strip().upper() for item in str(value).split(",") if item.strip()}
                )
            else:
                existing = pd.Series([set() for _ in range(len(raw))], index=raw.index)
            raw["related_tickers"] = [
                set(related) | relationship_map.get(str(ticker).strip().upper(), set())
                for ticker, related in zip(raw["ticker"], existing)
            ]
            raw = raw[raw["ticker"].eq(self.config.ticker) | raw["related_tickers"].map(lambda tickers: self.config.ticker in tickers)]
            for index, row in raw.iterrows():
                source_name = str(row.get("source_name", "") or "")
                source_category = str(row.get("source_category", "") or "")
                self.events.append(MemoryEvent(
                    event_id=str(index),
                    event_date=row["date"],
                    ticker=str(row["ticker"]),
                    source_type=str(row["source_type"]),
                    text=str(row["text"]),
                    summary=str(row.get("summary", "") or ""),
                    related_tickers=set(row["related_tickers"]),
                    source_name=source_name,
                    source_category=source_category,
                ))
        self.events.sort(key=lambda event: (event.event_date, event.event_id))
        self.prices = self._load_prices()

    def _learn_provenance(self) -> None:
        ours = self.systems["ours"]
        if not isinstance(ours, OursMemory):
            return
        prices = self.prices["close"]
        outcomes: dict[str, list[bool]] = {}
        for event in self.events:
            if event.event_date > self.config.train_end or event.event_date not in prices.index:
                continue
            future = [day for day in prices.index if day > event.event_date]
            if not future:
                continue
            direction = float(prices.loc[future[0]] / prices.loc[event.event_date] - 1)
            if direction == 0 or event.sentiment == 0:
                continue
            source = event.source_name if event.source_name and event.source_name != "unknown" else event.source_category
            outcomes.setdefault(source, []).append(event.sentiment * direction > 0)
        for source, hits in outcomes.items():
            if len(hits) >= 2:
                # Beta-style smoothing toward the neutral 0.7 prior keeps tiny
                # source samples from producing extreme reliability values.
                smoothed = (sum(hits) + 2 * 0.7) / (len(hits) + 2)
                ours.learn_source_reliability(source, smoothed)

    def _load_prices(self) -> pd.DataFrame:
        path = self.config.data_dir / "prices.csv"
        if path.exists():
            prices = pd.read_csv(path, parse_dates=["date"])
            prices["date"] = prices["date"].dt.date
            if "ticker" in prices.columns:
                prices = prices[prices["ticker"].eq(self.config.ticker)]
            prices = prices.drop_duplicates(subset=["date"])
            if prices.empty:
                raise ValueError(f"No prices found for ticker {self.config.ticker}")
            return prices.set_index("date")["close"].astype(float).to_frame()
        try:
            import yfinance as yf
            data = yf.download(self.config.ticker, start=self.config.start_date, end=self.config.end_date + timedelta(days=1), auto_adjust=False, progress=False)
            if data.empty:
                raise ValueError("No price data returned")
            close = data["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            prices = close.rename("close").to_frame()
            prices.index = pd.to_datetime(prices.index).date
            return prices
        except Exception as exc:
            raise RuntimeError("Price data unavailable; provide data/prices.csv for an offline run") from exc

    def _events_for(self, current_date: date) -> list[MemoryEvent]:
        return [event for event in self.events if event.event_date <= current_date]

    def _ingest_event(self, system: object, event: MemoryEvent) -> None:
        if system.name != "ours":
            system.add(event)
            return
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
            source_category=event.source_category,
        )
        if not self.config.use_reconciliation:
            copy.active = True
            system.events.append(copy)
        else:
            system.add(copy)

    def _signal(self, memories: list[MemoryEvent], previous_return: float) -> str:
        sentiment = sum(event.sentiment for event in memories) / len(memories) if memories else 0.0
        threshold = {"conservative": 0.25, "standard": 0.15, "aggressive": 0.05}.get(self.config.risk_mode, 0.15)
        if sentiment > threshold or previous_return > threshold / 10:
            return "Buy"
        if sentiment < -threshold or previous_return < -threshold / 10:
            return "Sell"
        return "Hold"

    def run(self) -> dict[str, dict]:
        if self.prices.empty:
            self.load_data()
        if self.config.use_provenance:
            self._learn_provenance()
        for event in self.events:
            if event.event_date <= self.config.train_end:
                for system in self.systems.values():
                    self._ingest_event(system, event)
        test_dates = [d for d in self.prices.index if self.config.test_start <= d <= self.config.end_date]
        for index, current_date in enumerate(test_dates):
            for event in self.events:
                if event.event_date == current_date:
                    for system in self.systems.values():
                        self._ingest_event(system, event)
            prev_date = self.prices.index[self.prices.index.get_loc(current_date) - 1] if self.prices.index.get_loc(current_date) > 0 else current_date
            previous_return = float(self.prices.loc[current_date, "close"] / self.prices.loc[prev_date, "close"] - 1) if current_date != prev_date else 0.0
            if index > 0:
                ours = self.systems["ours"]
                if isinstance(ours, OursMemory):
                    ours.resolve(test_dates[index - 1], previous_return)
            next_date = test_dates[index + 1] if index + 1 < len(test_dates) else None
            realized = float(self.prices.loc[next_date, "close"] / self.prices.loc[current_date, "close"] - 1) if next_date else 0.0
            topics = extract_topics(" ".join(event.text for event in self._events_for(current_date)))
            for name, system in self.systems.items():
                query_topics = topics if self.config.use_topic_matching else set()
                memories = system.retrieve(current_date, query_topics, self.config.top_k, ticker=self.config.ticker)
                outcome = None
                if name == "ours" and self.config.use_outcome_recall:
                    outcome = system.recall(topics)
                prompt_memories = [{"id": event.event_id, "source": event.source_type, "date": event.event_date.isoformat(), "text": event.text, "relationship": ",".join(sorted(event.related_tickers))} for event in memories]
                if outcome is not None:
                    prompt_memories.append({"id": f"outcome:{outcome['date']}", "source": "decision_history", "date": outcome["date"], "text": f"Prior similar decision: {outcome['action']}; realized return: {outcome['realized_return']:.6f}"})
                action = self._signal(memories, previous_return) if name != "no_memory" else ("Buy" if previous_return > 0.01 else "Sell" if previous_return < -0.01 else "Hold")
                cited = [event.event_id for event in memories[:3]]
                llm_result = {"action": action, "reasoning": "deterministic policy", "cited_memory_ids": cited, "llm_status": "disabled"}
                if self.llm is not None:
                    try:
                        llm_result = self.llm.decide(name, self.config.ticker, current_date.isoformat(), previous_return, prompt_memories, risk_mode=self.config.risk_mode)
                    except TypeError:
                        llm_result = self.llm.decide(name, self.config.ticker, current_date.isoformat(), previous_return, prompt_memories)
                    action = llm_result["action"]
                    cited = llm_result.get("cited_memory_ids", cited)
                if self.config.risk_mode == "conservative" and memories and abs(sum(event.sentiment for event in memories) / len(memories)) < 0.25:
                    action = "Hold"
                position = {"Buy": 1.0, "Sell": -1.0, "Hold": 0.0}[action]
                if self.config.risk_mode == "conservative":
                    position *= 0.5
                daily_return = position * realized
                row = {"date": current_date.isoformat(), "decision": action, "position": position, "price": float(self.prices.loc[current_date, "close"]), "next_return": realized, "daily_return": daily_return, "cited_memory_ids": cited, "reasoning": llm_result["reasoning"], "llm_status": llm_result["llm_status"], "trial_id": self.config.trial_id, "risk_mode": self.config.risk_mode, "top_k": self.config.top_k}
                self.results[name].append(row)
                for rank, event in enumerate(memories, 1):
                    self.evidence.append({"date": current_date.isoformat(), "memory_id": event.event_id, "layer": event.layer, "source": event.source_name if event.source_name and event.source_name != "unknown" else event.source_category, "source_type": event.source_type, "text": event.text[:280], "topic": ";".join(sorted(event.topic_tags)), "related_tickers": ";".join(sorted(event.related_tickers)), "reliability": event.source_reliability if self.config.use_provenance else 1.0, "active": event.active, "rank": rank, "score": float(event.source_reliability), "cited": event.event_id in cited, "decision": action, "realized_return": realized, "system": name, "ticker": self.config.ticker, "outcome_recalled": outcome is not None})
                if name == "ours":
                    system.record_decision({"date": current_date.isoformat(), "action": action, "topics": sorted(topics), "realized_return": None})
        metrics = {name: self._metrics(rows) for name, rows in self.results.items()}
        ours = self.systems["ours"]
        if isinstance(ours, OursMemory):
            metrics["ours"]["facts_retired"] = len(ours.retired_events)
            metrics["ours"]["decisions_recalled"] = ours.recalled_count
        if self.llm is not None:
            llm_stats = self.llm.stats.__dict__
            for values in metrics.values():
                values["llm_calls"] = llm_stats["calls"]
                values["llm_cache_hits"] = llm_stats["cache_hits"]
                values["llm_failures"] = llm_stats["failures"]
                values["llm_parse_failures"] = llm_stats["parse_failures"]
        metrics["buy_and_hold"] = self._metrics([{"daily_return": row["next_return"], "decision": "Buy"} for row in self.results["no_memory"]])
        return metrics

    @staticmethod
    def _metrics(rows: list[dict]) -> dict:
        returns = pd.Series([float(row.get("daily_return", 0.0)) for row in rows], dtype=float)
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
        volatility = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) else 0.0
        return {"cumulative_return": float(equity.iloc[-1] - 1) if not equity.empty else 0.0, "sharpe_ratio": sharpe, "annualized_volatility": volatility, "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0, "hold_rate": float(sum(row.get("decision") == "Hold" for row in rows) / len(rows)) if rows else 0.0}

    def write_artifacts(self, metrics: dict[str, dict]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in self.results.items():
            pd.DataFrame(rows).to_csv(self.config.output_dir / f"{name}_daily.csv", index=False)
        pd.DataFrame(self.evidence).to_csv(self.config.output_dir / "evidence_log.csv", index=False)
        (self.config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        self._plot(metrics)
        report = ["# Agentic Trade Research Run", "", "Deterministic offline experiment artifacts.", "", "## Metrics", "", "| System | Return | Sharpe | Volatility | Max Drawdown | Hold Rate |", "|---|---:|---:|---:|---:|---:|"]
        report += [f"| {name} | {values.get('cumulative_return', 0):.4f} | {values.get('sharpe_ratio', 0):.2f} | {values.get('annualized_volatility', 0):.4f} | {values.get('max_drawdown', 0):.4f} | {values.get('hold_rate', 0):.2f} |" for name, values in metrics.items()]
        report += ["", "## Caveats", "", "- Offline run uses checked-in data and deterministic signal policy.", "- No transaction costs, slippage, or statistical significance claim.", "- Live API and LLM adapters are intentionally opt-in."]
        (self.config.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    def _plot(self, metrics: dict[str, dict]) -> None:
        figure_dir = Path("paper/figures")
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rows in self.results.items():
            if name == "buy_and_hold":
                continue
            series = pd.Series([row["daily_return"] for row in rows]).add(1).cumprod()
            ax.plot(series.values, label=name)
        ax.set(title="Cumulative Return by Memory System", xlabel="Test observation", ylabel="Growth of $1")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(self.config.output_dir / "cumulative_returns.png", dpi=180)
        fig.savefig(figure_dir / "cumulative_returns.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(metrics)
        ax.bar(names, [metrics[name].get("cumulative_return", 0.0) for name in names])
        ax.set(title="Cumulative Return Comparison", ylabel="Return")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(self.config.output_dir / "metric_comparison.png", dpi=180)
        fig.savefig(figure_dir / "metric_comparison.png", dpi=180)
        plt.close(fig)

def load_all_events(data_dir: Path | str, tickers: Iterable[str]) -> list[MemoryEvent]:
    """Load and merge events for a ticker universe without the single-ticker filter."""
    merged: dict[tuple, MemoryEvent] = {}
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        engine = ExperimentEngine(ExperimentConfig(data_dir=Path(data_dir), ticker=ticker))
        engine.load_data()
        for event in engine.events:
            key = (event.ticker, event.event_date, event.source_type, event.text)
            if key in merged:
                merged[key].related_tickers |= event.related_tickers
                merged[key].topic_tags |= event.topic_tags
            else:
                merged[key] = event
    return sorted(merged.values(), key=lambda event: (event.event_date, event.ticker, event.event_id))


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, dict]:
    engine = ExperimentEngine(config or ExperimentConfig())
    metrics = engine.run()
    engine.write_artifacts(metrics)
    return metrics
