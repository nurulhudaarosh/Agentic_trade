from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from agentic_trade.engine import ExperimentConfig
from agentic_trade.live_trader import generate_decisions
from agentic_trade.paper_trader import PaperBroker


class FakeStats:
    def __init__(self) -> None:
        self.__dict__.update({"calls": 1, "cache_hits": 0, "failures": 0, "parse_failures": 0, "last_error": None})


class FakeLLM:
    model = "fake-groq-model"

    def __init__(self) -> None:
        self.stats = FakeStats()
        self.calls = []

    def decide(self, system_name, ticker, decision_date, momentum, memories):
        self.calls.append(ticker)
        return {
            "action": "Buy",
            "reasoning": "fake",
            "cited_memory_ids": [m["id"] for m in memories[:1]],
            "llm_status": "success",
        }


def _write_market_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([
        {"date": "2026-06-01", "ticker": "TSLA", "source_type": "news", "text": "Tesla earnings profit beat", "summary": ""},
        {"date": "2026-06-02", "ticker": "TSLA", "source_type": "news", "text": "Tesla guidance raised", "summary": ""},
        {"date": "2026-06-02", "ticker": "MSFT", "source_type": "news", "text": "Microsoft cloud growth", "summary": ""},
    ]).to_csv(data_dir / "events.csv", index=False)
    pd.DataFrame([
        {"date": "2026-06-01", "ticker": "TSLA", "close": 100.0},
        {"date": "2026-06-02", "ticker": "TSLA", "close": 101.0},
        {"date": "2026-06-03", "ticker": "TSLA", "close": 102.0},
        {"date": "2026-06-01", "ticker": "MSFT", "close": 50.0},
        {"date": "2026-06-02", "ticker": "MSFT", "close": 51.0},
        {"date": "2026-06-03", "ticker": "MSFT", "close": 52.0},
    ]).to_csv(data_dir / "prices.csv", index=False)
    (data_dir / "relationships.csv").write_text('ticker,related_tickers\nTSLA,"MSFT"\nMSFT,"TSLA"\n')
    return data_dir


def test_generate_decisions_uses_llm_and_persists_history(tmp_path: Path):
    data_dir = _write_market_data(tmp_path)
    config = ExperimentConfig(data_dir=data_dir, output_dir=tmp_path / "out")
    llm = FakeLLM()
    history_path = tmp_path / "history.jsonl"

    decisions, details, new_history = generate_decisions(
        config,
        ("TSLA", "MSFT"),
        llm,
        history_path=history_path,
        as_of=date(2026, 6, 3),
    )

    assert decisions == {"TSLA": "Buy", "MSFT": "Buy"}
    assert llm.calls == ["TSLA", "MSFT"]
    assert len(details) == 2
    assert len(new_history) == 2
    assert history_path.exists()
    assert "reasoning" in details[0]


def test_live_broker_requires_explicit_confirmation(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE_CONFIRM", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_LIVE_CONFIRM"):
        PaperBroker(execute=True, paper=False)
