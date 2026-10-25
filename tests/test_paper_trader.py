from pathlib import Path

import pytest

from agentic_trade.paper_trader import PaperBroker, PaperTradingAgent, TradingConfig


def test_buy_orders_are_capped_and_holds_are_not_submitted(tmp_path: Path):
    config = TradingConfig(tickers=("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT"), budget=100, audit_path=tmp_path / "orders.jsonl")
    results = PaperTradingAgent(config, PaperBroker()).execute_decisions({"TSLA": "Buy", "AAPL": "Hold"})
    assert results[0]["status"] == "dry_run"
    assert results[0]["notional"] == 20
    assert results[1]["status"] == "not_submitted"
    assert len(results) == 5
    assert len((tmp_path / "orders.jsonl").read_text().splitlines()) == 5


def test_invalid_budget_is_rejected():
    agent = PaperTradingAgent(TradingConfig(budget=0), PaperBroker())
    with pytest.raises(ValueError):
        agent.build_orders({"TSLA": "Buy"})
