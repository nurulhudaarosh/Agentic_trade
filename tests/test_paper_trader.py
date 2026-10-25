from pathlib import Path
from unittest.mock import MagicMock

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


def test_sell_without_position_skips_when_short_disabled(tmp_path: Path):
    config = TradingConfig(tickers=("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT"), budget=100, audit_path=tmp_path / "orders.jsonl", allow_short=False)
    agent = PaperTradingAgent(config, PaperBroker())
    results = agent.execute_decisions({"TSLA": "Sell"})
    assert results[0]["status"] == "skipped_no_position"
    assert results[0]["notional"] == 0.0


def test_sell_submits_short_when_short_enabled(tmp_path: Path):
    config = TradingConfig(tickers=("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT"), budget=100, audit_path=tmp_path / "orders.jsonl", allow_short=True)
    agent = PaperTradingAgent(config, PaperBroker())
    results = agent.execute_decisions({"TSLA": "Sell"})
    assert results[0]["status"] == "dry_run"
    assert results[0]["action"] == "sell"
    assert results[0]["notional"] == 20.0


def test_sell_closes_open_long_position(tmp_path: Path):
    broker = PaperBroker()
    broker.get_positions = MagicMock(return_value=[{"symbol": "TSLA", "qty": "10", "avg_entry_price": "200"}])
    config = TradingConfig(tickers=("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT"), budget=100, audit_path=tmp_path / "orders.jsonl")
    agent = PaperTradingAgent(config, broker)
    results = agent.execute_decisions({"TSLA": "Sell"})
    assert results[0]["status"] == "dry_run"
    assert results[0]["action"] == "close_position"


def test_portfolio_reconciliation():
    broker = PaperBroker()
    broker.get_positions = MagicMock(return_value=[
        {"symbol": "TSLA", "qty": "10", "avg_entry_price": "200"},
        {"symbol": "SPY", "qty": "50", "avg_entry_price": "500"},
    ])
    config = TradingConfig(tickers=("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT"))
    agent = PaperTradingAgent(config, broker)
    audit = agent.reconcile_portfolio()
    assert "TSLA" in audit["monitored_positions"]
    assert "SPY" in audit["unmonitored_positions"]
