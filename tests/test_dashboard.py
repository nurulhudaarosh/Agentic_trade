from __future__ import annotations

from pathlib import Path
from agentic_trade.dashboard_server import (
    PortfolioManager,
    get_benchmarks_data,
    get_equity_data,
    get_evidence_data,
    get_market_data,
    get_orders_data,
    get_summary_data,
    get_technical_analysis,
)


def test_dashboard_summary_returns_expected_structure():
    summary = get_summary_data()
    assert "project" in summary
    assert "tickers" in summary
    assert len(summary["tickers"]) > 0
    assert "latest_prices" in summary


def test_dashboard_equity_offline():
    eq = get_equity_data(dataset="offline")
    assert "series" in eq
    assert "ours" in eq["series"]
    assert len(eq["series"]["ours"]) > 0
    assert "metrics" in eq
    assert "ours" in eq["metrics"]


def test_dashboard_evidence_filtering():
    ev = get_evidence_data(dataset="offline", limit=10)
    assert "items" in ev
    assert len(ev["items"]) <= 10
    assert ev["total"] > 0


def test_dashboard_benchmarks():
    bm = get_benchmarks_data()
    assert "table2" in bm
    assert "table7" in bm
    assert len(bm["table7"]) > 0


def test_dashboard_orders():
    res = get_orders_data()
    assert "orders" in res
    assert "decisions" in res
    assert "portfolio" in res


def test_market_data_indicators_and_candles():
    m = get_market_data("TSLA")
    assert m["ticker"] == "TSLA"
    assert len(m["dates"]) > 0
    assert len(m["prices"]) == len(m["dates"])
    assert "indicators" in m
    ind = m["indicators"]
    assert "sma_20" in ind
    assert "sma_50" in ind
    assert "rsi_14" in ind
    assert "macd" in ind
    assert "bb_upper" in ind
    assert len(m["candles"]) == len(m["dates"])


def test_technical_analysis_signals():
    ta = get_technical_analysis("TSLA")
    assert ta["ticker"] == "TSLA"
    assert "recommendation" in ta
    assert ta["recommendation"] in ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
    assert "signals" in ta
    assert "rsi" in ta["signals"]
    assert "macd" in ta["signals"]
    assert "moving_averages" in ta["signals"]


def test_portfolio_manager_lifecycle(tmp_path: Path):
    mgr = PortfolioManager(tmp_path / "portfolio.json")
    mgr.reset(initial_capital=50000.0)
    state = mgr.get_state()
    assert state["cash"] == 50000.0
    assert state["equity"] == 50000.0
    assert len(state["positions"]) == 0

    # Execute Buy
    res = mgr.execute_order(ticker="TSLA", action="Buy", shares=10, order_type="Market")
    assert res["success"] is True
    assert "TSLA" in mgr.positions
    assert mgr.positions["TSLA"]["shares"] == 10
    assert mgr.cash < 50000.0

    # Execute Close
    res_close = mgr.close_position("TSLA")
    assert res_close["success"] is True
    assert "TSLA" not in mgr.positions

    # Update settings
    res_set = mgr.update_settings({"risk_profile": "conservative"})
    assert res_set["success"] is True
    assert mgr.settings["risk_profile"] == "conservative"
