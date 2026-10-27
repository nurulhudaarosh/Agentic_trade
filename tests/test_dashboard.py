from __future__ import annotations

from agentic_trade.dashboard_server import (
    get_benchmarks_data,
    get_equity_data,
    get_evidence_data,
    get_orders_data,
    get_summary_data,
)


def test_dashboard_summary_returns_expected_structure():
    summary = get_summary_data()
    assert "project" in summary
    assert "tickers" in summary
    assert len(summary["tickers"]) > 0


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

