from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_trade import analytics


def test_pattern_analogs_shapes_and_no_lookahead():
    res = analytics.pattern_analogs("TSLA", window=20, horizon=10, k=8)
    assert len(res["current_path"]) == 20
    assert res["current_path"][-1] == 0.0
    assert res["forecast"]["action"] in {"Buy", "Sell", "Hold"}
    assert len(res["forecast"]["mean_path"]) == 10
    as_of = pd.Timestamp(res["as_of"])
    dates = analytics.close_panel().index
    for m in res["matches"]:
        assert len(m["window_path"]) == 20 and len(m["forward_path"]) == 10
        end = dates.get_loc(pd.Timestamp(m["end"]))
        # The analog's whole outcome is observed by today, and it does not overlap today's window.
        assert end + 10 <= dates.get_loc(as_of)
        assert end <= dates.get_loc(as_of) - 20


def test_universe_scope_draws_from_other_tickers():
    res = analytics.pattern_analogs("NVDA", scope="universe", k=12)
    assert {m["ticker"] for m in res["matches"]} - {"NVDA"}


def test_analog_backtest_matches_buy_and_hold_when_always_long(monkeypatch):
    monkeypatch.setattr(analytics, "_forecast", lambda paths, sims, thr: {"expected": 1.0, "p_up": 1.0, "action": "Buy"})
    bt = analytics.analog_backtest("MSFT", cost_bps=0.0)
    assert bt["equity"]["Analog strategy"] == pytest.approx(bt["equity"]["Buy & Hold"])
    assert len(bt["dates"]) == len(bt["equity"]["Buy & Hold"]) == len(bt["positions"])


def test_sweep_uses_common_test_period():
    sweep = analytics.analog_sweep("AAPL", scope="ticker", windows=(10, 20), horizons=(5, 10))
    assert len(sweep["cells"]) == 4
    assert all(c["metrics"] is not None for c in sweep["cells"])


def test_portfolio_distribution_weights_and_hypothetical():
    state = {"equity": 100000.0, "cash": 60000.0, "positions": [
        {"ticker": "TSLA", "side": "Long", "shares": 100, "current_price": 300.0},
        {"ticker": "AMD", "side": "Short", "shares": 50, "current_price": 200.0}]}
    res = analytics.portfolio_distribution(state, days=10, paths=500)
    weights = {a["ticker"]: a["weight"] for a in res["allocation"]}
    assert weights == {"TSLA": 0.3, "AMD": -0.1, "Cash": 0.6}
    assert sum(a["risk_share"] for a in res["allocation"]) == pytest.approx(1.0, abs=1e-3)
    assert len(res["monte_carlo"]["fan"]["p50"]) == 11
    assert not res["hypothetical"]
    empty = analytics.portfolio_distribution({"equity": 50000.0, "positions": []}, ticker="AMD", paths=200)
    assert empty["hypothetical"] and empty["allocation"][0]["ticker"] == "AMD"


def test_risk_and_correlation():
    risk = analytics.risk_report("COIN")
    assert sum(risk["histogram"]["counts"]) == risk["n"]
    assert risk["stats"]["cvar_95"] >= risk["stats"]["var_95"] > 0
    corr = analytics.correlation_matrix(60)
    m = np.array(corr["matrix"], dtype=float)
    assert np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T)


def test_bad_inputs_raise_value_error():
    with pytest.raises(ValueError):
        analytics.pattern_analogs("NOPE")
    with pytest.raises(ValueError):
        analytics.pattern_analogs("TSLA", window=2)


def test_allocation_plan_respects_reserve_and_cap():
    for method in ("inverse_vol", "min_variance", "equal", "signal_tilt"):
        plan = analytics.allocation_plan(10000, reserve_pct=20, max_position_pct=12, method=method,
                                         holdings={"TSLA": 3000}, signals={"TSLA": "Sell"})
        weights = [r["weight"] for r in plan["rows"]]
        assert max(weights) <= 0.12 + 1e-6
        assert sum(weights) <= 0.8 + 1e-3
        assert plan["cash"]["weight"] >= 0.2 - 1e-6
        tsla = next(r for r in plan["rows"] if r["ticker"] == "TSLA")
        assert tsla["delta_usd"] == pytest.approx(tsla["target_usd"] - 3000, abs=0.02)
    tilt = analytics.allocation_plan(10000, method="signal_tilt", signals={"TSLA": "Sell"})
    assert next(r for r in tilt["rows"] if r["ticker"] == "TSLA")["weight"] == 0
    assert set(plan["backtest"]["metrics"]) >= {"Equal weight", "Minimum variance"}
