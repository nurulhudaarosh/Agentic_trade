from __future__ import annotations

import numpy as np
import pytest

from agentic_trade import algorithms as A


pytestmark = pytest.mark.skipif(not A.DAILY_PATH.exists(), reason="study results not generated")


def test_board_lists_every_algorithm_with_out_of_sample_record():
    b = A.algorithm_board("NVDA")
    assert len(b["rows"]) >= 12  # 12 study algorithms, plus the extension models once they have been run
    assert sum(b["votes"].values()) == len(b["rows"]) - 1  # the Buy & Hold benchmark does not vote
    for r in b["rows"]:
        assert r["today"]["action"] in {"Buy", "Sell", "Hold"}
        assert r["record"]["decisions"] > 100
        assert 0 <= r["record"]["hit_rate"] <= 1


def test_experience_log_is_consistent():
    e = A.experience("TSLA", "MA-Crossover", limit=20)
    assert len(e["log"]) == 20
    for x in e["log"]:
        assert x["next_date"] > x["date"]
        if x["held"] != "Flat":
            sign = 1 if x["held"] == "Long" else -1
            assert x["right"] == (np.sign(x["next_return"]) == sign)
    assert len(e["curve"]["dates"]) == len(e["curve"]["algo"]) == len(e["curve"]["buy_hold"])


def test_q_learning_agent_shapes_and_determinism():
    a = A.q_learning_agent("AAPL", epochs=5)
    b = A.q_learning_agent("AAPL", epochs=5)
    assert a["metrics"] == b["metrics"]
    assert len(a["learning_curve"]) == 5
    assert len(a["q_table"]) == A.N_STATES
    assert len(a["equity"]["dates"]) == len(a["equity"]["agent"])
    assert a["today"]["action"] in A.ACTIONS
    assert a["test_start"] >= "2025-12-01"


def test_q_learning_does_not_peek(monkeypatch):
    """Changing the last few days of prices must not change any earlier result."""
    base = A.q_learning_agent("MSFT", epochs=3)
    close = A.close_panel().copy()
    close.loc[close.index[-5]:, "MSFT"] *= 1.5
    A._rl_cache.clear()
    monkeypatch.setattr(A, "close_panel", lambda: close)
    shocked = A.q_learning_agent("MSFT", epochs=3)
    A._rl_cache.clear()
    assert base["learning_curve"] == shocked["learning_curve"]
    assert base["equity"]["agent"][:-8] == shocked["equity"]["agent"][:-8]
    assert base["equity"]["agent"][-1] != shocked["equity"]["agent"][-1]


def test_bad_inputs():
    with pytest.raises(ValueError):
        A.algorithm_board("XYZ")
    with pytest.raises(ValueError):
        A.q_learning_agent("TSLA", epochs=0)
