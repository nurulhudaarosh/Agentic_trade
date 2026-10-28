from datetime import date

import numpy as np
import pandas as pd
import pytest

from agentic_trade.study.backtest import apply_costs, positions_from_signal, split_of
from agentic_trade.study.data import assign_trading_day
from agentic_trade.study.evaluation import holm, metrics, paired_tests
from agentic_trade.study.memory import Credibility, MemoryConfig, MemoryStore, categorize


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_assign_trading_day_respects_close():
    days = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]
    stamps = pd.Series(pd.to_datetime([
        "2026-03-02T20:59:00Z",  # 15:59 ET -> same day
        "2026-03-02T21:01:00Z",  # 16:01 ET -> next day
        "2026-03-01T15:00:00Z",  # Sunday -> Monday
    ], utc=True))
    assert list(assign_trading_day(stamps, days)) == [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 2)]


def test_finmem_layers_decay_at_different_rates():
    store = MemoryStore("TSLA", MemoryConfig.fin_mem(), dim=3)
    store.add("news", 0, "shallow", unit(1, 0, 0), 0.5, "a", "other")
    store.add("10k", 0, "deep", unit(1, 0, 0), 0.5, "sec", "filing")
    rec = store._recency(np.array([0, 1]), now=30)
    assert rec[0] == pytest.approx(np.exp(-30 / 14))
    assert rec[1] == pytest.approx(np.exp(-30 / 365))


def test_finmem_purges_stale_shallow_memories_but_keeps_filings():
    store = MemoryStore("TSLA", MemoryConfig.fin_mem(), dim=3)
    store.add("news", 0, "shallow", unit(1, 0, 0), 0.5, "a", "other")
    store.add("10q", 0, "intermediate", unit(0, 1, 0), 0.5, "sec", "filing")
    store.maintain(now=60)
    assert not store.active[0] and store.active[1]


def test_retrieval_is_point_in_time_and_per_layer():
    store = MemoryStore("TSLA", MemoryConfig.fin_mem(top_k=1), dim=3)
    store.add("old", 0, "shallow", unit(1, 0, 0), 0.5, "a", "other")
    store.add("future", 10, "shallow", unit(1, 0, 0), 0.5, "a", "other")
    store.add("10k", 0, "deep", unit(1, 0, 0), 0.1, "sec", "filing")
    ret = store.retrieve(5, unit(1, 0, 0))
    ids = {store.item_id[i] for i in ret.idx}
    assert ids == {"old", "10k"}


def test_access_counter_promotes_pivotal_memory():
    store = MemoryStore("TSLA", MemoryConfig.fin_mem(), dim=3)
    store.add("n", 0, "shallow", unit(1, 0, 0), 0.5, "a", "other")
    store.v[0] = 55.0
    store.reinforce(np.array([0]), np.array([True]), now=3)
    assert store.layer[0] == 1 and store.t_reset[0] == 3 and store.promoted == 1


def test_dedup_merges_corroborating_story():
    store = MemoryStore("TSLA", MemoryConfig.ours(), dim=3)
    store.add("a", 0, "shallow", unit(1, 0.01, 0), 0.6, "x", "other")
    store.add("b", 1, "shallow", unit(1, 0.02, 0), 0.4, "y", "other")
    assert store.n == 1 and store.corrob[0] == 2 and store.merged == 1


def test_contradiction_retires_stale_opposite_memory():
    store = MemoryStore("TSLA", MemoryConfig.ours(dedup=False), dim=3)
    store.add("up", 0, "shallow", unit(1, 0.3, 0), 0.8, "x", "earnings")
    store.add("down", 1, "shallow", unit(1, 0.35, 0), -0.8, "x", "earnings")
    assert not store.active[0] and store.active[1] and store.retired == 1


def test_fin_mem_does_not_retire_or_merge():
    store = MemoryStore("TSLA", MemoryConfig.fin_mem(), dim=3)
    store.add("up", 0, "shallow", unit(1, 0, 0), 0.8, "x", "earnings")
    store.add("down", 1, "shallow", unit(1, 0, 0), -0.8, "x", "earnings")
    assert store.active[:2].all() and store.n == 2


def test_credibility_shrinks_toward_category_then_global():
    cred = Credibility(strength=8)
    assert cred.score("unknown", "analyst") == pytest.approx(0.5)
    for _ in range(40):
        cred.update("alice", "analyst", True)
    assert cred.score("alice", "analyst") > cred.score("bob", "analyst") > 0.5


def test_episodic_recall_needs_enough_similar_episodes():
    store = MemoryStore("TSLA", MemoryConfig.ours(), dim=3)
    for _ in range(4):
        store.remember_episode(unit(1, 0, 0), 0.5, -0.01)
    assert store.recall(unit(1, 0, 0)) == (None, 0)
    store.remember_episode(unit(1, 0, 0), 0.5, -0.01)
    edge, n = store.recall(unit(1, 0, 0))
    assert n == 5 and edge < 0


def test_hold_keeps_position_and_costs_charge_turnover():
    frame = pd.DataFrame({"ticker": "T", "date": [date(2026, 1, d) for d in (5, 6, 7, 8)],
                          "signal": [0.5, 0.0, -0.5, 0.0], "abstain": [False, False, False, True],
                          "next_return": [0.01, 0.01, -0.02, 0.03]})
    frame["position"] = positions_from_signal(frame, threshold=0.1)
    assert frame["position"].tolist() == [1.0, 1.0, -1.0, -1.0]
    net = apply_costs(frame, cost_bps=10)
    assert net.iloc[0] == pytest.approx(0.01 - 0.001)
    assert net.iloc[2] == pytest.approx(0.02 - 0.002)


def test_splits_do_not_overlap():
    assert split_of(date(2026, 2, 27)) == "train"
    assert split_of(date(2026, 3, 2)) == "validation"
    assert split_of(date(2026, 5, 1)) == "test"


def test_metrics_and_stats_helpers():
    r = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004] * 10)
    m = metrics(r)
    assert m["cumulative_return"] > 0 and m["max_drawdown"] <= 0
    res = paired_tests(r, r * 0.5)
    assert res["n"] == 50 and res["mean_diff_bps"] > 0
    adj = holm({"a": 0.01, "b": 0.04})
    assert adj["a"] == pytest.approx(0.02) and adj["b"] == pytest.approx(0.04)


def test_categorize_headlines():
    assert categorize("Morgan Stanley Maintains Overweight on Tesla, Raises Price Target") == "analyst"
    assert categorize("Netflix Q3 Earnings Beat Estimates") == "earnings"
