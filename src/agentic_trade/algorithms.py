"""Algorithm board and a reinforcement-learning agent for the Insights page.

* ``algorithm_board``: for one ticker, what every algorithm from the study says on
  the latest decision day and its track record ("experience") in the walk-forward
  test, read from ``artifacts/study/rolling/daily.parquet``. Each row of that file
  is one decision: the signal, the position taken and the next-day return that
  followed, so the track record is made only of out-of-sample decisions.
* ``experience``: the raw decision log of one algorithm on one ticker.
* ``q_learning_agent``: a tabular Q-learning agent trained walk-forward on daily
  prices. It pre-trains on the period before the study's test window, then trades
  the test window greedily while it keeps learning online from each realised
  return (an update only uses returns that were already known at decision time).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analytics import _f, _fl, _perf, close_panel
from .study.data import UNIVERSE

ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "artifacts" / "study" / "rolling" / "daily.parquet"
EXT_PATH = DAILY_PATH.with_name("extensions_daily.parquet")

# Algorithms shown on the board, grouped by family (ids are used for UI text).
BOARD = [
    ("Buy & Hold", "buy_hold", "benchmark"),
    ("Momentum (1d)", "momentum_1d", "rule"),
    ("TS-Momentum (20d)", "momentum_20d", "rule"),
    ("Mean-Reversion (5d)", "mean_reversion", "rule"),
    ("MA-Crossover", "ma_crossover", "rule"),
    ("Daily News Sentiment", "news_sentiment", "news"),
    ("LogReg (price)", "logreg", "ml"),
    ("LGBM (price)", "lgbm", "ml"),
    ("Plain RAG", "plain_rag", "memory"),
    ("FinMem", "finmem", "memory"),
    ("Ours", "ours", "memory"),
    ("LGBM (price+Ours)", "lgbm_ours", "ml"),
    # Extension experiments (scripts/run_extensions.py), same protocol and costs.
    ("GRU (price)", "gru", "deep"),
    ("Transformer (price)", "transformer", "deep"),
    ("Transformer (price+memory)", "transformer_mem", "deep"),
    ("RL Q-learning", "rl", "rl"),
    ("Pattern analogs (kNN)", "analogs", "rule"),
    ("LGBM + memory risk overlay", "overlay", "hybrid"),
]

_lock = threading.Lock()
_cache: dict[str, Any] = {"stamp": None, "daily": None}
_rl_cache: dict[tuple, dict[str, Any]] = {}


def _daily() -> pd.DataFrame:
    if not DAILY_PATH.exists():
        raise ValueError("no study results yet: run scripts/run_study.py first")
    stamp = (DAILY_PATH.stat().st_mtime, EXT_PATH.stat().st_mtime if EXT_PATH.exists() else 0)
    with _lock:
        if _cache["stamp"] != stamp:
            cols = ["system", "ticker", "date", "next_date", "split", "signal", "abstain", "position", "next_return", "net"]
            df = pd.read_parquet(DAILY_PATH, columns=cols)
            if EXT_PATH.exists():
                ext = pd.read_parquet(EXT_PATH)
                ext["next_date"] = pd.NaT
                df = pd.concat([df, ext[cols]], ignore_index=True)
            df = df[df["system"].isin([b[0] for b in BOARD])].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["next_date"] = pd.to_datetime(df["next_date"]).dt.strftime("%Y-%m-%d")
            # Baseline rows in the study carry no next_date: take it from another algorithm's row
            # for the same stock and day, else from the next trading day in the log.
            known = df.dropna(subset=["next_date"]).drop_duplicates(["ticker", "date"]).set_index(["ticker", "date"])["next_date"]
            fill = pd.Series(list(zip(df["ticker"], df["date"])), index=df.index).map(known)
            df["next_date"] = df["next_date"].fillna(fill)
            df = df.sort_values(["system", "ticker", "date"])
            df["next_date"] = df["next_date"].fillna(df.groupby(["system", "ticker"])["date"].shift(-1))
            _cache.update(stamp=stamp, daily=df)
        return _cache["daily"]


def _action(position: float) -> str:
    return "Buy" if position > 0 else "Sell" if position < 0 else "Hold"


def _track(rows: pd.DataFrame) -> dict[str, Any]:
    """Out-of-sample record of a decision log (test split only)."""
    net = rows["net"].to_numpy(dtype=float)
    active = rows["position"].to_numpy() != 0
    right = np.sign(rows["position"].to_numpy()) == np.sign(rows["next_return"].to_numpy())
    last = rows.tail(30)
    last_active = last["position"].to_numpy() != 0
    last_right = np.sign(last["position"].to_numpy()) == np.sign(last["next_return"].to_numpy())
    return {
        **_perf(net),
        "decisions": int(len(rows)),
        "active_days": int(active.sum()),
        "hit_rate": _f(right[active].mean(), 4) if active.any() else None,
        "hit_rate_30": _f(last_right[last_active].mean(), 4) if last_active.any() else None,
        "buy_share": _f((rows["position"] > 0).mean(), 4),
        "sell_share": _f((rows["position"] < 0).mean(), 4),
    }


def algorithm_board(ticker: str = "TSLA") -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in UNIVERSE:
        raise ValueError(f"no study results for {ticker}; available: {', '.join(UNIVERSE)}")
    daily = _daily()
    sub = daily[daily["ticker"] == ticker]
    test = sub[sub["split"] == "test"]
    rows = []
    for name, aid, family in BOARD:
        d = sub[(sub["system"] == name) & sub["position"].notna()].sort_values("date")
        if d.empty:
            continue
        latest = d.iloc[-1]
        t = test[test["system"] == name].sort_values("date")
        equity = np.cumprod(1.0 + t["net"].to_numpy(dtype=float))
        step = max(1, len(equity) // 60)
        rows.append({
            "name": name, "id": aid, "family": family,
            "today": {"date": latest["date"], "action": "Hold" if bool(latest["abstain"]) else _action(latest["position"]),
                      "signal": _f(latest["signal"], 4), "abstain": bool(latest["abstain"])},
            "record": _track(t) if not t.empty else None,
            "spark": _fl(equity[::step], 4),
        })
    votes = {a: sum(1 for r in rows if r["family"] != "benchmark" and r["today"]["action"] == a) for a in ("Buy", "Hold", "Sell")}
    t_all = test[test["system"] == "Buy & Hold"]
    return {
        "ticker": ticker, "rows": rows, "votes": votes,
        "decision_date": sub["date"].max(),
        "test_range": [t_all["date"].min(), t_all["date"].max()] if not t_all.empty else None,
        "source": {"file": str(DAILY_PATH.relative_to(ROOT)), "rows": int(len(daily)),
                   "algorithms": int(daily["system"].nunique()),
                   "tickers": int(daily["ticker"].nunique()), "first": daily["date"].min(), "last": daily["date"].max()},
    }


def experience(ticker: str = "TSLA", system: str = "Ours", limit: int = 40) -> dict[str, Any]:
    ticker = ticker.upper()
    daily = _daily()
    d = daily[(daily["ticker"] == ticker) & (daily["system"] == system)].sort_values("date")
    if d.empty:
        raise ValueError(f"no decisions for {system} on {ticker}")
    test = d[d["split"] == "test"]
    bh = daily[(daily["ticker"] == ticker) & (daily["system"] == "Buy & Hold") & (daily["split"] == "test")].sort_values("date")
    log = []
    for r in d.tail(limit).iloc[::-1].itertuples():
        action = "Hold" if bool(r.abstain) else _action(r.position)
        # In the study "Hold" keeps yesterday's position, so also report what was actually held.
        held = "Long" if r.position > 0 else "Short" if r.position < 0 else "Flat"
        log.append({"date": r.date, "next_date": r.next_date, "split": r.split, "signal": _f(r.signal, 4), "action": action, "held": held,
                    "next_return": _f(r.next_return), "net": _f(r.net),
                    "right": None if r.position == 0 else bool(np.sign(r.position) == np.sign(r.next_return))})
    return {
        "ticker": ticker, "system": system, "log": log,
        "curve": {"dates": list(test["date"]),
                  "algo": _fl(np.cumprod(1.0 + test["net"].to_numpy(dtype=float)), 4),
                  "buy_hold": _fl(np.cumprod(1.0 + bh["net"].to_numpy(dtype=float)), 4)},
        "record": _track(test) if not test.empty else None,
    }


# --------------------------------------------------------------------------- reinforcement learning
RET_EDGES = (-0.02, 0.02)   # 5-day return: falling / flat / rising
RSI_EDGES = (35.0, 65.0)    # RSI 14: oversold / neutral / overbought
ACTIONS = ("Flat", "Long")  # stay in cash, or hold the stock


def _features(px: pd.Series) -> pd.DataFrame:
    r5 = px.pct_change(5)
    trend = (px > px.rolling(20).mean()).astype(int)
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    f = pd.DataFrame({"ret5": np.digitize(r5, RET_EDGES), "trend": trend, "rsi": np.digitize(rsi, RSI_EDGES),
                      "next_ret": px.pct_change().shift(-1)}, index=px.index)
    return f.iloc[20:]  # indicators need 20 days of history


def _state(ret5: float, trend: float, rsi: float, pos: int) -> int:
    return ((int(ret5) * 2 + int(trend)) * 3 + int(rsi)) * 2 + int(pos)


def _describe(s: int) -> dict[str, Any]:
    pos, s = s % 2, s // 2
    rsi, s = s % 3, s // 3
    trend, ret5 = s % 2, s // 2
    return {"ret5": ["down", "flat", "up"][ret5], "trend": ["below", "above"][trend],
            "rsi": ["low", "mid", "high"][rsi], "position": ACTIONS[pos]}


N_STATES = 3 * 2 * 3 * 2


def q_learning_agent(ticker: str = "TSLA", epochs: int = 30, alpha: float = 0.1, gamma: float = 0.9,
                     cost_bps: float = 5.0, test_start: str = "2025-12-01", seed: int = 7) -> dict[str, Any]:
    """Tabular Q-learning, long/flat, trained walk-forward on daily prices.

    State = 5-day move (3 levels) × price vs 20-day average (2) × RSI band (3) × current position (2).
    Reward = next-day return of the chosen position minus trading cost (in basis points ×100).
    Pre-training replays the pre-test history of all ten stocks for ``epochs`` passes with an
    epsilon-greedy policy. In the test window the agent acts greedily on ``ticker`` and, after each
    day's return is known, updates Q from that day's experience of all ten stocks. Because the
    market return does not depend on the agent, both actions' rewards are known after the fact and
    both are learned from.
    """
    ticker = ticker.upper()
    if not 1 <= epochs <= 500:
        raise ValueError("epochs must be 1-500")
    close = close_panel()
    if ticker not in close.columns:
        raise ValueError(f"unknown ticker {ticker}")
    tickers = [t for t in UNIVERSE if t in close.columns]
    feats = {t: _features(close[t].dropna()) for t in tickers}
    split = pd.Timestamp(test_start)
    key = (ticker, epochs, alpha, gamma, cost_bps, test_start, seed, str(close.index[-1]))
    with _lock:
        if key in _rl_cache:
            return _rl_cache[key]
    cost = cost_bps / 1e4
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, 2))
    visits = np.zeros((N_STATES, 2), dtype=int)

    # Per stock: market part of the state (ret5 x trend x rsi) and the next-day return, as arrays.
    def arrays(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        f = f.dropna()
        base = ((f["ret5"].to_numpy(int) * 2 + f["trend"].to_numpy(int)) * 3 + f["rsi"].to_numpy(int)) * 2
        return base, f["next_ret"].to_numpy(float), f.index

    pre = {t: arrays(feats[t][feats[t].index < split]) for t in tickers}
    test = {t: arrays(feats[t][feats[t].index >= split]) for t in tickers}

    def reward(next_ret: float, pos: int, action: int) -> float:
        return (action * next_ret - cost * abs(action - pos)) * 100.0  # in % so values are readable

    def learn(base: np.ndarray, ret: np.ndarray, i: int, pos: int) -> None:
        s = base[i] + pos
        for a in (0, 1):
            target = reward(ret[i], pos, a) + gamma * Q[base[i + 1] + a].max()
            Q[s, a] += alpha * (target - Q[s, a])
            visits[s, a] += 1

    # Phase 1: pre-training on the history before the test window (all stocks), epsilon-greedy.
    curve = []
    for ep in range(epochs):
        eps = max(0.05, 1.0 - ep / max(1, epochs * 0.7))
        total, n = 0.0, 0
        for t in tickers:
            base, ret, _ = pre[t]
            pos = 0
            for i in range(len(base) - 1):
                s = base[i] + pos
                a = int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())
                total += reward(ret[i], pos, a)
                n += 1
                learn(base, ret, i, pos)
                pos = a
        curve.append(_f(total / max(n, 1), 4))
    q_pretrained = Q.copy()

    # Phase 2: walk-forward test on the selected ticker, learning online from every stock.
    base_t, ret_t, idx_t = test[ticker]
    if len(base_t) < 10:
        raise ValueError("not enough test data")
    where = {t: {d: k for k, d in enumerate(test[t][2])} for t in tickers}
    positions, rets, bh, log = [], [], [], []
    pos = {t: 0 for t in tickers}
    for i in range(len(base_t) - 1):
        day = idx_t[i]
        # 1) Every stock decides with today's Q, before the next-day return is known.
        decided = {}
        for t in tickers:
            j = where[t].get(day)
            if j is not None and j + 1 < len(test[t][0]):
                s_ = test[t][0][j] + pos[t]
                decided[t] = (j, s_, int(Q[s_].argmax()))
        _, s, a = decided[ticker]
        r = reward(ret_t[i], pos[ticker], a) / 100.0
        positions.append(a)
        rets.append(r)
        bh.append(ret_t[i])
        log.append({"date": day.strftime("%Y-%m-%d"), "state": _describe(s), "action": ACTIONS[a],
                    "q": _fl(Q[s], 4), "next_return": _f(ret_t[i]), "reward": _f(r)})
        # 2) The return is now known: learn from this day's experience of every stock.
        for t, (j, _, at) in decided.items():
            learn(test[t][0], test[t][1], j, pos[t])
            pos[t] = at

    s_now = base_t[-1] + pos[ticker]
    rets_a, bh_a, pos_a = np.array(rets), np.array(bh), np.array(positions)
    active = pos_a == 1
    dates = [d.strftime("%Y-%m-%d") for d in idx_t[:len(rets) + 1]]
    table = [{"state": s, **_describe(s), "q_flat": _f(Q[s, 0], 4), "q_long": _f(Q[s, 1], 4),
              "best": ACTIONS[int(Q[s].argmax())], "visits": int(visits[s].sum()),
              "changed": bool(np.argmax(Q[s]) != np.argmax(q_pretrained[s]))} for s in range(N_STATES)]
    result = {
        "ticker": ticker, "epochs": epochs, "alpha": alpha, "gamma": gamma, "cost_bps": cost_bps,
        "test_start": dates[0], "as_of": idx_t[-1].strftime("%Y-%m-%d"),
        "today": {"state": _describe(s_now), "action": ACTIONS[int(Q[s_now].argmax())], "q": _fl(Q[s_now], 4),
                  "visits": int(visits[s_now].sum())},
        "learning_curve": curve,
        "equity": {"dates": dates, "agent": _fl(np.concatenate([[1.0], np.cumprod(1.0 + rets_a)]), 4),
                   "buy_hold": _fl(np.concatenate([[1.0], np.cumprod(1.0 + bh_a)]), 4)},
        "metrics": {"agent": {**_perf(rets_a), "time_in_market": _f(active.mean(), 4),
                              "hit_rate": _f((bh_a[active] > 0).mean(), 4) if active.any() else None,
                              "switches": int(np.abs(np.diff(np.concatenate([[0], pos_a]))).sum())},
                    "buy_hold": _perf(bh_a)},
        "q_table": table,
        "decisions": {"dates": dates[:-1], "position": [int(x) for x in positions]},
        "experience": {"transitions": int(visits.sum()), "pretrain_days": int(sum(len(pre[t][0]) for t in tickers)),
                       "states_seen": int((visits.sum(axis=1) > 0).sum()), "states": N_STATES,
                       "recent": log[-15:][::-1]},
    }
    with _lock:
        _rl_cache[key] = result
    return result
