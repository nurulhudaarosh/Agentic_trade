"""Non-memory baselines and learned (ML) policies built on memory features."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "rsi_14", "dist_ma_20", "volume_z", "spy_ret_1", "spy_ret_5"]
MEMORY_FEATURES = ["signal", "day_sentiment", "n_today", "conflict", "redundancy", "mean_age",
                   "n_intermediate", "n_deep"]
OURS_EXTRA = ["peer_signal", "episodic_edge", "mean_cred"]


def price_features(bars: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    close = bars.pivot_table(index="date", columns="ticker", values="close").sort_index()
    volume = bars.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    rets = close.pct_change()
    frames = []
    for t in tickers:
        c, r, v = close[t], rets[t], volume[t]
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        f = pd.DataFrame({
            "ret_1": r, "ret_5": c.pct_change(5), "ret_20": c.pct_change(20),
            "vol_20": r.rolling(20).std(), "rsi_14": 100 - 100 / (1 + gain / loss.replace(0, np.nan)),
            "dist_ma_20": c / c.rolling(20).mean() - 1,
            "volume_z": (v - v.rolling(20).mean()) / v.rolling(20).std(),
            "spy_ret_1": rets["SPY"], "spy_ret_5": close["SPY"].pct_change(5),
        })
        f["ticker"] = t
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)


def rule_baselines(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Classic non-learning strategies evaluated on the same (ticker, date) grid."""
    out = {
        "Buy & Hold": pd.Series(1.0, index=frame.index),
        "Momentum (1d)": np.sign(frame["ret_1"]).fillna(0.0),
        "TS-Momentum (20d)": np.sign(frame["ret_20"]).fillna(0.0),
        "Mean-Reversion (5d)": -np.sign(frame["ret_5"]).fillna(0.0),
        "MA-Crossover": np.sign(frame["dist_ma_20"]).fillna(0.0),
        "Daily News Sentiment": np.sign(frame["day_sentiment"].where(frame["day_sentiment"].abs() > 0.1, 0)).fillna(0.0),
    }
    return out


def fit_predict(frame: pd.DataFrame, features: list[str], model: str = "lgbm", seed: int = 0) -> pd.Series:
    """Fit on the train split, return P(next return > 0) for every row."""
    train = frame[frame["split"] == "train"].dropna(subset=["next_return"])
    x_train = train[features].astype(float).fillna(0.0)
    y_train = (train["next_return"] > 0).astype(int)
    x_all = frame[features].astype(float).fillna(0.0)
    if model == "lgbm":
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, verbose=-1)
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_all)[:, 1]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_all)[:, 1]
    return pd.Series(prob, index=frame.index)


def positions_from_prob(prob: pd.Series, frame: pd.DataFrame, margin: float) -> pd.Series:
    """Long when P(up) > 0.5 + margin, short when < 0.5 - margin, otherwise keep."""
    out = pd.Series(0.0, index=frame.index)
    for _, g in frame.sort_values("date").groupby("ticker", sort=False):
        pos = 0.0
        for i in g.index:
            p = prob.at[i]
            if p > 0.5 + margin:
                pos = 1.0
            elif p < 0.5 - margin:
                pos = -1.0
            out.at[i] = pos
    return out
