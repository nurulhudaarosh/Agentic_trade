"""Extension experiments on the study's rolling protocol (deep learning, RL, analogs).

Every model here is evaluated exactly like the study's LightGBM baselines: the same
(ticker, date) grid, the same test window (2025-12-01 .. 2026-10-26), monthly
walk-forward refits that only use labels resolved before the month starts, the same
probability margin (0.02) and the same 5 bps cost on position changes.

Models
------
* ``GRU (price)``                recurrent network over the last 20 days of price features.
* ``Transformer (price)``        Transformer encoder over the same 20-day window.
* ``Transformer (price+memory)`` the same encoder with our memory features added.
* ``RL Q-learning``              tabular Q-learning, long/flat (``agentic_trade.algorithms``).
* ``Pattern analogs (kNN)``      nearest past 20-day shapes, 10-day outcome (``agentic_trade.analytics``).
* Exploratory, defined after the main results were known (reported as such):
  ``LGBM + memory risk overlay`` (LightGBM position, flat when our memory abstains) and
  ``Model vote`` (majority of LGBM, Transformer and our memory).
"""
from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import apply_costs
from .evaluation import block_bootstrap_sharpe_diff, information_coefficient, metrics, paired_tests
from .pipeline import COST_BPS, ROLLING_MARGIN, ROLLING_START, _month_starts, portfolio
from .policies import MEMORY_FEATURES, OURS_EXTRA, PRICE_FEATURES, positions_from_prob, price_features

SEQ_LEN = 20
SEEDS = (0, 1, 2)
EXT_MODELS = ["GRU (price)", "Transformer (price)", "Transformer (price+memory)", "RL Q-learning", "Pattern analogs (kNN)"]
EXPLORATORY = ["LGBM + memory risk overlay", "Model vote (LGBM, Transformer, Ours)"]


# --------------------------------------------------------------------------- data
def build_base(data_dir: Path, study_dir: Path) -> pd.DataFrame:
    """The study's (ticker, date) frame: our memory signals joined with price features."""
    signals = pd.read_parquet(study_dir / "signals.parquet")
    signals["date"] = pd.to_datetime(signals["date"]).dt.date
    signals = signals[signals["split"] != "warmup"]
    tickers = sorted(signals["ticker"].unique())
    bars = pd.read_parquet(data_dir / "prices.parquet")
    feats = price_features(bars, tickers)
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    base = signals[signals["system"] == "Ours"].merge(feats, on=["ticker", "date"], how="left")
    return base.sort_values(["ticker", "date"]).reset_index(drop=True)


def _sequences(base: pd.DataFrame, features: list[str]) -> np.ndarray:
    """(rows, SEQ_LEN, features): for each row, that ticker's last SEQ_LEN days ending on its date."""
    x = base[features].astype(float).to_numpy()
    out = np.zeros((len(base), SEQ_LEN, len(features)), dtype=np.float32)
    for _, idx in base.groupby("ticker").indices.items():
        idx = np.sort(idx)
        xt = x[idx]
        for k, row in enumerate(idx):
            lo = max(0, k - SEQ_LEN + 1)
            window = xt[lo:k + 1]
            out[row, SEQ_LEN - len(window):] = window
    return out


# --------------------------------------------------------------------------- deep models
def _make_model(kind: str, n_features: int, n_tickers: int):
    import torch
    from torch import nn

    class GRUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tick = nn.Embedding(n_tickers, 4)
            self.gru = nn.GRU(n_features + 4, 32, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x, t):
            e = self.tick(t).unsqueeze(1).expand(-1, x.shape[1], -1)
            h, _ = self.gru(torch.cat([x, e], dim=-1))
            return self.head(h[:, -1]).squeeze(-1)

    class TransformerNet(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.inp = nn.Linear(n_features, d)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
            self.tick = nn.Embedding(n_tickers, d)
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 2, 64, dropout=0.1, batch_first=True) for _ in range(2)])
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.last_attention = None

        def forward(self, x, t):
            h = self.inp(x) + self.pos + self.tick(t).unsqueeze(1)
            for layer in self.layers:
                h = layer(h)
            # Attention of the final day over the 20-day window, from the last layer (for figures).
            with torch.no_grad():
                last = self.layers[-1].self_attn
                _, w = last(h, h, h, need_weights=True, average_attn_weights=True)
                self.last_attention = w[:, -1, :].mean(0).cpu().numpy()
            return self.head(h[:, -1]).squeeze(-1)

    return GRUNet() if kind == "gru" else TransformerNet()


def _fit_predict_deep(kind: str, x: np.ndarray, tick: np.ndarray, y: np.ndarray, train: np.ndarray,
                      predict: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    # Standardise with training statistics only, then clip outliers.
    mu = x[train].reshape(-1, x.shape[-1]).mean(0)
    sd = x[train].reshape(-1, x.shape[-1]).std(0) + 1e-6
    xs = np.clip(np.nan_to_num((x - mu) / sd), -5, 5).astype(np.float32)
    tr = np.flatnonzero(train)
    # Time-ordered early-stopping split: the last 15% of training rows by date.
    n_val = max(20, int(len(tr) * 0.15))
    fit_idx, val_idx = tr[:-n_val], tr[-n_val:]
    model = _make_model(kind, x.shape[-1], int(tick.max()) + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    X, T, Y = torch.tensor(xs), torch.tensor(tick, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
    best, best_state, wait, log = math.inf, None, 0, {"train": [], "val": []}
    for _ in range(40):
        model.train()
        perm = np.random.permutation(fit_idx)
        total = 0.0
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            opt.zero_grad()
            loss = loss_fn(model(X[b], T[b]), Y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(X[val_idx], T[val_idx]), Y[val_idx]))
        log["train"].append(total / len(perm))
        log["val"].append(val)
        if val < best - 1e-4:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X[predict], T[predict])).numpy()
    if hasattr(model, "last_attention") and model.last_attention is not None:
        log["attention"] = [float(v) for v in model.last_attention]
    log["best_epoch"] = int(np.argmin(log["val"])) + 1
    return prob, log


def walk_forward_deep(base: pd.DataFrame, features: list[str], kind: str) -> tuple[pd.Series, list[dict]]:
    """Monthly refits exactly like the study's LightGBM: train on labels resolved before the month."""
    x = _sequences(base, features)
    tickers = sorted(base["ticker"].unique())
    tick = base["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    y = (base["next_return"] > 0).astype(float).to_numpy()
    has_label = base["next_return"].notna().to_numpy()
    days = sorted(base["date"].unique())
    starts = _month_starts(days)
    prob = pd.Series(0.5, index=base.index)
    logs = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else date.max
        last = max(d for d in days if d < start)
        train = ((base["date"] < start) & (base["date"] != last)).to_numpy() & has_label
        month = ((base["date"] >= start) & (base["date"] < end)).to_numpy()
        preds, month_logs = [], []
        for seed in SEEDS:
            p, log = _fit_predict_deep(kind, x, tick, y, train, np.flatnonzero(month), seed)
            preds.append(p)
            month_logs.append(log)
        prob[month] = np.mean(preds, axis=0)
        logs.append({"month": str(start), "n_train": int(train.sum()), "seeds": month_logs})
    return prob, logs


# --------------------------------------------------------------------------- assembling results
def _rows(base: pd.DataFrame, position: pd.Series, signal: pd.Series, label: str) -> pd.DataFrame:
    f = base[["ticker", "date", "next_return", "next_spy"]].copy()
    f["position"] = position.to_numpy(dtype=float)
    f["signal"] = signal.to_numpy(dtype=float)
    f["abstain"] = False
    f["system"] = label
    f["net"] = apply_costs(f, cost_bps=COST_BPS)
    f["gross"] = f["position"] * f["next_return"]
    f["split"] = np.where(f["date"] >= ROLLING_START, "test", "pre")
    return f


def _from_map(base: pd.DataFrame, mapping: dict[tuple[str, date], float], default: float = 0.0) -> pd.Series:
    keys = list(zip(base["ticker"], base["date"]))
    return pd.Series([mapping.get(k, default) for k in keys], index=base.index, dtype=float)


def rl_positions(base: pd.DataFrame) -> tuple[pd.Series, dict]:
    from ..algorithms import q_learning_agent

    mapping, logs = {}, {}
    for t in sorted(base["ticker"].unique()):
        r = q_learning_agent(t, epochs=30, test_start=str(ROLLING_START))
        for d, p in zip(r["decisions"]["dates"], r["decisions"]["position"]):
            mapping[(t, pd.Timestamp(d).date())] = float(p)
        logs[t] = {"learning_curve": r["learning_curve"], "q_table": r["q_table"], "metrics": r["metrics"],
                   "experience": {k: v for k, v in r["experience"].items() if k != "recent"}}
    return _from_map(base, mapping), logs


def analog_positions(base: pd.DataFrame) -> tuple[pd.Series, dict]:
    from ..analytics import analog_backtest

    mapping, logs = {}, {}
    for t in sorted(base["ticker"].unique()):
        # The app's default settings, fixed before any study result was looked at.
        r = analog_backtest(t, window=20, horizon=10, k=8, scope="ticker", threshold=0.01, long_only=False, cost_bps=COST_BPS)
        for d, p in zip(r["dates"][:-1], r["positions"][:-1]):
            mapping[(t, pd.Timestamp(d).date())] = float(p)
        logs[t] = r["metrics"]
    return _from_map(base, mapping), logs


def run(data_dir: Path, study_dir: Path) -> dict:
    t0 = time.time()
    out_dir = study_dir / "rolling"
    base = build_base(data_dir, study_dir)
    study = pd.read_parquet(out_dir / "daily.parquet")
    study["date"] = pd.to_datetime(study["date"]).dt.date
    rows, logs = [], {}

    deep = {
        "GRU (price)": (PRICE_FEATURES, "gru"),
        "Transformer (price)": (PRICE_FEATURES, "transformer"),
        "Transformer (price+memory)": (PRICE_FEATURES + MEMORY_FEATURES + OURS_EXTRA, "transformer"),
    }
    probs = {}
    for label, (feats, kind) in deep.items():
        print(f"[extensions] {label} …", flush=True)
        prob, log = walk_forward_deep(base, feats, kind)
        probs[label] = prob
        logs[label] = log
        rows.append(_rows(base, positions_from_prob(prob, base, ROLLING_MARGIN), prob - 0.5, label))

    print("[extensions] RL Q-learning …", flush=True)
    pos, logs["RL Q-learning"] = rl_positions(base)
    rows.append(_rows(base, pos, pos, "RL Q-learning"))
    print("[extensions] pattern analogs …", flush=True)
    pos, logs["Pattern analogs (kNN)"] = analog_positions(base)
    rows.append(_rows(base, pos, pos, "Pattern analogs (kNN)"))

    # Exploratory combinations (defined after the main results; reported separately).
    key = ["ticker", "date"]
    st = study.set_index(["system", "ticker", "date"])
    lgbm = st.loc["LGBM (price)"]["position"]
    ours = st.loc["Ours"]
    idx = pd.MultiIndex.from_frame(base[key])
    lg = lgbm.reindex(idx).fillna(0.0).to_numpy()
    abstain = ours["abstain"].reindex(idx).fillna(False).astype(bool).to_numpy()
    overlay = pd.Series(np.where(abstain, 0.0, lg), index=base.index)
    rows.append(_rows(base, overlay, overlay, EXPLORATORY[0]))
    tr = positions_from_prob(probs["Transformer (price)"], base, ROLLING_MARGIN).to_numpy()
    om = ours["position"].reindex(idx).fillna(0.0).to_numpy()
    vote = pd.Series(np.sign(lg + tr + om), index=base.index)
    rows.append(_rows(base, vote, vote, EXPLORATORY[1]))

    ext = pd.concat(rows, ignore_index=True)
    ext.to_parquet(out_dir / "extensions_daily.parquet", index=False)
    summary = summarize(study, ext)
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out_dir / "extensions_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "extensions_logs.json").write_text(json.dumps(logs, default=float))
    return summary


def summarize(study: pd.DataFrame, ext: pd.DataFrame) -> dict:
    allrows = pd.concat([study, ext], ignore_index=True)
    test = allrows[allrows["split"] == "test"]
    systems = ["Buy & Hold", "LGBM (price)", "LogReg (price)", "Ours", "FinMem"] + EXT_MODELS + EXPLORATORY
    port = {s: portfolio(test[test["system"] == s], "net") for s in systems}
    spy = test[test["system"] == "Buy & Hold"].groupby("date")["next_spy"].mean().sort_index()
    out = {"portfolio": {}, "per_ticker": {}, "tests": {}, "ic": {}}
    for s in systems:
        g = test[test["system"] == s]
        pos = g.groupby("date")["position"].mean()
        out["portfolio"][s] = metrics(port[s], pos, spy)
        out["per_ticker"][s] = {t: metrics(gt.sort_values("date")["net"].reset_index(drop=True))["sharpe"]
                                for t, gt in g.groupby("ticker")}
        if s in EXT_MODELS + EXPLORATORY and g["signal"].nunique() > 2:
            out["ic"][s] = information_coefficient(g)
    for s in EXT_MODELS + EXPLORATORY:
        for ref in ("Buy & Hold", "LGBM (price)", "Ours"):
            bs = block_bootstrap_sharpe_diff(port[s], port[ref])
            bs.pop("samples")
            out["tests"][f"{s} vs {ref}"] = {**paired_tests(port[s], port[ref]), **bs}
    return out
