from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_trade.study import extensions as X

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "artifacts" / "study" / "rolling"


def _toy(n: int = 30) -> pd.DataFrame:
    days = pd.bdate_range("2025-11-03", periods=n).date
    rows = [{"ticker": t, "date": d, "f": float(i) + (100 if t == "B" else 0), "next_return": 0.01 * (-1) ** i,
             "next_spy": 0.0} for t in ("A", "B") for i, d in enumerate(days)]
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_sequences_are_right_aligned_and_never_use_future_days():
    base = _toy()
    x = X._sequences(base, ["f"])
    a = base.index[base["ticker"] == "A"]
    # Row k of ticker A ends with its own value and contains only earlier values of A.
    k = a[25]
    assert x[k, -1, 0] == base.at[k, "f"]
    assert list(x[k, :, 0]) == list(base.loc[a[6:26], "f"])
    # The first row is zero-padded, and ticker B's window never contains ticker A's values.
    assert (x[a[0], :-1, 0] == 0).all()
    b = base.index[base["ticker"] == "B"]
    assert (x[b[10], -11:, 0] >= 100).all()


def test_rows_apply_the_study_cost_model():
    base = _toy(6)
    pos = pd.Series([1, 1, -1, -1, 0, 0] * 2, index=base.index, dtype=float)
    out = X._rows(base, pos, pos, "toy")
    a = out[out["ticker"] == "A"].sort_values("date")
    turnover = np.abs(np.diff(np.concatenate([[0.0], a["position"].to_numpy()])))
    expected = a["position"].to_numpy() * a["next_return"].to_numpy() - turnover * 5 / 1e4
    assert np.allclose(a["net"].to_numpy(), expected)
    assert set(out["split"]) == {"pre"}  # all toy dates are before the test window


@pytest.mark.skipif(not (RD / "extensions_daily.parquet").exists(), reason="extensions not run")
def test_saved_extension_results_cover_the_study_test_grid():
    ext = pd.read_parquet(RD / "extensions_daily.parquet")
    study = pd.read_parquet(RD / "daily.parquet", columns=["system", "ticker", "date", "split"])
    grid = study[(study["system"] == "Ours") & (study["split"] == "test")][["ticker", "date"]]
    for name in X.EXT_MODELS:
        g = ext[(ext["system"] == name) & (ext["split"] == "test")]
        assert len(g) == len(grid)
        assert set(map(tuple, g[["ticker", "date"]].astype(str).to_numpy())) == set(map(tuple, grid.astype(str).to_numpy()))
        assert g["position"].isin([-1.0, 0.0, 1.0]).all()
    assert pd.to_datetime(ext.loc[ext["split"] == "test", "date"]).min().date() == date(2025, 12, 1)
