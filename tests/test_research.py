from pathlib import Path
import pandas as pd

from agentic_trade.research import markdown_table, paired_wilcoxon, summarize_ablations


def test_paired_wilcoxon_identical_returns():
    df = pd.DataFrame([
        {"ticker": "TSLA", "trial": 0, "date": "2026-09-01", "system": "ours", "daily_return": 0.01},
        {"ticker": "TSLA", "trial": 0, "date": "2026-09-01", "system": "fin_mem", "daily_return": 0.01},
        {"ticker": "TSLA", "trial": 0, "date": "2026-09-02", "system": "ours", "daily_return": -0.01},
        {"ticker": "TSLA", "trial": 0, "date": "2026-09-02", "system": "fin_mem", "daily_return": -0.01},
    ])
    result = paired_wilcoxon(df)
    assert result["p_value"] == 1.0
    assert not result["significant"]


def test_paired_wilcoxon_insufficient_data():
    df = pd.DataFrame([
        {"ticker": "TSLA", "trial": 0, "date": "2026-09-01", "system": "ours", "daily_return": 0.01},
    ])
    result = paired_wilcoxon(df)
    assert result["p_value"] is None
    assert not result["significant"]


def test_markdown_table_formatting(tmp_path: Path):
    summary = pd.DataFrame([
        {"ticker": "TSLA", "system": "ours", "cumulative_return": 0.15, "sharpe_ratio": 1.5, "annualized_volatility": 0.2, "max_drawdown": -0.05, "hold_rate": 0.1, "significant": True},
        {"ticker": "TSLA", "system": "fin_mem", "cumulative_return": 0.05, "sharpe_ratio": 0.8, "annualized_volatility": 0.22, "max_drawdown": -0.08, "hold_rate": 0.2, "significant": False},
    ])
    out = tmp_path / "table.md"
    markdown_table(summary, out)
    content = out.read_text()
    assert "| TSLA | ours | **0.1500*** | 1.50 | 0.2000 | -0.0500 | 0.10 |" in content


def test_summarize_ablations(tmp_path: Path):
    sweeps_dir = tmp_path / "sweeps"
    sweeps_dir.mkdir()
    # Create fake ablation CSVs
    pd.DataFrame([
        {"setting": True, "system": "ours", "cumulative_return": 0.10, "sharpe_ratio": 1.2, "annualized_volatility": 0.25, "max_drawdown": -0.10},
        {"setting": False, "system": "ours", "cumulative_return": 0.02, "sharpe_ratio": 0.3, "annualized_volatility": 0.28, "max_drawdown": -0.15},
    ]).to_csv(sweeps_dir / "ablation_use_provenance.csv", index=False)

    pd.DataFrame([
        {"setting": True, "system": "ours", "cumulative_return": 0.10, "sharpe_ratio": 1.2, "annualized_volatility": 0.25, "max_drawdown": -0.10},
        {"setting": False, "system": "ours", "cumulative_return": 0.05, "sharpe_ratio": 0.6, "annualized_volatility": 0.26, "max_drawdown": -0.12},
    ]).to_csv(sweeps_dir / "ablation_use_reconciliation.csv", index=False)

    df = summarize_ablations(sweeps_dir, tmp_path)
    assert len(df) == 3
    assert df.iloc[0]["configuration"] == "Full System (Ours)"
    assert (tmp_path / "table6_ablation.csv").exists()
    assert (tmp_path / "table6_ablation.md").exists()

