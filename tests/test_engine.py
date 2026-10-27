from datetime import date
from pathlib import Path

import pandas as pd

from agentic_trade.engine import ExperimentConfig, ExperimentEngine


def test_engine_writes_one_row_per_trading_observation(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([
        {"date": "2026-08-31", "ticker": "TSLA", "source_type": "news", "text": "earnings profit beat", "summary": ""},
        {"date": "2026-09-01", "ticker": "TSLA", "source_type": "news", "text": "guidance raised", "summary": ""},
    ]).to_csv(data_dir / "events.csv", index=False)
    pd.DataFrame({"date": ["2026-08-31", "2026-09-01", "2026-09-02"], "close": [100, 101, 99]}).to_csv(data_dir / "prices.csv", index=False)
    config = ExperimentConfig(data_dir=data_dir, output_dir=tmp_path / "out", test_start=date(2026, 9, 1), end_date=date(2026, 9, 2))
    engine = ExperimentEngine(config)
    metrics = engine.run()
    assert len(engine.results["ours"]) == 2
    assert engine.results["ours"][0]["price"] == 101
    assert "buy_and_hold" in metrics
    assert metrics["ours"]["decisions_recalled"] == 1


def test_relationships_file_adds_cross_ticker_events(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame([
        {"date": "2026-08-31", "ticker": "TSLA", "source_type": "news", "text": "earnings profit beat", "summary": ""},
        {"date": "2026-08-31", "ticker": "MSFT", "source_type": "news", "text": "cloud growth", "summary": ""},
    ]).to_csv(data_dir / "events.csv", index=False)
    (data_dir / "relationships.csv").write_text('ticker,related_tickers\nTSLA,"MSFT"\nMSFT,"TSLA"\n')
    pd.DataFrame({
        "date": ["2026-08-31", "2026-09-01", "2026-09-02"],
        "close": [100, 101, 99],
    }).to_csv(data_dir / "prices.csv", index=False)
    config = ExperimentConfig(data_dir=data_dir, ticker="TSLA", test_start=date(2026, 9, 1), end_date=date(2026, 9, 2))
    engine = ExperimentEngine(config)
    engine.load_data()
    tickers = {event.ticker for event in engine.events}
    assert tickers == {"TSLA", "MSFT"}
    assert all("MSFT" in event.related_tickers for event in engine.events if event.ticker == "TSLA")
