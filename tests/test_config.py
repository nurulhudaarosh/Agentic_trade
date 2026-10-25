from datetime import date
from pathlib import Path

from agentic_trade.engine import ExperimentConfig


def test_config_from_yaml(tmp_path: Path):
    yaml_content = """
ticker: "NVDA"
risk_mode: "aggressive"
top_k: 10
start_date: "2026-05-01"
use_provenance: false
"""
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml_content)

    config = ExperimentConfig.from_yaml(yaml_path)
    assert config.ticker == "NVDA"
    assert config.risk_mode == "aggressive"
    assert config.top_k == 10
    assert config.start_date == date(2026, 5, 1)
    assert not config.use_provenance


def test_config_to_dict():
    config = ExperimentConfig(ticker="AAPL", risk_mode="conservative")
    d = config.to_dict()
    assert d["ticker"] == "AAPL"
    assert d["risk_mode"] == "conservative"
    assert d["start_date"] == "2026-06-01"

