from pathlib import Path
import pandas as pd

from agentic_trade.case_studies import build_case_cards


def test_build_case_cards_deduplicates_trials(tmp_path: Path):
    daily = pd.DataFrame([
        # Trial 0
        {"date": "2026-09-01", "system": "ours", "decision": "Buy", "next_return": 0.02, "trial": 0},
        {"date": "2026-09-01", "system": "fin_mem", "decision": "Hold", "next_return": 0.02, "trial": 0},
        {"date": "2026-09-02", "system": "ours", "decision": "Hold", "next_return": -0.01, "trial": 0},
        {"date": "2026-09-02", "system": "fin_mem", "decision": "Hold", "next_return": -0.01, "trial": 0},
        # Trial 1 (should be ignored by case card builder)
        {"date": "2026-09-01", "system": "ours", "decision": "Buy", "next_return": 0.02, "trial": 1},
        {"date": "2026-09-01", "system": "fin_mem", "decision": "Hold", "next_return": 0.02, "trial": 1},
    ])
    evidence = pd.DataFrame([
        # Trial 0 memories
        {"date": "2026-09-01", "system": "ours", "memory_id": "m1", "text": "battery beat", "source": "news", "layer": "shallow", "active": True, "rank": 1, "trial": 0},
        {"date": "2026-09-01", "system": "ours", "memory_id": "m2", "text": "guidance raised", "source": "news", "layer": "shallow", "active": True, "rank": 2, "trial": 0},
        # Trial 1 duplicate memory
        {"date": "2026-09-01", "system": "ours", "memory_id": "m1", "text": "battery beat", "source": "news", "layer": "shallow", "active": True, "rank": 1, "trial": 1},
    ])

    build_case_cards(daily, evidence, tmp_path)
    output_card = tmp_path / "case_study_cards.md"
    assert output_card.exists()
    content = output_card.read_text()
    assert "# Case Study Cards" in content
    assert "Ours-FinMem disagreement" in content
    # Ensure memory m1 is not duplicated 3 times for the same date
    lines = [line for line in content.splitlines() if "m1:" in line]
    assert len(lines) == 1

