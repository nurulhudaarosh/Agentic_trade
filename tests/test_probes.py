from datetime import date
from pathlib import Path

from agentic_trade.memory import MemoryEvent, NoMemory, OursMemory
from agentic_trade.probes import ProbeItem, evaluate_probes, get_default_probes, run_probe_benchmark


def test_default_probes_not_empty():
    probes = get_default_probes()
    assert len(probes) >= 15
    for probe in probes:
        assert probe.ticker
        assert probe.expected_entities
        assert probe.as_of


def test_evaluate_probes_metrics():
    events = [
        MemoryEvent("e1", date(2026, 6, 1), "TSLA", "news", "Tesla battery supplier Panasonic raises cell production with quarterly revenue beat"),
        MemoryEvent("e2", date(2026, 6, 2), "TSLA", "news", "Elon Musk speaks at autonomy summit"),
    ]
    ours = OursMemory()
    for e in events:
        ours.add(e)

    probes = [
        ProbeItem("P1", "TSLA", "battery partner", {"earnings"}, ("panasonic", "battery"), date(2026, 6, 5)),
        ProbeItem("P2", "TSLA", "unrelated topic", {"legal_risk"}, ("nonexistent_entity",), date(2026, 6, 5)),
    ]

    results = evaluate_probes({"no_memory": NoMemory(), "ours": ours}, probes, top_k=2)

    assert results["no_memory"]["hit_at_1"] == 0.0
    assert results["no_memory"]["mrr"] == 0.0

    assert results["ours"]["hit_at_1"] == 0.5  # 1 out of 2 probes matched at rank 1
    assert results["ours"]["mrr"] == 0.5


def test_probes_do_not_match_unrelated_tickers():
    events = [
        MemoryEvent("msft", date(2026, 6, 1), "MSFT", "news", "Microsoft cloud revenue beat"),
    ]
    ours = OursMemory()
    for event in events:
        ours.add(event)
    probe = ProbeItem("TSLA-X", "TSLA", "cloud revenue", {"earnings"}, ("cloud", "revenue"), date(2026, 6, 5))
    results = evaluate_probes({"ours": ours}, [probe], top_k=5)
    assert results["ours"]["hit_at_1"] == 0.0
    assert results["ours"]["mrr"] == 0.0


def test_run_probe_benchmark_file_outputs(tmp_path: Path):
    events = [
        MemoryEvent("e1", date(2026, 6, 1), "TSLA", "news", "Tesla battery supplier Panasonic raises cell production"),
    ]
    frame, md = run_probe_benchmark(events, tmp_path, probes=get_default_probes()[:3], top_k=3)
    assert (tmp_path / "table7_probes.csv").exists()
    assert (tmp_path / "table7_probes.md").exists()
    assert "# Table 7: Memory Probe Retrieval Accuracy" in md
    assert len(frame) == 4
