from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .memory import FinMem, MemoryEvent, MemorySystem, NoMemory, OursMemory, PlainRAG


@dataclass(frozen=True)
class ProbeItem:
    probe_id: str
    ticker: str
    query_text: str
    topics: set[str]
    expected_entities: tuple[str, ...]
    as_of: date
    category: str = "supply_chain"


def get_default_probes() -> list[ProbeItem]:
    """Pre-registered probe suite testing relationship, topic, and entity recall across tickers."""
    return [
        # TSLA Supply Chain & Partners
        ProbeItem("TSLA-01", "TSLA", "Who supplies lithium batteries and cells to Tesla?", {"earnings", "guidance"}, ("panasonic", "catl", "battery", "supplier", "cell"), date(2026, 7, 1), "supply_chain"),
        ProbeItem("TSLA-02", "TSLA", "Tesla vehicle deliveries and quarterly production forecast", {"earnings", "guidance"}, ("deliveries", "production", "quarterly", "beat", "record"), date(2026, 7, 5), "earnings_catalyst"),
        ProbeItem("TSLA-03", "TSLA", "Autonomous driving and regulatory compliance hurdles", {"legal_risk"}, ("autopilot", "fsd", "investigation", "nhtsa", "regulatory", "compliance"), date(2026, 7, 5), "legal_risk"),
        ProbeItem("TSLA-04", "TSLA", "Tesla relationship with SpaceX and cross-company deals", {"mergers_acquisitions", "guidance"}, ("spacex", "elon", "arms", "satellite", "ipo"), date(2026, 7, 9), "supply_chain"),
        ProbeItem("TSLA-05", "TSLA", "Tesla automotive margin beat and earnings profitability", {"earnings"}, ("profit", "margin", "eps", "revenue", "beat"), date(2026, 7, 1), "earnings_catalyst"),

        # MSFT Cloud & OpenAI Partnerships
        ProbeItem("MSFT-01", "MSFT", "Microsoft cloud infrastructure partner for OpenAI", {"guidance"}, ("openai", "azure", "partnership", "cloud", "ai"), date(2026, 7, 1), "supply_chain"),
        ProbeItem("MSFT-02", "MSFT", "Azure cloud quarterly growth rate and guidance", {"guidance", "earnings"}, ("azure", "growth", "cloud", "guidance", "raised"), date(2026, 7, 5), "guidance"),
        ProbeItem("MSFT-03", "MSFT", "Gaming acquisition integration and regulatory remedies", {"mergers_acquisitions", "legal_risk"}, ("activision", "ftc", "cma", "acquisition", "gaming"), date(2026, 7, 5), "mergers_acquisitions"),
        ProbeItem("MSFT-04", "MSFT", "Enterprise software licensing and margin expansion", {"earnings"}, ("commercial", "office", "margin", "profit", "eps"), date(2026, 7, 5), "earnings_catalyst"),

        # AMZN E-commerce & Cloud
        ProbeItem("AMZN-01", "AMZN", "AWS cloud revenue and infrastructure capital spend", {"earnings", "guidance"}, ("aws", "cloud", "capex", "revenue", "profit"), date(2026, 7, 1), "earnings_catalyst"),
        ProbeItem("AMZN-02", "AMZN", "Delivery fleet logistics and warehouse automation suppliers", {"guidance"}, ("logistics", "rivian", "delivery", "automation", "fleet"), date(2026, 7, 5), "supply_chain"),
        ProbeItem("AMZN-03", "AMZN", "FTC consumer marketplace antitrust litigation", {"legal_risk"}, ("ftc", "lawsuit", "marketplace", "antitrust", "legal"), date(2026, 7, 5), "legal_risk"),

        # COIN Crypto & Regulatory
        ProbeItem("COIN-01", "COIN", "SEC lawsuit and crypto staking regulatory proceedings", {"legal_risk"}, ("sec", "lawsuit", "staking", "crypto", "unregistered"), date(2026, 7, 1), "legal_risk"),
        ProbeItem("COIN-02", "COIN", "Institutional custody partnership with Bitcoin ETF issuers", {"mergers_acquisitions", "guidance"}, ("blackrock", "etf", "custody", "partnership", "bitcoin"), date(2026, 7, 5), "supply_chain"),
        ProbeItem("COIN-03", "COIN", "Transaction trading volume and subscription services revenue", {"earnings"}, ("volume", "revenue", "subscription", "eps", "trading"), date(2026, 7, 5), "earnings_catalyst"),

        # NFLX Streaming & Licensing
        ProbeItem("NFLX-01", "NFLX", "Global subscriber net additions beat and ad-tier growth", {"earnings"}, ("subscribers", "ad-tier", "growth", "beat", "quarterly"), date(2026, 7, 1), "earnings_catalyst"),
        ProbeItem("NFLX-02", "NFLX", "Content licensing partnerships and studio production deals", {"guidance"}, ("licensing", "studio", "content", "deal", "production"), date(2026, 7, 5), "supply_chain"),
    ]


def evaluate_probes(
    systems: dict[str, MemorySystem],
    probes: list[ProbeItem],
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    """Evaluate retrieval accuracy, precision, and MRR for each memory system."""
    results: dict[str, dict[str, float]] = {}

    for name, system in systems.items():
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        reciprocal_ranks: list[float] = []
        precision_scores: list[float] = []

        for probe in probes:
            # A probe may only be answered from its own ticker (or an explicitly
            # related ticker); without this filter keyword matches from unrelated
            # companies contaminate the benchmark.
            retrieved = system.retrieve(as_of=probe.as_of, topics=probe.topics, top_k=top_k, ticker=probe.ticker)
            ranks_of_hits = []
            for rank, event in enumerate(retrieved, 1):
                text_combined = f"{event.text} {event.summary}".lower()
                if any(entity in text_combined for entity in probe.expected_entities):
                    ranks_of_hits.append(rank)

            if 1 in ranks_of_hits:
                hit_1 += 1
            if any(r <= 3 for r in ranks_of_hits):
                hit_3 += 1
            if any(r <= 5 for r in ranks_of_hits):
                hit_5 += 1

            rr = 1.0 / ranks_of_hits[0] if ranks_of_hits else 0.0
            reciprocal_ranks.append(rr)
            precision_scores.append(len(ranks_of_hits) / top_k if top_k > 0 else 0.0)

        n = len(probes) or 1
        results[name] = {
            "hit_at_1": round(hit_1 / n, 4),
            "hit_at_3": round(hit_3 / n, 4),
            "hit_at_5": round(hit_5 / n, 4),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "precision_at_k": round(sum(precision_scores) / n, 4),
            "probe_count": float(len(probes)),
        }

    return results


def run_probe_benchmark(
    events: list[MemoryEvent],
    output_dir: Path,
    probes: list[ProbeItem] | None = None,
    top_k: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Run probe benchmark across memory systems, write Table 7 CSV and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_items = probes or get_default_probes()

    systems: dict[str, MemorySystem] = {
        "no_memory": NoMemory(),
        "plain_rag": PlainRAG(),
        "fin_mem": FinMem(),
        "ours": OursMemory(),
    }

    # Ingest events into all memory systems
    for event in events:
        for system in systems.values():
            copy = event if system.name != "ours" else MemoryEvent(
                event.event_id, event.event_date, event.ticker, event.source_type,
                event.text, event.summary, event.sentiment, set(event.topic_tags), set(event.related_tickers)
            )
            system.add(copy)

    metrics = evaluate_probes(systems, probe_items, top_k=top_k)

    rows = []
    for sys_name, res in metrics.items():
        rows.append({"system": sys_name, **res})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "table7_probes.csv", index=False)

    md_lines = [
        "# Table 7: Memory Probe Retrieval Accuracy",
        "",
        f"Benchmark of {len(probe_items)} pre-registered queries evaluating entity, topic, and relationship recall.",
        "",
        "| System | Hit@1 | Hit@3 | Hit@5 | MRR | Precision@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in frame.iterrows():
        md_lines.append(
            f"| {r['system']} | {r['hit_at_1']:.4f} | {r['hit_at_3']:.4f} | "
            f"{r['hit_at_5']:.4f} | {r['mrr']:.4f} | {r['precision_at_k']:.4f} |"
        )
    md_content = "\n".join(md_lines) + "\n"
    (output_dir / "table7_probes.md").write_text(md_content)

    return frame, md_content

