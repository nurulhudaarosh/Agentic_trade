from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def build_case_cards(daily: pd.DataFrame, evidence: pd.DataFrame, output: Path, count: int = 5) -> None:
    """Select cases by pre-registered rules, never by outcome."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if "trial" in daily.columns:
        daily = daily[daily["trial"].eq(0)]
    if "trial" in evidence.columns:
        evidence = evidence[evidence["trial"].eq(0)]
    ours = daily[daily["system"].eq("ours")].copy()
    fin = daily[daily["system"].eq("fin_mem")][["date", "decision"]].rename(columns={"decision": "fin_decision"})
    merged = ours.merge(fin, on="date", how="left").sort_values("date")
    candidates: list[tuple[str, pd.Series]] = []
    disagreement = merged[merged["decision"] != merged["fin_decision"]]
    if not disagreement.empty:
        candidates.append(("Ours-FinMem disagreement", disagreement.iloc[0]))
    if not evidence.empty:
        retirement = evidence[(evidence["system"].eq("ours")) & (~evidence["active"].astype(bool))]
        if not retirement.empty:
            ret_date = retirement.iloc[0]["date"]
            ret_match = merged[merged["date"].eq(ret_date)]
            row = ret_match.iloc[0] if not ret_match.empty else merged.iloc[min(len(merged) - 1, 1)]
            candidates.append(("Memory update candidate", row))
    if len(merged) > 2:
        candidates.append(("Pre-registered midpoint", merged.iloc[len(merged) // 2]))
    if len(merged) > 1:
        candidates.append(("Pre-registered first test day", merged.iloc[0]))
    rng = random.Random(17)
    if len(merged) > 1:
        candidates.append(("Deterministic random case", merged.iloc[rng.randrange(len(merged))]))
    used = set()
    cards = []
    for reason, row in candidates:
        if row["date"] in used or len(cards) >= count:
            continue
        used.add(row["date"])
        ev = evidence[(evidence["system"].eq("ours")) & evidence["date"].eq(row["date"])].sort_values("rank").head(3)
        evidence_lines = [f"- {item.memory_id}: {item.text} ({item.source}, {item.layer}, active={item.active}, rank={item.rank})" for item in ev.itertuples()]
        cards.append("\n".join([
            f"## {row['date']} | {reason}",
            f"- FinMem decision: {row.get('fin_decision', 'unknown')}",
            f"- Ours decision: {row['decision']}",
            f"- Realized next-period return: {float(row['next_return']):.4f}",
            "- Evidence:",
            *(evidence_lines or ["- No retrieved evidence"]),
        ]))
    (output / "case_study_cards.md").write_text("# Case Study Cards\n\n" + "\n\n".join(cards) + "\n")
