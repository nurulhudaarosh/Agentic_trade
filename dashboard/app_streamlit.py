"""
Streamlit Alternative Interface for Agentic Trade Research Dashboard.
Run with: streamlit run dashboard/app_streamlit.py
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import pandas as pd
    import streamlit as st
except ImportError:
    print("Streamlit or Pandas is not installed. Run 'pip install streamlit plotly pandas'.")
    exit(1)

st.set_page_config(page_title="Agentic Trade Dashboard", layout="wide", page_icon="📈")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"

st.title("📈 Agentic Trade Research • Cognitive Trading Memory")
st.markdown("Evaluating FinMem vs Plain RAG vs Ours (Source Provenance, Contradiction Retirement, Decision Recall)")

tab1, tab2, tab3, tab4 = st.tabs(["Performance Curves", "Memory Inspector", "Research Tables", "Paper Trading"])

with tab1:
    st.subheader("TSLA Baseline & Matrix Performance")
    metrics_path = ARTIFACTS_DIR / "offline" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        cols = st.columns(4)
        cols[0].metric("Ours Return", f"{metrics['ours']['cumulative_return']*100:.2f}%")
        cols[1].metric("FinMem Return", f"{metrics['fin_mem']['cumulative_return']*100:.2f}%")
        cols[2].metric("Sharpe (Ours)", f"{metrics['ours']['sharpe_ratio']:.2f}")
        cols[3].metric("Decisions Recalled", f"{metrics['ours'].get('decisions_recalled', 0)}")

    daily_csv = ARTIFACTS_DIR / "offline" / "ours_daily.csv"
    if daily_csv.exists():
        df = pd.read_csv(daily_csv)
        df["Ours Cumulative"] = (1.0 + df["daily_return"]).cumprod() - 1.0
        st.line_chart(df.set_index("date")["Ours Cumulative"])

with tab2:
    st.subheader("Memory & Evidence Log")
    ev_path = ARTIFACTS_DIR / "offline" / "evidence_log.csv"
    if ev_path.exists():
        ev_df = pd.read_csv(ev_path)
        st.dataframe(ev_df[["date", "memory_id", "layer", "source", "topic", "reliability", "active", "cited", "decision"]].head(100), use_container_width=True)

with tab3:
    st.subheader("Pre-Registered Retrieval Probes & Ablations")
    t7_path = ARTIFACTS_DIR / "research" / "matrix" / "table7_probes.csv"
    if t7_path.exists():
        st.write("### Table 7: Retrieval Probe Accuracy")
        st.dataframe(pd.read_csv(t7_path), use_container_width=True)

with tab4:
    st.subheader("Alpaca Paper Trading Audit")
    orders_path = ARTIFACTS_DIR / "paper_trading" / "orders.jsonl"
    if orders_path.exists():
        orders = [json.loads(line) for line in open(orders_path) if line.strip()]
        st.dataframe(pd.DataFrame(orders), use_container_width=True)

