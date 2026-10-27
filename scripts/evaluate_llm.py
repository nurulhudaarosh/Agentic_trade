#!/usr/bin/env python3
"""
Evaluate LLM-backed Trading Agent on historical point-in-time data.
Generates LLM trade decisions (Buy/Sell/Hold) per day, calculates exact Profit & Loss (USD and %),
win rates, Sharpe ratios, drawdowns, and benchmarks against FinMem, Plain RAG, and Buy & Hold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import numpy as np
import pandas as pd

from agentic_trade.engine import ExperimentConfig, ExperimentEngine
from agentic_trade.llm import OpenRouterClient


def calculate_trade_stats(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
    """Calculate detailed trading performance stats including dollar PnL and win rate."""
    if df.empty or "daily_return" not in df.columns:
        return {}

    daily_returns = df["daily_return"].fillna(0.0).values
    positions = df["position"].fillna(0.0).values
    decisions = df["decision"].fillna("Hold").values

    # Portfolio equity curve
    portfolio_wealth = (1.0 + daily_returns).cumprod()
    final_wealth = float(portfolio_wealth[-1]) if len(portfolio_wealth) > 0 else 1.0
    cum_return = final_wealth - 1.0

    final_portfolio_value = initial_capital * final_wealth
    net_pnl_usd = final_portfolio_value - initial_capital

    # Risk metrics
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    ann_vol = float(std_ret * np.sqrt(252))
    sharpe = float((mean_ret / std_ret * np.sqrt(252))) if std_ret > 1e-9 else 0.0

    # Max Drawdown
    peaks = np.maximum.accumulate(portfolio_wealth)
    drawdowns = (portfolio_wealth - peaks) / peaks
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Trade counts and win rate (days with non-zero position)
    active_days = df[df["position"] != 0.0]
    total_active = len(active_days)
    winning_days = len(df[df["daily_return"] > 0.0])
    losing_days = len(df[df["daily_return"] < 0.0])
    win_rate = (winning_days / total_active * 100.0) if total_active > 0 else 0.0

    buy_count = int(np.sum(decisions == "Buy"))
    sell_count = int(np.sum(decisions == "Sell"))
    hold_count = int(np.sum(decisions == "Hold"))

    return {
        "initial_capital": initial_capital,
        "final_portfolio_value": round(final_portfolio_value, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "cumulative_return_pct": round(cum_return * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "annualized_volatility_pct": round(ann_vol * 100.0, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "total_days": len(df),
        "active_trade_days": total_active,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "win_rate_pct": round(win_rate, 1),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }


def print_pnl_banner(summary_table: List[Dict[str, Any]], ticker: str, model_name: str, risk_mode: str, initial_capital: float) -> None:
    """Print executive terminal summary table."""
    print("\n" + "=" * 90)
    print(f" 📊 AGENTIC TRADE LLM BACKTEST & PROFIT/LOSS REPORT")
    print(f" 🎯 Ticker: {ticker} | Model: {model_name} | Risk Profile: {risk_mode.upper()}")
    print(f" 💰 Initial Capital: ${initial_capital:,.2f}")
    print("=" * 90)
    print(f"{'System Architecture':<28} | {'Return':<9} | {'Net PnL ($)':<12} | {'Final Value':<12} | {'Sharpe':<7} | {'Max DD':<8} | {'Win Rate':<8}")
    print("-" * 90)

    for row in summary_table:
        name = row["system"]
        ret = f"{row['cumulative_return_pct']:+.2f}%"
        pnl = f"{row['net_pnl_usd']:+,.2f}"
        final = f"${row['final_portfolio_value']:,.2f}"
        sharpe = f"{row['sharpe_ratio']:.2f}"
        dd = f"{row['max_drawdown_pct']:.2f}%"
        wr = f"{row['win_rate_pct']:.1f}%"
        print(f"{name:<28} | {ret:<9} | {pnl:<12} | {final:<12} | {sharpe:<7} | {dd:<8} | {wr:<8}")

    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM Trading Decisions & Profit/Loss on historical data")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker (e.g. TSLA, NFLX, AMZN, MSFT, COIN)")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), help="LLM model (Groq or OpenRouter)")
    parser.add_argument("--provider", choices=["auto", "groq", "openrouter"], default="auto", help="Primary LLM provider (auto fails over between Groq and OpenRouter)")
    parser.add_argument("--risk-mode", choices=["aggressive", "standard", "conservative"], default="aggressive", help="Risk profile")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/llm_eval"), help="Directory to save evaluation artifacts")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial investment capital in USD (default $10,000)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved memories per layer")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing LLM client with model '{args.model}' and risk mode '{args.risk_mode}'...")
    llm_client = OpenRouterClient(
        model=args.model,
        temperature=args.temperature,
        cache_dir=output_dir / "llm_cache",
        primary_provider=args.provider,
    )
    print(f"Primary backend: {llm_client.primary_provider.upper()} (Groq model: {llm_client.groq_model} | OpenRouter free models: {', '.join(llm_client.openrouter_free_models[:3])}...)")

    config = ExperimentConfig(
        data_dir=Path("data"),
        output_dir=output_dir,
        ticker=args.ticker,
        start_date=date(2026, 6, 1),
        train_end=date(2026, 6, 15),
        test_start=date(2026, 6, 16),
        end_date=date(2026, 7, 9),
        top_k=args.top_k,
        temperature=args.temperature,
        backbone=args.model,
        risk_mode=args.risk_mode,
        llm_cache_dir=output_dir / "llm_cache",
    )

    print(f"\nRunning point-in-time backtest for {args.ticker} (Test Window: {config.test_start} to {config.end_date})...")
    engine = ExperimentEngine(config)
    engine.llm = llm_client

    # Run the backtest
    metrics = engine.run()
    engine.write_artifacts(metrics)

    # Compute detailed trade statistics for each system
    summary_rows: List[Dict[str, Any]] = []

    names_map = {
        "ours": "Ours (Cognitive Architecture)",
        "fin_mem": "FinMem (3-Layer Baseline)",
        "plain_rag": "Plain RAG (Flat Vector)",
        "no_memory": "No Memory (Momentum)",
        "buy_and_hold": "Buy & Hold Benchmark",
    }

    # Process all evaluated systems
    for sys_key in ["ours", "fin_mem", "plain_rag", "no_memory"]:
        daily_csv = output_dir / f"{sys_key}_daily.csv"
        if daily_csv.exists():
            df = pd.read_csv(daily_csv)
            stats = calculate_trade_stats(df, initial_capital=args.capital)
            stats["system"] = names_map.get(sys_key, sys_key)
            stats["system_key"] = sys_key
            summary_rows.append(stats)

    # Process Buy & Hold
    ref_csv = output_dir / "no_memory_daily.csv"
    if ref_csv.exists():
        ref_df = pd.read_csv(ref_csv).copy()
        ref_df["daily_return"] = ref_df["next_return"]
        ref_df["position"] = 1.0
        ref_df["decision"] = "Hold"
        bnh_stats = calculate_trade_stats(ref_df, initial_capital=args.capital)
        bnh_stats["system"] = names_map["buy_and_hold"]
        bnh_stats["system_key"] = "buy_and_hold"
        summary_rows.append(bnh_stats)

    # Print terminal executive banner
    print_pnl_banner(summary_rows, args.ticker, llm_client.model, args.risk_mode, args.capital)

    # Print day-by-day decisions for "Ours"
    ours_csv = output_dir / "ours_daily.csv"
    if ours_csv.exists():
        ours_df = pd.read_csv(ours_csv)
        print(f"\n📋 Day-by-Day LLM Decisions & Realized Returns for 'Ours' ({args.ticker}):")
        print("-" * 105)
        print(f"{'Date':<12} | {'Decision':<8} | {'Position':<8} | {'Price ($)':<10} | {'Day Return':<10} | {'Cum Return':<10} | {'LLM Reasoning':<40}")
        print("-" * 105)
        ours_cum = (1.0 + ours_df["daily_return"]).cumprod() - 1.0
        for idx, row in ours_df.iterrows():
            d_str = str(row["date"])
            dec = str(row["decision"])
            pos = f"{row['position']:+.1f}"
            price = f"${row['price']:.2f}"
            day_ret = f"{row['daily_return']*100:+.2f}%"
            cum_ret = f"{ours_cum.iloc[idx]*100:+.2f}%"
            reason = str(row["reasoning"])[:38] + "..." if len(str(row["reasoning"])) > 40 else str(row["reasoning"])
            print(f"{d_str:<12} | {dec:<8} | {pos:<8} | {price:<10} | {day_ret:<10} | {cum_ret:<10} | {reason:<40}")
        print("-" * 105 + "\n")

    # Export detailed PnL summary JSON
    pnl_summary_path = output_dir / "pnl_summary.json"
    with open(pnl_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"✅ Saved complete Profit & Loss summary to: {pnl_summary_path}")
    print(f"✅ Evaluation artifacts written to: {output_dir}\n")


if __name__ == "__main__":
    main()

