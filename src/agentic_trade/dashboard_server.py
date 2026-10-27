"""
High-performance Dashboard & Agentic Trading Harness Server.
Provides zero-dependency HTTP server with REST APIs for:
- Real-time Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Technical Signals & Sentiment Recommendation Engine
- Manual Trading Terminal with Position & Risk Management
- Automated Agentic AI Trading Engine (Groq + OpenRouter Free Models Failover)
- Strategy & Risk Profile Switching (Aggressive, Standard, Conservative)
- Cognitive Memory Evidence & Research Benchmarks (Tables 2-7)
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None

from .llm import DEFAULT_GROQ_MODELS, DEFAULT_OPENROUTER_FREE_MODELS, OpenRouterClient, SYSTEM_PROMPTS


def find_workspace_root() -> Path:
    """Find the project root containing artifacts or pyproject.toml."""
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        if (curr / "pyproject.toml").exists() or (curr / "artifacts").exists():
            return curr
        curr = curr.parent
    return Path.cwd()


WORKSPACE_ROOT = find_workspace_root()
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
DATA_DIR = WORKSPACE_ROOT / "data"
DASHBOARD_DIR = WORKSPACE_ROOT / "dashboard"
PORTFOLIO_STATE_FILE = ARTIFACTS_DIR / "paper_trading" / "portfolio_state.json"
ORDERS_LOG_FILE = ARTIFACTS_DIR / "paper_trading" / "orders.jsonl"


def clean_val(val: Any) -> Any:
    """Ensure floating NaNs and Infs become None for JSON."""
    if val is None:
        return None
    if isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf")):
        return None
    return val


def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: clean_val(v) for k, v in d.items()}


def get_latest_prices() -> Dict[str, float]:
    """Retrieve the latest close prices for all tickers."""
    prices_path = DATA_DIR / "prices.csv"
    if not prices_path.exists() or pd is None:
        return {
            "TSLA": 372.22,
            "AMZN": 248.00,
            "MSFT": 524.51,
            "NFLX": 65.85,
            "COIN": 164.45,
        }
    try:
        df = pd.read_csv(prices_path)
        return df.groupby("ticker").last()["close"].to_dict()
    except Exception:
        return {
            "TSLA": 372.22,
            "AMZN": 248.00,
            "MSFT": 524.51,
            "NFLX": 65.85,
            "COIN": 164.45,
        }


def get_summary_data() -> Dict[str, Any]:
    """Overall summary data for header and KPI ribbon."""
    summary: Dict[str, Any] = {
        "project": "Agentic Trade Research & Live Harness",
        "description": "Cognitive Trading Memory System (FinMem Base + 4 Architectural Novelties)",
        "datasets": ["offline", "llm_eval"],
        "tickers": ["TSLA", "AMZN", "MSFT", "NFLX", "COIN"],
        "trials": [0, 1, 2],
        "has_offline": (ARTIFACTS_DIR / "offline" / "metrics.json").exists(),
        "has_matrix": (ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv").exists(),
        "has_orders": (ARTIFACTS_DIR / "paper_trading" / "orders.jsonl").exists(),
        "latest_prices": get_latest_prices(),
    }

    # Discover available tickers from data/prices.csv
    prices_path = DATA_DIR / "prices.csv"
    if prices_path.exists() and pd is not None:
        try:
            df = pd.read_csv(prices_path)
            tickers = sorted(df["ticker"].dropna().unique().tolist())
            if tickers:
                summary["tickers"] = tickers
        except Exception:
            pass

    # Wilcoxon summary
    wilcoxon_json = ARTIFACTS_DIR / "research" / "matrix" / "wilcoxon.json"
    if wilcoxon_json.exists():
        try:
            with open(wilcoxon_json) as f:
                summary["wilcoxon"] = json.load(f)
        except Exception:
            summary["wilcoxon"] = None

    return summary


def get_market_data(ticker: str = "TSLA") -> Dict[str, Any]:
    """Calculate full price series and technical indicators for a ticker.

    Indicators: SMA 20, SMA 50, EMA 12, EMA 26, RSI 14, MACD (12, 26, 9),
    Bollinger Bands (20, 2 std), ATR.
    """
    ticker = ticker.upper()
    prices_path = DATA_DIR / "prices.csv"
    if not prices_path.exists() or pd is None:
        return {"ticker": ticker, "dates": [], "prices": [], "indicators": {}}

    try:
        df = pd.read_csv(prices_path)
        sub = df[df["ticker"] == ticker].sort_values("date").copy()
        if sub.empty:
            sub = df[df["ticker"] == "TSLA"].sort_values("date").copy()
            ticker = "TSLA"

        close = sub["close"].astype(float)
        dates = [str(d) for d in sub["date"]]
        prices = [round(float(p), 2) for p in close]

        # Moving Averages
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        # MACD
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0)

        # Bollinger Bands (20, 2)
        bb_std = close.rolling(20, min_periods=1).std().fillna(0.0)
        bb_upper = sma20 + (2.0 * bb_std)
        bb_lower = sma20 - (2.0 * bb_std)

        # High/Low/Open synthesis for candlestick view
        opens = [prices[0]] + prices[:-1]
        highs = [round(max(o, c) * 1.004, 2) for o, c in zip(opens, prices)]
        lows = [round(min(o, c) * 0.996, 2) for o, c in zip(opens, prices)]

        # Relative change
        prev_c = prices[-2] if len(prices) > 1 else prices[-1]
        change_pct = round(((prices[-1] - prev_c) / prev_c) * 100.0, 2)

        candles = []
        for i in range(len(dates)):
            candles.append({
                "date": dates[i],
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": prices[i],
                "volume": int(abs(prices[i] - opens[i]) * 100000 + 500000),
            })

        return {
            "ticker": ticker,
            "latest_price": prices[-1] if prices else 0.0,
            "change_pct": change_pct,
            "dates": dates,
            "prices": prices,
            "candles": candles,
            "indicators": {
                "sma_20": [round(float(v), 2) if not np.isnan(v) else None for v in sma20],
                "sma_50": [round(float(v), 2) if not np.isnan(v) else None for v in sma50],
                "ema_12": [round(float(v), 2) if not np.isnan(v) else None for v in ema12],
                "ema_26": [round(float(v), 2) if not np.isnan(v) else None for v in ema26],
                "rsi_14": [round(float(v), 2) if not np.isnan(v) else 50.0 for v in rsi],
                "macd": [round(float(v), 2) if not np.isnan(v) else 0.0 for v in macd],
                "macd_signal": [round(float(v), 2) if not np.isnan(v) else 0.0 for v in macd_signal],
                "macd_hist": [round(float(v), 2) if not np.isnan(v) else 0.0 for v in macd_hist],
                "bb_upper": [round(float(v), 2) if not np.isnan(v) else None for v in bb_upper],
                "bb_mid": [round(float(v), 2) if not np.isnan(v) else None for v in sma20],
                "bb_lower": [round(float(v), 2) if not np.isnan(v) else None for v in bb_lower],
            },
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "dates": [], "prices": [], "indicators": {}}


def get_technical_analysis(ticker: str = "TSLA") -> Dict[str, Any]:
    """Evaluate technical indicators and output recommendation signals."""
    m_data = get_market_data(ticker)
    if "error" in m_data or not m_data.get("prices"):
        return {"ticker": ticker, "recommendation": "NEUTRAL", "score": 0, "signals": {}}

    prices = m_data["prices"]
    ind = m_data["indicators"]
    latest_price = prices[-1]

    latest_rsi = ind["rsi_14"][-1] if ind.get("rsi_14") else 50.0
    latest_macd = ind["macd"][-1] if ind.get("macd") else 0.0
    latest_signal = ind["macd_signal"][-1] if ind.get("macd_signal") else 0.0
    latest_hist = ind["macd_hist"][-1] if ind.get("macd_hist") else 0.0
    latest_sma20 = ind["sma_20"][-1] if ind.get("sma_20") else latest_price
    latest_sma50 = ind["sma_50"][-1] if ind.get("sma_50") else latest_price
    latest_bbupper = ind["bb_upper"][-1] if ind.get("bb_upper") else latest_price * 1.05
    latest_bblower = ind["bb_lower"][-1] if ind.get("bb_lower") else latest_price * 0.95

    # Scoring Engine (-5 to +5)
    score = 0
    signals = {}

    # 1. RSI
    if latest_rsi < 30:
        rsi_status = "Oversold (Rebound edge)"
        rsi_signal = "Bullish"
        score += 2
    elif latest_rsi > 70:
        rsi_status = "Overbought (Pullback risk)"
        rsi_signal = "Bearish"
        score -= 2
    elif latest_rsi >= 50:
        rsi_status = "Neutral / Bullish momentum"
        rsi_signal = "Bullish"
        score += 1
    else:
        rsi_status = "Neutral / Bearish bias"
        rsi_signal = "Bearish"
        score -= 1

    signals["rsi"] = {
        "value": round(latest_rsi, 1),
        "status": rsi_status,
        "signal": rsi_signal,
    }

    # 2. MACD
    if latest_macd > latest_signal:
        macd_status = "Bullish Crossover (Above Signal Line)"
        macd_signal = "Bullish"
        score += 2
    else:
        macd_status = "Bearish Crossover (Below Signal Line)"
        macd_signal = "Bearish"
        score -= 2

    signals["macd"] = {
        "macd": round(latest_macd, 2),
        "signal": round(latest_signal, 2),
        "histogram": round(latest_hist, 2),
        "status": macd_status,
        "signal": macd_signal,
    }

    # 3. Moving Averages Trend
    if latest_price > latest_sma20 and latest_sma20 > latest_sma50:
        ma_status = "Golden Alignment (Price > SMA20 > SMA50)"
        ma_signal = "Strong Bullish"
        score += 2
    elif latest_price < latest_sma20 and latest_sma20 < latest_sma50:
        ma_status = "Death Alignment (Price < SMA20 < SMA50)"
        ma_signal = "Strong Bearish"
        score -= 2
    elif latest_price > latest_sma20:
        ma_status = "Short-term Uptrend (Price > SMA20)"
        ma_signal = "Bullish"
        score += 1
    else:
        ma_status = "Short-term Downtrend (Price < SMA20)"
        ma_signal = "Bearish"
        score -= 1

    signals["moving_averages"] = {
        "sma_20": round(latest_sma20, 2),
        "sma_50": round(latest_sma50, 2),
        "status": ma_status,
        "signal": ma_signal,
    }

    # 4. Bollinger Bands (%B)
    bb_range = latest_bbupper - latest_bblower
    pct_b = (latest_price - latest_bblower) / bb_range if bb_range > 0 else 0.5
    if pct_b > 0.8:
        bb_status = "Testing Upper Band (High Volatility)"
    elif pct_b < 0.2:
        bb_status = "Testing Lower Band (Oversold Support)"
    else:
        bb_status = "Trading within Normal Volatility Channel"

    signals["bollinger"] = {
        "upper": round(latest_bbupper, 2),
        "lower": round(latest_bblower, 2),
        "percent_b": round(pct_b * 100.0, 1),
        "status": bb_status,
    }

    # Final Technical Recommendation
    if score >= 3:
        rec = "STRONG BUY"
        color = "#10b981"
    elif score in [1, 2]:
        rec = "BUY"
        color = "#3b82f6"
    elif score == 0:
        rec = "NEUTRAL"
        color = "#f59e0b"
    elif score in [-2, -1]:
        rec = "SELL"
        color = "#f97316"
    else:
        rec = "STRONG SELL"
        color = "#ef4444"

    return {
        "ticker": ticker,
        "price": round(latest_price, 2),
        "recommendation": rec,
        "recommendation_color": color,
        "score": score,
        "signals": signals,
    }


class PortfolioManager:
    """Manages paper trading portfolio state, live executions, and risk settings."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.cash: float = 100000.0
        self.initial_capital: float = 100000.0
        self.realized_pnl: float = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.settings: Dict[str, Any] = {
            "risk_profile": "aggressive",
            "provider": "auto",
            "groq_model": "openai/gpt-oss-120b",
            "openrouter_model": "recursive/flint-v1.0:free",
            "max_position_pct": 25.0,
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "auto_trading_enabled": False,
            "auto_interval_sec": 30,
        }
        self.load()

    def load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.cash = float(data.get("cash", 100000.0))
                self.initial_capital = float(data.get("initial_capital", 100000.0))
                self.realized_pnl = float(data.get("realized_pnl", 0.0))
                self.positions = data.get("positions", {})
                self.settings.update(data.get("settings", {}))
            except Exception:
                pass

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "positions": self.positions,
            "settings": self.settings,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_state(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        prices = current_prices or get_latest_prices()
        unrealized_pnl = 0.0
        position_list = []

        for ticker, pos in list(self.positions.items()):
            qty = float(pos.get("shares", 0.0))
            if qty <= 0.0001:
                continue
            avg = float(pos.get("avg_price", 0.0))
            mark = float(prices.get(ticker, avg))
            side = pos.get("side", "Long")

            if side == "Long":
                pos_val = qty * mark
                u_pnl = (mark - avg) * qty
            else:
                pos_val = qty * avg - (mark - avg) * qty
                u_pnl = (avg - mark) * qty

            u_pnl_pct = (u_pnl / (qty * avg) * 100.0) if (qty * avg) > 0 else 0.0
            unrealized_pnl += u_pnl

            position_list.append({
                "ticker": ticker,
                "side": side,
                "shares": round(qty, 4),
                "avg_price": round(avg, 2),
                "current_price": round(mark, 2),
                "current_value": round(pos_val, 2),
                "unrealized_pnl": round(u_pnl, 2),
                "unrealized_pnl_pct": round(u_pnl_pct, 2),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
            })

        total_equity = self.cash + sum(p["current_value"] for p in position_list)
        total_pnl = self.realized_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / self.initial_capital * 100.0) if self.initial_capital > 0 else 0.0

        # Load recent orders
        recent_orders = []
        if ORDERS_LOG_FILE.exists():
            try:
                with open(ORDERS_LOG_FILE, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            recent_orders.append(json.loads(line))
            except Exception:
                pass

        return {
            "cash": round(self.cash, 2),
            "equity": round(total_equity, 2),
            "initial_capital": round(self.initial_capital, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions": position_list,
            "recent_orders": recent_orders[::-1][:50],
            "settings": self.settings,
        }

    def execute_order(
        self,
        ticker: str,
        action: str,
        shares: float,
        order_type: str = "Market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        is_ai: bool = False,
        reasoning: str = "",
        provider: str = "",
        model: str = "",
    ) -> Dict[str, Any]:
        ticker = ticker.upper()
        action = action.title()
        prices = get_latest_prices()
        price = prices.get(ticker, 100.0)
        notional = shares * price

        if action == "Buy":
            if self.cash < notional:
                # Cap to available cash
                if self.cash < price:
                    return {"success": False, "error": f"Insufficient funds: ${self.cash:,.2f} available for ${notional:,.2f} order."}
                shares = self.cash / price
                notional = shares * price

            self.cash -= notional
            if ticker in self.positions and self.positions[ticker].get("side") == "Long":
                old_shares = self.positions[ticker]["shares"]
                old_avg = self.positions[ticker]["avg_price"]
                new_shares = old_shares + shares
                new_avg = (old_shares * old_avg + notional) / new_shares
                self.positions[ticker]["shares"] = new_shares
                self.positions[ticker]["avg_price"] = new_avg
            else:
                self.positions[ticker] = {
                    "ticker": ticker,
                    "side": "Long",
                    "shares": shares,
                    "avg_price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

        elif action == "Sell":
            if ticker in self.positions:
                pos = self.positions[ticker]
                old_shares = pos["shares"]
                old_avg = pos["avg_price"]
                sell_qty = min(old_shares, shares)
                pnl = (price - old_avg) * sell_qty
                self.realized_pnl += pnl
                self.cash += sell_qty * price
                remaining = old_shares - sell_qty
                if remaining <= 0.0001:
                    del self.positions[ticker]
                else:
                    pos["shares"] = remaining
            else:
                # Short position
                self.cash += notional
                self.positions[ticker] = {
                    "ticker": ticker,
                    "side": "Short",
                    "shares": shares,
                    "avg_price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

        # Log order
        order_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "decision": action,
            "action": action,
            "shares": round(shares, 4),
            "price": round(price, 2),
            "notional": round(notional, 2),
            "order_type": order_type,
            "is_ai": is_ai,
            "reasoning": reasoning or "Manual Execution",
            "provider": provider,
            "model": model,
            "status": "filled",
        }

        ORDERS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ORDERS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(order_record) + "\n")

        self.save()
        return {"success": True, "order": order_record, "portfolio": self.get_state()}

    def close_position(self, ticker: str) -> Dict[str, Any]:
        ticker = ticker.upper()
        if ticker not in self.positions:
            return {"success": False, "error": f"No open position in {ticker}."}

        pos = self.positions[ticker]
        shares = pos["shares"]
        return self.execute_order(
            ticker=ticker,
            action="Sell",
            shares=shares,
            order_type="Market",
            reasoning=f"Closed position in {ticker}",
        )

    def reset(self, initial_capital: float = 100000.0) -> Dict[str, Any]:
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.realized_pnl = 0.0
        self.positions = {}
        self.save()
        return {"success": True, "portfolio": self.get_state()}

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        self.settings.update(new_settings)
        self.save()
        return {"success": True, "settings": self.settings}


# Global portfolio manager instance
portfolio_mgr = PortfolioManager(PORTFOLIO_STATE_FILE)


def execute_ai_trade(
    ticker: str = "TSLA",
    risk_mode: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute live AI decision for ticker using Groq + OpenRouter failover."""
    ticker = ticker.upper()
    active_risk = risk_mode or portfolio_mgr.settings.get("risk_profile", "aggressive")
    active_provider = provider or portfolio_mgr.settings.get("provider", "auto")

    # Load news events from data
    memories: List[Dict[str, Any]] = []
    news_csv = DATA_DIR / "multi_ticker_news.csv"
    if news_csv.exists() and pd is not None:
        try:
            df = pd.read_csv(news_csv)
            sub = df[df["ticker"] == ticker].tail(5)
            for idx, row in sub.iterrows():
                memories.append({
                    "id": str(row.get("source_id", idx)),
                    "source": str(row.get("source_type", "news")),
                    "date": str(row.get("date", "2026-10-23")),
                    "text": str(row.get("text", "")),
                })
        except Exception:
            pass

    # Momentum
    momentum = 0.01
    prices_csv = DATA_DIR / "prices.csv"
    if prices_csv.exists() and pd is not None:
        try:
            p_df = pd.read_csv(prices_csv)
            ts = p_df[p_df["ticker"] == ticker].sort_values("date")["close"].values
            if len(ts) >= 5:
                momentum = float((ts[-1] - ts[-5]) / ts[-5])
        except Exception:
            pass

    client = OpenRouterClient(
        temperature=0.3,
        primary_provider=active_provider,
        cache_dir=ARTIFACTS_DIR / "llm_cache",
    )

    decision = client.decide(
        system_name="ours",
        ticker=ticker,
        decision_date="2026-10-23",
        momentum=momentum,
        memories=memories,
        risk_mode=active_risk,
    )

    action = decision.get("action", "Hold")
    reasoning = decision.get("reasoning", "")
    provider_used = decision.get("provider", "groq")
    model_used = decision.get("model", "openai/gpt-oss-120b")
    cited_ids = decision.get("cited_memory_ids", [])

    order_result = None
    if action == "Buy":
        # Calculate sizing: 15% of cash or $5,000
        budget = min(portfolio_mgr.cash * 0.15, 5000.0)
        p = get_latest_prices().get(ticker, 100.0)
        shares = budget / p if p > 0 else 1.0
        order_result = portfolio_mgr.execute_order(
            ticker=ticker,
            action="Buy",
            shares=shares,
            is_ai=True,
            reasoning=f"AI [{provider_used.upper()} / {model_used}]: {reasoning}",
            provider=provider_used,
            model=model_used,
        )
    elif action == "Sell":
        if ticker in portfolio_mgr.positions:
            shares = portfolio_mgr.positions[ticker]["shares"]
            order_result = portfolio_mgr.execute_order(
                ticker=ticker,
                action="Sell",
                shares=shares,
                is_ai=True,
                reasoning=f"AI [{provider_used.upper()} / {model_used}]: {reasoning}",
                provider=provider_used,
                model=model_used,
            )

    return {
        "success": True,
        "ticker": ticker,
        "action": action,
        "reasoning": reasoning,
        "provider": provider_used,
        "model": model_used,
        "cited_memory_ids": cited_ids,
        "risk_mode": active_risk,
        "order": order_result["order"] if order_result else None,
        "portfolio": portfolio_mgr.get_state(),
    }


def get_equity_data(dataset: str = "offline", ticker: str = "TSLA", trial: int = 0) -> Dict[str, Any]:
    """Returns time-series returns and daily decisions for all systems."""
    systems = ["ours", "fin_mem", "plain_rag", "no_memory", "buy_and_hold"]
    series_map: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}
    metrics_map: Dict[str, Dict[str, Any]] = {}
    dates: List[str] = []
    prices: List[float] = []

    if dataset in ["offline", "llm_eval"]:
        target_dir = ARTIFACTS_DIR / dataset
        metrics_file = target_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics_map = json.load(f)
            except Exception:
                pass

        for s in ["ours", "fin_mem", "plain_rag", "no_memory"]:
            daily_file = target_dir / f"{s}_daily.csv"
            if daily_file.exists() and pd is not None:
                try:
                    df = pd.read_csv(daily_file)
                    df["cum_return"] = (1.0 + df["daily_return"]).cumprod() - 1.0
                    for _, row in df.iterrows():
                        series_map[s].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_return"]),
                            "daily_return": float(row["daily_return"]),
                            "decision": str(row["decision"]),
                            "position": float(row["position"]),
                            "price": float(row["price"]),
                        })
                    if s == "ours" and not dates:
                        dates = [str(d) for d in df["date"]]
                        prices = [float(p) for p in df["price"]]
                except Exception:
                    pass

        if series_map["no_memory"]:
            daily_file = target_dir / "no_memory_daily.csv"
            if daily_file.exists() and pd is not None:
                try:
                    df = pd.read_csv(daily_file)
                    df["cum_bnh"] = (1.0 + df["next_return"]).cumprod() - 1.0
                    for _, row in df.iterrows():
                        series_map["buy_and_hold"].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_bnh"]),
                            "daily_return": float(row["next_return"]),
                            "decision": "Hold",
                            "position": 1.0,
                            "price": float(row["price"]),
                        })
                except Exception:
                    pass

    else:
        daily_long_csv = ARTIFACTS_DIR / "research" / "matrix" / "daily_long.csv"
        metrics_csv = ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv"

        if metrics_csv.exists() and pd is not None:
            try:
                m_df = pd.read_csv(metrics_csv)
                sub_m = m_df[(m_df["ticker"] == ticker) & (m_df["trial"] == trial)]
                for _, row in sub_m.iterrows():
                    metrics_map[str(row["system"])] = clean_dict(row.to_dict())
            except Exception:
                pass

        if daily_long_csv.exists() and pd is not None:
            try:
                d_df = pd.read_csv(daily_long_csv)
                sub_d = d_df[(d_df["ticker"] == ticker) & (d_df["trial"] == trial)]

                for s in ["ours", "fin_mem", "plain_rag", "no_memory"]:
                    sys_sub = sub_d[sub_d["system"] == s].sort_values("date")
                    if not sys_sub.empty:
                        sys_sub["cum_return"] = (1.0 + sys_sub["daily_return"]).cumprod() - 1.0
                        for _, row in sys_sub.iterrows():
                            series_map[s].append({
                                "date": str(row["date"]),
                                "return": float(row["cum_return"]),
                                "daily_return": float(row["daily_return"]),
                                "decision": str(row["decision"]),
                                "position": float(row["position"]),
                                "price": float(row["price"]),
                            })
                        if s == "ours" and not dates:
                            dates = [str(d) for d in sys_sub["date"]]
                            prices = [float(p) for p in sys_sub["price"]]

                ref = sub_d[sub_d["system"] == "no_memory"].sort_values("date")
                if not ref.empty:
                    ref["cum_bnh"] = (1.0 + ref["next_return"]).cumprod() - 1.0
                    for _, row in ref.iterrows():
                        series_map["buy_and_hold"].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_bnh"]),
                            "daily_return": float(row["next_return"]),
                            "decision": "Hold",
                            "position": 1.0,
                            "price": float(row["price"]),
                        })
            except Exception:
                pass

    return {
        "dataset": dataset,
        "ticker": ticker,
        "trial": trial,
        "dates": dates,
        "prices": prices,
        "series": series_map,
        "metrics": metrics_map,
    }


def get_evidence_data(
    dataset: str = "offline",
    ticker: Optional[str] = None,
    system: Optional[str] = None,
    layer: Optional[str] = None,
    cited: Optional[bool] = None,
    active: Optional[bool] = None,
    outcome_recalled: Optional[bool] = None,
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    """Retrieve filtered evidence log items."""
    file_path = ARTIFACTS_DIR / "offline" / "evidence_log.csv"
    if dataset == "llm_eval":
        file_path = ARTIFACTS_DIR / "llm_eval" / "evidence_log.csv"
    elif dataset != "offline":
        file_path = ARTIFACTS_DIR / "research" / "matrix" / "evidence_log.csv"

    if not file_path.exists():
        return {"items": [], "total": 0, "page": page, "limit": limit}

    items: List[Dict[str, Any]] = []

    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ticker and row.get("ticker", "").strip().upper() != ticker.upper():
                continue
            if system and row.get("system", "").strip() != system:
                continue
            if layer and row.get("layer", "").strip() != layer:
                continue
            if cited is not None:
                is_cited = row.get("cited", "").strip().lower() in ["true", "1"]
                if is_cited != cited:
                    continue
            if active is not None:
                is_active = row.get("active", "").strip().lower() in ["true", "1"]
                if is_active != active:
                    continue
            if outcome_recalled is not None:
                is_recalled = row.get("outcome_recalled", "").strip().lower() in ["true", "1"]
                if is_recalled != outcome_recalled:
                    continue
            if search:
                s_lower = search.lower()
                text = (row.get("text", "") + " " + row.get("topic", "")).lower()
                if s_lower not in text:
                    continue

            items.append({
                "date": row.get("date", ""),
                "ticker": row.get("ticker", ""),
                "system": row.get("system", ""),
                "layer": row.get("layer", ""),
                "memory_id": row.get("memory_id", ""),
                "topic": row.get("topic", ""),
                "cited": row.get("cited", "").strip().lower() in ["true", "1"],
                "active": row.get("active", "").strip().lower() in ["true", "1"],
                "outcome_recalled": row.get("outcome_recalled", "").strip().lower() in ["true", "1"],
                "text": row.get("text", ""),
            })

    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    paged = items[start:end]

    return {"items": paged, "total": total, "page": page, "limit": limit}


def get_benchmarks_data() -> Dict[str, Any]:
    """Retrieve full research benchmark tables 2 through 7."""
    res: Dict[str, Any] = {}
    for tbl in ["table2_metrics", "table3_ablation", "table4_retrieval", "table5_decay", "table6_outcomes", "table7_probes"]:
        t_file = ARTIFACTS_DIR / "research" / "matrix" / f"{tbl}.csv"
        if t_file.exists() and pd is not None:
            try:
                df = pd.read_csv(t_file).replace({np.nan: None})
                res[tbl.split("_")[0]] = df.to_dict(orient="records")
            except Exception:
                pass

    w_file = ARTIFACTS_DIR / "research" / "matrix" / "wilcoxon.json"
    if w_file.exists():
        try:
            with open(w_file) as f:
                res["wilcoxon"] = json.load(f)
        except Exception:
            pass

    return res


def get_orders_data() -> Dict[str, Any]:
    """Retrieve paper trading orders and latest decisions."""
    orders = []
    if ORDERS_LOG_FILE.exists():
        try:
            with open(ORDERS_LOG_FILE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        orders.append(json.loads(line))
        except Exception:
            pass

    decisions = []
    decisions_csv = WORKSPACE_ROOT / "decisions.csv"
    if decisions_csv.exists() and pd is not None:
        try:
            df = pd.read_csv(decisions_csv).replace({np.nan: None})
            decisions = df.to_dict(orient="records")
        except Exception:
            pass

    return {
        "orders": orders[::-1],
        "decisions": decisions,
        "order_count": len(orders),
        "portfolio": portfolio_mgr.get_state(),
    }


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the complete Agentic Trade Application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        if args and str(args[0]).startswith("50"):
            super().log_message(format, *args)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ["/", "/index.html"]:
            index_path = DASHBOARD_DIR / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"index.html not found.")
            return

        if path == "/api/summary":
            self._send_json(get_summary_data())
            return

        if path == "/api/market_data":
            ticker = query.get("ticker", ["TSLA"])[0]
            self._send_json(get_market_data(ticker=ticker))
            return

        if path == "/api/technical_analysis":
            ticker = query.get("ticker", ["TSLA"])[0]
            self._send_json(get_technical_analysis(ticker=ticker))
            return

        if path == "/api/portfolio":
            self._send_json(portfolio_mgr.get_state())
            return

        if path == "/api/settings":
            self._send_json({
                "settings": portfolio_mgr.settings,
                "groq_models": DEFAULT_GROQ_MODELS,
                "openrouter_free_models": DEFAULT_OPENROUTER_FREE_MODELS,
                "system_prompts": SYSTEM_PROMPTS,
            })
            return

        if path == "/api/equity":
            dataset = query.get("dataset", ["offline"])[0]
            ticker = query.get("ticker", ["TSLA"])[0]
            trial_str = query.get("trial", ["0"])[0]
            try:
                trial = int(trial_str)
            except ValueError:
                trial = 0
            self._send_json(get_equity_data(dataset=dataset, ticker=ticker, trial=trial))
            return

        if path == "/api/evidence":
            dataset = query.get("dataset", ["offline"])[0]
            ticker = query.get("ticker", [None])[0]
            system = query.get("system", [None])[0]
            layer = query.get("layer", [None])[0]
            search = query.get("search", [None])[0]
            cited = query.get("cited", [None])[0]
            active = query.get("active", [None])[0]
            recalled = query.get("outcome_recalled", [None])[0]
            page_str = query.get("page", ["1"])[0]
            limit_str = query.get("limit", ["50"])[0]

            is_cited = cited.lower() in ["true", "1"] if cited is not None else None
            is_active = active.lower() in ["true", "1"] if active is not None else None
            is_recalled = recalled.lower() in ["true", "1"] if recalled is not None else None

            try:
                page = max(1, int(page_str))
            except ValueError:
                page = 1
            try:
                limit = min(200, max(10, int(limit_str)))
            except ValueError:
                limit = 50

            data = get_evidence_data(
                dataset=dataset,
                ticker=ticker,
                system=system,
                layer=layer,
                cited=is_cited,
                active=is_active,
                outcome_recalled=is_recalled,
                search=search,
                page=page,
                limit=limit,
            )
            self._send_json(data)
            return

        if path == "/api/benchmarks":
            self._send_json(get_benchmarks_data())
            return

        if path == "/api/orders":
            self._send_json(get_orders_data())
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            raw_body = self.rfile.read(content_length)
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:
                body = {}

        if path == "/api/manual_trade":
            ticker = str(body.get("ticker", "TSLA")).upper()
            action = str(body.get("action", "Buy")).title()
            try:
                shares = float(body.get("shares", body.get("quantity", 1.0)))
            except (ValueError, TypeError):
                shares = 1.0
            order_type = str(body.get("order_type", "Market"))
            stop_loss = float(body["stop_loss"]) if body.get("stop_loss") else None
            take_profit = float(body["take_profit"]) if body.get("take_profit") else None

            res = portfolio_mgr.execute_order(
                ticker=ticker,
                action=action,
                shares=shares,
                order_type=order_type,
                stop_loss=stop_loss,
                take_profit=take_profit,
                is_ai=False,
                reasoning=body.get("notes", "Manual terminal order"),
            )
            self._send_json(res)
            return

        if path == "/api/ai_trade":
            ticker = str(body.get("ticker", "TSLA")).upper()
            risk_mode = body.get("risk_mode")
            provider = body.get("provider")
            res = execute_ai_trade(ticker=ticker, risk_mode=risk_mode, provider=provider)
            self._send_json(res)
            return

        if path == "/api/close_position":
            ticker = str(body.get("ticker", "TSLA")).upper()
            res = portfolio_mgr.close_position(ticker)
            self._send_json(res)
            return

        if path == "/api/reset_portfolio":
            try:
                capital = float(body.get("initial_capital", 100000.0))
            except (ValueError, TypeError):
                capital = 100000.0
            res = portfolio_mgr.reset(initial_capital=capital)
            self._send_json(res)
            return

        if path == "/api/update_settings":
            res = portfolio_mgr.update_settings(body)
            self._send_json(res)
            return

        if path == "/api/run_offline":
            try:
                cmd = [sys.executable, "run_experiment.py", "--data-dir", "data", "--output-dir", "artifacts/offline"]
                res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, timeout=120)
                out = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
                self._send_json({"success": res.returncode == 0, "output": out})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        if path == "/api/run_paper_dryrun":
            try:
                cmd = [sys.executable, "-m", "agentic_trade.paper_trader", "--decisions", "decisions.csv", "--budget", "100"]
                res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, timeout=60)
                out = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
                self._send_json({"success": res.returncode == 0, "output": out})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        blob = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(blob)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(port: int = 8050, host: str = "127.0.0.1") -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    actual_port = port
    for p in range(port, port + 20):
        try:
            httpd = ThreadedHTTPServer((host, p), DashboardHandler)
            actual_port = p
            break
        except OSError:
            continue
    else:
        print(f"Error: Could not bind to any port in range {port}-{port+20}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" 🚀 AGENTIC TRADE HARNESS & TRADING APPLICATION SERVER RUNNING")
    print(f" 🌐 URL: http://{host}:{actual_port}/")
    print(" 📊 Features: Interactive Candlesticks, Indicators (SMA, EMA, RSI, MACD, BB)")
    print(" 🤖 AI Agent Engine: Live Groq + OpenRouter Free Models Auto-Failover")
    print(" 💼 Portfolio: Manual Execution, Paper Broker, Realized/Unrealized PnL")
    print(" ⌨️  Press Ctrl+C to terminate.")
    print("=" * 70 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping application server...")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Trade Application Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default 8050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
