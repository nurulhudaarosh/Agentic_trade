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
from . import credentials
from .autopilot import Autopilot

credentials.apply()  # keys saved in the Settings tab override .env (imported after .llm loaded it)


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


def get_available_tickers() -> list[str]:
    """Return the study universe, local monitored symbols and symbols held at Alpaca."""
    from .study.data import UNIVERSE

    tickers: set[str] = set(UNIVERSE)
    prices_path = DATA_DIR / "prices.csv"
    if prices_path.exists() and pd is not None:
        try:
            frame = pd.read_csv(prices_path, usecols=["ticker"])
            tickers.update(frame["ticker"].dropna().astype(str).str.upper().unique().tolist())
        except Exception:
            pass
    tickers.update(position["ticker"] for position in get_alpaca_positions())
    return sorted(tickers or {"TSLA", "AMZN", "MSFT", "NFLX", "COIN"})


def _get_alpaca_trading_client() -> Any:
    """Create the Alpaca paper client when paper credentials are configured."""
    api_key = os.environ.get("ALPACA_PAPER_KEY") or os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_PAPER_SECRET") or os.environ.get("ALPACA_SECRET")
    if not api_key or not api_secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        return TradingClient(api_key, api_secret, paper=True)
    except Exception:
        return None


def alpaca_mode_enabled() -> bool:
    return _get_alpaca_trading_client() is not None


def get_alpaca_positions() -> list[Dict[str, Any]]:
    """Read current Alpaca paper positions without mutating the local simulator."""
    client = _get_alpaca_trading_client()
    if client is None:
        return []
    try:
        positions = []
        for position in client.get_all_positions():
            qty = float(position.qty)
            avg_price = float(position.avg_entry_price)
            market_value = float(position.market_value)
            positions.append({
                "ticker": str(position.symbol).upper(),
                "side": "Short" if qty < 0 else "Long",
                "shares": round(abs(qty), 4),
                "avg_price": round(avg_price, 2),
                "current_price": round(market_value / abs(qty), 2) if qty else avg_price,
                "invested_cost": round(abs(qty) * avg_price, 2),
                "current_value": round(abs(market_value), 2),
                "unrealized_pnl": round(float(position.unrealized_pl), 2),
                "unrealized_pnl_pct": round(float(position.unrealized_plpc) * 100.0, 2),
                "is_profit": float(position.unrealized_pl) >= 0,
                "source": "Alpaca paper",
                "broker_position": True,
            })
        return positions
    except Exception:
        return []


def submit_alpaca_paper_order(
    ticker: str,
    action: str,
    shares: float,
    order_type: str = "Market",
    is_ai: bool = False,
    reasoning: str = "",
    provider: str = "",
    model: str = "",
) -> Dict[str, Any]:
    """Submit a dashboard order to Alpaca Paper and return an audit-ready record."""
    client = _get_alpaca_trading_client()
    if client is None:
        return {"success": False, "error": "Alpaca Paper credentials are not configured."}
    if order_type.lower() != "market":
        return {"success": False, "error": "Dashboard Alpaca execution currently supports Market orders only."}
    if shares <= 0:
        return {"success": False, "error": "Share quantity must be greater than zero."}
    try:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL
        order = client.submit_order(MarketOrderRequest(
            symbol=ticker.upper(),
            qty=round(shares, 6),
            side=side,
            time_in_force=TimeInForce.DAY,
        ))
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker.upper(),
            "decision": action.title(),
            "action": action.title(),
            "shares": round(shares, 6),
            "price": float(get_latest_prices().get(ticker.upper(), 0.0)),
            "notional": None,
            "order_type": "Market",
            "is_ai": is_ai,
            "reasoning": reasoning or "Alpaca Paper dashboard execution",
            "provider": provider,
            "model": model,
            "status": str(getattr(order, "status", "accepted")),
            "order_id": str(getattr(order, "id", "")),
            "broker": "Alpaca paper",
        }
        ORDERS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ORDERS_LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return {"success": True, "order": record, "portfolio": None}
    except Exception as exc:
        return {"success": False, "error": f"Alpaca Paper order failed: {exc}"}


def _get_alpaca_latest_prices(tickers: Optional[list[str]] = None) -> Dict[str, float]:
    """Return latest Alpaca paper prices when market-data credentials are available."""
    api_key = os.environ.get("ALPACA_PAPER_KEY") or os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_PAPER_SECRET") or os.environ.get("ALPACA_SECRET")
    if not api_key or not api_secret:
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        symbols = tickers or get_available_tickers()
        client = StockHistoricalDataClient(api_key, api_secret)
        trades = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))
        return {
            symbol: round(float(trade.price), 4)
            for symbol, trade in trades.items()
            if getattr(trade, "price", None) is not None
        }
    except Exception:
        return {}


def get_latest_prices() -> Dict[str, float]:
    """Retrieve latest paper-market prices, falling back to the local dataset."""
    fallback = {"TSLA": 372.22, "AMZN": 248.00, "MSFT": 524.51, "NFLX": 65.85, "COIN": 164.45}

    live_prices = _get_alpaca_latest_prices(get_available_tickers())
    if live_prices:
        return {**fallback, **live_prices}

    prices_path = DATA_DIR / "prices.csv"
    if not prices_path.exists() or pd is None:
        return fallback
    try:
        df = pd.read_csv(prices_path)
        return df.groupby("ticker").last()["close"].to_dict()
    except Exception:
        return fallback


def get_summary_data() -> Dict[str, Any]:
    """Overall summary data for header and KPI ribbon."""
    summary: Dict[str, Any] = {
        "project": "Agentic Trade Research & Live Harness",
        "description": "Cognitive Trading Memory System (FinMem Base + 4 Architectural Novelties)",
        "datasets": ["offline", "llm_eval"],
        "tickers": get_available_tickers(),
        "trials": [0, 1, 2],
        "has_offline": (ARTIFACTS_DIR / "offline" / "metrics.json").exists(),
        "has_matrix": (ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv").exists(),
        "has_orders": (ARTIFACTS_DIR / "paper_trading" / "orders.jsonl").exists(),
        "latest_prices": get_latest_prices(),
            "alpaca_positions": get_alpaca_positions(),
    }

    # Wilcoxon summary
    wilcoxon_json = ARTIFACTS_DIR / "research" / "matrix" / "wilcoxon.json"
    if wilcoxon_json.exists():
        try:
            with open(wilcoxon_json) as f:
                summary["wilcoxon"] = json.load(f)
        except Exception:
            summary["wilcoxon"] = None

    return summary


def _daily_bars(ticker: str) -> Any:
    """Real daily OHLCV for ``ticker``: the study panel (10 stocks) first, else data/prices.csv."""
    study = WORKSPACE_ROOT / "data" / "study" / "prices.parquet"
    if study.exists():
        df = pd.read_parquet(study, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        sub = df[df["ticker"] == ticker].sort_values("date")
        if not sub.empty:
            return sub.assign(date=pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d"))[["date", "open", "high", "low", "close", "volume"]]
    csv = DATA_DIR / "prices.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        sub = df[df["ticker"] == ticker].sort_values("date")
        if not sub.empty:
            # This file has closes only: show each day as open = previous close, without invented wicks.
            close = sub["close"].astype(float)
            opens = close.shift(1).fillna(close)
            return pd.DataFrame({"date": sub["date"].astype(str), "open": opens, "high": np.maximum(opens, close),
                                 "low": np.minimum(opens, close), "close": close, "volume": 0})
    return None


def get_market_data(ticker: str = "TSLA", days: int = 180) -> Dict[str, Any]:
    """Daily price series (real OHLCV) plus indicators for a ticker.

    Indicators: SMA 20, SMA 50, EMA 12, EMA 26, RSI 14, MACD (12, 26, 9),
    Bollinger Bands (20, 2 std). They are computed on the full history and the
    last ``days`` sessions are returned. A live Alpaca quote, when available,
    updates (or adds) today's bar.
    """
    ticker = ticker.upper()
    if pd is None:
        return {"ticker": ticker, "dates": [], "prices": [], "indicators": {}}

    try:
        bars = _daily_bars(ticker)
        live_price = _get_alpaca_latest_prices([ticker]).get(ticker)
        if bars is None:
            if live_price is None:
                return {"ticker": ticker, "error": f"No price data for {ticker}", "dates": [], "prices": [], "indicators": {}}
            bars = pd.DataFrame({"date": [datetime.now(timezone.utc).date().isoformat()], "open": [live_price],
                                 "high": [live_price], "low": [live_price], "close": [live_price], "volume": [0]})
        bars = bars.reset_index(drop=True)
        price_source = "historical"
        if live_price is not None:
            today = datetime.now(timezone.utc).date().isoformat()
            lp = float(live_price)
            if bars["date"].iloc[-1] == today:
                i = bars.index[-1]
                bars.loc[i, ["close", "high", "low"]] = [lp, max(bars.at[i, "high"], lp), min(bars.at[i, "low"], lp)]
            else:
                prev = float(bars["close"].iloc[-1])
                bars.loc[len(bars)] = [today, prev, max(prev, lp), min(prev, lp), lp, 0]
            price_source = "alpaca_paper_live"
        close = bars["close"].astype(float)
        dates = list(bars["date"])
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

        prev_c = prices[-2] if len(prices) > 1 else prices[-1]
        change_pct = round(((prices[-1] - prev_c) / prev_c) * 100.0, 2)
        candles = [{"date": r.date, "open": round(float(r.open), 2), "high": round(float(r.high), 2),
                    "low": round(float(r.low), 2), "close": round(float(r.close), 2), "volume": int(r.volume or 0)}
                   for r in bars.itertuples()]

        keep = slice(-days, None) if days and len(dates) > days else slice(None)
        series = lambda values, fill=None: [round(float(v), 2) if not np.isnan(v) else fill for v in list(values)[keep]]
        return {
            "ticker": ticker,
            "latest_price": prices[-1] if prices else 0.0,
            "change_pct": change_pct,
            "price_source": price_source,
            "as_of": dates[-1] if dates else None,
            "dates": dates[keep],
            "prices": prices[keep],
            "candles": candles[keep],
            "indicators": {
                "sma_20": series(sma20), "sma_50": series(sma50), "ema_12": series(ema12), "ema_26": series(ema26),
                "rsi_14": series(rsi, 50.0), "macd": series(macd, 0.0), "macd_signal": series(macd_signal, 0.0),
                "macd_hist": series(macd_hist, 0.0), "bb_upper": series(bb_upper), "bb_mid": series(sma20),
                "bb_lower": series(bb_lower),
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

    def __init__(self, state_file: Path, use_alpaca: bool = False) -> None:
        self.state_file = state_file
        self.use_alpaca = use_alpaca
        self.cash: float = 100000.0
        self.initial_capital: float = 100000.0
        self.realized_pnl: float = 0.0
        self.daily_start_equity: float = 100000.0
        self.daily_start_date: str = datetime.now(timezone.utc).date().isoformat()
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
                self.daily_start_equity = float(data.get("daily_start_equity", self.initial_capital))
                self.daily_start_date = str(data.get("daily_start_date", datetime.now(timezone.utc).date().isoformat()))
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
            "daily_start_equity": self.daily_start_equity,
            "daily_start_date": self.daily_start_date,
            "positions": self.positions,
            "settings": self.settings,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _recent_orders(self) -> list[dict[str, Any]]:
        if not ORDERS_LOG_FILE.exists():
            return []
        try:
            with ORDERS_LOG_FILE.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()][::-1][:50]
        except Exception:
            return []

    def _get_alpaca_state(self) -> Dict[str, Any]:
        client = _get_alpaca_trading_client()
        if client is None:
            return {}
        try:
            account = client.get_account()
            equity = float(account.equity)
            last_equity = float(getattr(account, "last_equity", equity))
            positions = get_alpaca_positions()
            daily_pnl = equity - last_equity
            daily_loss_limit = last_equity * float(self.settings.get("max_daily_loss_pct", 2.0)) / 100.0
            largest_position_pct = max(
                ((position["current_value"] / equity * 100.0) if equity else 0.0 for position in positions),
                default=0.0,
            )
            return {
                "cash": round(float(account.cash), 2),
                "equity": round(equity, 2),
                "initial_capital": round(last_equity, 2),
                "realized_pnl": 0.0,
                "unrealized_pnl": round(sum(p["unrealized_pnl"] for p in positions), 2),
                "total_pnl": round(daily_pnl, 2),
                "total_pnl_pct": round((daily_pnl / last_equity * 100.0) if last_equity else 0.0, 2),
                "risk": {
                    "daily_pnl": round(daily_pnl, 2),
                    "daily_loss_limit": round(daily_loss_limit, 2),
                    "daily_loss_pct": round((daily_pnl / last_equity * 100.0) if last_equity else 0.0, 2),
                    "largest_position_pct": round(largest_position_pct, 2),
                    "max_position_pct": float(self.settings.get("max_position_pct", 25.0)),
                    "daily_start_date": datetime.now(timezone.utc).date().isoformat(),
                },
                "positions": positions,
                "alpaca_positions": positions,
                "recent_orders": self._recent_orders(),
                "settings": {**self.settings, "execution_mode": "alpaca_paper"},
                "execution_mode": "alpaca_paper",
            }
        except Exception as exc:
            return {"error": f"Unable to read Alpaca Paper account: {exc}"}

    def get_state(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if self.use_alpaca and alpaca_mode_enabled():
            broker_state = self._get_alpaca_state()
            if broker_state and "error" not in broker_state:
                return broker_state
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

            invested_cost = qty * avg
            if side == "Long":
                pos_val = qty * mark
                u_pnl = (mark - avg) * qty
            else:
                pos_val = qty * avg - (mark - avg) * qty
                u_pnl = (avg - mark) * qty

            u_pnl_pct = (u_pnl / invested_cost * 100.0) if invested_cost > 0 else 0.0
            unrealized_pnl += u_pnl

            position_list.append({
                "ticker": ticker,
                "side": side,
                "shares": round(qty, 4),
                "avg_price": round(avg, 2),
                "current_price": round(mark, 2),
                "invested_cost": round(invested_cost, 2),
                "current_value": round(pos_val, 2),
                "unrealized_pnl": round(u_pnl, 2),
                "unrealized_pnl_pct": round(u_pnl_pct, 2),
                "is_profit": u_pnl >= 0,
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
            })

        total_equity = self.cash + sum(p["current_value"] for p in position_list)
        total_pnl = self.realized_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / self.initial_capital * 100.0) if self.initial_capital > 0 else 0.0
        daily_pnl = total_equity - self.daily_start_equity
        daily_loss_limit = self.daily_start_equity * float(self.settings.get("max_daily_loss_pct", 2.0)) / 100.0
        largest_position_pct = max(
            ((p["current_value"] / total_equity * 100.0) if total_equity > 0 else 0.0 for p in position_list),
            default=0.0,
        )
        broker_positions = get_alpaca_positions()

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
            "risk": {
                "daily_pnl": round(daily_pnl, 2),
                "daily_loss_limit": round(daily_loss_limit, 2),
                "daily_loss_pct": round((daily_pnl / self.daily_start_equity * 100.0) if self.daily_start_equity else 0.0, 2),
                "largest_position_pct": round(largest_position_pct, 2),
                "max_position_pct": float(self.settings.get("max_position_pct", 25.0)),
                "daily_start_date": self.daily_start_date,
            },
            "positions": position_list,
            "alpaca_positions": broker_positions,
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
        if self.use_alpaca and alpaca_mode_enabled():
            return submit_alpaca_paper_order(
                ticker, action, shares, order_type, is_ai=is_ai,
                reasoning=reasoning, provider=provider, model=model,
            )
        prices = get_latest_prices()
        price = prices.get(ticker, 100.0)
        notional = shares * price

        today = datetime.now(timezone.utc).date().isoformat()
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_equity = self.cash + sum(
                float(pos.get("shares", 0.0)) * float(prices.get(symbol, pos.get("avg_price", 0.0)))
                for symbol, pos in self.positions.items()
            )

        if action == "Buy":
            current_equity = self.cash + sum(
                float(pos.get("shares", 0.0)) * float(prices.get(symbol, pos.get("avg_price", 0.0)))
                for symbol, pos in self.positions.items()
            )
            max_position_pct = float(self.settings.get("max_position_pct", 25.0)) / 100.0
            existing_value = float(self.positions.get(ticker, {}).get("shares", 0.0)) * price
            max_position_value = current_equity * max_position_pct
            if existing_value + notional > max_position_value + 0.01:
                remaining = max(0.0, max_position_value - existing_value)
                return {
                    "success": False,
                    "error": (
                        f"Risk limit: {ticker} position would exceed {max_position_pct:.0%} of equity. "
                        f"Maximum additional notional is ${remaining:,.2f}."
                    ),
                    "risk_limit": "max_position_pct",
                }

            current_equity = self.cash + sum(
                float(pos.get("shares", 0.0)) * float(prices.get(symbol, pos.get("avg_price", 0.0)))
                for symbol, pos in self.positions.items()
            )
            current_pnl = current_equity - self.daily_start_equity
            max_daily_loss_pct = float(self.settings.get("max_daily_loss_pct", 2.0)) / 100.0
            if current_pnl < -(self.daily_start_equity * max_daily_loss_pct):
                return {
                    "success": False,
                    "error": f"Risk limit: trading paused after {current_pnl:,.2f} daily loss exceeded {max_daily_loss_pct:.1%} limit.",
                    "risk_limit": "max_daily_loss_pct",
                }

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

    def close_position(self, ticker: str, shares: Optional[float] = None) -> Dict[str, Any]:
        ticker = ticker.upper()
        if self.use_alpaca and alpaca_mode_enabled():
            position = next((p for p in get_alpaca_positions() if p["ticker"] == ticker), None)
            if position is None:
                return {"success": False, "error": f"No Alpaca Paper position in {ticker}."}
            qty = min(float(position["shares"]), float(shares)) if shares and float(shares) > 0 else float(position["shares"])
            action = "Buy" if position["side"] == "Short" else "Sell"
            return submit_alpaca_paper_order(ticker, action, qty, "Market", reasoning=f"Closed Alpaca Paper position in {ticker}")
        if ticker not in self.positions:
            return {"success": False, "error": f"No open position in {ticker}."}

        pos = self.positions[ticker]
        total_shares = float(pos.get("shares", 0.0))
        shares_to_sell = min(total_shares, float(shares)) if (shares and float(shares) > 0) else total_shares
        side = pos.get("side", "Long")
        action = "Sell" if side == "Long" else "Buy"
        return self.execute_order(
            ticker=ticker,
            action=action,
            shares=shares_to_sell,
            order_type="Market",
            reasoning=f"Closed {shares_to_sell:g}/{total_shares:g} shares in {ticker}",
        )

    def reset(self, initial_capital: float = 100000.0) -> Dict[str, Any]:
        if self.use_alpaca and alpaca_mode_enabled():
            return {"success": False, "error": "Reset is disabled in Alpaca Paper mode. Manage the account in Alpaca."}
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.realized_pnl = 0.0
        self.daily_start_equity = initial_capital
        self.daily_start_date = datetime.now(timezone.utc).date().isoformat()
        self.positions = {}
        self.save()
        return {"success": True, "portfolio": self.get_state()}

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        self.settings.update(new_settings)
        self.save()
        return {"success": True, "settings": self.settings}


# Global portfolio manager instance
portfolio_mgr = PortfolioManager(PORTFOLIO_STATE_FILE, use_alpaca=True)


def _position_sign(ticker: str) -> float:
    """+1 long, -1 short, 0 flat — from the broker in Alpaca mode, else the local simulator."""
    for pos in portfolio_mgr.get_state().get("positions") or []:
        if pos.get("ticker") == ticker.upper() and float(pos.get("shares") or 0.0) > 0:
            return -1.0 if pos.get("side") == "Short" else 1.0
    return 0.0


class _DashboardBroker:
    """What the autopilot needs from the app: account state, prices, orders and decisions."""

    def state(self) -> Dict[str, Any]:
        return portfolio_mgr.get_state()

    def price(self, ticker: str) -> Optional[float]:
        price = get_latest_prices().get(ticker)
        if price:
            return float(price)
        from . import analytics

        close = analytics.close_panel()
        return float(close[ticker].dropna().iloc[-1]) if ticker in close.columns else None

    def place(self, ticker: str, action: str, shares: float, reasoning: str, provider: str, model: str) -> Dict[str, Any]:
        return portfolio_mgr.execute_order(ticker=ticker, action=action, shares=shares, is_ai=True,
                                           reasoning=reasoning, provider=provider, model=model)

    def decide(self, ticker: str, mode: str, backbone: str) -> Dict[str, Any]:
        from .agent_service import service

        position = _position_sign(ticker)
        brief = service.brief(ticker, position=position)
        cons = brief["consensus"]
        rule = {"action": cons["action"], "confidence": cons["confidence"], "reasoning": "; ".join(cons["reasons"]),
                "provider": "rule", "model": "memory + price consensus"}
        if mode == "consensus":
            return rule
        try:
            info = service.llm_decide(ticker, backbone, position)
            d = info.get("decision") or {}
            ok = str(d.get("status", "")).startswith("ok")
        except Exception as exc:
            ok, d, info = False, {"status": str(exc)}, {}
        if not ok:
            if mode == "llm_and_consensus":
                return {**rule, "action": "Hold", "reasoning": f"LLM unavailable ({d.get('status')}), so no agreement: Hold"}
            return {**rule, "reasoning": f"LLM unavailable ({d.get('status')}); rule consensus used. {rule['reasoning']}"}
        llm = {"action": d.get("action", "Hold"), "confidence": d.get("confidence"), "reasoning": d.get("reasoning", ""),
               "provider": info.get("provider", "llm"), "model": info.get("model", backbone)}
        if llm["action"] == rule["action"]:
            return llm
        note = f"LLM says {llm['action']}, rule consensus says {rule['action']}"
        if mode == "llm_and_consensus":
            return {**llm, "action": "Hold", "reasoning": f"{note}: both must agree, so Hold"}
        return {**llm, "disagreement": note}

    def targets(self, method: str) -> Dict[str, float]:
        plan = allocation_plan_for(method)
        return {r["ticker"]: r["weight"] for r in plan["rows"]}


def allocation_plan_for(method: str = "inverse_vol", lookback: int = 120) -> Dict[str, Any]:
    """Budget split using the limits from Settings and the account's current holdings."""
    from . import analytics

    b = autopilot.budget
    state = portfolio_mgr.get_state()
    holdings: Dict[str, float] = {}
    for pos in state.get("positions") or []:
        sign = -1.0 if pos.get("side") == "Short" else 1.0
        holdings[pos["ticker"]] = holdings.get(pos["ticker"], 0.0) + sign * float(pos.get("current_value") or 0.0)
    signals = None
    if method == "signal_tilt":
        from .agent_service import service

        signals = {r["ticker"]: r["consensus"] for r in service.scan()["rows"]}
    prices = get_latest_prices()
    return analytics.allocation_plan(float(b["total_budget"]), float(b["cash_reserve_pct"]), float(b["max_position_pct"]),
                                     list(b["allowed_tickers"]), method, lookback, holdings, prices, signals)


autopilot = Autopilot(ARTIFACTS_DIR / "paper_trading" / "autopilot.json",
                      ARTIFACTS_DIR / "paper_trading" / "autopilot_log.jsonl", _DashboardBroker())


def execute_ai_trade(
    ticker: str = "TSLA",
    risk_mode: Optional[str] = None,
    provider: Optional[str] = None,
    backbone: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Run the research memory agent (Ours) for ``ticker`` and route its decision to the
    paper portfolio. The LLM sees the same context as in the backbone experiment; if the
    LLM is unavailable the rule/price-model consensus is used instead."""
    from .agent_service import service

    ticker = ticker.upper()
    active_risk = risk_mode or portfolio_mgr.settings.get("risk_profile", "standard")
    position = _position_sign(ticker)
    llm_info: Dict[str, Any] = {}
    if use_llm:
        try:
            llm_info = service.llm_decide(ticker, backbone or portfolio_mgr.settings.get("agent_backbone", "gpt-oss-120b"), position)
        except Exception as exc:  # fall back to the deterministic consensus
            llm_info = {"error": str(exc)}
    decision = (llm_info.get("decision") or {}) if llm_info else {}
    if decision.get("status", "").startswith("ok"):
        brief = llm_info["brief"]
        action = decision["action"]
        reasoning = decision.get("reasoning", "")
        provider_used, model_used = llm_info["provider"], llm_info["model"]
        cited_ids = decision.get("cited_ids", [])
    else:
        brief = service.brief(ticker, position=position)
        action = brief["consensus"]["action"]
        reasoning = "; ".join(brief["consensus"]["reasons"])
        provider_used, model_used, cited_ids = "rule", "memory+price consensus", []
    order_result = _route_agent_order(ticker, action, reasoning, provider_used, model_used, brief.get("price"))
    return {
        "success": True,
        "ticker": ticker,
        "action": action,
        "reasoning": reasoning,
        "provider": provider_used,
        "model": model_used,
        "cited_memory_ids": cited_ids,
        "risk_mode": active_risk,
        "brief": brief,
        "llm_error": llm_info.get("error") or (decision.get("status") if decision and not decision.get("status", "").startswith("ok") else None),
        "order": order_result["order"] if order_result and "order" in order_result else None,
        "order_error": order_result.get("error") if order_result else None,
        "portfolio": portfolio_mgr.get_state(),
    }


def _route_agent_order(ticker: str, action: str, reasoning: str, provider: str, model: str,
                       ref_price: Optional[float]) -> Optional[Dict[str, Any]]:
    """A one-off agent trade the user clicked for. Sized and checked by the same budget
    guardrails as the autopilot. Buy opens/extends a long; Sell closes it; Hold does nothing."""
    if action not in ("Buy", "Sell"):
        return None
    return autopilot.execute_now(ticker, {"action": action, "reasoning": reasoning, "provider": provider,
                                          "model": model}, "manual agent run")


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

        if path.startswith("/api/agent/") or path == "/api/research" or path.startswith("/figures/"):
            self._agent_get(path, query)
            return

        if path.startswith("/api/analytics/"):
            self._analytics_get(path, query)
            return

        if path == "/api/autopilot/status":
            self._send_json(autopilot.status())
            return

        if path == "/api/credentials":
            if self._credentials_allowed():
                self._send_json(credentials.status())
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

        if path.startswith("/api/agent/"):
            self._agent_post(path, body)
            return

        if path.startswith("/api/credentials/"):
            self._credentials_post(path, body)
            return

        if path.startswith("/api/autopilot/"):
            self._autopilot_post(path, body)
            return

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
            shares = float(body.get("shares")) if body.get("shares") else None
            res = portfolio_mgr.close_position(ticker, shares=shares)
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

    def _agent_get(self, path: str, query: Dict[str, list]) -> None:
        from .agent_service import FIGURE_DIR, service

        q = lambda key, default=None: (query.get(key) or [default])[0]
        try:
            if path == "/api/agent/status":
                self._send_json(service.status())
            elif path == "/api/agent/brief":
                ticker = str(q("ticker", "TSLA")).upper()
                self._send_json(service.brief(ticker, q("date"), _position_sign(ticker)))
            elif path == "/api/agent/scan":
                self._send_json(service.scan({t: _position_sign(t) for t in portfolio_mgr.positions}))
            elif path == "/api/research":
                self._send_json(service.research(str(q("protocol", "rolling"))))
            elif path.startswith("/figures/"):
                name = Path(path).name
                target = FIGURE_DIR / name
                if target.suffix != ".png" or not target.exists() or target.parent != FIGURE_DIR:
                    self.send_response(404); self.end_headers(); return
                blob = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Cache-Control", "max-age=300")
                self.end_headers()
                self.wfile.write(blob)
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _credentials_allowed(self) -> bool:
        """Keys may only be read or changed by a browser on this machine, from this app's own page."""
        local_hosts = {"localhost", "127.0.0.1", "::1", str(self.server.server_address[0])}
        host = urllib.parse.urlsplit(f"//{self.headers.get('Host', '')}").hostname or ""
        origin = self.headers.get("Origin")
        same_origin = origin is None or urllib.parse.urlsplit(origin).netloc == self.headers.get("Host")
        if self.client_address[0] in ("127.0.0.1", "::1") and host in local_hosts and same_origin:
            return True
        self._send_json({"error": "API keys can only be managed from the app on this computer."}, status=403)
        return False

    def _credentials_post(self, path: str, body: Dict[str, Any]) -> None:
        if not self._credentials_allowed():
            return
        provider = str(body.get("provider", ""))
        values = body.get("values")
        try:
            if path == "/api/credentials/save":
                res = credentials.save(provider, values or {}, test_first=body.get("test", True) is not False)
                self._send_json({**res, "status": credentials.status()}, status=200 if res["saved"] else 400)
            elif path == "/api/credentials/test":
                self._send_json({"test": credentials.test(provider, values), "status": credentials.status()})
            elif path == "/api/credentials/remove":
                credentials.remove(provider)
                self._send_json({"removed": True, "status": credentials.status()})
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _autopilot_post(self, path: str, body: Dict[str, Any]) -> None:
        # Trading controls change what the agent may do with money: same-machine browser only.
        if not self._credentials_allowed():
            return
        try:
            if path == "/api/autopilot/config":
                autopilot.update(budget=body.get("budget"), config=body.get("config"))
                self._send_json(autopilot.status())
            elif path == "/api/autopilot/kill":
                cancelled = autopilot.engage_kill_switch(str(body.get("reason") or "manual emergency stop"))
                closed = []
                if body.get("close_all"):
                    for pos in portfolio_mgr.get_state().get("positions") or []:
                        res = portfolio_mgr.close_position(pos["ticker"])
                        closed.append({"ticker": pos["ticker"], "success": bool(res.get("success")), "error": res.get("error")})
                self._send_json({"cancelled": cancelled, "closed": closed, "status": autopilot.status()})
            elif path == "/api/autopilot/resume":
                autopilot.release_kill_switch()
                self._send_json(autopilot.status())
            elif path == "/api/autopilot/run_once":
                results = autopilot.run_cycle(manual=True)
                self._send_json({"results": results, "status": autopilot.status()})
            elif path == "/api/autopilot/resolve":
                ids = body.get("ids") or [body.get("id")]
                results = [autopilot.resolve(str(i), bool(body.get("approve"))) for i in ids if i]
                self._send_json({"results": results, "status": autopilot.status()})
            elif path == "/api/autopilot/rebalance":
                plan = allocation_plan_for(str(body.get("method", "inverse_vol")))
                rows = sorted(plan["rows"], key=lambda r: r["delta_usd"])  # sells first free up cash
                queued = []
                for r in rows:
                    if abs(r["delta_usd"]) < max(1.0, float(body.get("min_trade_usd", 25.0))):
                        continue
                    reason = f"Rebalance to {plan['method_label']}: target ${r['target_usd']:,.0f} ({r['weight']:.1%} of budget)"
                    if r["delta_usd"] > 0:
                        queued.append(autopilot.queue_order(r["ticker"], "Buy", reason, notional=r["delta_usd"]))
                    else:
                        shares = -r["delta_usd"] / r["price"]
                        queued.append(autopilot.queue_order(r["ticker"], "Sell", reason, shares=shares))
                self._send_json({"queued": queued, "status": autopilot.status()})
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _analytics_get(self, path: str, query: Dict[str, list]) -> None:
        from . import analytics

        q = lambda key, default=None: (query.get(key) or [default])[0]
        ticker = str(q("ticker", "TSLA")).upper()
        scope = "universe" if q("scope") == "universe" else "ticker"
        try:
            pattern = dict(ticker=ticker, window=int(q("window", 20)), horizon=int(q("horizon", 10)),
                           k=int(q("k", 8)), scope=scope, threshold=float(q("threshold", 0.01)))
            if path == "/api/analytics/analogs":
                self._send_json(analytics.pattern_analogs(**pattern))
            elif path == "/api/analytics/backtest":
                self._send_json(analytics.analog_backtest(**pattern, long_only=q("long_only") in ("1", "true"),
                                                          cost_bps=float(q("cost_bps", 5.0))))
            elif path == "/api/analytics/sweep":
                self._send_json(analytics.analog_sweep(ticker, scope, pattern["k"], pattern["threshold"],
                                                       q("long_only") in ("1", "true")))
            elif path == "/api/analytics/risk":
                self._send_json(analytics.risk_report(ticker))
            elif path == "/api/analytics/correlation":
                self._send_json(analytics.correlation_matrix(int(q("lookback", 120))))
            elif path == "/api/analytics/algorithms":
                from . import algorithms
                self._send_json(algorithms.algorithm_board(ticker))
            elif path == "/api/analytics/experience":
                from . import algorithms
                self._send_json(algorithms.experience(ticker, str(q("system", "Ours")), int(q("limit", 40))))
            elif path == "/api/analytics/rl":
                from . import algorithms
                self._send_json(algorithms.q_learning_agent(ticker, int(q("epochs", 30)), cost_bps=float(q("cost_bps", 5.0))))
            elif path == "/api/analytics/allocation":
                self._send_json(allocation_plan_for(str(q("method", "inverse_vol")), int(q("lookback", 120))))
            elif path == "/api/analytics/portfolio":
                self._send_json(analytics.portfolio_distribution(portfolio_mgr.get_state(), ticker, int(q("days", 21))))
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _agent_post(self, path: str, body: Dict[str, Any]) -> None:
        from .agent_service import service

        try:
            if path == "/api/agent/llm":
                ticker = str(body.get("ticker", "TSLA")).upper()
                self._send_json(service.llm_decide(ticker, str(body.get("backbone", "gpt-oss-120b")), _position_sign(ticker)))
            elif path == "/api/agent/execute":
                ticker = str(body.get("ticker", "TSLA")).upper()
                action = str(body.get("action", "Hold")).title()
                if action not in ("Buy", "Sell", "Hold"):
                    raise ValueError("action must be Buy, Sell or Hold")
                res = _route_agent_order(ticker, action, str(body.get("reasoning", "manual approval of agent recommendation")),
                                         str(body.get("provider", "agent")), str(body.get("model", "memory agent")), body.get("price"))
                self._send_json({"success": bool(res and "order" in res) or action == "Hold", "result": res,
                                 "portfolio": portfolio_mgr.get_state()})
            elif path == "/api/agent/run":
                ticker = str(body.get("ticker", "TSLA")).upper()
                self._send_json(execute_ai_trade(ticker, backbone=body.get("backbone"), use_llm=bool(body.get("use_llm", True))))
            elif path == "/api/agent/refresh":
                self._send_json(service.refresh())
            else:
                self._send_json({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

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

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The browser aborted the request (page refresh, tab switch, or a
        # superseded fetch) before the response was written. Nothing to do.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


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
    autopilot.start()

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
