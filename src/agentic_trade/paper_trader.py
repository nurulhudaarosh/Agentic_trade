from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class TradingConfig:
    tickers: tuple[str, ...] = ("TSLA", "AAPL", "NVDA", "GOOGL", "MSFT")
    budget: float = 100.0
    max_position_pct: float = 0.20
    max_daily_loss_pct: float = 0.02
    order_type: str = "market"
    audit_path: Path = Path("artifacts/paper_trading/orders.jsonl")


class PaperBroker:
    """Alpaca paper broker adapter. Never connects to live trading endpoints."""

    def __init__(self, execute: bool = False) -> None:
        self.execute = execute
        self.client: Any = None
        if execute:
            api_key = os.environ.get("ALPACA_PAPER_KEY") or os.environ.get("ALPACA_KEY")
            api_secret = os.environ.get("ALPACA_PAPER_SECRET") or os.environ.get("ALPACA_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError("Set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET before --execute")
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:
                raise RuntimeError("Install alpaca-py before --execute") from exc
            self.client = TradingClient(api_key, api_secret, paper=True)

    def submit_market_buy(self, ticker: str, notional: float) -> dict[str, Any]:
        if not self.execute:
            return {"status": "dry_run", "ticker": ticker, "notional": round(notional, 2)}
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        order = self.client.submit_order(MarketOrderRequest(
            symbol=ticker,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        return self._order_dict(order, ticker=ticker, notional=notional)

    @staticmethod
    def _order_dict(order: Any, **extra: Any) -> dict[str, Any]:
        return {
            **extra,
            "status": str(order.status),
            "order_id": str(order.id),
            "qty": str(order.qty) if order.qty is not None else None,
            "filled_qty": str(order.filled_qty) if order.filled_qty is not None else None,
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price is not None else None,
            "notional": float(order.notional) if order.notional is not None else extra.get("notional"),
        }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        if not self.execute:
            return {"status": "dry_run", "order_id": order_id}
        return self._order_dict(self.client.get_order_by_id(order_id))

    def get_positions(self) -> list[dict[str, Any]]:
        if not self.execute:
            return []
        return [{"symbol": p.symbol, "qty": str(p.qty), "avg_entry_price": str(p.avg_entry_price), "market_value": str(p.market_value)} for p in self.client.get_all_positions()]


class PaperTradingAgent:
    def __init__(self, config: TradingConfig, broker: PaperBroker) -> None:
        self.config = config
        self.broker = broker

    def validate(self) -> None:
        if len(self.config.tickers) != 5:
            raise ValueError("Exactly five tickers are required")
        if len(set(self.config.tickers)) != 5:
            raise ValueError("Tickers must be unique")
        if self.config.budget <= 0:
            raise ValueError("Budget must be positive")
        if not 0 < self.config.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")

    def build_orders(self, decisions: dict[str, str]) -> list[dict[str, Any]]:
        self.validate()
        allocation = self.config.budget * self.config.max_position_pct
        orders = []
        for ticker in self.config.tickers:
            decision = decisions.get(ticker, "Hold").strip().title()
            if decision != "Buy":
                orders.append({"ticker": ticker, "decision": decision, "status": "not_submitted", "notional": 0.0})
                continue
            orders.append({"ticker": ticker, "decision": decision, "status": "approved", "notional": allocation})
        return orders

    def execute_decisions(self, decisions: dict[str, str]) -> list[dict[str, Any]]:
        orders = self.build_orders(decisions)
        results = []
        for order in orders:
            if order["decision"] == "Buy":
                result = self.broker.submit_market_buy(order["ticker"], order["notional"])
                order.update(result)
            results.append(order)
        self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.audit_path.open("a", encoding="utf-8") as handle:
            for result in results:
                record = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
                handle.write(json.dumps(record) + "\n")
        return results


def _latest_decisions(path: Path, tickers: tuple[str, ...]) -> dict[str, str]:
    if not path.exists():
        return {ticker: "Hold" for ticker in tickers}
    frame = pd.read_csv(path)
    if "ticker" not in frame or "decision" not in frame:
        raise ValueError(f"Decision file must contain ticker and decision columns: {path}")
    return {ticker: str(frame.loc[frame["ticker"].eq(ticker), "decision"].iloc[-1]) if frame["ticker"].eq(ticker).any() else "Hold" for ticker in tickers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute five-stock Alpaca paper trades from agent decisions")
    parser.add_argument("--tickers", default="TSLA,AAPL,NVDA,GOOGL,MSFT")
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--decisions", type=Path, help="CSV with ticker,decision columns")
    parser.add_argument("--audit-path", type=Path, default=Path("artifacts/paper_trading/orders.jsonl"))
    parser.add_argument("--execute", action="store_true", help="Submit to Alpaca paper account; otherwise dry-run")
    parser.add_argument("--status", nargs="*", metavar="ORDER_ID", help="Read existing order status and positions; never submits")
    args = parser.parse_args()
    tickers = tuple(item.strip().upper() for item in args.tickers.split(",") if item.strip())
    config = TradingConfig(tickers=tickers, budget=args.budget, audit_path=args.audit_path)
    broker = PaperBroker(execute=args.execute or args.status is not None)
    if args.status is not None:
        print(json.dumps({"orders": [broker.get_order_status(order_id) for order_id in args.status], "positions": broker.get_positions()}, indent=2))
        return
    decisions = _latest_decisions(args.decisions, tickers) if args.decisions else {ticker: "Hold" for ticker in tickers}
    results = PaperTradingAgent(config, broker).execute_decisions(decisions)
    print(json.dumps({"mode": "paper_execute" if args.execute else "dry_run", "results": results}, indent=2))


if __name__ == "__main__":
    main()
