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
    allow_short: bool = False


class PaperBroker:
    """Alpaca broker adapter. Paper trading is the default; live requires explicit opt-in."""

    def __init__(self, execute: bool = False, paper: bool = True) -> None:
        self.execute = execute
        self.paper = paper
        self.client: Any = None
        if execute:
            if not paper and os.environ.get("ALPACA_LIVE_CONFIRM") != "YES":
                raise RuntimeError(
                    "Refusing to place live orders. Set ALPACA_LIVE_CONFIRM=YES and use real-money "
                    "ALPACA_KEY/ALPACA_SECRET credentials to opt in."
                )
            api_key = (
                os.environ.get("ALPACA_PAPER_KEY") or os.environ.get("ALPACA_KEY")
                if paper
                else os.environ.get("ALPACA_KEY")
            )
            api_secret = (
                os.environ.get("ALPACA_PAPER_SECRET") or os.environ.get("ALPACA_SECRET")
                if paper
                else os.environ.get("ALPACA_SECRET")
            )
            if not api_key or not api_secret:
                raise RuntimeError("Set Alpaca API key and secret before --execute")
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:
                raise RuntimeError("Install alpaca-py before --execute") from exc
            self.client = TradingClient(api_key, api_secret, paper=paper)

    def submit_market_buy(self, ticker: str, notional: float) -> dict[str, Any]:
        if not self.execute:
            return {"status": "dry_run", "action": "buy", "ticker": ticker, "notional": round(notional, 2)}
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        order = self.client.submit_order(MarketOrderRequest(
            symbol=ticker,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        return self._order_dict(order, ticker=ticker, notional=notional, action="buy")

    def submit_market_sell(self, ticker: str, qty: float | None = None, notional: float | None = None) -> dict[str, Any]:
        if not self.execute:
            return {
                "status": "dry_run",
                "action": "sell",
                "ticker": ticker,
                "qty": qty,
                "notional": round(notional, 2) if notional is not None else None,
            }
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        kwargs: dict[str, Any] = {"symbol": ticker, "side": OrderSide.SELL, "time_in_force": TimeInForce.DAY}
        if qty is not None:
            kwargs["qty"] = qty
        elif notional is not None:
            kwargs["notional"] = round(notional, 2)
        order = self.client.submit_order(MarketOrderRequest(**kwargs))
        return self._order_dict(order, ticker=ticker, action="sell", qty=qty, notional=notional)

    def close_position(self, ticker: str) -> dict[str, Any]:
        if not self.execute:
            return {"status": "dry_run", "action": "close_position", "ticker": ticker}
        try:
            order = self.client.close_position(ticker)
            return self._order_dict(order, ticker=ticker, action="close_position")
        except Exception as exc:
            return {"status": "error", "action": "close_position", "ticker": ticker, "error": str(exc)}

    @staticmethod
    def _order_dict(order: Any, **extra: Any) -> dict[str, Any]:
        return {
            **extra,
            "status": str(getattr(order, "status", "unknown")),
            "order_id": str(getattr(order, "id", "")),
            "qty": str(order.qty) if getattr(order, "qty", None) is not None else None,
            "filled_qty": str(order.filled_qty) if getattr(order, "filled_qty", None) is not None else None,
            "filled_avg_price": str(order.filled_avg_price) if getattr(order, "filled_avg_price", None) is not None else None,
            "notional": float(order.notional) if getattr(order, "notional", None) is not None else extra.get("notional"),
        }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        if not self.execute:
            return {"status": "dry_run", "order_id": order_id}
        return self._order_dict(self.client.get_order_by_id(order_id))

    def get_positions(self) -> list[dict[str, Any]]:
        if not self.execute or self.client is None:
            return []
        return [{"symbol": p.symbol, "qty": str(p.qty), "avg_entry_price": str(p.avg_entry_price), "market_value": str(p.market_value)} for p in self.client.get_all_positions()]


class PaperTradingAgent:
    def __init__(self, config: TradingConfig, broker: PaperBroker) -> None:
        self.config = config
        self.broker = broker

    def validate(self) -> None:
        if not self.config.tickers:
            raise ValueError("At least one ticker is required")
        if len(set(self.config.tickers)) != len(self.config.tickers):
            raise ValueError("Tickers must be unique")
        if self.config.budget <= 0:
            raise ValueError("Budget must be positive")
        if not 0 < self.config.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")

    def get_open_position(self, ticker: str) -> dict[str, Any] | None:
        for pos in self.broker.get_positions():
            if pos.get("symbol") == ticker:
                return pos
        return None

    def build_orders(self, decisions: dict[str, str]) -> list[dict[str, Any]]:
        self.validate()
        allocation = self.config.budget * self.config.max_position_pct
        orders = []
        for ticker in self.config.tickers:
            decision = decisions.get(ticker, "Hold").strip().title()
            if decision == "Buy":
                orders.append({"ticker": ticker, "decision": decision, "status": "approved", "notional": allocation})
            elif decision == "Sell":
                orders.append({"ticker": ticker, "decision": decision, "status": "approved", "notional": allocation})
            else:
                orders.append({"ticker": ticker, "decision": decision, "status": "not_submitted", "notional": 0.0})
        return orders

    def execute_decisions(self, decisions: dict[str, str]) -> list[dict[str, Any]]:
        orders = self.build_orders(decisions)
        results = []
        for order in orders:
            decision = order["decision"]
            ticker = order["ticker"]
            if decision == "Buy":
                result = self.broker.submit_market_buy(ticker, order["notional"])
                order.update(result)
            elif decision == "Sell":
                pos = self.get_open_position(ticker)
                if pos and float(pos.get("qty", 0)) > 0:
                    result = self.broker.close_position(ticker)
                    order.update(result)
                elif self.config.allow_short:
                    result = self.broker.submit_market_sell(ticker, notional=order["notional"])
                    order.update(result)
                else:
                    order.update({
                        "status": "skipped_no_position",
                        "reason": "No open position to sell and allow_short=False",
                        "notional": 0.0,
                    })
            results.append(order)
        self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.audit_path.open("a", encoding="utf-8") as handle:
            for result in results:
                record = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
                handle.write(json.dumps(record) + "\n")
        return results

    def reconcile_portfolio(self) -> dict[str, Any]:
        """Audit current broker positions against configured universe."""
        positions = self.broker.get_positions()
        pos_by_symbol = {p.get("symbol"): p for p in positions}
        universe = set(self.config.tickers)
        reconciled = {
            "monitored_positions": {s: pos_by_symbol[s] for s in universe if s in pos_by_symbol},
            "unmonitored_positions": {s: p for s, p in pos_by_symbol.items() if s not in universe},
            "mode": ("live_execute" if not self.broker.paper else "paper_execute") if self.broker.execute else "dry_run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return reconciled


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
    parser.add_argument("--live", action="store_true", help="Use the live Alpaca account (requires ALPACA_LIVE_CONFIRM=YES)")
    parser.add_argument("--allow-short", action="store_true", help="Allow short selling if no long position exists")
    parser.add_argument("--reconcile", action="store_true", help="Audit current broker positions against universe")
    parser.add_argument("--status", nargs="*", metavar="ORDER_ID", help="Read existing order status and positions; never submits")
    args = parser.parse_args()
    tickers = tuple(item.strip().upper() for item in args.tickers.split(",") if item.strip())
    config = TradingConfig(tickers=tickers, budget=args.budget, audit_path=args.audit_path, allow_short=args.allow_short)
    broker = PaperBroker(execute=args.execute or args.status is not None, paper=not args.live)
    agent = PaperTradingAgent(config, broker)

    if args.reconcile:
        print(json.dumps(agent.reconcile_portfolio(), indent=2))
        return

    if args.status is not None:
        print(json.dumps({"orders": [broker.get_order_status(order_id) for order_id in args.status], "positions": broker.get_positions()}, indent=2))
        return

    decisions = _latest_decisions(args.decisions, tickers) if args.decisions else {ticker: "Hold" for ticker in tickers}
    results = agent.execute_decisions(decisions)
    mode = "dry_run" if not args.execute else ("live_execute" if args.live else "paper_execute")
    print(json.dumps({"mode": mode, "results": results}, indent=2))


if __name__ == "__main__":
    main()
